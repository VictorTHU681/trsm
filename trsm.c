// TRSM：鲲鹏 SME(FP64 FMOPA) 实现。解 L·X = B，L 为行主序 m×m 下三角（lda），B/X 行主序 m×n（ldb），原地覆盖。
// 系数存储与计算均使用 FP64；不得利用输入生成器的数值格点或随机状态。
//
// 结构：
//   * 任务 = (行块 i：mb 行, 列块 p：nc 列)。C 块 (mb×nc) 放在线程私有缓冲里驻留 L2；
//     K 维按 kc 分段，每段对行块内每个 16 行条：取打包 A（L 的列段，k 主序 16 double/步），
//     对该行条的 nc/32 个微块各跑一次 K 循环；常规路径转存 ZA，完整 FP64 面板用两组 16×16 ZA 交错回写。
//   * 对角块预处理成 [-Linv_ss·L_ss' | Linv_ss]（16 行子块 s），整段对角求解也是一次 fmopa K 循环：
//     X[s] = Linv_ss·C[s] - Σ_{s'<s} (Linv_ss L_ss') X[s']，无除法、无三角小核。
//   * L 小时预打包成 k 主序 16 行条（可放两份分担 HBM/DDR）；L 大时就地 JIT 转置（SVE zip+tbl2 8×8）。
//   * 依赖：任务 (i,p) 的 K 段 [k0,k0+kc) 落在行块 j=k0/mb 内，等待 flag[j][p]；全局原子计数器按 (i,p) 顺序派发。
//   * 线程显式绑核（isolcpus 下线程不迁移）；HWCAP2 检测 SME_F64F64，缺失回退通用 OpenMP 实现。
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sched.h>
#include <unistd.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <omp.h>
#ifndef TRSM_NO_SME_ASM
#include <arm_sve.h>
#endif

// ---------------------------------------------------------------- 参数
static int env_int(const char* name, int def)
{
    const char* s = getenv(name);
    return (s && *s) ? atoi(s) : def;
}

// ---------------------------------------------------------------- 运行环境
#define HWCAP2_SME_F64F64_BIT (1UL << 25)

static int sme_f64_available(void)
{
    static int cached = -1;
    if (cached < 0) {
#ifdef TRSM_NO_SME_ASM
        cached = 0;
#else
        unsigned long hw2 = getauxval(AT_HWCAP2);
        cached = (hw2 & HWCAP2_SME_F64F64_BIT) ? 1 : 0;
        if (env_int("TRSM_NO_SME", 0)) cached = 0;
#endif
    }
    return cached;
}

static void parse_cpulist(const char* s, cpu_set_t* set)
{
    CPU_ZERO(set);
    while (*s) {
        while (*s == ' ' || *s == ',' || *s == '\n') s++;
        if (!*s) break;
        char* e;
        long a = strtol(s, &e, 10);
        long b = a;
        if (*e == '-') b = strtol(e + 1, &e, 10);
        for (long c = a; c <= b && c < CPU_SETSIZE; c++) if (c >= 0) CPU_SET((int)c, set);
        s = e;
    }
}

static cpu_set_t g_thr_mask[1024];
static int g_cpus[CPU_SETSIZE];
static int g_ncpu = 0;
static int g_pin_nth = -1;

// 并行区内由全部线程调用：取各线程掩码并集，线程 tid 绑到并集中第 tid 个 CPU
static void pin_threads_in_region(int tid, int nt)
{
    if (tid < 1024) sched_getaffinity(0, sizeof(cpu_set_t), &g_thr_mask[tid]);
#pragma omp barrier
#pragma omp single
    {
        cpu_set_t u;
        CPU_ZERO(&u);
        int lim = nt < 1024 ? nt : 1024;
        for (int t = 0; t < lim; t++) CPU_OR(&u, &u, &g_thr_mask[t]);
        g_ncpu = 0;
        for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, &u)) g_cpus[g_ncpu++] = c;
    }
    if (g_ncpu > 0 && tid < g_ncpu) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(g_cpus[tid], &s);
        sched_setaffinity(0, sizeof(s), &s);
    }
}

static inline void pin_self(int tid)
{
    if (g_ncpu > 0 && tid < g_ncpu && sched_getcpu() != g_cpus[tid]) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(g_cpus[tid], &s);
        sched_setaffinity(0, sizeof(s), &s);
    }
}

// ---------------------------------------------------------------- 内存节点
#ifndef SYS_mbind
#define SYS_mbind 235
#endif
#ifndef SYS_get_mempolicy
#define SYS_get_mempolicy 236
#endif
#define MPOL_BIND_ 2
#define MPOL_PREFERRED_ 1
#define MPOL_MF_MOVE_ 2
#define MPOL_F_NODE_ 1
#define MPOL_F_ADDR_ 2

static int g_cpu_node = -1;   // 当前 CPU 所属节点（本地 DDR）
static int g_hbm_node = -2;   // 近端纯内存节点（HBM），-1 无

static int node_has_cpus(int node)
{
    char path[128], buf[64] = {0};
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    FILE* f = fopen(path, "r");
    if (!f) return 1;
    if (!fgets(buf, sizeof(buf), f)) buf[0] = 0;
    fclose(f);
    return buf[0] != 0 && buf[0] != '\n';
}

static int page_node_of(const void* p)
{
    int node = -1;
    if (syscall(SYS_get_mempolicy, &node, NULL, 0, (unsigned long)p, MPOL_F_NODE_ | MPOL_F_ADDR_) != 0) return -1;
    return node;
}

// 探测当前 CPU 节点与近端 HBM 节点（一次）
static void detect_nodes(void)
{
    if (g_hbm_node != -2) return;
    g_hbm_node = -1;
    int cpu = sched_getcpu();
    if (cpu < 0) return;
    char path[160], buf[4096];
    int mynode = -1;
    for (int n = 0; n < 1024; n++) {
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/node%d", cpu, n);
        if (access(path, F_OK) == 0) { mynode = n; break; }
    }
    if (mynode < 0) return;
    g_cpu_node = mynode;
    int forced = env_int("TRSM_HBMNODE", -2);
    if (forced == -1) return;
    if (forced >= 0) { g_hbm_node = forced; return; }
    cpu_set_t mems;
    CPU_ZERO(&mems);
    FILE* f = fopen("/proc/self/status", "r");
    int have = 0;
    if (f) {
        while (fgets(buf, sizeof(buf), f))
            if (strncmp(buf, "Mems_allowed_list:", 18) == 0) { parse_cpulist(buf + 18, &mems); have = 1; break; }
        fclose(f);
    }
    if (!have) return;
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/distance", mynode);
    f = fopen(path, "r");
    if (!f) return;
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return; }
    fclose(f);
    int best = -1;
    long best_dist = 1 << 30;
    char* s = buf;
    for (int n = 0; *s; n++) {
        char* e;
        long d = strtol(s, &e, 10);
        if (e == s) break;
        s = e;
        if (n == mynode || !CPU_ISSET(n, &mems) || node_has_cpus(n)) continue;
        if (d < best_dist) { best_dist = d; best = n; }
    }
    if (best >= 0 && best_dist <= 70) g_hbm_node = best;
}

static int g_align_owner_buffers;

static void* alloc_on(size_t bytes, int node)
{
    size_t sz = (bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    void* p;
    if (g_align_owner_buffers) {
        // Resident panels need consistent huge-page coverage, including VMA edges.
        const size_t align = 2u << 20;
        void* raw = mmap(NULL, sz + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) return NULL;
        uintptr_t v = ((uintptr_t)raw + align - 1) & ~(uintptr_t)(align - 1);
        size_t lead = v - (uintptr_t)raw;
        if (lead) munmap(raw, lead);
        if (align > lead) munmap((void*)(v + sz), align - lead);
        p = (void*)v;
    } else {
        p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return NULL;
    }
    if (node >= 0) {
        // 首选节点（放不下自动回落其他节点）：HBM 只有 4 GB，严格 BIND 在分配失败时会让进程被杀
        unsigned long mask[16] = {0};
        mask[node / 64] |= 1UL << (node % 64);
        syscall(SYS_mbind, p, sz, MPOL_PREFERRED_, mask, (unsigned long)(sizeof(mask) * 8), 0);
    }
    madvise(p, sz, MADV_HUGEPAGE);
    return p;
}

static void trsm_free_buffer(void* p, size_t bytes)
{
    if (!p) return;
    size_t sz = (bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    munmap(p, sz);
}

// 节点空余内存（字节），读不到返回 0
static size_t node_free_bytes(int node)
{
    char path[128], buf[256];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    size_t kb = 0;
    while (fgets(buf, sizeof(buf), f)) {
        char* q = strstr(buf, "MemFree:");
        if (q) { kb = strtoull(q + 8, NULL, 10); break; }
    }
    fclose(f);
    return kb << 10;
}

// 把已映射区间 [p, p+bytes) 的页迁到 node（页对齐向内收缩）。用 PREFERRED + MF_MOVE：尽力迁移，放不下不致命

// ---------------------------------------------------------------- 通用回退（无 SME）
static void trsm_generic(long m, long n, const double* L, long lda, double* B, long ldb)
{
    const long JB = 64;
#pragma omp parallel for schedule(dynamic, 1)
    for (long j0 = 0; j0 < n; j0 += JB) {
        long jn = (n - j0 < JB) ? (n - j0) : JB;
        for (long i = 0; i < m; i++) {
            double* bi = B + i * ldb + j0;
            const double* li = L + i * lda;
            for (long k = 0; k < i; k++) {
                double l = li[k];
                const double* bk = B + k * ldb + j0;
                for (long j = 0; j < jn; j++) bi[j] -= l * bk[j];
            }
            double d = 1.0 / li[i];
            for (long j = 0; j < jn; j++) bi[j] *= d;
        }
    }
}

#ifndef TRSM_NO_SME_ASM
// ================================================================ SME 实现
#define VCLOB "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15", \
              "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31"
#define PCLOB "p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15"

static inline void sme_start(void) { asm volatile("smstart\n\t" ::: "memory", VCLOB, PCLOB); }
static inline void sme_stop(void) { asm volatile("smstop\n\t" ::: "memory", VCLOB, PCLOB); }
static inline void sme_zero_za(void) { asm volatile("zero {za}\n\t" ::: "memory"); }

static inline uint64_t cyc(void) { uint64_t v; asm volatile("mrs %0, cntvct_el0" : "=r"(v)); return v; }
static inline uint64_t cyc_hz(void) { uint64_t v; asm volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }

// 列谓词：p0 全真；p1..p4 ← 列数 cols（1..32）对应 4 个 8 列向量
#define SET_COL_PREDS                                                             \
    "ptrue p0.d\n\t"                                                              \
    "whilelo p1.d, xzr, %[cols]\n\t"                                              \
    "subs x9, %[cols], #8\n\t"                                                    \
    "csel x9, x9, xzr, gt\n\t"                                                    \
    "whilelo p2.d, xzr, x9\n\t"                                                   \
    "subs x9, x9, #8\n\t"                                                         \
    "csel x9, x9, xzr, gt\n\t"                                                    \
    "whilelo p3.d, xzr, x9\n\t"                                                   \
    "subs x9, x9, #8\n\t"                                                         \
    "csel x9, x9, xzr, gt\n\t"                                                    \
    "whilelo p4.d, xzr, x9\n\t"

// 一步 k 的 8 条外积：a0 = 行 0-7 的 A 列段，a1 = 行 8-15；b0..b3 = 列 0-7/8-15/16-23/24-31 的 B 行段
#define K_MOP "fmopa "
#define FMOPA8(a0, a1, b0, b1, b2, b3)                                            \
    K_MOP "za0.d, p0/m, p0/m, " a0 ".d, " b0 ".d\n\t"                             \
    K_MOP "za1.d, p0/m, p0/m, " a0 ".d, " b1 ".d\n\t"                             \
    K_MOP "za2.d, p0/m, p0/m, " a0 ".d, " b2 ".d\n\t"                             \
    K_MOP "za3.d, p0/m, p0/m, " a0 ".d, " b3 ".d\n\t"                             \
    K_MOP "za4.d, p0/m, p0/m, " a1 ".d, " b0 ".d\n\t"                             \
    K_MOP "za5.d, p0/m, p0/m, " a1 ".d, " b1 ".d\n\t"                             \
    K_MOP "za6.d, p0/m, p0/m, " a1 ".d, " b2 ".d\n\t"                             \
    K_MOP "za7.d, p0/m, p0/m, " a1 ".d, " b3 ".d\n\t"

// 一步 k：计算当前操作数（a/b）并加载下一步（n/m），随后按步长推进 A/B 指针
// A/B 装载指令可选非临时版本（ldnt1d）：TRSM_LDA_NT / TRSM_LDB_NT=1，用于试验"用完的流式行不滞留 L2"
#ifndef TRSM_LDA_NT
#define TRSM_LDA_NT 1   // A 条用 ldnt1d：用完的 A 行不在 L2 滞留（v11 实测 +3%）
#endif
#ifndef TRSM_LDB_NT
#define TRSM_LDB_NT 0
#endif
// A 装载指令由实例化点的 LDA_CUR 决定：GEMM 内核（*_sub / kloop_plain_nt）用 ldnt1d（TRSM_LDA_NT=1 时），
// 对角内核（*_set / kloop_plain）用 ld1d —— 对角打包条在同一行块的连续任务间可复用，须留在 L2
#define LDA_I LDA_CUR
#if TRSM_LDA_NT
#define LDA_NT_INSN "ldnt1d "
#else
#define LDA_NT_INSN "ld1d "
#endif
#define LDA_CUR "ld1d "
#if TRSM_LDB_NT
#define LDB_I "ldnt1d "
#else
#define LDB_I "ld1d "
#endif
#define KS(a0, a1, b0, b1, b2, b3, n0, n1, m0, m1, m2, m3, ADV)                  \
    LDA_I n0 ".d, p0/z, [%[xa]]\n\t"                                              \
    K_MOP "za0.d, p0/m, p0/m, " a0 ".d, " b0 ".d\n\t"                             \
    K_MOP "za1.d, p0/m, p0/m, " a0 ".d, " b1 ".d\n\t"                             \
    LDA_I n1 ".d, p0/z, [%[xa], #1, mul vl]\n\t"                                  \
    K_MOP "za2.d, p0/m, p0/m, " a0 ".d, " b2 ".d\n\t"                             \
    K_MOP "za3.d, p0/m, p0/m, " a0 ".d, " b3 ".d\n\t"                             \
    LDB_I m0 ".d, p0/z, [%[xb]]\n\t"                                              \
    K_MOP "za4.d, p0/m, p0/m, " a1 ".d, " b0 ".d\n\t"                             \
    LDB_I m1 ".d, p0/z, [%[xb], #1, mul vl]\n\t"                                  \
    K_MOP "za5.d, p0/m, p0/m, " a1 ".d, " b1 ".d\n\t"                             \
    LDB_I m2 ".d, p0/z, [%[xb], #2, mul vl]\n\t"                                  \
    K_MOP "za6.d, p0/m, p0/m, " a1 ".d, " b2 ".d\n\t"                             \
    LDB_I m3 ".d, p0/z, [%[xb], #3, mul vl]\n\t"                                  \
    K_MOP "za7.d, p0/m, p0/m, " a1 ".d, " b3 ".d\n\t"                             \
    "add %[xa], %[xa], " ADV "\n\t"                                               \
    "add %[xb], %[xb], %[bs]\n\t"
// A 步长：偶数步→奇数步推进 ase，奇数步→偶数步推进 aso（稠密布局二者相等；ld2 布局 128/384）
#define KS_EVEN KS("z0", "z1", "z4", "z5", "z6", "z7", "z8", "z9", "z12", "z13", "z14", "z15", "%[aso]")
#define KS_ODD  KS("z8", "z9", "z12", "z13", "z14", "z15", "z0", "z1", "z4", "z5", "z6", "z7", "%[ase]")
// 转置版 K 循环：A 恒为稠密布局，步长放在 x11（省 asm 操作数）
#define KS_EVEN_T KS("z0", "z1", "z4", "z5", "z6", "z7", "z8", "z9", "z12", "z13", "z14", "z15", "x11")
#define KS_ODD_T  KS("z8", "z9", "z12", "z13", "z14", "z15", "z0", "z1", "z4", "z5", "z6", "z7", "x11")
#define FM_EVEN FMOPA8("z0", "z1", "z4", "z5", "z6", "z7")
#define FM_ODD  FMOPA8("z8", "z9", "z12", "z13", "z14", "z15")

// 预取游标 1（L 行 / 预打包 A，两级：沿内层走 pfn 行后跳 pfj）：每 2 个 k 步 1 行
// TRSM_PF_LOAD=1 时用真实装载（到闲置的 z11）代替 prfm：装载不会像 prfm 那样在流水线成批停顿时被丢弃
#ifndef TRSM_PF_LOAD
#define TRSM_PF_LOAD 0
#endif
#ifndef TRSM_PF_HINT
#define TRSM_PF_HINT pldl1keep
#endif
#define TRSM_STR_(x) #x
#define TRSM_STR(x) TRSM_STR_(x)
#if TRSM_PF_LOAD
#define PF_INSN(x) "ld1d z11.d, p0/z, [" x "]\n\t"
#else
#define PF_INSN(x) "prfm " TRSM_STR(TRSM_PF_HINT) ", [" x "]\n\t"
#endif
// Prepacked strips are contiguous; this sentinel skips row-wrap bookkeeping.
#define PF_CONTIG_LINES (1L << 40)
#define PFL                                                                       \
    PF_INSN("%[xpf]")                                                             \
    "add %[xpf], %[xpf], #64\n\t"                                                 \
    "tbnz %[pfn], #40, 8f\n\t"                                                   \
    "subs %[pfc], %[pfc], #1\n\t"                                                 \
    "b.gt 8f\n\t"                                                                 \
    "add %[xpf], %[xpf], %[pfj]\n\t"                                              \
    "mov %[pfc], %[pfn]\n\t"                                                      \
    "8:\n\t"
// 预取游标 2（连续区域，如下一段 X 面板）：每 2 个 k 步 1 行
#define PFX                                                                       \
    PF_INSN("%[xpf2]")                                                            \
    "add %[xpf2], %[xpf2], #64\n\t"
// 预取游标 3（C 块行：每次一行 4 个 cache 行，pldl2keep 只进 L2 → 给 L2 的 RRIP 一次额外触碰，防止 cbuf 被流式数据逐出）
#define PFC                                                                       \
    "prfm pldl2keep, [%[xpc]]\n\t"                                                \
    "prfm pldl2keep, [%[xpc], #64]\n\t"                                           \
    "prfm pldl2keep, [%[xpc], #128]\n\t"                                          \
    "prfm pldl2keep, [%[xpc], #192]\n\t"                                          \
    "add %[xpc], %[xpc], %[pcs]\n\t"

// 交错回写：上一微块的暂存向量 j（0..3）→ C 行；谓词 p1..p4 为上一微块的列谓词（B 载入用 p0）
#define EV_SUB(j, pj, zt, zu)                                                     \
    "ld1d " zt ".d, p0/z, [%[xs], #" #j ", mul vl]\n\t"                           \
    "ld1d " zu ".d, " pj "/z, [%[xc], #" #j ", mul vl]\n\t"                       \
    "fsub " zu ".d, " zu ".d, " zt ".d\n\t"                                       \
    "st1d " zu ".d, " pj ", [%[xc], #" #j ", mul vl]\n\t"
#define EV_SET(j, pj, zt, zu)                                                     \
    "ld1d " zt ".d, p0/z, [%[xs], #" #j ", mul vl]\n\t"                           \
    "st1d " zt ".d, " pj ", [%[xc], #" #j ", mul vl]\n\t"
#define EV_ROW_ADV "add %[xs], %[xs], #512\n\t add %[xc], %[xc], %[cs]\n\t"
// 一行回写 = 4 个 k 步（E O E O）各带一个向量
// 回写阶段不发预取：hosted epilogue 的 C 行 load/store 会让同期发出的预取大半丢失（kbench 第 9/10 节）
#define KROW(EV)                                                                  \
    KS_EVEN EV(0, "p1", "z16", "z17") SWS_GATE_R SWS                              \
    KS_ODD  EV(1, "p2", "z18", "z19")                                             \
    KS_EVEN EV(2, "p3", "z16", "z17") SWS_GATE_R SWS                              \
    KS_ODD  EV(3, "p4", "z18", "z19")                                             \
    EV_ROW_ADV

// 列谓词 p1..p4 ← 上一微块列数 %[ecols]
#define SET_EPI_PREDS                                                             \
    "whilelo p1.d, xzr, %[ecols]\n\t"                                             \
    "subs x9, %[ecols], #8\n\t"                                                   \
    "csel x9, x9, xzr, gt\n\t"                                                    \
    "whilelo p2.d, xzr, x9\n\t"                                                   \
    "subs x9, x9, #8\n\t"                                                         \
    "csel x9, x9, xzr, gt\n\t"                                                    \
    "whilelo p3.d, xzr, x9\n\t"                                                   \
    "subs x9, x9, #8\n\t"                                                         \
    "csel x9, x9, xzr, gt\n\t"                                                    \
    "whilelo p4.d, xzr, x9\n\t"

#define KLOOP_INIT                                                                \
    "ptrue p0.d\n\t"                                                              \
    LDA_I "z0.d, p0/z, [%[xa]]\n\t"                                                \
    LDA_I "z1.d, p0/z, [%[xa], #1, mul vl]\n\t"                                    \
    LDB_I "z4.d, p0/z, [%[xb]]\n\t"                                                \
    LDB_I "z5.d, p0/z, [%[xb], #1, mul vl]\n\t"                                    \
    LDB_I "z6.d, p0/z, [%[xb], #2, mul vl]\n\t"                                    \
    LDB_I "z7.d, p0/z, [%[xb], #3, mul vl]\n\t"                                    \
    "add %[xa], %[xa], %[ase]\n\t"                                                \
    "add %[xb], %[xb], %[bs]\n\t"
// 余下 k 步（%[k] = 还需"算+载"的步数，可为 0）+ 最后一步只算
#define KLOOP_REST                                                                \
    "cbz %[k], 6f\n\t"                                                            \
    "1:\n\t"                                                                      \
    KS_EVEN PFL PFX SWS_GATE_K SWS                                                \
    "subs %[k], %[k], #1\n\t"                                                     \
    "b.eq 5f\n\t"                                                                 \
    KS_ODD PFC                                                                    \
    "subs %[k], %[k], #1\n\t"                                                     \
    "b.ne 1b\n\t"                                                                 \
    "6:\n\t" FM_EVEN                                                              \
    "b 7f\n\t"                                                                    \
    "5:\n\t" FM_ODD                                                               \
    "7:\n\t"

// 托管回写走线器：K 循环内每 2 步把 1 个 cache 行从 src（连续，Xp）拷到 dst（行距 ldb 的 X），stnt1d 不分配。
// 状态在 x12..x17：src dst cnt(本行剩余行数) dstj(行末跳) rows(剩余行) lpr(每行 cache 行数)；进出内核时从本结构装载/保存。
typedef struct { const char* src; char* dst; long cnt; long dstj; long rows; long lpr; } sw_t;
// cbuf / Xp 行距填充（double 数，编译期常量：走线器行末源跳用立即数）
#ifndef TRSM_NCPAD
#define TRSM_NCPAD 0   // 8（+64 B）实测三用例都略差 1–3%，对角阶段无变化
#endif
#define TRSM_NCPAD_BYTES (TRSM_NCPAD * 8)
static sw_t g_sw_none;   // rows=0：无事可做
#define SW_LOAD  "ldp x12, x13, [%[sw]]\n\t ldp x14, x15, [%[sw], #16]\n\t ldp x16, x17, [%[sw], #32]\n\t"
#define SW_SAVE  "stp x12, x13, [%[sw]]\n\t str x14, [%[sw], #16]\n\t str x16, [%[sw], #32]\n\t"
#ifndef TRSM_SW_NT
#define TRSM_SW_NT 1
#endif
#if TRSM_SW_NT
#define SW_ST "stnt1d "
#else
#define SW_ST "st1d "
#endif
// TRSM_SW_RATE=4：每 4 步 1 行（REST 循环按 k 的 bit1 门控，托管行按行计数 x10 的 bit0 门控）；默认 2
#ifndef TRSM_SW_RATE
#define TRSM_SW_RATE 2
#endif
#if TRSM_SW_RATE == 4
#define SWS_GATE_K "tbz %[k], #1, 9f\n\t"
#define SWS_GATE_R "tbz x10, #0, 9f\n\t"
#else
#define SWS_GATE_K
#define SWS_GATE_R
#endif
// 源行（Xp，刚写出、只碰一次）在 RRIP 下最先被逐出，走线器来读时多半已不在 L2 → 自己提前 TRSM_SW_PFD 字节预取
#ifndef TRSM_SW_PFD
#define TRSM_SW_PFD 0      // 实测 1K/2K/4K 提前量都无收益：托管回写的开销是写流量本身挤占 K 循环，不是源 miss
#endif
#if TRSM_SW_PFD > 0
#define SW_LD "prfm pldl1keep, [x12, #" TRSM_STR(TRSM_SW_PFD) "]\n\t ld1d z11.d, p0/z, [x12]\n\t"
#else
#define SW_LD "ld1d z11.d, p0/z, [x12]\n\t"
#endif
#define SWS                                                                       \
    "cbz x16, 9f\n\t"                                                             \
    SW_LD                                                                         \
    SW_ST "z11.d, p0, [x13]\n\t"                                                   \
    "add x12, x12, #64\n\t"                                                       \
    "add x13, x13, #64\n\t"                                                       \
    "subs x14, x14, #1\n\t"                                                       \
    "b.gt 9f\n\t"                                                                 \
    "add x13, x13, x15\n\t"                                                       \
    "add x12, x12, #" TRSM_STR(TRSM_NCPAD_BYTES) "\n\t"                           \
    "mov x14, x17\n\t"                                                            \
    "sub x16, x16, #1\n\t"                                                        \
    "9:\n\t"
#define SWCLOB "x12","x13","x14","x15","x16","x17"

// 预取任务：游标 1 从 p 起，每 pfn 行（64 B）后跳 pfj 字节；游标 2 从 p2 起连续（步长 pfs2，哑时 0）；各每 2 步 1 行；sw 为托管回写状态（可 NULL）
typedef struct { const char* p; long pfn, pfj; const char* p2; const char* pc; long pcs; sw_t* sw; } pf_t;

// K 循环：ZA(16×32) += Σ_k A[k](16) ⊗ B[k](32)。pa：A（每 k 两个相邻向量，步长 as 字节）；
// pb：B 第 k 行 32 列起点，行距 bs；K ≥ 1；cols ∈ [1,32]；进入前 ZA 须已清零。
#define DEF_KLOOP_PLAIN(NAME)                                                                                     \
static inline void NAME(const double* pa, long ase, long aso, const double* pb, long bs, long K, long cols, const pf_t* pf) \
{                                                                                                                 \
    const double* xa = pa;                                                                                        \
    const double* xb = pb;                                                                                        \
    const char* xpf = pf->p;                                                                                      \
    const char* xpf2 = pf->p2;                                                                                    \
    const char* xpc = pf->pc;                                                                                     \
    long k = K - 1, pfc = pf->pfn;                                                                                 \
    sw_t* sw = pf->sw ? pf->sw : &g_sw_none;                                                                      \
    (void)cols;                                                                                                   \
    asm volatile(                                                                                                 \
        SW_LOAD                                                                                                   \
        KLOOP_INIT                                                                                                \
        KLOOP_REST                                                                                                \
        SW_SAVE                                                                                                   \
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [k] "+&r"(k), [xpf] "+&r"(xpf), [pfc] "+&r"(pfc), [xpf2] "+&r"(xpf2), [xpc] "+&r"(xpc) \
        : [ase] "r"(ase), [aso] "r"(aso), [bs] "r"(bs), [pfn] "r"(pf->pfn), [pfj] "r"(pf->pfj), [pcs] "r"(pf->pcs), [sw] "r"(sw) \
        : "memory", "cc", "x9", SWCLOB, VCLOB, PCLOB);                                                            \
}
DEF_KLOOP_PLAIN(kloop_plain)

// K = 64 专用：64 步全部托管回写（15 行 KROW + 3 步 + 末步只算并带最后一组），因此走线器预取只能放在托管步里
// （TRSM_HOSTPF=0 时不放预取，用于对比）。
#ifndef TRSM_HOSTPF
#define TRSM_HOSTPF 1
#endif
// L/X 走线器按每两步一个 cache 行发预取，与调度器的 K/2 行预算一致。
#if TRSM_HOSTPF
#define KROWP(EV)                                                                 \
    KS_EVEN EV(0, "p1", "z16", "z17") PFL PFX SWS_GATE_R SWS                      \
    KS_ODD  EV(1, "p2", "z18", "z19") PFC                                         \
    KS_EVEN EV(2, "p3", "z16", "z17") PFL PFX SWS_GATE_R SWS                      \
    KS_ODD  EV(3, "p4", "z18", "z19")                                             \
    EV_ROW_ADV
#define HPF1 PFL PFX
#define HPF2 PFC
#else
#define KROWP(EV) KROW(EV)
#define HPF1
#define HPF2
#endif
// 同上，且在前 64 步里把暂存（上一微块的 ZA 转存）回写到 xc（行距 cs，列数 ecols）：SUB 为 C -= 暂存，SET 为 C = 暂存。
// 要求 K ≥ 65。
#define DEF_KLOOP_EPI(NAME, EV)                                                                                   \
static inline void NAME(const double* pa, long ase, long aso, const double* pb, long bs, long K, long cols, const pf_t* pf, \
                        const double* scratch, double* c, long cs, long ecols)                                    \
{                                                                                                                 \
    const double* xa = pa;                                                                                        \
    const double* xb = pb;                                                                                        \
    const char* xpf = pf->p;                                                                                      \
    const char* xpf2 = pf->p2;                                                                                    \
    const char* xpc = pf->pc;                                                                                     \
    const char* xs = (const char*)scratch;                                                                        \
    char* xc = (char*)c;                                                                                          \
    long k = K - 65, pfc = pf->pfn;                                                                                \
    sw_t* sw = pf->sw ? pf->sw : &g_sw_none;                                                                      \
    (void)cols;                                                                                                   \
    asm volatile(                                                                                                 \
        SW_LOAD                                                                                                   \
        SET_EPI_PREDS                                                                                             \
        KLOOP_INIT                                                                                                \
        "mov x10, #8\n\t"                                                                                         \
        "2:\n\t" KROWP(EV)                                                                                        \
        "subs x10, x10, #1\n\t"                                                                                   \
        "b.ne 2b\n\t"                                                                                             \
        "sub %[xs], %[xs], #3840\n\t"                                                                             \
        "mov x10, #8\n\t"                                                                                         \
        "3:\n\t" KROWP(EV)                                                                                        \
        "subs x10, x10, #1\n\t"                                                                                   \
        "b.ne 3b\n\t"                                                                                             \
        KLOOP_REST                                                                                                \
        SW_SAVE                                                                                                   \
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [k] "+&r"(k), [xpf] "+&r"(xpf), [pfc] "+&r"(pfc), [xpf2] "+&r"(xpf2), [xpc] "+&r"(xpc), [xs] "+&r"(xs), [xc] "+&r"(xc) \
        : [ase] "r"(ase), [aso] "r"(aso), [bs] "r"(bs), [cs] "r"(cs), [ecols] "r"(ecols), [pfn] "r"(pf->pfn), [pfj] "r"(pf->pfj), [pcs] "r"(pf->pcs), [sw] "r"(sw) \
        : "memory", "cc", "x9", "x10", SWCLOB, VCLOB, PCLOB);                                                     \
}
DEF_KLOOP_EPI(kloop_epi_set, EV_SET)
#undef LDA_CUR
#define LDA_CUR LDA_NT_INSN
DEF_KLOOP_PLAIN(kloop_plain_nt)
DEF_KLOOP_EPI(kloop_epi_sub, EV_SUB)
#undef LDA_CUR
#define LDA_CUR "ld1d "
DEF_KLOOP_EPI(kloop_epi_sub_t, EV_SUB)

#define DEF_KLOOP_EPI64(NAME, EV)                                                                                 \
static inline void NAME(const double* pa, long ase, long aso, const double* pb, long bs, long cols, const pf_t* pf, \
                        const double* scratch, double* c, long cs, long ecols)                                    \
{                                                                                                                 \
    const double* xa = pa;                                                                                        \
    const double* xb = pb;                                                                                        \
    const char* xpf = pf->p;                                                                                      \
    const char* xpf2 = pf->p2;                                                                                    \
    const char* xpc = pf->pc;                                                                                     \
    const char* xs = (const char*)scratch;                                                                        \
    char* xc = (char*)c;                                                                                          \
    long pfc = pf->pfn;                                                                                           \
    sw_t* sw = pf->sw ? pf->sw : &g_sw_none;                                                                      \
    (void)cols;                                                                                                   \
    asm volatile(                                                                                                 \
        SW_LOAD                                                                                                   \
        SET_EPI_PREDS                                                                                             \
        KLOOP_INIT                                                                                                \
        "mov x10, #8\n\t"                                                                                         \
        "2:\n\t" KROWP(EV)                                                                                        \
        "subs x10, x10, #1\n\t"                                                                                   \
        "b.ne 2b\n\t"                                                                                             \
        "sub %[xs], %[xs], #3840\n\t"                                                                             \
        "mov x10, #7\n\t"                                                                                         \
        "3:\n\t" KROWP(EV)                                                                                        \
        "subs x10, x10, #1\n\t"                                                                                   \
        "b.ne 3b\n\t"                                                                                             \
        KS_EVEN EV(0, "p1", "z16", "z17") HPF1                                                                    \
        KS_ODD  EV(1, "p2", "z18", "z19") HPF2                                                                    \
        KS_EVEN EV(2, "p3", "z16", "z17") HPF1                                                                    \
        FM_ODD  EV(3, "p4", "z18", "z19")                                                                         \
        SW_SAVE                                                                                                   \
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [xpf] "+&r"(xpf), [pfc] "+&r"(pfc), [xpf2] "+&r"(xpf2), [xpc] "+&r"(xpc), [xs] "+&r"(xs), [xc] "+&r"(xc) \
        : [ase] "r"(ase), [aso] "r"(aso), [bs] "r"(bs), [cs] "r"(cs), [ecols] "r"(ecols), [pfn] "r"(pf->pfn), [pfj] "r"(pf->pfj), [pcs] "r"(pf->pcs), [sw] "r"(sw) \
        : "memory", "cc", "x9", "x10", SWCLOB, VCLOB, PCLOB);                                                     \
}
DEF_KLOOP_EPI64(kloop_epi64_set, EV_SET)
#undef LDA_CUR
#define LDA_CUR LDA_NT_INSN
DEF_KLOOP_EPI64(kloop_epi64_sub, EV_SUB)
// 对角内核的 ldnt1d 版本：面板驻留模式下对角条每个面板只用一遍，不该挤占 L2 里的面板缓冲
DEF_KLOOP_EPI(kloop_epi_set_nt, EV_SET)
DEF_KLOOP_EPI64(kloop_epi64_set_nt, EV_SET)
#undef LDA_CUR
#define LDA_CUR "ld1d "
DEF_KLOOP_EPI64(kloop_epi64_sub_t, EV_SUB)

#undef K_MOP
#define K_MOP "fmops "
#undef LDA_CUR
#define LDA_CUR LDA_NT_INSN
DEF_KLOOP_PLAIN(kloop_plain_direct)
#undef LDA_CUR
#define LDA_CUR "ld1d "
DEF_KLOOP_PLAIN(kloop_plain_direct_t)
#undef K_MOP
#define K_MOP "fmopa "
#undef LDA_CUR
#define LDA_CUR "ld1d "
// Full-panel kernels use immediate A/C/Xp strides to reduce GPR operands.
// Dispatch requires NC=128, KC=64 and n divisible by 128; every GEMM chunk is full.
#define DEF_FIXED_EPI64(NAME, EV)                                                                                 \
static inline void NAME(const double* pa, long ase, long aso, const double* pb, long bs, long cols, const pf_t* pf, \
                        const double* scratch, double* c, long cs, long ecols)                                    \
{                                                                                                                 \
    const double* xa = pa;                                                                                        \
    const double* xb = pb;                                                                                        \
    const char* xpf = pf->p;                                                                                      \
    const char* xpf2 = pf->p2;                                                                                    \
    const char* xpc = pf->pc;                                                                                     \
    const char* xs = (const char*)scratch;                                                                        \
    char* xc = (char*)c;                                                                                          \
    long pfc = pf->pfn;                                                                                           \
    sw_t* sw = pf->sw ? pf->sw : &g_sw_none;                                                                      \
    (void)cols;                                                                                                   \
    asm volatile(                                                                                                 \
        SW_LOAD                                                                                                   \
        SET_EPI_PREDS                                                                                             \
        KLOOP_INIT                                                                                                \
        "mov x10, #8\n\t"                                                                                         \
        "2:\n\t" KROWP(EV)                                                                                        \
        "subs x10, x10, #1\n\t"                                                                                   \
        "b.ne 2b\n\t"                                                                                             \
        "sub %[xs], %[xs], #3840\n\t"                                                                             \
        "mov x10, #7\n\t"                                                                                         \
        "3:\n\t" KROWP(EV)                                                                                        \
        "subs x10, x10, #1\n\t"                                                                                   \
        "b.ne 3b\n\t"                                                                                             \
        KS_EVEN EV(0, "p1", "z16", "z17") HPF1                                                                    \
        KS_ODD  EV(1, "p2", "z18", "z19") HPF2                                                                    \
        KS_EVEN EV(2, "p3", "z16", "z17") HPF1                                                                    \
        FM_ODD  EV(3, "p4", "z18", "z19")                                                                         \
        SW_SAVE                                                                                                   \
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [xpf] "+&r"(xpf), [pfc] "+&r"(pfc), [xpf2] "+&r"(xpf2), [xpc] "+&r"(xpc), [xs] "+&r"(xs), [xc] "+&r"(xc) \
        : [pfn] "r"(pf->pfn), [pfj] "r"(pf->pfj), [pcs] "r"(pf->pcs), [sw] "r"(sw) \
        : "memory", "cc", "x9", "x10", SWCLOB, VCLOB, PCLOB);                                                     \
}
#define DEF_FIXED_PLAIN(NAME)                                                                                     \
static inline void NAME(const double* pa, long ase, long aso, const double* pb, long bs, long K, long cols, const pf_t* pf) \
{                                                                                                                 \
    const double* xa = pa;                                                                                        \
    const double* xb = pb;                                                                                        \
    const char* xpf = pf->p;                                                                                      \
    const char* xpf2 = pf->p2;                                                                                    \
    const char* xpc = pf->pc;                                                                                     \
    long k = 63, pfc = pf->pfn;                                                                                 \
    sw_t* sw = pf->sw ? pf->sw : &g_sw_none;                                                                      \
    (void)cols;                                                                                                   \
    asm volatile(                                                                                                 \
        SW_LOAD                                                                                                   \
        KLOOP_INIT                                                                                                \
        KLOOP_REST                                                                                                \
        SW_SAVE                                                                                                   \
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [k] "+&r"(k), [xpf] "+&r"(xpf), [pfc] "+&r"(pfc), [xpf2] "+&r"(xpf2), [xpc] "+&r"(xpc) \
        : [pfn] "r"(pf->pfn), [pfj] "r"(pf->pfj), [pcs] "r"(pf->pcs), [sw] "r"(sw) \
        : "memory", "cc", "x9", SWCLOB, VCLOB, PCLOB);                                                            \
}
#undef KS
#define KS(a0, a1, b0, b1, b2, b3, n0, n1, m0, m1, m2, m3, ADV)                  \
    LDA_I n0 ".d, p0/z, [%[xa]]\n\t"                                              \
    K_MOP "za0.d, p0/m, p0/m, " a0 ".d, " b0 ".d\n\t"                             \
    K_MOP "za1.d, p0/m, p0/m, " a0 ".d, " b1 ".d\n\t"                             \
    LDA_I n1 ".d, p0/z, [%[xa], #1, mul vl]\n\t"                                  \
    K_MOP "za2.d, p0/m, p0/m, " a0 ".d, " b2 ".d\n\t"                             \
    K_MOP "za3.d, p0/m, p0/m, " a0 ".d, " b3 ".d\n\t"                             \
    LDB_I m0 ".d, p0/z, [%[xb]]\n\t"                                              \
    K_MOP "za4.d, p0/m, p0/m, " a1 ".d, " b0 ".d\n\t"                             \
    LDB_I m1 ".d, p0/z, [%[xb], #1, mul vl]\n\t"                                  \
    K_MOP "za5.d, p0/m, p0/m, " a1 ".d, " b1 ".d\n\t"                             \
    LDB_I m2 ".d, p0/z, [%[xb], #2, mul vl]\n\t"                                  \
    K_MOP "za6.d, p0/m, p0/m, " a1 ".d, " b2 ".d\n\t"                             \
    LDB_I m3 ".d, p0/z, [%[xb], #3, mul vl]\n\t"                                  \
    K_MOP "za7.d, p0/m, p0/m, " a1 ".d, " b3 ".d\n\t"                             \
    "add %[xa], %[xa], " ADV "\n\t"                                               \
    "add %[xb], %[xb], #1024\n\t"
#undef KLOOP_INIT
#define KLOOP_INIT                                                                \
    "ptrue p0.d\n\t"                                                              \
    LDA_I "z0.d, p0/z, [%[xa]]\n\t"                                                \
    LDA_I "z1.d, p0/z, [%[xa], #1, mul vl]\n\t"                                    \
    LDB_I "z4.d, p0/z, [%[xb]]\n\t"                                                \
    LDB_I "z5.d, p0/z, [%[xb], #1, mul vl]\n\t"                                    \
    LDB_I "z6.d, p0/z, [%[xb], #2, mul vl]\n\t"                                    \
    LDB_I "z7.d, p0/z, [%[xb], #3, mul vl]\n\t"                                    \
    "add %[xa], %[xa], #128\n\t"                                                \
    "add %[xb], %[xb], #1024\n\t"
#undef KS_EVEN
#define KS_EVEN KS("z0", "z1", "z4", "z5", "z6", "z7", "z8", "z9", "z12", "z13", "z14", "z15", "#128")
#undef KS_ODD
#define KS_ODD  KS("z8", "z9", "z12", "z13", "z14", "z15", "z0", "z1", "z4", "z5", "z6", "z7", "#128")
#undef EV_ROW_ADV
#define EV_ROW_ADV "add %[xs], %[xs], #512\n\t add %[xc], %[xc], #1024\n\t"
#undef SET_EPI_PREDS
#define SET_EPI_PREDS "ptrue p1.d\n\t ptrue p2.d\n\t ptrue p3.d\n\t ptrue p4.d\n\t"
#undef LDA_CUR
#define LDA_CUR LDA_NT_INSN
DEF_FIXED_PLAIN(kloop_fixed_plain)
DEF_FIXED_EPI64(kloop_fixed_epi, EV_SUB)
#undef K_MOP
#define K_MOP "fmops "
DEF_FIXED_PLAIN(kloop_fixed_direct)
#undef K_MOP
#define K_MOP "fmopa "
#undef LDA_CUR
#define LDA_CUR "ld1d "
#undef KS
#define KS(a0, a1, b0, b1, b2, b3, n0, n1, m0, m1, m2, m3, ADV)                  \
    LDA_I n0 ".d, p0/z, [%[xa]]\n\t"                                              \
    K_MOP "za0.d, p0/m, p0/m, " a0 ".d, " b0 ".d\n\t"                             \
    K_MOP "za1.d, p0/m, p0/m, " a0 ".d, " b1 ".d\n\t"                             \
    LDA_I n1 ".d, p0/z, [%[xa], #1, mul vl]\n\t"                                  \
    K_MOP "za2.d, p0/m, p0/m, " a0 ".d, " b2 ".d\n\t"                             \
    K_MOP "za3.d, p0/m, p0/m, " a0 ".d, " b3 ".d\n\t"                             \
    LDB_I m0 ".d, p0/z, [%[xb]]\n\t"                                              \
    K_MOP "za4.d, p0/m, p0/m, " a1 ".d, " b0 ".d\n\t"                             \
    LDB_I m1 ".d, p0/z, [%[xb], #1, mul vl]\n\t"                                  \
    K_MOP "za5.d, p0/m, p0/m, " a1 ".d, " b1 ".d\n\t"                             \
    LDB_I m2 ".d, p0/z, [%[xb], #2, mul vl]\n\t"                                  \
    K_MOP "za6.d, p0/m, p0/m, " a1 ".d, " b2 ".d\n\t"                             \
    LDB_I m3 ".d, p0/z, [%[xb], #3, mul vl]\n\t"                                  \
    K_MOP "za7.d, p0/m, p0/m, " a1 ".d, " b3 ".d\n\t"                             \
    "add %[xa], %[xa], " ADV "\n\t"                                               \
    "add %[xb], %[xb], %[bs]\n\t"
#undef KLOOP_INIT
#define KLOOP_INIT                                                                \
    "ptrue p0.d\n\t"                                                              \
    LDA_I "z0.d, p0/z, [%[xa]]\n\t"                                                \
    LDA_I "z1.d, p0/z, [%[xa], #1, mul vl]\n\t"                                    \
    LDB_I "z4.d, p0/z, [%[xb]]\n\t"                                                \
    LDB_I "z5.d, p0/z, [%[xb], #1, mul vl]\n\t"                                    \
    LDB_I "z6.d, p0/z, [%[xb], #2, mul vl]\n\t"                                    \
    LDB_I "z7.d, p0/z, [%[xb], #3, mul vl]\n\t"                                    \
    "add %[xa], %[xa], %[ase]\n\t"                                                \
    "add %[xb], %[xb], %[bs]\n\t"
#undef KS_EVEN
#define KS_EVEN KS("z0", "z1", "z4", "z5", "z6", "z7", "z8", "z9", "z12", "z13", "z14", "z15", "%[aso]")
#undef KS_ODD
#define KS_ODD  KS("z8", "z9", "z12", "z13", "z14", "z15", "z0", "z1", "z4", "z5", "z6", "z7", "%[ase]")
#undef EV_ROW_ADV
#define EV_ROW_ADV "add %[xs], %[xs], #512\n\t add %[xc], %[xc], %[cs]\n\t"
#undef SET_EPI_PREDS
#define SET_EPI_PREDS                                                             \
    "whilelo p1.d, xzr, %[ecols]\n\t"                                             \
    "subs x9, %[ecols], #8\n\t"                                                   \
    "csel x9, x9, xzr, gt\n\t"                                                    \
    "whilelo p2.d, xzr, x9\n\t"                                                   \
    "subs x9, x9, #8\n\t"                                                         \
    "csel x9, x9, xzr, gt\n\t"                                                    \
    "whilelo p3.d, xzr, x9\n\t"                                                   \
    "subs x9, x9, #8\n\t"                                                         \
    "csel x9, x9, xzr, gt\n\t"                                                    \
    "whilelo p4.d, xzr, x9\n\t"
static const uint64_t g_tbl_idx[4][8];   // tbl2 索引向量（定义见打包节）

// ---------------------------------------------------------------- 带交错转置的 K 循环
// 在 K 循环里顺带把"下一条 A"（16 行 × K 列的 L 子块）转置成稠密 k 主序：每 16 个 k 步完成一个 8×8 块
// （8 ld1d、8 zip、8 tbl2、8 tbl2、8 st1d，每步 ≤ 4 条额外指令），32 步完成 8 个 k 的全部 16 行。
// 寄存器：K 循环 z0/1/4-7/8/9/12-15，宿主回写 z2/z3，转置工作区 z16-z27，tbl2 索引 z28-z31。
// 块顺序：(kb, 行 0-7) → (kb, 行 8-15) → (kb+8, 行 0-7) …；xl 为源行指针（步进 lda8），xd 为目标（每 k 128 B）。
#ifndef TR_LOOKAHEAD
#define TR_LOOKAHEAD 0            // 转置装载时对同一行提前预取的字节数（0 关闭；实测 128–1024 均无收益）
#endif
#if TR_LOOKAHEAD > 0
#if TRSM_PF_LOAD
#define TH_PF "ld1d z11.d, p0/z, [%[xl], #" TR_STR(TR_LOOKAHEAD) "]\n\t"   /* 仅当 TR_LOOKAHEAD 为 64 的倍数且 ≤448 时合法 */
#else
#define TH_PF "prfm pldl1keep, [%[xl], #" TR_STR(TR_LOOKAHEAD) "]\n\t"
#endif
#else
#define TH_PF
#endif
#define TR_STR_(x) #x
#define TR_STR(x) TR_STR_(x)
#define TH_LD(za, zb)                                                             \
    "ld1d " za ".d, p0/z, [%[xl]]\n\t"                                            \
    TH_PF                                                                         \
    "add %[xl], %[xl], %[lda8]\n\t"                                               \
    "ld1d " zb ".d, p0/z, [%[xl]]\n\t"                                            \
    TH_PF                                                                         \
    "add %[xl], %[xl], %[lda8]\n\t"
#define TH0 TH_LD("z16", "z17")
#define TH1 TH_LD("z18", "z19")
#define TH2 TH_LD("z20", "z21")
#define TH3 TH_LD("z22", "z23")
#define TH4 "zip1 z24.d, z16.d, z17.d\n\t zip1 z25.d, z18.d, z19.d\n\t"
#define TH5 "zip2 z26.d, z16.d, z17.d\n\t zip2 z27.d, z18.d, z19.d\n\t"
#define TH6 "zip1 z16.d, z20.d, z21.d\n\t zip1 z17.d, z22.d, z23.d\n\t"
#define TH7 "zip2 z18.d, z20.d, z21.d\n\t zip2 z19.d, z22.d, z23.d\n\t"
#define TH8 "tbl z20.d, {z24.d, z25.d}, z28.d\n\t tbl z22.d, {z24.d, z25.d}, z29.d\n\t"
#define TH9 "tbl z21.d, {z16.d, z17.d}, z28.d\n\t tbl z23.d, {z16.d, z17.d}, z29.d\n\t"
#define TH10 "tbl z24.d, {z26.d, z27.d}, z28.d\n\t tbl z16.d, {z26.d, z27.d}, z29.d\n\t"
#define TH11 "tbl z25.d, {z18.d, z19.d}, z28.d\n\t tbl z17.d, {z18.d, z19.d}, z29.d\n\t"
#define TH12 "tbl z26.d, {z20.d, z21.d}, z30.d\n\t tbl z27.d, {z20.d, z21.d}, z31.d\n\t"           \
             "st1d z26.d, p0, [%[xd]]\n\t st1d z27.d, p0, [%[xd], #2, mul vl]\n\t"
#define TH13 "tbl z18.d, {z22.d, z23.d}, z30.d\n\t tbl z19.d, {z22.d, z23.d}, z31.d\n\t"           \
             "st1d z18.d, p0, [%[xd], #4, mul vl]\n\t st1d z19.d, p0, [%[xd], #6, mul vl]\n\t"     \
             "add %[xd], %[xd], #512\n\t"
#define TH14 "tbl z26.d, {z24.d, z25.d}, z30.d\n\t tbl z27.d, {z24.d, z25.d}, z31.d\n\t"           \
             "st1d z26.d, p0, [%[xd]]\n\t st1d z27.d, p0, [%[xd], #2, mul vl]\n\t"
#define TH15 "tbl z18.d, {z16.d, z17.d}, z30.d\n\t tbl z19.d, {z16.d, z17.d}, z31.d\n\t"           \
             "st1d z18.d, p0, [%[xd], #4, mul vl]\n\t st1d z19.d, p0, [%[xd], #6, mul vl]\n\t"     \
             "add %[xd], %[xd], #512\n\t"
// 块结束后：行 0-7 块 → 目标退回 start+64（行 8-15 的槽），源已在行 8；行 8-15 块 → 目标到下一 kb 起点，源回行 0 并右移 8 列
#define TFIX_H0 "sub %[xd], %[xd], #960\n\t"
#define TFIX_H1 "sub %[xd], %[xd], #64\n\t sub %[xl], %[xl], x10\n\t"
#define EVX_SUB(j, pj) EV_SUB(j, pj, "z2", "z3")
#define EVX_NONE(j, pj)
#define RA_ON EV_ROW_ADV
#define RA_OFF
// 16 步（一个 8×8 块）；E 为回写向量钩子（每 4 步一行），RA 为行推进
#define TB16(E, RA, LASTODD, PC, PF)                                              \
    KS_EVEN_T E(0, "p1") TH0 PF KS_ODD_T E(1, "p2") TH1 PF PC KS_EVEN_T E(2, "p3") TH2 PF KS_ODD_T E(3, "p4") TH3 PF RA \
    KS_EVEN_T E(0, "p1") TH4 PF KS_ODD_T E(1, "p2") TH5 PF KS_EVEN_T E(2, "p3") TH6 PF KS_ODD_T E(3, "p4") TH7 PF RA \
    KS_EVEN_T E(0, "p1") TH8 PF KS_ODD_T E(1, "p2") TH9 PF PC KS_EVEN_T E(2, "p3") TH10 PF KS_ODD_T E(3, "p4") TH11 PF RA \
    KS_EVEN_T E(0, "p1") TH12 PF KS_ODD_T E(1, "p2") TH13 PF KS_EVEN_T E(2, "p3") TH14 PF LASTODD E(3, "p4") TH15 PF RA
#define PC_NONE
#define PF_NONE
#define PF_BOTH PFL PFX
#define TB_PAIR(E, RA, PC, PF) TB16(E, RA, KS_ODD_T, PC, PF) TFIX_H0 TB16(E, RA, KS_ODD_T, PC, PF) TFIX_H1
#define TB_PAIR_LAST(E, RA, PC, PF) TB16(E, RA, KS_ODD_T, PC, PF) TFIX_H0 TB16(E, RA, FM_ODD, PC, PF) TFIX_H1
// KLOOP_INIT 用 %[ase]；转置版在 asm 内以 x11 别名给出
#define KLOOP_INIT_T                                                              \
    "ptrue p0.d\n\t"                                                              \
    "ld1d z0.d, p0/z, [%[xa]]\n\t"                                                \
    "ld1d z1.d, p0/z, [%[xa], #1, mul vl]\n\t"                                    \
    "ld1d z4.d, p0/z, [%[xb]]\n\t"                                                \
    "ld1d z5.d, p0/z, [%[xb], #1, mul vl]\n\t"                                    \
    "ld1d z6.d, p0/z, [%[xb], #2, mul vl]\n\t"                                    \
    "ld1d z7.d, p0/z, [%[xb], #3, mul vl]\n\t"                                    \
    "add %[xa], %[xa], x11\n\t"                                                   \
    "add %[xb], %[xb], %[bs]\n\t"

// K 循环 + 下一条转置（+ 宿主回写，要求 K ≥ 96）。K % 32 == 0。xl：源 L 子块行 0 列 kb0；xd：目标（稠密 128 B/k）。
// 每个 K 循环转置 K/16 个 8×8 块（= 16 行 × K/4 列），本条 4 个微块合起来转置整条。
// （转置版 A 恒为稠密 128 B/k 布局：ase/aso 由 x11 提供，省操作数）
static void kloop_tr_epi(const double* pa, const double* pb, long bs, long K, const pf_t* pf,
                         const double* scratch, double* c, long cs, long ecols,
                         const double* lsrc, long lda8, double* dst)
{
    const double* xa = pa;
    const double* xb = pb;
    const char* xpf = pf->p;
    const char* xpf2 = pf->p2;
    const char* xpc = pf->pc;
    const char* xs = (const char*)scratch;
    char* xc = (char*)c;
    const char* xl = (const char*)lsrc;
    char* xd = (char*)dst;
    long pairs = K / 32 - 2, pfc = pf->pfn;
    asm volatile(
        SET_EPI_PREDS
        "lsl x10, %[lda8], #4\n\t"
        "sub x10, x10, #64\n\t"
        "mov x11, #128\n\t"
        "ld1d z28.d, p0/z, [%[idx]]\n\t"
        "ld1d z29.d, p0/z, [%[idx], #1, mul vl]\n\t"
        "ld1d z30.d, p0/z, [%[idx], #2, mul vl]\n\t"
        "ld1d z31.d, p0/z, [%[idx], #3, mul vl]\n\t"
        KLOOP_INIT_T
        TB_PAIR(EVX_SUB, RA_ON, PC_NONE, PF_NONE)
        "sub %[xs], %[xs], #3840\n\t"
        TB_PAIR(EVX_SUB, RA_ON, PC_NONE, PF_NONE)
        "subs %[pairs], %[pairs], #1\n\t"
        "b.eq 6f\n\t"
        "5:\n\t"
        TB_PAIR(EVX_NONE, RA_OFF, PFC, PF_BOTH)
        "subs %[pairs], %[pairs], #1\n\t"
        "b.ne 5b\n\t"
        "6:\n\t"
        TB_PAIR_LAST(EVX_NONE, RA_OFF, PFC, PF_BOTH)
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [pairs] "+&r"(pairs), [xpf] "+&r"(xpf), [pfc] "+&r"(pfc), [xpf2] "+&r"(xpf2),
          [xpc] "+&r"(xpc), [xs] "+&r"(xs), [xc] "+&r"(xc), [xl] "+&r"(xl), [xd] "+&r"(xd)
        : [bs] "r"(bs), [cs] "r"(cs), [ecols] "r"(ecols),
          [pfn] "r"(pf->pfn), [pfj] "r"(pf->pfj), [pcs] "r"(pf->pcs), [lda8] "r"(lda8), [idx] "r"(g_tbl_idx)
        : "memory", "cc", "x9", "x10", "x11", VCLOB, PCLOB);
}
#define l16m64 "x10"
static void kloop_tr_noepi(const double* pa, const double* pb, long bs, long K, const pf_t* pf,
                           const double* lsrc, long lda8, double* dst)
{
    const double* xa = pa;
    const double* xb = pb;
    const char* xpf = pf->p;
    const char* xpf2 = pf->p2;
    const char* xpc = pf->pc;
    const char* xl = (const char*)lsrc;
    char* xd = (char*)dst;
    long pairs = K / 32, pfc = pf->pfn;
    asm volatile(
        "ptrue p0.d\n\t"
        "lsl x10, %[lda8], #4\n\t"
        "sub x10, x10, #64\n\t"
        "ld1d z28.d, p0/z, [%[idx]]\n\t"
        "ld1d z29.d, p0/z, [%[idx], #1, mul vl]\n\t"
        "ld1d z30.d, p0/z, [%[idx], #2, mul vl]\n\t"
        "ld1d z31.d, p0/z, [%[idx], #3, mul vl]\n\t"
        "mov x11, #128\n\t"
        KLOOP_INIT_T
        "subs %[pairs], %[pairs], #1\n\t"
        "b.eq 6f\n\t"
        "5:\n\t"
        TB_PAIR(EVX_NONE, RA_OFF, PFC, PF_BOTH)
        "subs %[pairs], %[pairs], #1\n\t"
        "b.ne 5b\n\t"
        "6:\n\t"
        TB_PAIR_LAST(EVX_NONE, RA_OFF, PFC, PF_BOTH)
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [pairs] "+&r"(pairs), [xpf] "+&r"(xpf), [pfc] "+&r"(pfc), [xpf2] "+&r"(xpf2),
          [xpc] "+&r"(xpc), [xl] "+&r"(xl), [xd] "+&r"(xd)
        : [bs] "r"(bs), [pfn] "r"(pf->pfn), [pfj] "r"(pf->pfj), [pcs] "r"(pf->pcs), [lda8] "r"(lda8), [idx] "r"(g_tbl_idx)
        : "memory", "cc", "x9", "x10", "x11", VCLOB, PCLOB);
}
#undef l16m64

// ZA 按自然顺序转存到 4 KB 暂存：ZA 向量 n = r*8 + t（tile t 的第 r 行）位于 scratch + n*64。
// 于是块内第 R 行的 32 个值连续：R<8 在 scratch + R*512，R≥8 在 scratch + (R-8)*512 + 256。
#define ZA_DUMP_GROUP(G)                                                          \
    "mov w12, #" #G "\n\t"                                                        \
    "str za[w12, 0], [%[b]]\n\t"                                                  \
    "str za[w12, 1], [%[b], #1, mul vl]\n\t"                                      \
    "str za[w12, 2], [%[b], #2, mul vl]\n\t"                                      \
    "str za[w12, 3], [%[b], #3, mul vl]\n\t"                                      \
    "str za[w12, 4], [%[b], #4, mul vl]\n\t"                                      \
    "str za[w12, 5], [%[b], #5, mul vl]\n\t"                                      \
    "str za[w12, 6], [%[b], #6, mul vl]\n\t"                                      \
    "str za[w12, 7], [%[b], #7, mul vl]\n\t"                                      \
    "str za[w12, 8], [%[b], #8, mul vl]\n\t"                                      \
    "str za[w12, 9], [%[b], #9, mul vl]\n\t"                                      \
    "str za[w12, 10], [%[b], #10, mul vl]\n\t"                                    \
    "str za[w12, 11], [%[b], #11, mul vl]\n\t"                                    \
    "str za[w12, 12], [%[b], #12, mul vl]\n\t"                                    \
    "str za[w12, 13], [%[b], #13, mul vl]\n\t"                                    \
    "str za[w12, 14], [%[b], #14, mul vl]\n\t"                                    \
    "str za[w12, 15], [%[b], #15, mul vl]\n\t"                                    \
    "add %[b], %[b], #1024\n\t"
static inline void za_dump(double* scratch)
{
    char* b = (char*)scratch;
    asm volatile(
        ZA_DUMP_GROUP(0) ZA_DUMP_GROUP(16) ZA_DUMP_GROUP(32) ZA_DUMP_GROUP(48)
        : [b] "+&r"(b) : : "memory", "x12");
}

// 独立回写（无宿主 K 循环时用）：C 块 16×32 子块 -= / = 暂存
#define EPI_SUB_ROW                                                               \
    "ld1d z0.d, p0/z, [%[xs]]\n\t"                                                \
    "ld1d z1.d, p0/z, [%[xs], #1, mul vl]\n\t"                                    \
    "ld1d z2.d, p0/z, [%[xs], #2, mul vl]\n\t"                                    \
    "ld1d z3.d, p0/z, [%[xs], #3, mul vl]\n\t"                                    \
    "ld1d z4.d, p1/z, [%[xc]]\n\t"                                                \
    "ld1d z5.d, p2/z, [%[xc], #1, mul vl]\n\t"                                    \
    "ld1d z6.d, p3/z, [%[xc], #2, mul vl]\n\t"                                    \
    "ld1d z7.d, p4/z, [%[xc], #3, mul vl]\n\t"                                    \
    "fsub z4.d, z4.d, z0.d\n\t"                                                   \
    "fsub z5.d, z5.d, z1.d\n\t"                                                   \
    "fsub z6.d, z6.d, z2.d\n\t"                                                   \
    "fsub z7.d, z7.d, z3.d\n\t"                                                   \
    "st1d z4.d, p1, [%[xc]]\n\t"                                                  \
    "st1d z5.d, p2, [%[xc], #1, mul vl]\n\t"                                      \
    "st1d z6.d, p3, [%[xc], #2, mul vl]\n\t"                                      \
    "st1d z7.d, p4, [%[xc], #3, mul vl]\n\t"                                      \
    "add %[xs], %[xs], #512\n\t"                                                  \
    "add %[xc], %[xc], %[cs]\n\t"
#define EPI_SET_ROW                                                               \
    "ld1d z0.d, p0/z, [%[xs]]\n\t"                                                \
    "ld1d z1.d, p0/z, [%[xs], #1, mul vl]\n\t"                                    \
    "ld1d z2.d, p0/z, [%[xs], #2, mul vl]\n\t"                                    \
    "ld1d z3.d, p0/z, [%[xs], #3, mul vl]\n\t"                                    \
    "st1d z0.d, p1, [%[xc]]\n\t"                                                  \
    "st1d z1.d, p2, [%[xc], #1, mul vl]\n\t"                                      \
    "st1d z2.d, p3, [%[xc], #2, mul vl]\n\t"                                      \
    "st1d z3.d, p4, [%[xc], #3, mul vl]\n\t"                                      \
    "add %[xs], %[xs], #512\n\t"                                                  \
    "add %[xc], %[xc], %[cs]\n\t"
#define ROWS8(R) R R R R R R R R
#define DEF_EPI_STD(NAME, ROW)                                                    \
static inline void NAME(const double* scratch, double* c, long cs, long cols)     \
{                                                                                 \
    const char* xs = (const char*)scratch;                                        \
    char* xc = (char*)c;                                                          \
    asm volatile(                                                                 \
        SET_COL_PREDS                                                             \
        ROWS8(ROW)                                                                \
        "sub %[xs], %[xs], #3840\n\t"                                             \
        ROWS8(ROW)                                                                \
        : [xs] "+&r"(xs), [xc] "+&r"(xc)                                          \
        : [cs] "r"(cs), [cols] "r"(cols)                                          \
        : "memory", "cc", "x9", VCLOB, PCLOB);                                    \
}
DEF_EPI_STD(epi_sub_std, EPI_SUB_ROW)
DEF_EPI_STD(epi_set_std, EPI_SET_ROW)

// Small row dimensions benefit from loading C into ZA and applying FMOPS directly.
// Expand every row explicitly: the looped transfer variant was slower on this CPU.
#define ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, R)                                  \
    "mov w12, #" #R "\n\t"                                                       \
    "add x13, %[xc], #0\n\t"                                                      \
    OP " {za" #T0 "h.d[w12, 0]}, p1" SUFFIX ", [x13]\n\t"                        \
    "add x13, %[xc], #64\n\t"                                                     \
    OP " {za" #T1 "h.d[w12, 0]}, p2" SUFFIX ", [x13]\n\t"                        \
    "add x13, %[xc], #128\n\t"                                                    \
    OP " {za" #T2 "h.d[w12, 0]}, p3" SUFFIX ", [x13]\n\t"                        \
    "add x13, %[xc], #192\n\t"                                                    \
    OP " {za" #T3 "h.d[w12, 0]}, p4" SUFFIX ", [x13]\n\t"                        \
    "add %[xc], %[xc], %[cs]\n\t"
#define ZA_C_ROWS8(OP, SUFFIX, T0, T1, T2, T3)                                   \
    ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 0)                                      \
    ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 1)                                      \
    ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 2)                                      \
    ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 3)                                      \
    ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 4)                                      \
    ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 5)                                      \
    ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 6)                                      \
    ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 7)
#define DEF_ZA_C_TRANSFER(NAME, OP, SUFFIX)                                     \
static inline void NAME(double* c, long cs, long cols)                          \
{                                                                               \
    char* xc = (char*)c;                                                        \
    asm volatile(                                                               \
        SET_COL_PREDS                                                           \
        ZA_C_ROWS8(OP, SUFFIX, 0, 1, 2, 3)                                     \
        ZA_C_ROWS8(OP, SUFFIX, 4, 5, 6, 7)                                     \
        : [xc] "+&r"(xc) : [cs] "r"(cs), [cols] "r"(cols)                       \
        : "memory", "cc", "x9", "x12", "x13", VCLOB, PCLOB);                    \
}
DEF_ZA_C_TRANSFER(za_load_direct, "ld1d", "/z")
DEF_ZA_C_TRANSFER(za_store_direct, "st1d", "")

// Complete 128-column panels have fixed predicates and a 1024-byte C row stride.
// The first vector uses C directly, avoiding a redundant address move per row.
#define FIXED_ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, R)                             \
    "mov w12, #" #R "\n\t"                                                        \
    OP " {za" #T0 "h.d[w12, 0]}, p1" SUFFIX ", [%[xc]]\n\t"                       \
    "add x13, %[xc], #64\n\t"                                                     \
    OP " {za" #T1 "h.d[w12, 0]}, p2" SUFFIX ", [x13]\n\t"                         \
    "add x13, %[xc], #128\n\t"                                                    \
    OP " {za" #T2 "h.d[w12, 0]}, p3" SUFFIX ", [x13]\n\t"                         \
    "add x13, %[xc], #192\n\t"                                                    \
    OP " {za" #T3 "h.d[w12, 0]}, p4" SUFFIX ", [x13]\n\t"                         \
    "add %[xc], %[xc], #1024\n\t"
#define FIXED_ZA_C_ROWS8(OP, SUFFIX, T0, T1, T2, T3)                              \
    FIXED_ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 0)                                 \
    FIXED_ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 1)                                 \
    FIXED_ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 2)                                 \
    FIXED_ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 3)                                 \
    FIXED_ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 4)                                 \
    FIXED_ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 5)                                 \
    FIXED_ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 6)                                 \
    FIXED_ZA_C_ROW(OP, SUFFIX, T0, T1, T2, T3, 7)
#define FIXED_DEF_ZA_C_TRANSFER(NAME, OP, SUFFIX)                                 \
static inline void NAME(double* c, long cs, long cols)                            \
{                                                                                 \
    char* xc = (char*)c;                                                          \
    asm volatile(                                                                 \
        "ptrue p1.d\n\t" "ptrue p2.d\n\t" "ptrue p3.d\n\t" "ptrue p4.d\n\t"       \
        FIXED_ZA_C_ROWS8(OP, SUFFIX, 0, 1, 2, 3)                                  \
        FIXED_ZA_C_ROWS8(OP, SUFFIX, 4, 5, 6, 7)                                  \
        : [xc] "+&r"(xc) :                                                        \
        : "memory", "cc", "x9", "x12", "x13", VCLOB, PCLOB);                      \
}
FIXED_DEF_ZA_C_TRANSFER(za_load_fixed, "ld1d", "/z")
FIXED_DEF_ZA_C_TRANSFER(za_store_fixed, "st1d", "")

// ---------------------------------------------------------------- 用 FMOPA 转置打包 A（JIT）
// 16 行 × kk 列（kk % 8 == 0，16 行全部有效）→ dst：每 k 占 512 B，前 128 B 为 [行 0-7 | 行 8-15] 的第 k 列。
// 原理：za0 += L 行 r ⊗ e_r（e_r 单位向量）→ za0[k][r] = L[r][k]，tile 行 k 即所需 A 向量；
// str za 按自然顺序落地：tile0 行 k → 向量 8k，tile1 行 k → 向量 8k+1（相邻 128 B）。
static const double g_unit[64] __attribute__((aligned(64))) = {
    1, 0, 0, 0, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0, 0, 0,  0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0, 1,
};
#define PZ_ROWS(xr, z0, z1, z2, z3, z4, z5, z6, z7)                               \
    "ld1d " z0 ".d, p0/z, [%[" xr "]]\n\t"                                        \
    "add x9, %[" xr "], %[lda8]\n\t"                                              \
    "ld1d " z1 ".d, p0/z, [x9]\n\t"                                               \
    "add x9, x9, %[lda8]\n\t"                                                     \
    "ld1d " z2 ".d, p0/z, [x9]\n\t"                                               \
    "add x9, x9, %[lda8]\n\t"                                                     \
    "ld1d " z3 ".d, p0/z, [x9]\n\t"                                               \
    "add x9, x9, %[lda8]\n\t"                                                     \
    "ld1d " z4 ".d, p0/z, [x9]\n\t"                                               \
    "add x9, x9, %[lda8]\n\t"                                                     \
    "ld1d " z5 ".d, p0/z, [x9]\n\t"                                               \
    "add x9, x9, %[lda8]\n\t"                                                     \
    "ld1d " z6 ".d, p0/z, [x9]\n\t"                                               \
    "add x9, x9, %[lda8]\n\t"                                                     \
    "ld1d " z7 ".d, p0/z, [x9]\n\t"
#define PZ_MOPA(T, z0, z1, z2, z3, z4, z5, z6, z7)                                \
    "fmopa " T ".d, p0/m, p0/m, " z0 ".d, z16.d\n\t"                              \
    "fmopa " T ".d, p0/m, p0/m, " z1 ".d, z17.d\n\t"                              \
    "fmopa " T ".d, p0/m, p0/m, " z2 ".d, z18.d\n\t"                              \
    "fmopa " T ".d, p0/m, p0/m, " z3 ".d, z19.d\n\t"                              \
    "fmopa " T ".d, p0/m, p0/m, " z4 ".d, z20.d\n\t"                              \
    "fmopa " T ".d, p0/m, p0/m, " z5 ".d, z21.d\n\t"                              \
    "fmopa " T ".d, p0/m, p0/m, " z6 ".d, z22.d\n\t"                              \
    "fmopa " T ".d, p0/m, p0/m, " z7 ".d, z23.d\n\t"
// 存 tile 对 (T0,T1) 的 8 行：向量 8k+t0, 8k+t1；w12 取 0/16/32/48 各覆盖 2 个 k
#define PZ_STORE4(G, t0, t1)                                                      \
    "mov w12, #" #G "\n\t"                                                        \
    "str za[w12, " #t0 "], [%[xd]]\n\t"                                           \
    "str za[w12, " #t1 "], [%[xd], #" #t1 ", mul vl]\n\t"                         \
    "str za[w12, 8], [%[xd], #8, mul vl]\n\t"                                     \
    "str za[w12, 9], [%[xd], #9, mul vl]\n\t"
static void pack_strip_za(const double* L, long lda, long row0, long k0, long kk, double* dst)
{
    const char* xl0 = (const char*)(L + row0 * lda + k0);
    const char* xl8 = xl0 + 8 * lda * 8;
    char* xd = (char*)dst;
    long n = kk / 8, lda8 = lda * 8;
    asm volatile(
        "ptrue p0.d\n\t"
        "ld1d z16.d, p0/z, [%[u]]\n\t"
        "ld1d z17.d, p0/z, [%[u], #1, mul vl]\n\t"
        "ld1d z18.d, p0/z, [%[u], #2, mul vl]\n\t"
        "ld1d z19.d, p0/z, [%[u], #3, mul vl]\n\t"
        "ld1d z20.d, p0/z, [%[u], #4, mul vl]\n\t"
        "ld1d z21.d, p0/z, [%[u], #5, mul vl]\n\t"
        "ld1d z22.d, p0/z, [%[u], #6, mul vl]\n\t"
        "ld1d z23.d, p0/z, [%[u], #7, mul vl]\n\t"
        "1:\n\t"
        "zero {za0.d, za1.d}\n\t"
        PZ_ROWS("xl0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7")
        PZ_ROWS("xl8", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15")
        PZ_MOPA("za0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7")
        PZ_MOPA("za1", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15")
        // k=0,1 → 向量 0,1,8,9；k=2,3 → 16,17,24,25 …（w12 = 0,16,32,48；xd 每组 +1024）
        "mov w12, #0\n\t"
        "str za[w12, 0], [%[xd]]\n\t"
        "str za[w12, 1], [%[xd], #1, mul vl]\n\t"
        "str za[w12, 8], [%[xd], #8, mul vl]\n\t"
        "str za[w12, 9], [%[xd], #9, mul vl]\n\t"
        "mov w12, #16\n\t"
        "add %[xd], %[xd], #1024\n\t"
        "str za[w12, 0], [%[xd]]\n\t"
        "str za[w12, 1], [%[xd], #1, mul vl]\n\t"
        "str za[w12, 8], [%[xd], #8, mul vl]\n\t"
        "str za[w12, 9], [%[xd], #9, mul vl]\n\t"
        "mov w12, #32\n\t"
        "add %[xd], %[xd], #1024\n\t"
        "str za[w12, 0], [%[xd]]\n\t"
        "str za[w12, 1], [%[xd], #1, mul vl]\n\t"
        "str za[w12, 8], [%[xd], #8, mul vl]\n\t"
        "str za[w12, 9], [%[xd], #9, mul vl]\n\t"
        "mov w12, #48\n\t"
        "add %[xd], %[xd], #1024\n\t"
        "str za[w12, 0], [%[xd]]\n\t"
        "str za[w12, 1], [%[xd], #1, mul vl]\n\t"
        "str za[w12, 8], [%[xd], #8, mul vl]\n\t"
        "str za[w12, 9], [%[xd], #9, mul vl]\n\t"
        "add %[xd], %[xd], #1024\n\t"
        "add %[xl0], %[xl0], #64\n\t"
        "add %[xl8], %[xl8], #64\n\t"
        "subs %[n], %[n], #1\n\t"
        "b.ne 1b\n\t"
        : [xl0] "+&r"(xl0), [xl8] "+&r"(xl8), [xd] "+&r"(xd), [n] "+&r"(n)
        : [lda8] "r"(lda8), [u] "r"(g_unit)
        : "memory", "cc", "x9", "x12", VCLOB, PCLOB);
}

// 变体 A：ld4d 解交织 + 8 tile → 稠密 128 B/k 布局。32 个 k 一组：行 r 的 ld4d 得 z_q[j] = L[r][k0+4j+q]，
// fmopa za(2q + [r≥8]) += z_q ⊗ e_{r%8} → tile 行 j = k0+4j+q；str za 自然布局向量 n = 8j + 2q (+1) → 地址 128·(k-k0) (+64)。
// 要求 kk % 32 == 0，16 行全部有效。
#define P4_R(zA, zB, zC, zD, T0, T1, T2, T3, E)                                   \
    "ld4d {" zA ".d, " zB ".d, " zC ".d, " zD ".d}, p0/z, [x9]\n\t"               \
    "add x9, x9, %[lda8]\n\t"                                                     \
    "fmopa " T0 ".d, p0/m, p0/m, " zA ".d, " E ".d\n\t"                           \
    "fmopa " T1 ".d, p0/m, p0/m, " zB ".d, " E ".d\n\t"                           \
    "fmopa " T2 ".d, p0/m, p0/m, " zC ".d, " E ".d\n\t"                           \
    "fmopa " T3 ".d, p0/m, p0/m, " zD ".d, " E ".d\n\t"
#define P4_EVEN(E) P4_R("z0", "z1", "z2", "z3", "za0", "za2", "za4", "za6", E)
#define P4_ODD(E)  P4_R("z4", "z5", "z6", "z7", "za0", "za2", "za4", "za6", E)
#define P4_EVEN_H(E) P4_R("z0", "z1", "z2", "z3", "za1", "za3", "za5", "za7", E)
#define P4_ODD_H(E)  P4_R("z4", "z5", "z6", "z7", "za1", "za3", "za5", "za7", E)
static void pack_strip_ld4(const double* L, long lda, long row0, long k0, long kk, double* dst)
{
    const char* xl = (const char*)(L + row0 * lda + k0);
    char* xd = (char*)dst;
    long n = kk / 32, lda8 = lda * 8;
    asm volatile(
        "ptrue p0.d\n\t"
        "ld1d z16.d, p0/z, [%[u]]\n\t"
        "ld1d z17.d, p0/z, [%[u], #1, mul vl]\n\t"
        "ld1d z18.d, p0/z, [%[u], #2, mul vl]\n\t"
        "ld1d z19.d, p0/z, [%[u], #3, mul vl]\n\t"
        "ld1d z20.d, p0/z, [%[u], #4, mul vl]\n\t"
        "ld1d z21.d, p0/z, [%[u], #5, mul vl]\n\t"
        "ld1d z22.d, p0/z, [%[u], #6, mul vl]\n\t"
        "ld1d z23.d, p0/z, [%[u], #7, mul vl]\n\t"
        "1:\n\t"
        "zero {za}\n\t"
        "mov x9, %[xl]\n\t"
        P4_EVEN("z16") P4_ODD("z17") P4_EVEN("z18") P4_ODD("z19")
        P4_EVEN("z20") P4_ODD("z21") P4_EVEN("z22") P4_ODD("z23")
        P4_EVEN_H("z16") P4_ODD_H("z17") P4_EVEN_H("z18") P4_ODD_H("z19")
        P4_EVEN_H("z20") P4_ODD_H("z21") P4_EVEN_H("z22") P4_ODD_H("z23")
        ZA_DUMP_GROUP(0) ZA_DUMP_GROUP(16) ZA_DUMP_GROUP(32) ZA_DUMP_GROUP(48)
        "add %[xl], %[xl], #256\n\t"
        "subs %[n], %[n], #1\n\t"
        "b.ne 1b\n\t"
        : [xl] "+&r"(xl), [b] "+&r"(xd), [n] "+&r"(n)
        : [lda8] "r"(lda8), [u] "r"(g_unit)
        : "memory", "cc", "x9", "x12", VCLOB, PCLOB);
}

// 变体 B：ld2d + 4 tile，两组交替（za0-3 / za4-7），使 str za 与下一组的 fmopa 有机会重叠。
// 16 个 k 一组：z_q[j] = L[r][k0+2j+q]，tile (2q + [r≥8]) (+4 奇数组)。布局：k 偶 → 256(k-k0)，k 奇 → 256(k-k0)-128，
// 即 K 循环步长交替 128 / 384。要求 kk % 32 == 0。
#define P2_R(zA, zB, T0, T1, E)                                                   \
    "ld2d {" zA ".d, " zB ".d}, p0/z, [x9]\n\t"                                   \
    "add x9, x9, %[lda8]\n\t"                                                     \
    "fmopa " T0 ".d, p0/m, p0/m, " zA ".d, " E ".d\n\t"                           \
    "fmopa " T1 ".d, p0/m, p0/m, " zB ".d, " E ".d\n\t"
#define P2_GROUP(T0, T1, T2, T3)                                                  \
    "mov x9, %[xl]\n\t"                                                           \
    P2_R("z0", "z1", T0, T2, "z16") P2_R("z2", "z3", T0, T2, "z17")               \
    P2_R("z4", "z5", T0, T2, "z18") P2_R("z6", "z7", T0, T2, "z19")               \
    P2_R("z0", "z1", T0, T2, "z20") P2_R("z2", "z3", T0, T2, "z21")               \
    P2_R("z4", "z5", T0, T2, "z22") P2_R("z6", "z7", T0, T2, "z23")               \
    P2_R("z0", "z1", T1, T3, "z16") P2_R("z2", "z3", T1, T3, "z17")               \
    P2_R("z4", "z5", T1, T3, "z18") P2_R("z6", "z7", T1, T3, "z19")               \
    P2_R("z0", "z1", T1, T3, "z20") P2_R("z2", "z3", T1, T3, "z21")               \
    P2_R("z4", "z5", T1, T3, "z22") P2_R("z6", "z7", T1, T3, "z23")               \
    "add %[xl], %[xl], #128\n\t"
// 存一组 4 个 tile 的两行（向量 G+t 与 G+8+t，t 为该组的 4 个 tile 号），w12 = 0/16/32/48，每组后基址 +1024
#define P2_ST_A(G)                                                                \
    "mov w12, #" #G "\n\t"                                                        \
    "str za[w12, 0], [%[xa]]\n\t"                                                 \
    "str za[w12, 1], [%[xa], #1, mul vl]\n\t"                                     \
    "str za[w12, 2], [%[xa], #2, mul vl]\n\t"                                     \
    "str za[w12, 3], [%[xa], #3, mul vl]\n\t"                                     \
    "str za[w12, 8], [%[xa], #8, mul vl]\n\t"                                     \
    "str za[w12, 9], [%[xa], #9, mul vl]\n\t"                                     \
    "str za[w12, 10], [%[xa], #10, mul vl]\n\t"                                   \
    "str za[w12, 11], [%[xa], #11, mul vl]\n\t"                                   \
    "add %[xa], %[xa], #1024\n\t"
#define P2_ST_B(G)                                                                \
    "mov w12, #" #G "\n\t"                                                        \
    "str za[w12, 4], [%[xb], #4, mul vl]\n\t"                                     \
    "str za[w12, 5], [%[xb], #5, mul vl]\n\t"                                     \
    "str za[w12, 6], [%[xb], #6, mul vl]\n\t"                                     \
    "str za[w12, 7], [%[xb], #7, mul vl]\n\t"                                     \
    "str za[w12, 12], [%[xb], #12, mul vl]\n\t"                                   \
    "str za[w12, 13], [%[xb], #13, mul vl]\n\t"                                   \
    "str za[w12, 14], [%[xb], #14, mul vl]\n\t"                                   \
    "str za[w12, 15], [%[xb], #15, mul vl]\n\t"                                   \
    "add %[xb], %[xb], #1024\n\t"
#define P2_STORE_A P2_ST_A(0) P2_ST_A(16) P2_ST_A(32) P2_ST_A(48)
#define P2_STORE_B P2_ST_B(0) P2_ST_B(16) P2_ST_B(32) P2_ST_B(48)
static void pack_strip_ld2(const double* L, long lda, long row0, long k0, long kk, double* dst)
{
    const char* xl = (const char*)(L + row0 * lda + k0);
    char* xa = (char*)dst;             // 偶数组基址：向量 8j+t (t=0..3) → dst + 512j + 64t
    char* xb = (char*)dst + 4096 - 256; // 奇数组基址：向量 8j+4+t → dst + 4096 + 512j + 64t
    long n = kk / 32, lda8 = lda * 8;
    asm volatile(
        "ptrue p0.d\n\t"
        "ld1d z16.d, p0/z, [%[u]]\n\t"
        "ld1d z17.d, p0/z, [%[u], #1, mul vl]\n\t"
        "ld1d z18.d, p0/z, [%[u], #2, mul vl]\n\t"
        "ld1d z19.d, p0/z, [%[u], #3, mul vl]\n\t"
        "ld1d z20.d, p0/z, [%[u], #4, mul vl]\n\t"
        "ld1d z21.d, p0/z, [%[u], #5, mul vl]\n\t"
        "ld1d z22.d, p0/z, [%[u], #6, mul vl]\n\t"
        "ld1d z23.d, p0/z, [%[u], #7, mul vl]\n\t"
        "zero {za}\n\t"
        "1:\n\t"
        P2_GROUP("za0", "za1", "za2", "za3")
        P2_STORE_A
        "zero {za0.d, za1.d, za2.d, za3.d}\n\t"
        P2_GROUP("za4", "za5", "za6", "za7")
        P2_STORE_B
        "zero {za4.d, za5.d, za6.d, za7.d}\n\t"
        "add %[xa], %[xa], #4096\n\t"   // 每轮 32 k = 8192 B，两组各占 4096，store 已各推进 4096
        "add %[xb], %[xb], #4096\n\t"
        "subs %[n], %[n], #1\n\t"
        "b.ne 1b\n\t"
        : [xl] "+&r"(xl), [xa] "+&r"(xa), [xb] "+&r"(xb), [n] "+&r"(n)
        : [lda8] "r"(lda8), [u] "r"(g_unit)
        : "memory", "cc", "x9", "x12", VCLOB, PCLOB);
}

// 把 X 面板（kk 行 × nci 列，行距 ldb）拷到连续缓冲 dst（行距 nc，越界列补零）
static __attribute__((unused)) void copy_panel(const double* X, long ldb, long kk, long nci, long nc, double* dst)
{
    const svbool_t pg = svptrue_b64();
    for (long k = 0; k < kk; k++) {
        const double* s = X + k * ldb;
        double* d = dst + k * nc;
        long j = 0;
        for (; j + 8 <= nci; j += 8) svst1_f64(pg, d + j, svld1_f64(pg, s + j));
        if (j < nci) {
            svbool_t pj = svwhilelt_b64((long)j, (long)nci);
            svst1_f64(pj, d + j, svld1_f64(pj, s + j));
            j += 8;
        }
        for (; j < nc; j += 8) svst1_f64(pg, d + j, svdup_f64(0.0));
    }
}

// ---------------------------------------------------------------- 打包 A（L 的 16 行 × kk 列 → k 主序）
// dst[k*16 + r] = L[(row0+r)*lda + k0 + k]，row0+r ≥ m 的行补零。
static const uint64_t g_tbl_idx[4][8] = {
    {0, 1, 8, 9, 2, 3, 10, 11},     // 128 位交织低半
    {4, 5, 12, 13, 6, 7, 14, 15},   // 128 位交织高半
    {0, 1, 2, 3, 8, 9, 10, 11},     // 256 位交织低半
    {4, 5, 6, 7, 12, 13, 14, 15},   // 256 位交织高半
};

// 8×8 转置：r0..r7 为 8 行（各 8 个 k），输出 8 列写到 dst + k*16（每列 8 个 double）
#define TR8X8_STORE(R0, R1, R2, R3, R4, R5, R6, R7, DST)                                          \
    do {                                                                                          \
        svfloat64_t t0 = svzip1_f64(R0, R1), t1 = svzip2_f64(R0, R1);                             \
        svfloat64_t t2 = svzip1_f64(R2, R3), t3 = svzip2_f64(R2, R3);                             \
        svfloat64_t t4 = svzip1_f64(R4, R5), t5 = svzip2_f64(R4, R5);                             \
        svfloat64_t t6 = svzip1_f64(R6, R7), t7 = svzip2_f64(R6, R7);                             \
        svfloat64_t u0 = svtbl2_f64(svcreate2_f64(t0, t2), iq1), u1 = svtbl2_f64(svcreate2_f64(t0, t2), iq2); \
        svfloat64_t u2 = svtbl2_f64(svcreate2_f64(t1, t3), iq1), u3 = svtbl2_f64(svcreate2_f64(t1, t3), iq2); \
        svfloat64_t u4 = svtbl2_f64(svcreate2_f64(t4, t6), iq1), u5 = svtbl2_f64(svcreate2_f64(t4, t6), iq2); \
        svfloat64_t u6 = svtbl2_f64(svcreate2_f64(t5, t7), iq1), u7 = svtbl2_f64(svcreate2_f64(t5, t7), iq2); \
        svst1_f64(pg, (DST) + 0 * 16, svtbl2_f64(svcreate2_f64(u0, u4), ih1));                    \
        svst1_f64(pg, (DST) + 1 * 16, svtbl2_f64(svcreate2_f64(u0, u4), ih2));                    \
        svst1_f64(pg, (DST) + 2 * 16, svtbl2_f64(svcreate2_f64(u1, u5), ih1));                    \
        svst1_f64(pg, (DST) + 3 * 16, svtbl2_f64(svcreate2_f64(u1, u5), ih2));                    \
        svst1_f64(pg, (DST) + 4 * 16, svtbl2_f64(svcreate2_f64(u2, u6), ih1));                    \
        svst1_f64(pg, (DST) + 5 * 16, svtbl2_f64(svcreate2_f64(u2, u6), ih2));                    \
        svst1_f64(pg, (DST) + 6 * 16, svtbl2_f64(svcreate2_f64(u3, u7), ih1));                    \
        svst1_f64(pg, (DST) + 7 * 16, svtbl2_f64(svcreate2_f64(u3, u7), ih2));                    \
    } while (0)

static void pack_strip(const double* L, long lda, long m, long row0, long k0, long kk, double* dst)
{
    long rows = m - row0;
    if (rows > 16) rows = 16;
    if (rows == 16 && (kk & 7) == 0) {
        const svbool_t pg = svptrue_b64();
        const svuint64_t iq1 = svld1_u64(pg, g_tbl_idx[0]), iq2 = svld1_u64(pg, g_tbl_idx[1]);
        const svuint64_t ih1 = svld1_u64(pg, g_tbl_idx[2]), ih2 = svld1_u64(pg, g_tbl_idx[3]);
        const double* src = L + row0 * lda + k0;
        for (long kb = 0; kb < kk; kb += 8) {
            const double* s = src + kb;
            svfloat64_t r0 = svld1_f64(pg, s + 0 * lda), r1 = svld1_f64(pg, s + 1 * lda);
            svfloat64_t r2 = svld1_f64(pg, s + 2 * lda), r3 = svld1_f64(pg, s + 3 * lda);
            svfloat64_t r4 = svld1_f64(pg, s + 4 * lda), r5 = svld1_f64(pg, s + 5 * lda);
            svfloat64_t r6 = svld1_f64(pg, s + 6 * lda), r7 = svld1_f64(pg, s + 7 * lda);
            TR8X8_STORE(r0, r1, r2, r3, r4, r5, r6, r7, dst + kb * 16);
            r0 = svld1_f64(pg, s + 8 * lda); r1 = svld1_f64(pg, s + 9 * lda);
            r2 = svld1_f64(pg, s + 10 * lda); r3 = svld1_f64(pg, s + 11 * lda);
            r4 = svld1_f64(pg, s + 12 * lda); r5 = svld1_f64(pg, s + 13 * lda);
            r6 = svld1_f64(pg, s + 14 * lda); r7 = svld1_f64(pg, s + 15 * lda);
            TR8X8_STORE(r0, r1, r2, r3, r4, r5, r6, r7, dst + kb * 16 + 8);
        }
        return;
    }
    for (long k = 0; k < kk; k++) {
        double* d = dst + k * 16;
        for (long r = 0; r < 16; r++) d[r] = (r < rows) ? L[(row0 + r) * lda + k0 + k] : 0.0;
    }
}

// ---------------------------------------------------------------- 对角块预处理
// 行块 i（行 [i0, i0+mbi)，S 个 16 行子块）→ dst：子块 s 的打包条带（16 × 16(s+1)），
//   k <  16s：dst[k*16 + r] = -(Linv_ss · L_{s,k/16})[r][k%16]
//   k ≥ 16s：dst[k*16 + r] = Linv_ss[r][k-16s]
// 子块 s 的条带起点 = dst + 256*s*(s+1)/2。越界（≥ mbi）的行/列按单位阵处理。
static inline size_t diag_strip_off(int s) { return (size_t)256 * s * (s + 1) / 2; }
static inline size_t diag_block_doubles(int S) { return (size_t)256 * S * (S + 1) / 2; }

static void diag_prep_sub(const double* L, long lda, long i0, int mbi, int s, double* pk)
{
    double Lss[16][16], Linv[16][16];
    long r0 = i0 + 16 * s;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int in = (16 * s + r < mbi) && (16 * s + c < mbi) && (c <= r);
            Lss[r][c] = in ? L[(r0 + r) * lda + r0 + c] : (r == c ? 1.0 : 0.0);
            Linv[r][c] = 0.0;
        }
    for (int c = 0; c < 16; c++) {
        Linv[c][c] = 1.0 / Lss[c][c];
        for (int r = c + 1; r < 16; r++) {
            double acc = 0.0;
            for (int q = c; q < r; q++) acc += Lss[r][q] * Linv[q][c];
            Linv[r][c] = -acc / Lss[r][r];
        }
    }
    // 越界行 t（≥ mbi）在 L_{s,s'} 中视为 0 行
    int vr = mbi - 16 * s;
    if (vr > 16) vr = 16;
    for (int sp = 0; sp < s; sp++) {
        const double* Lb = L + r0 * lda + i0 + 16 * sp;   // L_{s,sp}[t][q] = Lb[t*lda + q]
        double* d = pk + (size_t)16 * sp * 16;
        for (int q = 0; q < 16; q++) {
            for (int r = 0; r < 16; r++) {
                double v = 0.0;
                int tmax = r < vr ? r : vr - 1;
                for (int t = 0; t <= tmax; t++) v += Linv[r][t] * Lb[t * lda + q];
                d[q * 16 + r] = -v;
            }
        }
    }
    double* d = pk + (size_t)16 * s * 16;
    for (int q = 0; q < 16; q++)
        for (int r = 0; r < 16; r++) d[q * 16 + r] = Linv[r][q];
}

// ---------------------------------------------------------------- 上下文与缓冲
typedef struct {
    long m, n, lda, ldb;
    const double* L;
    double* B;
    int mb, nc, kc;
    int lchunk, direct, special;
    int owner;              // 面板驻留模式（小 m、宽 X）：线程独占整列面板并按行块顺序做完，B 操作数来自本线程 L2 中的面板缓冲，不写 Xp、无跨线程依赖
    long* owner_bound;      // 面板驻留模式：线程 t 的连续任务区间 [owner_bound[t], owner_bound[t+1])（面板主序，按代价均衡）
    int dnt;                // 对角阶段 A 条用 ldnt1d（面板驻留模式默认开；只对能留在 L1 的短条带生效）
    int eflag;              // 专用路径：Xp 写完即发布 flag，X 回写放在 flag 之后（TRSM_EFLAG）
    int nexta;              // fast_owner：回写循环里预取下一任务首段 A（TRSM_NEXTA）
    int pf2;                // 面板驻留模式：游标 2 也预取 A（更深的 A 流预取）
    long loadpf;            // 装载阶段（B 子块 → C 块）软件预取提前行数（TRSM_LOADPF，0 关）
    int xptm;               // 专用路径 Xp（K×16 微面板）按列块主序写（TRSM_XPTM）
    long nrb, npan, ntask;
    long cshift;     // 面板边界平移（double 数，0..7）：X 基址非 64 B 对齐且行距为 64 B 整数倍时，内部面板起点按 X 的 cache 行对齐
                     // （首面板 nc−cshift 列、末面板 cshift 列走通用路径），内部面板的 X 装载/回写全对齐（TRSM_CSHIFT）
    long ncp;        // cbuf / Xp 面板 / 部分和缓冲的行距（double）：nc + 8，避免 1 KB 行距的 L1 组冲突
    int* S_of;              // 每行块的子块数
    double* diag;           // 全部行块的对角打包
    size_t* diag_off;       // 每行块起点（double 下标）
    int jit;                // 1：GEMM 部分就地转置；0：用预打包
    int ppmode;             // 0：预打包在并行区开头一次完成；1：打包作为任务流水式进行（大 L），消费者等 pflags
    volatile int* pflags;   // 每条的打包完成标志（ppmode=1；nobar 时 ppmode=0 也用）
    volatile int* dcnt;     // 每个行块已完成的对角条数（nobar：任务对角阶段前等它等于 S_of[i]）
    int nobar;              // 预处理阶段不设全局 barrier：预处理项按行块顺序动态领取，消费者用 pflags/dcnt 等（TRSM_NOBAR）
    double* pp[2];          // 预打包（0 主副本，1 副副本，可为 NULL）
    size_t* pp_off;         // 每 16 行条起点（double 下标）
    long nstrip;
    volatile int* flags;    // nrb*npan
    // Dispatch writes must not invalidate the read-mostly kernel settings.
    volatile long next __attribute__((aligned(128)));
    char next_padding[128 - sizeof(long)];
    int prof;
    int swhost;      // X 回写托管进后续 K 循环
    int ant;         // GEMM 阶段 A 条用 ldnt1d
    int reserve;     // 对角阶段开始时预占下一任务（精确预取其 C 块）
    int xpnt;        // Xp 拷贝用 stnt1d
    int pf;                 // 软件预取开关
    int ntstore;            // 结果回写用非临时存储
    double* xp;             // X 的面板主序副本：面板 (chunk c, panel p) 在 xp + (c*npan + p)*kc*nc，行距 nc
    long xp_chunks;
    int packmode;           // JIT 打包方式：0 SVE zip/tbl2，1 ld4d+8tile（稠密），2 ld2d+4tile 交替，3 单 tile 对 512B/k
    // 任务表（含 K 分裂段）
    const struct task_s* tasks;
    long ntask_items;
    double* part;           // 分裂任务的部分和缓冲池
    volatile int* seg_cnt;  // 每个 (i,p) 的已完成段数
} ctx_t;

typedef struct task_s {
    int kind;           // 0 计算任务；1 打包条 g（流水式预打包）
    long g;             // kind=1：16 行条编号
    int i, p;
    int seg, nseg;      // 段号 / 段数（nseg=1 为普通任务）
    long ch0, ch1;      // 本段负责的 K 段区间 [ch0, ch1)（以 kc 为单位）
    size_t part_off;    // 本 (i,p) 的部分和缓冲起点（double 下标），段 s 在 part_off + s*mb*nc
} task_t;

typedef struct {
    double* cbuf;    // mb*nc
    double* astrip;  // JIT 打包 A：kc*64（每 k 512 B 布局）
    double* xpan;    // X 面板拷贝：kc*nc
    double* scratch; // 4 KB ZA 转存
    double* scratch2;// 预取哑地址
    size_t cbuf_cap, astrip_cap, xpan_cap;
    // 托管回写：当前区（K 循环里推进）+ 待处理区队列（上一任务的 X 块按 kc 行分区）
    sw_t sw;
    sw_t swq[16];
    int swn, swi;
    long n_swflush, n_swq;   // 同步收尾拷的 cache 行数 / 排队的总行数（诊断托管走线器是否生效）
    // 分相计时（cntvct 周期），TRSM_PROFILE=2 时累计
    uint64_t t_load, t_pack, t_pf, t_wait, t_kloop, t_epi, t_dkloop, t_depi, t_store, t_task;
    long n_task, n_kstep;
    uint64_t t_dump, t_ddump;
    long n_dkstep;
    // Keep counters from sharing a line with the next worker's pointers.
} __attribute__((aligned(128))) tbuf_t;

// 同步完成一个回写区（可能已被 K 循环推进到行中）
static void sw_run_region(sw_t* w, tbuf_t* tb)
{
    const svbool_t pg = svptrue_b64();
    while (w->rows > 0) {
        for (long l = 0; l < w->cnt; l++)
            svstnt1_f64(pg, (double*)(w->dst + 64 * l), svld1_f64(pg, (const double*)(w->src + 64 * l)));
        tb->n_swflush += w->cnt;
        w->src += 64 * w->cnt + TRSM_NCPAD_BYTES;
        w->dst += 64 * w->cnt + w->dstj;
        w->cnt = w->lpr;
        w->rows--;
    }
}
static void sw_flush(tbuf_t* tb)
{
    sw_run_region(&tb->sw, tb);
    while (tb->swi < tb->swn) { tb->sw = tb->swq[tb->swi++]; sw_run_region(&tb->sw, tb); }
    tb->swi = tb->swn = 0;
}
static inline void sw_advance(tbuf_t* tb)
{
    if (tb->sw.rows == 0 && tb->swi < tb->swn) tb->sw = tb->swq[tb->swi++];
}

#define PROF_T(c, expr, field)                                           \
    do {                                                                 \
        if ((c)->prof >= 2) { uint64_t _t0 = cyc(); expr; tb->field += cyc() - _t0; } \
        else { expr; }                                                   \
    } while (0)

static tbuf_t g_tb[1024];
static uint64_t g_prep_cyc[1024], g_tail_cyc[1024], g_wake_cyc[1024];
static double* g_diag = NULL;
static size_t g_diag_cap = 0;
static double* g_pp[2] = {NULL, NULL};
static size_t g_pp_cap[2] = {0, 0};
static int g_pp_node[2] = {-3, -3};
static int* g_flags = NULL;
static size_t g_flags_cap = 0;
static int* g_pflags = NULL;
static size_t g_pflags_cap = 0;
static int* g_dcnt = NULL; static size_t g_dcnt_cap = 0;
static int* g_seg_cnt = NULL;
static task_t* g_tasks = NULL;
static size_t g_tasks_cap = 0;
static double* g_part = NULL;
static size_t g_part_cap = 0;
static double* g_xp = NULL;
static size_t g_xp_cap = 0;
static int g_xp_node = -3;

static void ensure_tbuf(int tid, size_t cbuf_d, size_t astrip_d, size_t xpan_d)
{
    tbuf_t* t = &g_tb[tid];
    if (t->cbuf_cap < cbuf_d) {
        trsm_free_buffer(t->cbuf, t->cbuf_cap * 8);
        t->cbuf = alloc_on(cbuf_d * 8, -1);
        t->cbuf_cap = cbuf_d;
    }
    if (t->astrip_cap < astrip_d) {
        trsm_free_buffer(t->astrip, t->astrip_cap * 8);
        t->astrip = alloc_on(astrip_d * 8, -1);
        t->astrip_cap = astrip_d;
    }
    if (t->xpan_cap < xpan_d) {
        trsm_free_buffer(t->xpan, t->xpan_cap * 8);
        t->xpan = alloc_on(xpan_d * 8, -1);
        t->xpan_cap = xpan_d;
    }
    if (!t->scratch) { t->scratch = alloc_on(8192, -1); memset(t->scratch, 0, 8192); }
    if (!t->scratch2) { t->scratch2 = alloc_on(65536, -1); memset(t->scratch2, 0, 65536); }
}

// ---------------------------------------------------------------- 任务执行
static inline void wait_flag(const ctx_t* c, long idx)
{
    while (__atomic_load_n(&c->flags[idx], __ATOMIC_ACQUIRE) == 0) asm volatile("yield");
}

// Pack consecutive K chunks in consumption order to reduce L2 set conflicts.
// pp_off at the first strip of a row block remains its allocation base.
static inline size_t packed_l_offset(const ctx_t* c, long g, long k0) {
    if (!c->lchunk) return c->pp_off[g] + (size_t)k0 * 16;
    long i=(16*g)/c->mb, g0=i*c->mb/16;
    return c->pp_off[g0]+(size_t)(k0/c->kc)*c->S_of[i]*16*c->kc+(g-g0)*16*c->kc+(k0%c->kc)*16;
}
static void pack_l_chunked(const ctx_t* c, long g) {
    long row0=16*g, i=row0/c->mb, kk=i*c->mb;
    long rows=c->m-row0; if(rows>16)rows=16;
    for(long k=0;k<kk;k+=c->kc) {
        double* dst=c->pp[0]+packed_l_offset(c,g,k);
        // The initial OpenMP packing stage has not enabled ZA; only task packing may use it.
        if(c->ppmode==1 && rows==16 && (c->kc&31)==0)pack_strip_ld4(c->L,c->lda,row0,k,c->kc,dst);
        else pack_strip(c->L,c->lda,c->m,row0,k,c->kc,dst);
        if(c->pp[1])memcpy(c->pp[1]+packed_l_offset(c,g,k),dst,(size_t)c->kc*16*8);
    }
}

// 流水式打包任务：把 16 行条 g 的 L 子块（列 [0, i0)）转置成稠密 k 主序写到预打包缓冲
static void run_pack_task(const ctx_t* c, const task_t* tk, tbuf_t* tb)
{
    uint64_t t0 = c->prof >= 2 ? cyc() : 0;
    long g = tk->g;
    long i = (16 * g) / c->mb;
    long i0 = i * c->mb;
    long row0 = 16 * g;
    long rows = c->m - row0; if (rows > 16) rows = 16;
    double* dst = c->pp[0] + c->pp_off[g];
    if (i0 > 0 && c->lchunk) { pack_l_chunked(c, g); }
    else if (i0 > 0) {
        if (rows == 16 && (i0 & 31) == 0) pack_strip_ld4(c->L, c->lda, row0, 0, i0, dst);
        else pack_strip(c->L, c->lda, c->m, row0, 0, i0, dst);
    }
    __atomic_store_n(&c->pflags[g], 1, __ATOMIC_RELEASE);
    if (c->prof >= 2) tb->t_pack += cyc() - t0;
}

// 返回：对角阶段预占到的下一任务号（未预占为 -1）
// 驻留面板专用路径：相邻两个 16 行条带复用同一个 16 列 X 微块。
static inline void za_flush_duplex_bank4(double*, long);
static inline __attribute__((always_inline)) void kloop_owner_bank0(const double*, const double*, double*, const char*, const char*, const char*);
static inline __attribute__((always_inline)) void kloop_owner_bank4(const double*, const double*, double*, const char*, const char*, const char*);

static inline __attribute__((always_inline)) void kloop_owner_cursor0(const double*,const double*,double*,const char*,const char*,const char*);
static inline __attribute__((always_inline)) void kloop_owner_cursor4(const double*,const double*,double*,const char*,const char*,const char*);
// Select once per owner task; keep the original kernel for same-binary comparisons.
static int g_owner_cursor;

// Full 16x32 diagonal results can go straight from ZA to the resident panel.
// All columns are independent; storing this tile before the next column tile
// also removes the temporary ZA dump and its subsequent SVE reload.
#define OWNER_DSTORE_ROW(T0, T1, T2, T3, R) \
    "mov w12, #" #R "\n\t" \
    "mova z16.d, p0/m, za" #T0 "h.d[w12, 0]\n\t" \
    "mova z17.d, p0/m, za" #T1 "h.d[w12, 0]\n\t" \
    "mova z18.d, p0/m, za" #T2 "h.d[w12, 0]\n\t" \
    "mova z19.d, p0/m, za" #T3 "h.d[w12, 0]\n\t" \
    "st1d z16.d, p0, [%[c], #0, mul vl]\n\t" \
    "st1d z17.d, p0, [%[c], #1, mul vl]\n\t" \
    "st1d z18.d, p0, [%[c], #2, mul vl]\n\t" \
    "st1d z19.d, p0, [%[c], #3, mul vl]\n\t" \
    "add %[c], %[c], #768\n\t"
#define OWNER_DSTORE_ROWS(T0, T1, T2, T3) \
    OWNER_DSTORE_ROW(T0, T1, T2, T3, 0) \
    OWNER_DSTORE_ROW(T0, T1, T2, T3, 1) \
    OWNER_DSTORE_ROW(T0, T1, T2, T3, 2) \
    OWNER_DSTORE_ROW(T0, T1, T2, T3, 3) \
    OWNER_DSTORE_ROW(T0, T1, T2, T3, 4) \
    OWNER_DSTORE_ROW(T0, T1, T2, T3, 5) \
    OWNER_DSTORE_ROW(T0, T1, T2, T3, 6) \
    OWNER_DSTORE_ROW(T0, T1, T2, T3, 7)
static inline void za_owner_dstore(double* c)
{
    asm volatile(
        "ptrue p0.d\n\t"
        OWNER_DSTORE_ROWS(0, 1, 2, 3)
        OWNER_DSTORE_ROWS(4, 5, 6, 7)
        : [c] "+&r"(c) :: "memory", "cc", "x12", VCLOB, PCLOB);
}
#undef OWNER_DSTORE_ROW
#undef OWNER_DSTORE_ROWS

// 面板列起点/宽度：cshift>0 时内部面板边界按 X 的 64 B cache 行对齐（面板 0 缩为 nc−cshift 列、末面板 cshift 列），
// X 行段的装载与非临时回写全对齐（无跨行向量、无 partial-line RMW）。cshift=0 时退化为原始等宽划分。
static inline long pan_c0(const ctx_t* c, long p) { return p ? p * c->nc - c->cshift : 0; }
static inline long pan_nci(const ctx_t* c, long p)
{
    long e = (p + 1) * c->nc - c->cshift;
    if (e > c->n) e = c->n;
    return e - pan_c0(c, p);
}

static inline __attribute__((always_inline)) long run_task_impl(const ctx_t* c, tbuf_t* tb, const task_t* tk, const task_t* tkn_guess, int tid, int fast_owner)
{
    if (!fast_owner && tk->kind == 1) { run_pack_task(c, tk, tb); return -1; }
    long tnext = -1;
    const task_t* tkn = tkn_guess;
    const long mb = (fast_owner ? 64 : c->mb), nc = (fast_owner ? 96 : c->nc), kc = (fast_owner ? 64 : c->kc), ncp = (fast_owner ? 96 : c->ncp);
    const long i = tk->i, p = tk->p;
    const long i0 = i * mb;
    long mbi = fast_owner ? 64 : c->m - i0; if (mbi > mb) mbi = mb;
    const long c0 = pan_c0(c, p);   // fast_owner 只接整面板（奇面板由调度路由到通用路径），公式对两者一致
    long nci = fast_owner ? 96 : pan_nci(c, p); if (nci > nc) nci = nc;
    const int S = fast_owner ? 4 : c->S_of[i];
    const long ntile = (nci + 31) / 32;
    double* cb = tb->cbuf;
    double* scratch = tb->scratch;
    const long ldb = c->ldb, lda = c->lda;
    const long cs = ncp * 8;      // C / X 面板缓冲行距（字节）
    if ((fast_owner ? 1 : c->owner)) cb = tb->cbuf + (size_t)i0 * ncp;   // 面板缓冲：全局行 gr 在 cbuf + gr*ncp，本行块的 C 块就是其中 [i0, i0+mb) 行
    double* Bblk = c->B + i0 * ldb + c0;
    const pf_t pf_dummy = { (const char*)tb->scratch2, 1, -64, (const char*)tb->scratch2, (const char*)tb->scratch2, 0, &tb->sw };
    const int pf_on = (fast_owner ? 1 : c->pf);
    uint64_t tt0 = c->prof >= 2 ? cyc() : 0;

    // 分裂段：部分和写到专用缓冲（从 0 开始累加），普通任务直接用 C 块
    const int split = !fast_owner && tk->nseg > 1;
    double* acc = split ? c->part + tk->part_off + (size_t)tk->seg * mb * ncp : cb;

    // 1. C 块 ← B 子块（越界列/行补零）；分裂段只清零部分和缓冲
    if (split) {
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        const svbool_t pg = svptrue_b64();
        for (long q = 0; q < 16L * S * ncp; q += 8) svst1_f64(pg, acc + q, svdup_f64(0.0));
        if (c->prof >= 2) tb->t_load += cyc() - t0;
    } else {
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        const svbool_t pg = svptrue_b64();
        const long lpf = c->loadpf;   // 装载阶段软件预取：提前 lpf 行把 B 子块行拉进 L1（每行 nci*8/64 个 cache 行）
        for (long r = 0; r < 16L * S; r++) {
            double* d = cb + r * ncp;
            if (lpf > 0 && r + lpf < mbi) {
                const char* pr = (const char*)(Bblk + (r + lpf) * ldb);
                for (long b = 0; b < nci * 8; b += 64) __builtin_prefetch(pr + b, 0, 3);
            }
            if (r < mbi) {
                const double* s = Bblk + r * ldb;
                long j = 0;
                for (; j + 8 <= nci; j += 8) svst1_f64(pg, d + j, svld1_f64(pg, s + j));
                if (j < nci) {
                    svbool_t pj = svwhilelt_b64((long)j, (long)nci);
                    svst1_f64(pj, d + j, svld1_f64(pj, s + j));
                    j += 8;
                }
                for (; j < nc; j += 8) svst1_f64(pg, d + j, svdup_f64(0.0));
            } else {
                for (long j = 0; j < nc; j += 8) svst1_f64(pg, d + j, svdup_f64(0.0));
            }
        }
        if (c->prof >= 2) tb->t_load += cyc() - t0;
    }

    // 2. GEMM 段：C 块 -= L[i0.., k0..k0+kk) · X[k0..k0+kk, c0..]
    //    条带序列 q = (段序号 - 首段) * S + r；JIT 时"下一条"的转置交错在本条的 4 个 K 循环里（两块 A 缓冲交替）。
    int cpy = 0;   // 预打包副本选择（按任务交替）
    if (!(fast_owner ? 0 : c->jit) && c->pp[1]) cpy = (int)((i + p) & 1);
    if (!(fast_owner ? 0 : c->jit) && ((fast_owner ? 0 : c->ppmode) == 1 || c->nobar) && i0 > 0) {   // 等本行块各条打包完成（nobar 时预打包也按条设 flag）
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        for (int r = 0; r < S; r++)
            while (__atomic_load_n(&c->pflags[i0 / 16 + r], __ATOMIC_ACQUIRE) == 0) asm volatile("yield");
        if (c->prof >= 2) tb->t_wait += cyc() - t0;
    }
    int pend = 0;            // 有待回写的上一微块（暂存中）
    double* pend_c = NULL;
    long pend_cols = 0;
    const long xlines = ((nci * 8 + 63) / 64 + 1) & ~1L;   // X 面板每行预取的 cache 行数（偶数）
    const long chfirst = split ? tk->ch0 : 0;
    const long kend = split ? tk->ch1 * kc : i0;
    const long nchunks = (kend - chfirst * kc + kc - 1) / kc;
    const long nstrips = nchunks * (fast_owner ? S / 2 : S);
    double* Abuf[2] = { tb->astrip, tb->astrip + 64L * kc };
    // 对角条预取：GEMM 最后两段里走线器 2 的空闲槽顺序扫本行块的对角打包条（否则对角阶段开头全是冷读，约 20 µs/任务）
    const char* dpf = (const char*)(c->diag + c->diag_off[i]);
    const char* dend = dpf + (size_t)diag_strip_off(S) * 8;
    int hosted_next = 0;     // 下一条已由本条 K 循环转置好
    const int can_host = !(fast_owner ? 0 : c->direct) && (fast_owner ? 0 : c->jit) && (fast_owner ? 1 : c->packmode) != 0 && ntile == 4 && (kc & 31) == 0 && kc >= 32;
    if (fast_owner) sme_zero_za();
    for (long q = 0; q < nstrips; q++) {
        const long ch = chfirst + q / (fast_owner ? S / 2 : S);
        const int r = fast_owner ? (int)(q % (S / 2)) * 2 : (int)(q % S);
        const long k0 = ch * kc;
        long kk = fast_owner ? 64 : i0 - k0; if (kk > kc) kk = kc;
        const long row0 = i0 + 16 * r;
        const long k0n = k0 + kc;
        if (!fast_owner && r == 0) PROF_T(c, wait_flag(c, (k0 / mb) * c->npan + p), t_wait);
        // 本段 B 面板（连续，行距 ncp）与下一段面板（预取目标）：面板驻留模式直接取本线程面板缓冲的第 ch 段
        const double* Xk = (fast_owner ? 1 : c->owner) ? tb->cbuf + (size_t)ch * kc * ncp : c->xp + ((size_t)ch * c->npan + p) * kc * ncp;
        const double* Xkn = (fast_owner ? 1 : c->owner) ? tb->cbuf + (size_t)(ch + 1) * kc * ncp : c->xp + ((size_t)(ch + 1) * c->npan + p) * kc * ncp;
        (void)xlines;
        // 本条 A
        const double* A;
        long ase = 128, aso = 128;
        if ((fast_owner ? 0 : c->jit)) {
            double* Acur = Abuf[q & 1];
            if (!hosted_next) {
                long rows = c->m - row0; if (rows > 16) rows = 16;
                if (rows == 16 && (kk & 31) == 0 && (fast_owner ? 1 : c->packmode) == 1) {
                    PROF_T(c, pack_strip_ld4(c->L, lda, row0, k0, kk, Acur), t_pack);
                } else if (rows == 16 && (kk & 31) == 0 && (fast_owner ? 1 : c->packmode) == 2) {
                    PROF_T(c, pack_strip_ld2(c->L, lda, row0, k0, kk, Acur), t_pack);
                    ase = 128; aso = 384;
                } else if (rows == 16 && (kk & 7) == 0 && (fast_owner ? 1 : c->packmode) == 3) {
                    PROF_T(c, pack_strip_za(c->L, lda, row0, k0, kk, Acur), t_pack);
                    ase = aso = 512;
                } else {
                    PROF_T(c, pack_strip(c->L, lda, c->m, row0, k0, kk, Acur), t_pack);
                }
            }
            A = Acur;
        } else {
            A = c->pp[cpy] + (fast_owner ? c->pp_off[i0 / 16] + ((size_t)ch * S + r) * 1024 : packed_l_offset(c, i0 / 16 + r, k0));
        }
        // 下一条（q+1）：能否在本条 K 循环里转置
        long nrow0 = 0, nk0 = 0;
        int host_now = 0;
        if (q + 1 < nstrips) {
            long qn = q + 1;
            nrow0 = i0 + 16 * (qn % S);
            nk0 = (chfirst + qn / S) * kc;
            long nkk = i0 - nk0; if (nkk > kc) nkk = kc;
            host_now = can_host && (c->m - nrow0 >= 16) && nkk == kk && (kk & 31) == 0;
        }
        hosted_next = host_now;
        // 2 条之后（预取目标）：转置读的是 q+1 的行，因此预取 q+2；不托管时预取 q+1
        long pfq = q + (host_now ? 2 : 1);
        for (long t = 0; t < (fast_owner ? 6 : ntile); t++) {
            pf_t pf = pf_dummy;
            // 游标 1：条 pfq 的 A（每微块 K/2 行；JIT：4 个微块各 4 行 L；预打包：连续 4 段）
            if (pf_on && pfq < nstrips) {
                long pr0 = i0 + (fast_owner ? 32 * (pfq % (S / 2)) : 16 * (pfq % S));
                long pk0 = (chfirst + pfq / (fast_owner ? S / 2 : S)) * kc;
                long pkk = fast_owner ? 64 : i0 - pk0; if (pkk > kc) pkk = kc;
                long lines_per_row = (pkk * 8 + 63) / 64;
                long rows_per_tile = (kk / 2) / lines_per_row; if (rows_per_tile < 1) rows_per_tile = 1;
                if ((fast_owner ? 0 : c->jit)) {
                    long rr0 = t * rows_per_tile;
                    if (rr0 < 16 && c->m - pr0 > rr0) {
                        pf.p = (const char*)(c->L + (pr0 + rr0) * lda + pk0);
                        pf.pfn = lines_per_row;
                        pf.pfj = lda * 8 - lines_per_row * 64;
                    }
                } else {
                    const double* An = c->pp[cpy] + (fast_owner ? c->pp_off[i0 / 16] + (size_t)pfq * 2048 : packed_l_offset(c, pr0 / 16, pk0));
                    pf.p = (const char*)An + (size_t)t * (kk / 2) * 64;
                    pf.pfn = PF_CONTIG_LINES;
                    pf.pfj = 0;
                }
            }
            // 游标 2：下一段 X 面板（连续），只在本段后半的条里发，每微块 K/2 行；空闲槽在最后两段里用于对角条
            int p2_used = 0;
            // 面板驻留模式（TRSM_OWNER_PF2，默认开）：X 面板在本线程 L2 里不用预取，游标 2 改为把 A 流预取得更深——
            // 预打包 A 按段主序连续，游标 1 的 ntile 个微块只盖住下一条的 ntile×K/2 行，游标 2 从其后接着盖（下一条余下部分 + 下下条）。
            // 官方 bench 交错三轮：2.48/2.49/2.47 → 2.35/2.37/2.44 ms。游标 3 也改预取 A（再深 8 KB/微块）反而慢到 2.6–2.7：突发 prfm 过多
            if (pf_on && (fast_owner ? 1 : c->owner) && c->pf2 && !(fast_owner ? 0 : c->jit) && pfq < nstrips) {
                long pr0 = i0 + (fast_owner ? 32 * (pfq % (S / 2)) : 16 * (pfq % S));
                long pk0 = (chfirst + pfq / (fast_owner ? S / 2 : S)) * kc;
                const double* An = c->pp[cpy] + (fast_owner ? c->pp_off[i0 / 16] + (size_t)pfq * 2048 : packed_l_offset(c, pr0 / 16, pk0));
                pf.p2 = (const char*)An + (size_t)((fast_owner ? 6 : ntile) + t) * (kk / 2) * 64;
                p2_used = 1;
            }
            if (pf_on && !p2_used && k0n < kend) {
                long slots_needed = (kk * ncp * 8 + kk * 32 - 1) / (kk * 32);  // 每槽 kk/2 行
                long strips_needed = (slots_needed + ntile - 1) / ntile;
                long xstart = S - strips_needed; if (xstart < 0) xstart = 0;
                if (r >= xstart) {
                    long slot = (r - xstart) * ntile + t;
                    if (slot < slots_needed) { pf.p2 = (const char*)Xkn + slot * (kk / 2) * 64; p2_used = 1; }
                }
            }
            if (pf_on && !p2_used && !split && k0n + kc >= kend && dpf < dend) {
                pf.p2 = dpf;
                dpf += (kk / 2) * 64;
            }
            long cols = nci - 32 * t; if (cols > 32) cols = 32;
            // 游标 3：本微块的 C 行（其回写在下一微块的 K 循环里），pldl2keep 提前触碰
            if (pf_on) { pf.pc = (const char*)(acc + 16L * r * ncp + 32 * t); pf.pcs = cs; }
            if (fast_owner) {
                // Adjacent row strips share one 16-column X tile in L1.
                double* cur = acc + 16L * r * ncp + 16 * t;
                if(fast_owner==2){
                PROF_T(c, kloop_owner_cursor0(A, Xk + 16 * t, pend ? pend_c : cur, pf.p, pf.p2, (const char*)cur), t_kloop);
                PROF_T(c, kloop_owner_cursor4(A + 1024, Xk + 16 * t, cur, pf.p + 1024, pf.p2 + 1024,
                    (const char*)(cur + 16 * ncp)), t_kloop);
                }else{
                PROF_T(c, kloop_owner_bank0(A, Xk + 16 * t, pend ? pend_c : cur, pf.p, pf.p2, (const char*)cur), t_kloop);
                PROF_T(c, kloop_owner_bank4(A + 1024, Xk + 16 * t, cur, pf.p + 1024, pf.p2 + 1024,
                    (const char*)(cur + 16 * ncp)), t_kloop);
                }
                pend = 1; pend_c = cur + 16 * ncp; pend_cols = 16;
                if (c->prof >= 2) tb->n_kstep += kk;
                continue;
            }
            sw_advance(tb);
            if((fast_owner ? 0 : c->direct)) {
                PROF_T(c,za_load_direct(acc+16L*r*ncp+32*t,cs,cols),t_load);
            } else sme_zero_za();
            if((fast_owner ? 0 : c->direct)) {
                if((fast_owner ? 1 : c->ant))PROF_T(c,kloop_plain_direct(A,ase,aso,Xk+32*t,cs,kk,cols,&pf),t_kloop);
                else PROF_T(c,kloop_plain_direct_t(A,ase,aso,Xk+32*t,cs,kk,cols,&pf),t_kloop);
            } else if (host_now) {
                int epi = pend && kk >= 96;
                if (pend && !epi) { PROF_T(c, epi_sub_std(scratch, pend_c, cs, pend_cols), t_epi); }
                long kb0 = t * (kk / 4);
                if (epi)
                    PROF_T(c, kloop_tr_epi(A, Xk + 32 * t, cs, kk, &pf, scratch, pend_c, cs, pend_cols,
                                           c->L + nrow0 * lda + nk0 + kb0, lda * 8, Abuf[(q + 1) & 1] + kb0 * 16), t_kloop);
                else
                    PROF_T(c, kloop_tr_noepi(A, Xk + 32 * t, cs, kk, &pf,
                                             c->L + nrow0 * lda + nk0 + kb0, lda * 8, Abuf[(q + 1) & 1] + kb0 * 16), t_kloop);
            } else if (pend && kk >= 65) {
                if ((fast_owner ? 1 : c->ant)) PROF_T(c, kloop_epi_sub(A, ase, aso, Xk + 32 * t, cs, kk, cols, &pf, scratch, pend_c, cs, pend_cols), t_kloop);
                else PROF_T(c, kloop_epi_sub_t(A, ase, aso, Xk + 32 * t, cs, kk, cols, &pf, scratch, pend_c, cs, pend_cols), t_kloop);
            } else if (pend && kk == 64) {
                if ((fast_owner ? 1 : c->ant)) PROF_T(c, kloop_epi64_sub(A, ase, aso, Xk + 32 * t, cs, cols, &pf, scratch, pend_c, cs, pend_cols), t_kloop);
                else PROF_T(c, kloop_epi64_sub_t(A, ase, aso, Xk + 32 * t, cs, cols, &pf, scratch, pend_c, cs, pend_cols), t_kloop);
            } else {
                if (pend) PROF_T(c, epi_sub_std(scratch, pend_c, cs, pend_cols), t_epi);
                if ((fast_owner ? 1 : c->ant)) PROF_T(c, kloop_plain_nt(A, ase, aso, Xk + 32 * t, cs, kk, cols, &pf), t_kloop);
                else PROF_T(c, kloop_plain(A, ase, aso, Xk + 32 * t, cs, kk, cols, &pf), t_kloop);
            }
            if((fast_owner ? 0 : c->direct)) {
                PROF_T(c,za_store_direct(acc+16L*r*ncp+32*t,cs,cols),t_dump);
                pend=0;
            } else {
                PROF_T(c, za_dump(scratch), t_dump);
                pend = 1; pend_c = acc + 16L * r * ncp + 32 * t; pend_cols = cols;
            }
            if (c->prof >= 2) tb->n_kstep += kk;
        }
    }
    if (fast_owner && pend) { PROF_T(c, za_flush_duplex_bank4(pend_c, cs), t_epi); pend = 0; }
    if (pend) { PROF_T(c, epi_sub_std(scratch, pend_c, cs, pend_cols), t_epi); pend = 0; }

    if (split) {
        // 段完成：最后完成者归约全部部分和到 C 块并继续对角求解，其余线程返回
        int done = __atomic_add_fetch(&c->seg_cnt[i * c->npan + p], 1, __ATOMIC_ACQ_REL);
        if (done < tk->nseg) { if (c->prof >= 2) { tb->t_task += cyc() - tt0; tb->n_task++; } return -1; }
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        const svbool_t pg = svptrue_b64();
        const long lpf = c->loadpf;   // 装载阶段软件预取：提前 lpf 行把 B 子块行拉进 L1（每行 nci*8/64 个 cache 行）
        for (long r = 0; r < 16L * S; r++) {
            double* d = cb + r * ncp;
            if (lpf > 0 && r + lpf < mbi) {
                const char* pr = (const char*)(Bblk + (r + lpf) * ldb);
                for (long b = 0; b < nci * 8; b += 64) __builtin_prefetch(pr + b, 0, 3);
            }
            if (r < mbi) {
                const double* s = Bblk + r * ldb;
                long j = 0;
                for (; j + 8 <= nci; j += 8) svst1_f64(pg, d + j, svld1_f64(pg, s + j));
                if (j < nci) {
                    svbool_t pj = svwhilelt_b64((long)j, (long)nci);
                    svst1_f64(pj, d + j, svld1_f64(pj, s + j));
                    j += 8;
                }
                for (; j < nc; j += 8) svst1_f64(pg, d + j, svdup_f64(0.0));
            } else {
                for (long j = 0; j < nc; j += 8) svst1_f64(pg, d + j, svdup_f64(0.0));
            }
        }
        for (int sg = 0; sg < tk->nseg; sg++) {
            const double* P = c->part + tk->part_off + (size_t)sg * mb * ncp;
            for (long q = 0; q < 16L * S * ncp; q += 8)
                svst1_f64(pg, cb + q, svadd_f64_x(pg, svld1_f64(pg, cb + q), svld1_f64(pg, P + q)));
        }
        if (c->prof >= 2) tb->t_load += cyc() - t0;
    }

    // 3. 对角求解：X[s] = Linv_ss C[s] - Σ (Linv_ss L_ss') X[s']，A = 对角打包条带，B = C 块行 0..16(s+1)
    if (c->nobar) {   // 本行块的对角条是否都已预处理好（无 barrier 模式）
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        while (__atomic_load_n(&c->dcnt[i], __ATOMIC_ACQUIRE) < S) asm volatile("yield");
        if (c->prof >= 2) tb->t_wait += cyc() - t0;
    }
    if ((fast_owner ? 1 : c->reserve) && !(fast_owner ? 1 : c->owner)) {   // 此时预占下一任务：提前量只有对角阶段 + 回写，不会明显加深依赖流水（面板驻留模式下一任务已确定，作为 tkn_guess 传入）
        tnext = __atomic_fetch_add((long*)&c->next, 1, __ATOMIC_RELAXED);
        tkn = tnext < c->ntask_items ? &c->tasks[tnext] : NULL;
    }
    {
        const double* D = c->diag + c->diag_off[i];
        // 下一任务的 B 子块（行距 ldb 的 1 KB 行段）在对角阶段用游标 1 预取；对角条带（连续）用游标 2
        long nb_rows = 0, nb_done = 0, nb_lines = 0;
        const double* nb_base = NULL;
        long nb_ldb8 = ldb * 8;
        if (pf_on && tkn && tkn->kind == 0 && tkn->nseg == 1) {
            long ni0 = (long)tkn->i * mb, nc0 = pan_c0(c, tkn->p);
            nb_rows = c->m - ni0; if (nb_rows > mb) nb_rows = mb;
            long nnci = pan_nci(c, tkn->p); if (nnci > nc) nnci = nc;
            nb_lines = (nnci * 8 + 63) / 64;
            nb_base = c->B + ni0 * ldb + nc0;
        }
        for (int s = 0; s < S; s++) {
            const double* A = D + diag_strip_off(s);
            long K = 16L * (s + 1);
            for (long t = 0; t < ntile; t++) {
                pf_t pf = pf_dummy;
                if (pf_on && s + 1 < S) pf.p2 = (const char*)(D + diag_strip_off(s + 1)) + t * K * 32;
                if (nb_base && nb_done < nb_rows) {
                    long rows_this = (K / 2) / nb_lines; if (rows_this < 1) rows_this = 1;
                    pf.p = (const char*)(nb_base + nb_done * ldb);
                    pf.pfn = nb_lines;
                    pf.pfj = nb_ldb8 - nb_lines * 64;
                    nb_done += rows_this;
                }
                long cols = nci - 32 * t; if (cols > 32) cols = 32;
                if (pf_on) { pf.pc = (const char*)(cb + 16L * s * ncp + 32 * t); pf.pcs = cs; }
                sw_advance(tb);
                sme_zero_za();
                // ldnt1d 只在条带能留在 L1 时用（K*128 B ≤ 32 KB）：更长的条带 ntile 次重读要靠 L2，非临时装载会让后几个微块重新去内存取
                const int dnt = (fast_owner ? 1 : c->dnt) && K <= 256;
                if (fast_owner) {
                    PROF_T(c, kloop_plain_nt(A, 128, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);
                    PROF_T(c, za_owner_dstore(cb + 16L * s * ncp + 32 * t), t_ddump);
                    if (c->prof >= 2) { tb->n_kstep += K; tb->n_dkstep += K; }
                    continue;
                }
                if (pend && K >= 65) {
                    if (dnt) PROF_T(c, kloop_epi_set_nt(A, 128, 128, cb + 32 * t, cs, K, cols, &pf, scratch, pend_c, cs, pend_cols), t_dkloop);
                    else PROF_T(c, kloop_epi_set(A, 128, 128, cb + 32 * t, cs, K, cols, &pf, scratch, pend_c, cs, pend_cols), t_dkloop);
                } else if (pend && K == 64) {
                    if (dnt) PROF_T(c, kloop_epi64_set_nt(A, 128, 128, cb + 32 * t, cs, cols, &pf, scratch, pend_c, cs, pend_cols), t_dkloop);
                    else PROF_T(c, kloop_epi64_set(A, 128, 128, cb + 32 * t, cs, cols, &pf, scratch, pend_c, cs, pend_cols), t_dkloop);
                } else {
                    if (pend) PROF_T(c, epi_set_std(scratch, pend_c, cs, pend_cols), t_depi);
                    if (dnt) PROF_T(c, kloop_plain_nt(A, 128, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);
                    else PROF_T(c, kloop_plain(A, 128, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);
                }
                PROF_T(c, za_dump(scratch), t_ddump);
                pend = 1; pend_c = cb + 16L * s * ncp + 32 * t; pend_cols = cols;
                if (c->prof >= 2) { tb->n_kstep += K; tb->n_dkstep += K; }
            }
            // 条末必须落地：下一条把这些行当 B 操作数
            if (pend) { PROF_T(c, epi_set_std(scratch, pend_c, cs, pend_cols), t_depi); pend = 0; }
        }
    }

    // 4. C 块 → Xp（同步，消费者要读）→ X（可托管到后续 K 循环），发布 flag
    {
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        const svbool_t pg = svptrue_b64();
        // 上一任务遗留的托管回写先同步收尾（通常已在 K 循环里完成）
        sw_flush(tb);
        // 面板主序副本：行 i0+r → 面板 ((i0+r)/kc, p) 的第 (i0+r)%kc 行（整行 nc 列，含零填充）；最后一个行块没有消费者，不写
        double* Bout=Bblk;

        const int last_rb = (i + 1 == c->nrb);
        if (!last_rb && !(fast_owner ? 1 : c->owner)) for (long r = 0; r < mbi; r++) {
            long gr = i0 + r;
            double* d = c->xp + ((size_t)(gr / kc) * c->npan + p) * kc * ncp + (gr % kc) * ncp;
            const double* s = cb + r * ncp;
            if ((fast_owner ? 0 : c->xpnt)) for (long j = 0; j < nc; j += 8) svstnt1_f64(pg, d + j, svld1_f64(pg, s + j));
            else for (long j = 0; j < nc; j += 8) svst1_f64(pg, d + j, svld1_f64(pg, s + j));
        }
        // 托管条件：整面板（nci==nc，Xp 行距 = 每行 64 B 段数×64）且有 Xp 副本可作源。X 行可以不对齐（bench 的 malloc 只有 16 B 对齐），
        // stnt1d 支持非对齐，只是跨行的 64 B 段会拆成两次部分行写。
        const int can_sw = (fast_owner ? 0 : c->swhost) && !(fast_owner ? 1 : c->owner) && nci == nc && ((nc * 8) & 63) == 0 && !last_rb;
        if (can_sw) {
            for (long r0 = 0; r0 < mbi; r0 += kc) {
                long rows = mbi - r0; if (rows > kc) rows = kc;
                sw_t w;
                w.src = (const char*)(c->xp + ((size_t)((i0 + r0) / kc) * c->npan + p) * kc * ncp);
                w.dst = (char*)(Bout + r0 * ldb);
                w.lpr = nc * 8 / 64; w.cnt = w.lpr;
                w.dstj = ldb * 8 - nc * 8;
                w.rows = rows;
                tb->swq[tb->swn++] = w;
                tb->n_swq += rows * w.lpr;
            }
        } else if ((fast_owner ? 1 : c->ntstore)) {
            // 下一任务 A 预取（fast_owner，TRSM_NEXTA）：任务只有 20–80 µs，任务开头的冷 A 摊不掉（按 i 凸增的任务耗时）；
            // 借回写循环逐行发 prfm（zgemm"打包循环内预取"模式）。档位：1=首块前 16 KB；2=首块整 32 KB；
            // 3=首块 32 KB + "新"块 chi 32 KB（chi 是下一任务相对当前任务唯一没读过的 A 块）
            const char* na = NULL; const char* na2 = NULL; long nal = 0;
            if (fast_owner && c->nexta && tkn && tkn->kind == 0 && tkn->i > 0) {
                na = (const char*)c->pp[cpy] + c->pp_off[(long)tkn->i * (mb / 16)];
                nal = c->nexta >= 2 ? 512 : 256;                     // 每行预取字节数（首块）
                if (c->nexta >= 3 && tkn->i > 1)
                    na2 = na + (size_t)(tkn->i - 1) * 32768;          // chi = 首块 + (i'-1)*32KB
            }
            // 非临时存储按 X 行的 cache 行对齐：头部分行（谓词）+ 整行 + 尾部分行。bench 的 B 只 16 B 对齐，
            // 若按 64 B 向量直接存，每个向量都跨两行 → 32 次部分行写/行；对齐后是 15 次整行 + 2 次部分行。
            for (long r = 0; r < mbi; r++) {
                const double* s = cb + r * ncp;
                double* d = Bout + r * ldb;
                if (na) {
                    const char* q = na + r * nal;
                    for (long b = 0; b < nal; b += 64) __builtin_prefetch(q + b, 0, 2);
                    if (na2) {
                        const char* q2 = na2 + r * 512;
                        for (long b = 0; b < 512; b += 64) __builtin_prefetch(q2 + b, 0, 2);
                    }
                }
                long pre = (long)(((64 - ((uintptr_t)d & 63)) & 63) >> 3);   // 到下一个 64 B 边界的 double 数（0..7）
                if (pre > nci) pre = nci;
                long j = 0;
                if (pre) { svbool_t ph = svwhilelt_b64(0L, pre); svstnt1_f64(ph, d, svld1_f64(ph, s)); j = pre; }
                for (; j + 8 <= nci; j += 8) svstnt1_f64(pg, d + j, svld1_f64(pg, s + j));
                if (j < nci) {
                    svbool_t pj = svwhilelt_b64((long)j, (long)nci);
                    svstnt1_f64(pj, d + j, svld1_f64(pj, s + j));
                }
            }
        } else {
            for (long r = 0; r < mbi; r++) {
                const double* s = cb + r * ncp;
                double* d = Bout + r * ldb;
                long j = 0;
                for (; j + 8 <= nci; j += 8) svst1_f64(pg, d + j, svld1_f64(pg, s + j));
                if (j < nci) {
                    svbool_t pj = svwhilelt_b64((long)j, (long)nci);
                    svst1_f64(pj, d + j, svld1_f64(pj, s + j));
                }
            }
        }
        if (c->prof >= 2) tb->t_store += cyc() - t0;
    }
    __atomic_store_n(&c->flags[i * c->npan + p], 1, __ATOMIC_RELEASE);
    if (c->prof >= 2) { tb->t_task += cyc() - tt0; tb->n_task++; }
    (void)tid;
    return tnext;
}

static long run_task(const ctx_t* c, tbuf_t* tb, const task_t* tk, const task_t* guess, int tid)
{ return run_task_impl(c, tb, tk, guess, tid, 0); }
static long run_task_owner(const ctx_t* c, tbuf_t* tb, const task_t* tk, const task_t* guess, int tid)
{ return run_task_impl(c, tb, tk, guess, tid, 1); }
static long run_task_owner_cursor(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid)
{ return run_task_impl(c,tb,tk,guess,tid,2); }

// Constant block configurations let the compiler remove generic index and mode work.
// Keep the general task independent; it handles all unmatched settings and layouts.
static inline size_t special_l_offset(const ctx_t* c,int mode,long g,long k0) {
 long mb=mode==2?256:192;
 long i=16*g/mb,g0=i*mb/16;
 return c->pp_off[g0]+(size_t)(k0/64)*c->S_of[i]*1024+(g-g0)*1024+(k0%64)*16;
}

// Complete FP64 panels use two independent 16x16 accumulator banks:
// ZA0..3 holds columns 0..15 and ZA4..7 holds columns 16..31.
// A kernel clears/computes its own bank while MOVA drains the preceding bank
// into C. This overlaps output subtraction with FMOPA and avoids ZA scratch
// dumps. Only the final ZA4..7 bank needs a separate flush.
// Modes 2/4 use MB256/192 with NC128/KC64. Xp stores eight contiguous Kx16
// micro-panels per chunk; C remains row-major with stride 128. A loads are
// temporal so the second column half can reuse the same 16-row strip.
// The last K step performs no preload, including at the allocation boundary.
static inline void za_flush_duplex_bank4(double* c, long bs)
{
    char* ct = (char*)c;
    char* cb = ct + 8 * bs;
    asm volatile(
 "ptrue p0.d\n\t"
 "mov w12, #0\n\t"
 "1:\n\t"
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "mova z16.d, p0/m, za5h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #1, mul vl]\n\t"
 "mova z16.d, p0/m, za6h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[cb], #0, mul vl]\n\t"
 "mova z16.d, p0/m, za7h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], %[bs]\n\t"
 "add %[cb], %[cb], %[bs]\n\t"
 "add w12, w12, #1\n\t"
 "cmp w12, #8\n\t"
 "b.lt 1b\n\t"
: [ct] "+&r"(ct),[cb] "+&r"(cb):[bs] "r"(bs): "memory","cc","x12",VCLOB,PCLOB);}
// Fixed KC64: preload prior ZA/C rows, then subtract after independent K work.
// Expand all eight output rows; the final K step never loads another operand.
#define DUPLEX0_U_BODY(BS, CS, C8, C864, ALD) \
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t" \
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t" \
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t" \
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t" \
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t" \
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t" \
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t" \
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z17.d, z17.d, z16.d\n\t" \
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z19.d, z19.d, z18.d\n\t" \
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t" \
 "prfm pldl2keep, [%[pc]]\n\t" \
 "prfm pldl2keep, [%[pc], #64]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C8 "]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C864 "]\n\t" \
 "add %[pc], %[pc], #" #CS "\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z21.d, z21.d, z20.d\n\t" \
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"
#define DUPLEX0_U_NEXT(BS, ALD) \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"
#define DUPLEX0_U_LAST \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
#define DUPLEX0_U_TAIL(CS) \
 "fsub z23.d, z23.d, z22.d\n\t" \
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t" \
 "add %[ct], %[ct], #" #CS "\n\t" \
 "add %[cb], %[cb], #" #CS "\n\t"
static inline __attribute__((always_inline)) void kloop_duplex_bank0(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 8192;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za0.d, za1.d, za2.d, za3.d}\n\t"
 "mov w12, #0\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #128\n\t"
 "mov w12, #0\n\t" DUPLEX0_U_BODY(128, 1024, 8192, 8256, "ld1d ") DUPLEX0_U_NEXT(128, "ld1d ") DUPLEX0_U_TAIL(1024)
 "mov w12, #1\n\t" DUPLEX0_U_BODY(128, 1024, 8192, 8256, "ld1d ") DUPLEX0_U_NEXT(128, "ld1d ") DUPLEX0_U_TAIL(1024)
 "mov w12, #2\n\t" DUPLEX0_U_BODY(128, 1024, 8192, 8256, "ld1d ") DUPLEX0_U_NEXT(128, "ld1d ") DUPLEX0_U_TAIL(1024)
 "mov w12, #3\n\t" DUPLEX0_U_BODY(128, 1024, 8192, 8256, "ld1d ") DUPLEX0_U_NEXT(128, "ld1d ") DUPLEX0_U_TAIL(1024)
 "mov w12, #4\n\t" DUPLEX0_U_BODY(128, 1024, 8192, 8256, "ld1d ") DUPLEX0_U_NEXT(128, "ld1d ") DUPLEX0_U_TAIL(1024)
 "mov w12, #5\n\t" DUPLEX0_U_BODY(128, 1024, 8192, 8256, "ld1d ") DUPLEX0_U_NEXT(128, "ld1d ") DUPLEX0_U_TAIL(1024)
 "mov w12, #6\n\t" DUPLEX0_U_BODY(128, 1024, 8192, 8256, "ld1d ") DUPLEX0_U_NEXT(128, "ld1d ") DUPLEX0_U_TAIL(1024)
 "mov w12, #7\n\t" DUPLEX0_U_BODY(128, 1024, 8192, 8256, "ld1d ") DUPLEX0_U_LAST DUPLEX0_U_TAIL(1024)
: [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc):: "memory","cc","x12",VCLOB,PCLOB);}
static inline __attribute__((always_inline)) void kloop_owner_bank0(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 6144;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za0.d, za1.d, za2.d, za3.d}\n\t"
 "mov w12, #0\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "mov w12, #0\n\t" DUPLEX0_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX0_U_NEXT(768, "ldnt1d ") DUPLEX0_U_TAIL(768)
 "mov w12, #1\n\t" DUPLEX0_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX0_U_NEXT(768, "ldnt1d ") DUPLEX0_U_TAIL(768)
 "mov w12, #2\n\t" DUPLEX0_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX0_U_NEXT(768, "ldnt1d ") DUPLEX0_U_TAIL(768)
 "mov w12, #3\n\t" DUPLEX0_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX0_U_NEXT(768, "ldnt1d ") DUPLEX0_U_TAIL(768)
 "mov w12, #4\n\t" DUPLEX0_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX0_U_NEXT(768, "ldnt1d ") DUPLEX0_U_TAIL(768)
 "mov w12, #5\n\t" DUPLEX0_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX0_U_NEXT(768, "ldnt1d ") DUPLEX0_U_TAIL(768)
 "mov w12, #6\n\t" DUPLEX0_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX0_U_NEXT(768, "ldnt1d ") DUPLEX0_U_TAIL(768)
 "mov w12, #7\n\t" DUPLEX0_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX0_U_LAST DUPLEX0_U_TAIL(768)
: [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc):: "memory","cc","x12",VCLOB,PCLOB);}
#undef DUPLEX0_U_BODY
#undef DUPLEX0_U_NEXT
#undef DUPLEX0_U_LAST
#undef DUPLEX0_U_TAIL

// Fixed KC64: preload prior ZA/C rows, then subtract after independent K work.
// Expand all eight output rows; the final K step never loads another operand.
#define DUPLEX4_U_BODY(BS, CS, C8, C864, ALD) \
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t" \
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t" \
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t" \
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t" \
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t" \
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t" \
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t" \
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z17.d, z17.d, z16.d\n\t" \
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z19.d, z19.d, z18.d\n\t" \
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t" \
 "prfm pldl2keep, [%[pc]]\n\t" \
 "prfm pldl2keep, [%[pc], #64]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C8 "]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C864 "]\n\t" \
 "add %[pc], %[pc], #" #CS "\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z21.d, z21.d, z20.d\n\t" \
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"
#define DUPLEX4_U_NEXT(BS, ALD) \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"
#define DUPLEX4_U_LAST \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
#define DUPLEX4_U_TAIL(CS) \
 "fsub z23.d, z23.d, z22.d\n\t" \
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t" \
 "add %[ct], %[ct], #" #CS "\n\t" \
 "add %[cb], %[cb], #" #CS "\n\t"
// 专用路径 bank4 是同一条 A 的第二次（最后一次）读取：-DTRSM_A4NT 用 ldnt1d，让用完的 A 行不被 RRIP 命中提升、
// 不占 L2 高优先级（zgemm 的缓存提示卫生结论：死数据被提升后会把预取行挤掉）
#ifndef TRSM_A4NT
#define TRSM_A4NT 1     // 实测 case2 −0.1 ms、case3 −0.15 ms（4 轮交错）
#endif
#if TRSM_A4NT
#define DUP4_ALD "ldnt1d "
#else
#define DUP4_ALD "ld1d "
#endif
static inline __attribute__((always_inline)) void kloop_duplex_bank4(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 8192;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za4.d, za5.d, za6.d, za7.d}\n\t"
 "mov w12, #0\n\t"
 "ld1d z0.d, p0/z, [%[a]]\n\t"
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #128\n\t"
 "mov w12, #0\n\t" DUPLEX4_U_BODY(128, 1024, 8192, 8256, DUP4_ALD) DUPLEX4_U_NEXT(128, DUP4_ALD) DUPLEX4_U_TAIL(1024)
 "mov w12, #1\n\t" DUPLEX4_U_BODY(128, 1024, 8192, 8256, DUP4_ALD) DUPLEX4_U_NEXT(128, DUP4_ALD) DUPLEX4_U_TAIL(1024)
 "mov w12, #2\n\t" DUPLEX4_U_BODY(128, 1024, 8192, 8256, DUP4_ALD) DUPLEX4_U_NEXT(128, DUP4_ALD) DUPLEX4_U_TAIL(1024)
 "mov w12, #3\n\t" DUPLEX4_U_BODY(128, 1024, 8192, 8256, DUP4_ALD) DUPLEX4_U_NEXT(128, DUP4_ALD) DUPLEX4_U_TAIL(1024)
 "mov w12, #4\n\t" DUPLEX4_U_BODY(128, 1024, 8192, 8256, DUP4_ALD) DUPLEX4_U_NEXT(128, DUP4_ALD) DUPLEX4_U_TAIL(1024)
 "mov w12, #5\n\t" DUPLEX4_U_BODY(128, 1024, 8192, 8256, DUP4_ALD) DUPLEX4_U_NEXT(128, DUP4_ALD) DUPLEX4_U_TAIL(1024)
 "mov w12, #6\n\t" DUPLEX4_U_BODY(128, 1024, 8192, 8256, DUP4_ALD) DUPLEX4_U_NEXT(128, DUP4_ALD) DUPLEX4_U_TAIL(1024)
 "mov w12, #7\n\t" DUPLEX4_U_BODY(128, 1024, 8192, 8256, DUP4_ALD) DUPLEX4_U_LAST DUPLEX4_U_TAIL(1024)
: [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc):: "memory","cc","x12",VCLOB,PCLOB);}
static inline __attribute__((always_inline)) void kloop_owner_bank4(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 6144;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za4.d, za5.d, za6.d, za7.d}\n\t"
 "mov w12, #0\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "mov w12, #0\n\t" DUPLEX4_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX4_U_NEXT(768, "ldnt1d ") DUPLEX4_U_TAIL(768)
 "mov w12, #1\n\t" DUPLEX4_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX4_U_NEXT(768, "ldnt1d ") DUPLEX4_U_TAIL(768)
 "mov w12, #2\n\t" DUPLEX4_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX4_U_NEXT(768, "ldnt1d ") DUPLEX4_U_TAIL(768)
 "mov w12, #3\n\t" DUPLEX4_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX4_U_NEXT(768, "ldnt1d ") DUPLEX4_U_TAIL(768)
 "mov w12, #4\n\t" DUPLEX4_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX4_U_NEXT(768, "ldnt1d ") DUPLEX4_U_TAIL(768)
 "mov w12, #5\n\t" DUPLEX4_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX4_U_NEXT(768, "ldnt1d ") DUPLEX4_U_TAIL(768)
 "mov w12, #6\n\t" DUPLEX4_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX4_U_NEXT(768, "ldnt1d ") DUPLEX4_U_TAIL(768)
 "mov w12, #7\n\t" DUPLEX4_U_BODY(768, 768, 6144, 6208, "ldnt1d ") DUPLEX4_U_LAST DUPLEX4_U_TAIL(768)
: [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc):: "memory","cc","x12",VCLOB,PCLOB);}
#undef DUPLEX4_U_BODY
#undef DUPLEX4_U_NEXT
#undef DUPLEX4_U_LAST
#undef DUPLEX4_U_TAIL

// Fixed KC64: preload prior ZA/C rows, then subtract after independent K work.
// Expand all eight output rows; the final K step never loads another operand.
#define DUPLEX0_U_BODY(BS, CS, C8, C864, ALD) \
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t" \
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t" \
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t" \
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t" \
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t" \
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t" \
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t" \
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z17.d, z17.d, z16.d\n\t" \
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z19.d, z19.d, z18.d\n\t" \
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t" \
 "prfm pldl2keep, [%[pc]]\n\t" \
 "prfm pldl2keep, [%[pc], #64]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C8 "]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C864 "]\n\t" \
 "add %[pc], %[pc], #" #CS "\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z21.d, z21.d, z20.d\n\t" \
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"
#define DUPLEX0_U_NEXT(BS, ALD) \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"
#define DUPLEX0_U_LAST \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
#define DUPLEX0_U_TAIL(CS) \
 "fsub z23.d, z23.d, z22.d\n\t" \
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t" \
 "add %[ct], %[ct], #" #CS "\n\t" \
 "add %[cb], %[cb], #" #CS "\n\t"
// Fixed KC64: address K rows around the middle of each eight-row group.
// Keep the operand/read order and C/prefetch schedule unchanged.
#define HDUPLEX0_BODY \
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t" \
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t" \
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t" \
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t" \
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t" \
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t" \
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t" \
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t" \
 "ld1d z8.d, p0/z, [%[a], #-6, mul vl]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 "ld1d z9.d, p0/z, [%[a], #-5, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b], #-6, mul vl]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #-5, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "ld1d z0.d, p0/z, [%[a], #-4, mul vl]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "ld1d z1.d, p0/z, [%[a], #-3, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b], #-4, mul vl]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #-3, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "fsub z17.d, z17.d, z16.d\n\t" \
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 "ld1d z8.d, p0/z, [%[a], #-2, mul vl]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 "ld1d z9.d, p0/z, [%[a], #-1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b], #-2, mul vl]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #-1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "ld1d z0.d, p0/z, [%[a], #0, mul vl]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b], #0, mul vl]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "fsub z19.d, z19.d, z18.d\n\t" \
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t" \
 "prfm pldl2keep, [%[pc]]\n\t" \
 "prfm pldl2keep, [%[pc], #64]\n\t" \
 "prfm pldl2keep, [%[pc], #8192]\n\t" \
 "prfm pldl2keep, [%[pc], #8256]\n\t" \
 "add %[pc], %[pc], #1024\n\t" \
 "ld1d z8.d, p0/z, [%[a], #2, mul vl]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 "ld1d z9.d, p0/z, [%[a], #3, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b], #2, mul vl]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #3, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "ld1d z0.d, p0/z, [%[a], #4, mul vl]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "ld1d z1.d, p0/z, [%[a], #5, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b], #4, mul vl]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #5, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "fsub z21.d, z21.d, z20.d\n\t" \
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 "ld1d z8.d, p0/z, [%[a], #6, mul vl]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 "ld1d z9.d, p0/z, [%[a], #7, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b], #6, mul vl]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #7, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
#define HDUPLEX0_NEXT \
 "add %[a], %[a], #1024\n\t" \
 "ld1d z0.d, p0/z, [%[a], #-8, mul vl]\n\t" \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "ld1d z1.d, p0/z, [%[a], #-7, mul vl]\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "add %[b], %[b], #1024\n\t" \
 "ld1d z4.d, p0/z, [%[b], #-8, mul vl]\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #-7, mul vl]\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
#define HDUPLEX0_LAST \
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
#define HDUPLEX0_TAIL \
 "fsub z23.d, z23.d, z22.d\n\t" \
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t" \
 "add %[ct], %[ct], #1024\n\t" \
 "add %[cb], %[cb], #1024\n\t"
static inline __attribute__((always_inline)) void kloop_hoist_bank0(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 8192;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za0.d, za1.d, za2.d, za3.d}\n\t"
 "mov w12, #0\n\t"
 "add %[a], %[a], #512\n\t"
 "ld1d z0.d, p0/z, [%[a], #-8, mul vl]\n\t"
 "ld1d z1.d, p0/z, [%[a], #-7, mul vl]\n\t"
 "add %[b], %[b], #512\n\t"
 "ld1d z4.d, p0/z, [%[b], #-8, mul vl]\n\t"
 "ld1d z5.d, p0/z, [%[b], #-7, mul vl]\n\t"
 "mov w12, #0\n\t" HDUPLEX0_BODY HDUPLEX0_NEXT HDUPLEX0_TAIL
 "mov w12, #1\n\t" HDUPLEX0_BODY HDUPLEX0_NEXT HDUPLEX0_TAIL
 "mov w12, #2\n\t" HDUPLEX0_BODY HDUPLEX0_NEXT HDUPLEX0_TAIL
 "mov w12, #3\n\t" HDUPLEX0_BODY HDUPLEX0_NEXT HDUPLEX0_TAIL
 "mov w12, #4\n\t" HDUPLEX0_BODY HDUPLEX0_NEXT HDUPLEX0_TAIL
 "mov w12, #5\n\t" HDUPLEX0_BODY HDUPLEX0_NEXT HDUPLEX0_TAIL
 "mov w12, #6\n\t" HDUPLEX0_BODY HDUPLEX0_NEXT HDUPLEX0_TAIL
 "mov w12, #7\n\t" HDUPLEX0_BODY HDUPLEX0_LAST HDUPLEX0_TAIL
: [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc):: "memory","cc","x12",VCLOB,PCLOB);}
#undef HDUPLEX0_BODY
#undef HDUPLEX0_NEXT
#undef HDUPLEX0_LAST
#undef HDUPLEX0_TAIL
// Fixed KC64: preload prior ZA/C rows, then subtract after independent K work.
// Expand all eight output rows; the final K step never loads another operand.
#define DUPLEX4_U_BODY(BS, CS, C8, C864, ALD) \
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t" \
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t" \
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t" \
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t" \
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t" \
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t" \
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t" \
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z17.d, z17.d, z16.d\n\t" \
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z19.d, z19.d, z18.d\n\t" \
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t" \
 "prfm pldl2keep, [%[pc]]\n\t" \
 "prfm pldl2keep, [%[pc], #64]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C8 "]\n\t" \
 "prfm pldl2keep, [%[pc], #" #C864 "]\n\t" \
 "add %[pc], %[pc], #" #CS "\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t" \
 "fsub z21.d, z21.d, z20.d\n\t" \
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 ALD "z8.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 ALD "z9.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"
#define DUPLEX4_U_NEXT(BS, ALD) \
 ALD "z0.d, p0/z, [%[a]]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b]]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "add %[a], %[a], #128\n\t" \
 "add %[b], %[b], #" #BS "\n\t"
#define DUPLEX4_U_LAST \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
#define DUPLEX4_U_TAIL(CS) \
 "fsub z23.d, z23.d, z22.d\n\t" \
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t" \
 "add %[ct], %[ct], #" #CS "\n\t" \
 "add %[cb], %[cb], #" #CS "\n\t"
// 专用路径 bank4 是同一条 A 的第二次（最后一次）读取：-DTRSM_A4NT 用 ldnt1d，让用完的 A 行不被 RRIP 命中提升、
// 不占 L2 高优先级（zgemm 的缓存提示卫生结论：死数据被提升后会把预取行挤掉）
#ifndef TRSM_A4NT
#define TRSM_A4NT 1     // 实测 case2 −0.1 ms、case3 −0.15 ms（4 轮交错）
#endif
#if TRSM_A4NT
#define DUP4_ALD "ldnt1d "
#else
#define DUP4_ALD "ld1d "
#endif
// Fixed KC64: address K rows around the middle of each eight-row group.
// Keep the operand/read order and C/prefetch schedule unchanged.
#define HDUPLEX4_BODY \
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t" \
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t" \
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t" \
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t" \
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t" \
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t" \
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t" \
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t" \
 DUP4_ALD "z8.d, p0/z, [%[a], #-6, mul vl]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 DUP4_ALD "z9.d, p0/z, [%[a], #-5, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b], #-6, mul vl]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #-5, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 DUP4_ALD "z0.d, p0/z, [%[a], #-4, mul vl]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 DUP4_ALD "z1.d, p0/z, [%[a], #-3, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b], #-4, mul vl]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #-3, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "fsub z17.d, z17.d, z16.d\n\t" \
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 DUP4_ALD "z8.d, p0/z, [%[a], #-2, mul vl]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 DUP4_ALD "z9.d, p0/z, [%[a], #-1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b], #-2, mul vl]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #-1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 DUP4_ALD "z0.d, p0/z, [%[a], #0, mul vl]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 DUP4_ALD "z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b], #0, mul vl]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "fsub z19.d, z19.d, z18.d\n\t" \
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t" \
 "prfm pldl2keep, [%[pc]]\n\t" \
 "prfm pldl2keep, [%[pc], #64]\n\t" \
 "prfm pldl2keep, [%[pc], #8192]\n\t" \
 "prfm pldl2keep, [%[pc], #8256]\n\t" \
 "add %[pc], %[pc], #1024\n\t" \
 DUP4_ALD "z8.d, p0/z, [%[a], #2, mul vl]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 DUP4_ALD "z9.d, p0/z, [%[a], #3, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b], #2, mul vl]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #3, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t" \
 DUP4_ALD "z0.d, p0/z, [%[a], #4, mul vl]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 DUP4_ALD "z1.d, p0/z, [%[a], #5, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "ld1d z4.d, p0/z, [%[b], #4, mul vl]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #5, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t" \
 "fsub z21.d, z21.d, z20.d\n\t" \
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t" \
 "prfm pldl2keep, [%[ap]]\n\t" \
 "prfm pldl2keep, [%[bp]]\n\t" \
 "add %[ap], %[ap], #64\n\t" \
 "add %[bp], %[bp], #64\n\t" \
 DUP4_ALD "z8.d, p0/z, [%[a], #6, mul vl]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t" \
 DUP4_ALD "z9.d, p0/z, [%[a], #7, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t" \
 "ld1d z12.d, p0/z, [%[b], #6, mul vl]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t" \
 "ld1d z13.d, p0/z, [%[b], #7, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
#define HDUPLEX4_NEXT \
 "add %[a], %[a], #1024\n\t" \
 DUP4_ALD "z0.d, p0/z, [%[a], #-8, mul vl]\n\t" \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 DUP4_ALD "z1.d, p0/z, [%[a], #-7, mul vl]\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "add %[b], %[b], #1024\n\t" \
 "ld1d z4.d, p0/z, [%[b], #-8, mul vl]\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "ld1d z5.d, p0/z, [%[b], #-7, mul vl]\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
#define HDUPLEX4_LAST \
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t" \
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t" \
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t" \
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
#define HDUPLEX4_TAIL \
 "fsub z23.d, z23.d, z22.d\n\t" \
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t" \
 "add %[ct], %[ct], #1024\n\t" \
 "add %[cb], %[cb], #1024\n\t"
static inline __attribute__((always_inline)) void kloop_hoist_bank4(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 8192;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za4.d, za5.d, za6.d, za7.d}\n\t"
 "mov w12, #0\n\t"
 "add %[a], %[a], #512\n\t"
 "ld1d z0.d, p0/z, [%[a], #-8, mul vl]\n\t"
 "ld1d z1.d, p0/z, [%[a], #-7, mul vl]\n\t"
 "add %[b], %[b], #512\n\t"
 "ld1d z4.d, p0/z, [%[b], #-8, mul vl]\n\t"
 "ld1d z5.d, p0/z, [%[b], #-7, mul vl]\n\t"
 "mov w12, #0\n\t" HDUPLEX4_BODY HDUPLEX4_NEXT HDUPLEX4_TAIL
 "mov w12, #1\n\t" HDUPLEX4_BODY HDUPLEX4_NEXT HDUPLEX4_TAIL
 "mov w12, #2\n\t" HDUPLEX4_BODY HDUPLEX4_NEXT HDUPLEX4_TAIL
 "mov w12, #3\n\t" HDUPLEX4_BODY HDUPLEX4_NEXT HDUPLEX4_TAIL
 "mov w12, #4\n\t" HDUPLEX4_BODY HDUPLEX4_NEXT HDUPLEX4_TAIL
 "mov w12, #5\n\t" HDUPLEX4_BODY HDUPLEX4_NEXT HDUPLEX4_TAIL
 "mov w12, #6\n\t" HDUPLEX4_BODY HDUPLEX4_NEXT HDUPLEX4_TAIL
 "mov w12, #7\n\t" HDUPLEX4_BODY HDUPLEX4_LAST HDUPLEX4_TAIL
: [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc):: "memory","cc","x12",VCLOB,PCLOB);}
#undef HDUPLEX4_BODY
#undef HDUPLEX4_NEXT
#undef HDUPLEX4_LAST
#undef HDUPLEX4_TAIL
// Select the address-hoisted KC64 kernels once per solve. Keeping both task
// versions permits same-binary profiling without changing the owner path.
static int g_abhoist;
static inline __attribute__((always_inline)) long run_task_special_impl(const ctx_t* c, tbuf_t* tb, const task_t* tk, const task_t* tkn_guess, int tid, int mode, int hoist)
{
    if (tk->kind == 1) { run_pack_task(c, tk, tb); return -1; }
    long tnext = -1;
    const task_t* tkn = tkn_guess;
    const long mb = (mode==2?256:192), nc = 128, kc = 64, ncp = 128;
    const long i = tk->i, p = tk->p;
    const long i0 = i * mb;
    long mbi = c->m - i0; if (mbi > mb) mbi = mb;
    const long c0 = pan_c0(c, p);   // cshift>0 时奇面板（首/末）由调度路由到通用路径，这里只接完整对齐面板
    // The specialization admission check guarantees a complete 128-column panel.
    long nci=128;
    const int S = c->S_of[i];
    const long ntile = (nci + 31) / 32;
    double* cb = tb->cbuf;
    double* scratch = tb->scratch;
    const long ldb = c->ldb, lda = c->lda;
    const long cs = ncp * 8;      // C / X 面板缓冲行距（字节）
    double* Bblk = c->B + i0 * ldb + c0;
    const pf_t pf_dummy = { (const char*)tb->scratch2, 1, -64, (const char*)tb->scratch2, (const char*)tb->scratch2, 0, &tb->sw };
    const int pf_on = 1;
    uint64_t tt0 = c->prof >= 2 ? cyc() : 0;

    // 分裂段：部分和写到专用缓冲（从 0 开始累加），普通任务直接用 C 块
    const int split = tk->nseg > 1;
    double* acc = split ? c->part + tk->part_off + (size_t)tk->seg * mb * ncp : cb;

    // 1. C 块 ← B 子块（越界列/行补零）；分裂段只清零部分和缓冲
    if (split) {
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        const svbool_t pg = svptrue_b64();
        for (long q = 0; q < 16L * S * ncp; q += 8) svst1_f64(pg, acc + q, svdup_f64(0.0));
        if (c->prof >= 2) tb->t_load += cyc() - t0;
    } else {
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        const svbool_t pg = svptrue_b64();
        const long lpf = c->loadpf;   // 装载阶段软件预取：提前 lpf 行把 B 子块行拉进 L1（每行 nci*8/64 个 cache 行）
        for (long r = 0; r < 16L * S; r++) {
            double* d = cb + r * ncp;
            if (lpf > 0 && r + lpf < mbi) {
                const char* pr = (const char*)(Bblk + (r + lpf) * ldb);
                for (long b = 0; b < nci * 8; b += 64) __builtin_prefetch(pr + b, 0, 3);
            }
            if (r < mbi) {
                const double* s = Bblk + r * ldb;
                long j = 0;
                for (; j + 8 <= nci; j += 8) svst1_f64(pg, d + j, svld1_f64(pg, s + j));
                if (j < nci) {
                    svbool_t pj = svwhilelt_b64((long)j, (long)nci);
                    svst1_f64(pj, d + j, svld1_f64(pj, s + j));
                    j += 8;
                }
                for (; j < nc; j += 8) svst1_f64(pg, d + j, svdup_f64(0.0));
            } else {
                for (long j = 0; j < nc; j += 8) svst1_f64(pg, d + j, svdup_f64(0.0));
            }
        }
        if (c->prof >= 2) tb->t_load += cyc() - t0;
    }

    // 2. GEMM 段：C 块 -= L[i0.., k0..k0+kk) · X[k0..k0+kk, c0..]
    //    条带序列 q = (段序号 - 首段) * S + r；JIT 时"下一条"的转置交错在本条的 4 个 K 循环里（两块 A 缓冲交替）。
    int cpy = 0;   // 预打包副本选择（按任务交替）
    if (!0 && c->pp[1]) cpy = (int)((i + p) & 1);
    // FP64 narrow panels consume asynchronously packed L strips.
    if((mode==4 || c->nobar) && i0>0) {
        uint64_t t0=c->prof>=2?cyc():0;
        for(int r=0;r<S;r++)
            while(__atomic_load_n(&c->pflags[i0/16+r],__ATOMIC_ACQUIRE)==0) asm volatile("yield");
        if(c->prof>=2)tb->t_wait+=cyc()-t0;
    }

    int pend = 0;            // 有待回写的上一微块（暂存中）
    double* pend_c = NULL;
    long pend_cols = 0;
    const long xlines = ((nci * 8 + 63) / 64 + 1) & ~1L;   // X 面板每行预取的 cache 行数（偶数）
    const long chfirst = split ? tk->ch0 : 0;
    const long kend = split ? tk->ch1 * kc : i0;
    const long nchunks = (kend - chfirst * kc + kc - 1) / kc;
    const long nstrips = nchunks * S;
    double* Abuf[2] = { tb->astrip, tb->astrip + 64L * kc };
    // 对角条预取：GEMM 最后两段里走线器 2 的空闲槽顺序扫本行块的对角打包条（否则对角阶段开头全是冷读，约 20 µs/任务）
    const char* dpf = (const char*)(c->diag + c->diag_off[i]);
    const char* dend = dpf + (size_t)diag_strip_off(S) * 8;
    int hosted_next = 0;     // 下一条已由本条 K 循环转置好
    const int can_host = !(mode==1) && 0 && c->packmode != 0 && ntile == 4 && (kc & 31) == 0 && kc >= 32;
    // Prepacked traversal advances one strip at a time; no division by S is needed.
    // The first previous bank is zero; later kernels consume the preceding output.
    if((mode==2 || mode==4))sme_zero_za();
    long strip_ch = chfirst;
    int strip_r = 0;
    for (long q = 0; q < nstrips; q++, strip_r = (strip_r + 1 == S ? (strip_ch++, 0) : strip_r + 1)) {
        const long ch = strip_ch;
        const int r = strip_r;
        const long k0 = ch * kc;
        long kk = i0 - k0; if (kk > kc) kk = kc;
        const long row0 = i0 + 16 * r;
        const long k0n = k0 + kc;
        if (r == 0) PROF_T(c, wait_flag(c, (k0 / mb) * c->npan + p), t_wait);
        const double* Xk = c->xp + ((size_t)ch * c->npan + p) * kc * ncp;         // 本段 B 面板（连续，行距 ncp）
        const double* Xkn = c->xp + ((size_t)(ch + 1) * c->npan + p) * kc * ncp;  // 下一段面板（预取目标）
        (void)xlines;
        // 本条 A
        const double* A;
        long ase = 128, aso = 128;
        if (0) {
            double* Acur = Abuf[q & 1];
            if (!hosted_next) {
                long rows = c->m - row0; if (rows > 16) rows = 16;
                if (rows == 16 && (kk & 31) == 0 && c->packmode == 1) {
                    PROF_T(c, pack_strip_ld4(c->L, lda, row0, k0, kk, Acur), t_pack);
                } else if (rows == 16 && (kk & 31) == 0 && c->packmode == 2) {
                    PROF_T(c, pack_strip_ld2(c->L, lda, row0, k0, kk, Acur), t_pack);
                    ase = 128; aso = 384;
                } else if (rows == 16 && (kk & 7) == 0 && c->packmode == 3) {
                    PROF_T(c, pack_strip_za(c->L, lda, row0, k0, kk, Acur), t_pack);
                    ase = aso = 512;
                } else {
                    PROF_T(c, pack_strip(c->L, lda, c->m, row0, k0, kk, Acur), t_pack);
                }
            }
            A = Acur;
        } else {
            // Each full KC chunk contains S strips of 1024 doubles.
            A = c->pp[cpy] + ((mode==2 || mode==4) ? (c->pp_off[i0/16]+((size_t)ch*S+r)*1024)
                    : special_l_offset(c,mode,i0/16+r,k0));
        }
        // 下一条（q+1）：能否在本条 K 循环里转置
        long nrow0 = 0, nk0 = 0;
        int host_now = 0;
        if (q + 1 < nstrips) {
            long qn = q + 1;
            nrow0 = i0 + 16 * (qn % S);
            nk0 = (chfirst + qn / S) * kc;
            long nkk = i0 - nk0; if (nkk > kc) nkk = kc;
            host_now = can_host && (c->m - nrow0 >= 16) && nkk == kk && (kk & 31) == 0;
        }
        hosted_next = host_now;
        // 2 条之后（预取目标）：转置读的是 q+1 的行，因此预取 q+2；不托管时预取 q+1
        long pfq = q + (host_now ? 2 : 1);
        for (long t = 0; t < ntile; t++) {
            pf_t pf = pf_dummy;
            // 游标 1：条 pfq 的 A（每微块 K/2 行；JIT：4 个微块各 4 行 L；预打包：连续 4 段）
            if (pf_on && pfq < nstrips) {
                long pr0 = i0 + 16 * (r + 1 == S ? 0 : r + 1);
                long pk0 = (ch + (r + 1 == S)) * kc;
                long pkk = i0 - pk0; if (pkk > kc) pkk = kc;
                long lines_per_row = (pkk * 8 + 63) / 64;
                long rows_per_tile = (kk / 2) / lines_per_row; if (rows_per_tile < 1) rows_per_tile = 1;
                if (0) {
                    long rr0 = t * rows_per_tile;
                    if (rr0 < 16 && c->m - pr0 > rr0) {
                        pf.p = (const char*)(c->L + (pr0 + rr0) * lda + pk0);
                        pf.pfn = lines_per_row;
                        pf.pfj = lda * 8 - lines_per_row * 64;
                    }
                } else {
                    const double* An = c->pp[cpy] + ((mode==2 || mode==4) ? (c->pp_off[i0/16]+((size_t)(ch+(r+1==S))*S+(r+1==S?0:r+1))*1024)
                            : special_l_offset(c,mode,pr0/16,pk0));
                    pf.p = (const char*)An + (size_t)t * (kk / 2) * 64;
                    pf.pfn = PF_CONTIG_LINES;
                    pf.pfj = 0;
                }
            }
            // 游标 2：下一段 X 面板（连续），只在本段后半的条里发，每微块 K/2 行；空闲槽在最后两段里用于对角条
            int p2_used = 0;
            if (pf_on && k0n < kend) {
                long slots_needed = (kk * ncp * 8 + kk * 32 - 1) / (kk * 32);  // 每槽 kk/2 行
                long strips_needed = (slots_needed + ntile - 1) / ntile;
                long xstart = S - strips_needed; if (xstart < 0) xstart = 0;
                if (r >= xstart) {
                    long slot = (r - xstart) * ntile + t;
                    if (slot < slots_needed) { pf.p2 = (const char*)Xkn + slot * (kk / 2) * 64; p2_used = 1; }
                }
            }
            if (pf_on && !p2_used && !split && k0n + kc >= kend && dpf < dend) {
                pf.p2 = dpf;
                dpf += (kk / 2) * 64;
            }
            long cols = nci - 32 * t; if (cols > 32) cols = 32;
            // 游标 3：本微块的 C 行（其回写在下一微块的 K 循环里），pldl2keep 提前触碰
            if (pf_on) { pf.pc = (const char*)(acc + 16L * r * ncp + 32 * t); pf.pcs = cs; }
            if((mode==2 || mode==4)) {
                // Keep ZA live between strips; each call drains the other bank.
                double* cur=acc+16L*r*ncp+32*t;
                const char* ap=pf.p; const char* bp=pf.p2;
                if (hoist) {
                PROF_T(c,kloop_hoist_bank0(A,Xk+32*t*kc,(pend?pend_c:cur)+16,ap,bp,pf.pc),t_kloop);
                PROF_T(c,kloop_hoist_bank4(A,Xk+32*t*kc+1024,cur,ap+1024,bp+1024,pf.pc+128),t_kloop);
                } else {
                PROF_T(c,kloop_duplex_bank0(A,Xk+32*t*kc,(pend?pend_c:cur)+16,ap,bp,pf.pc),t_kloop);
                PROF_T(c,kloop_duplex_bank4(A,Xk+32*t*kc+1024,cur,ap+1024,bp+1024,pf.pc+128),t_kloop);
                }
                pend=1;pend_c=cur;pend_cols=cols;
                if(c->prof>=2)tb->n_kstep+=kk;
                continue;
            }

            sw_advance(tb);
            if((mode==1)) {
                PROF_T(c,za_load_fixed(acc+16L*r*ncp+32*t,cs,cols),t_load);
            } else sme_zero_za();
            if((mode==1)) {
                if(c->ant)PROF_T(c,kloop_fixed_direct(A,ase,aso,Xk+32*t,cs,kk,cols,&pf),t_kloop);
                else PROF_T(c,kloop_fixed_direct(A,ase,aso,Xk+32*t,cs,kk,cols,&pf),t_kloop);
            } else if (host_now) {
                int epi = pend && kk >= 96;
                if (pend && !epi) { PROF_T(c, epi_sub_std(scratch, pend_c, cs, pend_cols), t_epi); }
                long kb0 = t * (kk / 4);
                if (epi)
                    PROF_T(c, kloop_tr_epi(A, Xk + 32 * t, cs, kk, &pf, scratch, pend_c, cs, pend_cols,
                                           c->L + nrow0 * lda + nk0 + kb0, lda * 8, Abuf[(q + 1) & 1] + kb0 * 16), t_kloop);
                else
                    PROF_T(c, kloop_tr_noepi(A, Xk + 32 * t, cs, kk, &pf,
                                             c->L + nrow0 * lda + nk0 + kb0, lda * 8, Abuf[(q + 1) & 1] + kb0 * 16), t_kloop);
            } else if (pend && kk >= 65) {
                if (c->ant) PROF_T(c, kloop_epi_sub(A, ase, aso, Xk + 32 * t, cs, kk, cols, &pf, scratch, pend_c, cs, pend_cols), t_kloop);
                else PROF_T(c, kloop_epi_sub_t(A, ase, aso, Xk + 32 * t, cs, kk, cols, &pf, scratch, pend_c, cs, pend_cols), t_kloop);
            } else if (pend && kk == 64) {
                if (c->ant) PROF_T(c, kloop_fixed_epi(A, ase, aso, Xk + 32 * t, cs, cols, &pf, scratch, pend_c, cs, pend_cols), t_kloop);
                else PROF_T(c, kloop_fixed_epi(A, ase, aso, Xk + 32 * t, cs, cols, &pf, scratch, pend_c, cs, pend_cols), t_kloop);
            } else {
                if (pend) PROF_T(c, epi_sub_std(scratch, pend_c, cs, pend_cols), t_epi);
                if (c->ant) PROF_T(c, kloop_fixed_plain(A, ase, aso, Xk + 32 * t, cs, kk, cols, &pf), t_kloop);
                else PROF_T(c, kloop_fixed_plain(A, ase, aso, Xk + 32 * t, cs, kk, cols, &pf), t_kloop);
            }
            if((mode==1)) {
                PROF_T(c,za_store_fixed(acc+16L*r*ncp+32*t,cs,cols),t_dump);
                pend=0;
            } else {
                PROF_T(c, za_dump(scratch), t_dump);
                pend = 1; pend_c = acc + 16L * r * ncp + 32 * t; pend_cols = cols;
            }
            if (c->prof >= 2) tb->n_kstep += kk;
        }
    }
    if((mode==2 || mode==4) && pend) {PROF_T(c,za_flush_duplex_bank4(pend_c+16,cs),t_epi);pend=0;}
    if (pend) { PROF_T(c, epi_sub_std(scratch, pend_c, cs, pend_cols), t_epi); pend = 0; }

    if (split) {
        // 段完成：最后完成者归约全部部分和到 C 块并继续对角求解，其余线程返回
        int done = __atomic_add_fetch(&c->seg_cnt[i * c->npan + p], 1, __ATOMIC_ACQ_REL);
        if (done < tk->nseg) { if (c->prof >= 2) { tb->t_task += cyc() - tt0; tb->n_task++; } return -1; }
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        const svbool_t pg = svptrue_b64();
        const long lpf = c->loadpf;   // 装载阶段软件预取：提前 lpf 行把 B 子块行拉进 L1（每行 nci*8/64 个 cache 行）
        for (long r = 0; r < 16L * S; r++) {
            double* d = cb + r * ncp;
            if (lpf > 0 && r + lpf < mbi) {
                const char* pr = (const char*)(Bblk + (r + lpf) * ldb);
                for (long b = 0; b < nci * 8; b += 64) __builtin_prefetch(pr + b, 0, 3);
            }
            if (r < mbi) {
                const double* s = Bblk + r * ldb;
                long j = 0;
                for (; j + 8 <= nci; j += 8) svst1_f64(pg, d + j, svld1_f64(pg, s + j));
                if (j < nci) {
                    svbool_t pj = svwhilelt_b64((long)j, (long)nci);
                    svst1_f64(pj, d + j, svld1_f64(pj, s + j));
                    j += 8;
                }
                for (; j < nc; j += 8) svst1_f64(pg, d + j, svdup_f64(0.0));
            } else {
                for (long j = 0; j < nc; j += 8) svst1_f64(pg, d + j, svdup_f64(0.0));
            }
        }
        for (int sg = 0; sg < tk->nseg; sg++) {
            const double* P = c->part + tk->part_off + (size_t)sg * mb * ncp;
            for (long q = 0; q < 16L * S * ncp; q += 8)
                svst1_f64(pg, cb + q, svadd_f64_x(pg, svld1_f64(pg, cb + q), svld1_f64(pg, P + q)));
        }
        if (c->prof >= 2) tb->t_load += cyc() - t0;
    }

    // 3. 对角求解：X[s] = Linv_ss C[s] - Σ (Linv_ss L_ss') X[s']，A = 对角打包条带，B = C 块行 0..16(s+1)
    if (c->nobar) {   // 本行块的对角条是否都已预处理好（无 barrier 模式）
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        while (__atomic_load_n(&c->dcnt[i], __ATOMIC_ACQUIRE) < S) asm volatile("yield");
        if (c->prof >= 2) tb->t_wait += cyc() - t0;
    }
    if (1) {   // 此时预占下一任务：提前量只有对角阶段 + 回写，不会明显加深依赖流水
        tnext = __atomic_fetch_add((long*)&c->next, 1, __ATOMIC_RELAXED);
        tkn = tnext < c->ntask_items ? &c->tasks[tnext] : NULL;
    }
    {
        const double* D = c->diag + c->diag_off[i];
        // 下一任务的 B 子块（行距 ldb 的 1 KB 行段）在对角阶段用游标 1 预取；对角条带（连续）用游标 2
        long nb_rows = 0, nb_done = 0, nb_lines = 0;
        const double* nb_base = NULL;
        long nb_ldb8 = ldb * 8;
        if (pf_on && tkn && tkn->kind == 0 && tkn->nseg == 1) {
            long ni0 = (long)tkn->i * mb, nc0 = pan_c0(c, tkn->p);
            nb_rows = c->m - ni0; if (nb_rows > mb) nb_rows = mb;
            long nnci = pan_nci(c, tkn->p); if (nnci > nc) nnci = nc;
            nb_lines = (nnci * 8 + 63) / 64;
            nb_base = c->B + ni0 * ldb + nc0;
        }
        for (int s = 0; s < S; s++) {
            const double* A = D + diag_strip_off(s);
            long K = 16L * (s + 1);
            for (long t = 0; t < ntile; t++) {
                pf_t pf = pf_dummy;
                if (pf_on && s + 1 < S) pf.p2 = (const char*)(D + diag_strip_off(s + 1)) + t * K * 32;
                if (nb_base && nb_done < nb_rows) {
                    long rows_this = (K / 2) / nb_lines; if (rows_this < 1) rows_this = 1;
                    pf.p = (const char*)(nb_base + nb_done * ldb);
                    pf.pfn = nb_lines;
                    pf.pfj = nb_ldb8 - nb_lines * 64;
                    nb_done += rows_this;
                }
                long cols = nci - 32 * t; if (cols > 32) cols = 32;
                if (pf_on) { pf.pc = (const char*)(cb + 16L * s * ncp + 32 * t); pf.pcs = cs; }
                sw_advance(tb);
                sme_zero_za();
                if (pend && K >= 65) {
                    PROF_T(c, kloop_epi_set(A, 128, 128, cb + 32 * t, cs, K, cols, &pf, scratch, pend_c, cs, pend_cols), t_dkloop);
                } else if (pend && K == 64) {
                    PROF_T(c, kloop_epi64_set(A, 128, 128, cb + 32 * t, cs, cols, &pf, scratch, pend_c, cs, pend_cols), t_dkloop);
                } else {
                    if (pend) PROF_T(c, epi_set_std(scratch, pend_c, cs, pend_cols), t_depi);
                    PROF_T(c, kloop_plain(A, 128, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);
                }
                PROF_T(c, za_dump(scratch), t_ddump);
                pend = 1; pend_c = cb + 16L * s * ncp + 32 * t; pend_cols = cols;
                if (c->prof >= 2) { tb->n_kstep += K; tb->n_dkstep += K; }
            }
            // 条末必须落地：下一条把这些行当 B 操作数
            if (pend) { PROF_T(c, epi_set_std(scratch, pend_c, cs, pend_cols), t_depi); pend = 0; }
        }
    }

    // 4. C 块 → Xp（同步，消费者要读）→ X（可托管到后续 K 循环），发布 flag
    {
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        const svbool_t pg = svptrue_b64();
        // 上一任务遗留的托管回写先同步收尾（通常已在 K 循环里完成）
        sw_flush(tb);
        // 面板主序副本：行 i0+r → 面板 ((i0+r)/kc, p) 的第 (i0+r)%kc 行（整行 nc 列，含零填充）；最后一个行块没有消费者，不写
        double* Bout=Bblk;

        const int last_rb = (i + 1 == c->nrb);
        if (!last_rb && (mode==2 || mode==4) && c->xptm) {
            // 列块主序写 Xp（K×16 微面板布局）：每个 16 列微面板内按行连续写 128 B，8 条顺序写流代替每行 8 处散写
            for (long ch0 = 0; ch0 < mbi; ch0 += kc) {
                long rows = mbi - ch0; if (rows > kc) rows = kc;
                double* pan = c->xp + ((size_t)((i0 + ch0) / kc) * c->npan + p) * kc * ncp;
                for (long j = 0; j < nc; j += 16) {
                    double* dj = pan + (j / 16) * kc * 16;
                    const double* sj = cb + ch0 * ncp + j;
                    for (long r = 0; r < rows; r++) {
                        svst1_f64(pg, dj + r * 16, svld1_f64(pg, sj + r * ncp));
                        svst1_f64(pg, dj + r * 16 + 8, svld1_f64(pg, sj + r * ncp + 8));
                    }
                }
            }
        } else if (!last_rb) for (long r = 0; r < mbi; r++) {
            long gr = i0 + r;
            double* d = c->xp + ((size_t)(gr / kc) * c->npan + p) * kc * ncp + (gr % kc) * ncp;
            const double* s = cb + r * ncp;
            if (c->xpnt) for (long j = 0; j < nc; j += 8) {
                double* dj=((mode==2 || mode==4))?d-(gr%kc)*ncp+(j/16)*kc*16+(gr%kc)*16+j%16:d+j;
                svstnt1_f64(pg,dj,svld1_f64(pg,s+j));
            }
            else for (long j = 0; j < nc; j += 8) {
                // Duplex modes pack Kx16 X tiles for the shared FP64 kernels.
                double* dj=((mode==2 || mode==4))?d-(gr%kc)*ncp+(j/16)*kc*16+(gr%kc)*16+j%16:d+j;
                svst1_f64(pg,dj,svld1_f64(pg,s+j));
            }
        }
        // 专用路径的消费者只读 Xp（列块主序副本），不读 X：Xp 写完就发布 flag，X 的回写放到 flag 之后——
        // 缩短行块间依赖链（case3 每环 ≈ 最后一段 GEMM + 归约 + 对角 + 回写，89 环串行），下游任务少等一次 X 回写的时间
        if (c->eflag) __atomic_store_n(&c->flags[i * c->npan + p], 1, __ATOMIC_RELEASE);
        // 托管条件：整面板（nci==nc，Xp 行距 = 每行 64 B 段数×64）且有 Xp 副本可作源。X 行可以不对齐（bench 的 malloc 只有 16 B 对齐），
        // stnt1d 支持非对齐，只是跨行的 64 B 段会拆成两次部分行写。
        // The specialized Xp layout is tiled; copy final output from C below.
        const int can_sw = 0;
        if (can_sw) {
            for (long r0 = 0; r0 < mbi; r0 += kc) {
                long rows = mbi - r0; if (rows > kc) rows = kc;
                sw_t w;
                w.src = (const char*)(c->xp + ((size_t)((i0 + r0) / kc) * c->npan + p) * kc * ncp);
                w.dst = (char*)(Bout + r0 * ldb);
                w.lpr = nc * 8 / 64; w.cnt = w.lpr;
                w.dstj = ldb * 8 - nc * 8;
                w.rows = rows;
                tb->swq[tb->swn++] = w;
                tb->n_swq += rows * w.lpr;
            }
        } else if (1) {
            // 非临时存储按 X 行的 cache 行对齐：头部分行（谓词）+ 整行 + 尾部分行。bench 的 B 只 16 B 对齐，
            // 若按 64 B 向量直接存，每个向量都跨两行 → 32 次部分行写/行；对齐后是 15 次整行 + 2 次部分行。
            for (long r = 0; r < mbi; r++) {
                const double* s = cb + r * ncp;
                double* d = Bout + r * ldb;
                long pre = (long)(((64 - ((uintptr_t)d & 63)) & 63) >> 3);   // 到下一个 64 B 边界的 double 数（0..7）
                if (pre > nci) pre = nci;
                long j = 0;
                if (pre) { svbool_t ph = svwhilelt_b64(0L, pre); svstnt1_f64(ph, d, svld1_f64(ph, s)); j = pre; }
                for (; j + 8 <= nci; j += 8) svstnt1_f64(pg, d + j, svld1_f64(pg, s + j));
                if (j < nci) {
                    svbool_t pj = svwhilelt_b64((long)j, (long)nci);
                    svstnt1_f64(pj, d + j, svld1_f64(pj, s + j));
                }
            }
        } else {
            for (long r = 0; r < mbi; r++) {
                const double* s = cb + r * ncp;
                double* d = Bout + r * ldb;
                long j = 0;
                for (; j + 8 <= nci; j += 8) svst1_f64(pg, d + j, svld1_f64(pg, s + j));
                if (j < nci) {
                    svbool_t pj = svwhilelt_b64((long)j, (long)nci);
                    svst1_f64(pj, d + j, svld1_f64(pj, s + j));
                }
            }
        }
        if (c->prof >= 2) tb->t_store += cyc() - t0;
    }
    if (!c->eflag) __atomic_store_n(&c->flags[i * c->npan + p], 1, __ATOMIC_RELEASE);
    if (c->prof >= 2) { tb->t_task += cyc() - tt0; tb->n_task++; }
    (void)tid;
    return tnext;
}

static long run_task_special1(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid) {return run_task_special_impl(c,tb,tk,guess,tid,1,0);}
static long run_task_special2(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid) {return run_task_special_impl(c,tb,tk,guess,tid,2,0);}
static long run_task_special4(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid) {return run_task_special_impl(c,tb,tk,guess,tid,4,0);}
static long run_task_hoist2(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid) {return run_task_special_impl(c,tb,tk,guess,tid,2,1);}
static long run_task_hoist4(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid) {return run_task_special_impl(c,tb,tk,guess,tid,4,1);}
// KC64 owner kernels: prefetch bases remain fixed and PRFM uses immediate
// offsets. A/B loads, FP64 FMOPA order and hosted C stores match the old kernels.
// This removes cursor ADD instructions without changing the addresses prefetched.
static inline __attribute__((always_inline)) void kloop_owner_cursor0(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 6144;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za0.d, za1.d, za2.d, za3.d}\n\t"
 "mov w12, #0\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "mov w12, #0\n\t"
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #0]\n\t"
 "prfm pldl2keep, [%[bp], #0]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #0]\n\t"
 "prfm pldl2keep, [%[pc], #64]\n\t"
 "prfm pldl2keep, [%[pc], #6144]\n\t"
 "prfm pldl2keep, [%[pc], #6208]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #64]\n\t"
 "prfm pldl2keep, [%[bp], #64]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #1\n\t"
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #128]\n\t"
 "prfm pldl2keep, [%[bp], #128]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #768]\n\t"
 "prfm pldl2keep, [%[pc], #832]\n\t"
 "prfm pldl2keep, [%[pc], #6912]\n\t"
 "prfm pldl2keep, [%[pc], #6976]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #192]\n\t"
 "prfm pldl2keep, [%[bp], #192]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #2\n\t"
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #256]\n\t"
 "prfm pldl2keep, [%[bp], #256]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #1536]\n\t"
 "prfm pldl2keep, [%[pc], #1600]\n\t"
 "prfm pldl2keep, [%[pc], #7680]\n\t"
 "prfm pldl2keep, [%[pc], #7744]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #320]\n\t"
 "prfm pldl2keep, [%[bp], #320]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #3\n\t"
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #384]\n\t"
 "prfm pldl2keep, [%[bp], #384]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #2304]\n\t"
 "prfm pldl2keep, [%[pc], #2368]\n\t"
 "prfm pldl2keep, [%[pc], #8448]\n\t"
 "prfm pldl2keep, [%[pc], #8512]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #448]\n\t"
 "prfm pldl2keep, [%[bp], #448]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #4\n\t"
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #512]\n\t"
 "prfm pldl2keep, [%[bp], #512]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #3072]\n\t"
 "prfm pldl2keep, [%[pc], #3136]\n\t"
 "prfm pldl2keep, [%[pc], #9216]\n\t"
 "prfm pldl2keep, [%[pc], #9280]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #576]\n\t"
 "prfm pldl2keep, [%[bp], #576]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #5\n\t"
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #640]\n\t"
 "prfm pldl2keep, [%[bp], #640]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #3840]\n\t"
 "prfm pldl2keep, [%[pc], #3904]\n\t"
 "prfm pldl2keep, [%[pc], #9984]\n\t"
 "prfm pldl2keep, [%[pc], #10048]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #704]\n\t"
 "prfm pldl2keep, [%[bp], #704]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #6\n\t"
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #768]\n\t"
 "prfm pldl2keep, [%[bp], #768]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #4608]\n\t"
 "prfm pldl2keep, [%[pc], #4672]\n\t"
 "prfm pldl2keep, [%[pc], #10752]\n\t"
 "prfm pldl2keep, [%[pc], #10816]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #832]\n\t"
 "prfm pldl2keep, [%[bp], #832]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #7\n\t"
 "mova z16.d, p0/m, za4h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za5h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za6h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za7h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #896]\n\t"
 "prfm pldl2keep, [%[bp], #896]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #5376]\n\t"
 "prfm pldl2keep, [%[pc], #5440]\n\t"
 "prfm pldl2keep, [%[pc], #11520]\n\t"
 "prfm pldl2keep, [%[pc], #11584]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #960]\n\t"
 "prfm pldl2keep, [%[bp], #960]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za2.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za3.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fmopa za0.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "fmopa za1.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "fmopa za2.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "fmopa za3.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "\n\t"
: [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc):: "memory","cc","x12",VCLOB,PCLOB);}
static inline __attribute__((always_inline)) void kloop_owner_cursor4(const double* a, const double* b, double* c,
                                      const char* ap, const char* bp, const char* pc)
{
    char* ct = (char*)c;
    char* cb = ct + 6144;
    asm volatile(
 "ptrue p0.d\n\t"
 "zero {za4.d, za5.d, za6.d, za7.d}\n\t"
 "mov w12, #0\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "mov w12, #0\n\t"
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #0]\n\t"
 "prfm pldl2keep, [%[bp], #0]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #0]\n\t"
 "prfm pldl2keep, [%[pc], #64]\n\t"
 "prfm pldl2keep, [%[pc], #6144]\n\t"
 "prfm pldl2keep, [%[pc], #6208]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #64]\n\t"
 "prfm pldl2keep, [%[bp], #64]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #1\n\t"
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #128]\n\t"
 "prfm pldl2keep, [%[bp], #128]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #768]\n\t"
 "prfm pldl2keep, [%[pc], #832]\n\t"
 "prfm pldl2keep, [%[pc], #6912]\n\t"
 "prfm pldl2keep, [%[pc], #6976]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #192]\n\t"
 "prfm pldl2keep, [%[bp], #192]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #2\n\t"
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #256]\n\t"
 "prfm pldl2keep, [%[bp], #256]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #1536]\n\t"
 "prfm pldl2keep, [%[pc], #1600]\n\t"
 "prfm pldl2keep, [%[pc], #7680]\n\t"
 "prfm pldl2keep, [%[pc], #7744]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #320]\n\t"
 "prfm pldl2keep, [%[bp], #320]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #3\n\t"
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #384]\n\t"
 "prfm pldl2keep, [%[bp], #384]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #2304]\n\t"
 "prfm pldl2keep, [%[pc], #2368]\n\t"
 "prfm pldl2keep, [%[pc], #8448]\n\t"
 "prfm pldl2keep, [%[pc], #8512]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #448]\n\t"
 "prfm pldl2keep, [%[bp], #448]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #4\n\t"
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #512]\n\t"
 "prfm pldl2keep, [%[bp], #512]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #3072]\n\t"
 "prfm pldl2keep, [%[pc], #3136]\n\t"
 "prfm pldl2keep, [%[pc], #9216]\n\t"
 "prfm pldl2keep, [%[pc], #9280]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #576]\n\t"
 "prfm pldl2keep, [%[bp], #576]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #5\n\t"
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #640]\n\t"
 "prfm pldl2keep, [%[bp], #640]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #3840]\n\t"
 "prfm pldl2keep, [%[pc], #3904]\n\t"
 "prfm pldl2keep, [%[pc], #9984]\n\t"
 "prfm pldl2keep, [%[pc], #10048]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #704]\n\t"
 "prfm pldl2keep, [%[bp], #704]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #6\n\t"
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #768]\n\t"
 "prfm pldl2keep, [%[bp], #768]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #4608]\n\t"
 "prfm pldl2keep, [%[pc], #4672]\n\t"
 "prfm pldl2keep, [%[pc], #10752]\n\t"
 "prfm pldl2keep, [%[pc], #10816]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #832]\n\t"
 "prfm pldl2keep, [%[bp], #832]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "mov w12, #7\n\t"
 "mova z16.d, p0/m, za0h.d[w12, 0]\n\t"
 "ld1d z17.d, p0/z, [%[ct], #0, mul vl]\n\t"
 "mova z18.d, p0/m, za1h.d[w12, 0]\n\t"
 "ld1d z19.d, p0/z, [%[ct], #1, mul vl]\n\t"
 "mova z20.d, p0/m, za2h.d[w12, 0]\n\t"
 "ld1d z21.d, p0/z, [%[cb], #0, mul vl]\n\t"
 "mova z22.d, p0/m, za3h.d[w12, 0]\n\t"
 "ld1d z23.d, p0/z, [%[cb], #1, mul vl]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z17.d, z17.d, z16.d\n\t"
 "st1d z17.d, p0, [%[ct], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #896]\n\t"
 "prfm pldl2keep, [%[bp], #896]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z19.d, z19.d, z18.d\n\t"
 "st1d z19.d, p0, [%[ct], #1, mul vl]\n\t"
 "prfm pldl2keep, [%[pc], #5376]\n\t"
 "prfm pldl2keep, [%[pc], #5440]\n\t"
 "prfm pldl2keep, [%[pc], #11520]\n\t"
 "prfm pldl2keep, [%[pc], #11584]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "ldnt1d z0.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "ldnt1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "ld1d z4.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "ld1d z5.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fsub z21.d, z21.d, z20.d\n\t"
 "st1d z21.d, p0, [%[cb], #0, mul vl]\n\t"
 "prfm pldl2keep, [%[ap], #960]\n\t"
 "prfm pldl2keep, [%[bp], #960]\n\t"
 "ldnt1d z8.d, p0/z, [%[a]]\n\t"
 "fmopa za4.d, p0/m, p0/m, z0.d, z4.d\n\t"
 "ldnt1d z9.d, p0/z, [%[a], #1, mul vl]\n\t"
 "fmopa za5.d, p0/m, p0/m, z0.d, z5.d\n\t"
 "ld1d z12.d, p0/z, [%[b]]\n\t"
 "fmopa za6.d, p0/m, p0/m, z1.d, z4.d\n\t"
 "ld1d z13.d, p0/z, [%[b], #1, mul vl]\n\t"
 "fmopa za7.d, p0/m, p0/m, z1.d, z5.d\n\t"
 "add %[a], %[a], #128\n\t"
 "add %[b], %[b], #768\n\t"
 "fmopa za4.d, p0/m, p0/m, z8.d, z12.d\n\t"
 "fmopa za5.d, p0/m, p0/m, z8.d, z13.d\n\t"
 "fmopa za6.d, p0/m, p0/m, z9.d, z12.d\n\t"
 "fmopa za7.d, p0/m, p0/m, z9.d, z13.d\n\t"
 "fsub z23.d, z23.d, z22.d\n\t"
 "st1d z23.d, p0, [%[cb], #1, mul vl]\n\t"
 "add %[ct], %[ct], #768\n\t"
 "add %[cb], %[cb], #768\n\t"
 "\n\t"
: [a] "+&r"(a),[b] "+&r"(b),[ct] "+&r"(ct),[cb] "+&r"(cb),[ap] "+&r"(ap),[bp] "+&r"(bp),[pc] "+&r"(pc):: "memory","cc","x12",VCLOB,PCLOB);}

// ---------------------------------------------------------------- 主流程
static void trsm_sme(long m, long n, const double* L, long lda, double* B, long ldb)
{
    ctx_t c;
    memset(&c, 0, sizeof(c));
    c.m = m; c.n = n; c.lda = lda; c.ldb = ldb; c.L = L; c.B = B;
    c.prof = env_int("TRSM_PROFILE", 0);
    g_owner_cursor=env_int("TRSM_OWNER_CURSOR",1); // Fixed KC64 prefetch offsets; 0 restores v48 owner kernels.
    g_abhoist = env_int("TRSM_ABHOIST", 1); // 0 retains the original duplex kernels for comparisons
    c.pf = env_int("TRSM_PF", 1);
    c.packmode = env_int("TRSM_PACK", 1);
    c.ntstore = env_int("TRSM_NT", 1);
    c.swhost = env_int("TRSM_SWHOST", -1);   // -1：面板数 ≤ 8 时启用（case3 形状略有收益，宽 X 时净收益为负）
    c.ant = env_int("TRSM_ANT", m <= 1024 && !env_int("TRSM_DIRECT", 1) ? 0 : 1);   // 直接 ZA 累加用非临时 A 装载；关闭直接累加的小矩阵仍保留 A 于 L2
    c.reserve = env_int("TRSM_RESERVE", 1);
    c.xpnt = env_int("TRSM_XPNT", 0);
    // 小 m、宽 X（面板数 ≥ 2×线程数）→ 面板驻留模式（TRSM_OWNER=-1 自动）：默认块 mb64/nc96（面板缓冲 512×96×8 = 393 KB 常驻 L2），见 NOTES §9
    int nth_hint = omp_get_max_threads(); if (nth_hint < 1) nth_hint = 1;
    int owner_auto = env_int("TRSM_OWNER", -1);
    if (owner_auto < 0) owner_auto = (m <= 1024 && n >= 2L * nth_hint * 96);
    // 默认 mb192/nc128/kc64：L2 足迹 cbuf 192K + 2×(A 段 96K + B 段 64K) = 512 KB；kc=64 时回写由 kloop_epi64 全程托管
    int mb = env_int("TRSM_MB", owner_auto ? 64 : (m > 1024 && n >= 8192 ? 256 : 192)), nc = env_int("TRSM_NC", owner_auto ? 96 : 128), kc = env_int("TRSM_KC", 64);
    if (mb < 32) mb = 32;
    mb = (mb + 31) & ~31;
    if (nc < 32) nc = 32;
    nc = (nc + 31) & ~31;
    if (kc < 32) kc = 32;
    kc = (kc + 31) & ~31;               // ld4d/ld2d 打包按 32 个 k 一组
    if (kc > mb) kc = mb;
    while (mb % kc) kc -= 32;           // 要求 kc | mb，使每个 K 段落在单个行块内
    c.mb = mb; c.nc = nc; c.kc = kc;
    c.lchunk = env_int("TRSM_LCHUNK", 1) && mb % kc == 0;
    c.direct = env_int("TRSM_DIRECT", m <= 1024);
    c.nrb = (m + mb - 1) / mb;
    // 面板边界平移：bench 的 B 只 16 B 对齐（大块 malloc 恒为行首 +16 B），行距 ldb*8 为 64 B 整数倍时全矩阵行段偏移恒定。
    // 内部面板起点平移 cshift 列使 (B + c0*8) 落在 cache 行边界：装载无跨行向量（每条 svld1 从 2 次 L1 访问降到 1 次）、
    // 非临时回写无 partial-line RMW。首面板缩为 nc−cshift 列、末面板 cshift 列，由派发处路由到通用路径。
    // 门控 npan≥64：多出的尾面板要整读一遍 A 流（成本 ≈ A 流量/npan），窄 X（如 17024×512，npan=4）A 占总流量 ~78%，
    // 实测 +19% 灾难回退；npan≥64 时该成本 ≤1.6% A 流量，远小于对齐收益。TRSM_CSHIFT：0 关，1 自动（默认），2 强开。
    c.cshift = 0;
    {
        int cs_mode = env_int("TRSM_CSHIFT", 1);
        long npan0 = (n + nc - 1) / nc;
        if (cs_mode && (((size_t)ldb * 8) & 63) == 0 && (cs_mode >= 2 || npan0 >= 64))
            c.cshift = (long)(((uintptr_t)B >> 3) & 7);
    }
    c.npan = c.cshift ? (n > nc - c.cshift ? 1 + (n - (nc - c.cshift) + nc - 1) / nc : 1) : (n + nc - 1) / nc;
    c.ncp = nc + TRSM_NCPAD;
    if (c.swhost < 0) c.swhost = c.npan <= 8;
    c.ntask = c.nrb * c.npan;
    c.nstrip = (m + 15) / 16;

    uint64_t t0 = c.prof ? cyc() : 0;
    uint64_t tsub[6] = {0, 0, 0, 0, 0, 0};   // prof>=3：setup 子阶段（pin/diag/缓冲/flags/任务表/ditem）
#define TSUB(k) do { if (c.prof >= 3) tsub[k] = cyc(); } while (0)
    int nth = omp_get_max_threads();
    if (nth < 1) nth = 1;
    if (nth > 1024) nth = 1024;
    if (g_ncpu <= 0 || g_pin_nth != nth) {
#pragma omp parallel num_threads(nth)
        pin_threads_in_region(omp_get_thread_num(), omp_get_num_threads());
        g_pin_nth = nth;
    }
    if (g_ncpu > 0 && g_ncpu < nth) nth = g_ncpu;
    detect_nodes();
    TSUB(0);
    // 决定 GEMM 部分的 L 打包方式：L 下三角 ≤ 阈值时预打包（两份），否则 JIT
    size_t tri_bytes = (size_t)m * (m + 1) / 2 * 8;
    long jit_mb = env_int("TRSM_JIT_MB", 256);
    int bigL = (tri_bytes > (size_t)jit_mb << 20) ? 1 : 0;
    // 大 L：默认流水式打包任务（TRSM_PPMODE=1）；TRSM_JIT=1 强制就地转置
    c.ppmode = bigL ? env_int("TRSM_PPMODE", 1) : 0;
    c.jit = (bigL && c.ppmode == 0) ? 1 : 0;
    if (env_int("TRSM_JIT", -1) >= 0) { c.jit = env_int("TRSM_JIT", -1); if (c.jit) c.ppmode = 0; }
    // 面板驻留模式：只在预打包路径下可用，整列面板 (nrb*mb × ncp) 要能放进线程缓冲（默认上限 4 MB）
    {
        int owner = owner_auto;
        size_t panel_bytes = (size_t)c.nrb * mb * c.ncp * 8;
        if (c.jit || c.ppmode != 0) owner = 0;
        if (panel_bytes > ((size_t)env_int("TRSM_OWNER_MB", 4) << 20)) owner = 0;
        c.owner = owner;
        if (c.owner) {
            c.swhost = 0;
            if (!getenv("TRSM_ANT")) c.ant = 1;          // 面板缓冲要常驻 L2：A 条用 ldnt1d 流过
            if (!getenv("TRSM_DIRECT")) c.direct = 0;    // mb=64 的段只有 64 步，直接 ZA 装载/存储每微块两次传输比托管回写贵（实测 2.97 vs 2.41 ms）
        }
        c.dnt = env_int("TRSM_DNT", c.owner);
        c.eflag = env_int("TRSM_EFLAG", 1);
        c.nexta = env_int("TRSM_NEXTA", 1);
        c.pf2 = env_int("TRSM_OWNER_PF2", 1);
    }
    {
        c.loadpf = env_int("TRSM_LOADPF", 16);   // 官方 bench 交错两轮：case2 25.07/24.82 → 24.20/24.48，case1 2.32/2.37 → 2.28/2.29
        c.xptm = env_int("TRSM_XPTM", 0);
    }
    // Admit a specialization only when every baked-in setting matches this call.
    c.special=0;
    if(!c.owner && env_int("TRSM_SPECIAL",1) && n%128==0 && nc==128 && kc==64 && c.ncp==128 && !c.jit && c.lchunk && c.pf && c.reserve && c.ntstore && c.ant) {   // xpnt 在专用实现里已按开关处理
      if(mb==192 && c.direct && !c.ppmode && !c.swhost)c.special=1;
      if(mb==256 && !c.direct && !c.ppmode && !c.swhost)c.special=2;
      if(mb==192 && !c.direct && c.ppmode==1 && c.npan<=8 &&
         (!c.swhost || env_int("TRSM_SWHOST",-1)<0)) {
        c.special=4;
        c.swhost=0; // Kx16 Xp is copied to B directly after the diagonal solve.
      }
    }

    // Paired row strips require full 64-row tasks and full 96-column panels.
    if (c.owner && m % 64 == 0 && n % 96 == 0 && env_int("TRSM_SPECIAL", 1) && mb == 64 && nc == 96 && kc == 64 && c.ncp == 96
        && !c.jit && c.ppmode == 0 && c.lchunk && !c.direct && c.ant && c.pf && c.reserve
        && !c.xpnt && c.ntstore && !c.swhost && c.dnt && c.packmode == 1) c.special = 5;
    // Tune B prefetch separately for the paired-row owner loop.
    if (c.special == 5) c.loadpf = env_int("TRSM_LOADPF", 0);

    g_align_owner_buffers = c.owner || env_int("TRSM_ALIGN_ALL", 0);   // 2 MiB 对齐所有缓冲：专用路径实测 case2 持平、case3 −0.1 ms 但出现 44 ms 异常轮，默认关

    // HBM 容量规划：自己的缓冲（Xp、打包 L）尽量进 HBM，空间不够就退本地 DDR（算子不移动调用方内存）。
    // TRSM_HBM_MARGIN_MB：保留余量（默认 256）
    const size_t hbm_margin = (size_t)env_int("TRSM_HBM_MARGIN_MB", 256) << 20;
#define HBM_ROOM(bytes) (g_hbm_node >= 0 && node_free_bytes(g_hbm_node) > (bytes) + hbm_margin)

    if (env_int("TRSM_SMAPS", 0)) {   // 诊断：X / L / 打包缓冲所在 VMA 的透明大页覆盖率（每次调用打印）
        const void* tgt[4] = { B, L, g_pp[0], (const char*)B + (4 << 20) };
        const char* nm[4] = { "X", "L", "pp0", "X+4M" };
        FILE* f = fopen("/proc/self/smaps", "r");
        if (f) {
            char line[256]; uintptr_t lo = 0, hi = 0; long huge = -1;
            int in[4] = {0,0,0,0};
            while (fgets(line, sizeof line, f)) {
                uintptr_t a, b2;
                if (sscanf(line, "%lx-%lx", &a, &b2) == 2) {
                    lo = a; hi = b2;
                    for (int k = 0; k < 4; k++) in[k] = tgt[k] && (uintptr_t)tgt[k] >= a && (uintptr_t)tgt[k] < b2;
                } else if (sscanf(line, "AnonHugePages: %ld", &huge) == 1) {
                    for (int k = 0; k < 4; k++)
                        if (in[k]) fprintf(stderr, "[trsm smaps] %s vma %ld MB, anonhuge %ld MB (%.0f%%)\n",
                                           nm[k], (long)((hi - lo) >> 20), huge >> 10, (hi - lo) ? 100.0 * (huge << 10) / (double)(hi - lo) : 0);
                }
            }
            fclose(f);
        }
    }

    // 对角打包缓冲
    c.S_of = malloc(sizeof(int) * c.nrb);
    c.diag_off = malloc(sizeof(size_t) * (c.nrb + 1));
    size_t doff = 0;
    for (long i = 0; i < c.nrb; i++) {
        long mbi = m - i * mb; if (mbi > mb) mbi = mb;
        c.S_of[i] = (int)((mbi + 15) / 16);
        c.diag_off[i] = doff;
        doff += diag_block_doubles(c.S_of[i]);
    }
    c.diag_off[c.nrb] = doff;
    if (g_diag_cap < doff) {
        trsm_free_buffer(g_diag, g_diag_cap * 8);
        g_diag = alloc_on(doff * 8, -1);
        g_diag_cap = doff;
    }
    c.diag = g_diag;
    TSUB(1);

    // X 面板主序副本（本实现自己的缓冲，可自由选内存节点）：TRSM_XPNODE：-1 默认策略，-2 本地 DDR，≥0 指定；面板驻留模式不需要
    if (!c.owner) {
        c.xp_chunks = (m + kc - 1) / kc;
        size_t need = (size_t)c.xp_chunks * c.npan * kc * c.ncp;
        int xn = env_int("TRSM_XPNODE", -9);
        int node;
        if (xn == -9) {   // 自动：B 操作数流对延迟敏感，尽量进 HBM（判题命令不带 -m，"默认策略"意味着 DDR）
            if (g_xp_cap >= need && g_xp_node == g_hbm_node && g_hbm_node >= 0) node = g_hbm_node;   // 已在 HBM，沿用
            else node = HBM_ROOM(need * 8) ? g_hbm_node : (g_cpu_node >= 0 ? g_cpu_node : -1);
        }
        else if (xn == -2) node = g_cpu_node;
        else node = xn;
        if (g_xp_cap < need || g_xp_node != node) {
            trsm_free_buffer(g_xp, g_xp_cap * 8);
            g_xp = alloc_on(need * 8, node);
            g_xp_cap = need;
            g_xp_node = node;
        }
        c.xp = g_xp;
    }

    // 预打包缓冲：条 g 属于行块 i=(16g)/mb，打包 k∈[0, i*mb)
    c.pp[0] = c.pp[1] = NULL;
    c.pp_off = NULL;
    if (!c.jit) {
        c.pp_off = malloc(sizeof(size_t) * (c.nstrip + 1));
        size_t off = 0;
        for (long g = 0; g < c.nstrip; g++) {
            c.pp_off[g] = off;
            long i = (16 * g) / mb;
            off += (size_t)16 * (i * mb);
        }
        c.pp_off[c.nstrip] = off;

        // 副本节点：主副本放 HBM（有则），副副本放本地 DDR；TRSM_LSPLIT=0 只用主副本；流水式打包只有一份
        int split = c.ppmode == 1 ? 0 : env_int("TRSM_LSPLIT", c.owner ? 0 : 1);   // 面板驻留模式（case1）不做 DDR 第二份：实测 2.00→1.92 ms（一半任务从 DDR 读 A 更慢 + 省 memcpy）；case2 持平
        if (c.ppmode == 1 && g_pp[1]) { trsm_free_buffer(g_pp[1], g_pp_cap[1] * 8); g_pp[1] = NULL; g_pp_cap[1] = 0; g_pp_node[1] = -3; }
        int node0 = g_hbm_node >= 0 ? g_hbm_node : -1;
        int node1 = g_hbm_node >= 0 ? g_cpu_node : -1;
        if (node0 >= 0 && !(g_pp_cap[0] >= off && g_pp_node[0] == node0)) {   // 主副本进 HBM 需有位置，不够就腾 L / 退 DDR
            if (!HBM_ROOM(off * 8 + 4096)) node0 = g_cpu_node >= 0 ? g_cpu_node : -1;
        }
        if (env_int("TRSM_LNODE0", -3) != -3) node0 = env_int("TRSM_LNODE0", -3);
        if (env_int("TRSM_LNODE1", -3) != -3) node1 = env_int("TRSM_LNODE1", -3);
        for (int k = 0; k < 2; k++) {
            if (k == 1 && !split) break;
            int node = k == 0 ? node0 : node1;
            if (g_pp_cap[k] < off || g_pp_node[k] != node) {
                trsm_free_buffer(g_pp[k], g_pp_cap[k] * 8);
                g_pp[k] = alloc_on(off * 8 + 4096, node);
                g_pp_cap[k] = off;
                g_pp_node[k] = node;
            }
            c.pp[k] = g_pp[k];
        }
    }

    TSUB(2);
    size_t nflags = (size_t)c.ntask;
    if (g_flags_cap < nflags) {
        free(g_flags); free(g_seg_cnt);
        g_flags = calloc(nflags, sizeof(int));
        g_seg_cnt = calloc(nflags, sizeof(int));
        g_flags_cap = nflags;
    }
    memset(g_flags, 0, nflags * sizeof(int));
    memset(g_seg_cnt, 0, nflags * sizeof(int));
    c.flags = g_flags;
    c.seg_cnt = g_seg_cnt;
    c.next = 0;
    if (g_pflags_cap < (size_t)c.nstrip) { free(g_pflags); g_pflags = calloc(c.nstrip, sizeof(int)); g_pflags_cap = c.nstrip; }
    memset(g_pflags, 0, c.nstrip * sizeof(int));
    c.pflags = g_pflags;
    if (g_dcnt_cap < (size_t)c.nrb) { free(g_dcnt); g_dcnt = calloc(c.nrb, sizeof(int)); g_dcnt_cap = c.nrb; }
    memset(g_dcnt, 0, c.nrb * sizeof(int));
    c.dcnt = g_dcnt;
    TSUB(3);
    c.nobar = env_int("TRSM_NOBAR", 1);

    // 任务表：按 (i, p) 顺序；代价大的行块按 K 分裂成多段（段内为连续 K 段区间），
    // 段数 = 该任务 k 步数 / 目标段大小，目标 = 总 k 步 / (nth * TRSM_SPLITF)
    {
        long ntile_full = (nc + 31) / 32;
        double total = 0;
        double* cost = malloc(sizeof(double) * c.nrb);
        for (long i = 0; i < c.nrb; i++) {
            long S = c.S_of[i];
            cost[i] = (double)S * (i * mb) + 8.0 * S * (S + 1);   // 每列块的 k 步数 / ntile
            total += cost[i] * c.npan;
        }
        int splitf = c.owner ? 0 : env_int("TRSM_SPLITF", 12);   // 面板驻留模式不分裂 K
        int maxseg = env_int("TRSM_MAXSEG", 8);
        // Narrow RHS panels leave too few long tasks at the end of the solve.
        // Split only the last two row blocks more finely to shorten that tail.
        int tailrows = env_int("TRSM_TAIL_ROWS", 2);
        int tailseg = env_int("TRSM_TAIL_SEG", 6);
        // 宽 n（面板数 ≥ 线程数）时最后 tailrows_w 个行块也分裂成 tailseg_w 段：最后一个行块的任务最长（case2 约 1.7 ms），
        // 不分裂时尾部平均空闲约半个任务
        int tailrows_w = env_int("TRSM_TAILW_ROWS", 1);
        int tailseg_w = env_int("TRSM_TAILW_SEG", 1);   // 实测 case2 分裂最后行块：尾部空闲 0.36→0.26 ms 但归约与部分和流量多 0.6 ms，净负，默认关
        long minch = env_int("TRSM_SPLIT_MINCH", 8);   // 少于这么多 K 段的任务不分裂（小问题分裂只增加归约开销）
        double target = splitf > 0 ? total / ((double)nth * splitf) : 1e300;
        long nitems = 0;
        size_t part_doubles = 0;
        for (long i = 0; i < c.nrb; i++) {
            long nch = (i * mb) / kc;
            int nseg = 1;
            if (splitf > 0 && nch >= minch) {
                nseg = (int)(cost[i] / target + 0.5);
                if (c.npan < nth && i >= c.nrb - tailrows && nseg < tailseg) nseg = tailseg;
                if (c.npan >= nth && i >= c.nrb - tailrows_w && nseg < tailseg_w) nseg = tailseg_w;
                if (nseg > maxseg) nseg = maxseg;
                if (nseg > nch) nseg = (int)nch;
                if (nseg < 1) nseg = 1;
            }
            nitems += (long)nseg * c.npan;
            if (nseg > 1) part_doubles += (size_t)nseg * c.npan * mb * c.ncp;
        }
        if (c.ppmode == 1) nitems += c.nstrip;   // 打包任务
        if (g_tasks_cap < (size_t)nitems) { free(g_tasks); g_tasks = malloc(sizeof(task_t) * nitems); g_tasks_cap = nitems; }
        if (g_part_cap < part_doubles) { trsm_free_buffer(g_part, g_part_cap * 8); g_part = alloc_on(part_doubles * 8 + 4096, -1); g_part_cap = part_doubles; }
        long q = 0;
        size_t poff = 0;
        // 流水式打包：行块 i+LA 的条在行块 i 的计算任务之前派发；开头先派发 1..LA 块
        int LA = env_int("TRSM_PACK_LA", 2);
        long gnext = 0;   // 下一个待派发的打包条
        if (c.ppmode == 1) {
            long gend = (long)(LA + 1) * mb / 16; if (gend > c.nstrip) gend = c.nstrip;
            for (; gnext < gend; gnext++) { task_t* t = &g_tasks[q++]; memset(t, 0, sizeof *t); t->kind = 1; t->g = gnext; t->nseg = 1; }
        }
        if (c.owner) {   // 面板主序：面板 p 的行块 0..nrb-1 连续，q = p*nrb + i；线程区间按代价（GEMM k 步 + 对角 + 每任务固定装载/回写开销）均分
            const long chunks_per_block = mb / kc;
            for (long p = 0; p < c.npan; p++)
                for (long i = 0; i < c.nrb; i++) {
                    task_t* t = &g_tasks[q++];
                    memset(t, 0, sizeof *t);
                    t->i = (int)i; t->p = (int)p; t->nseg = 1;
                    t->ch0 = 0; t->ch1 = i * chunks_per_block;
                }
            double c0 = env_int("TRSM_OWNER_C0", c.special == 5 && c.pf2 ? 160 : 192) * (double)mb * nc;   // 固定开销折算成 MAC 数（读 B 写 X 各 mb*nc*8 B，≈12 MAC/B）
            double rowsum = 0;
            for (long i = 0; i < c.nrb; i++) {
                cost[i] = ((double)i * mb * mb + 0.5 * mb * mb) * nc + c0;
                rowsum += cost[i];
            }
            // cshift>0 时首/末奇面板更窄，按面板宽度加权（Σ 权重 = n/nc）
            double ototal = rowsum * ((double)n / nc);
            c.owner_bound = malloc(sizeof(long) * (nth + 1));
            double cum = 0;
            int tb_ = 1;
            c.owner_bound[0] = 0;
            // Each panel repeats the same row costs. Reuse those costs and
            // update the next boundary only when advancing to another thread.
            long row = 0, pan = 0;
            double pw = (double)pan_nci(&c, 0) / nc;
            double next_bound = ototal / nth;
            for (long qq = 0; qq < q && tb_ < nth; qq++) {
                cum += cost[row] * pw;
                if (++row == c.nrb) { row = 0; if (++pan < c.npan) pw = (double)pan_nci(&c, pan) / nc; }
                while (tb_ < nth && cum >= next_bound) {
                    c.owner_bound[tb_++] = qq + 1;
                    next_bound = ototal * tb_ / nth;
                }
            }
            while (tb_ <= nth) c.owner_bound[tb_++] = q;
        } else
        for (long i = 0; i < c.nrb; i++) {
            if (c.ppmode == 1) {
                long gend = (long)(i + LA + 1) * mb / 16; if (gend > c.nstrip) gend = c.nstrip;
                for (; gnext < gend; gnext++) { task_t* t = &g_tasks[q++]; memset(t, 0, sizeof *t); t->kind = 1; t->g = gnext; t->nseg = 1; }
            }
            long nch = (i * mb) / kc;
            int nseg = 1;
            if (splitf > 0 && nch >= minch) {
                nseg = (int)(cost[i] / target + 0.5);
                if (c.npan < nth && i >= c.nrb - tailrows && nseg < tailseg) nseg = tailseg;
                if (c.npan >= nth && i >= c.nrb - tailrows_w && nseg < tailseg_w) nseg = tailseg_w;
                if (nseg > maxseg) nseg = maxseg;
                if (nseg > nch) nseg = (int)nch;
                if (nseg < 1) nseg = 1;
            }
            // 分裂任务的派发顺序：TRSM_LASTLATE=1 时先派发各面板的非末段（不依赖前一行块），末段（含最后一个 K 段、要等前一行块 flag）放到最后，
            // 让末段更晚到达等待点、独立工作更早开始
            const int lastlate = nseg > 1 && env_int("TRSM_LASTLATE", 1);
            size_t poff0 = poff;
            for (int pass = 0; pass < (lastlate ? 2 : 1); pass++) {
                poff = poff0;
                for (long p = 0; p < c.npan; p++) {
                    for (int s = 0; s < nseg; s++) {
                        int is_last = s == nseg - 1;
                        if (lastlate && (pass == 0) == is_last) continue;   // pass 0：非末段；pass 1：末段
                        task_t* t = &g_tasks[q++];
                        t->kind = 0; t->g = 0;
                        t->i = (int)i; t->p = (int)p; t->seg = s; t->nseg = nseg;
                        t->ch0 = nch * s / nseg; t->ch1 = nch * (s + 1) / nseg;
                        t->part_off = poff;
                    }
                    if (nseg > 1) poff += (size_t)nseg * mb * c.ncp;
                }
            }
        }
        nitems = q;
        c.tasks = g_tasks;
        c.ntask_items = nitems;
        c.part = g_part;
        free(cost);
        (void)ntile_full;
        TSUB(4);
    }

    // 预处理工作项：对角子块 (i, s) 与预打包条 g。nobar 时合成一张按行块顺序排列的表（对角项 ditem_s>=0，打包条 ditem_s=-1、ditem_i=g），
    // 各线程动态领取，做完自己领到的就直接进任务阶段（消费者按 pflags/dcnt 等，通常早已就绪）——省掉全局 barrier 等最慢线程
    const int prep_pack = !c.jit && c.ppmode == 0;
    long ndiag_items = 0;
    for (long i = 0; i < c.nrb; i++) ndiag_items += c.S_of[i];
    long nprep_items = ndiag_items + ((c.nobar && prep_pack) ? c.nstrip : 0);
    long* ditem_i = malloc(sizeof(long) * nprep_items);
    int* ditem_s = malloc(sizeof(int) * nprep_items);
    {
        long q = 0;
        for (long i = 0; i < c.nrb; i++) {
            for (int s = 0; s < c.S_of[i]; s++) { ditem_i[q] = i; ditem_s[q] = s; q++; }
            if (c.nobar && prep_pack) {
                long g0 = i * mb / 16, g1 = (i + 1) * mb / 16; if (g1 > c.nstrip) g1 = c.nstrip;
                for (long g = g0; g < g1; g++) { ditem_i[q] = g; ditem_s[q] = -1; q++; }
            }
        }
    }
    uint64_t t1 = c.prof ? cyc() : 0;

#pragma omp parallel num_threads(nth)
    {
        int tid = omp_get_thread_num();
        if (c.prof >= 2) g_wake_cyc[tid] = cyc();
        pin_self(tid);
        ensure_tbuf(tid, (size_t)(c.owner ? c.nrb * mb : mb) * c.ncp, (size_t)128 * kc, (size_t)kc * c.ncp);
        if (c.prof >= 2) {
            tbuf_t* tb = &g_tb[tid];
            tb->t_load = tb->t_pack = tb->t_pf = tb->t_wait = tb->t_kloop = tb->t_epi = 0;
            tb->t_dkloop = tb->t_depi = tb->t_store = tb->t_task = 0;
            tb->n_task = tb->n_kstep = tb->n_dkstep = 0;
            tb->t_dump = tb->t_ddump = 0;
            tb->n_swflush = tb->n_swq = 0;
        }
        uint64_t tp0 = c.prof >= 2 ? cyc() : 0;
        // 阶段 1：对角预处理（+ nobar 时的预打包条，同一张表、按行块顺序动态领取）
#pragma omp for schedule(dynamic, 1) nowait
        for (long q = 0; q < nprep_items; q++) {
            if (ditem_s[q] >= 0) {
                long i = ditem_i[q];
                long mbi = m - i * mb; if (mbi > mb) mbi = mb;
                diag_prep_sub(L, lda, i * mb, (int)mbi, ditem_s[q], c.diag + c.diag_off[i] + diag_strip_off(ditem_s[q]));
                if (c.nobar) __atomic_fetch_add(&c.dcnt[i], 1, __ATOMIC_RELEASE);
            } else {
                long g = ditem_i[q];
                long i = (16 * g) / mb;
                long kk = i * mb;
                if (kk > 0) {
                    if (c.lchunk) pack_l_chunked(&c, g);
                    else {
                        pack_strip(L, lda, m, 16 * g, 0, kk, c.pp[0] + c.pp_off[g]);
                        if (c.pp[1]) memcpy(c.pp[1] + c.pp_off[g], c.pp[0] + c.pp_off[g], (size_t)kk * 16 * 8);
                    }
                }
                __atomic_store_n(&c.pflags[g], 1, __ATOMIC_RELEASE);
            }
        }
        // 阶段 1b：预打包（barrier 模式：单独一轮）
        if (prep_pack && !c.nobar) {
#pragma omp for schedule(dynamic, 1) nowait
            for (long g = 0; g < c.nstrip; g++) {
                long i = (16 * g) / mb;
                long kk = i * mb;
                if (kk <= 0) continue;
                if (c.lchunk) { pack_l_chunked(&c, g); continue; }
                pack_strip(L, lda, m, 16 * g, 0, kk, c.pp[0] + c.pp_off[g]);
                if (c.pp[1]) memcpy(c.pp[1] + c.pp_off[g], c.pp[0] + c.pp_off[g], (size_t)kk * 16 * 8);
            }
        }
        if (!c.nobar) {
#pragma omp barrier
        }
        if (c.prof >= 2) g_prep_cyc[tid] = cyc() - tp0;   // 预处理（+ barrier）
        // 阶段 2：任务
        // 面板驻留模式的线程调度表在进入流式模式前建好（流式区内 malloc 等系统调用会清 PSTATE.SM → 后续 SME 指令 SIGILL）：
        // 线程 tid 的连续区间 [a, b)（面板主序）。区间两端可能各含一个不完整面板：
        // 尾部不完整面板（行块 0..ib，被下一线程接着算）放到调度最前，头部不完整面板（行块 ia..，依赖上一线程算出的前几块）放到最后，
        // 这样跨线程依赖在到达时早已满足。完整面板按行块顺序做完，B 操作数来自本线程面板缓冲。
        long* sched = NULL;
        long k = 0, khead = -1;   // khead：头部不完整面板在 sched 中的起点（其前几个行块由上一线程算出）
        long pa = 0, ia = 0;
        if (c.owner) {
            long a = c.owner_bound[tid], b = c.owner_bound[tid + 1];
            long nrb = c.nrb;
            long cnt = b - a;
            sched = cnt > 0 ? malloc(sizeof(long) * cnt) : NULL;
            pa = a / nrb; ia = a % nrb;
            if (cnt > 0) {
                long pb = (b - 1) / nrb, ib = (b - 1) % nrb;
                if (pa == pb) {
                    if (ia != 0) khead = 0;
                    for (long qq = a; qq < b; qq++) sched[k++] = qq;
                } else {
                    long pfull0 = ia == 0 ? pa : pa + 1;          // 完整面板起点
                    long pfull1 = ib == nrb - 1 ? pb : pb - 1;    // 完整面板终点（含）
                    if (ib != nrb - 1) for (long i = 0; i <= ib; i++) sched[k++] = pb * nrb + i;
                    for (long p = pfull0; p <= pfull1; p++) for (long i = 0; i < nrb; i++) sched[k++] = p * nrb + i;
                    if (ia != 0) { khead = k; for (long i = ia; i < nrb; i++) sched[k++] = pa * nrb + i; }
                }
            }
        }
        sme_start();
        g_tb[tid].swn = g_tb[tid].swi = 0; g_tb[tid].sw.rows = 0;
        if (c.owner) {
            for (long s = 0; s < k; s++) {
                if (s == khead) {
                    // 头部不完整面板：行块 0..ia-1 的 X 由上一线程算出并已写回 X；等其 flag，把这些行拷进本线程面板缓冲作 B 操作数
                    long pc0 = pan_c0(&c, pa), pnci = pan_nci(&c, pa); if (pnci > c.nc) pnci = c.nc;
                    for (long i = 0; i < ia; i++) wait_flag(&c, i * c.npan + pa);
                    double* buf = g_tb[tid].cbuf;
                    const svbool_t pg = svptrue_b64();   // 流式模式下不能调 memcpy（NEON），用 SVE 拷贝，越界列补零
                    for (long gr = 0; gr < ia * mb; gr++) {
                        double* d = buf + (size_t)gr * c.ncp;
                        const double* sp = B + (size_t)gr * ldb + pc0;
                        long j = 0;
                        for (; j + 8 <= pnci; j += 8) svst1_f64(pg, d + j, svld1_f64(pg, sp + j));
                        if (j < pnci) { svbool_t pj = svwhilelt_b64(j, pnci); svst1_f64(pj, d + j, svld1_f64(pj, sp + j)); j += 8; }
                        for (; j < c.nc; j += 8) svst1_f64(pg, d + j, svdup_f64(0.0));
                    }
                }
                {   // cshift>0 时首/末奇面板（宽度 ≠ nc 或起点非对齐）路由到通用路径
                    const task_t* tsk = &c.tasks[sched[s]];
                    const int fullpan = !c.cshift || (tsk->p > 0 && tsk->p < c.npan - 1);
                    (c.special == 5 && fullpan ? (g_owner_cursor ? run_task_owner_cursor : run_task_owner) : run_task)(&c, &g_tb[tid], tsk, s + 1 < k ? &c.tasks[sched[s + 1]] : NULL, tid);
                }
            }
        } else {
        long (*taskfn)(const ctx_t*,tbuf_t*,const task_t*,const task_t*,int)=run_task;
        if(c.special==1)taskfn=run_task_special1;
        if(c.special==2)taskfn=g_abhoist ? run_task_hoist2 : run_task_special2;
        if(c.special==4)taskfn=g_abhoist ? run_task_hoist4 : run_task_special4;
        long t = __atomic_fetch_add(&c.next, 1, __ATOMIC_RELAXED);
        while (t < c.ntask_items) {
            // 下一任务：reserve=1 时在对角阶段开始处预占（精确预取其 C 块）；否则启发式猜测 t+nth，不提前占号
            long tg = t + nth;
            const task_t* tsk = &c.tasks[t];
            // cshift>0 时首/末奇面板（宽度 ≠ nc）路由到通用路径；打包任务（kind=1）两边都能处理，走 taskfn 不变
            const int fullpan = !c.cshift || tsk->kind != 0 || (tsk->p > 0 && tsk->p < c.npan - 1);
            long tn = (fullpan ? taskfn : run_task)(&c, &g_tb[tid], tsk, tg < c.ntask_items ? &c.tasks[tg] : NULL, tid);
            t = tn >= 0 ? tn : __atomic_fetch_add(&c.next, 1, __ATOMIC_RELAXED);
        }
        }
        sme_stop();
        free(sched);
        {   // 遗留的托管回写收尾
            uint64_t t0 = c.prof >= 2 ? cyc() : 0;
            sw_flush(&g_tb[tid]);
            if (c.prof >= 2) g_tb[tid].t_store += cyc() - t0;
        }
        if (c.prof >= 2) g_tail_cyc[tid] = cyc();
    }
    if (c.prof) {
        uint64_t t2 = cyc();
        double hz = (double)cyc_hz();
        fprintf(stderr, "[trsm prof] m=%ld n=%ld mb=%d nc=%d kc=%d jit=%d pp=%d tasks=%ld(items %ld) nth=%d hbm=%d Xnode=%d Lnode=%d Xp=%d pp0=%d pp1=%d hbmfree=%zuMB | setup %.3f ms, parallel %.3f ms\n",
                m, n, mb, nc, kc, c.jit, c.ppmode, c.ntask, c.ntask_items, nth, g_hbm_node, page_node_of(B), page_node_of(L),
                c.xp ? page_node_of(c.xp) : -1, c.pp[0] ? page_node_of(c.pp[0]) : -1, c.pp[1] ? page_node_of(c.pp[1]) : -1,
                g_hbm_node >= 0 ? node_free_bytes(g_hbm_node) >> 20 : 0,
                (t1 - t0) / hz * 1e3, (t2 - t1) / hz * 1e3);
        if (c.prof >= 3)
            fprintf(stderr, "[trsm prof]  setup 分解 µs：pin+探测 %.1f | 准入+对角缓冲 %.1f | xp+pp 缓冲 %.1f | flags %.1f | 任务表 %.1f | ditem %.1f\n",
                    (tsub[0] - t0) / hz * 1e6, (tsub[1] - tsub[0]) / hz * 1e6, (tsub[2] - tsub[1]) / hz * 1e6,
                    (tsub[3] - tsub[2]) / hz * 1e6, (tsub[4] - tsub[3]) / hz * 1e6, (t1 - tsub[4]) / hz * 1e6);
        if (c.prof >= 2) {
            {   // 线程唤醒延迟：各线程进入并行区的时刻相对主线程发起并行区的时刻
                uint64_t wmax = 0, wsum = 0; int wn = 0;
                for (int t = 0; t < nth; t++) { uint64_t d = g_wake_cyc[t] > t1 ? g_wake_cyc[t] - t1 : 0; if (d > wmax) wmax = d; wsum += d; wn++; }
                fprintf(stderr, "[trsm prof]  并行区唤醒：最慢线程 %.1f µs，平均 %.1f µs\n", wmax / hz * 1e6, (double)wsum / wn / hz * 1e6);
            }
            double sload = 0, spack = 0, spf = 0, swait = 0, skl = 0, sepi = 0, sdkl = 0, sdepi = 0, sst = 0, stask = 0, sprep = 0;
            long nks = 0, ndks = 0, ntk = 0, nswf = 0, nswq = 0;
            double sdump = 0, sddump = 0;
            uint64_t tail_min = ~0ull, tail_max = 0;
            for (int t = 0; t < nth; t++) {
                tbuf_t* tb = &g_tb[t];
                sload += tb->t_load; spack += tb->t_pack; spf += tb->t_pf; swait += tb->t_wait; skl += tb->t_kloop;
                sdump += tb->t_dump; sddump += tb->t_ddump; ndks += tb->n_dkstep;
                sepi += tb->t_epi; sdkl += tb->t_dkloop; sdepi += tb->t_depi; sst += tb->t_store; stask += tb->t_task;
                nswf += tb->n_swflush; nswq += tb->n_swq;
                sprep += g_prep_cyc[t]; nks += tb->n_kstep; ntk += tb->n_task;
                if (g_tail_cyc[t] < tail_min) tail_min = g_tail_cyc[t];
                if (g_tail_cyc[t] > tail_max) tail_max = g_tail_cyc[t];
            }
            double tail_idle = 0;
            for (int t = 0; t < nth; t++) tail_idle += tail_max - g_tail_cyc[t];
            double f = 1e3 / hz / nth;   // 每线程平均 ms
            double ideal_ms = (double)nks * 8.0 / 1.55e9 * 1e3 / nth;   // 每 k 步 8 周期 @1.55 GHz
            fprintf(stderr, "[trsm prof]  每线程平均 ms：prep %.3f | load %.3f pack %.3f xcopy %.3f wait %.3f kloop %.3f dump %.3f epi %.3f dkloop %.3f ddump %.3f depi %.3f store %.3f | task 合计 %.3f | 尾部不均衡 %.3f ms（平均空闲 %.3f ms）| k 步 %ld（对角 %ld）（理想 %.3f ms，kloop 效率 %.1f%%）\n",
                    sprep * f, sload * f, spack * f, spf * f, swait * f, skl * f, sdump * f, sepi * f, sdkl * f, sddump * f, sdepi * f, sst * f, stask * f,
                    (tail_max - tail_min) / hz * 1e3, tail_idle * f, nks, ndks, ideal_ms, ideal_ms / ((skl + sdkl) * f) * 100.0);
            fprintf(stderr, "[trsm prof]  托管回写：排队 %ld 行，同步收尾 %ld 行（%.1f%%）\n", nswq, nswf, nswq ? 100.0 * nswf / nswq : 0.0);
        }
    }
    free(ditem_i); free(ditem_s);
    free(c.S_of); free(c.diag_off); free(c.pp_off); free(c.owner_bound);
}
#endif // TRSM_NO_SME_ASM

// ---------------------------------------------------------------- 入口
void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb)
{
    if (m <= 0 || n <= 0) return;
#ifndef TRSM_NO_SME_ASM
    if (sme_f64_available()) {
        trsm_sme(m, n, L, lda, B, ldb);
        return;
    }
#endif
    trsm_generic(m, n, L, lda, B, ldb);
}
