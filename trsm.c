// TRSM：鲲鹏 SME(FP64 FMOPA) 实现。解 L·X = B，L 为行主序 m×m 下三角（lda），B/X 行主序 m×n（ldb），原地覆盖。
//
// 结构：
//   * 任务 = (行块 i：mb 行, 列块 p：nc 列)。C 块 (mb×nc) 放在线程私有缓冲里驻留 L2；
//     K 维按 kc 分段，每段对行块内每个 16 行条：取打包 A（L 的列段，k 主序 16 double/步），
//     对该行条的 nc/32 个微块各跑一次 K 循环（ZA 8 tile = 16×32 FP64 累加器），ZA 转存后从 C 块中减去。
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
    if (g_ncpu > 0 && tid < g_ncpu) {
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

static void* alloc_on(size_t bytes, int node)
{
    size_t sz = (bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    void* p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    if (node >= 0) {
        unsigned long mask[16] = {0};
        mask[node / 64] |= 1UL << (node % 64);
        syscall(SYS_mbind, p, sz, MPOL_BIND_, mask, (unsigned long)(sizeof(mask) * 8), 0);
    }
    madvise(p, sz, MADV_HUGEPAGE);
    return p;
}

static void free_sized(void* p, size_t bytes)
{
    if (!p) return;
    size_t sz = (bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    munmap(p, sz);
}

// 把已映射区间 [p, p+bytes) 的页迁到 node（页对齐向内收缩）
static int move_range_to_node(void* p, size_t bytes, int node)
{
    uintptr_t a = ((uintptr_t)p + 4095) & ~(uintptr_t)4095;
    uintptr_t e = ((uintptr_t)p + bytes) & ~(uintptr_t)4095;
    if (e <= a || node < 0) return -1;
    unsigned long mask[16] = {0};
    mask[node / 64] |= 1UL << (node % 64);
    return (int)syscall(SYS_mbind, (void*)a, (unsigned long)(e - a), MPOL_BIND_, mask,
                        (unsigned long)(sizeof(mask) * 8), MPOL_MF_MOVE_);
}

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
#define FMOPA8(a0, a1, b0, b1, b2, b3)                                            \
    "fmopa za0.d, p0/m, p0/m, " a0 ".d, " b0 ".d\n\t"                             \
    "fmopa za1.d, p0/m, p0/m, " a0 ".d, " b1 ".d\n\t"                             \
    "fmopa za2.d, p0/m, p0/m, " a0 ".d, " b2 ".d\n\t"                             \
    "fmopa za3.d, p0/m, p0/m, " a0 ".d, " b3 ".d\n\t"                             \
    "fmopa za4.d, p0/m, p0/m, " a1 ".d, " b0 ".d\n\t"                             \
    "fmopa za5.d, p0/m, p0/m, " a1 ".d, " b1 ".d\n\t"                             \
    "fmopa za6.d, p0/m, p0/m, " a1 ".d, " b2 ".d\n\t"                             \
    "fmopa za7.d, p0/m, p0/m, " a1 ".d, " b3 ".d\n\t"

// 一步 k：计算当前操作数（a/b）并加载下一步（n/m），随后按步长推进 A/B 指针
#define KS(a0, a1, b0, b1, b2, b3, n0, n1, m0, m1, m2, m3)                       \
    "ld1d " n0 ".d, p0/z, [%[xa]]\n\t"                                            \
    "fmopa za0.d, p0/m, p0/m, " a0 ".d, " b0 ".d\n\t"                             \
    "fmopa za1.d, p0/m, p0/m, " a0 ".d, " b1 ".d\n\t"                             \
    "ld1d " n1 ".d, p0/z, [%[xa], #1, mul vl]\n\t"                                \
    "fmopa za2.d, p0/m, p0/m, " a0 ".d, " b2 ".d\n\t"                             \
    "fmopa za3.d, p0/m, p0/m, " a0 ".d, " b3 ".d\n\t"                             \
    "ld1d " m0 ".d, p0/z, [%[xb]]\n\t"                                            \
    "fmopa za4.d, p0/m, p0/m, " a1 ".d, " b0 ".d\n\t"                             \
    "ld1d " m1 ".d, p0/z, [%[xb], #1, mul vl]\n\t"                                \
    "fmopa za5.d, p0/m, p0/m, " a1 ".d, " b1 ".d\n\t"                             \
    "ld1d " m2 ".d, p0/z, [%[xb], #2, mul vl]\n\t"                                \
    "fmopa za6.d, p0/m, p0/m, " a1 ".d, " b2 ".d\n\t"                             \
    "ld1d " m3 ".d, p0/z, [%[xb], #3, mul vl]\n\t"                                \
    "fmopa za7.d, p0/m, p0/m, " a1 ".d, " b3 ".d\n\t"                             \
    "add %[xa], %[xa], %[as]\n\t"                                                 \
    "add %[xb], %[xb], %[bs]\n\t"
#define KS_EVEN KS("z0", "z1", "z4", "z5", "z6", "z7", "z8", "z9", "z12", "z13", "z14", "z15")
#define KS_ODD  KS("z8", "z9", "z12", "z13", "z14", "z15", "z0", "z1", "z4", "z5", "z6", "z7")
#define FM_EVEN FMOPA8("z0", "z1", "z4", "z5", "z6", "z7")
#define FM_ODD  FMOPA8("z8", "z9", "z12", "z13", "z14", "z15")

// 两级预取游标（每 2 个 k 步发 2 行）：沿内层走 pfc 行后跳 pfj 到下一段，pfn 为内层长度（偶数）
#define PF2                                                                       \
    "prfm pldl1keep, [%[xpf]]\n\t"                                                \
    "prfm pldl1keep, [%[xpf], #64]\n\t"                                           \
    "add %[xpf], %[xpf], #128\n\t"                                                \
    "subs %[pfc], %[pfc], #2\n\t"                                                 \
    "b.gt 8f\n\t"                                                                 \
    "add %[xpf], %[xpf], %[pfj]\n\t"                                              \
    "mov %[pfc], %[pfn]\n\t"                                                      \
    "8:\n\t"

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
#define KROW(EV)                                                                  \
    KS_EVEN EV(0, "p1", "z16", "z17") PF2                                         \
    KS_ODD  EV(1, "p2", "z18", "z19")                                             \
    KS_EVEN EV(2, "p3", "z16", "z17") PF2                                         \
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
    "ld1d z0.d, p0/z, [%[xa]]\n\t"                                                \
    "ld1d z1.d, p0/z, [%[xa], #1, mul vl]\n\t"                                    \
    "ld1d z4.d, p0/z, [%[xb]]\n\t"                                                \
    "ld1d z5.d, p0/z, [%[xb], #1, mul vl]\n\t"                                    \
    "ld1d z6.d, p0/z, [%[xb], #2, mul vl]\n\t"                                    \
    "ld1d z7.d, p0/z, [%[xb], #3, mul vl]\n\t"                                    \
    "add %[xa], %[xa], %[as]\n\t"                                                 \
    "add %[xb], %[xb], %[bs]\n\t"
// 余下 k 步（%[k] = 还需"算+载"的步数，可为 0）+ 最后一步只算
#define KLOOP_REST                                                                \
    "cbz %[k], 6f\n\t"                                                            \
    "1:\n\t"                                                                      \
    KS_EVEN PF2                                                                   \
    "subs %[k], %[k], #1\n\t"                                                     \
    "b.eq 5f\n\t"                                                                 \
    KS_ODD                                                                        \
    "subs %[k], %[k], #1\n\t"                                                     \
    "b.ne 1b\n\t"                                                                 \
    "6:\n\t" FM_EVEN                                                              \
    "b 7f\n\t"                                                                    \
    "5:\n\t" FM_ODD                                                               \
    "7:\n\t"

// 预取任务：从 p 起，每 pfn 行（64 B）后跳 pfj 字节；每个 k 步 1 行
typedef struct { const char* p; long pfn, pfj; } pf_t;

// K 循环：ZA(16×32) += Σ_k A[k](16) ⊗ B[k](32)。pa：A（每 k 两个相邻向量，步长 as 字节）；
// pb：B 第 k 行 32 列起点，行距 bs；K ≥ 1；cols ∈ [1,32]；进入前 ZA 须已清零。
static inline void kloop_plain(const double* pa, long as, const double* pb, long bs, long K, long cols, const pf_t* pf)
{
    const double* xa = pa;
    const double* xb = pb;
    const char* xpf = pf->p;
    long k = K - 1, pfc = pf->pfn;
    (void)cols;   // B 来自零填充的连续缓冲，无需列谓词
    asm volatile(
        KLOOP_INIT
        KLOOP_REST
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [k] "+&r"(k), [xpf] "+&r"(xpf), [pfc] "+&r"(pfc)
        : [as] "r"(as), [bs] "r"(bs), [pfn] "r"(pf->pfn), [pfj] "r"(pf->pfj)
        : "memory", "cc", "x9", VCLOB, PCLOB);
}

// 同上，且在前 64 步里把暂存（上一微块的 ZA 转存）回写到 xc（行距 cs，列数 ecols）：SUB 为 C -= 暂存，SET 为 C = 暂存。
// 要求 K ≥ 65。
#define DEF_KLOOP_EPI(NAME, EV)                                                                                   \
static inline void NAME(const double* pa, long as, const double* pb, long bs, long K, long cols, const pf_t* pf,    \
                        const double* scratch, double* c, long cs, long ecols)                                    \
{                                                                                                                 \
    const double* xa = pa;                                                                                        \
    const double* xb = pb;                                                                                        \
    const char* xpf = pf->p;                                                                                      \
    const char* xs = (const char*)scratch;                                                                        \
    char* xc = (char*)c;                                                                                          \
    long k = K - 65, pfc = pf->pfn;                                                                                \
    (void)cols;                                                                                                   \
    asm volatile(                                                                                                 \
        SET_EPI_PREDS                                                                                             \
        KLOOP_INIT                                                                                                \
        "mov x10, #8\n\t"                                                                                         \
        "2:\n\t" KROW(EV)                                                                                         \
        "subs x10, x10, #1\n\t"                                                                                   \
        "b.ne 2b\n\t"                                                                                             \
        "sub %[xs], %[xs], #3840\n\t"                                                                             \
        "mov x10, #8\n\t"                                                                                         \
        "3:\n\t" KROW(EV)                                                                                         \
        "subs x10, x10, #1\n\t"                                                                                   \
        "b.ne 3b\n\t"                                                                                             \
        KLOOP_REST                                                                                                \
        : [xa] "+&r"(xa), [xb] "+&r"(xb), [k] "+&r"(k), [xpf] "+&r"(xpf), [pfc] "+&r"(pfc), [xs] "+&r"(xs), [xc] "+&r"(xc) \
        : [as] "r"(as), [bs] "r"(bs), [cs] "r"(cs), [ecols] "r"(ecols), [pfn] "r"(pf->pfn), [pfj] "r"(pf->pfj) \
        : "memory", "cc", "x9", "x10", VCLOB, PCLOB);                                                             \
}
DEF_KLOOP_EPI(kloop_epi_sub, EV_SUB)
DEF_KLOOP_EPI(kloop_epi_set, EV_SET)

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

// 把 X 面板（kk 行 × nci 列，行距 ldb）拷到连续缓冲 dst（行距 nc，越界列补零）
static void copy_panel(const double* X, long ldb, long kk, long nci, long nc, double* dst)
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
    long nrb, npan, ntask;
    int* S_of;              // 每行块的子块数
    double* diag;           // 全部行块的对角打包
    size_t* diag_off;       // 每行块起点（double 下标）
    int jit;                // 1：GEMM 部分就地转置；0：用预打包
    double* pp[2];          // 预打包（0 主副本，1 副副本，可为 NULL）
    size_t* pp_off;         // 每 16 行条起点（double 下标）
    long nstrip;
    volatile int* flags;    // nrb*npan
    volatile long next;
    int prof;
    int pf;                 // 软件预取开关
} ctx_t;

typedef struct {
    double* cbuf;    // mb*nc
    double* astrip;  // JIT 打包 A：kc*64（每 k 512 B 布局）
    double* xpan;    // X 面板拷贝：kc*nc
    double* scratch; // 4 KB ZA 转存
    double* scratch2;// 预取哑地址
    size_t cbuf_cap, astrip_cap, xpan_cap;
    // 分相计时（cntvct 周期），TRSM_PROFILE=2 时累计
    uint64_t t_load, t_pack, t_pf, t_wait, t_kloop, t_epi, t_dkloop, t_depi, t_store, t_task;
    long n_task, n_kstep;
} tbuf_t;

#define PROF_T(c, expr, field)                                           \
    do {                                                                 \
        if ((c)->prof >= 2) { uint64_t _t0 = cyc(); expr; tb->field += cyc() - _t0; } \
        else { expr; }                                                   \
    } while (0)

static tbuf_t g_tb[1024];
static uint64_t g_prep_cyc[1024], g_tail_cyc[1024];
static double* g_diag = NULL;
static size_t g_diag_cap = 0;
static double* g_pp[2] = {NULL, NULL};
static size_t g_pp_cap[2] = {0, 0};
static int g_pp_node[2] = {-3, -3};
static int* g_flags = NULL;
static size_t g_flags_cap = 0;
static int g_first_call = 1;

static void ensure_tbuf(int tid, size_t cbuf_d, size_t astrip_d, size_t xpan_d)
{
    tbuf_t* t = &g_tb[tid];
    if (t->cbuf_cap < cbuf_d) {
        free_sized(t->cbuf, t->cbuf_cap * 8);
        t->cbuf = alloc_on(cbuf_d * 8, -1);
        t->cbuf_cap = cbuf_d;
    }
    if (t->astrip_cap < astrip_d) {
        free_sized(t->astrip, t->astrip_cap * 8);
        t->astrip = alloc_on(astrip_d * 8, -1);
        t->astrip_cap = astrip_d;
    }
    if (t->xpan_cap < xpan_d) {
        free_sized(t->xpan, t->xpan_cap * 8);
        t->xpan = alloc_on(xpan_d * 8, -1);
        t->xpan_cap = xpan_d;
    }
    if (!t->scratch) { t->scratch = alloc_on(8192, -1); memset(t->scratch, 0, 8192); }
    if (!t->scratch2) { t->scratch2 = alloc_on(4096, -1); memset(t->scratch2, 0, 4096); }
}

// ---------------------------------------------------------------- 任务执行
static inline void wait_flag(const ctx_t* c, long idx)
{
    while (__atomic_load_n(&c->flags[idx], __ATOMIC_ACQUIRE) == 0) asm volatile("yield");
}

static void run_task(const ctx_t* c, tbuf_t* tb, long i, long p, int tid)
{
    const long mb = c->mb, nc = c->nc, kc = c->kc;
    const long i0 = i * mb;
    long mbi = c->m - i0; if (mbi > mb) mbi = mb;
    const long c0 = p * nc;
    long nci = c->n - c0; if (nci > nc) nci = nc;
    const int S = c->S_of[i];
    const long ntile = (nci + 31) / 32;
    double* cb = tb->cbuf;
    double* scratch = tb->scratch;
    const long ldb = c->ldb, lda = c->lda;
    const long cs = nc * 8;       // C 块 / X 面板缓冲行距字节
    double* Bblk = c->B + i0 * ldb + c0;
    const pf_t pf_dummy = { (const char*)tb->scratch2, 2, -128 };
    const int pf_on = c->pf;
    uint64_t tt0 = c->prof >= 2 ? cyc() : 0;

    // 1. C 块 ← B 子块（越界列/行补零）
    {
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        const svbool_t pg = svptrue_b64();
        for (long r = 0; r < 16L * S; r++) {
            double* d = cb + r * nc;
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
    int cpy = 0;   // 预打包副本选择（按任务交替）
    if (!c->jit && c->pp[1]) cpy = (int)((i + p) & 1);
    int pend = 0;            // 有待回写的上一微块（暂存中）
    double* pend_c = NULL;
    long pend_cols = 0;
    const long xlines = ((nci * 8 + 63) / 64 + 1) & ~1L;   // X 面板每行预取的 cache 行数（偶数）
    for (long k0 = 0; k0 < i0; k0 += kc) {
        long kk = i0 - k0; if (kk > kc) kk = kc;
        PROF_T(c, wait_flag(c, (k0 / mb) * c->npan + p), t_wait);
        PROF_T(c, copy_panel(c->B + k0 * ldb + c0, ldb, kk, nci, nc, tb->xpan), t_pf);
        const double* Xk = tb->xpan;
        const long k0n = k0 + kc;
        long rows_per_tile = kk / xlines; if (rows_per_tile < 1) rows_per_tile = 1;
        long xrow_next = 0;   // 下一段 X 面板已安排预取的行数
        for (int r = 0; r < S; r++) {
            const double* A;
            long as;
            const long row0 = i0 + 16 * r;
            if (c->jit) {
                long rows = c->m - row0; if (rows > 16) rows = 16;
                if (rows == 16 && (kk & 7) == 0) {
                    PROF_T(c, pack_strip_za(c->L, lda, row0, k0, kk, tb->astrip), t_pack);
                    as = 512;
                } else {
                    PROF_T(c, pack_strip(c->L, lda, c->m, row0, k0, kk, tb->astrip), t_pack);
                    as = 128;
                }
                A = tb->astrip;
            } else {
                A = c->pp[cpy] + c->pp_off[i0 / 16 + r] + (size_t)k0 * 16;
                as = 128;
            }
            for (long t = 0; t < ntile; t++) {
                pf_t pf = pf_dummy;
                if (pf_on && t < 2) {
                    // 下一条的 A（同段的下一条，或下一段的第 0 条）
                    long nr = (r + 1 < S) ? row0 + 16 : i0;
                    long nk = (r + 1 < S) ? k0 : k0n;
                    if (nk < i0) {
                        if (c->jit) {
                            long nrows = c->m - nr;
                            if (nrows > 8 * t) {
                                pf.p = (const char*)(c->L + (nr + 8 * t) * lda + nk);
                                pf.pfn = ((kk * 8 + 63) / 64 + 1) & ~1L;
                                pf.pfj = lda * 8 - pf.pfn * 64;
                            }
                        } else {
                            long g = (r + 1 < S) ? i0 / 16 + r + 1 : i0 / 16;
                            const double* An = c->pp[cpy] + c->pp_off[g] + (size_t)nk * 16;
                            pf.p = (const char*)An + (size_t)t * kk * 64;
                            pf.pfn = 1L << 40;
                            pf.pfj = 0;
                        }
                    }
                } else if (pf_on && k0n < i0 && xrow_next < kk) {
                    // 下一段 X 面板：本微块负责 rows_per_tile 行
                    pf.p = (const char*)(c->B + (k0n + xrow_next) * ldb + c0);
                    pf.pfn = xlines;
                    pf.pfj = ldb * 8 - xlines * 64;
                    xrow_next += rows_per_tile;
                }
                long cols = nci - 32 * t; if (cols > 32) cols = 32;
                sme_zero_za();
                if (pend && kk >= 65) {
                    PROF_T(c, kloop_epi_sub(A, as, Xk + 32 * t, cs, kk, cols, &pf, scratch, pend_c, cs, pend_cols), t_kloop);
                } else {
                    if (pend) PROF_T(c, epi_sub_std(scratch, pend_c, cs, pend_cols), t_epi);
                    PROF_T(c, kloop_plain(A, as, Xk + 32 * t, cs, kk, cols, &pf), t_kloop);
                }
                PROF_T(c, za_dump(scratch), t_epi);
                pend = 1; pend_c = cb + 16L * r * nc + 32 * t; pend_cols = cols;
                tb->n_kstep += kk;
            }
        }
    }
    if (pend) { PROF_T(c, epi_sub_std(scratch, pend_c, cs, pend_cols), t_epi); pend = 0; }

    // 3. 对角求解：X[s] = Linv_ss C[s] - Σ (Linv_ss L_ss') X[s']，A = 对角打包条带，B = C 块行 0..16(s+1)
    {
        const double* D = c->diag + c->diag_off[i];
        for (int s = 0; s < S; s++) {
            const double* A = D + diag_strip_off(s);
            long K = 16L * (s + 1);
            pf_t pf = pf_dummy;
            if (pf_on && s + 1 < S) { pf.p = (const char*)(D + diag_strip_off(s + 1)); pf.pfn = 1L << 40; pf.pfj = 0; }
            for (long t = 0; t < ntile; t++) {
                long cols = nci - 32 * t; if (cols > 32) cols = 32;
                sme_zero_za();
                if (pend && K >= 65) {
                    PROF_T(c, kloop_epi_set(A, 128, cb + 32 * t, cs, K, cols, &pf, scratch, pend_c, cs, pend_cols), t_dkloop);
                } else {
                    if (pend) PROF_T(c, epi_set_std(scratch, pend_c, cs, pend_cols), t_depi);
                    PROF_T(c, kloop_plain(A, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);
                }
                PROF_T(c, za_dump(scratch), t_depi);
                pend = 1; pend_c = cb + 16L * s * nc + 32 * t; pend_cols = cols;
                tb->n_kstep += K;
            }
            // 条末必须落地：下一条把这些行当 B 操作数
            if (pend) { PROF_T(c, epi_set_std(scratch, pend_c, cs, pend_cols), t_depi); pend = 0; }
        }
    }

    // 4. C 块 → X（非临时存储），发布 flag
    {
        uint64_t t0 = c->prof >= 2 ? cyc() : 0;
        const svbool_t pg = svptrue_b64();
        for (long r = 0; r < mbi; r++) {
            const double* s = cb + r * nc;
            double* d = Bblk + r * ldb;
            long j = 0;
            for (; j + 8 <= nci; j += 8) svstnt1_f64(pg, d + j, svld1_f64(pg, s + j));
            if (j < nci) {
                svbool_t pj = svwhilelt_b64((long)j, (long)nci);
                svstnt1_f64(pj, d + j, svld1_f64(pj, s + j));
            }
        }
        if (c->prof >= 2) tb->t_store += cyc() - t0;
    }
    __atomic_store_n(&c->flags[i * c->npan + p], 1, __ATOMIC_RELEASE);
    if (c->prof >= 2) { tb->t_task += cyc() - tt0; tb->n_task++; }
    (void)tid;
}


// ---------------------------------------------------------------- 主流程
static void trsm_sme(long m, long n, const double* L, long lda, double* B, long ldb)
{
    ctx_t c;
    memset(&c, 0, sizeof(c));
    c.m = m; c.n = n; c.lda = lda; c.ldb = ldb; c.L = L; c.B = B;
    c.prof = env_int("TRSM_PROFILE", 0);
    c.pf = env_int("TRSM_PF", 1);
    int mb = env_int("TRSM_MB", 256), nc = env_int("TRSM_NC", 128), kc = env_int("TRSM_KC", 128);
    if (mb < 16) mb = 16;
    mb = (mb + 15) & ~15;
    if (nc < 32) nc = 32;
    nc = (nc + 31) & ~31;
    if (kc < 16) kc = 16;
    kc = (kc + 15) & ~15;
    if (kc > mb) kc = mb;
    while (mb % kc) kc -= 16;           // 要求 kc | mb，使每个 K 段落在单个行块内
    c.mb = mb; c.nc = nc; c.kc = kc;
    c.nrb = (m + mb - 1) / mb;
    c.npan = (n + nc - 1) / nc;
    c.ntask = c.nrb * c.npan;
    c.nstrip = (m + 15) / 16;

    uint64_t t0 = c.prof ? cyc() : 0;
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

    // 决定 GEMM 部分的 L 打包方式：L 下三角 ≤ 阈值时预打包（两份），否则 JIT
    size_t tri_bytes = (size_t)m * (m + 1) / 2 * 8;
    long jit_mb = env_int("TRSM_JIT_MB", 256);
    c.jit = (tri_bytes > (size_t)jit_mb << 20) ? 1 : 0;
    if (env_int("TRSM_JIT", -1) >= 0) c.jit = env_int("TRSM_JIT", -1);

    // 首次调用：X 的内存节点安排（不计时阶段）。TRSM_XNODE：-1 不动（默认），-2 迁到本地 DDR，≥0 指定节点
    if (g_first_call) {
        int xn = env_int("TRSM_XNODE", -1);
        int target = xn == -2 ? g_cpu_node : xn;
        if (target >= 0) move_range_to_node(B, (size_t)m * ldb * 8, target);
        g_first_call = 0;
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
        free_sized(g_diag, g_diag_cap * 8);
        g_diag = alloc_on(doff * 8, -1);
        g_diag_cap = doff;
    }
    c.diag = g_diag;

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
        // 副本节点：主副本放 HBM（有则），副副本放本地 DDR；TRSM_LSPLIT=0 只用主副本
        int split = env_int("TRSM_LSPLIT", 1);
        int node0 = g_hbm_node >= 0 ? g_hbm_node : -1;
        int node1 = g_hbm_node >= 0 ? g_cpu_node : -1;
        if (env_int("TRSM_LNODE0", -3) != -3) node0 = env_int("TRSM_LNODE0", -3);
        if (env_int("TRSM_LNODE1", -3) != -3) node1 = env_int("TRSM_LNODE1", -3);
        for (int k = 0; k < 2; k++) {
            if (k == 1 && !split) break;
            int node = k == 0 ? node0 : node1;
            if (g_pp_cap[k] < off || g_pp_node[k] != node) {
                free_sized(g_pp[k], g_pp_cap[k] * 8);
                g_pp[k] = alloc_on(off * 8 + 4096, node);
                g_pp_cap[k] = off;
                g_pp_node[k] = node;
            }
            c.pp[k] = g_pp[k];
        }
    }

    size_t nflags = (size_t)c.ntask;
    if (g_flags_cap < nflags) {
        free(g_flags);
        g_flags = calloc(nflags, sizeof(int));
        g_flags_cap = nflags;
    }
    memset(g_flags, 0, nflags * sizeof(int));
    c.flags = g_flags;
    c.next = 0;

    // 预处理工作项：对角子块 (i, s) 与预打包条 g
    long ndiag_items = 0;
    for (long i = 0; i < c.nrb; i++) ndiag_items += c.S_of[i];
    long* ditem_i = malloc(sizeof(long) * ndiag_items);
    int* ditem_s = malloc(sizeof(int) * ndiag_items);
    {
        long q = 0;
        for (long i = 0; i < c.nrb; i++)
            for (int s = 0; s < c.S_of[i]; s++) { ditem_i[q] = i; ditem_s[q] = s; q++; }
    }
    uint64_t t1 = c.prof ? cyc() : 0;

#pragma omp parallel num_threads(nth)
    {
        int tid = omp_get_thread_num();
        pin_self(tid);
        ensure_tbuf(tid, (size_t)mb * nc, (size_t)64 * kc, (size_t)kc * nc);
        if (c.prof >= 2) {
            tbuf_t* tb = &g_tb[tid];
            tb->t_load = tb->t_pack = tb->t_pf = tb->t_wait = tb->t_kloop = tb->t_epi = 0;
            tb->t_dkloop = tb->t_depi = tb->t_store = tb->t_task = 0;
            tb->n_task = tb->n_kstep = 0;
        }
        uint64_t tp0 = c.prof >= 2 ? cyc() : 0;
        // 阶段 1：对角预处理
#pragma omp for schedule(dynamic, 1) nowait
        for (long q = 0; q < ndiag_items; q++) {
            long i = ditem_i[q];
            long mbi = m - i * mb; if (mbi > mb) mbi = mb;
            diag_prep_sub(L, lda, i * mb, (int)mbi, ditem_s[q], c.diag + c.diag_off[i] + diag_strip_off(ditem_s[q]));
        }
        // 阶段 1b：预打包
        if (!c.jit) {
#pragma omp for schedule(dynamic, 1) nowait
            for (long g = 0; g < c.nstrip; g++) {
                long i = (16 * g) / mb;
                long kk = i * mb;
                if (kk <= 0) continue;
                pack_strip(L, lda, m, 16 * g, 0, kk, c.pp[0] + c.pp_off[g]);
                if (c.pp[1]) memcpy(c.pp[1] + c.pp_off[g], c.pp[0] + c.pp_off[g], (size_t)kk * 16 * 8);
            }
        }
#pragma omp barrier
        if (c.prof >= 2) g_prep_cyc[tid] = cyc() - tp0;   // 预处理 + barrier
        // 阶段 2：任务
        sme_start();
        for (;;) {
            long t = __atomic_fetch_add(&c.next, 1, __ATOMIC_RELAXED);
            if (t >= c.ntask) break;
            run_task(&c, &g_tb[tid], t / c.npan, t % c.npan, tid);
        }
        sme_stop();
        if (c.prof >= 2) g_tail_cyc[tid] = cyc();
    }
    if (c.prof) {
        uint64_t t2 = cyc();
        double hz = (double)cyc_hz();
        fprintf(stderr, "[trsm prof] m=%ld n=%ld mb=%d nc=%d kc=%d jit=%d tasks=%ld nth=%d hbm=%d Xnode=%d | setup %.3f ms, parallel %.3f ms\n",
                m, n, mb, nc, kc, c.jit, c.ntask, nth, g_hbm_node, page_node_of(B),
                (t1 - t0) / hz * 1e3, (t2 - t1) / hz * 1e3);
        if (c.prof >= 2) {
            double sload = 0, spack = 0, spf = 0, swait = 0, skl = 0, sepi = 0, sdkl = 0, sdepi = 0, sst = 0, stask = 0, sprep = 0;
            long nks = 0, ntk = 0;
            uint64_t tail_min = ~0ull, tail_max = 0;
            for (int t = 0; t < nth; t++) {
                tbuf_t* tb = &g_tb[t];
                sload += tb->t_load; spack += tb->t_pack; spf += tb->t_pf; swait += tb->t_wait; skl += tb->t_kloop;
                sepi += tb->t_epi; sdkl += tb->t_dkloop; sdepi += tb->t_depi; sst += tb->t_store; stask += tb->t_task;
                sprep += g_prep_cyc[t]; nks += tb->n_kstep; ntk += tb->n_task;
                if (g_tail_cyc[t] < tail_min) tail_min = g_tail_cyc[t];
                if (g_tail_cyc[t] > tail_max) tail_max = g_tail_cyc[t];
            }
            double f = 1e3 / hz / nth;   // 每线程平均 ms
            double ideal_ms = (double)nks * 8.0 / 1.55e9 * 1e3 / nth;   // 每 k 步 8 周期 @1.55 GHz
            fprintf(stderr, "[trsm prof]  每线程平均 ms：prep %.3f | load %.3f pack %.3f xcopy %.3f wait %.3f kloop %.3f epi %.3f dkloop %.3f depi %.3f store %.3f | task 合计 %.3f | 尾部不均衡 %.3f ms | k 步 %ld（理想 %.3f ms，kloop 效率 %.1f%%）\n",
                    sprep * f, sload * f, spack * f, spf * f, swait * f, skl * f, sepi * f, sdkl * f, sdepi * f, sst * f, stask * f,
                    (tail_max - tail_min) / hz * 1e3, nks, ideal_ms, ideal_ms / ((skl + sdkl) * f) * 100.0);
        }
    }
    free(ditem_i); free(ditem_s);
    free(c.S_of); free(c.diag_off); free(c.pp_off);
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
