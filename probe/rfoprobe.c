// RFO 探针（审计 #5）：load 相把 B 行（768 B，行距 160256 B=TC1 ldb*8）svst1 进 cbuf 行。
// 若目标行已被 A 流逐出 L2，全行覆盖写是否触发 write-allocate 读（RFO）？三态对比：
//   A) 目标 L2 热（同一 48 KB 块反复用）——in-app 乐观假设
//   B) 目标 L2 冷（64 个 48 KB 槽轮转，3 MB > L2 768 KB）——in-app 悲观假设
//   C) 目标冷 + 每行先 dc zva（12 条 cache 行清零占有，绕过 RFO 读）
// 判读：B/A ≥ 1.5 → RFO 真实存在，且 C≈A 则 dc zva 是修法（load 相省 768 行读 ≈ −40%）；B≈A → RFO 不存在，关闭该方向。
// 编译：gcc -O3 -march=armv9-a+sve2 -fopenmp probe/rfoprobe.c -o rfoprobe
// 运行：numactl -N 1 --preferred=<hbm> ./rfoprobe（单线程）
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <arm_sve.h>

static inline uint64_t cyc(void){ uint64_t v; asm volatile("mrs %0, cntvct_el0" : "=r"(v)); return v; }
static inline uint64_t cyc_hz(void){ uint64_t v; asm volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }

#define ROWB 768L            // 每行字节（96 列 double）
#define LDBB 160256L         // 源行距（TC1 ldb*8）
#define ROWS 64L             // 每次任务 64 行（mb=64）
#define ITER 192L            // 任务数/变体（源足迹 192*64*160256/3 ≈ 0.65 GB/变体）

static void copy_task(const char* src, char* dst, int zva)
{
    for (long r = 0; r < ROWS; r++) {
        const double* s = (const double*)(src + r * LDBB);
        double* d = (double*)(dst + r * ROWB);
        if (zva)
            for (long l = 0; l < ROWB; l += 64) asm volatile("dc zva, %0" :: "r"((char*)d + l) : "memory");
        svbool_t pg = svptrue_b64();
        for (long j = 0; j < 96; j += 8) svst1_f64(pg, d + j, svld1_f64(pg, s + j));
    }
    asm volatile("dsb ish" ::: "memory");
}

int main(void)
{
    uint64_t dcz; asm volatile("mrs %0, dczid_el0" : "=r"(dcz));
    long zva_bytes = (dcz & 0x10) ? 0 : (4L << (dcz & 0xf));
    printf("dczid=0x%lx zva=%ldB hz=%lu\n", dcz, zva_bytes, cyc_hz());
    if (zva_bytes != 64) { printf("zva!=64B，C 变体不可比\n"); }

    size_t srcbytes = 3 * ITER * ROWS * LDBB;
    char* src = mmap(NULL, srcbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char* dsthot = aligned_alloc(4096, ROWS * ROWB);
    char* dstcold = aligned_alloc(4096, 64 * ROWS * ROWB);   // 64 槽 × 48 KB = 3 MB
    if (src == MAP_FAILED || !dsthot || !dstcold) { printf("alloc fail\n"); return 1; }
    madvise(src, srcbytes, MADV_HUGEPAGE);
    memset(src, 1, srcbytes);                                 // 预触碰：页就位，数据远超 L2 必冷
    memset(dsthot, 0, ROWS * ROWB);
    memset(dstcold, 0, 64 * ROWS * ROWB);

    const char* name[3] = { "A_hot ", "B_cold", "C_zva " };
    double ghz = cyc_hz() * 1e-9;
    for (int v = 0; v < 3; v++) {
        const char* s0 = src + (size_t)v * ITER * ROWS * LDBB;
        // 变体内预热 8 次任务（不计时），随后计时 ITER-8 次
        uint64_t t0 = 0; long done = 0;
        for (long it = 0; it < ITER; it++) {
            char* d = (v == 0) ? dsthot : dstcold + (it & 63) * ROWS * ROWB;
            if (v == 0) { volatile double x = ((double*)dsthot)[5]; (void)x; }  // 保持热
            if (it == 8) t0 = cyc();
            copy_task(s0 + it * ROWS * LDBB, d, v == 2);
            if (it >= 8) done++;
        }
        double tk = (double)(cyc() - t0) / ghz;               // ns 总
        double nsrow = tk / (done * ROWS);
        printf("%s  %8.1f ns/任务  %6.2f ns/行  %6.1f 行/µs\n", name[v], tk / done, nsrow, 1000.0 / nsrow);
    }
    return 0;
}
