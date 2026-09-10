// TRSM 理论上限探针：FP64 FMOPA 计算峰值 + 四种流式访存模式的带宽（read / write / rw 原地 / copy）。
// 编译：gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp peak.c -o peak
// 运行：OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores numactl -N 1 [-m 17] ./peak [MB_per_thread]
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <omp.h>

#define HWCAP2_SME_F64F64_BIT (1UL << 25)
#define VCLOB "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15", \
              "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31"
#define PCLOB "p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 8 个 ZA.D tile 轮转 FMOPA：吞吐（每条 = 64 FMA = 128 flop）
static double fmopa_gflops(long iters)
{
    long n = iters;
    double t0 = now();
    asm volatile(
        "smstart\n\t"
        "ptrue p0.d\n\t ptrue p1.d\n\t"
        "fdup z0.d, #1.0\n\t fdup z1.d, #0.5\n\t"
        "zero {za}\n\t"
        "1:\n\t"
        "fmopa za0.d, p0/m, p1/m, z0.d, z1.d\n\t"
        "fmopa za1.d, p0/m, p1/m, z0.d, z1.d\n\t"
        "fmopa za2.d, p0/m, p1/m, z0.d, z1.d\n\t"
        "fmopa za3.d, p0/m, p1/m, z0.d, z1.d\n\t"
        "fmopa za4.d, p0/m, p1/m, z0.d, z1.d\n\t"
        "fmopa za5.d, p0/m, p1/m, z0.d, z1.d\n\t"
        "fmopa za6.d, p0/m, p1/m, z0.d, z1.d\n\t"
        "fmopa za7.d, p0/m, p1/m, z0.d, z1.d\n\t"
        "subs %0, %0, #1\n\t"
        "b.ne 1b\n\t"
        "smstop\n\t"
        : "+r"(n) :: "memory", "cc", VCLOB, PCLOB);
    double t1 = now();
    return (double)iters * 8.0 * 128.0 / (t1 - t0) * 1e-9;
}

// 真实 GEMM 微内核形态：每 8 条 fmopa（16×32 块）配 6 条 ld1d（2 A + 4 B），数据在 L1
static double fmopa_16x32_gflops(long iters, const double* a, const double* b)
{
    long n = iters;
    double t0 = now();
    asm volatile(
        "smstart\n\t"
        "ptrue p0.d\n\t"
        "zero {za}\n\t"
        "1:\n\t"
        "ld1d z0.d, p0/z, [%1]\n\t"
        "ld1d z1.d, p0/z, [%1, #1, mul vl]\n\t"
        "ld1d z4.d, p0/z, [%2]\n\t"
        "ld1d z5.d, p0/z, [%2, #1, mul vl]\n\t"
        "ld1d z6.d, p0/z, [%2, #2, mul vl]\n\t"
        "ld1d z7.d, p0/z, [%2, #3, mul vl]\n\t"
        "fmopa za0.d, p0/m, p0/m, z0.d, z4.d\n\t"
        "fmopa za1.d, p0/m, p0/m, z0.d, z5.d\n\t"
        "fmopa za2.d, p0/m, p0/m, z0.d, z6.d\n\t"
        "fmopa za3.d, p0/m, p0/m, z0.d, z7.d\n\t"
        "fmopa za4.d, p0/m, p0/m, z1.d, z4.d\n\t"
        "fmopa za5.d, p0/m, p0/m, z1.d, z5.d\n\t"
        "fmopa za6.d, p0/m, p0/m, z1.d, z6.d\n\t"
        "fmopa za7.d, p0/m, p0/m, z1.d, z7.d\n\t"
        "subs %0, %0, #1\n\t"
        "b.ne 1b\n\t"
        "smstop\n\t"
        : "+r"(n) : "r"(a), "r"(b) : "memory", "cc", VCLOB, PCLOB);
    double t1 = now();
    return (double)iters * 8.0 * 128.0 / (t1 - t0) * 1e-9;
}

// ---- 带宽内核（n 为 double 个数，256 B 展开，n 须为 32 的倍数）
static double bw_read(const double* p, size_t n)
{
    double out[8] __attribute__((aligned(64)));
    asm volatile(
        "ptrue p0.d\n\t"
        "dup z0.d, #0\n\t dup z1.d, #0\n\t dup z2.d, #0\n\t dup z3.d, #0\n\t"
        "mov x9, %2\n\t"
        "1:\n\t"
        "ld1d z4.d, p0/z, [%1]\n\t"
        "ld1d z5.d, p0/z, [%1, #1, mul vl]\n\t"
        "ld1d z6.d, p0/z, [%1, #2, mul vl]\n\t"
        "ld1d z7.d, p0/z, [%1, #3, mul vl]\n\t"
        "fadd z0.d, z0.d, z4.d\n\t fadd z1.d, z1.d, z5.d\n\t"
        "fadd z2.d, z2.d, z6.d\n\t fadd z3.d, z3.d, z7.d\n\t"
        "add %1, %1, #256\n\t"
        "subs x9, x9, #32\n\t"
        "b.gt 1b\n\t"
        "fadd z0.d, z0.d, z1.d\n\t fadd z2.d, z2.d, z3.d\n\t fadd z0.d, z0.d, z2.d\n\t"
        "st1d z0.d, p0, [%0]\n\t"
        : : "r"(out), "r"(p), "r"(n) : "memory", "cc", "x9", VCLOB, PCLOB);
    return out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7];
}

static void bw_write(double* p, size_t n)
{
    asm volatile(
        "ptrue p0.d\n\t"
        "fdup z0.d, #1.0\n\t"
        "mov x9, %1\n\t"
        "1:\n\t"
        "st1d z0.d, p0, [%0]\n\t"
        "st1d z0.d, p0, [%0, #1, mul vl]\n\t"
        "st1d z0.d, p0, [%0, #2, mul vl]\n\t"
        "st1d z0.d, p0, [%0, #3, mul vl]\n\t"
        "add %0, %0, #256\n\t"
        "subs x9, x9, #32\n\t"
        "b.gt 1b\n\t"
        : "+r"(p) : "r"(n) : "memory", "cc", "x9", VCLOB, PCLOB);
}

static void bw_rw(double* p, size_t n)
{
    asm volatile(
        "ptrue p0.d\n\t"
        "fdup z0.d, #1.0\n\t"
        "mov x9, %1\n\t"
        "1:\n\t"
        "ld1d z4.d, p0/z, [%0]\n\t"
        "ld1d z5.d, p0/z, [%0, #1, mul vl]\n\t"
        "ld1d z6.d, p0/z, [%0, #2, mul vl]\n\t"
        "ld1d z7.d, p0/z, [%0, #3, mul vl]\n\t"
        "fadd z4.d, z4.d, z0.d\n\t fadd z5.d, z5.d, z0.d\n\t"
        "fadd z6.d, z6.d, z0.d\n\t fadd z7.d, z7.d, z0.d\n\t"
        "st1d z4.d, p0, [%0]\n\t"
        "st1d z5.d, p0, [%0, #1, mul vl]\n\t"
        "st1d z6.d, p0, [%0, #2, mul vl]\n\t"
        "st1d z7.d, p0, [%0, #3, mul vl]\n\t"
        "add %0, %0, #256\n\t"
        "subs x9, x9, #32\n\t"
        "b.gt 1b\n\t"
        : "+r"(p) : "r"(n) : "memory", "cc", "x9", VCLOB, PCLOB);
}

static void bw_copy(const double* s, double* d, size_t n)
{
    asm volatile(
        "ptrue p0.d\n\t"
        "mov x9, %2\n\t"
        "1:\n\t"
        "ld1d z4.d, p0/z, [%0]\n\t"
        "ld1d z5.d, p0/z, [%0, #1, mul vl]\n\t"
        "ld1d z6.d, p0/z, [%0, #2, mul vl]\n\t"
        "ld1d z7.d, p0/z, [%0, #3, mul vl]\n\t"
        "st1d z4.d, p0, [%1]\n\t"
        "st1d z5.d, p0, [%1, #1, mul vl]\n\t"
        "st1d z6.d, p0, [%1, #2, mul vl]\n\t"
        "st1d z7.d, p0, [%1, #3, mul vl]\n\t"
        "add %0, %0, #256\n\t"
        "add %1, %1, #256\n\t"
        "subs x9, x9, #32\n\t"
        "b.gt 1b\n\t"
        : "+r"(s), "+r"(d) : "r"(n) : "memory", "cc", "x9", VCLOB, PCLOB);
}

static void* alloc_touch(size_t bytes)
{
    void* p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(p, bytes, MADV_HUGEPAGE);
    memset(p, 1, bytes);
    return p;
}

static void bandwidth(int nth, size_t bytes_per_thread)
{
    size_t n = bytes_per_thread / sizeof(double);
    n &= ~(size_t)31;
    double** a = malloc(sizeof(double*) * nth);
    double** b = malloc(sizeof(double*) * nth);
#pragma omp parallel num_threads(nth)
    {
        int t = omp_get_thread_num();
        a[t] = alloc_touch(n * 8);
        b[t] = alloc_touch(n * 8);
    }
    double sink = 0;
    const int reps = 5;
    double best;

    best = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now(), sum = 0;
#pragma omp parallel num_threads(nth) reduction(+ : sum)
        sum += bw_read(a[omp_get_thread_num()], n);
        double t1 = now();
        sink += sum;
        double g = (double)nth * n * 8 / (t1 - t0) * 1e-9;
        if (g > best) best = g;
    }
    printf("  read   (ld1d)            %8.1f GB/s\n", best);

    best = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
#pragma omp parallel num_threads(nth)
        bw_write(b[omp_get_thread_num()], n);
        double t1 = now();
        double g = (double)nth * n * 8 / (t1 - t0) * 1e-9;
        if (g > best) best = g;
    }
    printf("  write  (st1d)            %8.1f GB/s\n", best);

    best = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
#pragma omp parallel num_threads(nth)
        bw_rw(a[omp_get_thread_num()], n);
        double t1 = now();
        double g = (double)nth * n * 16 / (t1 - t0) * 1e-9;
        if (g > best) best = g;
    }
    printf("  rw     (ld1d+st1d 原地)  %8.1f GB/s (读+写计两次)\n", best);

    best = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
#pragma omp parallel num_threads(nth)
        bw_copy(a[omp_get_thread_num()], b[omp_get_thread_num()], n);
        double t1 = now();
        double g = (double)nth * n * 16 / (t1 - t0) * 1e-9;
        if (g > best) best = g;
    }
    printf("  copy   (ld1d src, st1d dst) %5.1f GB/s (读+写计两次)\n", best);
    if (sink == 12345.0) printf("x");
    for (int t = 0; t < nth; t++) { munmap(a[t], n * 8); munmap(b[t], n * 8); }
    free(a); free(b);
}

int main(int argc, char** argv)
{
    unsigned long hw2 = getauxval(AT_HWCAP2);
    int nth = omp_get_max_threads();
    size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 32;
    int cpus[1024] = {0};
#pragma omp parallel num_threads(nth)
    cpus[omp_get_thread_num()] = sched_getcpu();
    int distinct = 0;
    for (int i = 0; i < nth; i++) { int d = 1; for (int j = 0; j < i; j++) if (cpus[j] == cpus[i]) d = 0; distinct += d; }
    printf("threads=%d distinct_cpus=%d cpu0=%d cpuN=%d SME_F64F64=%d\n", nth, distinct, cpus[0], cpus[nth - 1],
           !!(hw2 & HWCAP2_SME_F64F64_BIT));
    if (!(hw2 & HWCAP2_SME_F64F64_BIT)) { printf("no SME F64, skip compute\n"); }
    else {
        long iters = 5000000;
        double* small = aligned_alloc(64, 8192);
        memset(small, 0, 8192);
        double g1 = fmopa_gflops(iters);
        printf("== 单核 FMOPA.D 8-tile      : %8.2f GFLOPS (%.3f fmopa/ns → %.3f GHz)\n", g1, g1 / 128.0, g1 / 128.0);
        printf("== 单核 16x32 内核形态(L1)  : %8.2f GFLOPS\n", fmopa_16x32_gflops(iters, small, small + 16));
        double agg = 0;
#pragma omp parallel num_threads(nth) reduction(+ : agg)
        agg += fmopa_gflops(iters);
        printf("== %d 核 FMOPA.D 聚合        : %8.1f GFLOPS = %.3f TFLOPS\n", nth, agg, agg * 1e-3);
        agg = 0;
#pragma omp parallel num_threads(nth) reduction(+ : agg)
        agg += fmopa_16x32_gflops(iters, small, small + 16);
        printf("== %d 核 16x32 内核形态聚合  : %8.1f GFLOPS = %.3f TFLOPS\n", nth, agg, agg * 1e-3);
    }
    printf("== 带宽：%d 线程，每线程 %zu MB ==\n", nth, mb);
    bandwidth(nth, mb << 20);
    if (nth > 1) {
        printf("== 带宽：1 线程，%zu MB ==\n", mb);
        bandwidth(1, mb << 20);
    }
    return 0;
}
