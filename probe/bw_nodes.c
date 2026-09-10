// 多节点并发读带宽探针：38 线程绑核，线程 t 的缓冲 mbind 到 nodes[t % k]，全部同时流式读。
// 用法：OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores numactl -N 1 ./bw_nodes MB_per_thread node[,node...]
// 输出各节点组合的聚合读带宽（ld1d，SVE），以及每节点分摊。
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <arm_sve.h>

#define MPOL_PREFERRED_ 1
static void* alloc_on(size_t bytes, int node)
{
    void* p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    madvise(p, bytes, MADV_HUGEPAGE);
    if (node >= 0) {
        unsigned long mask[16] = {0};
        mask[node / 64] |= 1UL << (node % 64);
        syscall(SYS_mbind, p, bytes, MPOL_PREFERRED_, mask, 1024, 0);
    }
    memset(p, 1, bytes);
    return p;
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static double read_sum(const double* p, size_t n)
{
    svbool_t pg = svptrue_b64();
    svfloat64_t a0 = svdup_f64(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i + 32 <= n; i += 32) {
        a0 = svadd_f64_x(pg, a0, svld1_f64(pg, p + i));
        a1 = svadd_f64_x(pg, a1, svld1_f64(pg, p + i + 8));
        a2 = svadd_f64_x(pg, a2, svld1_f64(pg, p + i + 16));
        a3 = svadd_f64_x(pg, a3, svld1_f64(pg, p + i + 24));
    }
    return svaddv_f64(pg, svadd_f64_x(pg, svadd_f64_x(pg, a0, a1), svadd_f64_x(pg, a2, a3)));
}

// 第 3 参数 share：每 share 个线程共用同一缓冲（同时读同一数据，测 L2 间转发 / 共享读的有效带宽）；
// 第 4 参数 stagger：共享组内线程 t 的起点错开 t*stagger KB（0 = 同起点）。
int main(int argc, char** argv)
{
    size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 32;
    int nodes[32], k = 0;
    if (argc > 2) {
        char* s = strdup(argv[2]);
        for (char* t = strtok(s, ","); t && k < 32; t = strtok(NULL, ",")) nodes[k++] = atoi(t);
    }
    if (k == 0) { nodes[0] = -1; k = 1; }
    int share = argc > 3 ? atoi(argv[3]) : 1; if (share < 1) share = 1;
    long stagger = argc > 4 ? atol(argv[4]) * 1024 / 8 : 0;
    int nth = omp_get_max_threads();
    size_t n = mb << 20 >> 3;
    double** bufs = calloc(nth, sizeof(*bufs));
    int ok = 1;
#pragma omp parallel
    {
        int t = omp_get_thread_num();
        if (t % share == 0) { bufs[t] = alloc_on(n * 8 + (size_t)stagger * 8 * share, nodes[(t / share) % k]); if (!bufs[t]) ok = 0; }
#pragma omp barrier
        if (t % share) bufs[t] = bufs[t - t % share];
    }
    if (!ok) { fprintf(stderr, "alloc failed\n"); return 1; }
    double best = 0, sink = 0;
    for (int rep = 0; rep < 4; rep++) {
        double t0 = now();
#pragma omp parallel reduction(+ : sink)
        {
            int t = omp_get_thread_num();
            sink += read_sum(bufs[t] + (t % share) * stagger, n);
        }
        double dt = now() - t0;
        double gbs = (double)nth * n * 8 / dt / 1e9;
        if (gbs > best) best = gbs;
    }
    printf("  nodes=%-10s share=%-2d stagger=%ldKB threads=%d  read %.1f GB/s (%.1f GB/s/thread)  [sink %g]\n",
           argc > 2 ? argv[2] : "default", share, stagger * 8 / 1024, nth, best, best / nth, sink);
    return 0;
}
