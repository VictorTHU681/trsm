// 快速基准：与官方 bench_trsm.c 相同的数据生成（LCG、同种子）、相同的内存布局与调用序列（先一次不计时调用，再计时），
// 但不用 cblas_dgemm 算参考解（case3 要 296 GFLOP），改用抽样残差校验：抽 NS 行算 |L·X − B| 相对于 Σ|L||X|+|B| 的比值。
// 编译：gcc -O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/fastbench.c trsm.c -o fastbench -lm
// 运行：OMP_NUM_THREADS=38 numactl -N 1 ./fastbench [-r reps] [-s rows] m n [m n ...]
//   -r 计时轮数（默认 3），-s 抽样行数（默认 64，0 关闭校验）
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <sys/mman.h>

void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb);

static double rand_lcg(int* localSeed)
{
    int high = *localSeed / 127773;
    int low = *localSeed % 127773;
    int test = 16807 * low - 2836 * high;
    int value = test + ((test > 0) ? 0 : 2147483647);
    *localSeed = value;
    return (double)value / 2147483647.0;
}

static const int gMatGenThreads = 38;

// 与官方一致
static void generate_matrix(int m, int n, double* A, int lda, bool isLower, int seed)
{
#pragma omp parallel num_threads(gMatGenThreads)
    {
        int localSeed = seed + omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < m; ++i) {
            int jend = isLower ? i : n;
            double abssum = 0.0;
            for (int j = 0; j < jend; ++j) {
                A[(size_t)i * lda + j] = rand_lcg(&localSeed) - 0.5;
                abssum += fabs(A[(size_t)i * lda + j]);
            }
            if (isLower) {
                A[(size_t)i * lda + i] = rand_lcg(&localSeed) + abssum + 1.0;
                for (int j = i + 1; j < n; ++j) A[(size_t)i * lda + j] = 0.0;
            }
        }
    }
}

static void copy_matrix(int m, int n, const double* src, int lds, double* dst, int ldd)
{
#pragma omp parallel for
    for (int i = 0; i < m; i++) memcpy(dst + (size_t)i * ldd, src + (size_t)i * lds, (size_t)n * 8);
}

// 抽样残差：行 i 取自 [0, m) 的均匀分布（含首末行），每行全部 n 列
static double sampled_residual(int m, int n, const double* L, int lda, const double* X, const double* B, int ldb, int ns, int* worst_row)
{
    double worst = 0.0; int wrow = -1;
    for (int s = 0; s < ns; s++) {
        int i = ns == 1 ? m - 1 : (int)((long long)s * (m - 1) / (ns - 1));
        double rowmax = 0.0;
#pragma omp parallel for reduction(max : rowmax)
        for (int j = 0; j < n; j++) {
            double acc = 0.0, scale = fabs(B[(size_t)i * ldb + j]);
            const double* li = L + (size_t)i * lda;
            for (int k = 0; k <= i; k++) {
                double t = li[k] * X[(size_t)k * ldb + j];
                acc += t; scale += fabs(t);
            }
            double r = fabs(acc - B[(size_t)i * ldb + j]) / (scale > 0 ? scale : 1.0);
            if (r > rowmax) rowmax = r;
        }
        if (rowmax > worst) { worst = rowmax; wrow = i; }
    }
    if (worst_row) *worst_row = wrow;
    return worst;
}

// FB_NOHUGE：首触前对指定缓冲打 MADV_NOHUGEPAGE 强制 4K 页（本机 THP enabled=always 缺页即得大页，
// 这是本地复现判题机低覆盖态的唯一手段）。取值含 'x'=解矩阵 X、'l'=三角阵 L。
// 用途：x + TRSM_REHUGE=0/1 交错 → 本地量化 X 页税与 REHUGE 回收上界（REHUGE 的 MADV_HUGEPAGE 会清掉
// NOHUGEPAGE 标志并重缺页为大页，重铺照常生效）；l → REHUGE-L 假想收益的绝对上界（无人再碰 L 的页结构）。
static void nohuge_if(const char* name, char tag, void* p, size_t bytes)
{
    const char* e = getenv("FB_NOHUGE");
    if (!e || !p || !strchr(e, tag)) return;
    uintptr_t lo = (uintptr_t)p & ~(uintptr_t)4095;
    uintptr_t hi = ((uintptr_t)p + bytes + 4095) & ~(uintptr_t)4095;
    int rc = madvise((void*)lo, hi - lo, MADV_NOHUGEPAGE);
    fprintf(stderr, "[fb nohuge] %s %zu MB rc=%d\n", name, bytes >> 20, rc);
}

// FB_XOFF：控制解矩阵 X 的 64B 对齐偏移（字节，0..63）。未设置 = 原样 malloc（大块 mmap → 恒定 +16B）。
// 用途：定价 X 回写 partial-line NT 写的端到端代价（FB_XOFF=0 为全对齐上界，16 ≈ malloc 现状）。
static void* g_xbase = NULL;
static double* alloc_x(size_t bytes)
{
    const char* e = getenv("FB_XOFF");
    if (!e) { g_xbase = malloc(bytes); return (double*)g_xbase; }
    long off = atol(e) & 63;
    size_t sz = (bytes + off + 8191) & ~(size_t)4095;
    char* p = aligned_alloc(4096, sz);
    g_xbase = p;
    return p ? (double*)(p + off) : NULL;
}

static int run_case(int m, int n, int reps, int ns)
{
    int lda = m + 64, ldb = n + 64;
    double* L = malloc((size_t)lda * m * sizeof(double));
    double* B = malloc((size_t)ldb * m * sizeof(double));
    double* X = alloc_x((size_t)ldb * m * sizeof(double));
    if (!L || !B || !X) { printf("malloc failed\n"); return 1; }
    nohuge_if("L", 'l', L, (size_t)lda * m * sizeof(double));
    nohuge_if("X", 'x', g_xbase, (size_t)ldb * m * sizeof(double) + 64);
    double tg0 = omp_get_wtime();
    generate_matrix(m, m, L, lda, true, 12345);
    generate_matrix(m, n, B, ldb, false, 67890);   // 直接把随机矩阵当 B（官方是 B = L·X_true；解的统计性质相同，数值尺度不同）
    double tg1 = omp_get_wtime();

    // 预热（对应官方的校验调用，不计时）
    copy_matrix(m, n, B, ldb, X, ldb);
    double tw0 = omp_get_wtime();
    l_trsm(m, n, L, lda, X, ldb);
    double tw1 = omp_get_wtime();

    double tmin = 1e30, tsum = 0.0;
    for (int r = 0; r < reps; r++) {
        copy_matrix(m, n, B, ldb, X, ldb);
        double t0 = omp_get_wtime();
        l_trsm(m, n, L, lda, X, ldb);
        double t1 = omp_get_wtime();
        double dt = t1 - t0;
        tsum += dt; if (dt < tmin) tmin = dt;
        printf("  rep %d: %8.3f ms  %9.2f GFLOPS\n", r, dt * 1e3, (double)m * m * n / dt * 1e-9);
    }
    double tavg = tsum / reps;
    int wrow = -1;
    double res = ns > 0 ? sampled_residual(m, n, L, lda, X, B, ldb, ns, &wrow) : 0.0;
    const char* verdict = ns == 0 ? "SKIP" : (res < 1e-13 && !isnan(res)) ? "PASS" : "FAIL";
    printf("%6d x%6d  min %8.3f ms %9.2f GFLOPS | avg %8.3f ms %9.2f GFLOPS | 预热 %.1f ms 生成 %.2f s | 抽样残差 %.2e (行 %d) %s\n",
           m, n, tmin * 1e3, (double)m * m * n / tmin * 1e-9, tavg * 1e3, (double)m * m * n / tavg * 1e-9,
           (tw1 - tw0) * 1e3, tg1 - tg0, res, wrow, verdict);
    free(L); free(B); free(g_xbase);
    return strcmp(verdict, "FAIL") == 0;
}

int main(int argc, char** argv)
{
    int reps = 3, ns = 64, a = 1, bad = 0;
    while (a < argc && argv[a][0] == '-') {
        if (!strcmp(argv[a], "-r") && a + 1 < argc) { reps = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-s") && a + 1 < argc) { ns = atoi(argv[++a]); }
        else { fprintf(stderr, "用法：%s [-r reps] [-s rows] m n [m n ...]\n", argv[0]); return 2; }
        a++;
    }
    if (a >= argc) { bad |= run_case(512, 19968, reps, ns); bad |= run_case(2432, 17024, reps, ns); bad |= run_case(17024, 512, reps, ns); return bad; }
    for (; a + 1 < argc; a += 2) bad |= run_case(atoi(argv[a]), atoi(argv[a + 1]), reps, ns);
    return bad;
}
