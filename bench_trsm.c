#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

#include "kblas.h"

void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb);

// 生成随机数，线性同余法
double rand_lcg(int* localSeed)
{
    int high = *localSeed / 127773;
    int low = *localSeed % 127773;
    int test = 16807 * low - 2836 * high;
    int value = test + ((test > 0) ? 0 : 2147483647);
    *localSeed = value;
    return (double)value / 2147483647.0;
}

static int gMatGenThreads = 38;

void generate_matrix(int m, int n, double* A, int lda, bool isLower, int seed)
{
#pragma omp parallel num_threads(gMatGenThreads) /* fix num threads to make it reproducible */
    {
        int localSeed = seed + omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < m; ++i) {
            int jend = isLower ? i : n;
            double abssum = 0.0;
            for (int j = 0; j < jend; ++j) {
                A[i * lda + j] = rand_lcg(&localSeed) - 0.5;
                abssum += fabs(A[i * lda + j]);
            }
            if (isLower) {
                A[i * lda + i] = rand_lcg(&localSeed) + abssum + 1.0; // 确保对角线足够大
                for (int j = i + 1; j < n; ++j) {
                    A[i * lda + j] = 0.0;
                }
            }
        }
    }
}

// 计算误差的无穷范数
double validate_result(int m, int n, const double* X, const double* X_true, int ldx, double tol,
                       int num_report)
{
    double max_error = 0.0;
    int error_count = 0;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double diff = X[i * ldx + j] - X_true[i * ldx + j];
            double err = fabs(diff);
            if (err > tol || isnan(diff)) {
                if ((++error_count) <= num_report) {
                    printf(" Error at [%d][%d]: got %.6e, expected %.6e "
                           "diff = %.6e \n",
                           i, j, X[i * ldx + j], X_true[i * ldx + j], err);
                }
            }
            max_error = (max_error < err || isnan(err) ? err : max_error);
        }
    }
    return max_error;
}

void copy_matrix(int m, int n, const double* src, int lds, double* dst, int ldd)
{
    cblas_domatcopy(CblasRowMajor, CblasNoTrans, m, n, 1.0, src, lds, dst, ldd);
}

// 性能测试
void test_performance(int m, int n, int test_runs)
{
    int lda = m + 64;
    int ldb = n + 64;

    printf("\n================================= TRSM测试 ================================\n");
    printf("%6s  %6s %12s %12s %12s %8s\n", "M", "N", "Time(ms)", "GFLOPS", "最大误差", "校验");
    printf("---------------------------------------------------------------------------\n");

    double* L = (double*)malloc(lda * m * sizeof(double));
    double* B = (double*)malloc(ldb * m * sizeof(double));
    double* X = (double*)malloc(ldb * m * sizeof(double));
    double* X_true = (double*)malloc(ldb * m * sizeof(double));

    if (L == NULL || B == NULL || X == NULL || X_true == NULL) {
        free(L);
        free(X);
        free(B);
        free(X_true);
        printf("malloc failed\n");
        return;
    }

    generate_matrix(m, m, L, lda, true, 12345);
    generate_matrix(m, n, X_true, ldb, false, 67890);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, m, 1.0, L, lda, X_true, ldb, 0.0,
                B, ldb);

    copy_matrix(m, n, B, ldb, X, ldb);
    l_trsm(m, n, L, lda, X, ldb);

    /* 使用相同的随机种子重新生成L，避免L被修改 */
    generate_matrix(m, m, L, lda, true, 12345);

    double tol = 1e-12;
    double max_err = validate_result(m, n, X, X_true, ldb, tol, 10);

    if (max_err > tol || isnan(max_err)) {
        printf("%6d x%6d %12s %12s %12.2e %8s\n", m, n, "/", "/", max_err, "FAIL");
        free(L);
        free(B);
        free(X);
        free(X_true);
        return;
    }

    double total_time = 0.0;
    for (int t = 0; t < test_runs; ++t) {
        copy_matrix(m, n, B, ldb, X, ldb);

        double start = omp_get_wtime();
        l_trsm(m, n, L, lda, X, ldb);
        double end = omp_get_wtime();

        total_time += end - start;
    }

    double avg_time = total_time / test_runs;
    double flops = (double)m * m * n;
    double gflops = flops / avg_time * 1e-9;
    double avg_time_ms = avg_time * 1e3;

    printf("%6d x%6d %12.2f %12.4f %12.2e %8s\n", m, n, avg_time_ms, gflops, max_err, "PASS");

    free(L);
    free(B);
    free(X);
    free(X_true);
}

int main(int argc, char* argv[])
{
    int m = 512;
    int n = 512;
    int test_runs = 2;

    if (argc > 1) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        test_runs = atoi(argv[3]);
    }

    test_performance(m, n, test_runs);

    return 0;
}
