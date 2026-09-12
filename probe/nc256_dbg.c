// nc=256 几何 bug 定位：小形状全矩阵比对（SME 解 vs 朴素参考），输出错误分布
// gcc -O2 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp probe/nc256_dbg.c -o probe/nc256_dbg -lm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
void l_trsm(long m, long n, const double* L, long lda, double* B, long ldb);

int main(int argc, char** argv)
{
    long m = argc > 1 ? atol(argv[1]) : 1536;
    long n = argc > 2 ? atol(argv[2]) : 512;
    long lda = m + 8, ldb = n + 8;
    double* L = malloc((size_t)m * lda * 8);
    double* B = malloc((size_t)m * ldb * 8);
    double* R = malloc((size_t)m * ldb * 8);
    srand(7);
    for (long i = 0; i < m; i++) {
        for (long k = 0; k <= i; k++) L[i * lda + k] = (double)(rand() % 1999 - 999) / 1024.0;
        L[i * lda + i] = 2.0 + (double)(rand() % 999) / 1024.0;   // 对角远离 0
        for (long j = 0; j < n; j++) B[i * ldb + j] = (double)(rand() % 1999 - 999) / 512.0;
    }
    memcpy(R, B, (size_t)m * ldb * 8);
    // 参考：朴素前代
    for (long i = 0; i < m; i++) {
        for (long k = 0; k < i; k++) {
            double a = L[i * lda + k];
            const double* xr = R + k * ldb;
            double* br = R + i * ldb;
            for (long j = 0; j < n; j++) br[j] -= a * xr[j];
        }
        double d = 1.0 / L[i * lda + i];
        for (long j = 0; j < n; j++) R[i * ldb + j] *= d;
    }
    l_trsm(m, n, L, lda, B, ldb);
    // 比对
    long bad = 0; long first_r = -1, first_c = -1;
    static long colhist[64], rowblk[512];
    double maxrel = 0;
    for (long i = 0; i < m; i++) for (long j = 0; j < n; j++) {
        double got = B[i * ldb + j], ref = R[i * ldb + j];
        double rel = fabs(got - ref) / (fabs(ref) + 1e-300);
        if (rel > 1e-6) {
            bad++;
            if (first_r < 0) { first_r = i; first_c = j; }
            colhist[(j / 16) & 63]++;
            rowblk[(i / 64) & 511]++;
            if (rel > maxrel) maxrel = rel;
        }
    }
    printf("m=%ld n=%ld bad=%ld/%ld maxrel=%.3e first=(%ld,%ld)\n", m, n, bad, m * n, maxrel, first_r, first_c);
    if (bad) {
        printf("按 16 列瓦片错误数:"); for (int t = 0; t < n / 16 && t < 64; t++) printf(" t%d=%ld", t, colhist[t]); printf("\n");
        printf("按 64 行块错误数(前 24):"); for (int rb = 0; rb < 24 && rb < (m + 63) / 64; rb++) printf(" r%d=%ld", rb, rowblk[rb]); printf("\n");
        for (long i = first_r, shown = 0; i < first_r + 1 && shown < 8; i++)
            for (long j = 0; j < n && shown < 8; j++) {
                double got = B[i * ldb + j], ref = R[i * ldb + j];
                if (fabs(got - ref) / (fabs(ref) + 1e-300) > 1e-6) { printf("  (%ld,%ld) got=%.6g ref=%.6g\n", i, j, got, ref); shown++; }
            }
    }
    return bad != 0;
}
