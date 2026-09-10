
// 前代：求解 L * X = B，L是下三角
void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb)
{
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double s = 0.0;
            for (int k = 0; k < i; ++k) {
                s += L[i * lda + k] * B[k * ldb + j];
            }
            B[i * ldb + j] = (B[i * ldb + j] - s) / L[i * lda + i];
        }
    }
}
