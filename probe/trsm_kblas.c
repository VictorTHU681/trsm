// 基线对照：直接调用 KML kblas 的 cblas_dtrsm（只用于测参照性能，不是提交实现）
#include "kblas.h"

void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb)
{
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, m, n, 1.0, L, lda, B, ldb);
}
