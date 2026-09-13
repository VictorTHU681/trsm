#ifndef TRSM_FUSED_RESEARCH_H
#define TRSM_FUSED_RESEARCH_H
struct fused_stats {
    double setup_ms, apack_ms, bpack_ms, compute_ms;
    long leaves;
    int level;
};
void fused_product(int m,int n,int k,const double* a,int lda,const double* b,int ldb,
                   double* c,int ldc,int alpha,int beta,int level);
struct fused_stats fused_last_stats(void);
#endif
