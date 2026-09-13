// Research wrapper: two complete diagonal solves and a complete off-diagonal update.
// No matrix or schedule preprocessing is retained between calls.
#include "fused.c"
static long completed_splits;
long fused_split_count(void){return completed_splits;}
void l_trsm(int m,int n,const double* L,int lda,double* B,int ldb)
{
    int enabled=env_int("TRSM_FUSED",0);
    if(!enabled || m<128 || m%64 || n<=0 || n%64 || !sme_f64_available()){
        unused_trsm_solver(m,n,L,lda,B,ldb);return;
    }
    int cut=(m/2/32)*32;
    int level=env_int("TRSM_FUSED_LEVEL",1);
    double start=omp_get_wtime();
    unused_trsm_solver(cut,n,L,lda,B,ldb);
    double first=omp_get_wtime();
    fused_product(m-cut,n,cut,L+(size_t)cut*lda,lda,B,ldb,
                  B+(size_t)cut*ldb,ldb,-1,1,level);
    struct fused_stats detail=fused_last_stats();
    double update=omp_get_wtime();
    unused_trsm_solver(m-cut,n,L+(size_t)cut*lda+cut,lda,B+(size_t)cut*ldb,ldb);
    double last=omp_get_wtime();
    completed_splits++;
    if(env_int("TRSM_FUSED_PROFILE",0))
        fprintf(stderr,"FUSED_TRSM m=%d n=%d cut=%d level=%d actual=%d first_ms=%.6f update_ms=%.6f last_ms=%.6f total_ms=%.6f apack_ms=%.6f bpack_ms=%.6f compute_ms=%.6f leaves=%ld\n",
                m,n,cut,level,detail.level,1e3*(first-start),1e3*(update-first),1e3*(last-update),1e3*(last-start),detail.apack_ms,detail.bpack_ms,detail.compute_ms,detail.leaves);
}
