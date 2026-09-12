#define main original_bench_main
#include "bench_trsm.c"
#undef main
int main(void){
 int dims[][2]={{512,19968},{2432,17024},{17024,512}};
 int fail=0;
 for(int c=0;c<3;c++){
  int m=dims[c][0],n=dims[c][1],lda=m+64,ldb=n+64;
  double *L=malloc((size_t)m*lda*8),*B=malloc((size_t)m*ldb*8),*X=malloc((size_t)m*ldb*8),*T=malloc((size_t)m*ldb*8);
  for(int iter=0;iter<4;iter++){
   generate_matrix(m,m,L,lda,true,12345+iter*1001);
   generate_matrix(m,n,T,ldb,false,67890+iter*997);
   cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,m,1,L,lda,T,ldb,0,B,ldb);
   copy_matrix(m,n,B,ldb,X,ldb);
   l_trsm(m,n,L,lda,X,ldb);
   double e=validate_result(m,n,X,T,ldb,1e-12,3);
   printf("changed-input case=%d iteration=%d maxerr=%.3e %s\n",c+1,iter,e,!(e<=1e-12)?"FAIL":"PASS");
   if(!(e<=1e-12))fail++;
  }
  free(L);free(B);free(X);free(T);
 }
 return fail?1:0;
}
