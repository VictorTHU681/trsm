from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'trsm-v49-final.c').read_text()
key='static void diag_prep_sub(const double* L, long lda, long i0, int mbi, int s, double* pk)\n{'
assert key in s
s=s.replace(key,'static long audit_diag_count,audit_pack_count;\n'+key+'\n    __atomic_fetch_add(&audit_diag_count,1,__ATOMIC_RELAXED);',1)
key='static void pack_l_chunked(const ctx_t* c, long g) {'
assert key in s
s=s.replace(key,key+'\n    __atomic_fetch_add(&audit_pack_count,1,__ATOMIC_RELAXED);',1)
key='static void trsm_sme(long m, long n, const double* L, long lda, double* B, long ldb)\n{'
assert key in s
s=s.replace(key,key+'\n    audit_diag_count=audit_pack_count=0;',1)
key='    free(ditem_i); free(ditem_s);'
assert key in s
s=s.replace(key,r'''    long expected_pack=0;
    for(long g=0;g<c.nstrip;g++)if((16*g)/mb>0)expected_pack++;
    int ok=audit_diag_count==ndiag_items && audit_pack_count==expected_pack;
    printf("PREPROCESS_AUDIT m=%ld n=%ld diag=%ld expected_diag=%ld pack=%ld expected_pack=%ld threads=%d cpus=%d..%d %s\n",
      m,n,audit_diag_count,ndiag_items,audit_pack_count,expected_pack,nth,g_cpus[0],g_cpus[nth-1],ok?"PASS":"FAIL");
    if(!ok)abort();
'''+key,1)
(p/'trsm-close-audit50.c').write_text(s)
(p/'close-input-audit50.c').write_text(r'''#define main original_bench_main
#include "bench_trsm.c"
#undef main
int main(void){
 const int shapes[3][2]={{512,19968},{2432,17024},{17024,512}};int fail=0;
 for(int shape=0;shape<3;shape++){
  int m=shapes[shape][0],n=shapes[shape][1],lda=m+64,ldb=n+64;
  double*L=malloc((size_t)m*lda*8),*B=malloc((size_t)m*ldb*8),*X=malloc((size_t)m*ldb*8),*T=malloc((size_t)m*ldb*8);
  if(!L||!B||!X||!T)return 2;
  generate_matrix(m,m,L,lda,true,41723);generate_matrix(m,n,T,ldb,false,79843);
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,m,1,L,lda,T,ldb,0,B,ldb);
  for(int call=0;call<4;call++){
   if(call==2){L[(size_t)(m-2)*lda+17]+=.125;for(int j=0;j<n;j++)B[(size_t)(m-2)*ldb+j]=fma(.125,T[(size_t)17*ldb+j],B[(size_t)(m-2)*ldb+j]);}
   if(call==3){T[(size_t)23*ldb+n-3]+=.125;for(int i=23;i<m;i++)B[(size_t)i*ldb+n-3]=fma(.125,L[(size_t)i*lda+23],B[(size_t)i*ldb+n-3]);}
   copy_matrix(m,n,B,ldb,X,ldb);l_trsm(m,n,L,lda,X,ldb);
   double error=validate_result(m,n,X,T,ldb,1e-12,3);
   int bad=!(error<=1e-12);fail+=bad;
   printf("INPUT_AUDIT case=%d call=%d kind=%s maxerr=%.3e %s\n",shape+1,call,
    call==0?"initial":call==1?"identical_input":call==2?"one_L_entry_and_matching_rhs":"rhs_only",error,bad?"FAIL":"PASS");
  }
  free(L);free(B);free(X);free(T);
 }
 return fail?1:0;
}
''')
(p/'405-close-audit50.sh').write_text('''set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
sha256sum trsm-v49-final.c bench_trsm.c
gcc $flags close-input-audit50.c trsm-close-audit50.c -o close-input-audit50 -lm -lkblas
numactl -N 1 ./close-input-audit50
gcc $flags bench_trsm.c trsm-v49-final.c -o v50-close-production -lm -lkblas
for shape in '512 19968' '2432 17024' '17024 512';do
 echo CLOSE50_FORMAL shape=$shape
 numactl -N 1 ./v50-close-production $shape 1
done
echo CLOSE50_DONE
''')
