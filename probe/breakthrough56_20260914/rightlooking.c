// Research only: blocked right-looking triangular solve, with fresh packing.
#define l_trsm baseline_trsm
#include "trsm-v55.c"
#undef l_trsm
#include "right-kernel.h"
static long right_completed;
long get_inv_completed(void) { return right_completed; }
static double *ra,*rx;
static size_t ca,cx;
static int reserve_right(double**ptr,size_t*cap,size_t need) {
    if(*ptr&&*cap>=need)return 1;
    free(*ptr);*ptr=NULL;*cap=0;
    size_t sz=(need+2097151)&~(size_t)2097151;
    *ptr=aligned_alloc(2097152,sz);
    if(!*ptr)return 0;
    madvise(*ptr,sz,MADV_HUGEPAGE);*cap=sz;return 1;
}
void l_trsm(int m,int n,const double*L,int lda,double*B,int ldb) {
    if(m<=0||m%16||n<=0||n%32||!sme_f64_available()||!env_int("TRSM_RIGHT",1)) {
        baseline_trsm(m,n,L,lda,B,ldb);return;
    }
    int nb=env_int("TRSM_RIGHT_NB",256),nc=env_int("TRSM_RIGHT_NC",128),mb=env_int("TRSM_RIGHT_MB",256);
    if(nb<16||nb>1024||nb%16)nb=256;
    if(nc<32||nc>512||nc%32||n%nc)nc=32;
    if(mb<16||mb>2048||mb%16)mb=256;
    double t0=omp_get_wtime(),diag=0,pack=0,update=0;
    if(!reserve_right(&ra,&ca,(size_t)m*nb*8)||!reserve_right(&rx,&cx,(size_t)n*nb*8)) {
        baseline_trsm(m,n,L,lda,B,ldb);return;
    }
    for(int k0=0;k0<m;k0+=nb) {
        int K=m-k0<nb?m-k0:nb,row0=k0+K,remaining=m-row0;
        double t=omp_get_wtime();
        baseline_trsm(K,n,L+(size_t)k0*lda+k0,lda,B+(size_t)k0*ldb,ldb);
        diag+=omp_get_wtime()-t;
        if(!remaining)break;
        t=omp_get_wtime();
        #pragma omp parallel
        {
            pin_self(omp_get_thread_num());
            #pragma omp for schedule(static) nowait
            for(int r=0;r<remaining;r+=16) {
                sme_start();
                pack_strip(L,lda,m,row0+r,k0,K,ra+(size_t)r*K);
                sme_stop();
            }
            #pragma omp for schedule(static)
            for(int p=0;p<n/nc;p++) {
                sme_start();
                for(int k=0;k<K;k++)for(int j=0;j<nc;j+=32)
                    inv_copy32(B+(size_t)(k0+k)*ldb+p*nc+j,rx+((size_t)p*nc+j)*K+k*32);
                sme_stop();
            }
        }
        pack+=omp_get_wtime()-t;t=omp_get_wtime();
        int blocks=(remaining+mb-1)/mb;
        #pragma omp parallel
        {
            pin_self(omp_get_thread_num());
            #pragma omp for schedule(dynamic,1)
            for(int task=0;task<(n/nc)*blocks;task++) {
                int rb=task%blocks,p=task/blocks,rend=(rb+1)*mb;
                if(rend>remaining)rend=remaining;
                sme_start();
                for(int r=rb*mb;r<rend;r+=16)for(int j=0;j<nc;j+=32) {
                    double*dst=B+(size_t)(row0+r)*ldb+p*nc+j;
                    za_load_direct(dst,(long)ldb*8,32);
                    right_compact(ra+(size_t)r*K,rx+((size_t)p*nc+j)*K,K);
                    za_store_direct(dst,(long)ldb*8,32);
                }
                sme_stop();
            }
        }
        update+=omp_get_wtime()-t;
    }
    right_completed++;
    if(env_int("TRSM_RIGHT_PROFILE",0))fprintf(stderr,"RIGHT m=%d n=%d nb=%d nc=%d mb=%d diag_ms=%.6f pack_ms=%.6f update_ms=%.6f total_ms=%.6f\n",m,n,nb,nc,mb,diag*1e3,pack*1e3,update*1e3,(omp_get_wtime()-t0)*1e3);
}
