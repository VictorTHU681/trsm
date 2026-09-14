// Research only: per-call block inversion and a persistent tiled RHS.
#define l_trsm baseline_trsm
#include "trsm-v55.c"
#undef l_trsm
#include "fullinv-compact.h"
#include "right-update.h"
static long tiled_completed;
long get_inv_completed(void) { return tiled_completed; }
static double *ta,*tx,*ti,*tip,*tb;
static size_t tca,tcx,tci,tcip,tcb;
static int tile_reserve(double**ptr,size_t*cap,size_t need) {
    if(*ptr&&*cap>=need)return 1;
    free(*ptr);*ptr=NULL;*cap=0;
    size_t sz=(need+2097151)&~(size_t)2097151;
    *ptr=aligned_alloc(2097152,sz);
    if(!*ptr)return 0;
    madvise(*ptr,sz,MADV_HUGEPAGE);*cap=sz;return 1;
}
void l_trsm(int m,int n,const double*L,int lda,double*B,int ldb) {
    if(m<=0||m%16||n<=0||n%32||!sme_f64_available()||!env_int("TRSM_TILED_RIGHT",1)) {
        baseline_trsm(m,n,L,lda,B,ldb);return;
    }
    int nb=env_int("TRSM_TILE_NB",512),nc=env_int("TRSM_TILE_NC",128),mb=env_int("TRSM_TILE_MB",256);
    if(nb<16||nb>1024||nb%16)nb=512;
    if(nc<32||nc>512||nc%32||n%nc)nc=32;
    if(mb<16||mb>2048||mb%16)mb=256;
    double t0=omp_get_wtime(),prep=0,diag=0,pack=0,update=0,copyin=0,copyout=0;
    size_t invsz=(size_t)((m+nb-1)/nb)*nb*nb*8;
    if(!tile_reserve(&ta,&tca,(size_t)m*nb*8)||!tile_reserve(&tx,&tcx,(size_t)m*n*8)
       ||!tile_reserve(&ti,&tci,invsz)||!tile_reserve(&tip,&tcip,invsz)
       ||!tile_reserve(&tb,&tcb,(size_t)nb*n*8)) {
        baseline_trsm(m,n,L,lda,B,ldb);return;
    }
    double t=omp_get_wtime();
    #pragma omp parallel for schedule(dynamic,1)
    for(int block=0;block<(m+nb-1)/nb;block++) {
        int k0=block*nb,K=m-k0<nb?m-k0:nb;
        double*inv=ti+(size_t)block*nb*nb;
        for(int i=0;i<K;i++) {
            double*row=inv+(size_t)i*nb;
            for(int j=0;j<K;j++)row[j]=0;
            row[i]=1.0/L[(size_t)(k0+i)*lda+k0+i];
            for(int k=0;k<i;k++) {
                double v=-L[(size_t)(k0+i)*lda+k0+k]/L[(size_t)(k0+i)*lda+k0+i];
                for(int j=0;j<=k;j+=8) {
                    svbool_t pg=svwhilelt_b64((uint64_t)j,(uint64_t)(k+1));
                    svst1_f64(pg,row+j,svmla_n_f64_x(pg,svld1_f64(pg,row+j),svld1_f64(pg,inv+(size_t)k*nb+j),v));
                }
            }
        }
        sme_start();
        for(int r=0;r<K;r+=16)pack_strip(inv,nb,K,r,0,r+16,tip+(size_t)block*nb*nb+(size_t)r*nb);
        sme_stop();
    }
    prep=omp_get_wtime()-t;t=omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for(long task=0;task<(long)(m/16)*(n/nc);task++) {
        int r0=task/(n/nc)*16,p=task%(n/nc);
        sme_start();
        for(int r=r0;r<r0+16;r++)for(int j=0;j<nc;j+=32)
            inv_copy32(B+(size_t)r*ldb+p*nc+j,tx+((size_t)p*nc+j)*m+(size_t)r*32);
        sme_stop();
    }
    copyin=omp_get_wtime()-t;
    for(int k0=0;k0<m;k0+=nb) {
        int K=m-k0<nb?m-k0:nb,row0=k0+K,remaining=m-row0;
        t=omp_get_wtime();
        #pragma omp parallel
        {
            pin_self(omp_get_thread_num());
            #pragma omp for schedule(static)
            for(int p=0;p<n/32;p++) {
                sme_start();
                for(int k=0;k<K;k++)inv_copy32(tx+(size_t)p*m*32+(size_t)(k0+k)*32,tb+(size_t)p*nb*32+k*32);
                sme_stop();
            }
            #pragma omp for schedule(dynamic,1)
            for(int task=0;task<(n/32)*(K/16);task++) {
                int p=task/(K/16),r=task%(K/16)*16;
                sme_start();sme_zero_za();
                inv_compact(tip+(size_t)(k0/nb)*nb*nb+(size_t)r*nb,tb+(size_t)p*nb*32,r+16);
                za_store_direct(tx+(size_t)p*m*32+(size_t)(k0+r)*32,256,32);
                sme_stop();
            }
        }
        diag+=omp_get_wtime()-t;
        if(!remaining)break;
        t=omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for(int r=0;r<remaining;r+=16) {
            sme_start();pack_strip(L,lda,m,row0+r,k0,K,ta+(size_t)r*K);sme_stop();
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
                    double*dst=tx+((size_t)p*nc+j)*m+(size_t)(row0+r)*32;
                    za_load_direct(dst,256,32);
                    right_compact(ta+(size_t)r*K,tx+((size_t)p*nc+j)*m+(size_t)k0*32,K);
                    za_store_direct(dst,256,32);
                }
                sme_stop();
            }
        }
        update+=omp_get_wtime()-t;
    }
    t=omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for(long task=0;task<(long)(m/16)*(n/nc);task++) {
        int r0=task/(n/nc)*16,p=task%(n/nc);
        sme_start();
        for(int r=r0;r<r0+16;r++)for(int j=0;j<nc;j+=32)
            inv_copy32(tx+((size_t)p*nc+j)*m+(size_t)r*32,B+(size_t)r*ldb+p*nc+j);
        sme_stop();
    }
    copyout=omp_get_wtime()-t;tiled_completed++;
    if(env_int("TRSM_TILE_PROFILE",0))fprintf(stderr,"TILED_RIGHT m=%d n=%d nb=%d nc=%d mb=%d prep_ms=%.6f copyin_ms=%.6f diag_ms=%.6f pack_ms=%.6f update_ms=%.6f copyout_ms=%.6f total_ms=%.6f\n",m,n,nb,nc,mb,prep*1e3,copyin*1e3,diag*1e3,pack*1e3,update*1e3,copyout*1e3,(omp_get_wtime()-t0)*1e3);
}
