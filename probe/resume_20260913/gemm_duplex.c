// Component adapter: a conventional GEMM using the production KC64 duplex kernels.
// All input-dependent packing and every output update occur in each call.
#define l_trsm unused_trsm_solver
#include "trsm-control.c"
#undef l_trsm
#include "kblas.h"

static double *dense_a, *dense_b, *dense_c, *dense_dummy;
static size_t dense_acap, dense_bcap, dense_ccap, dense_dcap;
static int dense_reserve(double** p,size_t* cap,size_t need)
{
    if(*p && *cap>=need)return 1;
    trsm_free_buffer(*p,*cap);*p=alloc_on(need,-1);*cap=*p?need:0;
    return *p!=NULL;
}
static inline __attribute__((always_inline)) void dense_copy16(const double* src,double* dst)
{
    asm volatile("ptrue p0.d\n\t"
                 "ld1d z0.d, p0/z, [%[s]]\n\t"
                 "ld1d z1.d, p0/z, [%[s], #1, mul vl]\n\t"
                 "st1d z0.d, p0, [%[d]]\n\t"
                 "st1d z1.d, p0, [%[d], #1, mul vl]\n\t"
                 ::[s]"r"(src),[d]"r"(dst):"memory","v0","v1","p0");
}
void pilot_gemm(int m,int n,int k,const double* a,int lda,const double* b,int ldb,double* c,int ldc)
{
    if(m%16 || n%32 || !sme_f64_available()){
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a,lda,b,ldb,0,c,ldc);return;
    }
    const long mb=192,nc=128,kc=64;
    const long strips=m/16,chunks=(k+kc-1)/kc,np=(n+nc-1)/nc,blocks=(m+mb-1)/mb;
    const int nth=omp_get_max_threads();
    if(g_ncpu<=0){
#pragma omp parallel num_threads(nth)
        pin_threads_in_region(omp_get_thread_num(),omp_get_num_threads());
    }
    g_align_owner_buffers=1;
    if(!dense_reserve(&dense_a,&dense_acap,(size_t)strips*chunks*1024*8)||
       !dense_reserve(&dense_b,&dense_bcap,(size_t)np*chunks*kc*nc*8)||
       !dense_reserve(&dense_c,&dense_ccap,(size_t)nth*mb*nc*8)||
       !dense_reserve(&dense_dummy,&dense_dcap,262144))abort();
#pragma omp parallel num_threads(nth)
    {
        int tid=omp_get_thread_num();pin_self(tid);
#pragma omp for schedule(static)
        for(long strip=0;strip<strips;strip++){
            for(long ch=0;ch<chunks;ch++){
                double* dst=dense_a+(ch*strips+strip)*1024;
                if(ch*kc+kc<=k){
                    sme_start();pack_strip_ld4(a,lda,strip*16,ch*kc,kc,dst);sme_stop();
                }else{
                    for(long q=0;q<kc;q++)for(long r=0;r<16;r++)
                        dst[q*16+r]=ch*kc+q<k?a[(strip*16+r)*lda+ch*kc+q]:0;
                }
            }
        }
#pragma omp for schedule(static)
        for(long idx=0;idx<chunks*np;idx++){
            long ch=idx/np,p=idx%np;
            double* dst=dense_b+idx*kc*nc;
            for(long t=0;t<8;t++){
                if(p*nc+16*t+16<=n && ch*kc+kc<=k){
                    sme_start();
                    for(long q=0;q<kc;q++)dense_copy16(b+(ch*kc+q)*ldb+p*nc+16*t,dst+t*1024+q*16);
                    sme_stop();
                }else{
                    for(long q=0;q<kc;q++)for(long j=0;j<16;j++)
                        dst[t*1024+q*16+j]=(ch*kc+q<k && p*nc+16*t+j<n)?b[(ch*kc+q)*ldb+p*nc+16*t+j]:0;
                }
            }
        }
#pragma omp for schedule(dynamic,1)
        for(long task=0;task<blocks*np;task++){
            long ib=task/np,p=task%np,rows=m-ib*mb,cols=n-p*nc;
            if(rows>mb)rows=mb;if(cols>nc)cols=nc;
            long S=rows/16,NT=cols/32;
            double* cb=dense_c+(size_t)tid*mb*nc;
            memset(cb,0,(size_t)mb*nc*8);
            double* pending=NULL;
            sme_start();sme_zero_za();
            for(long ch=0;ch<chunks;ch++){
                const double* Bch=dense_b+(ch*np+p)*kc*nc;
                const double* Bnext=ch+1<chunks?dense_b+((ch+1)*np+p)*kc*nc:NULL;
                for(long r=0;r<S;r++){
                    const double* A=dense_a+(ch*strips+ib*(mb/16)+r)*1024;
                    const double* Anext=r+1<S?A+1024:ch+1<chunks?dense_a+((ch+1)*strips+ib*(mb/16))*1024:NULL;
                    for(long t=0;t<NT;t++){
                        double* C=cb+16*r*nc+32*t;
                        const char* ap=Anext?(const char*)Anext+t*2048:(const char*)dense_dummy;
                        long slot=r*NT+t;
                        const char* bp=Bnext&&slot<32?(const char*)Bnext+slot*2048:(const char*)dense_dummy;
                        kloop_duplex_bank0(A,Bch+2*t*1024,pending?pending:C+16,ap,bp,(const char*)C);
                        kloop_duplex_bank4(A,Bch+(2*t+1)*1024,C,ap+1024,bp+1024,(const char*)(C+16));
                        pending=C+16;
                    }
                }
            }
            if(pending)za_flush_duplex_bank4(pending,nc*8);
            sme_stop();
            for(long i=0;i<rows;i++)for(long j=0;j<cols;j++)c[(ib*mb+i)*ldc+p*nc+j]=-cb[i*nc+j];
        }
    }
}
