// Research component only. All input-dependent values are repacked every call.
// One-level Strassen with addition/subtraction fused into packing and epilogues.
#define l_trsm unused_trsm_solver
#include "trsm-control.c"
#undef l_trsm
#include "kblas.h"
#include "fused.h"
#include "fused-pack.h"

static double *work_a,*work_b,*work_c,*dummy;
static size_t cap_a,cap_b,cap_c,cap_dummy;
static struct fused_stats stats;
struct fused_stats fused_last_stats(void){return stats;}
static int reserve(double** p,size_t* cap,size_t bytes)
{
    if(*p && *cap>=bytes)return 1;
    trsm_free_buffer(*p,*cap);
    *p=alloc_on(bytes,-1);*cap=*p?bytes:0;
    return *p!=NULL;
}
static inline __attribute__((always_inline)) void copy16(const double* a,double* d)
{
    asm volatile("ptrue p0.d\n\t"
                 "ld1d z0.d, p0/z, [%[a]]\n\t"
                 "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t"
                 "st1d z0.d, p0, [%[d]]\n\t"
                 "st1d z1.d, p0, [%[d], #1, mul vl]\n\t"
                 ::[a]"r"(a),[d]"r"(d):"memory","v0","v1","p0");
}
#define BCOMBINE(NAME,OP) \
static inline __attribute__((always_inline)) void NAME(const double* a,const double* b,double* d) \
{ asm volatile("ptrue p0.d\n\t" \
    "ld1d z0.d, p0/z, [%[a]]\n\t" \
    "ld1d z1.d, p0/z, [%[a], #1, mul vl]\n\t" \
    "ld1d z2.d, p0/z, [%[b]]\n\t" \
    "ld1d z3.d, p0/z, [%[b], #1, mul vl]\n\t" \
    OP " z0.d, z0.d, z2.d\n\t" OP " z1.d, z1.d, z3.d\n\t" \
    "st1d z0.d, p0, [%[d]]\n\t" \
    "st1d z1.d, p0, [%[d], #1, mul vl]\n\t" \
    ::[a]"r"(a),[b]"r"(b),[d]"r"(d):"memory","v0","v1","v2","v3","p0"); }
BCOMBINE(add16,"fadd")
BCOMBINE(sub16,"fsub")

// cb holds minus the product; sign is the requested coefficient of the product.
static inline void emit(const double* cb,double* c,long rows,long cols,long ldc,int sign,int init)
{
    svbool_t pg=svptrue_b64();
    for(long i=0;i<rows;i++)for(long j=0;j<cols;j+=8){
        svfloat64_t x=svld1_f64(pg,cb+i*128+j);
        svfloat64_t y;
        if(init)y=sign>0?svneg_f64_x(pg,x):x;
        else{
            y=svld1_f64(pg,c+i*ldc+j);
            y=sign>0?svsub_f64_x(pg,y,x):svadd_f64_x(pg,y,x);
        }
        svst1_f64(pg,c+i*ldc+j,y);
    }
}

static void leaf(int m,int n,int k,const double* a,int lda,const double* a2,int lda2,int as,
                 const double* b,int ldb,const double* b2,int ldb2,int bs,
                 double* c,int ldc,int cs,int ci,double* c2,int ldc2,int cs2,int ci2)
{
    if(m%16 || n%32 || m<=0 || n<=0 || k<=0)abort();
    double begin=omp_get_wtime(),packed_a=0,packed_b=0,allocated;
    const long mb=env_int("FUSED_MB",128),nc=128,kc=64;
    if(mb<=0 || mb%16)abort();
    const int order=env_int("FUSED_ORDER",1);
    const long strips=m/16,chunks=(k+63)/64,np=(n+127)/128,blocks=(m+mb-1)/mb;
    int nth=omp_get_max_threads();
    g_align_owner_buffers=1;
    if(g_ncpu<=0 || g_pin_nth!=nth){
#pragma omp parallel num_threads(nth)
        pin_threads_in_region(omp_get_thread_num(),omp_get_num_threads());
        g_pin_nth=nth;
    }
    if(!reserve(&work_a,&cap_a,(size_t)blocks*(mb/16)*chunks*1024*8)||
       !reserve(&work_b,&cap_b,(size_t)chunks*np*kc*nc*8)||
       !reserve(&work_c,&cap_c,(size_t)nth*mb*nc*8)||
       !reserve(&dummy,&cap_dummy,262144))abort();
    allocated=omp_get_wtime();
#pragma omp parallel num_threads(nth)
    {
        int tid=omp_get_thread_num();pin_self(tid);
#pragma omp for schedule(static)
        for(long strip=0;strip<strips;strip++)for(long ch=0;ch<chunks;ch++){
            double* dst=work_a+((strip/(mb/16))*chunks*(mb/16)+ch*(mb/16)+strip%(mb/16))*1024;
            if(ch*kc+kc<=k){
                sme_start();
                if(!a2)pack_strip_ld4(a,lda,strip*16,ch*kc,kc,dst);
                else if(as>0)fused_pack_add(a,lda,a2,lda2,strip*16,ch*kc,kc,dst);
                else fused_pack_sub(a,lda,a2,lda2,strip*16,ch*kc,kc,dst);
                sme_stop();
            }else{
                for(long q=0;q<kc;q++)for(long r=0;r<16;r++){
                    double v=0;
                    if(ch*kc+q<k){
                        v=a[(strip*16+r)*lda+ch*kc+q];
                        if(a2){double v2=a2[(strip*16+r)*lda2+ch*kc+q];v=as>0?v+v2:v-v2;}
                    }
                    dst[q*16+r]=v;
                }
            }
        }
#pragma omp master
        packed_a=omp_get_wtime();
#pragma omp for schedule(static)
        for(long idx=0;idx<chunks*np;idx++){
            long ch=idx/np,p=idx%np;
            double* dst=work_b+idx*kc*nc;
            for(long t=0;t<8;t++){
                if(p*nc+t*16+16<=n && ch*kc+kc<=k){
                    sme_start();
                    for(long q=0;q<kc;q++){
                        const double* src=b+(ch*kc+q)*ldb+p*nc+t*16;
                        double* d=dst+t*1024+q*16;
                        if(!b2)copy16(src,d);
                        else if(bs>0)add16(src,b2+(ch*kc+q)*ldb2+p*nc+t*16,d);
                        else sub16(src,b2+(ch*kc+q)*ldb2+p*nc+t*16,d);
                    }
                    sme_stop();
                }else{
                    for(long q=0;q<kc;q++)for(long j=0;j<16;j++){
                        double v=0;
                        if(ch*kc+q<k && p*nc+t*16+j<n){
                            v=b[(ch*kc+q)*ldb+p*nc+t*16+j];
                            if(b2){double v2=b2[(ch*kc+q)*ldb2+p*nc+t*16+j];v=bs>0?v+v2:v-v2;}
                        }
                        dst[t*1024+q*16+j]=v;
                    }
                }
            }
        }
#pragma omp master
        packed_b=omp_get_wtime();
#pragma omp for schedule(dynamic,1)
        for(long task=0;task<blocks*np;task++){
            long ib=order?task%blocks:task/np,p=order?task/blocks:task%np;
            long rows=m-ib*mb,cols=n-p*nc;
            if(rows>mb)rows=mb;if(cols>nc)cols=nc;
            long S=rows/16,NT=cols/32;
            double* cb=work_c+(size_t)tid*mb*nc;
            memset(cb,0,(size_t)mb*nc*8);
            double* pending=NULL;
            sme_start();sme_zero_za();
            for(long ch=0;ch<chunks;ch++){
                const double* Bch=work_b+(ch*np+p)*kc*nc;
                const double* Bnext=ch+1<chunks?work_b+((ch+1)*np+p)*kc*nc:NULL;
                for(long r=0;r<S;r++){
                    const double* A=work_a+(ib*chunks*(mb/16)+ch*(mb/16)+r)*1024;
                    const double* Anext=r+1<S?A+1024:ch+1<chunks?work_a+(ib*chunks*(mb/16)+(ch+1)*(mb/16))*1024:NULL;
                    for(long t=0;t<NT;t++){
                        double* C=cb+16*r*nc+32*t;
                        const char* ap=Anext?(const char*)Anext+t*2048:(const char*)dummy;
                        long first=S-(32+NT-1)/NT;if(first<0)first=0;
                        long slot=(r-first)*NT+t;
                        const char* bp=Bnext&&slot>=0&&slot<32?(const char*)Bnext+slot*2048:(const char*)dummy;
                        kloop_hoist_bank0(A,Bch+t*2048,pending?pending:C+16,ap,bp,(const char*)C);
                        kloop_hoist_bank4(A,Bch+t*2048+1024,C,ap+1024,bp+1024,(const char*)(C+16));
                        pending=C+16;
                    }
                }
            }
            if(pending)za_flush_duplex_bank4(pending,nc*8);
            emit(cb,c+ib*mb*ldc+p*nc,rows,cols,ldc,cs,ci);
            if(c2)emit(cb,c2+ib*mb*ldc2+p*nc,rows,cols,ldc2,cs2,ci2);
            sme_stop();
        }
    }
    double end=omp_get_wtime();
    stats.setup_ms+=(allocated-begin)*1e3;
    stats.apack_ms+=(packed_a-allocated)*1e3;
    stats.bpack_ms+=(packed_b-packed_a)*1e3;
    stats.compute_ms+=(end-packed_b)*1e3;
    stats.leaves++;
}

void fused_product(int m,int n,int k,const double* a,int lda,const double* b,int ldb,
                   double* c,int ldc,int alpha,int beta,int level)
{
    stats=(struct fused_stats){0};
    if((alpha!=1 && alpha!=-1)||(beta!=0 && beta!=1)||level<0||level>1)abort();
    if(m%16 || n%32 || !sme_f64_available()){
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
        stats.level=-1;return;
    }
    if(!level || m%32 || n%64 || k%2){
        leaf(m,n,k,a,lda,NULL,0,0,b,ldb,NULL,0,0,c,ldc,alpha,!beta,NULL,0,0,0);
        return;
    }
    stats.level=1;
    int mh=m/2,nh=n/2,kh=k/2;
    const double *a11=a,*a12=a+kh,*a21=a+(size_t)mh*lda,*a22=a21+kh;
    const double *b11=b,*b12=b+nh,*b21=b+(size_t)kh*ldb,*b22=b21+nh;
    double *c11=c,*c12=c+nh,*c21=c+(size_t)mh*ldc,*c22=c21+nh;
    // Every first contribution overwrites only for beta=0; updates use beta=1.
    leaf(mh,nh,kh,a11,lda,a22,lda, 1,b11,ldb,b22,ldb, 1,c11,ldc,alpha,!beta,c22,ldc,alpha,!beta);
    leaf(mh,nh,kh,a21,lda,a22,lda, 1,b11,ldb,NULL,0,0,c21,ldc,alpha,!beta,c22,ldc,-alpha,0);
    leaf(mh,nh,kh,a11,lda,NULL,0,0,b12,ldb,b22,ldb,-1,c12,ldc,alpha,!beta,c22,ldc,alpha,0);
    leaf(mh,nh,kh,a22,lda,NULL,0,0,b21,ldb,b11,ldb,-1,c11,ldc,alpha,0,c21,ldc,alpha,0);
    leaf(mh,nh,kh,a11,lda,a12,lda, 1,b22,ldb,NULL,0,0,c11,ldc,-alpha,0,c12,ldc,alpha,0);
    leaf(mh,nh,kh,a21,lda,a11,lda,-1,b11,ldb,b12,ldb, 1,c22,ldc,alpha,0,NULL,0,0,0);
    leaf(mh,nh,kh,a12,lda,a22,lda,-1,b21,ldb,b22,ldb, 1,c11,ldc,alpha,0,NULL,0,0,0);
}
