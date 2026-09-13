// Batch all seven products within this call. Source quadrants are read once for
// packing; there is no cross-call matrix/preprocessing reuse.
#define fused_product serial_fused_product
#define fused_last_stats serial_fused_last_stats
#include "fused.c"
#undef fused_product
#undef fused_last_stats

static double *all_a,*all_b,*all_m;
static size_t all_acap,all_bcap,all_mcap;
static double compose_ms;
double fused_compose_ms(void){return compose_ms;}
struct fused_stats fused_last_stats(void){return stats;}
static inline __attribute__((always_inline)) void put(svbool_t pg,double* d,svfloat64_t x,int nt)
{if(nt)svstnt1_f64(pg,d,x);else svst1_f64(pg,d,x);}
void fused_product(int m,int n,int k,const double* a,int lda,const double* b,int ldb,
                   double* c,int ldc,int alpha,int beta,int level)
{
    compose_ms=0;
    if(!level || m%32 || n%64 || k%2 || !sme_f64_available()){
        serial_fused_product(m,n,k,a,lda,b,ldb,c,ldc,alpha,beta,level);return;
    }
    if((alpha!=1&&alpha!=-1)||(beta!=0&&beta!=1)||level!=1)abort();
    stats=(struct fused_stats){0};stats.level=1;stats.leaves=7;
    double begin=omp_get_wtime(),a_end=0,b_end=0,gemm_end=0,allocated;
    const long mh=m/2,nh=n/2,kh=k/2,mb=env_int("FUSED_MB",128),nc=128,kc=64;
    if(mb<=0||mb%16)abort();
    const long strips=mh/16,chunks=(kh+63)/64,np=(nh+127)/128,blocks=(mh+mb-1)/mb;
    const size_t asize=(size_t)blocks*(mb/16)*chunks*1024,bsize=(size_t)chunks*np*kc*nc;
    const long ntasks=blocks*np;
    const size_t msize=(size_t)ntasks*mb*nc;
    const int nt=env_int("FUSED_PACK_NT",1),outnt=env_int("FUSED_OUTPUT_NT",0);
    const int order=env_int("FUSED_ORDER",1),interleave=env_int("FUSED_INTERLEAVE",0);
    const int nth=omp_get_max_threads();
    if(g_ncpu<=0||g_pin_nth!=nth){
#pragma omp parallel num_threads(nth)
        pin_threads_in_region(omp_get_thread_num(),omp_get_num_threads());
        g_pin_nth=nth;
    }
    g_align_owner_buffers=1;
    if(!reserve(&all_a,&all_acap,7*asize*8)||!reserve(&all_b,&all_bcap,7*bsize*8)||
       !reserve(&all_m,&all_mcap,7*msize*8)||!reserve(&work_c,&cap_c,(size_t)nth*mb*nc*8)||
       !reserve(&dummy,&cap_dummy,262144))abort();
    allocated=omp_get_wtime();
    const double *aq[4]={a,a+kh,a+mh*lda,a+mh*lda+kh};
    const double *bq[4]={b,b+nh,b+kh*ldb,b+kh*ldb+nh};
#pragma omp parallel num_threads(nth)
    {
        int tid=omp_get_thread_num();pin_self(tid);
        double tmp[4][1024] __attribute__((aligned(64)));
#pragma omp for schedule(static)
        for(long strip=0;strip<strips;strip++)for(long ch=0;ch<chunks;ch++){
            if(ch*kc+kc<=kh){
                sme_start();
                for(int q=0;q<4;q++)pack_strip_ld4(aq[q],lda,strip*16,ch*kc,kc,tmp[q]);
            }else{
                for(int q=0;q<4;q++)for(long t=0;t<kc;t++)for(long r=0;r<16;r++)
                    tmp[q][t*16+r]=ch*kc+t<kh?aq[q][(strip*16+r)*lda+ch*kc+t]:0;
                sme_start();
            }
            size_t off=((strip/(mb/16))*chunks*(mb/16)+ch*(mb/16)+strip%(mb/16))*1024;
            svbool_t pg=svptrue_b64();
            for(long v=0;v<1024;v+=8){
                svfloat64_t x0=svld1_f64(pg,tmp[0]+v),x1=svld1_f64(pg,tmp[1]+v);
                svfloat64_t x2=svld1_f64(pg,tmp[2]+v),x3=svld1_f64(pg,tmp[3]+v);
                put(pg,all_a+0*asize+off+v,svadd_f64_x(pg,x0,x3),nt);
                put(pg,all_a+1*asize+off+v,svadd_f64_x(pg,x2,x3),nt);
                put(pg,all_a+2*asize+off+v,x0,nt);
                put(pg,all_a+3*asize+off+v,x3,nt);
                put(pg,all_a+4*asize+off+v,svadd_f64_x(pg,x0,x1),nt);
                put(pg,all_a+5*asize+off+v,svsub_f64_x(pg,x2,x0),nt);
                put(pg,all_a+6*asize+off+v,svsub_f64_x(pg,x1,x3),nt);
            }
            sme_stop();
        }
#pragma omp master
        a_end=omp_get_wtime();
#pragma omp for schedule(static)
        for(long idx=0;idx<chunks*np;idx++){
            long ch=idx/np,p=idx%np;
            sme_start();
            svbool_t pg=svptrue_b64();
            for(long tile=0;tile<8;tile++)for(long r=0;r<kc;r++)for(long lane=0;lane<16;lane+=8){
                long row=ch*kc+r,col=p*nc+tile*16+lane;
                svfloat64_t x0=svdup_f64(0),x1=svdup_f64(0),x2=svdup_f64(0),x3=svdup_f64(0);
                if(row<kh&&col<nh){
                    x0=svld1_f64(pg,bq[0]+row*ldb+col);x1=svld1_f64(pg,bq[1]+row*ldb+col);
                    x2=svld1_f64(pg,bq[2]+row*ldb+col);x3=svld1_f64(pg,bq[3]+row*ldb+col);
                }
                size_t off=(size_t)idx*kc*nc+tile*1024+r*16+lane;
                put(pg,all_b+0*bsize+off,svadd_f64_x(pg,x0,x3),nt);
                put(pg,all_b+1*bsize+off,x0,nt);
                put(pg,all_b+2*bsize+off,svsub_f64_x(pg,x1,x3),nt);
                put(pg,all_b+3*bsize+off,svsub_f64_x(pg,x2,x0),nt);
                put(pg,all_b+4*bsize+off,x3,nt);
                put(pg,all_b+5*bsize+off,svadd_f64_x(pg,x0,x1),nt);
                put(pg,all_b+6*bsize+off,svadd_f64_x(pg,x2,x3),nt);
            }
            sme_stop();
        }
#pragma omp master
        b_end=omp_get_wtime();
#pragma omp for schedule(dynamic,1)
        for(long task=0;task<7*ntasks;task++){
            long product=interleave?task%7:task/ntasks,local=interleave?task/7:task%ntasks;
            long ib=order?local%blocks:local/np,p=order?local/blocks:local%np;
            long rows=mh-ib*mb,cols=nh-p*nc;if(rows>mb)rows=mb;if(cols>nc)cols=nc;
            long S=rows/16,NT=cols/32;
            const double *apack=all_a+product*asize,*bpack=all_b+product*bsize;
            double* cb=work_c+(size_t)tid*mb*nc;
            memset(cb,0,(size_t)mb*nc*8);
            double* pending=NULL;
            sme_start();sme_zero_za();
            for(long ch=0;ch<chunks;ch++){
                const double* Bch=bpack+(ch*np+p)*kc*nc;
                const double* Bnext=ch+1<chunks?bpack+((ch+1)*np+p)*kc*nc:NULL;
                for(long r=0;r<S;r++){
                    const double* A=apack+(ib*chunks*(mb/16)+ch*(mb/16)+r)*1024;
                    const double* Anext=r+1<S?A+1024:ch+1<chunks?apack+(ib*chunks*(mb/16)+(ch+1)*(mb/16))*1024:NULL;
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
            emit(cb,all_m+product*msize+local*mb*nc,rows,cols,nc,1,1);
            sme_stop();
        }
#pragma omp master
        gemm_end=omp_get_wtime();
#pragma omp for schedule(static)
        for(long local=0;local<ntasks;local++){
            long ib=order?local%blocks:local/np,p=order?local/blocks:local%np;
            long rows=mh-ib*mb,cols=nh-p*nc;if(rows>mb)rows=mb;if(cols>nc)cols=nc;
            double *c11=c+ib*mb*ldc+p*nc,*c12=c11+nh,*c21=c11+mh*ldc,*c22=c21+nh;
            sme_start();svbool_t pg=svptrue_b64();
            for(long i=0;i<rows;i++)for(long j=0;j<cols;j+=8){
                size_t off=(size_t)local*mb*nc+i*nc+j;
                svfloat64_t m1=svld1_f64(pg,all_m+0*msize+off),m2=svld1_f64(pg,all_m+1*msize+off);
                svfloat64_t m3=svld1_f64(pg,all_m+2*msize+off),m4=svld1_f64(pg,all_m+3*msize+off);
                svfloat64_t m5=svld1_f64(pg,all_m+4*msize+off),m6=svld1_f64(pg,all_m+5*msize+off),m7=svld1_f64(pg,all_m+6*msize+off);
                svfloat64_t y11=svadd_f64_x(pg,svsub_f64_x(pg,svadd_f64_x(pg,m1,m4),m5),m7);
                svfloat64_t y12=svadd_f64_x(pg,m3,m5),y21=svadd_f64_x(pg,m2,m4);
                svfloat64_t y22=svadd_f64_x(pg,svadd_f64_x(pg,svsub_f64_x(pg,m1,m2),m3),m6);
                if(alpha<0){y11=svneg_f64_x(pg,y11);y12=svneg_f64_x(pg,y12);y21=svneg_f64_x(pg,y21);y22=svneg_f64_x(pg,y22);}
                if(beta){y11=svadd_f64_x(pg,y11,svld1_f64(pg,c11+i*ldc+j));y12=svadd_f64_x(pg,y12,svld1_f64(pg,c12+i*ldc+j));
                         y21=svadd_f64_x(pg,y21,svld1_f64(pg,c21+i*ldc+j));y22=svadd_f64_x(pg,y22,svld1_f64(pg,c22+i*ldc+j));}
                put(pg,c11+i*ldc+j,y11,outnt);put(pg,c12+i*ldc+j,y12,outnt);
                put(pg,c21+i*ldc+j,y21,outnt);put(pg,c22+i*ldc+j,y22,outnt);
            }
            sme_stop();
        }
    }
    double end=omp_get_wtime();
    stats.setup_ms=1e3*(allocated-begin);stats.apack_ms=1e3*(a_end-allocated);
    stats.bpack_ms=1e3*(b_end-a_end);stats.compute_ms=1e3*(end-b_end);
    compose_ms=1e3*(end-gemm_end);
}
