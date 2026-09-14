// Independent matrix-product research; not a TRSM implementation or submission.
#include "trsm-v55.c"
#include <float.h>
#include "digit-kernels.h"
typedef struct { int p,bits,top,karatsuba; } config;
static size_t offset(int rows,int K,int KC,int r,int k) {
    return (size_t)(k/KC)*rows*KC+(size_t)(r/32)*32*KC+(size_t)((k%KC)/4)*128+(r%32)*4+k%4;
}
static void encode(const double* x,int8_t*d,int rows,int K,int KC,config c) {
    const size_t plane=(size_t)rows*K;
    const double scale=ldexp(1.0,c.top+c.bits*(c.p-1));
    #pragma omp parallel for schedule(static)
    for(int task=0;task<(rows/32)*(K/KC);task++) {
        int r0=task/(K/KC)*32,k0=task%(K/KC)*KC;
        svbool_t pg=svptrue_b64();
        for(int r=r0;r<r0+32;r++)for(int k=k0;k<k0+KC;k+=8) {
            svfloat64_t f=svrintn_f64_x(pg,svmul_n_f64_x(pg,svld1_f64(pg,x+(size_t)r*K+k),scale));
            svint64_t q=svcvt_s64_f64_x(pg,f);
            size_t dst=offset(rows,K,KC,r,k);
            for(int p=c.p-1;p>=0;p--) {
                svint64_t digit=q;
                if(p) {
                    svint64_t shifted=svadd_n_s64_x(pg,q,1L<<(c.bits-1));
                    digit=svsub_n_s64_x(pg,svand_n_s64_x(pg,shifted,(1L<<c.bits)-1),1L<<(c.bits-1));
                    q=c.bits==8?svasr_n_s64_x(pg,shifted,8):svasr_n_s64_x(pg,shifted,7);
                }
                int8_t temp[8];svst1b_s64(pg,temp,digit);
                memcpy(d+(size_t)p*plane+dst,temp,4);
                memcpy(d+(size_t)p*plane+dst+128,temp+4,4);
            }
        }
    }
}
static inline __attribute__((always_inline)) void accum(const int32_t*tmp,double*C,int ld,double scale) {
    svbool_t pg=svptrue_b64();
    for(int r=0;r<32;r++)for(int j=0;j<32;j+=8) {
        svfloat64_t v=svcvt_f64_s64_x(pg,svld1sw_s64(pg,tmp+32*r+j));
        svfloat64_t old=svld1_f64(pg,C+(size_t)r*ld+j);
        svst1_f64(pg,C+(size_t)r*ld+j,svmla_n_f64_x(pg,old,v,scale));
    }
}
static void product(const int8_t*A,const int8_t*B,double*C,int M,int N,int K,int KC,config c) {
    double weights[8],levels[8];
    for(int i=0;i<c.p;i++) {
        levels[i]=ldexp(1.0,-2*c.top-c.bits*i);
        weights[i]=2*i<c.p?ldexp(1.0,-c.bits*2*i):0;
        for(int j=0;j<c.p;j++)if(i!=j&&i+j<c.p)weights[i]-=ldexp(1.0,-c.bits*(i+j));
        weights[i]=ldexp(weights[i],-2*c.top);
    }
    #pragma omp parallel
    {
        _Alignas(64) int32_t tmp[1024];
        #pragma omp for schedule(dynamic,1)
        for(int task=0;task<(M/32)*(N/32);task++) {
            int i=task/(N/32)*32,j=task%(N/32)*32;
            double*dest=C+(size_t)i*N+j;
            sme_start();
            svbool_t pg=svptrue_b64();
            for(int r=0;r<32;r++)for(int col=0;col<32;col+=8)svst1_f64(pg,dest+(size_t)r*N+col,svdup_f64(0));
            for(int k=0;k<K;k+=KC) {
                const int8_t*ap=A+offset(M,K,KC,i,k),*bp=B+offset(N,K,KC,j,k);
                if(!c.karatsuba) {
                    for(int level=0;level<c.p;level++) {
                        sme_zero_za();
                        for(int u=0;u<=level;u++)digit_dot(ap+(size_t)u*M*K,bp+(size_t)(level-u)*N*K,KC);
                        digit_dump(tmp);accum(tmp,dest,N,levels[level]);
                    }
                } else {
                    for(int u=0;u<c.p;u++) {
                        sme_zero_za();digit_dot(ap+(size_t)u*M*K,bp+(size_t)u*N*K,KC);
                        digit_dump(tmp);accum(tmp,dest,N,weights[u]);
                    }
                    for(int u=0;u<c.p;u++)for(int v=u+1;v<c.p-u;v++) {
                        sme_zero_za();digit_sumdot(ap+(size_t)u*M*K,bp+(size_t)u*N*K,KC,ap+(size_t)v*M*K,bp+(size_t)v*N*K);
                        digit_dump(tmp);accum(tmp,dest,N,levels[u+v]);
                    }
                }
            }
            sme_stop();
        }
    }
}
static uint64_t rng=UINT64_C(0x697914eba8cc7311);
static double random53(void) {
    rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;
    return (double)((rng*UINT64_C(2685821657736338717))>>11)*0x1p-53-0.5;
}
static double bound(config c,int K) {
    long double delta=ldexpl(0.5L,-c.top-c.bits*(c.p-1)),tail=0;
    for(int u=0;u<c.p;u++)for(int v=0;v<c.p;v++)if(u+v>=c.p)
        tail+=ldexpl(1.0L,2*(c.bits-1)-2*c.top-c.bits*(u+v));
    return K*(double)(delta+delta*delta+tail)+64*DBL_EPSILON*K;
}
int main(int argc,char**argv) {
    int M=argc>1?atoi(argv[1]):32,N=M,K=argc>2?atoi(argv[2]):512,mode=argc>3?atoi(argv[3]):0;
    if(M%32||K%2048&&K!=512||svcntd()!=8)return 2;
    double*A=aligned_alloc(64,(size_t)M*K*8),*B=aligned_alloc(64,(size_t)N*K*8),*C=aligned_alloc(64,(size_t)M*N*8);
    int8_t*ad=aligned_alloc(64,(size_t)7*M*K),*bd=aligned_alloc(64,(size_t)7*N*K);
    if(!A||!B||!C||!ad||!bd)return 2;
    for(int i=0;i<M*K;i++)A[i]=mode==1?0.499999999999:mode==2?(i%2?0.499999999999:-0.499999999999):random53();
    for(int i=0;i<N*K;i++)B[i]=mode==1?0.499999999999:mode==2?(i%3?0.499999999999:-0.499999999999):random53();
    int count=M==32?1024:64,fail=0;double ref[1024];
    for(int z=0;z<count;z++) {
        int i=M==32?z/32:(z*73)%M,j=M==32?z%32:(z*121+z/17)%N;long double s=0;
        for(int k=0;k<K;k++)s+=(long double)A[(size_t)i*K+k]*B[(size_t)j*K+k];ref[z]=(double)s;
    }
    config configs[]={{5,8,7,0},{6,8,7,0},{7,7,6,1}};
    for(int cfg=0;cfg<3;cfg++)for(int KC=128;KC<=2048&&KC<=K;KC*=2) {
        config c=configs[cfg];double enc=omp_get_wtime();
        encode(A,ad,M,K,KC,c);encode(B,bd,N,K,KC,c);enc=omp_get_wtime()-enc;
        product(ad,bd,C,M,N,K,KC,c);
        double error=0;
        for(int z=0;z<count;z++) {
            int i=M==32?z/32:(z*73)%M,j=M==32?z%32:(z*121+z/17)%N;
            double e=fabs(C[(size_t)i*N+j]-ref[z]);if(!isfinite(e))e=INFINITY;error=fmax(error,e);
        }
        double limit=bound(c,K);int bad=!(error<=limit);fail+=bad;
        printf("DIGIT_CHECK M=%d K=%d KC=%d p=%d bits=%d karatsuba=%d input=%d outputs=%d maxerr=%.3e declared_bound=%.3e %s\n",M,K,KC,c.p,c.bits,c.karatsuba,mode,count,error,limit,bad?"FAIL":"PASS");fflush(stdout);
        for(int rep=0;rep<3;rep++) {
            double t=omp_get_wtime();encode(A,ad,M,K,KC,c);encode(B,bd,N,K,KC,c);double t1=omp_get_wtime();
            product(ad,bd,C,M,N,K,KC,c);double t2=omp_get_wtime();
            printf("DIGIT_PERF M=%d K=%d KC=%d p=%d bits=%d karatsuba=%d rep=%d encode_ms=%.6f product_ms=%.6f total_ms=%.6f product_GFLOPS=%.3f total_GFLOPS=%.3f\n",M,K,KC,c.p,c.bits,c.karatsuba,rep,1e3*(t1-t),1e3*(t2-t1),1e3*(t2-t),2.0*M*N*K/(t2-t1)*1e-9,2.0*M*N*K/(t2-t)*1e-9);
        }
    }
    return fail?1:0;
}
