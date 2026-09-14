// Same independent product and input generation as digits.c, FP64 control.
#define main digit_main
#include "digits.c"
#undef main
#include "fullinv-compact.h"
static void pack_fp(const double*x,double*d,int rows,int K,int KC,int nr) {
    #pragma omp parallel for schedule(static)
    for(int task=0;task<(rows/16)*(K/KC);task++) {
        int r=task/(K/KC)*16,k0=task%(K/KC)*KC;
        _Alignas(64) double tmp[128];
        for(int k=k0;k<k0+KC;k+=8) {
            pack_strip(x,K,rows,r,k,8,tmp);
            for(int kk=0;kk<8;kk++)for(int j=0;j<16;j+=8) {
                size_t dst=(size_t)(k/KC)*rows*KC+(size_t)(r/nr)*nr*KC+(size_t)(k%KC+kk)*nr+r%nr+j;
                svbool_t pg=svptrue_b64();svst1_f64(pg,d+dst,svld1_f64(pg,tmp+kk*16+j));
            }
        }
    }
}
static void multiply_fp(const double*A,const double*B,double*C,int M,int N,int K,int KC) {
    const int MB=M<128?M:128,NC=N<128?N:128;
    #pragma omp parallel
    {
        _Alignas(64) double acc[128*128];
        #pragma omp for schedule(dynamic,1)
        for(int task=0;task<(M/MB)*(N/NC);task++) {
            int r0=task/(N/NC)*MB,c0=task%(N/NC)*NC;
            sme_start();
            for(int k=0;k<K;k+=KC)for(int r=0;r<MB;r+=16)for(int j=0;j<NC;j+=32) {
                if(!k)sme_zero_za();else za_load_direct(acc+r*NC+j,NC*8,32);
                inv_compact(A+(size_t)(k/KC)*M*KC+(r0+r)*KC,B+(size_t)(k/KC)*N*KC+(c0+j)*KC,KC);
                za_store_direct(acc+r*NC+j,NC*8,32);
            }
            for(int r=0;r<MB;r++)for(int j=0;j<NC;j+=32)inv_copy32(acc+r*NC+j,C+(size_t)(r0+r)*N+c0+j);
            sme_stop();
        }
    }
}
int main(int argc,char**argv) {
    int M=argc>1?atoi(argv[1]):512,N=M,K=2048;
    if((M!=32&&M!=512)||svcntd()!=8)return 2;
    double*A=aligned_alloc(64,(size_t)M*K*8),*B=aligned_alloc(64,(size_t)N*K*8),*C=aligned_alloc(64,(size_t)M*N*8);
    double*ap=aligned_alloc(64,(size_t)M*K*8),*bp=aligned_alloc(64,(size_t)N*K*8);
    if(!A||!B||!C||!ap||!bp)return 2;
    for(int i=0;i<M*K;i++)A[i]=random53();for(int i=0;i<N*K;i++)B[i]=random53();
    int count=M==32?1024:64;double ref[1024];
    for(int z=0;z<count;z++) {
        int i=M==32?z/32:(z*73)%M,j=M==32?z%32:(z*121+z/17)%N;long double sum=0;
        for(int k=0;k<K;k++)sum+=(long double)A[(size_t)i*K+k]*B[(size_t)j*K+k];ref[z]=(double)sum;
    }
    for(int KC=64;KC<=256;KC*=2) {
        for(int rep=-1;rep<3;rep++) {
            double t=omp_get_wtime();pack_fp(A,ap,M,K,KC,16);pack_fp(B,bp,N,K,KC,32);double t1=omp_get_wtime();
            multiply_fp(ap,bp,C,M,N,K,KC);double t2=omp_get_wtime();
            double err=0;
            for(int z=0;z<count;z++) {
                int i=M==32?z/32:(z*73)%M,j=M==32?z%32:(z*121+z/17)%N;double e=fabs(C[(size_t)i*N+j]-ref[z]);
                if(!isfinite(e)||e>1e-10)return 1;err=fmax(err,e);
            }
            printf("FP64_PRODUCT M=%d K=%d KC=%d rep=%d outputs=%d maxerr=%.3e pack_ms=%.6f product_ms=%.6f total_ms=%.6f total_GFLOPS=%.3f PASS\n",M,K,KC,rep,count,err,(t1-t)*1e3,(t2-t1)*1e3,(t2-t)*1e3,2.0*M*N*K/(t2-t)*1e-9);
        }
    }
    return 0;
}
