#define _GNU_SOURCE
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "fused.h"

static double value(uint64_t x){
    x+=UINT64_C(0x9e3779b97f4a7c15);
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);x^=x>>31;
    return (double)(x>>11)*0x1p-53-0.5;
}
static double* buffer(size_t count){
    size_t alignment=(size_t)2<<20,bytes=(count*8+alignment-1)&~(alignment-1);
    double* p=aligned_alloc(alignment,bytes);if(!p)exit(2);
    madvise(p,bytes,MADV_HUGEPAGE);return p;
}
static uint64_t fingerprint(const double* a,size_t count){
    uint64_t hash=0;
#pragma omp parallel for reduction(^:hash)
    for(size_t i=0;i<count;i++){
        uint64_t bits;memcpy(&bits,a+i,8);
        hash^=(bits+UINT64_C(0x9e3779b97f4a7c15)*(i+1))*UINT64_C(0xbf58476d1ce4e5b9);
    }
    return hash;
}
static int check(int m,int n,int k,const double* a,int lda,const double* b,int ldb,
                 const double* c,int ldc,int alpha,int beta,int rep,double* worst){
    int ri[32],ci[32],nr=0,nc=0;
    const int edge[]={0,1,7,8,15,16,31,32,63,64,95,96,127,128,191,192,255,256};
    for(int t=0;t<18;t++){if(edge[t]<m)ri[nr++]=edge[t];if(edge[t]<n)ci[nc++]=edge[t];}
    const int tail[]={1,8,16,32,64};
    for(int t=0;t<5;t++){if(tail[t]<=m)ri[nr++]=m-tail[t];if(tail[t]<=n)ci[nc++]=n-tail[t];}
    ri[nr++]=m/2;ri[nr++]=(m-1)/2;ci[nc++]=n/2;ci[nc++]=(n-1)/2;
    int full=m<=384&&n<=384;
    long count=full?(long)m*n:128+nr*nc;
    for(long q=0;q<count;q++){
        int i=full?q/n:(q<128?(q*47+3)%m:ri[(q-128)/nc]);
        int j=full?q%n:(q<128?(q*131+7)%n:ci[(q-128)%nc]);
        long double ref=0,norm=0;
        for(int p=0;p<k;p++){
            long double t=(long double)a[(size_t)i*lda+p]*b[(size_t)p*ldb+j];ref+=t;norm+=fabsl(t);
        }
        double initial=value((uint64_t)i*n+j+UINT64_C(918271)+UINT64_C(104729)*(rep+1));
        ref=alpha*ref+(beta?initial:0);
        double error=fabs(c[(size_t)i*ldc+j]-(double)ref);
        if(error>*worst)*worst=error;
        if(!isfinite(error)||error>128*DBL_EPSILON*(double)(norm+fabsl(initial))+1e-14){
            printf("FUSED_CHECK FAIL i=%d j=%d actual=%.17g ref=%.17Lg error=%.17g\n",i,j,c[(size_t)i*ldc+j],ref,error);return 1;
        }
    }
    for(int i=0;i<m;i++)for(int j=n;j<ldc;j++)if(c[(size_t)i*ldc+j]!=0x1.deadbeefp+400){
        printf("FUSED_PADDING FAIL i=%d j=%d\n",i,j);return 1;
    }
    return 0;
}
int main(int argc,char**argv){
    int m=argc>1?atoi(argv[1]):64,n=argc>2?atoi(argv[2]):192,k=argc>3?atoi(argv[3]):176;
    int reps=argc>4?atoi(argv[4]):2,mode=argc>5?atoi(argv[5]):2;
    int alpha=argc>6?atoi(argv[6]):1,beta=argc>7?atoi(argv[7]):0,pad=argc>8?atoi(argv[8]):64;
    int exponent=argc>9?atoi(argv[9]):0;
    if(m<=0||n<=0||k<=0||reps<1||mode<0||mode>2||pad<0)return 2;
    int lda=k+pad,ldb=n+pad+1,ldc=n+pad+3;
    double *a=buffer((size_t)m*lda),*b=buffer((size_t)k*ldb),*c=buffer((size_t)m*ldc);
    for(int rep=-1;rep<reps;rep++){
#pragma omp parallel for schedule(static)
        for(long i=0;i<(long)m*lda;i++)a[i]=ldexp(value(i+19+UINT64_C(104729)*(rep+1)),exponent);
#pragma omp parallel for schedule(static)
        for(long i=0;i<(long)k*ldb;i++)b[i]=ldexp(value(i+UINT64_C(12491257)+UINT64_C(15485863)*(rep+1)),-exponent);
        uint64_t ah=fingerprint(a,(size_t)m*lda),bh=fingerprint(b,(size_t)k*ldb);
        for(int pass=0;pass<(mode==2?2:1);pass++){
            int level=mode==2?((rep&1)?1-pass:pass):mode;
#pragma omp parallel for schedule(static)
            for(int i=0;i<m;i++)for(int j=0;j<ldc;j++)c[(size_t)i*ldc+j]=j<n?value((uint64_t)i*n+j+UINT64_C(918271)+UINT64_C(104729)*(rep+1)):0x1.deadbeefp+400;
            double begin=omp_get_wtime();
            fused_product(m,n,k,a,lda,b,ldb,c,ldc,alpha,beta,level);
            double seconds=omp_get_wtime()-begin,worst=0;
            struct fused_stats st=fused_last_stats();
            int expected=m%16||n%32?-1:level&&m%32==0&&n%64==0&&k%2==0?1:0;
            if(st.level!=expected||st.leaves!=(expected<0?0:expected?7:1)){
                printf("FUSED_PATH FAIL expected=%d actual=%d leaves=%ld\n",expected,st.level,st.leaves);return 1;
            }
            if(check(m,n,k,a,lda,b,ldb,c,ldc,alpha,beta,rep,&worst))return 1;
            if(ah!=fingerprint(a,(size_t)m*lda)||bh!=fingerprint(b,(size_t)k*ldb)){
                puts("FUSED_INPUT_CHANGED FAIL");return 1;
            }
            printf("FUSED m=%d n=%d k=%d rep=%d level=%d actual=%d alpha=%d beta=%d pad=%d exp=%d total_ms=%.6f setup_ms=%.6f apack_ms=%.6f bpack_ms=%.6f compute_ms=%.6f nominal_GFLOPS=%.4f leaves=%ld max_error=%.3e coverage=%s PASS\n",
                   m,n,k,rep,level,st.level,alpha,beta,pad,exponent,seconds*1e3,st.setup_ms,st.apack_ms,st.bpack_ms,st.compute_ms,2.0*m*n*k/seconds*1e-9,st.leaves,worst,m<=384&&n<=384?"full":"samples-plus-edges");
        }
    }
    free(a);free(b);free(c);return 0;
}
