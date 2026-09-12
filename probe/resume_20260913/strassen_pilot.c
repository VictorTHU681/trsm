// Component experiment, not a TRSM implementation or a submission result.
// Classical seven-product Strassen, with explicit temporary matrices to measure costs.
#define _GNU_SOURCE
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include "kblas.h"

typedef struct { double *s, *t, *u; } level_workspace;
static level_workspace ws[3];
static double leaf_seconds, sum_seconds, output_seconds;

static double* buffer(size_t elements)
{
    size_t align=(size_t)2<<20, bytes=(elements*8+align-1)&~(align-1);
    double* p=aligned_alloc(align,bytes);
    if(!p)exit(2);
    madvise(p,bytes,MADV_HUGEPAGE);
    return p;
}
static double value(uint64_t i)
{
    i+=UINT64_C(0x9e3779b97f4a7c15);
    i=(i^(i>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    i=(i^(i>>27))*UINT64_C(0x94d049bb133111eb);
    i^=i>>31;
    return (double)(i>>11)*0x1p-53-0.5;
}
static void add_matrix(int m,int n,const double* a,int lda,const double* b,int ldb,double sign,double* out)
{
    double t=omp_get_wtime();
#pragma omp parallel for schedule(static)
    for(int i=0;i<m;i++)for(int j=0;j<n;j++)out[(size_t)i*n+j]=a[(size_t)i*lda+j]+sign*b[(size_t)i*ldb+j];
    sum_seconds+=omp_get_wtime()-t;
}
static void accumulate(int m,int n,const double* u,double* c,int ldc,double sign)
{
    double t=omp_get_wtime();
#pragma omp parallel for schedule(static)
    for(int i=0;i<m;i++)for(int j=0;j<n;j++)c[(size_t)i*ldc+j]+=sign*u[(size_t)i*n+j];
    output_seconds+=omp_get_wtime()-t;
}
static void product(int m,int n,int k,const double* a,int lda,const double* b,int ldb,double* c,int ldc,int depth,int level)
{
    if(!depth || ((m|n|k)&1)){
        double t=omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a,lda,b,ldb,0,c,ldc);
        leaf_seconds+=omp_get_wtime()-t;
        return;
    }
    int mh=m/2,nh=n/2,kh=k/2;
    const double *a11=a,*a12=a+kh,*a21=a+(size_t)mh*lda,*a22=a21+kh;
    const double *b11=b,*b12=b+nh,*b21=b+(size_t)kh*ldb,*b22=b21+nh;
    double *c11=c,*c12=c+nh,*c21=c+(size_t)mh*ldc,*c22=c21+nh;
    double *s=ws[level].s,*t=ws[level].t,*u=ws[level].u;
    double tick=omp_get_wtime();
#pragma omp parallel for schedule(static)
    for(int i=0;i<m;i++)for(int j=0;j<n;j++)c[(size_t)i*ldc+j]=0;
    output_seconds+=omp_get_wtime()-tick;
    // M1=(A11+A22)(B11+B22).
    add_matrix(mh,kh,a11,lda,a22,lda,1,s);add_matrix(kh,nh,b11,ldb,b22,ldb,1,t);
    product(mh,nh,kh,s,kh,t,nh,u,nh,depth-1,level+1);
    accumulate(mh,nh,u,c11,ldc,1);accumulate(mh,nh,u,c22,ldc,1);
    // M2=(A21+A22)B11.
    add_matrix(mh,kh,a21,lda,a22,lda,1,s);
    product(mh,nh,kh,s,kh,b11,ldb,u,nh,depth-1,level+1);
    accumulate(mh,nh,u,c21,ldc,1);accumulate(mh,nh,u,c22,ldc,-1);
    // M3=A11(B12-B22).
    add_matrix(kh,nh,b12,ldb,b22,ldb,-1,t);
    product(mh,nh,kh,a11,lda,t,nh,u,nh,depth-1,level+1);
    accumulate(mh,nh,u,c12,ldc,1);accumulate(mh,nh,u,c22,ldc,1);
    // M4=A22(B21-B11).
    add_matrix(kh,nh,b21,ldb,b11,ldb,-1,t);
    product(mh,nh,kh,a22,lda,t,nh,u,nh,depth-1,level+1);
    accumulate(mh,nh,u,c11,ldc,1);accumulate(mh,nh,u,c21,ldc,1);
    // M5=(A11+A12)B22.
    add_matrix(mh,kh,a11,lda,a12,lda,1,s);
    product(mh,nh,kh,s,kh,b22,ldb,u,nh,depth-1,level+1);
    accumulate(mh,nh,u,c11,ldc,-1);accumulate(mh,nh,u,c12,ldc,1);
    // M6=(A21-A11)(B11+B12).
    add_matrix(mh,kh,a21,lda,a11,lda,-1,s);add_matrix(kh,nh,b11,ldb,b12,ldb,1,t);
    product(mh,nh,kh,s,kh,t,nh,u,nh,depth-1,level+1);
    accumulate(mh,nh,u,c22,ldc,1);
    // M7=(A12-A22)(B21+B22).
    add_matrix(mh,kh,a12,lda,a22,lda,-1,s);add_matrix(kh,nh,b21,ldb,b22,ldb,1,t);
    product(mh,nh,kh,s,kh,t,nh,u,nh,depth-1,level+1);
    accumulate(mh,nh,u,c11,ldc,1);
}
static int validate(int m,int n,int k,const double*a,const double*b,const double*c,int full,double* worst)
{
    long checks=full?(long)m*n:128;
    for(long q=0;q<checks;q++){
        int i=full?q/n:(q*47+3)%m,j=full?q%n:(q*131+7)%n;
        long double ref=0,norm=0;
        for(int p=0;p<k;p++){
            long double term=(long double)a[(size_t)i*k+p]*b[(size_t)p*n+j];
            ref+=term;norm+=fabsl(term);
        }
        double error=fabs(c[(size_t)i*n+j]-(double)ref);
        if(error>*worst)*worst=error;
        if(!isfinite(error)||error>128*DBL_EPSILON*(double)norm+1e-14){
            printf("STRASSEN_CHECK FAIL i=%d j=%d error=%.17g norm=%.17Lg\n",i,j,error,norm);return 1;
        }
    }
    return 0;
}
int main(int argc,char**argv)
{
    int m=argc>1?atoi(argv[1]):64,n=argc>2?atoi(argv[2]):96,k=argc>3?atoi(argv[3]):80;
    int reps=argc>4?atoi(argv[4]):2;
    if(m<=0||n<=0||k<=0||reps<=0)return 2;
    double *a=buffer((size_t)m*k),*b=buffer((size_t)k*n),*c=buffer((size_t)m*n);
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)m*k;i++)a[i]=value(i+23879);
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)k*n;i++)b[i]=value(i+UINT64_C(127421711));
    for(int level=0;level<3;level++){
        size_t div=(size_t)1<<(2*(level+1));
        ws[level].s=buffer((size_t)m*k/div+32);
        ws[level].t=buffer((size_t)k*n/div+32);
        ws[level].u=buffer((size_t)m*n/div+32);
    }
    for(int rep=-1;rep<reps;rep++)for(int order=0;order<3;order++){
        int depth=(rep&1)?2-order:order;
        leaf_seconds=sum_seconds=output_seconds=0;
        double tick=omp_get_wtime();
        product(m,n,k,a,k,b,n,c,n,depth,0);
        double seconds=omp_get_wtime()-tick,worst=0;
        if(validate(m,n,k,a,b,c,m<=128&&n<=128,&worst))return 1;
        printf("STRASSEN m=%d n=%d k=%d depth=%d rep=%d total_ms=%.6f leaf_ms=%.6f sum_ms=%.6f output_ms=%.6f nominal_GFLOPS=%.4f error=%.3e coverage=%s PASS\n",
               m,n,k,depth,rep,1e3*seconds,1e3*leaf_seconds,1e3*sum_seconds,1e3*output_seconds,
               2.0*m*n*k/seconds*1e-9,worst,m<=128&&n<=128?"full":"128-samples");
    }
    for(int level=0;level<3;level++){free(ws[level].s);free(ws[level].t);free(ws[level].u);}
    free(a);free(b);free(c);return 0;
}
