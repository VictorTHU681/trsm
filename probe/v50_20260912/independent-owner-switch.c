#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void l_trsm(int,int,const double*,int,double*,int);
static uint64_t state=UINT64_C(0x72b90fa6873d21c5);
static double sample(void) {
    state^=state>>12; state^=state<<25; state^=state>>27;
    return (double)((state*UINT64_C(2685821657736338717))>>11)*0x1p-53-0.5;
}
int main(void) {
    const int shapes[][2]={{64,96},{128,192},{192,288},{256,384},{320,96},{384,192},{512,288},{768,96}};
    int cases=0,failures=0;
    double worst=0;
    for(unsigned sh=0;sh<sizeof(shapes)/sizeof(shapes[0]);sh++) {
        int m=shapes[sh][0],n=shapes[sh][1],lda=m+19,ldb=n+13;
        size_t ls=(size_t)m*lda*8,bs=(size_t)m*ldb*8;
        double *lr=malloc(ls+64),*br=malloc(bs+64),*saved=malloc(ls),*x=malloc(bs);
        if(!lr||!br||!saved||!x)return 2;
        double *L=lr+sh,*B=br+(7-sh);
        for(int mode=0;mode<4;mode++)for(int reuse=0;reuse<2;reuse++) {
            for(int i=0;i<m;i++) {
                double scale=mode==1?1e100:mode==2?1e-100:mode==3?ldexp(1.0,i%61-30):1.0;
                double norm=0;
                for(int k=0;k<lda;k++) {
                    double a=k<i?sample()*(mode==3?0.04:8.0):0;
                    L[(size_t)i*lda+k]=a*scale;
                    norm+=fabs(a);
                }
                L[(size_t)i*lda+i]=(reuse?-1:1)*(mode==3?2.0:norm+1.0)*scale;
                for(int j=0;j<ldb;j++) {
                    x[(size_t)i*ldb+j]=sample();
                    B[(size_t)i*ldb+j]=37.125;
                }
            }
            // Independent known solution; form the RHS with long-double sums.
            for(int i=0;i<m;i++)for(int j=0;j<n;j++) {
                long double sum=0;
                for(int k=0;k<=i;k++)sum+=(long double)L[(size_t)i*lda+k]*x[(size_t)k*ldb+j];
                B[(size_t)i*ldb+j]=(double)sum;
            }
            memcpy(saved,L,ls);
            setenv("TRSM_OWNER",reuse?"0":"1",1);
            setenv("TRSM_EXPECT_OWNER",reuse?"0":"1",1);
            l_trsm(m,n,L,lda,B,ldb);
            int bad=memcmp(saved,L,ls)!=0;double err=0;
            for(int i=0;i<m;i++)for(int j=0;j<ldb;j++) {
                double a=B[(size_t)i*ldb+j];
                if(j>=n) {if(a!=37.125)bad=1;continue;}
                double e=fabs(a-x[(size_t)i*ldb+j]);
                if(!isfinite(e)||e>1e-12)bad=1;
                err=fmax(err,e);
            }
            worst=fmax(worst,err);failures+=bad;cases++;
            printf("INDEPENDENT m=%d n=%d mode=%d reuse=%d maxerr=%.3e %s\n",m,n,mode,reuse,err,bad?"FAIL":"PASS");
        }
        free(lr);free(br);free(saved);free(x);
    }
    printf("INDEPENDENT_TOTAL cases=%d failures=%d maxerr=%.3e\n",cases,failures,worst);
    return failures?1:0;
}
