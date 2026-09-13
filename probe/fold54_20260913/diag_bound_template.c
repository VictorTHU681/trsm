#include "trsm-diag-order.c"
#include "diag-reference.h"
#include <float.h>
static double sample(uint64_t *state)
{
    uint64_t x=*state; x^=x>>12;x^=x<<25;x^=x>>27;*state=x;
    return (double)((x*UINT64_C(2685821657736338717))>>11)*0x1p-53-0.5;
}
int main(void)
{
    const int dims[]={1,15,16,17,31,32,63,64,191,192,255,256};
    const int m=320,lda=339,offset=32;
    double *L=malloc((size_t)m*lda*8),*actual=malloc(4128*8),*ref=malloc(4128*8);
    uint64_t seed=UINT64_C(98127361829);long count[4]={0};double worst[4]={0};
    for(int scale=-400;scale<=400;scale+=400)for(int sign=-1;sign<=1;sign+=2){
        for(int i=0;i<m;i++){
            double norm=0;
            for(int j=0;j<lda;j++){
                double x=j<i?sample(&seed):0;norm+=fabs(x);L[(size_t)i*lda+j]=ldexp(x,scale);
            }
            L[(size_t)i*lda+i]=sign*ldexp(norm+1,scale);
        }
        for(int shape=0;shape<12;shape++)for(int strip=0;strip<(dims[shape]+15)/16;strip++){
            int nvalues=256*(strip+1);
            diag_prep_reference(L,lda,offset,dims[shape],strip,ref);
            for(int mode=0;mode<4;mode++){
                g_diag_order=mode;
                for(int q=0;q<4128;q++)actual[q]=0x1.deadbeefp+500;
                diag_prep_sub(L,lda,offset,dims[shape],strip,actual);
                for(int q=0;q<nvalues;q++){
                    double err=fabs(actual[q]-ref[q]);
                    double relative=err/(fabs(ref[q])+1e-300);
                    if(relative>worst[mode])worst[mode]=relative;
                    int row=q%16,col=q/16,vr=dims[shape]-strip*16;
                    if(vr>16)vr=16;
                    long r0=offset+16*strip;
                    long double norm=0;
                    if(col<16*strip){
                        for(int t=0;t<vr;t++)
                            norm+=fabsl((long double)ref[256*strip+t*16+row]*L[(r0+t)*lda+offset+col]);
                    }else{
                        int invcol=col-16*strip;
                        if(row>=vr)norm=row==invcol?1:0;
                        else{
                            norm=row==invcol?1:0;
                            for(int t=0;t<row;t++)
                                norm+=fabsl((long double)L[(r0+row)*lda+r0+t]*ref[256*strip+invcol*16+t]);
                            norm/=fabsl(L[(r0+row)*lda+r0+row]);
                        }
                    }
                    if(!isfinite(actual[q]) || err>256*DBL_EPSILON*(double)norm+1e-300){
                        printf("DIAG_ORDER FAIL mode=%d shape=%d strip=%d q=%d actual=%.17g ref=%.17g err=%.3e\n",mode,dims[shape],strip,q,actual[q],ref[q],err);return 1;
                    }
                    count[mode]++;
                }
                for(int q=nvalues;q<4128;q++)if(actual[q]!=0x1.deadbeefp+500){puts("DIAG_ORDER_PADDING FAIL");return 1;}
            }
        }
    }
    for(int mode=0;mode<4;mode++)printf("DIAG_BOUND_CHECK mode=%d comparisons=%ld worst_relative=%.3e PASS\n",mode,count[mode],worst[mode]);
    // A component timing only: every call repeats the complete preprocessing.
    for(int round=0;round<4;round++)for(int pass=0;pass<4;pass++){
        int mode=round%2?3-pass:pass;g_diag_order=mode;
        double begin=omp_get_wtime();
        for(int repeat=0;repeat<128;repeat++)for(int strip=0;strip<16;strip++)
            diag_prep_sub(L,lda,offset,256,strip,actual);
        printf("DIAG_ORDER_MICRO round=%d mode=%d ms=%.6f calls=2048\n",round,mode,1e3*(omp_get_wtime()-begin));
    }
    return 0;
}
