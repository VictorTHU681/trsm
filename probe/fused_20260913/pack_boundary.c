#include "fused.c"
#include <float.h>
static double value(uint64_t x){
    x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);x^=x>>31;
    return (double)(x>>11)*0x1p-53-0.5;
}
static double* guarded(size_t bytes,void** mapping,size_t* size){
    size_t page=sysconf(_SC_PAGESIZE),rounded=(bytes+page-1)&~(page-1);
    *size=rounded+page;*mapping=mmap(NULL,*size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(*mapping==MAP_FAILED)abort();
    if(mprotect((char*)*mapping+rounded,page,PROT_NONE))abort();
    return(double*)((char*)*mapping+rounded-bytes);
}
int main(void){
    long comparisons=0;
    for(int K=32;K<=256;K+=32)for(int pad=0;pad<3;pad++)for(int scale=-400;scale<=400;scale+=400){
        int lda=K+pad,ldb=K+pad+7;
        void *ma,*mb,*md;size_t sa,sb,sd;
        double *a=guarded(16L*lda*8,&ma,&sa),*b=guarded(16L*ldb*8,&mb,&sb),*d=guarded(16L*K*8,&md,&sd);
        for(long i=0;i<16L*lda;i++)a[i]=ldexp(value(i+19),scale);
        for(long i=0;i<16L*ldb;i++)b[i]=ldexp(value(i+17119),scale);
        for(int sign=-1;sign<=1;sign+=2){
            sme_start();
            if(sign>0)fused_pack_add(a,lda,b,ldb,0,0,K,d);
            else fused_pack_sub(a,lda,b,ldb,0,0,K,d);
            sme_stop();
            for(int p=0;p<K;p++)for(int r=0;r<16;r++){
                double ref=sign>0?a[(long)r*lda+p]+b[(long)r*ldb+p]:a[(long)r*lda+p]-b[(long)r*ldb+p];
                if(d[p*16+r]!=ref){printf("FUSED_PACK FAIL K=%d r=%d p=%d sign=%d actual=%.17g ref=%.17g\n",K,r,p,sign,d[p*16+r],ref);return 1;}
                comparisons++;
            }
        }
        munmap(ma,sa);munmap(mb,sb);munmap(md,sd);
    }
    for(int sign=-1;sign<=1;sign+=2)for(int scale=-400;scale<=400;scale+=400){
        void *ma,*mb,*md;size_t sa,sb,sd;
        double *a=guarded(128,&ma,&sa),*b=guarded(128,&mb,&sb),*d=guarded(128,&md,&sd);
        for(int j=0;j<16;j++){a[j]=ldexp(value(j+311),scale);b[j]=ldexp(value(j+731),scale);}
        sme_start();if(sign>0)add16(a,b,d);else sub16(a,b,d);sme_stop();
        for(int j=0;j<16;j++){
            double ref=sign>0?a[j]+b[j]:a[j]-b[j];
            if(d[j]!=ref){puts("FUSED_BPACK FAIL");return 1;}comparisons++;
        }
        munmap(ma,sa);munmap(mb,sb);munmap(md,sd);
    }
    printf("FUSED_PACK_BOUNDARY PASS comparisons=%ld protected-input-output independent-53bit exp=-400,0,400\n",comparisons);
    return 0;
}
