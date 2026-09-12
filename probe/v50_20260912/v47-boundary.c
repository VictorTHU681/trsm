#ifndef PROBE_SOURCE
#define PROBE_SOURCE "trsm-v47-codex.c"
#endif
#include PROBE_SOURCE
#include <sys/mman.h>
static double* guarded_duplex(size_t bytes) {
 void* p=mmap(0,bytes+4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 if(p==MAP_FAILED || mprotect((char*)p+bytes,4096,PROT_NONE))exit(2);
 return p;
}
static int check_duplex(void){
 double*a=guarded_duplex(8192),*b=guarded_duplex(65536),*c=guarded_duplex(16384),*ref=malloc(16384),*pf=calloc(8192,8);
 uint64_t u=39881;double worst=0;long count=0;
 for(int it=0;it<50;it++) {
  double scale=it%3==0?0x1p-400:it%3==1?0x1p400:1;
  for(int q=0;q<1024;q++){u=u*6364136223846793005ULL+1;a[q]=((double)(int64_t)(u>>11)-(double)(1ULL<<52))*0x1p-54*scale;}
  for(int q=0;q<8192;q++){u=u*6364136223846793005ULL+1;b[q]=((double)(int64_t)(u>>11)-(double)(1ULL<<52))*0x1p-54/scale;}
  for(int q=0;q<2048;q++)ref[q]=c[q]=q*0x1p-12;
  for(int r=0;r<16;r++)for(int j=96;j<128;j++){double x=0;for(int k=0;k<64;k++)x=fma(a[k*16+r],b[(j/16)*1024+k*16+j%16],x);ref[r*128+j]-=x;}
  sme_start();sme_zero_za();
  kloop_hoist_bank0(a,b+6*1024,c+112,(char*)pf,(char*)pf,(char*)c+96*8);
  kloop_hoist_bank4(a,b+7*1024,c+96,(char*)pf,(char*)pf,(char*)c+112*8);
  za_flush_duplex_bank4(c+112,1024);sme_stop();
  for(int q=0;q<2048;q++){double d=fabs(ref[q]-c[q]);if(d>worst)worst=d;if(d!=0 || !isfinite(c[q])){printf("DUPLEX_BOUNDARY FAIL it=%d q=%d diff=%.17g\n",it,q,d);return 1;}count++;}
 }
 printf("DUPLEX_BOUNDARY PASS comparisons=%ld maxdiff=%.3g protected A/B/C pages\n",count,worst);
return 0;
}
#include <sys/mman.h>
static double* guarded_owner(size_t bytes) {
 void* p=mmap(0,bytes+4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 if(p==MAP_FAILED || mprotect((char*)p+bytes,4096,PROT_NONE))exit(2);
 return p;
}
static int check_owner(void){
 double*a=guarded_owner(16384),*b=guarded_owner(49152),*c=guarded_owner(24576),*ref=malloc(24576),*pf=calloc(8192,8);
 uint64_t u=39881;double worst=0;long count=0;
 for(int it=0;it<50;it++) {
  double scale=it%3==0?0x1p-400:it%3==1?0x1p400:1;
  for(int q=0;q<2048;q++){u=u*6364136223846793005ULL+1;a[q]=((double)(int64_t)(u>>11)-(double)(1ULL<<52))*0x1p-54*scale;}
  for(int q=0;q<6144;q++){u=u*6364136223846793005ULL+1;b[q]=((double)(int64_t)(u>>11)-(double)(1ULL<<52))*0x1p-54/scale;}
  for(int q=0;q<3072;q++)ref[q]=c[q]=q*0x1p-12;
  for(int r=0;r<32;r++)for(int j=80;j<96;j++){double x=0;for(int k=0;k<64;k++)x=fma(a[(r/16)*1024+k*16+r%16],b[k*96+j],x);ref[r*96+j]-=x;}
  sme_start();sme_zero_za();
  kloop_owner_bank0(a,b+80,c+80,(char*)pf,(char*)pf,(char*)c+80*8);
  kloop_owner_bank4(a+1024,b+80,c+80,(char*)pf,(char*)pf,(char*)(c+16*96+80));
  za_flush_duplex_bank4(c+16*96+80,768);sme_stop();
  for(int q=0;q<3072;q++){double d=fabs(ref[q]-c[q]);if(d>worst)worst=d;if(d!=0 || !isfinite(c[q])){printf("OWNER_BOUNDARY FAIL it=%d q=%d diff=%.17g\n",it,q,d);return 1;}count++;}
 }
 printf("OWNER_BOUNDARY PASS comparisons=%ld maxdiff=%.3g protected A/B/C pages\n",count,worst);
return 0;
}
int main(void) { if(check_duplex())return 1;return check_owner(); }
