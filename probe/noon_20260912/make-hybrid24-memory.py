from pathlib import Path
import re,json
p=Path(__file__).resolve().parent;old=(p/'hybrid24-replicas.c').read_text();a=old.index('static __attribute__((noinline)) void hybrid24_4');b=old.index('static double val',a)
fn=old[a:b];body=fn[fn.index('asm volatile(\n')+len('asm volatile(\n'):fn.index(' :[out]')]
ls=[json.loads(l.strip())[:-2] for l in body.splitlines() if l.strip().startswith('"')]
# Each parsed instruction formerly ended in newline/tab; strip those two bytes.
assert ls[0]=='ptrue p0.d'
ls=ls[:ls.index('mov w12,#0')]
# Host A/B prefetch across each K step. Each column tile contributes one quarter of the next A strip.
out=[]
for l in ls:
 out.append(l)
 if l=='add x10,x10,#256':out += ['prfm pldl2keep,[%[ap]]','prfm pldl2keep,[%[bp]]','add %[ap],%[ap],#64','add %[bp],%[bp],#64']
ls=out+['mov w12,#0','3:']
for za in range(8):ls += [f'ld1d z{za+8}.d,p0/z,[%[out],#{za},mul vl]',f'mova z{za}.d,p0/m,za{za}h.d[w12,0]']
for za in range(8):ls += [f'fadd z{za}.d,z{za}.d,z{za+8}.d',f'st1d z{za}.d,p0,[%[out],#{za},mul vl]']
ls += ['add %[out],%[out],#512','add w12,w12,#1','cmp w12,#8','b.lt 3b']
for v in range(8):ls += [f'ld1d z{v}.d,p0/z,[%[out],#{v},mul vl]']
for v in range(8):ls += [f'fadd z{v+16}.d,z{v+16}.d,z{v}.d',f'st1d z{16+v}.d,p0,[%[out],#{v},mul vl]']
s='#include "trsm-v47-codex.c"\n'
for K in [32,64,128]:
 sub=[l.replace('mov x11,#64',f'mov x11,#{K}') for l in ls]
 s+=f'static inline __attribute__((always_inline)) void hmem{K}(const double*a,const double*b,double*out,const char*ap,const char*bp){{long n=1;asm volatile(\n'+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in sub)+''' :[out]"+&r"(out),[n]"+&r"(n),[ap]"+&r"(ap),[bp]"+&r"(bp):[a]"r"(a),[b]"r"(b):"memory","cc","x9","x10","x11","x12",VCLOB,PCLOB);}
'''
for K in [32,64,128]:
 s+=f'''static __attribute__((noinline)) double runh{K}(const double*A,const double*B,double*C,char*dummy,int mode,int steps){{
 const int S=8,NT=4,K={K};size_t na=(mode==2||mode==4)?1024:1,nb=(mode==3||mode==4)?1024:1;uint64_t t=cyc();
 for(int chunk=0;chunk<steps;chunk++)for(int r=0;r<S;r++){{
  size_t ai=chunk%na,bi=chunk%nb,aa=(ai*S+r)*32*K,an=(ai*S+r+1)*32*K;if(r==S-1)an=((chunk+1)%na)*S*32*K;
  for(int j=0;j<NT;j++){{const double*b=B+(bi*NT+j)*32*K;double*c=C+(r*NT+j)*576;
   const char*ap=na>1?(const char*)(A+an)+j*K*64:dummy;const char*bp=dummy;int call=r*NT+j;
   if(nb>1&&call>=16)bp=(char*)(B+((chunk+1)%nb)*NT*32*K)+(call-16)*K*64;
   hmem{K}(A+aa,b,c,ap,bp);
  }}
 }}
 return (cyc()-t)/(double)cyc_hz()*1e9/(steps*S*NT*K);
}}
'''
s+=r'''
static __attribute__((noinline)) double runbase(const double*A,const double*B,double*C,char*dummy,int mode,int steps) {
 const int S=12,NT=4,K=64;size_t na=(mode==2||mode==4)?1024:1,nb=(mode==3||mode==4)?1024:1;long pend=0;uint64_t t=cyc();
 for(int chunk=0;chunk<steps;chunk++)for(int r=0;r<S;r++) {
  size_t ai=chunk%na,bi=chunk%nb,aa=(ai*S+r)*1024,an=(ai*S+r+1)*1024;if(r==S-1)an=((chunk+1)%na)*S*1024;
  for(int j=0;j<NT;j++) {
   const double*b=B+bi*NT*2048+j*2048;double*c=C+r*16*128+j*32;
   const char*ap=na>1?(char*)(A+an)+j*2048:dummy;const char*bp=dummy;int call=r*NT+j;
   if(nb>1&&call>=16)bp=(char*)(B+((chunk+1)%nb)*NT*2048)+(call-16)*2048;
   kloop_hoist_bank0(A+aa,b,C+pend+16,ap,bp,(char*)c);kloop_hoist_bank4(A+aa,b+1024,c,ap+1024,bp+1024,(char*)c+128);pend=r*16*128+j*32;
  }
 }
 za_flush_duplex_bank4(C+pend+16,1024);return (cyc()-t)/(double)cyc_hz()*1e9/(steps*S*NT*K);
}
static double val(uint64_t i){i=(i^(i>>30))*UINT64_C(0xbf58476d1ce4e5b9);i=(i^(i>>27))*UINT64_C(0x94d049bb133111eb);i^=i>>31;return (i>>11)*0x1p-53-.5;}
static double input(int chunk,int strip,int k,int r,int salt){return val((((size_t)chunk*8+strip)*128+k)*24+r+salt);}
int main(void) {
 size_t acount=(size_t)1024*8*32*128,bcount=(size_t)1024*4*32*128;
 double*A=aligned_alloc(4096,acount*8),*B=aligned_alloc(4096,bcount*8),*C=aligned_alloc(4096,192*128*8);
 double*AB=aligned_alloc(4096,(size_t)1024*12*1024*8),*BB=aligned_alloc(4096,(size_t)1024*4*2048*8);char*dummy=aligned_alloc(4096,1<<20);
 if(!A||!B||!C||!AB||!BB||!dummy)return 2;memset(dummy,0,1<<20);
 for(size_t i=0;i<(size_t)1024*12*1024;i++)AB[i]=val(i+1123);for(size_t i=0;i<(size_t)1024*4*2048;i++)BB[i]=val(i+3231);
 double(*runs[])(const double*,const double*,double*,char*,int,int)={runh32,runh64,runh128};
 for(int which=0;which<3;which++) {
  int K=32<<which;
  for(int ch=0;ch<1024;ch++)for(int st=0;st<8;st++)for(int k=0;k<K;k++)for(int r=0;r<32;r++)A[((size_t)ch*8+st)*32*K+k*32+r]=input(ch,st,k,r<24?r:16+(r-24+4)%8,1);
  for(int ch=0;ch<1024;ch++)for(int st=0;st<4;st++)for(int k=0;k<K;k++)for(int r=0;r<32;r++)B[((size_t)ch*4+st)*32*K+k*32+r]=input(ch,st,k,r<24?r:16+(r-24+2)%8,99123);
  memset(C,0,192*128*8);sme_start();runs[which](A,B,C,dummy,4,3);sme_stop();int bad=0;double worst=0;
  for(int st=0;st<8;st++)for(int jt=0;jt<4;jt++)for(int i=0;i<24;i++)for(int j=0;j<24;j++) {
   int ix;
   if(i<16||j<16){int tile=(i/8)*3+j/8;ix=(i%8)*64+tile*8+j%8;}
   else {int row=i-16,col=j-16,rot=(col/2-row/2+4)%4,lane=(row+(rot>=2?4:0))%8;ix=512+(2*rot+col%2)*8+lane;}
   long double sum=0;for(int ch=0;ch<3;ch++)for(int k=0;k<K;k++)sum+=(long double)input(ch,st,k,i,1)*input(ch,jt,k,j,99123);
   double err=fabs(C[(st*4+jt)*576+ix]-(double)sum);bad+=!isfinite(err)||err>1e-12;worst=fmax(worst,err);
  }
  printf("H24_MEMORY_CHECK K=%d outputs=18432 chunks=3 error=%.3e bad=%d %s\n",K,worst,bad,bad?"FAIL":"PASS");fflush(stdout);if(bad)return 1;
  for(int mode=1;mode<=4;mode++)for(int rep=0;rep<3;rep++)for(int h=0;h<2;h++) {
   sme_start();sme_zero_za();double ns=h?runs[which](A,B,C,dummy,mode,256):runbase(AB,BB,C,dummy,mode,256);sme_stop();
   printf("H24_MEMORY_PERF K=%d mode=%d rep=%d hybrid=%d ns_per_k=%.6f cycles155=%.6f GFLOPS=%.6f\n",K,mode,rep,h,ns,ns*1.55,(h?1152:1024)/ns);fflush(stdout);
  }
 }
 free(A);free(B);free(C);free(AB);free(BB);free(dummy);return 0;
}
'''
(p/'hybrid24-memory.c').write_text(s)
