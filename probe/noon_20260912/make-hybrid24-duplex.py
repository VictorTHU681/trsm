from pathlib import Path
import json
p=Path(__file__).resolve().parent;s='#include "trsm-v47-codex.c"\n'
for bank in (0,4):
 other=4-bank;ls=['ptrue p0.d',f'zero {{za{bank}.d,za{bank+1}.d,za{bank+2}.d,za{bank+3}.d}}']+[f'dup z{v}.d,#0' for v in range(24,28)]
 pos={'a':0,'b':0}
 def load(ptr,k,regs):
  stride=128 if ptr=='a' else 256;group=8 if ptr=='a' else 4;base=(k//group)*group*stride+512
  if base!=pos[ptr]:ls.append(f'add %[{ptr}],%[{ptr}],#{base-pos[ptr]}');pos[ptr]=base
  for j,v in enumerate(regs):
   off=(k*stride+j*64-base)//64;assert -8<=off<=7
   ls.append(f'ld1d z{v}.d,p0/z,[%[{ptr}],#{off},mul vl]')
 load('a',0,[8,9]);load('b',0,[4,5,6,7])
 for k in range(64):
  row=k//8;phase=k%8
  if phase==0:
   ls.append(f'mov w12,#{row}')
   for za in range(4):ls += [f'mova z{16+za*2}.d,p0/m,za{other+za}h.d[w12,0]',f'ld1d z{17+za*2}.d,p0/z,[%[c],#{za},mul vl]']
  ar=[8,9] if k%2==0 else [14,15];br=[4,5,6,7] if k%2==0 else [0,1,2,3]
  nrA=[14,15] if k%2==0 else [8,9];nrB=[0,1,2,3] if k%2==0 else [4,5,6,7]
  # Prepare next pointers, but load each next operand between independent outer products.
  nextloads=[]
  if k<63:
   oldlen=len(ls);load('a',k+1,nrA);load('b',k+1,nrB);nextloads=ls[oldlen:];ls=ls[:oldlen]
   # Pointer additions must precede their associated loads; preserve list order.
  pairs=[(ar[0],br[0]),(ar[0],br[1]),(ar[0],br[2]),(ar[1],br[0 if bank==0 else 1])]
  cursor=0
  for za,(av,bv) in enumerate(pairs):
   while cursor<len(nextloads):
    l=nextloads[cursor];ls.append(l);cursor+=1
    if l.startswith('ld1d'):break
   ls += [f'fmopa za{bank+za}.d,p0/m,p0/m,z{av}.d,z{bv}.d',f'fmla z{24+za}.d,z{ar[1]}.d,z{br[2+za//2]}.d[{za%2}]']
  ls+=nextloads[cursor:]
  if phase in [1,3,5,7]:
   z=phase//2;ls += [f'fsub z{17+2*z}.d,z{17+2*z}.d,z{16+2*z}.d',f'st1d z{17+2*z}.d,p0,[%[c],#{z},mul vl]']
   if phase in [1,5]:ls+=['prfm pldl2keep,[%[ap]]','prfm pldl2keep,[%[ap],#64]','prfm pldl2keep,[%[bp]]','prfm pldl2keep,[%[bp],#64]','add %[ap],%[ap],#128','add %[bp],%[bp],#128']
   if phase==3:ls += [f'prfm pldl2keep,[%[pc],#{row*256+z*64}]' for z in range(4)]
   if phase==7:ls += ['add %[c],%[c],#256']
 for z in range(4):ls += [f'ld1d z{z}.d,p0/z,[%[ec],#{z},mul vl]']
 for z in range(4):ls += [f'fsub z{z}.d,z{z}.d,z{24+z}.d',f'st1d z{z}.d,p0,[%[ec],#{z},mul vl]']
 s+=f'static inline __attribute__((always_inline)) void hduplex{bank}(const double*a,const double*b,double*c,double*ec,const char*ap,const char*bp,const char*pc){{asm volatile(\n'+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in ls)+''' :[a]"+&r"(a),[b]"+&r"(b),[c]"+&r"(c),[ap]"+&r"(ap),[bp]"+&r"(bp):[ec]"r"(ec),[pc]"r"(pc):"memory","cc","x12",VCLOB,PCLOB);}
'''
s+=r'''
static inline void hflush(double*c){asm volatile(
 "ptrue p0.d\n\tmov w12,#0\n\t1:\n\t"
 "mova z0.d,p0/m,za4h.d[w12,0]\n\tld1d z4.d,p0/z,[%[c]]\n\tfsub z4.d,z4.d,z0.d\n\tst1d z4.d,p0,[%[c]]\n\t"
 "mova z1.d,p0/m,za5h.d[w12,0]\n\tld1d z5.d,p0/z,[%[c],#1,mul vl]\n\tfsub z5.d,z5.d,z1.d\n\tst1d z5.d,p0,[%[c],#1,mul vl]\n\t"
 "mova z2.d,p0/m,za6h.d[w12,0]\n\tld1d z6.d,p0/z,[%[c],#2,mul vl]\n\tfsub z6.d,z6.d,z2.d\n\tst1d z6.d,p0,[%[c],#2,mul vl]\n\t"
 "mova z3.d,p0/m,za7h.d[w12,0]\n\tld1d z7.d,p0/z,[%[c],#3,mul vl]\n\tfsub z7.d,z7.d,z3.d\n\tst1d z7.d,p0,[%[c],#3,mul vl]\n\t"
 "add %[c],%[c],#256\n\tadd w12,w12,#1\n\tcmp w12,#8\n\tb.lt 1b\n\t"
 :[c]"+&r"(c)::"memory","cc","x12",VCLOB,PCLOB);}
static __attribute__((noinline)) double runhd(const double*A,const double*B,double*C,char*dummy,int mode,int steps) {
 const int S=8,NT=4,K=64;size_t na=(mode==2||mode==4)?1024:1,nb=(mode==3||mode==4)?1024:1;uint64_t t=cyc();long pend=0;
 for(int chunk=0;chunk<steps;chunk++)for(int r=0;r<S;r++){
  size_t ai=chunk%na,bi=chunk%nb,aa=(ai*S+r)*32*K,an=(ai*S+r+1)*32*K;if(r==S-1)an=((chunk+1)%na)*S*32*K;
  for(int j=0;j<NT;j++) {
   const double*b=B+(bi*NT+j)*32*K;double*c=C+(r*NT+j)*576;
   const char*ap=na>1?(const char*)(A+an)+j*K*64:dummy;const char*bp=dummy;int call=r*NT+j;
   if(nb>1&&call>=16)bp=(char*)(B+((chunk+1)%nb)*NT*32*K)+(call-16)*K*64;
   hduplex0(A+aa,b,C+pend+288,c+256,ap,bp,(char*)c);
   hduplex4(A+aa+16*K,b,c,c+288+256,ap+2048,bp+2048,(char*)(c+288));
   pend=(r*NT+j)*576;
  }
 }
 hflush(C+pend+288);return (cyc()-t)/(double)cyc_hz()*1e9/(steps*S*NT*K);
}
'''
old=(p/'hybrid24-memory.c').read_text();a=old.index('static __attribute__((noinline)) double runbase(');s+=old[a:old.index('int main(void)',a)]
s+=r'''
int main(void){
 const int K=64;size_t acount=(size_t)1024*8*32*K,bcount=(size_t)1024*4*32*K;
 double*A=aligned_alloc(4096,acount*8),*B=aligned_alloc(4096,bcount*8),*C=aligned_alloc(4096,192*128*8);
 double*AB=aligned_alloc(4096,(size_t)1024*12*1024*8),*BB=aligned_alloc(4096,(size_t)1024*4*2048*8);char*dummy=aligned_alloc(4096,1<<20);
 if(!A||!B||!C||!AB||!BB||!dummy)return 2;memset(dummy,0,1<<20);
 for(size_t i=0;i<(size_t)1024*12*1024;i++)AB[i]=val(i+1123);for(size_t i=0;i<(size_t)1024*4*2048;i++)BB[i]=val(i+3231);
 for(int ch=0;ch<1024;ch++)for(int st=0;st<8;st++)for(int bank=0;bank<2;bank++)for(int k=0;k<K;k++)for(int r=0;r<16;r++) {
  int row=r<8?bank*8+r:16+(r-8+bank*4)%8;
  A[((size_t)ch*8+st)*32*K+bank*16*K+k*16+r]=input(ch,st,k,row,1);
 }
 for(int ch=0;ch<1024;ch++)for(int st=0;st<4;st++)for(int k=0;k<K;k++)for(int r=0;r<32;r++)B[((size_t)ch*4+st)*32*K+k*32+r]=input(ch,st,k,r<24?r:16+(r-24+2)%8,99123);
 memset(C,0,192*128*8);sme_start();sme_zero_za();runhd(A,B,C,dummy,4,3);sme_stop();int bad=0;double worst=0;
 for(int st=0;st<8;st++)for(int jt=0;jt<4;jt++)for(int i=0;i<24;i++)for(int j=0;j<24;j++) {
  int ix;
  if(i<16){int bank=i/8;ix=bank*288+(i%8)*32+(j/8)*8+j%8;}
  else if(j<16){int bank=j/8,row=(i-16+bank*4)%8;ix=bank*288+row*32+24+j%8;}
  else {int row=i-16,col=j-16,rot=(col/2-row/2+4)%4,bank=rot/2,lane=(row+bank*4)%8;ix=bank*288+256+(2*(rot%2)+col%2)*8+lane;}
  long double sum=0;for(int ch=0;ch<3;ch++)for(int k=0;k<K;k++)sum+=(long double)input(ch,st,k,i,1)*input(ch,jt,k,j,99123);
  double err=fabs(C[(st*4+jt)*576+ix]+(double)sum);bad+=!isfinite(err)||err>1e-12;worst=fmax(worst,err);
 }
 printf("H24_DUPLEX_CHECK outputs=18432 chunks=3 error=%.3e bad=%d %s\n",worst,bad,bad?"FAIL":"PASS");fflush(stdout);if(bad)return 1;
 for(int mode=1;mode<=4;mode++)for(int rep=0;rep<3;rep++)for(int h=0;h<2;h++) {
  sme_start();sme_zero_za();double ns=h?runhd(A,B,C,dummy,mode,256):runbase(AB,BB,C,dummy,mode,256);sme_stop();
  printf("H24_DUPLEX_PERF mode=%d rep=%d hybrid=%d ns_per_k=%.6f cycles155=%.6f GFLOPS=%.6f\n",mode,rep,h,ns,ns*1.55,(h?1152:1024)/ns);fflush(stdout);
 }
 free(A);free(B);free(C);free(AB);free(BB);free(dummy);return 0;
}
'''
(p/'hybrid24-duplex.c').write_text(s)
