from pathlib import Path
import json
p=Path(__file__).resolve().parent;src='#include "trsm-v47-codex.c"\n'
for variant in range(4):
 ls=['ptrue p0.d','zero {za}']+[f'dup z{r}.d,#0' for r in range(16,24)]+['1:','mov x9,%[a]','mov x10,%[b]','mov x11,#64','2:']
 for u in range(4):
  ls += [f'ld1d z{r}.d,p0/z,[{ptr},#{off},mul vl]' for r,ptr,off in [(0,'x9',0),(1,'x9',1),(2,'x9',2),(3,'x10',0),(4,'x10',1),(5,'x10',2)]]
  # Execute both uses of B16 before rotating it. Remaining SME work is independent.
  ops=['fmopa za2.d,p0/m,p0/m,z0.d,z5.d','fmopa za5.d,p0/m,p0/m,z1.d,z5.d','fmopa za0.d,p0/m,p0/m,z0.d,z3.d','fmopa za1.d,p0/m,p0/m,z0.d,z4.d','fmopa za3.d,p0/m,p0/m,z1.d,z3.d','fmopa za4.d,p0/m,p0/m,z1.d,z4.d','fmopa za6.d,p0/m,p0/m,z2.d,z3.d','fmopa za7.d,p0/m,p0/m,z2.d,z4.d']
  if variant==0:ls+=ops
  elif variant in (1,2):
   for rot in range(4):
    ls+=ops[2*rot:2*rot+2]
    if rot:
     ls += ['ext z5.b,z5.b,z5.b,#16'] if variant==1 else [f'ld1d z5.d,p0/z,[x10,#{rot+2},mul vl]']
    ls += [f'fmla z{16+rot*2}.d,z2.d,z5.d[0]',f'fmla z{17+rot*2}.d,z2.d,z5.d[1]']
  else:
   for row in range(8):
    ls += [ops[row],f'dup z6.d,z2.d[{row}]',f'fmla z{16+row}.d,p0/m,z5.d,z6.d']
  ls += ['add x9,x9,#192',f'add x10,x10,#{384 if variant==2 else 192}']
 ls += ['subs x11,x11,#4','b.gt 2b','subs %[n],%[n],#1','b.ne 1b','mov w12,#0','3:']
 for za in range(8):ls += [f'mova z{za}.d,p0/m,za{za}h.d[w12,0]',f'st1d z{za}.d,p0,[%[out],#{za},mul vl]']
 ls += ['add %[out],%[out],#512','add w12,w12,#1','cmp w12,#8','b.lt 3b']
 for v in range(8):ls += [f'st1d z{16+v}.d,p0,[%[out],#{v},mul vl]']
 src+=f'static __attribute__((noinline)) void hybrid24_{variant}(const double*a,const double*b,double*out,long n){{asm volatile(\n'+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in ls)+''' :[out]"+&r"(out),[n]"+&r"(n):[a]"r"(a),[b]"r"(b):"memory","cc","x9","x10","x11","x12",VCLOB,PCLOB);}
'''
src+=r'''
static double val(uint64_t i){i=(i^(i>>30))*UINT64_C(0xbf58476d1ce4e5b9);i=(i^(i>>27))*UINT64_C(0x94d049bb133111eb);i^=i>>31;return (i>>11)*0x1p-53-.5;}
int main(void){
 double*A=aligned_alloc(64,64*24*8),*B=aligned_alloc(64,64*24*8),*BP=aligned_alloc(64,64*48*8),*out=aligned_alloc(64,576*8);
 void(*fn[])(const double*,const double*,double*,long)={hybrid24_0,hybrid24_1,hybrid24_2,hybrid24_3};
 int bad=0;
 for(int mode=0;mode<2;mode++) {
  for(int k=0;k<64;k++)for(int r=0;r<24;r++){A[k*24+r]=mode?((k*17+r*3)%5)-2:val(k*24+r+1);B[k*24+r]=mode?((k*7+r*13)%5)-2:val(k*24+r+981);}
  for(int k=0;k<64;k++){for(int j=0;j<24;j++)BP[k*48+j]=B[k*24+j];for(int rot=1;rot<4;rot++)for(int j=0;j<8;j++)BP[k*48+16+rot*8+j]=B[k*24+16+(j+2*rot)%8];}
  for(int v=0;v<4;v++) {
   long n=mode?100:3;sme_start();fn[v](A,v==2?BP:B,out,n);sme_stop();double err=0;int checked=0;
   for(int i=0;i<24;i++)for(int j=0;j<24;j++) {
    if(v==0&&i>=16&&j>=16)continue;
    int index;
    if(i<16||j<16){int tile=(i/8)*3+j/8;index=(i%8)*64+tile*8+j%8;}
    else if(v==3)index=512+(i-16)*8+j-16;
    else {int row=i-16,col=j-16,rot=(col/2-row/2+4)%4;index=512+(2*rot+col%2)*8+row;}
    long double dot=0;for(int k=0;k<64;k++)dot+=(long double)A[k*24+i]*B[k*24+j];
    double e=fabs(out[index]-n*(double)dot);err=fmax(err,e);bad+=!isfinite(e)||e>(mode?0:1e-12);checked++;
   }
   printf("H24_CHECK variant=%d exact=%d outputs=%d error=%.3e %s\n",v,mode,checked,err,bad?"FAIL":"PASS");fflush(stdout);if(bad)return 1;
   if(mode)for(int rep=0;rep<3;rep++) {
    sme_start();uint64_t t=cyc();fn[v](A,v==2?BP:B,out,10000);uint64_t dt=cyc()-t;sme_stop();
    double ns=(double)dt/cyc_hz()*1e9/640000;
    printf("H24_PERF variant=%d rep=%d ns_per_k=%.6f cycles155=%.6f GFLOPS=%.6f\n",v,rep,ns,ns*1.55,(v?1152:1024)/ns);fflush(stdout);
   }
  }
 }
 free(A);free(B);free(BP);free(out);return 0;
}
'''
(p/'hybrid24.c').write_text(src)
