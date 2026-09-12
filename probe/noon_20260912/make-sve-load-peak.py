from pathlib import Path
import json
p=Path(__file__).resolve().parent;s='#include "trsm-v47-codex.c"\n';cases=[(n,0,0) for n in [4,8,12,16]]+[(n,8,0) for n in [4,8,12,16]]+[(n,8,8) for n in [4,8,12,16]]
for idx,(nl,nm,nf) in enumerate(cases):
 ls=['ptrue p0.d','fdup z0.d,#1.0','fdup z1.d,#0.5','zero {za}']+[f'dup z{r}.d,#0' for r in range(16,24)]+['1:']
 for un in range(16):
  for j in range(8):
   for k in range(j*nl//8,(j+1)*nl//8):ls.append(f'ld1d z{8+k%8}.d,p0/z,[%[a],#{k%8},mul vl]')
   if nm:ls.append(f'fmopa za{j}.d,p0/m,p0/m,z0.d,z1.d')
   if nf:ls.append(f'fmla z{16+j}.d,z0.d,z1.d[0]')
 ls+=['subs %[n],%[n],#1','b.ne 1b','st1d z8.d,p0,[%[out]]','mov w12,#0','mova z2.d,p0/m,za0h.d[w12,0]','st1d z2.d,p0,[%[out],#1,mul vl]','st1d z16.d,p0,[%[out],#2,mul vl]']
 s+=f'static __attribute__((noinline)) void loadpeak{idx}(const double*a,double*out,long n){{asm volatile(\n'+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in ls)+''' :[n]"+&r"(n):[a]"r"(a),[out]"r"(out):"memory","cc","x12",VCLOB,PCLOB);}
'''
s+='int main(void){double a[64] __attribute__((aligned(64))),out[24] __attribute__((aligned(64)));for(int i=0;i<64;i++)a[i]=i+1;long n=40000;\n'
for idx,(nl,nm,nf) in enumerate(cases):
 s+=f'''for(int rep=0;rep<3;rep++){{sme_start();uint64_t t=cyc();loadpeak{idx}(a,out,n);uint64_t dt=cyc()-t;sme_stop();if(out[0]!=1||out[8]!={1 if nm else 0}*n*8.0||out[16]!={1 if nf else 0}*n*8.0){{printf("LOAD_PEAK_FAIL idx={idx}\\n");return 1;}}double ns=(double)dt/cyc_hz()*1e9/(n*16);printf("LOAD_PEAK loads={nl} fmopa={nm} fmla={nf} rep=%d cycles155=%.6f GBps=%.6f PASS\\n",rep,ns*1.55,{nl}*64/ns);}}\n'''
s+='return 0;}\n';(p/'sve-load-peak.c').write_text(s)
