from pathlib import Path
import json
p=Path(__file__).resolve().parent;s='#include "trsm-v47-codex.c"\n';cases=[(0,1,0,0),(4,1,0,0),(4,2,0,0),(4,4,0,0),(4,1,1,0),(4,2,1,0),(4,1,0,1),(8,1,0,0),(8,2,0,0),(16,1,0,0)]
for idx,(nf,ways,pred,end) in enumerate(cases):
 ls=['smstart','ptrue p0.d','fdup z0.d, #1.0','fdup z1.d, #0.5','zero {za}']+[f'dup z{r}.d, #0' for r in range(16,32)]+['1:']
 for k in range(16):
  fs=[]
  for f in range(nf):
   v=16+(k%ways)*nf+f
   fs.append(f'fmla z{v}.d, p0/m, z0.d, z1.d' if pred else f'fmla z{v}.d, z0.d, z1.d[0]')
  for za in range(8):
   ls.append(f'fmopa za{za}.d, p0/m, p0/m, z0.d, z1.d')
   if not end:
    for f in range((za*nf)//8,((za+1)*nf)//8):ls.append(fs[f])
  if end:ls+=fs
 ls+=['subs %[n], %[n], #1','b.ne 1b','mov w12, #0','mova z2.d, p0/m, za0h.d[w12, 0]','st1d z2.d, p0, [%[out]]','st1d z16.d, p0, [%[out], #1, mul vl]','smstop']
 s+=f'static __attribute__((noinline)) double peak{idx}(long n,double*out) {{uint64_t t=cyc();asm volatile(\n'+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in ls)+''' :[n]"+&r"(n):[out]"r"(out):"memory","cc","x12",VCLOB,PCLOB);return (cyc()-t)/(double)cyc_hz();}
'''
s+='int main(void){double out[16] __attribute__((aligned(64)));long n=20000;\n'
for idx,(nf,ways,pred,end) in enumerate(cases):
 s+=f'''for(int rep=0;rep<3;rep++){{double sec=peak{idx}(n,out);double az=n*16.0*.5,sv={1 if nf else 0}*n*16.0*.5/{ways};if(out[0]!=az||out[8]!=sv){{printf("HYBRID_PEAK_FAIL case={idx} za=%.9g/%.9g sve=%.9g/%.9g\\n",out[0],az,out[8],sv);return 1;}}printf("HYBRID_PEAK case={idx} nf={nf} ways={ways} pred={pred} end={end} rep=%d cycles_per_k=%.6f GFLOPS=%.6f PASS\\n",rep,sec*1.55e9/(n*16),(1024.0+16*{nf})*n*16/sec*1e-9);}}
'''
s+='return 0;}\n';(p/'hybrid-peak.c').write_text(s)
