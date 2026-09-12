from pathlib import Path
import json
p=Path(__file__).resolve().parent;s=(p/'trsm-v48-synced.c').read_text();defs=''
for skip in (0,1):
 for K in (16,32,48,64):
  ls=['ptrue p0.d'];pos={'a':0,'b':0}
  def load(ptr,k,regs):
   stride=128 if ptr=='a' else 768;group=8 if ptr=='a' else 2;bias=512;base=k//group*group*stride+bias
   if base!=pos[ptr]:ls.append(f'add %[{ptr}],%[{ptr}],#{base-pos[ptr]}');pos[ptr]=base
   for j,v in enumerate(regs):
    if ptr=='a' and j==0 and skip and k>=K-8:continue
    off=(k*stride+j*64-base)//64;assert -8<=off<=7
    ld='ldnt1d' if ptr=='a' else 'ld1d';ls.append(f'{ld} z{v}.d,p0/z,[%[{ptr}],#{off},mul vl]')
  load('a',0,[0,1]);load('b',0,[4,5,6,7])
  for k in range(K):
   ar=[0,1] if k%2==0 else [8,9];br=[4,5,6,7] if k%2==0 else [12,13,14,15]
   nxt=[]
   if k<K-1:
    n=len(ls);load('a',k+1,[8,9] if k%2==0 else [0,1]);load('b',k+1,[12,13,14,15] if k%2==0 else [4,5,6,7]);nxt=ls[n:];ls=ls[:n]
   ix=0
   for za in range(8):
    if skip and k>=K-8 and za<4:continue
    if za in [0,2,4,5,6,7]:
     while ix<len(nxt):
      line=nxt[ix];ls.append(line);ix+=1
      if line.startswith('ld'):break
    ls.append(f'fmopa za{za}.d,p0/m,p0/m,z{ar[za//4]}.d,z{br[za%4]}.d')
   ls+=nxt[ix:]
   if k%2==0:
    q=k//2
    if q>0 and q%12==0:ls.append('add %[pfp],%[pfp],%[rs]')
    ls += [f'prfm pldl1keep,[%[pfp],#{(q%12)*64}]',f'prfm pldl1keep,[%[pfx],#{q*64}]']
   elif k<K-1:
    q=k//2;ls += [f'prfm pldl2keep,[%[pc],#{q*768+j*64}]' for j in range(4)]
  defs+=f'static inline __attribute__((always_inline)) void odiag{skip}_{K}(const double*a,const double*b,const pf_t*pf){{const char*pfp=pf->p;long rs=pf->pfj+pf->pfn*64;asm volatile(\n'+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in ls)+''' :[a]"+&r"(a),[b]"+&r"(b),[pfp]"+&r"(pfp):[rs]"r"(rs),[pfx]"r"(pf->p2),[pc]"r"(pf->pc):"memory","cc",VCLOB,PCLOB);}
'''
# These only operate within the existing fixed owner path (full 16x32 tiles).
helper='static int g_odiag;\n'+defs
s=s.replace('// ---------------------------------------------------------------- 任务执行',helper+'\n// ---------------------------------------------------------------- 任务执行',1)
old='                    PROF_T(c, kloop_plain_nt(A, 128, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);'
new=''
for skip in (0,1):
 new+=f'                    {"if" if skip==0 else "else if"}(fast_owner=={2+skip}){{\n'
 for ki,K in enumerate((16,32,48,64)):
  new+=f'                        {"if" if ki==0 else "else if"}(s=={ki}){{PROF_T(c,odiag{skip}_{K}(A,cb+32*t,&pf),t_dkloop);}}\n'
 new+='                    }'
new+='else{\n'+old+'\n                    }';assert old in s;s=s.replace(old,new,1)
needle='{ return run_task_impl(c, tb, tk, guess, tid, 1); }'
s=s.replace(needle,needle+'\nstatic long run_task_owner_diag0(const ctx_t*c,tbuf_t*tb,const task_t*tk,const task_t*guess,int tid)\n{return run_task_impl(c,tb,tk,guess,tid,2);}\nstatic long run_task_owner_diag1(const ctx_t*c,tbuf_t*tb,const task_t*tk,const task_t*guess,int tid)\n{return run_task_impl(c,tb,tk,guess,tid,3);}',1)
s=s.replace('    c.prof = env_int("TRSM_PROFILE", 0);','    c.prof = env_int("TRSM_PROFILE", 0);\n    g_odiag=env_int("TRSM_ODIAG",0);',1)
s=s.replace('(c.special == 5 ? run_task_owner : run_task)','(c.special == 5 ? (g_odiag==1?run_task_owner_diag0:g_odiag==2?run_task_owner_diag1:run_task_owner) : run_task)',1)
(p/'trsm-owner-diag48.c').write_text(s)
# Independent protected-page kernel check for all K, not the benchmark generator.
t='''#include "trsm-owner-diag48.c"
static double* guard(size_t bytes){size_t n=(bytes+4095)&~4095UL;char*b=mmap(0,n+4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(b==MAP_FAILED||mprotect(b+n,4096,PROT_NONE))exit(2);return (double*)(b+n-bytes);}
static double val(uint64_t i){i=(i^(i>>30))*UINT64_C(0xbf58476d1ce4e5b9);i=(i^(i>>27))*UINT64_C(0x94d049bb133111eb);i^=i>>31;return (i>>11)*0x1p-53-.5;}
int main(void){int bad=0;long count=0;double worst=0;char*dummy=calloc(1,65536);
'''
for skip in (0,1):
 for K in (16,32,48,64):
  t+=f'''{{const int K={K};double*a=guard(K*128),*b=guard(K*768),*c=guard(16*768);double*ref=malloc(16*768);
  for(int rep=0;rep<12;rep++){{double scale=rep%3==0?0x1p400:rep%3==1?0x1p-400:1;
   for(int k=0;k<K;k++)for(int i=0;i<16;i++)a[k*16+i]=(k>=K-8&&i<8)?0:val(rep*891+k*16+i+1)*scale;
   for(int k=0;k<K;k++)for(int j=0;j<96;j++)b[k*96+j]=val(rep*1311+k*96+j+919)/scale;
   for(int i=0;i<16;i++)for(int j=0;j<96;j++){{c[i*96+j]=ref[i*96+j]=37.125;if(j>=64){{double sum=0;for(int k=0;k<K;k++)sum=fma(a[k*16+i],b[k*96+j],sum);ref[i*96+j]=sum;}}}}
   pf_t pf={{dummy,12,0,dummy,dummy,768,&g_sw_none}};sme_start();sme_zero_za();odiag{skip}_{K}(a,b+64,&pf);za_owner_dstore(c+64);sme_stop();
   for(int i=0;i<16*96;i++){{double err=fabs(c[i]-ref[i]);bad+=!isfinite(err)||err!=0;worst=fmax(worst,err);count++;}}
  }}
  printf("ODIAG_CHECK skip={skip} K={K} comparisons=%ld error=%.3e %s\\n",count,worst,bad?"FAIL":"PASS");fflush(stdout);if(bad)return 1;free(ref);
 }}\n'''
t+='return 0;}\n';(p/'owner-diag48-boundary.c').write_text(t)
