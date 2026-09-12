from pathlib import Path
import json

p=Path(__file__).resolve().parent
s=(p/'trsm-v49-final.c').read_text()
ls=['ptrue p0.d','zero {za}','mov %[a],%[abase]','add %[a],%[a],#64','mov %[b],%[bbase]']

def initial():
    ls.append('ldnt1d z0.d,p0/z,[%[a]]')
    ls.extend(f'ld1d z{4+j}.d,p0/z,[%[b],#{j},mul vl]' for j in range(4))
    ls.extend(['add %[a],%[a],#128','add %[b],%[b],#768'])

def step(k,bank,last=False,host=False,prefetch=False):
    ar=0 if k%2==0 else 8
    br=4 if k%2==0 else 12
    na=8 if k%2==0 else 0
    nb=12 if k%2==0 else 4
    if host:ls.append(f'mov w12,#{k}')
    if not last:ls.append(f'ldnt1d z{na}.d,p0/z,[%[a]]')
    for j in range(4):
        if not last:ls.append(f'ld1d z{nb+j}.d,p0/z,[%[b],#{j},mul vl]')
        if host:ls.append(f'mova z{16+j}.d,p0/m,za{4+j}h.d[w12,0]')
        ls.append(f'fmopa za{bank+j}.d,p0/m,p0/m,z{ar}.d,z{br+j}.d')
        if host:ls.append(f'st1d z{16+j}.d,p0,[%[out],#{j},mul vl]')
    if not last:ls.extend(['add %[a],%[a],#128','add %[b],%[b],#768'])
    if host:ls.append('add %[out],%[out],#768')
    if prefetch and k%2==0:
        ls.extend(['prfm pldl1keep,[%[xpf]]','add %[xpf],%[xpf],#64',
            'tbnz %[pfn],#40,8f','subs %[pfc],%[pfc],#1','b.gt 8f',
            'add %[xpf],%[xpf],%[pfj]','mov %[pfc],%[pfn]','8:',
            'prfm pldl1keep,[%[xpf2]]','add %[xpf2],%[xpf2],#64'])
    elif prefetch and not last:
        ls.extend(f'prfm pldl2keep,[%[xpc],#{j*64}]' for j in range(4))
        ls.append('add %[xpc],%[xpc],#768')

initial()
ls.append('1:')
for k in range(8):step(k,4,prefetch=True)
ls.extend(['subs %[bottom],%[bottom],#1','b.ne 1b'])
for k in range(8):step(k,4,last=k==7,prefetch=True)
ls.extend(['mov %[a],%[abase]','mov %[b],%[bbase]'])
initial()
ls.extend(['cbz %[prefix],3f','2:'])
for k in range(8):step(k,0)
ls.extend(['subs %[prefix],%[prefix],#1','b.ne 2b','3:'])
for k in range(8):step(k,0,last=k==7,host=True)
ls.extend(['sub %[out],%[out],#3,lsl #12'])
for r in range(8):
    ls.append(f'mov w12,#{r}')
    ls.extend(f'mova z{16+j}.d,p0/m,za{j}h.d[w12,0]' for j in range(4))
    ls.extend(f'st1d z{16+j}.d,p0,[%[out],#{j},mul vl]' for j in range(4))
    ls.append('add %[out],%[out],#768')

helper='''// Bottom rows are computed first. Their stores cannot affect top rows:
// the last eight input rows have zero coefficients in the top half.
static inline __attribute__((always_inline)) void odiagduplex50(const double*abase,
 const double*bbase,double*out,long K,const pf_t*pf){
 const double*a,*b;out+=8*96;long bottom=(K-8)/8,prefix=(K-16)/8;
 const char*xpf=pf->p,*xpf2=pf->p2,*xpc=pf->pc;long pfc=pf->pfn;
 asm volatile(
'''+''.join(' '+json.dumps(line+'\n\t')+'\n' for line in ls)+'''
 :[a]"=&r"(a),[b]"=&r"(b),[out]"+&r"(out),[bottom]"+&r"(bottom),[prefix]"+&r"(prefix),
 [xpf]"+&r"(xpf),[xpf2]"+&r"(xpf2),[xpc]"+&r"(xpc),[pfc]"+&r"(pfc)
 :[abase]"r"(abase),[bbase]"r"(bbase),[pfn]"r"(pf->pfn),[pfj]"r"(pf->pfj)
 :"memory","cc","x12",VCLOB,PCLOB);}
'''
s=s.replace('// ---------------------------------------------------------------- 任务执行',helper+'\nstatic int g_odiagduplex;\n// ---------------------------------------------------------------- 任务执行',1)
s=s.replace('if(fast_owner==2){','if(fast_owner>=2){',1)
old='''                    PROF_T(c, kloop_plain_nt(A, 128, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);
                    PROF_T(c, za_owner_dstore(cb + 16L * s * ncp + 32 * t), t_ddump);'''
assert old in s
s=s.replace(old,'''                    if(fast_owner==3){PROF_T(c,odiagduplex50(A,cb+32*t,cb+16L*s*ncp+32*t,K,&pf),t_dkloop);}
                    else{
'''+old+'\n                    }',1)
needle='{ return run_task_impl(c,tb,tk,guess,tid,2); }'
assert needle in s
s=s.replace(needle,needle+'''
static long run_task_owner_diagduplex(const ctx_t*c,tbuf_t*tb,const task_t*tk,const task_t*guess,int tid)
{return run_task_impl(c,tb,tk,guess,tid,3);}
''',1)
s=s.replace('    g_owner_cursor=env_int','    g_odiagduplex=env_int("TRSM_ODIAGDUP",0);\n    g_owner_cursor=env_int',1)
s=s.replace('(g_owner_cursor ? run_task_owner_cursor : run_task_owner)',
 '(g_odiagduplex ? run_task_owner_diagduplex : g_owner_cursor ? run_task_owner_cursor : run_task_owner)',1)
(p/'trsm-owner-diagduplex50.c').write_text(s)

guard=r'''#include "trsm-owner-diagduplex50.c"
static double*guard(size_t bytes){size_t n=(bytes+4095)&~4095UL;char*b=mmap(0,n+4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(b==MAP_FAILED||mprotect(b+n,4096,PROT_NONE))exit(2);return(double*)(b+n-bytes);}
static double val(uint64_t i){i=(i^(i>>30))*UINT64_C(0xbf58476d1ce4e5b9);i=(i^(i>>27))*UINT64_C(0x94d049bb133111eb);i^=i>>31;return(i>>11)*0x1p-53-.5;}
int main(void){int bad=0;long count=0;double worst=0;char*dummy=calloc(1,65536);
for(int K=16;K<=64;K+=16)for(int alias=0;alias<2;alias++){
 double*a=guard(K*128),*b=guard(K*768),*separate=guard(16*768);
 double*ref=malloc(K*768),*saved=malloc(K*768);
 for(int rep=0;rep<12;rep++){double scale=rep%3==0?0x1p400:rep%3==1?0x1p-400:1;
  for(int k=0;k<K;k++)for(int i=0;i<16;i++)a[k*16+i]=(k>=K-8&&i<8)?0:val(rep*891+k*16+i+1)*scale;
  for(int k=0;k<K;k++)for(int j=0;j<96;j++)b[k*96+j]=val(rep*1311+k*96+j+919)/scale;
  memcpy(saved,b,K*768);memcpy(ref,b,K*768);for(int i=0;i<16*96;i++)separate[i]=37.125;
  double*out=alias?b+(K-16)*96:separate;
  if(!alias)for(int i=0;i<16*96;i++)ref[i]=37.125;
  for(int i=0;i<16;i++)for(int j=64;j<96;j++){double sum=0;for(int k=0;k<K;k++)sum=fma(a[k*16+i],saved[k*96+j],sum);ref[((alias?K-16:0)+i)*96+j]=sum;}
  pf_t pf={dummy,12,0,dummy,dummy,768,&g_sw_none};sme_start();odiagduplex50(a,b+64,out+64,K,&pf);sme_stop();
  double*result=alias?b:separate;int n=(alias?K:16)*96;
  for(int i=0;i<n;i++){double err=fabs(result[i]-ref[i]);bad+=!isfinite(err)||err!=0;worst=fmax(worst,err);count++;}
 }
 printf("ODIAGDUP_CHECK K=%d alias=%d comparisons=%ld error=%.3e %s\n",K,alias,count,worst,bad?"FAIL":"PASS");fflush(stdout);if(bad)return 1;
 free(ref);free(saved);
}return 0;}
'''
(p/'owner-diagduplex50-boundary.c').write_text(guard)
(p/'403-owner-diagduplex50.sh').write_text('''set -euo pipefail
flags='-O3 -march=armv9-a+sme+sme-f64f64+sve2 -fopenmp'
gcc $flags owner-diagduplex50-boundary.c -o owner-diagduplex50-boundary -lm
OMP_NUM_THREADS=1 numactl -N 1 --preferred=17 taskset -c 40 ./owner-diagduplex50-boundary
gcc $flags bench_trsm.c trsm-owner-diagduplex50.c -o owner-diagduplex50 -lm -lkblas
for round in 1 2 3 4;do
 order='0 1';if [ "$((round % 2))" = 0 ];then order='1 0';fi
 for flag in $order;do
  echo OWNER_DIAGDUP50_SCORE round=$round flag=$flag
  TRSM_ODIAGDUP=$flag numactl -N 1 ./owner-diagduplex50 512 19968 11
 done
done
for flag in 0 1;do
 echo OWNER_DIAGDUP50_PROFILE flag=$flag
 TRSM_ODIAGDUP=$flag TRSM_PROFILE=2 numactl -N 1 ./owner-diagduplex50 512 19968 5
done
''')
