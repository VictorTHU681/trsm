#!/usr/bin/env python3
from pathlib import Path
import json

here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'trsm-fold.c').read_text()

def kernel(name,ld,prefetch):
    lines=['ptrue p0.d']
    def loads(reg):
        return [f'{ld} z{reg[0]}.d,p0/z,[%[a]]',f'{ld} z{reg[1]}.d,p0/z,[%[a],#1,mul vl]']+[
            f'ld1d z{reg[2+j]}.d,p0/z,[%[b],#{j},mul vl]' for j in range(4)]
    def fmas(reg):
        return [f'fmopa za{r*4+c}.d,p0/m,p0/m,z{reg[r]}.d,z{reg[2+c]}.d' for r in range(2) for c in range(4)]
    even,odd=(0,1,4,5,6,7),(8,9,12,13,14,15)
    def step(old,new,k):
        ld,fm=loads(new),fmas(old)
        seq=[ld[0],fm[0],fm[1],ld[1],fm[2],fm[3],ld[2],fm[4],ld[3],fm[5],ld[4],fm[6],ld[5],fm[7]]
        if prefetch and k%2==0:
            seq+=['prfm pldl2keep,[%[a],#2048]','prfm pldl2keep,[%[a],#2112]']
            if prefetch==1:seq+=['prfm pldl2keep,[%[b],#12288]','prfm pldl2keep,[%[b],#12416]']
        return seq+['add %[a],%[a],#128','add %[b],%[b],#768']
    lines+=loads(even)+['add %[a],%[a],#128','add %[b],%[b],#768','1:']
    for k in range(7):lines+=step(even if k%2==0 else odd,odd if k%2==0 else even,k)
    lines+=['subs %[groups],%[groups],#1','b.eq 2f']+step(odd,even,7)+['b 1b','2:']+fmas(odd)
    return f'''static inline __attribute__((always_inline)) void {name}(const double* a,const double* b,long K){{
    long groups=K/8;
    asm volatile(
'''+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in lines)+'''
        :[a]"+&r"(a),[b]"+&r"(b),[groups]"+&r"(groups)::"memory","cc",VCLOB,PCLOB);
}
'''
definitions='static int g_fold_kernel;\n'
for name,ld,pf in [('fold_nt','ldnt1d',0),('fold_keep','ld1d',0),('fold_pf','ld1d',1),('fold_pfa','ld1d',2)]:
    definitions+=kernel(name,ld,pf)
marker='static inline __attribute__((always_inline)) long run_task_impl('
assert s.count(marker)==1
s=s.replace(marker,definitions+'\n'+marker)
marker='    g_fold_owner=env_int('
s=s.replace(marker,'    g_fold_kernel=env_int("TRSM_FOLD_KERNEL",0);\n'+marker)
s=s.replace('&& c.cshift==0 && c.ddirect && !c.nobar;', '&& c.cshift==0 && c.ddirect;')
old='''                    PROF_T(c, kloop_plain_nt(A,128,128,(fast_owner==3 ? tb->cbuf : cb)+32*t,cs,K,cols,&pf),t_dkloop);'''
assert s.count(old)==1
s=s.replace(old,'''                    const double* source=(fast_owner==3 ? tb->cbuf : cb)+32*t;
                    if(fast_owner==3 && g_fold_kernel==1 && K>256){PROF_T(c,kloop_plain(A,128,128,source,cs,K,cols,&pf),t_dkloop);}
                    else if(fast_owner==3 && g_fold_kernel==2){PROF_T(c,fold_nt(A,source,K),t_dkloop);}
                    else if(fast_owner==3 && g_fold_kernel==3){PROF_T(c,fold_keep(A,source,K),t_dkloop);}
                    else if(fast_owner==3 && g_fold_kernel==4){PROF_T(c,fold_pf(A,source,K),t_dkloop);}
                    else if(fast_owner==3 && g_fold_kernel==5){PROF_T(c,fold_pfa(A,source,K),t_dkloop);}
                    else{PROF_T(c,kloop_plain_nt(A,128,128,source,cs,K,cols,&pf),t_dkloop);}''')
(out/'trsm-fold-kernels.c').write_text(s)
(out/'fold-boundary.c').write_text(r'''#include "trsm-fold-kernels.c"
static double* guard(size_t bytes){size_t n=(bytes+4095)&~4095UL;char*b=mmap(0,n+4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(b==MAP_FAILED||mprotect(b+n,4096,PROT_NONE))exit(2);return (double*)(b+n-bytes);}
static double value(uint64_t i){i=(i^(i>>30))*UINT64_C(0xbf58476d1ce4e5b9);i=(i^(i>>27))*UINT64_C(0x94d049bb133111eb);i^=i>>31;return (i>>11)*0x1p-53-.5;}
int main(void){
 const int sizes[]={16,32,48,64,96,192,256,384,512,1024};long count=0;
 for(int z=0;z<10;z++)for(int mode=2;mode<=5;mode++){
  int K=sizes[z];double*a=guard(K*128),*b=guard(K*768),*c=guard(16*768),*ref=malloc(16*768);
  for(int rep=0;rep<4;rep++){
   double scale=rep==1?0x1p400:rep==2?0x1p-400:1;
   for(int k=0;k<K;k++)for(int i=0;i<16;i++)a[k*16+i]=value(k*16+i+17+rep*733)*scale;
   for(int k=0;k<K;k++)for(int j=0;j<96;j++)b[k*96+j]=value(k*96+j+831+rep*913)/scale;
   for(int i=0;i<16;i++)for(int j=0;j<96;j++){
    c[i*96+j]=ref[i*96+j]=37.125;
    if(j>=64){double acc=0;for(int k=0;k<K;k++)acc=fma(a[k*16+i],b[k*96+j],acc);ref[i*96+j]=acc;}
   }
   sme_start();sme_zero_za();
   if(mode==2)fold_nt(a,b+64,K);else if(mode==3)fold_keep(a,b+64,K);else if(mode==4)fold_pf(a,b+64,K);else fold_pfa(a,b+64,K);
   za_owner_dstore(c+64);sme_stop();
   for(int q=0;q<16*96;q++){if(!isfinite(c[q])||c[q]!=ref[q]){printf("FOLD_BOUNDARY FAIL mode=%d K=%d rep=%d q=%d\n",mode,K,rep,q);return 1;}count++;}
  }
  printf("FOLD_BOUNDARY PASS mode=%d K=%d comparisons=%ld\n",mode,K,count);
  free(ref);
 }
 return 0;
}
''')
