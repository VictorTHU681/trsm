from pathlib import Path
import json
p=Path(__file__).resolve().parent;s=(p/'trsm-v48-synced.c').read_text();defs=''
for skip in (0,1):
 ls=['ptrue p0.d','ldnt1d z0.d,p0/z,[%[a]]','ldnt1d z1.d,p0/z,[%[a],#1,mul vl]']+[f'ld1d z{4+j}.d,p0/z,[%[b],#{j},mul vl]' for j in range(4)]+['add %[a],%[a],#128','add %[b],%[b],#768','1:']
 for section in (0,1):
  for k in range(8):
   ar=[0,1] if k%2==0 else [8,9];br=[4,5,6,7] if k%2==0 else [12,13,14,15];na=[8,9] if k%2==0 else [0,1];nb=[12,13,14,15] if k%2==0 else [4,5,6,7]
   tail=section==1;last=tail and k==7;full=not tail or not skip
   for za in range(8):
    if not last:
     if za==0 and full:ls.append(f'ldnt1d z{na[0]}.d,p0/z,[%[a]]')
     if za==2:ls.append(f'ldnt1d z{na[1]}.d,p0/z,[%[a],#1,mul vl]')
     if za>=4:ls.append(f'ld1d z{nb[za-4]}.d,p0/z,[%[b],#{za-4},mul vl]')
    if full or za>=4:ls.append(f'fmopa za{za}.d,p0/m,p0/m,z{ar[za//4]}.d,z{br[za%4]}.d')
   if not last:ls+=['add %[a],%[a],#128','add %[b],%[b],#768']
   if k%2==0:
    ls+=['prfm pldl1keep,[%[xpf]]','add %[xpf],%[xpf],#64','subs %[pfc],%[pfc],#1','b.gt 8f','add %[xpf],%[xpf],%[pfj]','mov %[pfc],%[pfn]','8:','prfm pldl1keep,[%[xpf2]]','add %[xpf2],%[xpf2],#64']
   elif not last:ls += [f'prfm pldl2keep,[%[xpc],#{j*64}]' for j in range(4)]+['add %[xpc],%[xpc],#768']
  if section==0:ls+=['subs %[blocks],%[blocks],#1','b.ne 1b']
 defs+=f'''static inline __attribute__((always_inline)) void odiagcompact{skip}(const double*a,const double*b,long K,const pf_t*pf){{
 const char*xpf=pf->p,*xpf2=pf->p2,*xpc=pf->pc;long pfc=pf->pfn,blocks=(K-8)/8;
 asm volatile(\n'''+''.join(' '+json.dumps(l+'\n\t')+'\n' for l in ls)+''' :[a]"+&r"(a),[b]"+&r"(b),[xpf]"+&r"(xpf),[xpf2]"+&r"(xpf2),[xpc]"+&r"(xpc),[pfc]"+&r"(pfc),[blocks]"+&r"(blocks):[pfn]"r"(pf->pfn),[pfj]"r"(pf->pfj):"memory","cc",VCLOB,PCLOB);}
'''
s=s.replace('// ---------------------------------------------------------------- 任务执行','static int g_odiag;\n'+defs+'\n// ---------------------------------------------------------------- 任务执行',1)
old='                    PROF_T(c, kloop_plain_nt(A, 128, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);'
new='''                    if(fast_owner==2){PROF_T(c,odiagcompact0(A,cb+32*t,K,&pf),t_dkloop);}
                    else if(fast_owner==3){PROF_T(c,odiagcompact1(A,cb+32*t,K,&pf),t_dkloop);}
                    else{
'''+old+'\n                    }';s=s.replace(old,new,1)
needle='{ return run_task_impl(c, tb, tk, guess, tid, 1); }';s=s.replace(needle,needle+'\nstatic long run_task_owner_diag0(const ctx_t*c,tbuf_t*tb,const task_t*tk,const task_t*guess,int tid)\n{return run_task_impl(c,tb,tk,guess,tid,2);}\nstatic long run_task_owner_diag1(const ctx_t*c,tbuf_t*tb,const task_t*tk,const task_t*guess,int tid)\n{return run_task_impl(c,tb,tk,guess,tid,3);}',1)
s=s.replace('    c.prof = env_int("TRSM_PROFILE", 0);','    c.prof = env_int("TRSM_PROFILE", 0);\n    g_odiag=env_int("TRSM_ODIAG",0);',1)
s=s.replace('(c.special == 5 ? run_task_owner : run_task)','(c.special == 5 ? (g_odiag==1?run_task_owner_diag0:g_odiag==2?run_task_owner_diag1:run_task_owner) : run_task)',1)
(p/'trsm-owner-diagcompact48.c').write_text(s)
s=(p/'owner-diag48-boundary.c').read_text().replace('trsm-owner-diag48.c','trsm-owner-diagcompact48.c')
for skip in (0,1):
 for K in (16,32,48,64):s=s.replace(f'odiag{skip}_{K}(a,b+64,&pf)',f'odiagcompact{skip}(a,b+64,K,&pf)')
(p/'owner-diagcompact48-boundary.c').write_text(s)
