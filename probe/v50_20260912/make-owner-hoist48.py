from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'trsm-v48-synced.c').read_text();old=(p/'trsm-hoist-ab45.c').read_text();clones='';protos=''
for bank in (0,4):
 name=f'kloop_owner_bank{bank}';a=old.index(f'static inline __attribute__((always_inline)) void {name}(const double* a');b=old.index('PCLOB);}',a)+len('PCLOB);}')
 clones+=old[a:b].replace(name,f'kloop_owner_hoist{bank}')+'\n'
 protos+=f'static inline __attribute__((always_inline)) void kloop_owner_hoist{bank}(const double*,const double*,double*,const char*,const char*,const char*);\n'
s=s.replace('// Full 16x32 diagonal results',protos+'static int g_owner_hoist;\n\n// Full 16x32 diagonal results',1)
a=s.index('                PROF_T(c, kloop_owner_bank0(');b=s.index('                pend = 1;',a);orig=s[a:b];new=orig.replace('kloop_owner_bank0','kloop_owner_hoist0').replace('kloop_owner_bank4','kloop_owner_hoist4');s=s[:a]+'                if(fast_owner==2){\n'+new+'                }else{\n'+orig+'                }\n'+s[b:]
needle='{ return run_task_impl(c, tb, tk, guess, tid, 1); }'
s=s.replace(needle,needle+'\nstatic long run_task_owner_hoist(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid)\n{ return run_task_impl(c,tb,tk,guess,tid,2); }',1)
s=s.replace('// ---------------------------------------------------------------- 主流程',clones+'\n// ---------------------------------------------------------------- 主流程',1)
s=s.replace('    c.prof = env_int("TRSM_PROFILE", 0);','    c.prof = env_int("TRSM_PROFILE", 0);\n    g_owner_hoist=env_int("TRSM_OWNER_HOIST",0);',1)
s=s.replace('(c.special == 5 ? run_task_owner : run_task)','(c.special == 5 ? (g_owner_hoist ? run_task_owner_hoist : run_task_owner) : run_task)',1)
(p/'trsm-owner-hoist48.c').write_text(s)
s=(p/'v47-boundary.c').read_text().replace('trsm-v47-codex.c','trsm-owner-hoist48.c').replace('kloop_owner_bank0','kloop_owner_hoist0').replace('kloop_owner_bank4','kloop_owner_hoist4');(p/'owner-hoist48-boundary.c').write_text(s)
