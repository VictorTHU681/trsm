from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'trsm-owner-hoist48.c').read_text();clones='';protos=''
for tag,family in [('pf','pfcursor'),('all','allcursor')]:
 old=(p/f'trsm-hoist-{tag}45.c').read_text()
 for bank in (0,4):
  name=f'kloop_owner_bank{bank}';a=old.index(f'static inline __attribute__((always_inline)) void {name}(const double* a');b=old.index('PCLOB);}',a)+len('PCLOB);}')
  clones+=old[a:b].replace(name,f'kloop_owner_{family}{bank}')+'\n';protos+=f'static inline __attribute__((always_inline)) void kloop_owner_{family}{bank}(const double*,const double*,double*,const char*,const char*,const char*);\n'
s=s.replace('static int g_owner_hoist;',protos+'static int g_owner_hoist;',1)
a=s.index('                if(fast_owner==2){');b=s.index('                pend = 1;',a)
part=s[a:b];orig=part[part.index('                }else{\n')+len('                }else{\n'):part.rindex('                }\n')]
extra=''
for mode,family in [(3,'pfcursor'),(4,'allcursor')]:extra+=f'                }}else if(fast_owner=={mode}){{\n'+orig.replace('kloop_owner_bank0',f'kloop_owner_{family}0').replace('kloop_owner_bank4',f'kloop_owner_{family}4')
part=part.replace('                }else{\n',extra+'                }else{\n',1);s=s[:a]+part+s[b:]
needle='{ return run_task_impl(c,tb,tk,guess,tid,2); }'
s=s.replace(needle,needle+'\nstatic long run_task_owner_pfcursor(const ctx_t*c,tbuf_t*tb,const task_t*tk,const task_t*guess,int tid)\n{ return run_task_impl(c,tb,tk,guess,tid,3); }\nstatic long run_task_owner_allcursor(const ctx_t*c,tbuf_t*tb,const task_t*tk,const task_t*guess,int tid)\n{ return run_task_impl(c,tb,tk,guess,tid,4); }',1)
s=s.replace('// ---------------------------------------------------------------- 主流程',clones+'\n// ---------------------------------------------------------------- 主流程',1)
s=s.replace('(g_owner_hoist ? run_task_owner_hoist : run_task_owner)','(g_owner_hoist==1 ? run_task_owner_hoist : g_owner_hoist==2 ? run_task_owner_pfcursor : g_owner_hoist==3 ? run_task_owner_allcursor : run_task_owner)',1)
(p/'trsm-owner-cursors48.c').write_text(s)
