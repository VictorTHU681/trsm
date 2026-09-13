#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'trsm-diag-late.c').read_text()
s=s.replace('if(g_late_diag)nitems+=c->nstrip;', 'if(g_late_diag==1)nitems+=c->nstrip;')
s=s.replace('if(g_late_diag)for(int ds=0;ds<c->S_of[i];ds++)',
            'if(g_late_diag==1)for(int ds=0;ds<c->S_of[i];ds++)')
s=s.replace('g_late_diag=env_int("TRSM_LATE_DIAG",0) && c.nobar && c.ppmode==1 && !c.owner;',
            'g_late_diag=c.nobar && c.ppmode==1 && !c.owner ? env_int("TRSM_LATE_DIAG",0) : 0;')
old='''            else tn = (fullpan ? taskfn : run_task)(&c, &g_tb[tid], tsk, tg < c.ntask_items ? &c.tasks[tg] : NULL, tid);'''
new='''            else {
                tn = (fullpan ? taskfn : run_task)(&c, &g_tb[tid], tsk, tg < c.ntask_items ? &c.tasks[tg] : NULL, tid);
                if(g_late_diag==2 && tsk->kind==1){
                    task_t diag_task;
                    diag_task.i=(16*tsk->g)/c.mb;
                    diag_task.g=tsk->g%(c.mb/16);
                    run_late_diag(&c,&g_tb[tid],&diag_task);
                }
            }'''
assert s.count(old)==1
s=s.replace(old,new)
(out/'trsm-diag-fuse.c').write_text(s)
