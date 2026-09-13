#!/usr/bin/env python3
from pathlib import Path
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=subprocess.check_output(['git','show','36a092c:trsm.c'],cwd=root).decode()
def clone(start,end,name,newname):
    global s
    body=s[s.index(start):s.index(end,s.index(start))]
    body=body.replace(name,newname).replace('PROF_T(', 'NOPROF_T(').replace('c->prof','0')
    s=s.replace(end,body+end,1)
s=s.replace('#define PROF_T(c, expr, field)', '#define NOPROF_T(c,expr,field) do { expr; } while(0)\n#define PROF_T(c, expr, field)',1)
clone('static inline __attribute__((always_inline)) long run_task_impl(',
      'static long run_task(', 'run_task_impl','run_task_noinstr_impl')
marker='static long run_task_owner_cursor('
new='''static int g_noinstr;
static long run_task_owner_noinstr(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid)
{return run_task_noinstr_impl(c,tb,tk,guess,tid,2);}
'''
s=s.replace(marker,new+marker,1)
clone('static inline __attribute__((always_inline)) long run_task_special_impl(',
      'static long run_task_special1(', 'run_task_special_impl','run_task_special_noinstr_impl')
marker='static long run_task_hoist4('
new='''static long run_task_noinstr2(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid)
{return run_task_special_noinstr_impl(c,tb,tk,guess,tid,2,1);}
static long run_task_noinstr4(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid)
{return run_task_special_noinstr_impl(c,tb,tk,guess,tid,4,1);}
'''
s=s.replace(marker,new+marker,1)
s=s.replace('    c.prof = env_int("TRSM_PROFILE", 0);',
            '    c.prof = env_int("TRSM_PROFILE", 0);\n    g_noinstr=env_int("TRSM_NOINSTR",0) && c.prof==0;',1)
s=s.replace('(g_owner_cursor ? run_task_owner_cursor : run_task_owner)',
            '(g_owner_cursor ? (g_noinstr ? run_task_owner_noinstr : run_task_owner_cursor) : run_task_owner)')
s=s.replace('g_abhoist ? run_task_hoist2 : run_task_special2',
            'g_abhoist ? (g_noinstr ? run_task_noinstr2 : run_task_hoist2) : run_task_special2')
s=s.replace('g_abhoist ? run_task_hoist4 : run_task_special4',
            'g_abhoist ? (g_noinstr ? run_task_noinstr4 : run_task_hoist4) : run_task_special4')
(out/'trsm-noinstr.c').write_text(s)
