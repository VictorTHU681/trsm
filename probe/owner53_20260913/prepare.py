#!/usr/bin/env python3
"""v53 owner task construction experiments; no cross-call schedule reuse."""
from pathlib import Path
import hashlib
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
out.mkdir(exist_ok=True)
def pinned(name):
    return subprocess.check_output(['git','show','3b24f38:'+name],cwd=root).decode()
s=pinned('trsm.c')
assert hashlib.sha256(s.encode()).hexdigest()=='dc8c8b4907c37362bd4d3d108dd4a9e35474736e7570091e7a2b2a79dd84f294'
(out/'trsm-v53.c').write_text(s)
(out/'bench_trsm.c').write_text(pinned('bench_trsm.c'))
def replace(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
replace('    long* owner_bound;', '    int implicit_owner, stack_sched;\n    long* owner_bound;')
marker='static void build_tasks(ctx_t* c, int nth)\n{'
helper='''// Owner tasks have a regular (panel,row-block) index. Construct each actual
// task at its use site; all task fields and thread boundaries are fresh per call.
static inline __attribute__((always_inline)) task_t owner_task_at(const ctx_t* c, long q)
{
    task_t t = {0};
    t.i = (int)(q % c->nrb); t.p = (int)(q / c->nrb);
    t.nseg = 1; t.ch1 = (long)t.i * (c->mb / c->kc);
    return t;
}
static void build_owner_bounds(ctx_t* c, int nth)
{
    double c0 = env_int("TRSM_OWNER_C0", c->special == 5 && c->pf2 ? 160 : 192) * (double)c->mb * c->nc;
    double a = (double)c->mb*c->mb*c->nc;
    double b = 0.5*a+c0;
    double rowsum = a*c->nrb*(c->nrb-1)/2+b*c->nrb;
    long count=c->nrb*c->npan;
    double total=rowsum*c->npan;
    c->owner_bound=malloc(sizeof(long)*(nth+1));
    if(!c->owner_bound)abort();
    c->owner_bound[0]=0;c->owner_bound[nth]=count;
    for(int t=1;t<nth;t++){
        double target=total*t/nth;
        long lo=0,hi=count;
        while(lo<hi){
            long q=lo+(hi-lo)/2,r=q%c->nrb;
            double prefix=(q/c->nrb)*rowsum+a*r*(r-1)/2+b*r;
            if(prefix>=target)hi=q;else lo=q+1;
        }
        c->owner_bound[t]=lo;
    }
    c->tasks=NULL;c->ntask_items=count;c->part=NULL;
}
'''
replace(marker,helper+marker+'\n    if(c->implicit_owner){build_owner_bounds(c,nth);return;}')
marker='    g_align_owner_buffers ='
replace(marker,'''    c.implicit_owner = env_int("TRSM_IMPLICIT_OWNER",0) && c.special==5
        && c.cshift==0 && c.ppmode==0 && c.mb==64 && c.nc==96 && c.kc==64
        && c.m%64==0 && c.n%96==0 && env_int("TRSM_KHEADW",0)==0
        && env_int("TRSM_OWNER_C0",160)>=0;
    c.stack_sched = env_int("TRSM_STACK_SCHED",0) && c.owner;
    if(env_int("TRSM_EXPECT_IMPLICIT",-1)>=0 && c.implicit_owner!=env_int("TRSM_EXPECT_IMPLICIT",-1))abort();
'''+marker)
replace('        long* sched = NULL;','        long sched_stack[128];\n        long* sched = NULL;')
replace('            sched = cnt > 0 ? malloc(sizeof(long) * cnt) : NULL;',
'''            sched = cnt > 0 ? (c.stack_sched && cnt<=128 ? sched_stack : malloc(sizeof(long)*cnt)) : NULL;''')
old='''                    const task_t* tsk = &c.tasks[sched[s]];
                    const int fullpan = !c.cshift || (tsk->p > 0 && tsk->p < c.npan - 1);
                    (c.special == 5 && fullpan ? (g_owner_cursor ? run_task_owner_cursor : run_task_owner) : run_task)(&c, &g_tb[tid], tsk, s + 1 < k ? &c.tasks[sched[s + 1]] : NULL, tid);'''
new='''                    task_t current_task, next_task;
                    const task_t *tsk, *guess=NULL;
                    if(c.implicit_owner){
                        current_task=owner_task_at(&c,sched[s]);tsk=&current_task;
                        if(s+1<k){next_task=owner_task_at(&c,sched[s+1]);guess=&next_task;}
                    }else{
                        tsk=&c.tasks[sched[s]];
                        if(s+1<k)guess=&c.tasks[sched[s+1]];
                    }
                    const int fullpan = !c.cshift || (tsk->p > 0 && tsk->p < c.npan - 1);
                    (c.special == 5 && fullpan ? (g_owner_cursor ? run_task_owner_cursor : run_task_owner) : run_task)(&c, &g_tb[tid], tsk, guess, tid);'''
replace(old,new)
replace('        free(sched);','        if(sched!=sched_stack)free(sched);')
(out/'trsm-owner53.c').write_text(s)
for name in ('bounds_check.c',):
    if (here/name).exists():(out/name).write_bytes((here/name).read_bytes())
driver=pinned('probe/v50_20260912/independent-owner-switch.c')
driver=driver.replace('setenv("TRSM_OWNER",reuse?"0":"1",1);',
'''setenv("TRSM_OWNER","1",1);
            setenv("TRSM_IMPLICIT_OWNER",reuse?"0":"1",1);
            setenv("TRSM_EXPECT_IMPLICIT",reuse?"0":"1",1);
            setenv("TRSM_STACK_SCHED",reuse?"0":"1",1);''')
driver=driver.replace('setenv("TRSM_EXPECT_OWNER",reuse?"0":"1",1);','setenv("TRSM_EXPECT_OWNER","1",1);')
(out/'independent-owner53.c').write_text(driver)
