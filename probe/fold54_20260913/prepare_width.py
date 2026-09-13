#!/usr/bin/env python3
from pathlib import Path
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=subprocess.check_output(['git','show','36a092c:trsm.c'],cwd=root).decode()
def replace(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
marker='static inline long pan_c0('
helper='''static int g_width_owner;
static long* g_width_start;
static int* g_width_size;
static long g_width_bound[1025];
static size_t g_width_capacity;
static void prepare_owner_width(ctx_t* c,int nth)
{
    long units=c->n/16,base=units/nth,extra=units%nth,np=0;
    for(int t=0;t<nth;t++)np+=(base+(t<extra)+5)/6;
    if(g_width_capacity<(size_t)np){
        free(g_width_start);free(g_width_size);
        g_width_start=malloc((size_t)np*sizeof(long));
        g_width_size=malloc((size_t)np*sizeof(int));
        if(!g_width_start || !g_width_size)abort();
        g_width_capacity=np;
    }
    long q=0,col=0;
    for(int t=0;t<nth;t++){
        g_width_bound[t]=q*c->nrb;
        long left=base+(t<extra);
        while(left){
            int amount=left>6?6:(int)left;
            g_width_start[q]=col;g_width_size[q]=16*amount;
            q++;col+=16*amount;left-=amount;
        }
    }
    c->npan=q;c->ntask=q*c->nrb;g_width_bound[nth]=c->ntask;
    if(col!=c->n)abort();
}

'''
replace(marker,helper+marker)
replace('{ return p ? p * c->nc - c->cshift : 0; }',
        '{ return g_width_owner ? g_width_start[p] : p ? p*c->nc-c->cshift : 0; }')
replace('''static inline long pan_nci(const ctx_t* c, long p)
{''','''static inline long pan_nci(const ctx_t* c, long p)
{
    if(g_width_owner)return g_width_size[p];''')
replace('    g_implicit_owner = env_int(','''    g_width_owner=env_int("TRSM_WIDTH_OWNER",0) && c.special==5 && c.m<=1024
        && c.cshift==0 && c.ddirect && g_owner_cursor && env_int("TRSM_IMPLICIT_OWNER",1);
    if(g_width_owner)prepare_owner_width(&c,nth);
    if(env_int("TRSM_EXPECT_WIDTH",-1)>=0 && g_width_owner!=env_int("TRSM_EXPECT_WIDTH",-1))abort();
    g_implicit_owner = env_int(''')
replace('''static void build_tasks(ctx_t* c, int nth)
{''','''static void build_tasks(ctx_t* c, int nth)
{
    if(g_width_owner){
        c->owner_bound=malloc(sizeof(long)*(nth+1));
        if(!c->owner_bound)abort();
        memcpy(c->owner_bound,g_width_bound,sizeof(long)*(nth+1));
        c->tasks=NULL;c->ntask_items=c->ntask;c->part=NULL;return;
    }''')
start=s.index('static inline __attribute__((always_inline)) long run_task_impl(')
end=s.index('static long run_task(',start)
body=s[start:end]
old='    long nci = fast_owner ? 96 : pan_nci(c, p); if (nci > nc) nci = nc;'
assert body.count(old)==1
body=body.replace(old,'    long nci = fast_owner>=3 ? 16*(fast_owner-2) : fast_owner ? 96 : pan_nci(c,p); if(nci>nc)nci=nc;')
body=body.replace('(fast_owner ? 6 : ntile)', '(fast_owner ? nci/16 : ntile)')
body=body.replace('if(fast_owner==2)', 'if(fast_owner>=2)')
s=s[:start]+body+s[end:]
marker='''{ return run_task_impl(c,tb,tk,guess,tid,2); }'''
funcs=''
for n in (16,32,48,64,80):
    funcs+=f'''\nstatic long run_task_owner_w{n}(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid)
{{return run_task_impl(c,tb,tk,guess,tid,{n//16+2});}}
'''
funcs+='''static long run_task_owner_width(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid)
{
    switch(g_width_size[tk->p]){
    case 16:return run_task_owner_w16(c,tb,tk,guess,tid);
    case 32:return run_task_owner_w32(c,tb,tk,guess,tid);
    case 48:return run_task_owner_w48(c,tb,tk,guess,tid);
    case 64:return run_task_owner_w64(c,tb,tk,guess,tid);
    case 80:return run_task_owner_w80(c,tb,tk,guess,tid);
    default:return run_task_owner_cursor(c,tb,tk,guess,tid);
    }
}
'''
replace(marker,marker+funcs)
replace('(g_owner_cursor ? run_task_owner_cursor : run_task_owner)',
        '(g_width_owner ? run_task_owner_width : g_owner_cursor ? run_task_owner_cursor : run_task_owner)')
(out/'trsm-width.c').write_text(s)
driver=subprocess.check_output(['git','show','36a092c:probe/v50_20260912/independent-owner-switch.c'],cwd=root).decode()
driver=driver.replace('setenv("TRSM_OWNER",reuse?"0":"1",1);', '''setenv("TRSM_OWNER","1",1);
            setenv("TRSM_WIDTH_OWNER",reuse?"0":"1",1);
            setenv("TRSM_EXPECT_WIDTH",reuse?"0":"1",1);''')
driver=driver.replace('setenv("TRSM_EXPECT_OWNER",reuse?"0":"1",1);','setenv("TRSM_EXPECT_OWNER","1",1);')
(out/'independent-width.c').write_text(driver)
