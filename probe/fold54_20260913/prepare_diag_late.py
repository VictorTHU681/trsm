#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'trsm-diag-final.c').read_text()
def replace(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
replace('static int g_diag_sme,g_diag_row;', 'static int g_diag_sme,g_diag_row,g_late_diag;')
replace('''        if (c->ppmode == 1) nitems += c->nstrip;   // 打包任务''',
'''        if (c->ppmode == 1) nitems += c->nstrip;   // 打包任务
        if(g_late_diag)nitems+=c->nstrip;''')
replace('''        } else
        for (long i = 0; i < c->nrb; i++) {
            if (c->ppmode == 1) {''',
'''        } else
        for (long i = 0; i < c->nrb; i++) {
            if(g_late_diag)for(int ds=0;ds<c->S_of[i];ds++)
                TASK_INIT(&g_tasks[q++],2,ds,(int)i,0,0,1,0,0,0);
            if (c->ppmode == 1) {''')
replace('    c.nobar = env_int("TRSM_NOBAR", 1);',
'''    c.nobar = env_int("TRSM_NOBAR", 1);
    g_late_diag=env_int("TRSM_LATE_DIAG",0) && c.nobar && c.ppmode==1 && !c.owner;
    if(env_int("TRSM_EXPECT_LATE",-1)>=0 && g_late_diag!=env_int("TRSM_EXPECT_LATE",-1))abort();''')
replace('''    long nprep_items = ndiag_items + ((c.nobar && prep_pack) ? c.nstrip : 0);''',
'''    long nprep_items = g_late_diag ? 0 : ndiag_items + ((c.nobar && prep_pack) ? c.nstrip : 0);''')
replace('''        long q = 0;
        for (long i = 0; i < c.nrb; i++) {
            for (int s = 0; s < c.S_of[i]; s++) { ditem_i[q] = i; ditem_s[q] = s; q++; }''',
'''        long q = 0;
        for (long i = 0; i < (g_late_diag ? 0 : c.nrb); i++) {
            for (int s = 0; s < c.S_of[i]; s++) { ditem_i[q] = i; ditem_s[q] = s; q++; }''')
marker='// ---------------------------------------------------------------- 主流程'
helper='''static void run_late_diag(const ctx_t* c,tbuf_t* tb,const task_t* tk)
{
    uint64_t t0=c->prof>=2?cyc():0;
    long i=tk->i,mbi=c->m-i*c->mb;
    if(mbi>c->mb)mbi=c->mb;
    // Ordinary preprocessing may use NEON; it is called outside streaming
    // mode. No live ZA output crosses task boundaries in the task dispatcher.
    sme_stop();
    diag_prep_sub(c->L,c->lda,i*c->mb,(int)mbi,(int)tk->g,
                  c->diag+c->diag_off[i]+diag_strip_off((int)tk->g));
    sme_start();
    __atomic_fetch_add(&c->dcnt[i],1,__ATOMIC_RELEASE);
    // This experimental pipeline accounts late preprocessing in the pack
    // bucket; it must be included when comparing full phase totals.
    if(c->prof>=2)tb->t_pack+=cyc()-t0;
}

'''
replace(marker,helper+marker)
old='''            long tn = (fullpan ? taskfn : run_task)(&c, &g_tb[tid], tsk, tg < c.ntask_items ? &c.tasks[tg] : NULL, tid);'''
new='''            long tn=-1;
            if(tsk->kind==2)run_late_diag(&c,&g_tb[tid],tsk);
            else tn = (fullpan ? taskfn : run_task)(&c, &g_tb[tid], tsk, tg < c.ntask_items ? &c.tasks[tg] : NULL, tid);'''
replace(old,new)
(out/'trsm-diag-late.c').write_text(s)
