#!/usr/bin/env python3
"""Per-call 16-row left preconditioning, with the existing owner scheduling."""
from pathlib import Path
import hashlib
import subprocess

here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
out.mkdir(exist_ok=True)
def pinned(name):return subprocess.check_output(['git','show','36a092c:'+name],cwd=root).decode()
s=pinned('trsm.c')
assert hashlib.sha256(s.encode()).hexdigest()=='936704cb25f79f9b497e33e1e9ba7ceaa972173421dd3863be049fb673dc0589'
(out/'trsm-v54.c').write_text(s)
(out/'bench_trsm.c').write_text(pinned('bench_trsm.c'))
def replace(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
def replace_run(old,new):
    global s
    start=s.index('static inline __attribute__((always_inline)) long run_task_impl(')
    end=s.index('static long run_task(',start)
    body=s[start:end]
    assert body.count(old)==1,(old,body.count(old))
    s=s[:start]+body.replace(old,new)+s[end:]
marker='static void diag_prep_sub('
replace(marker,'''static int g_fold_owner;
static double* g_fold_coeff;
static size_t g_fold_cap;

'''+marker)
end='''d[q * 16 + r] = Linv[r][q];
}'''
replace(end,'''d[q * 16 + r] = Linv[r][q];
    if(g_fold_owner){
        // Compute -inv(Lss)*L_left from this call's L, extending the existing
        // within-block transformation to all preceding row blocks.
        double* dest=g_fold_coeff+diag_strip_off((int)(r0/16));
        for(long col0=0;col0<i0;col0+=16){
            double block[16][16],product[16][16];
            for(int t=0;t<16;t++)for(int q=0;q<16;q++)
                block[t][q]=L[(r0+t)*lda+col0+q];
            for(int r=0;r<16;r++){
                for(int q=0;q<16;q++)product[r][q]=0;
                for(int t=0;t<=r;t++){
                    double a=Linv[r][t];
                    for(int q=0;q<16;q++)product[r][q]+=a*block[t][q];
                }
            }
            for(int q=0;q<16;q++)for(int r=0;r<16;r++)
                dest[(col0+q)*16+r]=-product[r][q];
        }
        memcpy(dest+i0*16,pk,(size_t)(s+1)*256*sizeof(double));
    }
}''')
marker='    g_align_owner_buffers = c.owner ||'
replace(marker,'''    g_fold_owner=env_int("TRSM_FOLD_OWNER",0) && c.special==5 && m<=1024
        && c.cshift==0 && c.ddirect && !c.nobar;
    if(env_int("TRSM_EXPECT_FOLD",-1)>=0 && g_fold_owner!=env_int("TRSM_EXPECT_FOLD",-1))abort();
'''+marker)
marker='    c.diag = g_diag;'
replace(marker,marker+'''
    if(g_fold_owner){
        size_t need=diag_block_doubles((int)(m/16));
        if(g_fold_cap<need){
            trsm_free_buffer(g_fold_coeff,g_fold_cap*8);
            g_fold_coeff=alloc_on(need*8,-1);g_fold_cap=need;
        }
    }''')
replace_run('    for (long q = 0; q < nstrips; q++) {',
        '    for (long q = 0; q < (fast_owner==3 ? 0 : nstrips); q++) {')
replace_run('''            const double* A = D + diag_strip_off(s);
            long K = 16L * (s + 1);''',
'''            const double* A = fast_owner==3 ? g_fold_coeff+diag_strip_off((int)(i0/16)+s) : D+diag_strip_off(s);
            long K = (fast_owner==3 ? i0 : 0) + 16L*(s+1);''')
replace_run('''                if (pf_on && s + 1 < S) pf.p2 = (const char*)(D + diag_strip_off(s + 1)) + t * K * 32;''',
'''                if (pf_on && s + 1 < S) pf.p2 = (const char*)(fast_owner==3
                    ? g_fold_coeff+diag_strip_off((int)(i0/16)+s+1) : D+diag_strip_off(s+1)) + t*K*32;''')
replace_run('''                    PROF_T(c, kloop_plain_nt(A, 128, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);''',
'''                    PROF_T(c, kloop_plain_nt(A,128,128,(fast_owner==3 ? tb->cbuf : cb)+32*t,cs,K,cols,&pf),t_dkloop);''')
marker='''{ return run_task_impl(c,tb,tk,guess,tid,2); }'''
replace(marker,marker+'''
static long run_task_owner_fold(const ctx_t* c,tbuf_t* tb,const task_t* tk,const task_t* guess,int tid)
{ return run_task_impl(c,tb,tk,guess,tid,3); }''')
replace('(g_owner_cursor ? run_task_owner_cursor : run_task_owner)',
        '(g_fold_owner ? run_task_owner_fold : g_owner_cursor ? run_task_owner_cursor : run_task_owner)')
(out/'trsm-fold.c').write_text(s)
driver=pinned('probe/v50_20260912/independent-owner-switch.c')
driver=driver.replace('setenv("TRSM_OWNER",reuse?"0":"1",1);', '''setenv("TRSM_OWNER","1",1);
            setenv("TRSM_FOLD_OWNER",reuse?"0":"1",1);
            setenv("TRSM_EXPECT_FOLD",reuse?"0":"1",1);''')
driver=driver.replace('setenv("TRSM_EXPECT_OWNER",reuse?"0":"1",1);','setenv("TRSM_EXPECT_OWNER","1",1);')
(out/'independent-fold.c').write_text(driver)
