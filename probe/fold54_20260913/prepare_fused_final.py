#!/usr/bin/env python3
from pathlib import Path
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=(out/'trsm-diag-fuse.c').read_text()
s=s.replace('        if(g_late_diag==1)nitems+=c->nstrip;\n','')
s=s.replace('''            if(g_late_diag==1)for(int ds=0;ds<c->S_of[i];ds++)
                TASK_INIT(&g_tasks[q++],2,ds,(int)i,0,0,1,0,0,0);
''','')
s=s.replace('''    // Large triangular matrices benefit from faster per-call diagonal packing.
    // 0 retains the original inverse/transform for same-binary comparisons.
    g_diag_sme=env_int("TRSM_DIAG_SME",m>=4096);
    g_diag_row=g_diag_sme!=0;
''','')
old='''    g_late_diag=c.nobar && c.ppmode==1 && !c.owner ? env_int("TRSM_LATE_DIAG",0) : 0;
    if(env_int("TRSM_EXPECT_LATE",-1)>=0 && g_late_diag!=env_int("TRSM_EXPECT_LATE",-1))abort();'''
new='''    // Recompute each diagonal strip in the task that packs the same L rows.
    // 0 restores the original preprocessing; 2 forces this path for testing.
    int fused_mode=env_int("TRSM_FUSED_DIAG",1);
    g_late_diag=fused_mode && (m>=4096 || fused_mode>1) && c.nobar && c.ppmode==1 && !c.owner;
    g_diag_sme=g_diag_row=g_late_diag;'''
assert s.count(old)==1
s=s.replace(old,new)
s=s.replace('''            long tn=-1;
            if(tsk->kind==2)run_late_diag(&c,&g_tb[tid],tsk);
            else {
                tn =''','''            long tn;
            {
                tn =''')
s=s.replace('if(g_late_diag==2 && tsk->kind==1)', 'if(g_late_diag && tsk->kind==1)')
s=s.replace('g_late_diag','g_fused_diag').replace('run_late_diag','run_packed_diag')
s=s.replace('''    // This experimental pipeline accounts late preprocessing in the pack
    // bucket; it must be included when comparing full phase totals.''',
'''    // Pack time includes the diagonal work fused into this producer task.''')
(out/'trsm-fused-final.c').write_text(s)
old=subprocess.check_output(['git','show','36a092c:probe/v50_20260912/make-close-audit50.py'],cwd=root).decode()
body=old[old.index("key='static void diag_prep_sub"):old.index("(p/'405-close-audit50.sh')")]
body=body.replace('trsm-close-audit50.c','trsm-fused-audit.c').replace('close-input-audit50.c','fused-input-audit.c')
exec(compile(body,'pinned-close-audit-generator','exec'),{'s':s,'p':out})
a=(out/'trsm-fused-audit.c').read_text()
a=a.replace('static int g_diag_sme,g_diag_row,g_fused_diag;',
            'static int g_diag_sme,g_diag_row,g_fused_diag;\nstatic long audit_sme_transforms,audit_fused_tasks;')
a=a.replace('    double at[256],bt[256];','    __atomic_fetch_add(&audit_sme_transforms,1,__ATOMIC_RELAXED);\n    double at[256],bt[256];')
marker='''static void run_packed_diag(const ctx_t* c,tbuf_t* tb,const task_t* tk)
{'''
assert a.count(marker)==1
a=a.replace(marker,marker+'\n    __atomic_fetch_add(&audit_fused_tasks,1,__ATOMIC_RELAXED);')
a=a.replace('    audit_diag_count=audit_pack_count=0;',
            '    audit_diag_count=audit_pack_count=0;\n    audit_sme_transforms=audit_fused_tasks=0;')
a=a.replace('    long expected_pack=0;', '''    long expected_sme=g_diag_sme?ndiag_items-c.nrb:0;
    long expected_fused=g_fused_diag?ndiag_items:0;
    int fused_ok=audit_sme_transforms==expected_sme && audit_fused_tasks==expected_fused;
    printf("FUSED_DIAG_AUDIT m=%ld n=%ld transforms=%ld expected_transforms=%ld tasks=%ld expected_tasks=%ld %s\\n",m,n,audit_sme_transforms,expected_sme,audit_fused_tasks,expected_fused,fused_ok?"PASS":"FAIL");
    if(!fused_ok)abort();
    long expected_pack=0;''')
(out/'trsm-fused-audit.c').write_text(a)
driver=(out/'independent-diag-extreme-duplex.c').read_text().replace('setenv("TRSM_DIAG_SME",reuse?"0":"1",1);', 'setenv("TRSM_FUSED_DIAG",reuse?"0":"2",1);')
(out/'independent-fused.c').write_text(driver)
gate=(out/'diag-gate.c').read_text().replace('TRSM_DIAG_SME','TRSM_FUSED_DIAG').replace('mode==1?"0":"1"','mode==1?"0":"2"').replace('DIAG_GATE','FUSED_GATE')
(out/'fused-gate.c').write_text(gate)
check=(out/'diag-final-check.c').read_text().replace('"trsm-diag-final.c"','"trsm-fused-final.c"').replace('DIAG_FINAL','FUSED_DIAG')
(out/'fused-check.c').write_text(check)
