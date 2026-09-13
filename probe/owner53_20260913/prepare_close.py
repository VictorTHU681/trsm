#!/usr/bin/env python3
"""Generate final-source audit copies; never alter the production candidate."""
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
root = here.parents[1]
out = here / 'generated'
source = (out / 'trsm-implicit-final.c').read_text()
old = subprocess.check_output(['git', 'show', '3b24f38:probe/v50_20260912/make-close-audit50.py'], cwd=root).decode()
# Reuse the established current-input mutation and preprocessing-count audit.
body = old[old.index("key='static void diag_prep_sub"):old.index("(p/'405-close-audit50.sh')")]
body = body.replace("trsm-close-audit50.c", "trsm-implicit-close.c").replace("close-input-audit50.c", "implicit-input-audit.c")
exec(compile(body, 'pinned-close-audit-generator', 'exec'), {'s': source, 'p': out})
s = (out / 'trsm-implicit-close.c').read_text()
s = s.replace('static int g_implicit_owner;', 'static int g_implicit_owner, audit_build_count, audit_bounds_count;')
s = s.replace('static void build_tasks(ctx_t* c, int nth)\n{', 'static void build_tasks(ctx_t* c, int nth)\n{\n    audit_build_count++;')
s = s.replace('static void build_owner_bounds(ctx_t* c, int nth)\n{', 'static void build_owner_bounds(ctx_t* c, int nth)\n{\n    audit_bounds_count++;')
s = s.replace('    audit_diag_count=audit_pack_count=0;', '    audit_diag_count=audit_pack_count=0;\n    audit_build_count=audit_bounds_count=0;')
s = s.replace('    long expected_pack=0;', '''    int task_ok=audit_build_count==1 && audit_bounds_count==g_implicit_owner;
    printf("TASK_BUILD_AUDIT m=%ld n=%ld builds=%d bounds=%d implicit=%d %s\\n",
        m,n,audit_build_count,audit_bounds_count,g_implicit_owner,task_ok?"PASS":"FAIL");
    if(!task_ok)abort();
    long expected_pack=0;''')
(out / 'trsm-implicit-close.c').write_text(s)

s = (here / 'bounds_check.c').read_text().replace('trsm-owner53.c', 'trsm-implicit-final.c')
s = s.replace('        build_tasks(&c,threads[th]);', '        g_implicit_owner=0;build_tasks(&c,threads[th]);')
s = s.replace('c.implicit_owner=1;', 'g_implicit_owner=1;')
(out / 'final-bounds.c').write_text(s)

s = (out / 'gen-strict.c').read_text()
s = s[:s.index('int main(int argc, char** argv)')] + r'''
int main(void)
{
    const int cases[][3]={{64,96,1},{1024,96,1},{1024,192,1},
        {1024,19968,1},{1088,96,0},{1023,96,0},{1024,97,0},{512,19967,0}};
    int fail=0;
    setenv("TRSM_OWNER","1",1);
    for(int build=0;build<2;build++){
        setenv("TRSM_TBUILD",build?"1":"0",1);
        for(unsigned c=0;c<sizeof(cases)/sizeof(cases[0]);c++){
            int m=cases[c][0],n=cases[c][1];
            setenv("TRSM_EXPECT_IMPLICIT",cases[c][2]?"1":"0",1);
            fail+=run_case(m,n,m+7,n+11,49103+c);
        }
    }
    printf("IMPLICIT_GATE %s: %d failures\n",fail?"FAIL":"PASS",fail);
    return !!fail;
}
'''
(out / 'implicit-gate.c').write_text(s)
(out / 'run.sh').write_bytes(subprocess.check_output(['git', 'show', '3b24f38:run.sh'], cwd=root))
