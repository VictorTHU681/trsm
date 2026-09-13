#!/usr/bin/env python3
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'generated'
s=(out/'trsm-owner53.c').read_text()
def replace(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
replace('    int implicit_owner, stack_sched;\n','')
replace('// Owner tasks have a regular', 'static int g_implicit_owner;\n\n// Owner tasks have a regular')
s=s.replace('c->implicit_owner','g_implicit_owner').replace('c.implicit_owner','g_implicit_owner')
replace('    c.stack_sched = env_int("TRSM_STACK_SCHED",0) && c.owner;\n','')
replace('        long sched_stack[128];\n','')
replace('            sched = cnt > 0 ? (c.stack_sched && cnt<=128 ? sched_stack : malloc(sizeof(long)*cnt)) : NULL;',
        '            sched = cnt > 0 ? malloc(sizeof(long) * cnt) : NULL;')
replace('        if(sched!=sched_stack)free(sched);','        free(sched);')
(out/'trsm-implicit-only.c').write_text(s)
