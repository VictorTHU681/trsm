#!/usr/bin/env python3
"""NC96 owner diagonal: same FMOPA/prefetch schedule, fixed strides, no idle SWS."""
from pathlib import Path
import re,subprocess
here=Path(__file__).resolve().parent;root=here.parents[1];out=here/'generated'
s=(out/'trsm-v55.c').read_text()
begin=s.index('#define DEF_KLOOP_PLAIN(');end=s.index('DEF_KLOOP_PLAIN(kloop_plain)',begin)
definition=s[begin:end].replace('DEF_KLOOP_PLAIN','DEF_OWNERDIAG57')
for name in ('ase','aso','bs'):
    definition=definition.replace(f'[{name}] "r"({name}), ','')
definition=re.sub(r'^\s*sw_t\* sw =.*\\\n','',definition,flags=re.M)
definition=definition.replace('SW_LOAD','').replace('SW_SAVE','').replace('SWCLOB,','').replace(', [sw] "r"(sw)','')
# The existing fixed macros here already have NC128 and A stride 128.
begin=s.index('#undef KS\n#define KS(',s.index('#define DEF_FIXED_PLAIN'))
end=s.index('#undef KS_EVEN',begin)
fixed=s[begin:end].replace('#1024','#768')
helpers=definition+'\n'
for name in ('KS','KLOOP_INIT','SWS'):
    helpers+=f'#pragma push_macro("{name}")\n'
helpers+=fixed+'\n#undef SWS\n#define SWS\nDEF_OWNERDIAG57(ownerdiag57_plain)\n'
for name in ('SWS','KLOOP_INIT','KS'):
    helpers+=f'#pragma pop_macro("{name}")\n'
marker='#undef K_MOP\n#define K_MOP "fmops "\nDEF_FIXED_PLAIN(kloop_fixed_direct)'
assert s.count(marker)==1;s=s.replace(marker,helpers+'\n'+marker)
marker='static inline __attribute__((always_inline)) long run_task_impl('
s=s.replace(marker,'static int g_owner_diag_simple;\n'+marker)
begin=s.index(marker);end=s.index('static inline size_t special_l_offset',begin)
part=s[begin:end]
needle='''                if (fast_owner && c->ddirect) {
                    PROF_T(c, kloop_plain_nt(A, 128, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);'''
assert part.count(needle)==1
part=part.replace(needle,'''                if (fast_owner && c->ddirect) {
                    PROF_T(c, (g_owner_diag_simple ? ownerdiag57_plain(A,128,128,cb+32*t,cs,K,cols,&pf) : kloop_plain_nt(A,128,128,cb+32*t,cs,K,cols,&pf)), t_dkloop);''')
s=s[:begin]+part+s[end:]
marker='    g_implicit_owner = env_int('
s=s.replace(marker,'    g_owner_diag_simple=c.special==5 && c.ddirect && !c.swhost && env_int("TRSM_OWNER_DIAG_SIMPLE",0);\n'+marker)
(out/'trsm-ownerdiag.c').write_text(s)
driver=subprocess.check_output(['git','show','a989f3a:probe/v50_20260912/independent-owner-switch.c'],cwd=root).decode()
driver='#include "trsm-ownerdiag.c"\n'+driver
driver=driver.replace('            setenv("TRSM_OWNER",reuse?"0":"1",1);','            setenv("TRSM_OWNER","1",1);\n            setenv("TRSM_OWNER_DIAG_SIMPLE",reuse?"0":"1",1);')
driver=driver.replace('            setenv("TRSM_EXPECT_OWNER",reuse?"0":"1",1);','')
driver=driver.replace('            l_trsm(m,n,L,lda,B,ldb);','            l_trsm(m,n,L,lda,B,ldb);\n            if(g_owner_diag_simple!=!reuse){puts("OWNER_DIAG_GATE FAIL");return 1;}')
(out/'independent-ownerdiag.c').write_text(driver)
