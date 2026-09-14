#!/usr/bin/env python3
from pathlib import Path
import subprocess,hashlib,re
here=Path(__file__).resolve().parent;root=here.parents[1];out=here/'generated';out.mkdir(exist_ok=True)
def pinned(path):return subprocess.check_output(['git','show','0624902:'+path],cwd=root)
base=pinned('trsm.c').decode()
assert hashlib.sha256(base.encode()).hexdigest()=='951a1594b22b5f424f47fc5b0a7b7ce2c56cbb38601267ef45efbcdf2bd9c6f6'
(out/'trsm-v56.c').write_text(base)
for name in ('bench_trsm.c','run.sh'):(out/name).write_bytes(pinned(name))
old='g_diag_compact=env_int("TRSM_DIAG_COMPACT",1) && c.special==2 && !c.swhost;'
assert base.count(old)==1
s=base.replace(old,'{ int mode=env_int("TRSM_DIAG_COMPACT",1);\n      g_diag_compact=mode && !c.swhost && (c.special==2 || (mode>=2 && c.special==4)); }')
(out/'trsm-tc3compact.c').write_text(s)
final=base.replace(old,'g_diag_compact=env_int("TRSM_DIAG_COMPACT",1) && (c.special==2 || c.special==4) && !c.swhost;')
final=final.replace('// Full NC128, MB256 specialization already requires swhost=0.','// Full NC128 specializations use fixed strides and no hosted X copy.')
(out/'trsm-v57.c').write_text(final)
driver=pinned('probe/v50_20260912/independent-duplex43.c').decode()
driver='#include "trsm-tc3compact.c"\n'+driver
driver=driver.replace('            l_trsm(m,n,L,lda,B,ldb);','            l_trsm(m,n,L,lda,B,ldb);\n            if(g_diag_compact!=(atoi(getenv("TRSM_DIAG_COMPACT"))>=2)){puts("TC3_COMPACT_GATE FAIL");return 1;}')
(out/'independent-tc3compact.c').write_text(driver)
driver=driver.replace('trsm-tc3compact.c','trsm-v57.c').replace('(atoi(getenv("TRSM_DIAG_COMPACT"))>=2)','(atoi(getenv("TRSM_DIAG_COMPACT"))!=0)')
(out/'independent-v57.c').write_text(driver)
for target,source in [('gen-strict.c','probe/gen_test.c'),('validate.c','probe/v50_20260912/validate.c'),('compact-guard.c','probe/alternative57_20260914/compact-guard.c')]:
    data=pinned(source).decode().replace('e > 1e-10','e > 1e-12').replace('trsm-diagcompact.c','trsm-v57.c')
    (out/target).write_text(data)

# The previous inline owner experiment increased the main hot function size.
# Keep this version out of line and remove the runtime selector at the call site.
s=base
begin=s.index('#define DEF_KLOOP_PLAIN(');end=s.index('DEF_KLOOP_PLAIN(kloop_plain)',begin)
definition=s[begin:end].replace('DEF_KLOOP_PLAIN','DEF_OWNERDIAG58')
definition=definition.replace('static inline void NAME','static __attribute__((noinline)) void NAME')
for name in ('ase','aso','bs'):definition=definition.replace(f'[{name}] "r"({name}), ','')
definition=re.sub(r'^\s*sw_t\* sw =.*\\\n','',definition,flags=re.M)
definition=definition.replace('SW_LOAD','').replace('SW_SAVE','').replace('SWCLOB,','').replace(', [sw] "r"(sw)','')
begin=s.index('#undef KS\n#define KS(',s.index('#define DEF_FIXED_PLAIN'));end=s.index('#undef KS_EVEN',begin)
fixed=s[begin:end].replace('#1024','#768')
helpers='#ifndef TRSM_OWNER_DIAG58\n#define TRSM_OWNER_DIAG58 1\n#endif\n'+definition+'\n'
for name in ('KS','KLOOP_INIT','SWS','SWS_GATE_K','SWS_GATE_R','PFC'):helpers+=f'#pragma push_macro("{name}")\n'
helpers+=fixed+'\n'
for name in ('SWS','SWS_GATE_K','SWS_GATE_R'):helpers+=f'#undef {name}\n#define {name}\n'
helpers+='#if TRSM_OWNER_DIAG58==2\n#undef PFC\n#define PFC\n#endif\n'
helpers+='DEF_OWNERDIAG58(ownerdiag58_plain)\n'
for name in ('PFC','SWS_GATE_R','SWS_GATE_K','SWS','KLOOP_INIT','KS'):helpers+=f'#pragma pop_macro("{name}")\n'
marker='#undef K_MOP\n#define K_MOP "fmops "\nDEF_FIXED_PLAIN(kloop_fixed_direct)'
assert s.count(marker)==1;s=s.replace(marker,helpers+'\n'+marker)
needle='''                if (fast_owner && c->ddirect) {
                    PROF_T(c, kloop_plain_nt(A, 128, 128, cb + 32 * t, cs, K, cols, &pf), t_dkloop);'''
assert s.count(needle)==1
s=s.replace(needle,'''                if (fast_owner && c->ddirect) {
#if TRSM_OWNER_DIAG58
                    PROF_T(c, ownerdiag58_plain(A,128,128,cb+32*t,cs,K,cols,&pf),t_dkloop);
#else
                    PROF_T(c, kloop_plain_nt(A,128,128,cb+32*t,cs,K,cols,&pf),t_dkloop);
#endif''')
(out/'trsm-ownerdiag58.c').write_text(s)
driver=pinned('probe/v50_20260912/independent-owner-switch.c').decode()
driver='#include "trsm-ownerdiag58.c"\n'+driver
driver=driver.replace('            setenv("TRSM_OWNER",reuse?"0":"1",1);','            setenv("TRSM_OWNER","1",1);')
driver=driver.replace('            setenv("TRSM_EXPECT_OWNER",reuse?"0":"1",1);','')
driver=driver.replace('            l_trsm(m,n,L,lda,B,ldb);','            l_trsm(m,n,L,lda,B,ldb);\n            if(!g_implicit_owner){puts("OWNER58_GATE FAIL");return 1;}')
(out/'independent-owner58.c').write_text(driver)
subprocess.run(['python3',str(here/'make_preload.py')],check=True)
