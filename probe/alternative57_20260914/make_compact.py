#!/usr/bin/env python3
"""Clean TC2 candidate: keep FP64 order; compact fixed diagonal/no idle copier."""
from pathlib import Path
import re,subprocess
here=Path(__file__).resolve().parent;root=here.parents[1];out=here/'generated'
s=(out/'trsm-v55.c').read_text()
helpers='// NC128 diagonal kernels: no hosted X copy; prefetch C only during the first 64 hosted steps.\n'
names={'PLAIN':'kloop_diag_plain_compact','EPI':'kloop_diag_set_compact','EPI64':'kloop_diag64_set_compact'}
for kind,end in [('PLAIN','DEF_KLOOP_PLAIN(kloop_plain)'),('EPI','DEF_KLOOP_EPI(kloop_epi_set, EV_SET)'),('EPI64','DEF_KLOOP_EPI64(kloop_epi64_set, EV_SET)')]:
    begin=s.index('#define DEF_KLOOP_'+kind+'(')
    text=s[begin:s.index(end,begin)].replace('DEF_KLOOP_'+kind,'DEF_COMPACT_DIAG_'+kind)
    for name in ('ase','aso','bs','cs','ecols'):
        text=text.replace(f'[{name}] "r"({name}), ','')
    text=re.sub(r'^\s*sw_t\* sw =.*\\\n','',text,flags=re.M)
    text=text.replace('SW_LOAD','').replace('SW_SAVE','').replace('SWCLOB,','').replace(', [sw] "r"(sw)','')
    helpers+=text+'\n'
for name in ('SWS','SWS_GATE_K','SWS_GATE_R','KLOOP_REST'):
    helpers+=f'#pragma push_macro("{name}")\n#undef {name}\n'
for name in ('SWS','SWS_GATE_K','SWS_GATE_R'):helpers+=f'#define {name}\n'
begin=s.index('#define KLOOP_REST');end=s.index('// 托管回写走线器',begin)
helpers+=s[begin:end].replace('KS_ODD PFC','KS_ODD')
helpers+='\n#undef LDA_CUR\n#define LDA_CUR "ld1d "\n'
for kind,name in names.items():
    helpers+=f'DEF_COMPACT_DIAG_{kind}({name}'+(', EV_SET' if kind!='PLAIN' else '')+')\n'
    helpers+=f'#undef DEF_COMPACT_DIAG_{kind}\n'
for name in ('KLOOP_REST','SWS_GATE_R','SWS_GATE_K','SWS'):
    helpers+=f'#pragma pop_macro("{name}")\n'
helpers+='\n#undef LDA_CUR\n#define LDA_CUR LDA_NT_INSN\n'
marker='#undef K_MOP\n#define K_MOP "fmops "\nDEF_FIXED_PLAIN(kloop_fixed_direct)'
assert s.count(marker)==1;s=s.replace(marker,helpers+'\n'+marker)
marker='static inline __attribute__((always_inline)) long run_task_special_impl('
s=s.replace(marker,'static int g_diag_compact;\n'+marker)
begin=s.index(marker);end=s.index('static long run_task_special1(',begin);part=s[begin:end]
for old,new,args in [
    ('kloop_epi_set',names['EPI'],'A, 128, 128, cb + 32 * t, cs, K, cols, &pf, scratch, pend_c, cs, pend_cols'),
    ('kloop_epi64_set',names['EPI64'],'A, 128, 128, cb + 32 * t, cs, cols, &pf, scratch, pend_c, cs, pend_cols'),
    ('kloop_plain',names['PLAIN'],'A, 128, 128, cb + 32 * t, cs, K, cols, &pf')]:
    call=old+'('+args+')';assert part.count(call)==1
    part=part.replace(call,f'(g_diag_compact ? {new}({args}) : {call})')
s=s[:begin]+part+s[end:]
marker='    g_implicit_owner = env_int('
s=s.replace(marker,'    // Full NC128, MB256 specialization already requires swhost=0.\n    g_diag_compact=env_int("TRSM_DIAG_COMPACT",1) && c.special==2 && !c.swhost;\n'+marker)
(out/'trsm-diagcompact.c').write_text(s)
driver=(out/'independent-diagfixed.c').read_text().replace('trsm-diagfixed.c','trsm-diagcompact.c')
driver=driver.replace('g_diag_fixed!=atoi(getenv("TRSM_DIAG_FIXED"))','g_diag_compact!=atoi(getenv("TRSM_DIAG_COMPACT"))')
(out/'independent-diagcompact.c').write_text(driver)
gen=subprocess.check_output(['git','show','a989f3a:probe/gen_test.c'],cwd=root).decode().replace('e > 1e-10','e > 1e-12')
(out/'gen-strict.c').write_text(gen)
# Pinned changed-input driver was archived with the v50 experiments.
(out/'validate.c').write_bytes(subprocess.check_output(['git','show','a989f3a:probe/v50_20260912/validate.c'],cwd=root))
(out/'compact-guard.c').write_bytes((here/'compact-guard.c').read_bytes())
