#!/usr/bin/env python3
"""Specialize only the existing full NC128 diagonal kernels; preserve FP64 order."""
from pathlib import Path
import re, subprocess
here=Path(__file__).resolve().parent;root=here.parents[1];out=here/'generated'
s=(out/'trsm-v55.c').read_text()
definitions=[]
for kind,end in [('PLAIN','DEF_KLOOP_PLAIN(kloop_plain)'),('EPI','DEF_KLOOP_EPI(kloop_epi_set, EV_SET)'),('EPI64','DEF_KLOOP_EPI64(kloop_epi64_set, EV_SET)')]:
    begin=s.index('#define DEF_KLOOP_'+kind+'(')
    text=s[begin:s.index(end,begin)]
    # The signature remains unchanged; the assembler no longer needs these GPRs.
    for name in ('ase','aso','bs','cs','ecols'):
        text=text.replace(f'[{name}] "r"({name}), ','')
    for version in (1,2):
        part=text.replace('DEF_KLOOP_'+kind,'DEF_DIAG57_'+kind+'_'+str(version))
        if version==2:
            part=re.sub(r'^\s*sw_t\* sw =.*\\\n','',part,flags=re.M)
            part=part.replace('SW_LOAD','').replace('SW_SAVE','').replace('SWCLOB,','')
            part=part.replace(', [sw] "r"(sw)','')
        definitions.append(part)
helpers='\n'.join(definitions)+'\n'
# Insert while KS/EV_ROW_ADV/SET_EPI_PREDS have the existing NC128 definitions.
marker='#undef K_MOP\n#define K_MOP "fmops "\nDEF_FIXED_PLAIN(kloop_fixed_direct)'
assert s.count(marker)==1
helpers+='\n#undef LDA_CUR\n#define LDA_CUR "ld1d "\n'
for version in (1,2):
    if version==2:
        helpers+='\n#pragma push_macro("SWS")\n#undef SWS\n#define SWS\n'
    for kind in ('PLAIN','EPI','EPI64'):
        helpers+=f'DEF_DIAG57_{kind}_{version}(diag57_{kind.lower()}_{version}'+(', EV_SET' if kind!='PLAIN' else '')+')\n'
    if version==2:helpers+='#pragma pop_macro("SWS")\n'
helpers+='\n#undef LDA_CUR\n#define LDA_CUR LDA_NT_INSN\n'
s=s.replace(marker,helpers+'\n'+marker)
marker='static inline __attribute__((always_inline)) long run_task_special_impl('
s=s.replace(marker,'static int g_diag_fixed;\n'+marker)
begin=s.index(marker);end=s.index('static long run_task_special1(',begin)
part=s[begin:end]
for old,new,args in [
    ('kloop_epi_set','epi','A, 128, 128, cb + 32 * t, cs, K, cols, &pf, scratch, pend_c, cs, pend_cols'),
    ('kloop_epi64_set','epi64','A, 128, 128, cb + 32 * t, cs, cols, &pf, scratch, pend_c, cs, pend_cols'),
    ('kloop_plain','plain','A, 128, 128, cb + 32 * t, cs, K, cols, &pf')]:
    call=old+'('+args+')';assert part.count(call)==1
    part=part.replace(call,f'(g_diag_fixed==2 ? diag57_{new}_2({args}) : g_diag_fixed==1 ? diag57_{new}_1({args}) : {call})')
s=s[:begin]+part+s[end:]
marker='    g_implicit_owner = env_int('
s=s.replace(marker,'    g_diag_fixed=(c.special==1 || c.special==2 || c.special==4) && !c.swhost ? env_int("TRSM_DIAG_FIXED",0) : 0;\n'+marker)
(out/'trsm-diagfixed.c').write_text(s)
driver=subprocess.check_output(['git','show','a989f3a:probe/v50_20260912/independent-duplex43.c'],cwd=root).decode()
# Compile solver and driver as one TU so the driver can check the selected mode.
driver='#include "trsm-diagfixed.c"\n'+driver
driver=driver.replace('            l_trsm(m,n,L,lda,B,ldb);','            l_trsm(m,n,L,lda,B,ldb);\n            if(g_diag_fixed!=atoi(getenv("TRSM_DIAG_FIXED"))) {puts("DIAG_GATE FAIL");return 1;}')
(out/'independent-diagfixed.c').write_text(driver)
