#!/usr/bin/env python3
from pathlib import Path
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=(out/'trsm-diag-combo.c').read_text()
s=s.replace('g_diag_row==2?-accum[col]*inverse:-accum[col]/Lss[r][r]', '-accum[col]/Lss[r][r]')
s=s.replace('''    g_diag_sme=env_int("TRSM_DIAG_SME",0);
    g_diag_row=env_int("TRSM_DIAG_ROW",0);''', '''    // Large triangular matrices benefit from faster per-call diagonal packing.
    // 0 retains the original inverse/transform for same-binary comparisons.
    g_diag_sme=env_int("TRSM_DIAG_SME",m>=4096);
    g_diag_row=g_diag_sme!=0;''')
(out/'trsm-diag-final.c').write_text(s)
for name,original in [('gen-strict.c','probe/v50_20260912/gen-strict.c'),
                      ('independent-duplex.c','probe/v50_20260912/independent-duplex43.c'),
                      ('validate.c','probe/v50_20260912/validate.c'),('run.sh','run.sh')]:
    (out/name).write_bytes(subprocess.check_output(['git','show','36a092c:'+original],cwd=root))
test=(out/'diag-sme-check.c').read_text().replace('"trsm-diag-sme.c"','"trsm-diag-final.c"')
test=test.replace('g_diag_sme=mode','g_diag_sme=mode;g_diag_row=mode')
test=test.replace('DIAG_SME','DIAG_FINAL')
(out/'diag-final-check.c').write_text(test)
old=subprocess.check_output(['git','show','36a092c:probe/v50_20260912/make-close-audit50.py'],cwd=root).decode()
body=old[old.index("key='static void diag_prep_sub"):old.index("(p/'405-close-audit50.sh')")]
body=body.replace('trsm-close-audit50.c','trsm-diag-final-audit.c').replace('close-input-audit50.c','diag-final-input-audit.c')
exec(compile(body,'pinned-close-audit-generator','exec'),{'s':s,'p':out})
