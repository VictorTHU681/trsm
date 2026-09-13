#!/usr/bin/env python3
from pathlib import Path
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=(out/'trsm-implicit-only.c').read_text()
s=s.replace('env_int("TRSM_IMPLICIT_OWNER",0) && c.special==5',
'''env_int("TRSM_IMPLICIT_OWNER",1) && c.special==5 && c.m<=1024''')
assert s.count('    if(env_int("TRSM_EXPECT_IMPLICIT",-1)>=0')==1
check=next(line for line in s.splitlines(keepends=True) if line.startswith('    if(env_int("TRSM_EXPECT_IMPLICIT",-1)>=0'))
s=s.replace(check,'')
(out/'trsm-implicit-final.c').write_text(s)
marker='    g_align_owner_buffers ='
assert s.count(marker)==1
(out/'trsm-implicit-final-audit.c').write_text(s.replace(marker,check+marker))
for name,original in [('gen-strict.c','probe/v50_20260912/gen-strict.c'),
                      ('independent-duplex.c','probe/v50_20260912/independent-duplex43.c'),
                      ('validate.c','probe/v50_20260912/validate.c')]:
    (out/name).write_bytes(subprocess.check_output(['git','show','3b24f38:'+original],cwd=root))
