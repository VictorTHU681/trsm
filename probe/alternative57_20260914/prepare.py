#!/usr/bin/env python3
from pathlib import Path
import subprocess,hashlib
here=Path(__file__).resolve().parent;root=here.parents[1];out=here/'generated';out.mkdir(exist_ok=True)
def pinned(path):return subprocess.check_output(['git','show','a989f3a:'+path],cwd=root)
base=pinned('trsm.c');assert hashlib.sha256(base).hexdigest()=='1f6a3dc4712bbb04c4a7f8cfbc042ba8c306681bcfd253a51242a47b4c518350'
(out/'trsm-v55.c').write_bytes(base)
for name in ('bench_trsm.c','run.sh'):(out/name).write_bytes(pinned(name))
(out/'rns13-fixed-constants.h').write_bytes(pinned('probe/noon_20260912/rns13-fixed-constants.h'))
script=pinned('probe/breakthrough56_20260914/make_digits.py').decode()
exec(compile(script,'pinned-digit-kernels','exec'),{'__file__':str(here/'make_digits.py')})
for name in ('residue_fused.c',):
    if (here/name).exists():(out/name).write_bytes((here/name).read_bytes())
for name in ('make_fastmod.py','make_diagfixed.py','make_ownerdiag.py','make_diagprefetch.py','make_compact.py'):
    subprocess.run(['python3',str(here/name)],check=True)
