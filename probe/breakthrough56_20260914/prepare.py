#!/usr/bin/env python3
from pathlib import Path
import hashlib,subprocess,runpy
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated';out.mkdir(exist_ok=True)
def pinned(path):return subprocess.check_output(['git','show','658ac18:'+path],cwd=root)
base=pinned('trsm.c')
assert hashlib.sha256(base).hexdigest()=='1f6a3dc4712bbb04c4a7f8cfbc042ba8c306681bcfd253a51242a47b4c518350'
(out/'trsm-v55.c').write_bytes(base)
for name in ('bench_trsm.c','run.sh'):(out/name).write_bytes(pinned(name))
# Reuse the previous checked inverse prototype without changing its old archive.
(here/'fullinv.c').write_bytes(pinned('probe/resume_20260913/fullinv.c'))
(here/'fullinv-boundary.c').write_bytes(pinned('probe/resume_20260913/fullinv-boundary.c'))
script=pinned('probe/resume_20260913/prepare_fullinv.py').decode()
assert hashlib.sha256((here/'independent-base.c').read_bytes()).hexdigest()=='ed843267013b69694949b4f3a037aee4468ff1f846fe1eaff899c85b1eef7062'
script=script.replace("HERE.parents[1]/'probe/v50_20260912/independent-cursor-switch.c'", "HERE/'independent-base.c'")
exec(compile(script,'pinned-inverse-generator','exec'),{'__file__':str(here/'prepare-inverse.py')})
for p in out.glob('fullinv*.c'):
    p.write_text(p.read_text().replace('"trsm-control.c"','"trsm-v55.c"'))
worker=pinned('probe/fold54_20260913/worker.sh')
(here/'worker.sh').write_bytes(worker)
for generator in ('make_digits.py','make_blocked.py','make_right.py','make_tiled.py','make_recursive.py'):
    runpy.run_path(str(here/generator),run_name='__main__')
for name in ('digits.c','fp64_product.c','vector_limits.c'):
    (out/name).write_bytes((here/name).read_bytes())
