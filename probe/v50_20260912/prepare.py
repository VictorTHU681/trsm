#!/usr/bin/env python3
"""Rebuild v48 follow-up experiments and the exact v49 validation sources."""
from pathlib import Path
import hashlib, subprocess, sys
here=Path(__file__).resolve().parent
repo=here.parents[1]
s=subprocess.run(['git','-C',str(repo),'show','2184f49:trsm.c'],check=True,capture_output=True).stdout
assert hashlib.sha256(s).hexdigest()=='04c6e356a287bc6afe460500ed177a7fd68ff847d217868481e4be77191a346f'
for name in ('trsm-v48-synced.c','trsm-v45-codex.c'):(here/name).write_bytes(s)
for script in ('make-hoist45.py','make-owner-hoist48.py','make-owner-cursors48.py','make-owner-diag48.py','make-owner-diagcompact48.py','make-owner-cursor-final48.py'):
 subprocess.run([sys.executable,str(here/script)],check=True)
subprocess.run([sys.executable,str(here/'make-solve-pmu.py'),'owner-cursor-final48'],check=True)
s=(here/'trsm-owner-cursor-final48.c').read_text()
s=s.replace('static int g_owner_cursor;', '// Select once per owner task; keep the original kernel for same-binary comparisons.\nstatic int g_owner_cursor;',1)
s=s.replace('    g_owner_cursor=env_int("TRSM_OWNER_CURSOR",1);','    g_owner_cursor=env_int("TRSM_OWNER_CURSOR",1); // Fixed KC64 prefetch offsets; 0 restores v48 owner kernels.',1)
s=s.replace('static inline __attribute__((always_inline)) void kloop_owner_cursor0(const double* a,','// KC64 owner kernels: prefetch bases remain fixed and PRFM uses immediate\n// offsets. A/B loads, FP64 FMOPA order and hosted C stores match the old kernels.\n// This removes cursor ADD instructions without changing the addresses prefetched.\nstatic inline __attribute__((always_inline)) void kloop_owner_cursor0(const double* a,',1)
s=s.replace('// 忙核探测（TRSM_CPROBE，默认开，','// 忙核探测（TRSM_CPROBE，默认关，',1)
(here/'trsm-v49-final.c').write_text(s)
(here/'v49-final-boundary.c').write_text((here/'owner-cursor-final48-boundary.c').read_text().replace('trsm-owner-cursor-final48.c','trsm-v49-final.c'))
(here/'bench_trsm.c').write_bytes((repo/'bench_trsm.c').read_bytes())
print('Generated pinned v48 experiments and v49 validation sources.')
