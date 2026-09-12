#!/usr/bin/env python3
"""Add strict independent-input tests and phase/PMU builds to generated candidates."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'generated'
spec = importlib.util.spec_from_file_location('old_pmu', ROOT/'probe/v50_20260912/make-solve-pmu.py')
pmu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pmu)
test = (ROOT/'probe/v50_20260912/independent-cursor-switch.c').read_text()
(OUT/'independent.c').write_text(test)
for name in ('control', 'pfspread', 'spread', 'front32', 'late32'):
    source = (OUT/f'trsm-{name}.c').read_text()
    audit = '''    int expect_owner = env_int("TRSM_EXPECT_OWNER", -1);
    if (expect_owner >= 0 && (c.special == 5) != expect_owner) abort();
    int expect_cursor = env_int("TRSM_EXPECT_CURSOR", -1);
    if (expect_cursor >= 0 && (g_owner_cursor && c.special == 5) != expect_cursor) abort();
'''
    assert source.count('    g_align_owner_buffers =') == 1
    (OUT/f'trsm-{name}-audit.c').write_text(source.replace('    g_align_owner_buffers =', audit+'    g_align_owner_buffers =', 1))
    (OUT/f'trsm-{name}-pmu.c').write_text(pmu.patch(source))
