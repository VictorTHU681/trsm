#!/usr/bin/env python3
"""Rebuild isolated research sources against the exact v47 baseline."""
from pathlib import Path
import hashlib
import subprocess
import sys

here = Path(__file__).resolve().parent
repo = here.parents[1]
source = subprocess.run(
    ['git', '-C', str(repo), 'show', '245bb36:trsm.c'],
    check=True, capture_output=True,
).stdout
assert hashlib.sha256(source).hexdigest() == 'adc925c72074056b4379cfd68f0347673f720852967ff05f73923aefaa5c6568'
(here / 'trsm-v47-codex.c').write_bytes(source)
for script in (
    'make-hybrid-peak.py', 'make-hybrid24.py', 'make-hybrid24-replicas.py',
    'make-hybrid24-memory.py', 'make-hybrid24-duplex.py',
    'make-hybrid24-duplex-acc2.py', 'make-hybrid24-ant.py', 'make-rns-shortk.py',
    'make-shortk-pmu.py', 'make-owner-import47.py', 'make-sve-load-peak.py',
):
    subprocess.run([sys.executable, str(here / script)], check=True)
print('Generated isolated research sources from v47 / 245bb36.')
