#!/usr/bin/env python3
"""Validate closing logs and retain all final measurements."""
from pathlib import Path
import csv
import re

logs = Path(__file__).resolve().parents[2] / 'logs/noon_20260912'
rows = []
for number in (403, 404, 405):
    path, = logs.glob(f'{number}-*.log')
    text = path.read_text()
    assert not re.search(r'\bFAIL(?:ED)?\b', text), path
    marker = ''
    for line in text.splitlines():
        if line.startswith(('OWNER_DIAGDUP50_', 'CLOSE50_FORMAL')):
            marker = line
        if line.startswith('CBUF_PERF '):
            v = dict(re.findall(r'(\w+)=([\d.]+)', line))
            rows.append(dict(log=path.name, kind='component', configuration=line.split(' ns_per_k=')[0],
                             m=v['mb'], n=v['nc'], gflops=v['GFLOPS']))
        result = re.fullmatch(r'\s*(\d+) x\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+PASS', line)
        if result and ('SCORE' in marker or marker.startswith('CLOSE50_FORMAL')):
            rows.append(dict(log=path.name, kind='formal', configuration=marker,
                             m=result[1], n=result[2], gflops=result[4]))
    if number == 403:
        assert text.count('ODIAGDUP_CHECK') == 8
        assert 'comparisons=258048 error=0.000e+00 PASS' in text
    elif number == 404:
        checks = [l for l in text.splitlines() if l.startswith('CBUF_CHECK ')]
        assert len(checks) == 18 and all('quant_reference_error=0.000e+00' in l for l in checks)
    else:
        assert text.count('PREPROCESS_AUDIT') == 12 and text.count('INPUT_AUDIT') == 12
        assert 'CLOSE50_DONE' in text
        for line in text.splitlines():
            if line.startswith('PREPROCESS_AUDIT '):
                v = dict(re.findall(r'(\w+)=(\d+)', line))
                assert v['diag'] == v['expected_diag'] and v['pack'] == v['expected_pack']
                assert v['threads'] == '38' and 'cpus=38..75' in line
assert len(rows) == 119 and sum(r['kind'] == 'formal' for r in rows) == 11
with (logs / 'v50_closing_measurements.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
print('Closing validation PASS: 12 per-call preprocessing audits, 12 changed/identical-input checks, 11 formal and 108 component measurements.')
