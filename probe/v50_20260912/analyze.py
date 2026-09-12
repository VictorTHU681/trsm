#!/usr/bin/env python3
"""Extract every formal process and PMU sample from requests 395–402."""
from pathlib import Path
import csv
import math
import re
import statistics as st

root = Path(__file__).resolve().parents[2]
logs = root / 'logs/noon_20260912'
formal, pmu = [], []
for number in range(395, 403):
    files = list(logs.glob(f'{number}-*.log'))
    assert len(files) == 1, (number, files)
    path = files[0]
    text = path.read_text()
    assert not re.search(r'\bFAIL(?:ED)?\b', text), path
    marker, fields, sample = '', {}, 0
    for line in text.splitlines():
        if re.match(r'\w+ (?:round|flag)=', line):
            marker = line.split()[0]
            fields = dict(re.findall(r'(\w+)=(\S+)', line))
            sample = 0
        result = re.fullmatch(
            r'\s*(\d+) x\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(PASS|FAIL)', line)
        if result and ('SCORE' in marker or marker.endswith(('_SAME', '_TC3'))):
            m, n, ms, gflops, error, check = result.groups()
            formal.append(dict(log=path.name, experiment=marker,
                round=fields['round'], variant=fields.get('tag', fields.get('flag')),
                m=m, n=n, ms=ms, gflops=gflops, max_error=error, check=check))
        if line.startswith('PMUSOLVE '):
            row = dict(re.findall(r'(\w+)=(\d+)', line))
            assert row['enabled_ns'] == row['running_ns'], row
            pmu.append(dict(log=path.name, flag=fields['flag'], sample=sample, **row))
            sample += 1
assert len(formal) == 110 and len(pmu) == 36, (len(formal), len(pmu))
for name, rows in [('v48_followup_formal.csv', formal), ('v50_pmu.csv', pmu)]:
    with (logs / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

for experiment, count, a, b in [
    ('OWNER_CURSOR48_SCORE', 6, 'v48-synced', 'owner-cursor-final48'),
    ('OWNER_CURSOR48_SAME', 6, '0', '1'),
    ('OWNER_CURSOR48_TC3', 4, 'v48-synced', 'owner-cursor-final48'),
]:
    subset = [r for r in formal if r['experiment'] == experiment]
    for shape in sorted({(r['m'], r['n']) for r in subset}):
        pairs = {(int(r['round']), r['variant']): float(r['gflops']) for r in subset
                 if (r['m'], r['n']) == shape}
        values = [[pairs[r, variant] for r in range(1, count + 1)] for variant in (a, b)]
        delta = 100 * math.expm1(st.mean(math.log(y / x) for x, y in zip(*values)))
        print(experiment, shape, 'medians', list(map(st.median, values)), 'paired %', delta)
