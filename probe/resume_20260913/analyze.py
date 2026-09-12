#!/usr/bin/env python3
"""Parse every recorded sample; warmup exclusion only affects grouped summaries.

Micro cycles are timer-derived at 1.55 GHz, not PMU core cycles. Full-inverse
and dense timings include instrumentation; only 407 is a formal paired screen.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOG = ROOT / 'logs/resume_20260913'
KV = re.compile(r'(\w+)=([^\s]+)')
SCORE = re.compile(r'^\s*(\d+)\s+x\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\deE+.-]+)\s+(PASS|FAIL)$')
MICRO = re.compile(r'^([ov]\S+)\s+.+[:：]\s*([\d.]+)\s+效率\s+([\d.]+)%')
HEADERS = ('FORMAL ', 'PROFILE ', 'MICRO ', 'INVERSE_SCREEN ', 'INVERSE_PRIVATE ',
           'INVERSE_COMPACT ', 'PRIVATE_SW_ONLY ', 'INVERSE_SHAPE ',
           'DENSE_PLACEMENT ', 'DENSE_CONTEXT ', 'DATA_MICRO ', 'DENSE_TASKS ',
           'DENSE_COLOR ', 'DENSE_COLOR_CHECK ', 'PRODUCTION_MB ', 'PRODUCTION_MB_PROFILE ',
           'SPECIAL_MB ', 'SPECIAL_MB_PROFILE ', 'STRASSEN_REBLOCKED ',
           'DENSE_CLEAR ', 'DENSE_CLEAR_CHECK ')
samples = defaultdict(list)


def write_csv(name, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (LOG / (name + '.csv')).open('w', newline='') as f:
        w = csv.DictWriter(f, columns, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


for path in sorted(LOG.glob('*.log')):
    context = ''
    counters = Counter()
    phase = {}
    solve = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith(HEADERS):
            context = line
            counters.clear()
        meta = dict(file=path.name, line=lineno, context=context)
        match = SCORE.match(line)
        if match:
            m, n, ms, gf, error, status = match.groups()
            samples['formal'].append({**meta, **dict(KV.findall(context)), 'm': m, 'n': n,
                                      'time_ms': ms, 'gflops': gf, 'max_error': error, 'status': status})
        match = MICRO.match(line)
        if match:
            name, cycles, eff = match.groups()
            samples['micro'].append(dict(meta, **dict(KV.findall(context)), scenario=name,
                                         timer_cycles=cycles, efficiency_pct=eff))
        if line.startswith('[trsm prof] m='):
            solve = dict(KV.findall(line))
            match = re.search(r'setup ([\d.]+) ms, parallel ([\d.]+) ms', line)
            if match:
                solve.update(setup_ms=match[1], parallel_ms=match[2])
        if '每线程平均 ms：' in line:
            phase = {key + '_ms': value for key, value in re.findall(
                r'\b(prep|load|pack|xcopy|wait|kloop|dump|epi|dkloop|ddump|depi|store) ([\d.]+)', line)}
            index = counters['trsm_phase']
            counters['trsm_phase'] += 1
            samples['trsm_phase'].append(dict(meta, call_index=index, warmup=int(index == 0), **solve, **phase))
        if line.startswith(('FULLINV ', 'DENSE_PROFILE ', 'DENSE_PMU ', 'PMUSOLVE ', 'STRASSEN ')):
            kind = line.split()[0].lower()
            values = dict(KV.findall(line))
            key = (kind, values.get('m'), values.get('n'), values.get('k'))
            index = counters[key]
            counters[key] += 1
            warm = int(values.get('rep', '0')) < 0 if kind == 'strassen' else index == 0
            # Recursive leaf records cannot be assigned to warmup by shape alone.
            if kind == 'dense_profile' and path.name.startswith('425-'):
                warm = -1
            row = dict(meta, call_index=index, warmup=int(warm), **values)
            if kind == 'pmusolve':
                row.update(phase)
            samples[kind].append(row)

for name, rows in samples.items():
    write_csv(name, rows)

formal = [r for r in samples['formal'] if r['file'].startswith('407-')]
assert len(formal) == 20 and all(r['status'] == 'PASS' for r in formal)
controls = {r['round']: float(r['gflops']) for r in formal if r['mode'] == 'control'}
schedule = []
for mode in ('control', 'pfspread', 'spread', 'front32', 'late32'):
    rows = [r for r in formal if r['mode'] == mode]
    values = [float(r['gflops']) for r in rows]
    ratio = math.exp(st.mean(math.log(float(r['gflops']) / controls[r['round']]) for r in rows))
    schedule.append(dict(mode=mode, rounds=len(rows), median_gflops=st.median(values),
                         min_gflops=min(values), max_gflops=max(values), paired_change_pct=100*(ratio-1)))
write_csv('schedule_summary', schedule)

grouped = []
for kind in ('pmusolve', 'trsm_phase', 'fullinv', 'dense_profile', 'dense_pmu', 'strassen'):
    groups = defaultdict(list)
    for r in samples[kind]:
        if kind == 'dense_profile' and r['file'].startswith('425-'):
            continue
        if not r['warmup']:
            groups[(r['file'], r['context'], r.get('m'), r.get('n'), r.get('k'), r.get('depth'))].append(r)
    for key, rows in groups.items():
        file, context, m, n, k, depth = key
        result = dict(kind=kind, file=file, context=context, m=m, n=n, k=k, depth=depth, samples=len(rows))
        for field in rows[0]:
            if field.endswith('_ms') or field in ('cpu_cycles', 'inst_retired', 'l2d_cache_refill',
                                                'l1d_cache_refill', 'stall_backend', 'stall_frontend',
                                                'nominal_GFLOPS'):
                result['mean_' + field] = st.mean(float(r[field]) for r in rows)
        grouped.append(result)
write_csv('phase_summary', grouped)

for r in samples['pmusolve'] + samples['dense_pmu']:
    assert r['enabled_ns'] == r['running_ns'], (r['file'], r['line'], 'PMU multiplexed')
for p in LOG.glob('*.status'):
    # 422 compilation ran before the failed source transfer was retried.
    # The complete failure is preserved, and 422b reruns the identical script.
    expected = '1' if p.name == '422-production-mb.sh.status' else '0'
    assert p.read_text().strip() == expected, (p, p.read_text())

manifest = {}
for folder, extensions in ((HERE, {'.c', '.py', '.sh', '.md'}),
                           (HERE/'generated', {'.c', '.h'}), (LOG, {'.log', '.status', '.out', '.err'})):
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix in extensions:
            manifest[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
for filename in ('trsm.c', 'bench_trsm.c', 'run.sh'):
    manifest[filename] = hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
(LOG/'sha256.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
print(json.dumps({'sample_counts': {k: len(v) for k, v in samples.items()},
                  'schedule': schedule}, indent=2))
