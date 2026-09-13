#!/usr/bin/env python3
"""Pool all final-code confirmation rounds, retaining every slow observation."""
from pathlib import Path
import csv, math, random
from collections import defaultdict
from statistics import mean, median
logs=Path(__file__).resolve().parents[2]/'logs/fold54_20260913'
nodes={'479-align-crossnode.sh.log':'cn20487',
       '480-align-recheck.sh.log':'cn20486','482-align-confirm.sh.log':'cn20486'}
rows=list(csv.DictReader((logs/'formal.csv').open()))
assert set(nodes)<=set(r['file'] for r in rows)
groups=defaultdict(lambda:defaultdict(dict))
for r in rows:
    if r['file'] not in nodes:continue
    groups[nodes[r['file']],r['m'],r['n']][r['mode']][r['file'],r['round']]=float(r['gflops'])
result=[]
for (node,m,n),modes in sorted(groups.items()):
    final=modes['final']; row=dict(node=node,m=m,n=n,rounds=len(final))
    for mode in ('control','final','off'):
        assert modes[mode].keys()==final.keys()
        row[mode+'_median']=median(modes[mode].values())
    for base in ('control','off'):
        ratios=[math.log(v/modes[base][key]) for key,v in final.items()]
        rng=random.Random(13729)
        samples=sorted(mean(rng.choices(ratios,k=len(ratios))) for _ in range(10000))
        row[base+'_paired_pct']=100*math.expm1(mean(ratios))
        row[base+'_lo95_pct']=100*math.expm1(samples[250])
        row[base+'_hi95_pct']=100*math.expm1(samples[9749])
        row[base+'_wins']=sum(x>0 for x in ratios)
    result.append(row)
with (logs/'final_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,result[0].keys(),lineterminator='\n');w.writeheader();w.writerows(result)
for r in result:print(r)
