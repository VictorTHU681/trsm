#!/usr/bin/env python3
from pathlib import Path
import csv,json,statistics,math,random
root=Path(__file__).resolve().parents[2];logs=root/'logs/alternative57_20260914'
rows=list(csv.DictReader((logs/'measurements.csv').open()))
rng=random.Random(20260914)
result=[]
for filename,node in [('497-compact-confirm.sh.log','cn20486'),('498-compact-second.sh.log','cn20487')]:
    for m,n in [(512,19968),(2432,17024),(17024,512)]:
        group=[r for r in rows if r['file']==filename and r.get('header')=='COMPACT_FORMAL' and int(r['m'])==m and int(r['n'])==n]
        points={mode:{int(r['round']):float(r['gflops']) for r in group if r['mode']==mode} for mode in ('control','on','off')}
        if not group:continue
        assert all(len(v)==12 for v in points.values()),(filename,m)
        item=dict(node=node,m=m,n=n,rounds=12,median={k:statistics.median(v.values()) for k,v in points.items()},range={k:[min(v.values()),max(v.values())] for k,v in points.items()})
        for reference in ('control','off'):
            differences=[math.log(points['on'][r]/points[reference][r]) for r in sorted(points['on'])]
            boot=sorted(100*math.expm1(statistics.mean(rng.choices(differences,k=len(differences)))) for _ in range(10000))
            item['vs_'+reference]=dict(paired_percent=100*math.expm1(statistics.mean(differences)),bootstrap95=[boot[249],boot[9749]],positive_pairs=sum(d>0 for d in differences))
        result.append(item)
(logs/'final-summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,ensure_ascii=False))
