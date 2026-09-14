#!/usr/bin/env python3
"""Parse every bench row; pair formal runs without excluding slow rounds."""
from pathlib import Path
import csv,json,re,statistics,math,random
root=Path(__file__).resolve().parents[2];logs=root/'logs/steady58_20260914'
rows=[];phases=[]
for path in sorted(logs.glob('*.sh.log')):
    context={};call=-1
    for line in path.read_text().splitlines():
        if re.match(r'^\w+_(?:FORMAL|PROFILE)\b',line):
            context=dict(re.findall(r'(\w+)=([^ ]+)',line))|{'header':line.split()[0]};call=-1
        if line.startswith('[trsm prof] m='):
            call+=1
        if '每线程平均 ms' in line:
            data=dict(re.findall(r'\b(prep|load|pack|xcopy|wait|kloop|dump|epi|dkloop|ddump|depi|store) ([\d.]+)',line))
            phases.append(context|data|{'file':path.name,'call':call})
        found=re.fullmatch(r'\s*(\d+) x\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\deE+.-]+)\s+(PASS|FAIL)',line)
        if found:
            rows.append(context|dict(zip(('m','n','ms','gflops','error','result'),found.groups()))|{'file':path.name})
for name,items in [('measurements.csv',rows),('phases.csv',phases)]:
    with (logs/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in items for k in r}),lineterminator='\n');w.writeheader();w.writerows(items)
summary=[];rng=random.Random(20260914)
for file,m,n,header in sorted({(r['file'],r['m'],r['n'],r.get('header','')) for r in rows if r.get('header','').endswith('_FORMAL')}):
    group=[r for r in rows if (r['file'],r['m'],r['n'],r.get('header',''))==(file,m,n,header)]
    points={mode:{int(r['round']):float(r['gflops']) for r in group if r['mode']==mode} for mode in sorted({r['mode'] for r in group})}
    item=dict(file=file,m=int(m),n=int(n),header=header,modes={})
    for mode,values in points.items():
        rec=dict(rounds=len(values),median=statistics.median(values.values()),range=[min(values.values()),max(values.values())])
        for reference in ('control','off'):
            if reference not in points or mode==reference:continue
            diffs=[math.log(values[r]/points[reference][r]) for r in sorted(values) if r in points[reference]]
            if not diffs:continue
            boot=sorted(100*math.expm1(statistics.mean(rng.choices(diffs,k=len(diffs)))) for _ in range(10000))
            rec['vs_'+reference]=dict(paired_percent=100*math.expm1(statistics.mean(diffs)),bootstrap95=[boot[249],boot[9749]],positive_pairs=sum(d>0 for d in diffs),pairs=len(diffs))
        item['modes'][mode]=rec
    summary.append(item)
(logs/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
