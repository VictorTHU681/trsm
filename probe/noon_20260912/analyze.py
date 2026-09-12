#!/usr/bin/env python3
"""Export every retained sample; no outlier deletion or timing/profile mixing."""
from pathlib import Path
import collections
import csv
import re
import statistics

root=Path(__file__).resolve().parents[2]
raw=root/'logs/noon_20260912'
rows=[]
for path in sorted(raw.glob('*.log')):
    phase=-1; last_k=0; ant=0; owner_round=None; owner_tag=None
    for line in path.read_text().splitlines():
        values=dict(re.findall(r'(\w+)=([\w.+-]+)',line))
        if line.startswith('H24_ANT='):ant=int(values['H24_ANT'])
        if line.startswith('SHORTK_CHECK'):
            if int(values['KC'])<last_k:phase+=1
            last_k=int(values['KC'])
        if line.startswith('OWNER_IMPORT_SCORE'):
            owner_round=values['round'];owner_tag=values['tag']
        if line.startswith('OWNER_IMPORT_PROFILE'):owner_tag=None
        m=re.search(r'512 x 19968\s+([\d.]+)\s+([\d.]+)\s+([\deE.+-]+)\s+(PASS|FAIL)',line)
        if m and owner_tag:
            rows.append(dict(source=path.name,event='FORMAL',round=owner_round,tag=owner_tag,ms=m[1],GFLOPS=m[2],error=m[3],check=m[4]))
        if values and any(line.startswith(prefix) for prefix in ['HYBRID_PEAK ','H24_PERF ','H24_PMU ','H24_MEMORY_PERF ','H24_DUPLEX_PERF ','SHORTK_PERF ','SHORTK_PMU ','LOAD_PEAK ']):
            event=line.split()[0];row=dict(source=path.name,event=event,**values)
            if event=='SHORTK_PERF':row['threads']=1 if phase<1 else 38
            if '394-' in path.name:row['ant']=ant
            rows.append(row)
columns=['source','event']+sorted(set().union(*(r.keys() for r in rows))-{'source','event'})
with (raw/'measurements.csv').open('w') as f:
    writer=csv.DictWriter(f,columns);writer.writeheader();writer.writerows(rows)
print('Retained samples:',len(rows))
for source in ['389-hybrid24-memory.sh.log','391-hybrid24-duplex.sh.log','393-hybrid24-acc2.sh.log','394-hybrid24-ant.sh.log']:
    groups=collections.defaultdict(list)
    for r in rows:
        if r['source']==source and 'GFLOPS' in r:groups[(r.get('K','64'),r.get('ant','-'),r['mode'],r['hybrid'])].append(float(r['GFLOPS']))
    for k in sorted(groups):
        if k[-1]=='1':
            b=statistics.median(groups[k[:-1]+('0',)]);v=statistics.median(groups[k]);print(source,k,'base',b,'candidate',v,'change %',100*(v/b-1))
