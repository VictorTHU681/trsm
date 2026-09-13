#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
import csv
import math
import re
import random
from statistics import mean,median

here=Path(__file__).resolve().parent
logs=here.parents[1]/'logs/fold54_20260913'
headers=('FOLD54_BASE ','FOLD54_PROFILE ','FOLD54_FORMAL ','FOLD54_KERNEL ',
         'FOLD54_PREFETCH ','FOLD54_TILE ','DIAGSTORE_FORMAL ','DIAGSTORE_PROFILE ',
         'OWNER_WIDTH_FORMAL ','OWNER_WIDTH_PROFILE ','NOINSTR_FORMAL ',
         'DIAG_SME_FORMAL ','DIAG_SME_PROFILE ','DIAG_COMBO_FORMAL ','DIAG_COMBO_PROFILE ',
         'FOLD_LAST_FORMAL ','FOLD_LAST_PROFILE ','DIAG_FINAL_FORMAL ',
         'DIAG_LATE_FORMAL ','DIAG_LATE_PROFILE ','DIAG_FUSE_FORMAL ','DIAG_FUSE_PROFILE ',
         'DIAG8_FORMAL ','DIAG8_PROFILE ','DIAG8_COMPACT_FORMAL ','DIAG8_COMPACT_PROFILE ',
         'FUSED_FINAL_FORMAL ','FUSED_FINAL_PROFILE ')
score=re.compile(r'^\s*(\d+) x\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([\deE.+-]+)\s+(PASS|FAIL)$')
data=defaultdict(list)
for path in sorted(logs.glob('*.log')):
    context='';call=0;setup={}
    for line_no,line in enumerate(path.read_text().splitlines(),1):
        if line.startswith(headers):context=line;call=0
        meta=dict(file=path.name,line=line_no,context=context,**dict(re.findall(r'(\w+)=([^\s]+)',context)))
        match=score.match(line)
        if match:
            m,n,ms,gf,err,status=match.groups()
            data['formal'].append(dict(meta,m=m,n=n,ms=ms,gflops=gf,max_error=err,status=status))
        if line.startswith('[trsm prof] m='):
            match=re.search(r'setup ([0-9.]+) ms, parallel ([0-9.]+) ms',line)
            setup=dict(setup_ms=match[1],parallel_ms=match[2]) if match else {}
        if '每线程平均 ms' in line:
            phase={k+'_ms':v for k,v in re.findall(r'\b(prep|load|pack|wait|kloop|epi|dkloop|ddump|depi|store) ([0-9.]+)',line)}
            data['phases'].append(dict(meta,call_index=call,warmup=int(call==0),**setup,**phase));call+=1
        if re.search(r'\bFAIL\b',line):data['failures'].append(dict(meta,output=line))
def write(name,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (logs/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,columns,lineterminator='\n');w.writeheader();w.writerows(rows)
for name,rows in data.items():write(name,rows)
assert all(r['status']=='PASS' for r in data['formal'])
groups=defaultdict(list)
for row in data['formal']:
    if 'round' in row and 'mode' in row:groups[(row['file'],row['m'],row['n'])].append(row)
summary=[]
for (file,m,n),rows in groups.items():
    control={r['round']:float(r['gflops']) for r in rows if r['mode']=='control'}
    off={r['round']:float(r['gflops']) for r in rows if r['mode']=='off'}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        rr=[r for r in rows if r['mode']==mode];values=[float(r['gflops']) for r in rr]
        result=dict(file=file,m=m,n=n,mode=mode,rounds=len(rr),median_gflops=median(values),min_gflops=min(values),max_gflops=max(values))
        for tag,base in [('control',control),('off',off)]:
            ratios=[math.log(float(r['gflops'])/base[r['round']]) for r in rr if r['round'] in base]
            if ratios:
                result['paired_'+tag+'_pct']=100*math.expm1(mean(ratios))
                rng=random.Random(724913)
                samples=sorted(mean(rng.choices(ratios,k=len(ratios))) for _ in range(5000))
                result['paired_'+tag+'_lo95_pct']=100*math.expm1(samples[125])
                result['paired_'+tag+'_hi95_pct']=100*math.expm1(samples[4874])
                result['paired_'+tag+'_wins']=sum(x>0 for x in ratios)
        summary.append(result)
write('summary',summary)
groups=defaultdict(list)
for r in data['phases']:
    if not r['warmup']:groups[(r['file'],r['context'])].append(r)
phases=[]
for (file,context),rows in groups.items():
    result=dict(file=file,context=context,calls=len(rows))
    for key in rows[0]:
        if key.endswith('_ms'):result['mean_'+key]=mean(float(r[key]) for r in rows)
    phases.append(result)
write('phase_summary',phases)
print({k:len(v) for k,v in data.items()})
print({p.name:int(p.read_text()) for p in logs.glob('*.status')})
