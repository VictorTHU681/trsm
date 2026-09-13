#!/usr/bin/env python3
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re
from statistics import mean,median
here=Path(__file__).resolve().parent
log=here.parents[1]/'logs/owner53_20260913'
kv=re.compile(r'(\w+)=([^\s]+)')
score=re.compile(r'^\s*(\d+) x\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\deE+.-]+)\s+(PASS|FAIL)$')
headers=('OWNER53_FORMAL ','OWNER53_PROFILE ','IMPLICIT_ONLY_FORMAL ','IMPLICIT_ONLY_PROFILE ',
         'DIAG_FORMAL ','DIAG_PROFILE ','IMPLICIT_CROSSNODE ','IMPLICIT_FINAL_FORMAL ',
         'IMPLICIT_FINAL_PROFILE ')
samples=defaultdict(list)
for path in sorted(log.glob('*.log')):
    context='';calls=0;setup={}
    for number,line in enumerate(path.read_text().splitlines(),1):
        if line in ('IMPLICIT_RUN_SH','IMPLICIT_ORIGINAL_COMMANDS'):context=line;calls=0
        if line.startswith(headers):context=line;calls=0
        meta=dict(file=path.name,line=number,context=context,**dict(kv.findall(context)))
        match=score.match(line)
        if match:
            m,n,ms,gf,err,status=match.groups()
            samples['formal'].append(dict(meta,m=m,n=n,ms=ms,gflops=gf,max_error=err,status=status))
        if line.startswith('[trsm prof] m='):
            setup=dict(kv.findall(line))
            match=re.search(r'setup ([\d.]+) ms, parallel ([\d.]+) ms',line)
            if match:setup.update(setup_ms=match[1],parallel_ms=match[2])
        if '每线程平均 ms：' in line:
            phase={k+'_ms':v for k,v in re.findall(r'\b(prep|load|pack|xcopy|wait|kloop|dump|epi|dkloop|ddump|depi|store) ([\d.]+)',line)}
            samples['phases'].append(dict(meta,call_index=calls,warmup=int(calls==0),**setup,**phase));calls+=1
        if line.startswith('DIAG_ORDER_MICRO '):samples['diag_micro'].append(dict(meta,**dict(kv.findall(line))))
        if re.search(r'\bFAIL\b',line):samples['failures'].append(dict(meta,output=line))

def write(name,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (log/(name+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,columns,lineterminator='\n');writer.writeheader();writer.writerows(rows)
for name,rows in samples.items():write(name,rows)
assert all(r['status']=='PASS' for r in samples['formal'])
groups=defaultdict(list)
for row in samples['formal']:
    if 'round' in row:groups[(row['file'],row['m'],row['n'])].append(row)
summary=[]
for (file,m,n),rows in groups.items():
    baseline={r['round']:float(r['gflops']) for r in rows if r['mode']=='control'}
    off={r['round']:float(r['gflops']) for r in rows if r['mode'] in ('off','0')}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        rr=[r for r in rows if r['mode']==mode];values=[float(r['gflops']) for r in rr]
        result=dict(file=file,m=m,n=n,mode=mode,rounds=len(rr),median_gflops=median(values),min_gflops=min(values),max_gflops=max(values))
        for tag,base in [('v53',baseline),('off',off)]:
            ratios=[math.log(float(r['gflops'])/base[r['round']]) for r in rr if r['round'] in base]
            if not ratios:continue
            result['paired_'+tag+'_pct']=100*(math.exp(mean(ratios))-1)
            # Descriptive paired bootstrap, retaining every slow sample.
            rng=random.Random(1591041);boot=sorted(mean(rng.choices(ratios,k=len(ratios))) for _ in range(4000))
            result['bootstrap_'+tag+'_lo_pct']=100*(math.exp(boot[100])-1)
            result['bootstrap_'+tag+'_hi_pct']=100*(math.exp(boot[3899])-1)
        summary.append(result)
write('summary',summary)
node_summary=[]
for node,files in [('cn20486',('440','443')),('cn20485',('444',)),('cn20487',('441',))]:
    controls=[];finals=[];paired=[];disabled=[]
    for prefix in files:
        rows=[r for r in samples['formal'] if r['file'].startswith(prefix) and r['m']=='512' and 'round' in r]
        base={r['round']:float(r['gflops']) for r in rows if r['mode']=='control'}
        off={r['round']:float(r['gflops']) for r in rows if r['mode']=='off'}
        controls.extend(base.values())
        for r in rows:
            if r['mode']!='final':continue
            value=float(r['gflops']);finals.append(value)
            paired.append(math.log(value/base[r['round']]))
            disabled.append(math.log(value/off[r['round']]))
    if finals:
        node_summary.append(dict(node=node,rounds=len(finals),v53_median_gflops=median(controls),
            v54_median_gflops=median(finals),median_pct=100*(median(finals)/median(controls)-1),
            paired_v53_pct=100*math.expm1(mean(paired)),paired_off_pct=100*math.expm1(mean(disabled))))
write('node_summary',node_summary)
groups=defaultdict(list)
for r in samples['phases']:
    if not r['warmup']:groups[(r['file'],r['context'])].append(r)
phase_summary=[]
for (file,context),rows in groups.items():
    rr=dict(file=file,context=context,calls=len(rows))
    for field in rows[0]:
        if field.endswith('_ms'):rr['mean_'+field]=mean(float(r[field]) for r in rows)
    phase_summary.append(rr)
write('phase_summary',phase_summary)
statuses={p.name:int(p.read_text()) for p in log.glob('*.status')}
print(json.dumps({'records':{k:len(v) for k,v in samples.items()},'statuses':statuses},indent=2))
