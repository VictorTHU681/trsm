#!/usr/bin/env python3
"""Keep every sample; exclude only explicitly identified warmups in summaries."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean, median

here=Path(__file__).resolve().parent
log=here.parents[1]/'logs/fused_20260913'
kv=re.compile(r'(\w+)=([^\s]+)')
score=re.compile(r'^\s*(\d+) x\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\deE+.-]+)\s+(PASS|FAIL)$')
headers=('RECURSIVE_FORMAL ','RECURSIVE_PROFILE ','BATCH_POLICY ',
         'FRONT_FORMAL ','FRONT_RECHECK ','FRONT_PROFILE ','FRONT_BOUNDARY ','DIRECT_BALANCED_MB112')
samples=defaultdict(list)
checks={}
for path in sorted(log.glob('*.log')):
    context='';counts=Counter();solve={}
    contents=path.read_text()
    assert not re.search(r'\bFAIL\b',contents),path
    checks[path.name]={'pass_lines':len(re.findall(r'\bPASS\b',contents))}
    for line_number,line in enumerate(contents.splitlines(),1):
        if line.startswith(headers):context=line;counts.clear()
        meta=dict(file=path.name,line=line_number,context=context,**dict(kv.findall(context)))
        if line.startswith(('FUSED m=','BATCH_FUSED m=','DIRECT_FUSED m=')):
            d=dict(kv.findall(line))
            samples['component'].append(dict(meta,family=line.split()[0],warmup=int(int(d['rep'])<0),**d))
        if line.startswith('FUSED_TRSM '):
            index=counts['recursive'];counts['recursive']+=1
            samples['recursive_phase'].append(dict(meta,call_index=index,warmup=int(index==0),**dict(kv.findall(line))))
        m=score.match(line)
        if m:
            a,b,ms,gf,error,status=m.groups()
            samples['formal'].append(dict(meta,m=a,n=b,time_ms=ms,gflops=gf,max_error=error,status=status))
        if line.startswith('[trsm prof] m='):
            solve=dict(kv.findall(line))
            times=re.search(r'setup ([\d.]+) ms, parallel ([\d.]+) ms',line)
            if times:solve.update(setup_ms=times[1],parallel_ms=times[2])
        if '每线程平均 ms：' in line:
            phase={key+'_ms':v for key,v in re.findall(r'\b(prep|load|pack|xcopy|wait|kloop|dump|epi|dkloop|ddump|depi|store) ([\d.]+)',line)}
            index=counts['trsm_phase'];counts['trsm_phase']+=1
            samples['trsm_phase'].append(dict(meta,call_index=index,warmup=int(index==0),**solve,**phase))

def write(name,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (log/(name+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,columns,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
for name,rows in samples.items():write(name,rows)
groups=defaultdict(list)
for family in ('component','recursive_phase','trsm_phase'):
    for row in samples[family]:
        if row['warmup']:continue
        key=(family,row['file'],row['context'],row.get('m'),row.get('n'),row.get('k'),
             row.get('level'),row.get('alpha'),row.get('beta'),row.get('pad'),row.get('exp'))
        groups[key].append(row)
summary=[]
for key,rows in groups.items():
    family,file,context,m,n,k,level,alpha,beta,pad,exp=key
    row=dict(family=family,file=file,context=context,m=m,n=n,k=k,level=level,alpha=alpha,beta=beta,pad=pad,exp=exp,count=len(rows))
    for field in rows[0]:
        if field.endswith('_ms') or field=='nominal_GFLOPS':
            row['median_'+field]=median(float(r[field]) for r in rows)
            row['mean_'+field]=mean(float(r[field]) for r in rows)
    summary.append(row)
write('phase_summary',summary)
for prefix,output in (('FRONT_FORMAL ','front_summary'),('FRONT_RECHECK ','front_recheck_summary')):
    front=[r for r in samples['formal'] if r['context'].startswith(prefix)]
    if not front:continue
    control={r['round']:float(r['gflops']) for r in front if r['window']=='control'}
    paired=[]
    for window in ('control','16','32','48','64'):
        rows=[r for r in front if r['window']==window]
        if not rows:continue
        ratios=[float(r['gflops'])/control[r['round']] for r in rows if r['round'] in control]
        paired.append(dict(window=window,rounds=len(rows),median_gflops=median(float(r['gflops']) for r in rows),
                           min_gflops=min(float(r['gflops']) for r in rows),max_gflops=max(float(r['gflops']) for r in rows),
                           paired_change_pct=100*(math.exp(mean(math.log(x) for x in ratios))-1) if ratios else ''))
    write(output,paired)
for status in sorted(log.glob('*.status')):
    assert status.read_text().strip()=='0',(status,status.read_text())
    checks[status.name]={'exit_status':0}
print(json.dumps({'records':{k:len(v) for k,v in samples.items()},'logs':checks},indent=2))
