#!/usr/bin/env python3
"""Extract every archived observation, preserving warm-up and slow measurements."""
from pathlib import Path
import re,csv,json,statistics
here=Path(__file__).resolve().parent;logs=here.parents[1]/'logs/breakthrough56_20260914'
rows=[]
for path in sorted(logs.glob('*.sh.log')):
    context={};threads='1' if path.name.startswith('484-') else '38';profile_call=0
    for line in path.read_text().splitlines():
        if line.startswith(('DIGIT_THREADS=','FP64_THREADS=')):threads=line.split('=')[1]
        if line.startswith(('INV56_FORMAL','INV56_PROFILE','BLOCKED_INV ','RIGHT_FORMAL','RIGHT_CONTROL',
                            'TILED_RIGHT_FORMAL','RECURSIVE_RIGHT_FORMAL','CONTROL_PROFILE')):
            context=dict(re.findall(r'(\w+)=([^ ]+)',line));context['mode_header']=line.split()[0];profile_call=0
        match=re.fullmatch(r'\s*(\d+) x\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\deE+.-]+)\s+(PASS|FAIL)',line)
        if match:
            m,n,ms,gflops,error,result=match.groups()
            row=dict(context);row.update(file=path.name,kind='formal',m=m,n=n,ms=ms,gflops=gflops,error=error,result=result)
            rows.append(row)
        elif line.startswith(('FULLINV ','RIGHT ','TILED_RIGHT ','DIGIT_CHECK ','DIGIT_PERF ','FP64_PRODUCT ')):
            kind=line.split()[0];data=dict(re.findall(r'(\w+)=([^ ]+)',line))
            row=dict(context);row.update(data);row.update(file=path.name,kind=kind,threads=threads)
            if kind in ('FULLINV','RIGHT','TILED_RIGHT'):
                row['call_in_process']=profile_call;profile_call+=1
            rows.append(row)
fields=sorted({k for row in rows for k in row})
with (logs/'measurements.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
formal=[r for r in rows if r['kind']=='formal']
summary={'archived_logs':len(list(logs.glob('*.sh.log'))),'observations':len(rows),'formal_rows':len(formal),
         'formal_failures':sum(r['result']!='PASS' for r in formal),'best_screening_by_file_shape':[]}
for file in sorted({r['file'] for r in formal}):
    for m in (512,2432,17024):
        group=[r for r in formal if r['file']==file and int(r['m'])==m]
        if group:summary['best_screening_by_file_shape'].append(max(group,key=lambda r:float(r['gflops'])))
(logs/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='best_screening_by_file_shape'},ensure_ascii=False))
