#!/usr/bin/env python3
"""Keep every timed row, including warmups and slow runs, with its experiment."""
from pathlib import Path
import csv,json,re,statistics,math
here=Path(__file__).resolve().parent;logs=here.parents[1]/'logs/alternative57_20260914'
rows=[]
for path in sorted(logs.glob('*.sh.log')):
    context={}
    for line in path.read_text().splitlines():
        data=dict(re.findall(r'(\w+)=([^ ]+)',line))
        if line.startswith(('DIAGFIXED_FORMAL','DIAGFIXED_PROFILE','OWNERDIAG_FORMAL','OWNERDIAG_PROFILE','DIAGPREFETCH_FORMAL','DIAGPREFETCH_PROFILE','COMPACT_FORMAL','COMPACT_PROFILE')):
            context=data|{'header':line.split()[0]}
        elif line.startswith(('ENCODE_PERF ','FUSED_RNS ','FASTMOD_RNS ')):
            rows.append(data|{'file':path.name,'kind':line.split()[0]})
        else:
            found=re.fullmatch(r'\s*(\d+) x\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\deE+.-]+)\s+(PASS|FAIL)',line)
            if found:
                row=dict(zip(('m','n','ms','gflops','error','result'),found.groups()))
                rows.append(context|row|{'file':path.name,'kind':'formal'})
fields=sorted({k for row in rows for k in row})
with (logs/'measurements.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
result={'observations':len(rows),'encoding':[],'product_best_screening':[],'formal':[]}
for file in sorted({r['file'] for r in rows}):
    enc=[r for r in rows if r['file']==file and r['kind']=='ENCODE_PERF']
    for m,k in sorted({(int(r['M']),int(r['K'])) for r in enc}):
        med={mode:statistics.median(float(r['total_ms']) for r in enc if int(r['M'])==m and int(r['K'])==k and int(r['sequential'])==mode) for mode in (0,1)}
        result['encoding'].append(dict(file=file,m=m,k=k,direct_ms=med[0],sequential_ms=med[1],speedup=med[0]/med[1]))
    products=[r for r in rows if r['file']==file and r['kind'] in ('FUSED_RNS','FASTMOD_RNS') and int(r['rep'])>=0]
    for m,k,fused in sorted({(int(r['M']),int(r['K']),int(r['fused'])) for r in products}):
        group=[r for r in products if (int(r['M']),int(r['K']),int(r['fused']))==(m,k,fused)]
        candidates=[]
        for mb,nc in sorted({(int(r['MB']),int(r['NC'])) for r in group}):
            points=[r for r in group if (int(r['MB']),int(r['NC']))==(mb,nc)]
            candidates.append(dict(mb=mb,nc=nc,ms=statistics.median(float(r['ms']) for r in points),gflops=statistics.median(float(r['GFLOPS']) for r in points),count=len(points)))
        best=max(candidates,key=lambda r:r['gflops'])
        result['product_best_screening'].append(dict(file=file,m=m,k=k,fused=fused,**best))
    formal=[r for r in rows if r['file']==file and r['kind']=='formal' and r.get('header','').endswith('_FORMAL')]
    for m,n in sorted({(r['m'],r['n']) for r in formal}):
        group=[r for r in formal if (r['m'],r['n'])==(m,n)]
        for mode in sorted({r['mode'] for r in group}):
            points=[r for r in group if r['mode']==mode]
            values=[float(r['gflops']) for r in points]
            item=dict(file=file,m=int(m),n=int(n),mode=mode,rounds=len(points),median=statistics.median(values),minimum=min(values),maximum=max(values))
            for reference in ('control','off'):
                ref={r['round']:float(r['gflops']) for r in group if r['mode']==reference}
                pairs=[math.log(float(r['gflops'])/ref[r['round']]) for r in points if r['round'] in ref]
                if pairs:item['paired_vs_'+reference+'_pct']=100*math.expm1(statistics.mean(pairs))
            result['formal'].append(item)
(logs/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,ensure_ascii=False))
