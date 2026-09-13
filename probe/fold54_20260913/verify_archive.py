#!/usr/bin/env python3
"""Validate reconstructed files against hashes collected from executed roots."""
from pathlib import Path
import hashlib,json,re,subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
logs=root/'logs/fold54_20260913'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
matched={}
for manifest in sorted(logs.glob('source-sha256-*.txt')):
    count=0
    for line in manifest.read_text().splitlines():
        digest,name=line.split(maxsplit=1)
        path=Path(name)
        local=here/(path.name.removesuffix('.done') if path.parts[0]=='results' else path)
        assert sha(local)==digest,(manifest.name,local)
        count+=1
    matched[manifest.name]=count
assert matched.get('source-sha256-fold.txt')==33,matched
original=subprocess.check_output(['git','cat-file','blob','a16843e2c49f16fa5b76ce6cd0c6f739c06b42ab'],cwd=root)
assert (root/'bench_trsm.c').read_bytes()==original
assert (here/'generated/bench_trsm.c').read_bytes()==original
assert sha(here/'generated/trsm-v54.c')=='936704cb25f79f9b497e33e1e9ba7ceaa972173421dd3863be049fb673dc0589'
assert sha(here/'generated/trsm-diag-final.c')=='971e4b59bc28eb1bb92586e2ad0523c957bbb5cc400a96c96cb36bed4287e51f'
assert sha(here/'generated/trsm-fused-align.c')=='7a6eeb3803b32dcad0f40c2180e8a6246d5c150a7a121d06d8e69bcff7b29e97'
assert sha(root/'run.sh')=='c7a21ad9e66ba9c3794f1c5940b39b0acb9722dd224ba6217329563d4cb1824b'
statuses={p.name.removesuffix('.status'):int(p.read_text()) for p in logs.glob('*.status')}
known_failures={'451-fold-screen.sh':1,'472-fused-close.sh':134}
for task,status in statuses.items():assert status==known_failures.get(task,0),(task,status)
assert {int(name[:3]) for name in statuses}==set(range(450,483)),statuses
for name,marker in [('462-diag-final.sh','DIAG_FINAL_SCREEN_DONE'),
                    ('463-diag-crossnode.sh','DIAG_FINAL_CROSSNODE_DONE'),
                    ('464-diag-close.sh','DIAG_FINAL_CLOSE_DONE'),
                    ('470-fused-final.sh','FUSED_FINAL_SCREEN_DONE'),
                    ('471-fused-crossnode.sh','FUSED_CROSSNODE_DONE'),
                    ('473-fused-balanced.sh','FUSED_BALANCED_DONE'),
                    ('474-fused-balanced.sh','FUSED_BALANCED_DONE'),
                    ('475-fused-close.sh','FUSED_FINAL_CLOSE_DONE'),
                    ('476-audit-jit-check.sh','AUDIT_JIT_CHECK_DONE'),
                    ('478-fused-align.sh','FUSED_ALIGN_DONE'),
                    ('479-align-crossnode.sh','FUSED_ALIGN_CROSS_DONE'),
                    ('480-align-recheck.sh','FUSED_ALIGN_CROSS_DONE'),
                    ('481-align-close.sh','FUSED_ALIGN_CLOSE_DONE'),
                    ('482-align-confirm.sh','FUSED_ALIGN_CONFIRM_DONE')]:
    if name in statuses:assert marker in (logs/(name+'.log')).read_text(),name
failures=[(p.name,line) for p in logs.glob('*.log') for line in p.read_text().splitlines() if re.search(r'\bFAIL\b',line)]
assert len(failures)==1 and failures[0][0]=='476-audit-jit-check.sh.log',failures
assert 'pack=0 expected_pack=4' in failures[0][1],failures
expected_jobs={'1591640':'SUCCEEDED','1593212':'SUCCEEDED','1593270':'SUCCEEDED',
               '1593837':'FAILED','1593872':'SUCCEEDED'}
actual_jobs={}
for line in (logs/'jobs-final.txt').read_text().splitlines():
    fields=line.split()
    if fields and fields[0] in expected_jobs:actual_jobs[fields[0]]=fields[2]
assert actual_jobs==expected_jobs,actual_jobs
assert (root/'trsm.c').read_bytes()==(here/'generated/trsm-fused-align.c').read_bytes()
oldzip=root/'trsm_submit_v53_984972a.zip'
if oldzip.exists():assert sha(oldzip)=='96546b272831921105d07f6efc6f339427b1ef6b6ab597e3d6b39703473ce640'
print(json.dumps(dict(matched=matched,statuses=statuses,final_jobs=actual_jobs,official_bench_unchanged=True,
                     production_sha256=sha(root/'trsm.c')),indent=2,ensure_ascii=False))
