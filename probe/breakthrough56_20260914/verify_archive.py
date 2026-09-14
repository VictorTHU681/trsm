#!/usr/bin/env python3
"""Verify actual archived hashes and pinned production; no inferred pass results."""
from pathlib import Path
import hashlib,re,subprocess,json
here=Path(__file__).resolve().parent;root=here.parents[1];out=here/'generated';logs=root/'logs/breakthrough56_20260914'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=0
for path in sorted(logs.glob('*.sh.log')):
    for line in path.read_text().splitlines():
        match=re.fullmatch(r'([a-f0-9]{64})  ([\w.-]+)',line)
        if match:
            digest,name=match.groups();assert sha(out/name)==digest,(path,name);checks+=1
    assert not re.search(r'\bFAIL\b',path.read_text()),path
expected={483:'INVERSE_TC2_DONE',484:'DIGIT_SCREEN_DONE',485:'BLOCKED_INVERSE_DONE',486:'RIGHTLOOKING_DONE',
          487:'TILED_RIGHT_DONE',488:'PROFILE_CONTROL_DONE',489:'RECURSIVE_RIGHT_DONE',490:'PLATFORM_LIMITS_DONE'}
present=[]
for i,marker in expected.items():
    found=list(logs.glob(f'{i}-*.sh.log'))
    if found:
        assert len(found)==1 and marker in found[0].read_text(),i;present.append(i)
for path in logs.glob('*.status'):assert path.read_text().strip()=='0',path
original=subprocess.check_output(['git','cat-file','blob','a16843e2c49f16fa5b76ce6cd0c6f739c06b42ab'],cwd=root)
assert (root/'bench_trsm.c').read_bytes()==original==(out/'bench_trsm.c').read_bytes()
assert sha(out/'trsm-v55.c')=='1f6a3dc4712bbb04c4a7f8cfbc042ba8c306681bcfd253a51242a47b4c518350'
assert sha(root/'run.sh')=='c7a21ad9e66ba9c3794f1c5940b39b0acb9722dd224ba6217329563d4cb1824b'
oldzip=root/'trsm_submit_v53_984972a.zip'
if oldzip.exists():assert sha(oldzip)=='96546b272831921105d07f6efc6f339427b1ef6b6ab597e3d6b39703473ce640'
print(json.dumps(dict(hash_checks=checks,complete_logs=present,missing_logs=sorted(set(expected)-set(present)),
                     production_matches_archived_baseline=sha(root/'trsm.c')==sha(out/'trsm-v55.c')),ensure_ascii=False))
