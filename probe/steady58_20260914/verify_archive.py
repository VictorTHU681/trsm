#!/usr/bin/env python3
"""Check complete executions, source identity, unchanged harness, and counts."""
from pathlib import Path
import hashlib,json,re,subprocess
here=Path(__file__).resolve().parent;root=here.parents[1];out=here/'generated';logs=root/'logs/steady58_20260914'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
markers={500:'STEADY_PROFILE_DONE',501:'TC3COMPACT_DONE',502:'OWNER58_DONE',503:'PRELOAD58_DONE',504:'FINAL58_CONFIRM_DONE',505:'FINAL58_VALIDATE_DONE',506:'FINAL58_CONFIRM_DONE'}
complete=[];hashes=0;independent=0;formal=0;guard=0;changed=0;binaries={}
for number,marker in markers.items():
    paths=list(logs.glob(f'{number}-*.sh.log'))
    assert len(paths)==1,(number,paths)
    path=paths[0];data=path.read_text()
    assert marker in data and not re.search(r'\bFAIL\b',data),path
    assert path.with_suffix('.status').read_text().strip()=='0',path
    assert path.with_suffix('.done').read_bytes()==(here/path.name.removesuffix('.log')).read_bytes(),path
    for digest,name in re.findall(r'^([a-f0-9]{64})  ([\w.-]+)$',data,re.M):
        if name in ('control','candidate'):
            binaries.setdefault(name,[]).append(digest)
        else:
            source=out/('trsm-v57.c' if name=='trsm.c' else name)
            assert sha(source)==digest,(path,name);hashes+=1
    for count,fail in re.findall(r'INDEPENDENT_TOTAL cases=(\d+) failures=(\d+)',data):
        assert int(fail)==0;independent+=int(count)
    formal+=len(re.findall(r'^\s*\d+ x\s*\d+\s+[\d.]+\s+[\d.]+\s+[\deE+.-]+\s+PASS$',data,re.M))
    guard+=sum(map(int,re.findall(r'COMPACT_GUARD comparisons=(\d+) PASS',data)))
    changed+=len(re.findall(r'changed-input case=\d+ iteration=\d+ maxerr=[^ ]+ PASS',data))
    complete.append(number)
for name,values in binaries.items():assert len(values)==2 and len(set(values))==1,(name,values)
assert (root/'bench_trsm.c').read_bytes()==subprocess.check_output(['git','cat-file','blob','a16843e2c49f16fa5b76ce6cd0c6f739c06b42ab'],cwd=root)
assert sha(root/'run.sh')==sha(out/'run.sh')=='c7a21ad9e66ba9c3794f1c5940b39b0acb9722dd224ba6217329563d4cb1824b'
assert sha(out/'trsm-v56.c')=='951a1594b22b5f424f47fc5b0a7b7ce2c56cbb38601267ef45efbcdf2bd9c6f6'
assert sha(out/'trsm-v57.c')=='15d1e2bfd7994ca5cb752ae8f0678d0f82fbab6773041ddd38079faf3025ab24'
assert sha(root/'trsm.c')==sha(out/'trsm-v57.c')
result=dict(complete=complete,source_hashes=hashes,independent_cases=independent,bench_pass_rows=formal,guard_comparisons=guard,changed_inputs=changed,binary_hashes={k:v[0] for k,v in binaries.items()},production_matches_validated_candidate=True)
(logs/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
