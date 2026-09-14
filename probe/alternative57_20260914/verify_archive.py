#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,re,subprocess
here=Path(__file__).resolve().parent;root=here.parents[1];out=here/'generated';logs=root/'logs/alternative57_20260914'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={491:'RESIDUE_FUSED_DONE',492:'RESIDUE_FASTMOD_DONE',493:None,494:'DIAGFIXED_FORMAL_DONE',495:'OWNERDIAG_DONE',496:'DIAGPREFETCH_DONE',497:'COMPACT_CONFIRM_DONE',498:'COMPACT_SECOND_DONE',499:'COMPACT_CLOSE_DONE'}
checks=0;complete=[];known_solutions=0;formal=0;binaries={};guard_comparisons=0;changed_inputs=0
for number,marker in expected.items():
    found=list(logs.glob(f'{number}-*.sh.log'))
    if not found:continue
    assert len(found)==1
    path=found[0];text=path.read_text()
    if marker is not None:assert marker in text,path
    else:
        assert number==493 and 'fatal error: bench_trsm.c: No such file or directory' in text
        assert text.count('INDEPENDENT_TOTAL cases=64 failures=0')==4
    assert not re.search(r'\bFAIL\b',text),path
    status=path.with_suffix('.status');assert status.read_text().strip()==('1' if number==493 else '0'),status
    for digest,name in re.findall(r'^([a-f0-9]{64})  ([\w.-]+)$',text,re.M):
        if name in ('control','diagcompact'):
            binaries.setdefault(name,[]).append(digest)
        else:
            source=out/('trsm-diagcompact.c' if name=='trsm.c' and number==499 else name)
            assert sha(source)==digest,(path,name);checks+=1
    for count,fail in re.findall(r'INDEPENDENT_TOTAL cases=(\d+) failures=(\d+)',text):
        assert int(fail)==0;known_solutions+=int(count)
    formal+=len(re.findall(r'^\s*\d+ x\s*\d+\s+[\d.]+\s+[\d.]+\s+[\deE+.-]+\s+PASS$',text,re.M))
    guard_comparisons+=sum(map(int,re.findall(r'COMPACT_GUARD comparisons=(\d+) PASS',text)))
    changed_inputs+=len(re.findall(r'changed-input case=\d+ iteration=\d+ maxerr=[^ ]+ PASS',text))
    complete.append(number)
for name,values in binaries.items():assert len(set(values))==1,(name,values)
assert (root/'bench_trsm.c').read_bytes()==subprocess.check_output(['git','cat-file','blob','a16843e2c49f16fa5b76ce6cd0c6f739c06b42ab'],cwd=root)
assert sha(out/'trsm-v55.c')=='1f6a3dc4712bbb04c4a7f8cfbc042ba8c306681bcfd253a51242a47b4c518350'
assert sha(root/'trsm.c')==sha(out/'trsm-diagcompact.c')=='951a1594b22b5f424f47fc5b0a7b7ce2c56cbb38601267ef45efbcdf2bd9c6f6'
assert sha(root/'run.sh')=='c7a21ad9e66ba9c3794f1c5940b39b0acb9722dd224ba6217329563d4cb1824b'
print(json.dumps(dict(complete=complete,missing=sorted(set(expected)-set(complete)),source_hashes=checks,independent_cases=known_solutions,formal_pass_rows=formal,guard_comparisons=guard_comparisons,changed_inputs=changed_inputs,binary_hash_observations={k:len(v) for k,v in binaries.items()},production_matches_validated_candidate=True)))
