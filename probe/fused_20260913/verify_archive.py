#!/usr/bin/env python3
"""Verify exact executed sources, original bench, production and delivered ZIP."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

here=Path(__file__).resolve().parent
root=here.parents[1]
log=root/'logs/fused_20260913'
count=0
for line in (log/'source-sha256.txt').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    remote=Path(name)
    if remote.parts[0]=='generated':local=here/remote
    elif remote.parts[0]=='results':local=here/remote.name.removesuffix('.done').removesuffix('.running')
    elif remote.name=='worker.sh':local=root/'probe/resume_20260913/worker.sh'
    else:raise AssertionError(remote)
    assert hashlib.sha256(local.read_bytes()).hexdigest()==digest,local
    count+=1
production={}
for name in ('trsm.c','bench_trsm.c','run.sh'):
    archived=subprocess.check_output(['git','show','b138512:'+name],cwd=root)
    if name!='run.sh':assert (here/'generated'/('trsm-control.c' if name=='trsm.c' else name)).read_bytes()==archived,name
    production[name]=hashlib.sha256(archived).hexdigest()
assert (here/'generated/bench_trsm.c').read_bytes()==subprocess.check_output(
    ['git','cat-file','blob','a16843e2c49f16fa5b76ce6cd0c6f739c06b42ab'],cwd=root)
text_hashes=[line.split()[0] for line in (log/'binary-text-sha256.txt').read_text().splitlines()]
assert len(text_hashes)==2 and len(set(text_hashes))==1
result=dict(executed_source_and_script_hashes_matched=count,archived_production_commit='b138512',
            production_sha256=production,official_bench_matches_original_blob=True,
            front64_reconstruction_identical_text_sha256=text_hashes[0])
archive=root/'trsm_submit_a536122.zip'
if archive.exists():
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest=='1fb3680d7b025315892f0bac9989fd622e0249b6f4d16f0b9277c85552b2e05a'
    with zipfile.ZipFile(archive) as z:
        assert sorted(i.filename for i in z.infolist() if not i.is_dir())==['trsm/bench_trsm.c','trsm/run.sh','trsm/trsm.c']
        for name in production:assert hashlib.sha256(z.read('trsm/'+name)).hexdigest()==production[name]
        assert (z.getinfo('trsm/run.sh').external_attr>>16)&0o777==0o755
    result['previous_submission_zip_unchanged_sha256']=digest
statuses=list(log.glob('*.status'))
assert len(statuses)==8,len(statuses)
for status in statuses:assert status.read_text().strip()=='0',status
result['completed_remote_tasks']=len(statuses)
print(json.dumps(result,indent=2,ensure_ascii=False))
