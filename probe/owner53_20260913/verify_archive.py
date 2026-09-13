#!/usr/bin/env python3
"""Check reconstructed sources against executed copies and retained outcomes."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

here = Path(__file__).resolve().parent
root = here.parents[1]
log = root / 'logs/owner53_20260913'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
matched = {}
for manifest in sorted(log.glob('source-sha256-*.txt')):
    count = 0
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = Path(name)
        if path.parts[0] == 'generated':
            local = here / path
        elif path.parts[0] == 'results':
            local = here / path.name.removesuffix('.done')
            if local.name == '438-diag-bound.sh': local = here / 'generated' / local.name
        else:
            local = here / path
        assert sha(local) == digest, (manifest.name, local)
        count += 1
    matched[manifest.name] = count
assert len(matched) == 4, matched
official = subprocess.check_output(['git', 'cat-file', 'blob', 'a16843e2c49f16fa5b76ce6cd0c6f739c06b42ab'], cwd=root)
assert (here / 'generated/bench_trsm.c').read_bytes() == official
assert (root / 'bench_trsm.c').read_bytes() == official
assert sha(here / 'generated/trsm-v53.c') == 'dc8c8b4907c37362bd4d3d108dd4a9e35474736e7570091e7a2b2a79dd84f294'
candidate = '936704cb25f79f9b497e33e1e9ba7ceaa972173421dd3863be049fb673dc0589'
assert sha(here / 'generated/trsm-implicit-final.c') == candidate
assert (here / 'generated/run.sh').read_bytes() == subprocess.check_output(['git', 'show', '3b24f38:run.sh'], cwd=root)
expected = {'435-owner53-screen.sh': 0, '436-implicit-only.sh': 0,
            '437-diag-order.sh': 1, '438-diag-bound.sh': 0,
            '440-implicit-final.sh': 0, '442-implicit-close.sh': 1,
            '443-implicit-close.sh': 0}
statuses = {p.name.removesuffix('.status'): int(p.read_text()) for p in log.glob('*.status')}
assert statuses == expected, statuses
assert 'IMPLICIT_FINAL_VALIDATION_DONE' in (log / '440-implicit-final.sh.log').read_text()
assert 'IMPLICIT_CLOSE_DONE' in (log / '443-implicit-close.sh.log').read_text()
assert 'IMPLICIT_FINAL_CROSSNODE_DONE' in (log / '441-implicit-final-node87.sh.log').read_text()
assert 'IMPLICIT_FINAL_CROSSNODE_DONE' in (log / '444-implicit-final-node85.sh.log').read_text()
for name in ('440-implicit-final.sh.log', '441-implicit-final-node87.sh.log', '443-implicit-close.sh.log', '444-implicit-final-node85.sh.log'):
    assert 'FAIL' not in (log / name).read_text(), name
jobs = (log / 'jobs-terminal.txt').read_text()
assert jobs.count('SUCCEEDED') == 4
for job in ('1591041', '1591209', '1591253', '1591530'): assert job in jobs
result = dict(executed_source_and_script_hashes_matched=matched, official_bench_matches_original_blob=True,
              final_candidate_sha256=candidate, task_statuses=statuses,
              root_production_matches_final_candidate=sha(root / 'trsm.c') == candidate)
oldzip = root / 'trsm_submit_v53_984972a.zip'
if oldzip.exists():
    assert sha(oldzip) == '96546b272831921105d07f6efc6f339427b1ef6b6ab597e3d6b39703473ce640'
    with zipfile.ZipFile(oldzip) as z:
        assert z.testzip() is None
        assert z.read('trsm/bench_trsm.c') == official
    result['delivered_v53_zip_unchanged'] = True
print(json.dumps(result, indent=2, ensure_ascii=False))
