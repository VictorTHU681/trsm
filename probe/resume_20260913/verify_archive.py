#!/usr/bin/env python3
"""Check rebuilt sources against the files actually used by both ARM workers."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOG = ROOT/'logs/resume_20260913'
checks = {}
for manifest in sorted(LOG.glob('resume*-source-sha256.txt')):
    count = 0
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = Path(name)
        if path.parts[0] == 'generated':
            path = HERE/path
        elif path.name.endswith('.done'):
            path = HERE/path.name.removesuffix('.done').replace('422b-', '422-')
        else:
            path = HERE/path.name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
        count += 1
    checks[manifest.name] = count

production = {}
for name in ('trsm.c', 'bench_trsm.c', 'run.sh'):
    pinned = subprocess.check_output(['git', 'show', '2fcdc7a:'+name], cwd=ROOT)
    current = (ROOT/name).read_bytes()
    assert current == pinned, name
    production[name] = hashlib.sha256(current).hexdigest()
original_bench = subprocess.check_output(
    ['git', 'cat-file', 'blob', 'a16843e2c49f16fa5b76ce6cd0c6f739c06b42ab'], cwd=ROOT)
assert original_bench == (ROOT/'bench_trsm.c').read_bytes()

result = {'rebuilt_sources_and_scripts_matched': checks,
          'production_unchanged_from': '2fcdc7a', 'production_sha256': production,
          'official_bench_matches_original_blob': True}
archive = ROOT/'trsm_submit_a536122.zip'
if archive.exists():
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == '1fb3680d7b025315892f0bac9989fd622e0249b6f4d16f0b9277c85552b2e05a'
    with zipfile.ZipFile(archive) as z:
        assert sorted(i.filename for i in z.infolist() if not i.is_dir()) == [
            'trsm/bench_trsm.c', 'trsm/run.sh', 'trsm/trsm.c']
        for name in production:
            assert z.read('trsm/'+name) == (ROOT/name).read_bytes()
        assert (z.getinfo('trsm/run.sh').external_attr >> 16) & 0o777 == 0o755
    result['previous_submission_zip_unchanged_sha256'] = digest
print(json.dumps(result, indent=2, ensure_ascii=False))
