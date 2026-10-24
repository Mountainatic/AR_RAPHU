"""Stdlib-only immutable experiment archive check, without loading any model/data."""
from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,sys
ROOT=Path(__file__).resolve().parents[1];archive=Path(sys.argv[1]);receipt=json.loads((ROOT/'PACKAGE_RECEIPT.json').read_text())
def digest(stream):
    h=hashlib.sha256()
    while True:
        b=stream.read(1024*1024)
        if not b:break
        h.update(b)
    return h.hexdigest()
with archive.open('rb') as f:assert digest(f)==receipt['archive_sha256']
assert archive.stat().st_size==receipt['archive_bytes']
prefix='sru_prism_window_selection_20261024/'
with tarfile.open(archive,'r:gz') as t:
    members=t.getmembers();assert len({m.name for m in members})==len(members)
    for m in members:
        p=PurePosixPath(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and m.name.startswith(prefix)
    manifest=json.load(t.extractfile(prefix+'SHA256SUMS.json'))
    assert len(manifest)==receipt['verified_members']==147 and len(members)==len(manifest)+1
    for name,row in manifest.items():
        m=t.getmember(prefix+name);assert m.size==row['bytes']
        assert digest(t.extractfile(m))==row['sha256'],name
print(json.dumps(dict(status='COMPLETED',archive_bytes=archive.stat().st_size,archive_sha256=receipt['archive_sha256'],verified_members=len(manifest),
    no_models_loaded=True,no_source_tail_read=True),indent=2))
