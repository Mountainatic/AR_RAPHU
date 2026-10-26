from pathlib import Path
import json,tarfile,hashlib
BASE=Path('/root/autodl-tmp');ROOT=BASE/'PRISM_PAPER_OPTIMIZATION_20261026'
roots=[ROOT,BASE/'PRISM_PAPER_SPLINE_REPAIR_20261026',BASE/'PRISM_CZ_E128_20261026',BASE/'PRISM_CZ_E128_TRANSPORT_20261026']
def digest(stream):
    h=hashlib.sha256()
    while True:
        b=stream.read(1024*1024)
        if not b:return h.hexdigest()
        h.update(b)
def sha(path):
    with Path(path).open('rb') as f:return digest(f)
manifest={}
for root in roots:
    assert (root/'QUEUE_COMPLETED.json').exists()
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='PACKAGE_FILE_MANIFEST.json':continue
        assert not p.is_symlink()
        manifest[str(p.relative_to(BASE))]=dict(bytes=p.stat().st_size,sha256=sha(p))
mf=ROOT/'PACKAGE_FILE_MANIFEST.json';mf.write_text(json.dumps(manifest,indent=2))
archive=BASE/'PRISM_PAPER_OPTIMIZATION_20261026_PRIVATE.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=6) as t:
    for name in manifest:t.add(BASE/name,arcname=name,recursive=False)
    t.add(mf,arcname=str(mf.relative_to(BASE)),recursive=False)
with tarfile.open(archive,'r:gz') as t:
    members=t.getmembers();assert len(members)==len(manifest)+1
    for member in members:
        assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
        if member.name==str(mf.relative_to(BASE)):assert digest(t.extractfile(member))==sha(mf)
        else:
            expect=manifest[member.name];assert member.size==expect['bytes']
            assert digest(t.extractfile(member))==expect['sha256'],member.name
receipt=dict(status='VERIFIED',archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=len(members),
    per_file_manifest_members=len(manifest),manifest_sha256=sha(mf),all_members_verified=True,
    contains_private_CZ_development_predictions_and_models=True,raw_workbook_included=False,reserved_tail_included=False,
    GitHub_modified=False)
(BASE/'PRISM_PAPER_OPTIMIZATION_20261026_PACKAGE.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)
