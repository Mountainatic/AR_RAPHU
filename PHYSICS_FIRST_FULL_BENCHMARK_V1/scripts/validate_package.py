#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gpu_common import sha256_file

FORBIDDEN_NAMES = {'.git', '__pycache__', '.pytest_cache', 'wandb', 'shared'}
FORBIDDEN_SUFFIXES = {'.xlsx', '.xls', '.tmp', '.pyc'}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-dir', required=True)
    args = parser.parse_args()
    root = Path(args.package_dir).resolve()
    manifest_path = root / 'MANIFEST.json'
    problems=[]
    if not manifest_path.is_file():
        problems.append('MANIFEST_MISSING')
        manifest={'files':[]}
    else:
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    seen=set()
    for record in manifest.get('files',[]):
        path=root/record['path']
        seen.add(record['path'])
        if not path.is_file():
            problems.append('MISSING:'+record['path'])
        elif path.stat().st_size != record['size']:
            problems.append('SIZE:'+record['path'])
        elif sha256_file(path) != record['sha256']:
            problems.append('HASH:'+record['path'])
    for path in root.rglob('*'):
        rel=path.relative_to(root)
        if any(part in FORBIDDEN_NAMES for part in rel.parts):
            problems.append('FORBIDDEN_PART:'+rel.as_posix())
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append('FORBIDDEN_SUFFIX:'+rel.as_posix())
        if path.is_file() and path.name != 'MANIFEST.json' and rel.as_posix() not in seen:
            problems.append('UNMANIFESTED:'+rel.as_posix())
    report={'status':'PASS' if not problems else 'FAIL','file_count':len(seen),'problems':problems}
    print('PACKAGE_VALIDATION='+json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if report['status']=='PASS' else 2
if __name__=='__main__':
    raise SystemExit(main())
