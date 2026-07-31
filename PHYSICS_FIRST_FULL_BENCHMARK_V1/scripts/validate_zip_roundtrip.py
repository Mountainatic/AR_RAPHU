#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, tempfile, zipfile, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gpu_common import sha256_file

def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip',required=True)
    args=parser.parse_args()
    archive=Path(args.zip).resolve()
    problems=[]
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(archive) as bundle:
            bad=bundle.testzip()
            if bad: problems.append('CRC:'+bad)
            bundle.extractall(temp)
        roots=[p for p in Path(temp).iterdir() if p.is_dir()]
        if len(roots)!=1:
            problems.append('ROOT_COUNT:'+str(len(roots)))
        else:
            root=roots[0]
            manifest_path=root/'MANIFEST.json'
            if not manifest_path.is_file():
                problems.append('MANIFEST_MISSING')
            else:
                manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
                for record in manifest.get('files',[]):
                    path=root/record['path']
                    if not path.is_file(): problems.append('MISSING:'+record['path'])
                    elif sha256_file(path)!=record['sha256']: problems.append('HASH:'+record['path'])
    report={'status':'PASS' if not problems else 'FAIL','zip_sha256':sha256_file(archive),'problems':problems}
    print('ZIP_ROUNDTRIP='+json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if report['status']=='PASS' else 2
if __name__=='__main__':
    raise SystemExit(main())
