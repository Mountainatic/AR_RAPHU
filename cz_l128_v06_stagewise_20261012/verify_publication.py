"""Standard-library public-archive integrity and rowwise/credential exclusion check."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
FORBIDDEN = {'.xlsx','.xls','.xlsm','.xlsb','.parquet','.arrow','.feather','.npz','.npy',
             '.pt','.pth','.ckpt','.pkl','.pickle','.h5','.hdf5','.zip','.7z','.tar','.gz',
             '.bundle','.pyc','.sqlite','.db','.pem','.key'}
ROW_FIELDS = {'y_true','y_pred','y_target','r_base','r_final','base_residual','final_residual',
              'base_prediction','final_prediction','sample_id','origin','origin_index','raw_index',
              'recorded_level','registered_target'}
SECRET = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}')


def check_json_vectors(value, path):
    if isinstance(value,dict):
        for key,item in value.items():
            assert not (key in ROW_FIELDS and isinstance(item,list) and len(item)>1), (str(path),key)
            check_json_vectors(item,path)
    elif isinstance(value,list):
        for item in value:check_json_vectors(item,path)


def verify():
    manifest=json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    expected={item['path'] for item in manifest['files']}|{'PUBLICATION_MANIFEST.json'}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc'}
    assert expected==actual, f'Unexpected/missing files: {expected^actual}'
    counts={'files':len(actual),'csv':0,'python':0,'json':0,'figures':0}
    for item in manifest['files']:
        rel=Path(item['path']);assert not rel.is_absolute() and '..' not in rel.parts
        path=ROOT/rel;assert not path.is_symlink() and path.suffix.lower() not in FORBIDDEN,path
        payload=path.read_bytes()
        assert len(payload)==item['bytes'] and hashlib.sha256(payload).hexdigest()==item['sha256'],path
        assert not SECRET.search(payload),f'Credential marker: {rel}'
        if path.suffix=='.py':
            ast.parse(payload.decode('utf-8-sig'),filename=str(rel));counts['python']+=1
        elif path.suffix=='.csv':
            rows=list(csv.reader(payload.decode('utf-8-sig').splitlines()))
            assert rows and not ROW_FIELDS.intersection(rows[0]),(str(rel),rows[0] if rows else [])
            assert all(len(row)==len(rows[0]) for row in rows[1:]),rel
            counts['csv']+=1
        elif path.suffix=='.json':
            check_json_vectors(json.loads(payload.decode('utf-8-sig')),rel);counts['json']+=1
        elif path.suffix=='.png':
            assert payload.startswith(b'\x89PNG\r\n\x1a\n'),rel;counts['figures']+=1
        elif path.suffix=='.pdf':
            assert payload.startswith(b'%PDF'),rel;counts['figures']+=1
    print(json.dumps({'status':'PASS',**counts,'checks':['manifest hashes','source syntax','CSV rectangularity',
        'no rowwise data columns/vectors','no private key or token markers','no private binary formats']},indent=2))


if __name__=='__main__':verify()
