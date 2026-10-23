"""Only existing train/development parquet, current X authorized by user."""
from pathlib import Path
import json,hashlib,glob
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
SRU=BASE/'PRISM_V06_SRU_CZ_SCALE_SEARCH_20261020'
DEBUT=BASE/'PRISM_V06_DEBUTANIZER_KCWA_SEARCH_20261022'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def source_records():
    s=read(SRU/'DATASETS.json')['datasets'];b=read(DEBUT/'DATASETS.json')['datasets']
    out={}
    for name in ['SRU_H2S','SRU_SO2','debutanizer']:
        d=next(x for x in s+b if x['id']==name)
        out[name]=dict(name=name,target=d.get('target','y'),inputs=d.get('columns',d.get('inputs')),train=d['train_end'],dev=d['validation_end'],total=d.get('total_rows',d.get('total')),upper=3328 if name.startswith('SRU') else 850,shared=d['shared'],dataset=d['dataset'],historical_exposure=name=='debutanizer',source_metadata=d)
    return out

def records():
    out={}
    for name,v in source_records().items():
        if name!='SRU_H2S':continue
        for delay in [4,8,16]:
            key=name+f'_D{delay}';out[key]=dict(v,name=key,source_case=name,delay=delay,upper=3328 if name.startswith('SRU') else 650)
    return out

def prefix(case,development=False):
    r=records()[case];p=Path(r['shared'])/'base_data'/r['dataset'];files=[p/'train.parquet']+([p/'validation.parquet'] if development else [])
    a=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    n=r['dev'] if development else r['train'];assert len(a)==n
    assert np.array_equal(a.row_in_entity.to_numpy(),np.arange(n))
    x=a[[r['target']]+r['inputs']].to_numpy(float);assert np.isfinite(x).all()
    return {case:x}

def prepare():
    assert not (ROOT/'DATA_VIEW_BEFORE_FITS.json').exists();r=records();ledger_refs=[]
    # Audit all existing project final ledgers; do not overwrite/reset any.
    for p in sorted(glob.glob(str(BASE/'*/*EXPOS*.json'))):
        obj=read(p);text=json.dumps(obj)
        if any(s in text.lower() for s in ['sru','debutanizer']):ledger_refs.append(dict(path=p,sha256=sha(p),content=obj))
    inputs={}
    for case,v in r.items():
        p=Path(v['shared'])/'base_data'/v['dataset']
        v['files']={str(p/(n+'.parquet')):sha(p/(n+'.parquet')) for n in ['train','validation']}
        inputs.update(prefix(case))
    np.savez_compressed(ROOT/'TRAINING_DATA.npz',**inputs)
    (ROOT/'DATA_VIEW_BEFORE_FITS.json').write_text(json.dumps(dict(records=r,existing_final_ledger_references=ledger_refs,final_numeric_read=False,SRU_final_source_status='Original branch records final20% not evaluated; shared historical project-ledger audit retained, no claim of a newly acquired source.',debutanizer_final_status='Historically exposed; any results retrospective, never fresh test.'),indent=2))
    print('SOFTSENSOR_TRAINING_ONLY_PREPARED',list(r),len(ledger_refs),flush=True)
if __name__=='__main__':prepare()
