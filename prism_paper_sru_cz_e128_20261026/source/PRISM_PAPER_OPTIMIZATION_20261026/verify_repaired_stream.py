from pathlib import Path
import os,sys,pickle,json
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;FIX=Path('/root/autodl-tmp/PRISM_PAPER_SPLINE_REPAIR_20261026')
sys.path[:0]=[str(ROOT),str(FIX/'robust')]
from streaming import run_stream
from suite import geometry,dump
records=[]
for e in ('SRU_H2S','CZ_Rod1','CZ_Rod2'):
    g=geometry(e);a=np.vstack([np.load(Path(g['root'])/'data/train.npz')['values'],np.load(Path(g['root'])/'data/development.npz')['values']])
    for d in (4,8,16):
        p=FIX/'runs/repaired'/e/f'D{d}/full/model.pkl';obj=pickle.loads(p.read_bytes())
        pred,state=run_stream(obj,g['windows'][d],d,a,g['train_end'],g['dev_end'])
        saved=pd.read_parquet(FIX/'results/repaired'/e/f'D{d}_predictions.parquet').full.to_numpy()
        okay=bool(np.allclose(pred,saved,rtol=1e-12,atol=1e-12))
        records.append(dict(entity=e,delay=d,passed=okay,max_absolute_error=float(np.max(np.abs(pred-saved)))))
assert all(x['passed'] for x in records if x['entity']=='SRU_H2S')
dump(ROOT/'REPAIRED_STREAM_VERIFICATION.json',dict(status='COMPLETED' if all(x['passed'] for x in records) else 'PARTIAL_NUMERICAL_REJECTION',records=records,rtol=1e-12,atol=1e-12,reserved_tail_read=False))
print(json.dumps(records,indent=2))
