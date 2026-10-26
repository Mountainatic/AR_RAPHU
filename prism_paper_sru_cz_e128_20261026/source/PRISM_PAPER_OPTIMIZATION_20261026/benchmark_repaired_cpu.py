from pathlib import Path
import os,sys,json,pickle,time
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;FIX=Path('/root/autodl-tmp/PRISM_PAPER_SPLINE_REPAIR_20261026');SRC=Path('/root/autodl-tmp/PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025')
sys.path[:0]=[str(ROOT),str(FIX/'robust')]
from streaming import StreamingPRISM,compact_model
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
assert all(x['passed'] for x in json.loads((ROOT/'REPAIRED_STREAM_VERIFICATION.json').read_text())['records'] if x['entity']=='SRU_H2S')
a=np.vstack([np.load(SRC/'data/train.npz')['values'],np.load(SRC/'data/development.npz')['values']]);rows=[]
for d,L in ((4,8),(8,16),(16,16)):
    obj=pickle.loads((FIX/f'runs/repaired/SRU_H2S/D{d}/full/model.pkl').read_bytes());times=[]
    for repeat in range(4):
        state=StreamingPRISM(obj,L,d,a[6048-L+1:6048,1:],a[6048-d-L+1:6048-d,0],6048);pred=[]
        for t in range(6048,8064):
            start=time.perf_counter();q=state.step(a[t,1:],a[t-d,0]);elapsed=time.perf_counter()-start;pred.append(q)
            if repeat and 6112<=t<6368:times.append(elapsed*1e6)
    saved=pd.read_parquet(FIX/f'results/repaired/SRU_H2S/D{d}_predictions.parquet').full.to_numpy()
    np.testing.assert_allclose(pred,saved,rtol=1e-12,atol=1e-12)
    p=obj['upper']['parent']['ridge']['parameter_count'];w=obj['upper']['W'];wc=w['ridge']['parameter_count'] if w else 0
    ac=len(obj['A']['model']['theta']) if obj['A'] and obj['A']['admitted'] else 0
    rows.append(dict(model='PRISM_spline_repaired',delay=d,L=L,ensemble=1,regression_coefficients=p+wc+ac,
        compact_deployment_bytes=len(pickle.dumps(compact_model(obj),protocol=5)),median_us=float(np.median(times)),p95_us=float(np.quantile(times,.95)),samples=len(times),includes_RLS_update=True))
(ROOT/'REPAIRED_CPU_COMPLEXITY.json').write_text(json.dumps(dict(status='COMPLETED',device='CPU',cpu_threads=1,cpu_affinity=sorted(os.sched_getaffinity(0)),numpy=np.__version__,python=sys.version,
    method='Identical timed origins6112..6367, four chronological passes with first warmup excluded. Includes mature-label updates. Existing neural runtime; single-thread FP64 PRISM versus FP32 neural models on the same CPU.',records=rows,reserved_tail_read=False),indent=2))
pd.DataFrame(rows).to_csv(ROOT/'REPAIRED_CPU_COMPLEXITY.csv',index=False)
print(json.dumps(rows,indent=2))
