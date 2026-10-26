"""Same CPU/thread deployment benchmark; no training and no raw/tail access.

Times include finite-window preparation and scaler application. PRISM includes
the newly matured label, feature construction, RLS updates and certificates.
Neural timings cover one seed and the actual three-seed ensemble separately.
Model resident tensor bytes exclude interpreter/library overhead; compact
serialized deployment bytes include preprocessing constants, not train ledgers.
"""
from pathlib import Path
import os,sys,json,pickle,time,io,platform,hashlib,statistics
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['AR_RAPHU_RUNTIME_MANAGER']='uv'
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp');SRC=BASE/'PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025'
sys.path[:0]=[str(ROOT),str(SRC)]
from streaming import StreamingPRISM,compact_model
from baseline_models import build
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.mha.set_fastpath_enabled(False)
torch.use_deterministic_algorithms(True)
if hasattr(os,'sched_getaffinity'):
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
WINDOWS={4:8,8:16,16:16}

def summary(samples):
    a=np.array(samples)*1e6
    return dict(median_us=float(np.median(a)),p95_us=float(np.quantile(a,.95)),mean_us=float(a.mean()),samples=len(a))

def sequence(a,t,L,d):
    ix=np.arange(t-L+1,t+1)
    return np.c_[a[ix,1:],a[ix-d,0]].astype(np.float32)

def infer_one(model,obj,raw,anchor):
    sc=obj['scaler'];x=(raw-sc['xmean'])/sc['xstd']
    z=model(torch.from_numpy(x[None]))[0].item()
    return z*sc['ystd']+sc['ymean']+(anchor if obj['config']['target']=='delta' else 0.)

def numeric_bytes(x,seen=None):
    seen=set() if seen is None else seen
    if id(x) in seen:return 0
    seen.add(id(x))
    if isinstance(x,np.ndarray):return x.nbytes
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    if isinstance(x,(float,int,np.number)):return 8
    if isinstance(x,dict):return sum(numeric_bytes(v,seen) for v in x.values())
    if isinstance(x,(list,tuple)):return sum(numeric_bytes(v,seen) for v in x)
    if hasattr(x,'__dict__'):return numeric_bytes(x.__dict__,seen)
    return 0

records=[];checks=[]
a=np.vstack([np.load(SRC/'data/train.npz')['values'],np.load(SRC/'data/development.npz')['values']]);assert a.shape==(8064,6)
verification=json.loads((ROOT/'STREAMING_VERIFICATION.json').read_text())
for d in (4,8,16):
    assert any(x['entity']=='SRU_H2S' and x['delay']==d and x['original_tolerance_passed'] for x in verification['records'])
    L=WINDOWS[d];obj=pickle.loads((SRC/f'prism_reference/D{d}/full.pkl').read_bytes());compact=compact_model(obj)
    all_times=[];pred=None;state=None
    for repeat in range(4):
        state=StreamingPRISM(obj,L,d,a[6048-L+1:6048,1:],a[6048-d-L+1:6048-d,0],6048);times=[];out=[]
        for t in range(6048,8064):
            begin=time.perf_counter();p=state.step(a[t,1:],a[t-d,0]);elapsed=time.perf_counter()-begin
            if 6112<=t<6368:times.append(elapsed)
            out.append(p)
        if repeat>0:all_times.extend(times)
        pred=np.array(out)
    stored=pd.read_parquet(SRC/f'data/prism_D{d}.parquet').PRISM.to_numpy()
    np.testing.assert_allclose(pred,stored,rtol=1e-12,atol=1e-12)
    records.append(dict(delay=d,L=L,model='PRISM_stream',ensemble=1,regression_coefficients=36,
        compact_deployment_bytes=len(pickle.dumps(compact,protocol=5)),model_numeric_bytes=numeric_bytes(compact),
        includes_RLS_update=True,bounded_history=True,**summary(all_times)))
    target=np.arange(6112,6368);wide=pd.read_csv(SRC/f'results/point_predictions_D{d}.csv')
    for name in ('LSTM','Transformer','iTransformer','TimeXer'):
        models=[];objects=[];weights_bytes=0;parameters=0;packed_bytes=0
        for seed in (0,1,2):
            path=SRC/f'runs/D{d}/{name}/seed_{seed}.pt';ob=torch.load(path,map_location='cpu',weights_only=False)
            net=build(name,6,ob['config']['d'],L).cpu().eval();net.load_state_dict(ob['state']);models.append(net);objects.append(ob)
            parameters+=sum(p.numel() for p in net.parameters());weights_bytes+=sum(v.numel()*v.element_size() for v in net.state_dict().values())
            buff=io.BytesIO();torch.save({k:ob[k] for k in ('state','scaler','config','delay','name')},buff);packed_bytes+=len(buff.getvalue())
        with torch.inference_mode():
            # Numerical audit against archived GPU-derived ensemble on the same points.
            predictions=[]
            for t in target:
                raw=sequence(a,t,L,d);predictions.append(np.mean([infer_one(n,o,raw,a[t-d,0]) for n,o in zip(models,objects)]))
            expected=wide.set_index('origin').loc[target,name].to_numpy()
            np.testing.assert_allclose(predictions,expected,rtol=3e-5,atol=2e-6)
            for ensemble in (1,3):
                ts=[]
                for repeat in range(4):
                    for t in target:
                        before=time.perf_counter();raw=sequence(a,t,L,d)
                        p=np.mean([infer_one(models[i],objects[i],raw,a[t-d,0]) for i in range(ensemble)])
                        if repeat:ts.append(time.perf_counter()-before)
                records.append(dict(delay=d,L=L,model=name,ensemble=ensemble,parameters=parameters//3*ensemble,
                    compact_deployment_bytes=packed_bytes//3*ensemble,model_tensor_bytes=weights_bytes//3*ensemble,
                    includes_RLS_update=False,**summary(ts)))
            checks.append(dict(delay=d,model=name,n=len(target),max_absolute_error=float(np.max(np.abs(np.array(predictions)-expected))),rtol=3e-5,atol=2e-6))
        del models,objects
cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unavailable')
result=dict(status='COMPLETED',device='CPU',cpu_threads=1,cpu_model=cpu_model,cpu_affinity=sorted(os.sched_getaffinity(0)),platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
    method='Four chronological passes, first warmup excluded; identical timed origins6112..6367 for all models, after64-startup. PRISM correctness checked on all2016 development targets. Three measured passes. Median/p95 observed wall latency, no FLOPs inference. No GPU work.',
    includes='Input indexing/scaling, model computation; PRISM includes all matured-label online updates and periodic positive-definite checks. Process imports/model load excluded for every method.',
    excludes='Full process peak RSS, training/selection time and analytical FLOPs; tensor bytes and serialized deployment bytes reported separately.',
    records=records,prediction_checks=checks,reserved_tail_read=False)
(ROOT/'CPU_COMPLEXITY.json').write_text(json.dumps(result,indent=2));pd.DataFrame(records).to_csv(ROOT/'CPU_COMPLEXITY.csv',index=False)
print(json.dumps(dict(status='COMPLETED',records=len(records),checks=len(checks)),indent=2),flush=True)
