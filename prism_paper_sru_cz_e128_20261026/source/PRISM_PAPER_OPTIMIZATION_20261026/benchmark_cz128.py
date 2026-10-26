"""Long-history deployment complexity versus the strongest existing neural comparator."""
from pathlib import Path
import os,sys,json,pickle,time,io
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp');EXP=BASE/'PRISM_CZ_E128_20261026'
sys.path[:0]=[str(ROOT),str(EXP/'E128'),str(BASE/'PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025')]
from streaming import StreamingPRISM,compact_model
from baseline_models import build
from suite import geometry,dump
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
assert (EXP/'QUEUE_COMPLETED.json').exists()
records=[];checks=[]
bins=list(zip([0,1,2,4,8,16,32,64],[1,2,4,8,16,32,64,128]))
def stats(times):
    a=np.asarray(times)*1e6
    return dict(median_us=float(np.median(a)),p95_us=float(np.quantile(a,.95)),samples=len(a))
for entity in ('CZ_Rod1','CZ_Rod2'):
    g=geometry(entity);src=Path(g['root']);a=np.vstack([np.load(src/'data/train.npz')['values'],np.load(src/'data/development.npz')['values']])
    start=g['train_end'];timed=np.arange(start+64,start+320)
    for d in (4,8,16):
        for mode in ('fixed128','E128'):
            obj=pickle.loads((EXP/f'runs/{mode}/{entity}/D{d}/full/model.pkl').read_bytes());all_times=[]
            for repeat in range(4):
                state=StreamingPRISM(obj,128,d,a[start-127:start,1:],a[start-d-127:start-d,0],start,bins if mode=='E128' else None)
                pred=[]
                for t in range(start,g['dev_end']):
                    before=time.perf_counter();p=state.step(a[t,1:],a[t-d,0]);elapsed=time.perf_counter()-before;pred.append(p)
                    if repeat and t in timed:all_times.append(elapsed)
            saved=pd.read_parquet(EXP/f'results/{mode}/{entity}/D{d}_predictions.parquet').full.to_numpy();err=float(np.max(np.abs(pred-saved)))
            passed=bool(np.allclose(pred,saved,rtol=1e-12,atol=1e-12));checks.append(dict(entity=entity,delay=d,mode=mode,passed=passed,max_absolute_error=err))
            parent=obj['upper']['parent'];w=obj['upper']['W'];A=obj['A'];count=parent['ridge']['parameter_count']+(w['ridge']['parameter_count'] if w else 0)+(len(A['model']['theta']) if A and A['admitted'] else 0)
            records.append(dict(entity=entity,delay=d,model=mode,history=128,ensemble=1,regression_coefficients=count,
                compact_deployment_bytes=len(pickle.dumps(dict(model=compact_model(obj),L=128,delay=d,temporal_bins=bins if mode=='E128' else None),protocol=5)),
                warm_runtime_checkpoint_bytes=len(pickle.dumps(state,protocol=5)),includes_online_update=True,numerical_equivalence_passed=passed,**stats(all_times)))
        nets=[];objects=[];packed=0;params=0
        for seed in (0,1,2):
            ob=torch.load(src/f'runs/D{d}/Transformer/seed_{seed}.pt',map_location='cpu',weights_only=False)
            net=build('Transformer',6,ob['config']['d'],128).cpu().eval();net.load_state_dict(ob['state']);nets.append(net);objects.append(ob)
            params+=sum(v.numel() for v in net.parameters());buf=io.BytesIO();torch.save({k:ob[k] for k in ('state','scaler','config','delay','name')},buf);packed+=len(buf.getvalue())
        def infer(n,ob,t):
            ix=np.arange(t-127,t+1);x=np.c_[a[ix,1:],a[ix-d,0]].astype(np.float32);sc=ob['scaler'];x=(x-sc['xmean'])/sc['xstd']
            z=n(torch.from_numpy(x[None]))[0].item();return z*sc['ystd']+sc['ymean']+(a[t-d,0] if ob['config']['target']=='delta' else 0.)
        with torch.inference_mode():
            expected=pd.read_csv(src/f'results/point_predictions_D{d}.csv').set_index('origin').loc[timed,'Transformer'].to_numpy()
            actual=np.array([np.mean([infer(n,ob,t) for n,ob in zip(nets,objects)]) for t in timed]);np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=2e-6)
            checks.append(dict(entity=entity,delay=d,model='Transformer',passed=True,max_absolute_error=float(np.max(np.abs(actual-expected))),rtol=3e-5,atol=2e-6))
            for ensemble in (1,3):
                times=[]
                for rep in range(4):
                    for t in timed:
                        before=time.perf_counter();p=np.mean([infer(nets[i],objects[i],t) for i in range(ensemble)])
                        if rep:times.append(time.perf_counter()-before)
                records.append(dict(entity=entity,delay=d,model='Transformer',history=128,ensemble=ensemble,parameters=params//3*ensemble,
                    compact_deployment_bytes=packed//3*ensemble,includes_online_update=False,numerical_equivalence_passed=True,**stats(times)))
        del nets,objects
dump(ROOT/'CZ128_CPU_COMPLEXITY.json',dict(status='COMPLETED',device='CPU',cpu_threads=1,numpy=np.__version__,torch=torch.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),
    records=records,checks=checks,all_inputs_history=128,method='Identical256 timed origins per entity/delay, three measured chronological passes after warmup. Includes input encoding/scaling and complete PRISM online updates. Compact/warm checkpoints are serialized object sizes, not peak process RAM.',reserved_tail_read=False))
pd.DataFrame(records).to_csv(ROOT/'CZ128_CPU_COMPLEXITY.csv',index=False)
print('CZ128_CPU_COMPLEXITY_COMPLETED',len(records),flush=True)
