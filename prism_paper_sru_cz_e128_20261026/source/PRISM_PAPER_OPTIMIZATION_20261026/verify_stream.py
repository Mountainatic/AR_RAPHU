"""Verify deployment predictions against the original batch scientific path."""
from pathlib import Path
import os,sys,pickle,json,time
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp')
sys.path.insert(0,str(ROOT/'standard'))
import model as m
from streaming import run_stream
from suite import geometry,reference,dump
records=[]
for entity in ('SRU_H2S','CZ_Rod1','CZ_Rod2'):
    g=geometry(entity);a=np.vstack([np.load(Path(g['root'])/'data/train.npz')['values'],np.load(Path(g['root'])/'data/development.npz')['values']])
    D={entity:a}
    for d in (4,8,16):
        L=g['windows'][d];m.L=L;m.configure(d)
        obj=pickle.loads(reference(g,entity,d).read_bytes());V=m.frame(D,g['train_end'],g['dev_end']);q=m.predict_kcw(D,V,obj['upper']);f=m.source(V,q,'stream_verification')
        pred=q+(np.zeros(len(V)) if obj['A'] is None else m.a_predict(f,obj['A'])[0])
        start=time.perf_counter();stream,state=run_stream(obj,L,d,a,g['train_end'],g['dev_end']);seconds=time.perf_counter()-start
        error=float(np.max(np.abs(stream-pred)))
        equivalent=bool(np.allclose(stream,pred,rtol=1e-12,atol=1e-12))
        # Independent replay prefixes prove state resets and no hidden full-array dependency.
        for n in (1,d,d+1,33,65,257):
            short,_=run_stream(obj,L,d,a[:g['train_end']+n],g['train_end'],g['train_end']+n)
            np.testing.assert_allclose(short,stream[:n],rtol=1e-12,atol=1e-12)
        assert len(state.q)<=max(32,d+8)+4 and len(state.residual)<=max(32,d+8)+4 and len(state.pending)<=d
        records.append(dict(entity=entity,delay=d,L=L,n=len(V),max_absolute_error=error,original_tolerance_passed=equivalent,seconds=seconds,
                            q_buffer=len(state.q),residual_buffer=len(state.residual),pending=len(state.pending),prefix_checks=6))
dump(ROOT/'STREAMING_VERIFICATION.json',dict(status='COMPLETED' if all(x['original_tolerance_passed'] for x in records) else 'PARTIAL_NUMERICAL_REJECTION',records=records,rtol=1e-12,atol=1e-12,
    no_fitting=True,reserved_tail_read=False,bounded_state=True))
print(json.dumps(records,indent=2),flush=True)
