import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'code'),str(ROOT/'code/src')]
from stagewise_entry import runner
from fast_inference import FastInferencePipeline
from canonical.cz_crossrod_l128_20261008.stagewise_engine import StagewisePipeline
from canonical.cz_crossrod_l128_20261008.data import load_inference
from canonical.cz_crossrod_l128_20261008.invariants import dump
def main():
    reports=[]
    for case in ('Rod_1_to_Rod_2','Rod_2_to_Rod_1'):
        ctx=dict(runner(case)._boot(),mode='inference_only',deployment_checkpoint_verified=True,inference_operation='source_validation')
        state=json.loads((ROOT/'formal_actual'/case/'deployment_state.json').read_text())
        T,_,shared,split=load_inference(ctx,'source_validation')
        # Exercise cold-start and separated entity runs without any test label.
        T=T.groupby('entity_id',sort=False,group_keys=False).head(180).reset_index(drop=True)
        p=FastInferencePipeline(ctx,ROOT/'fast_acceptance'/case,mode='inference_only');p.bind_deployment(state)
        p.accessor=p._make_accessor(shared,split)
        start=time.time();old,om,_=StagewisePipeline.online_a(p,state,T);old_seconds=time.time()-start
        start=time.time();new,nm,_=p.online_a(state,T);new_seconds=time.time()-start
        error=float(np.max(np.abs(old-new),initial=0));bound=8*np.finfo(float).eps*max(1.,float(np.max(np.abs(old),initial=0)))
        assert np.array_equal(om,nm) and error<=bound
        baseline=p.emit(state['parent'],T);assert np.array_equal(new[~nm],baseline[~nm])
        changed=T.copy();changed.loc[changed.index>=len(T)//2,'y_true']=1e9
        alternative,_,_=p.online_a(state,changed)
        assert np.array_equal(new[:len(T)//2],alternative[:len(T)//2])
        reports.append(dict(case=case,rows=len(T),max_abs_error=error,tolerance=bound,mask_equal=True,
            NA_parent_bitwise_equal=True,future_feedback_cannot_change_earlier_predictions=True,
            legacy_seconds=old_seconds,fast_seconds=new_seconds,fit_calls=p.fit_calls))
        print(json.dumps(reports[-1]),flush=True)
    dump(ROOT/'INFERENCE_EXTENSION_ACCEPTANCE.json',dict(status='PASS',reports=reports,target_test_accessed=False))
if __name__=='__main__':main()
