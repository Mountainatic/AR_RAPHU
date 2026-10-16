"""Real source-only K reuse, fixed-fold anti-refit and maturity acceptance."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import json,sys,time,copy,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'code'),str(ROOT/'code/src')]
from stagewise_entry import runner
from canonical.cz_crossrod_l128_20261008.stagewise_engine import StagewisePipeline
from canonical.cz_crossrod_l128_20261008.invariants import digest,frame_hash,dump,ProtocolError,prediction_hash
from canonical.cz_crossrod_l128_20261008.fold_geometry import folds,selection_blocks
from canonical.cz_crossrod_l128_20261008.a_state import build_a_training_state_from_cf_residual,feature_rows

def accept(cases=('Rod_1_to_Rod_2','Rod_2_to_Rod_1')):
    started=time.time();reports=[]
    for case in cases:
        context=dict(runner(case)._boot(),resume=False)
        p=StagewisePipeline(context,ROOT/'formal_actual'/case,mode='formal_development');T=p.load_development()
        K=p.learn('K',T)
        key=digest(dict(policy=p.migration['legacy_policy_id'],mode=p.mode,stage='K',role='ADMISSION',frame=frame_hash(T)))
        oldpath=p.legacy/'formal_actual'/case/'checkpoints/ADMISSION_K'/(key+'.json')
        old=json.loads(oldpath.read_text())['value']['state']
        before=p.emit(old,T);after=p.emit(K,T)
        assert np.array_equal(before,after) and old['routing_status']==K['routing_status']
        assert old['training_prediction_hash']==prediction_hash(after)
        # Build one real C model; then forbid every adaptive upstream call.
        C=p.learn('C',T)
        fi,ei,_,_,_=selection_blocks(T,p.protocol['geometry']['inner'],640)[1]
        selected=copy.deepcopy(C);original_learn=p.learn;original_tune=p.tune
        def forbidden(*a,**k):raise AssertionError('Fixed recipe fold refit attempted an adaptive learn/tune')
        p.learn=forbidden;p.tune=forbidden
        empty=p.refit_fixed(selected,T.iloc[:0])
        assert empty['domain'] is False and empty['routing_status']=='BYPASSED'
        local=p.refit_fixed(selected,T.iloc[fi]);ev=T.iloc[ei].drop(columns=['y_true'])
        pred=p.emit(local,ev)
        contaminated=ev.copy();contaminated['y_true']=np.linspace(-1e9,1e9,len(ev))
        assert np.array_equal(pred,p.emit(local,contaminated))
        source,proof,path=p.fixed_stream('C',T,selected)
        assert len(proof)==4 and len(source)<=3840 and all(not x['target_used_to_fit_own_parameters'] for x in proof)
        assert source[['entity_id','origin']].duplicated().sum()==0
        # Mature residual features must ignore an arbitrary future CF value.
        typed=build_a_training_state_from_cf_residual(source,policy_id=p.policy_id,h_steps=1)
        sample=source.loc[source.CF_fold>=0].iloc[:200]
        x,mask,_=feature_rows(sample,typed,p.A_profiles[0],p.A_max,0.)
        altered=source.copy();latest=int(sample.latest_available_target_index.max())
        altered.loc[altered.target_stop_exclusive-1>latest,'residual']=1e12
        future=build_a_training_state_from_cf_residual(altered,policy_id=p.policy_id,h_steps=1)
        z,mask2,_=feature_rows(sample,future,p.A_profiles[0],p.A_max,0.)
        assert np.array_equal(mask,mask2) and np.array_equal(x,z)
        p.learn=original_learn;p.tune=original_tune
        # Exercise the complete real W/A path, including sparse inner folds.
        # This is the authorized source-development run, kept for exact resume.
        final=p.learn('A',T);p._verify_chain(final)
        assert final['stage']=='A' and p.development_complete(final)
        if final.get('domain'):
            # Cached A inputs must preserve the original numerical function.
            import pandas as pd
            from canonical.cz_crossrod_l128_20261008.canonical_engine import CanonicalPipeline
            cf=pd.read_parquet(final['source_path']);ev_cf=cf.loc[cf.CF_fold==1].copy()
            fit_cf=cf.loc[cf._position<int(ev_cf._position.min())].copy()
            for spec in [p.A_specs[0],p.A_specs[len(p.A_specs)//2],p.A_specs[-1]]:
                ct,pc,mc,ac=p.fit_residual_recipe('A',spec,fit_cf,ev_cf,cf,comparison=True)
                rt,rp,rm,ra=CanonicalPipeline.fit_residual_recipe(p,'A',spec,fit_cf,ev_cf,cf,comparison=True)
                assert np.array_equal(pc,rp) and np.array_equal(mc,rm) and digest(ct)==digest(rt)
                assert digest(ac)==digest(ra)
            fi,ei=folds(T,p.protocol['geometry']['outer'])[0]
            child,mask,_=p.online_a(final,T.iloc[ei]);parent=p.emit(final['parent'],T.iloc[ei])
            assert np.array_equal(child[~mask],parent[~mask])
        reports.append(dict(case=case,K_route=K['routing_status'],K_max_abs_prediction_difference=float(np.max(np.abs(before-after))),
            K_predictions_bitwise_equal=True,K_exact_imports=p.import_count,C_route=C['routing_status'],
            finite_CF_rows=len(source),no_adaptive_calls_in_fixed_refits=True,heldout_target_values_not_predictor_inputs=True,
            future_residual_does_not_change_mature_features=True,empty_fold_identity_passed=True,
            full_source_K_C_W_A_completed=True,fit_calls=p.fit_calls))
        print(json.dumps(reports[-1]),flush=True)
    destination=ROOT/'STAGEWISE_ACCEPTANCE.json' if len(cases)==2 else ROOT/('ACCEPTANCE_'+cases[0]+'.json')
    dump(destination,dict(status='PASS',seconds=time.time()-started,reports=reports,
        target_test_accessed=False,full_C_W_A_equivalence_claim=False,
        scientific_difference='FINITE_PARAMETER_CF_RECIPES_FROZEN_WITHIN_OUTER_FIT'))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case-id');a=parser.parse_args()
    accept((a.case_id,) if a.case_id else ('Rod_1_to_Rod_2','Rod_2_to_Rod_1'))
