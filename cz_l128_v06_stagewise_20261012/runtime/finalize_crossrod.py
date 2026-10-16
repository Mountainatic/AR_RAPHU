"""After both-direction freeze: materialize target, replay, export, aggregate.

This controller does not fit/select a model. It cannot run until every
registered job has a sealed deployment or an explicit inherited NA status.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path('/root/autodl-tmp/CZ_CROSSROD_L128_STAGEWISE_V3_20261012')
sys.path[:0]=[str(ROOT),str(ROOT/'code'),str(ROOT/'code/src')]
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
from baseline_adapter.run_cpu import TASK,view,dump,verify_release
PIN=json.loads((ROOT/'release/RELEASE_PIN.json').read_text())['release_sha256']
CASES=['Rod_1_to_Rod_2','Rod_2_to_Rod_1']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(j):return j['direction'],j['information_set'],j['model']

def validate_job_coverage(jobs,records):
    if len(records)!=len(jobs) or len({key(j) for j in records})!=len(records):raise RuntimeError('Incomplete/duplicate baseline jobs')
    if {key(j) for j in records}!={key(j) for j in jobs}:raise RuntimeError('Unexpected baseline job universe')
    for r in records:
        if r['status']=='FROZEN':continue
        if r['model'] in {'SEASONAL_PERSISTENCE','N4SID'} and r['status']=='NOT_RUN_PROTOCOL_INCOMPATIBLE':continue
        raise RuntimeError('Baseline deployment is incomplete: '+str(key(r)))

def ready():
    paths=[ROOT/'formal_actual'/c/'DEPLOYMENT_SEAL.json' for c in CASES]
    paths += [ROOT/'baselines'/c/'cpu_verified/CPU_FREEZE_MANIFEST.json' for c in CASES]
    paths += [ROOT/'baselines/neural/NEURAL_FREEZE_MANIFEST.json']
    if not all(p.is_file() for p in paths):return None
    manifests=[json.loads(p.read_text()) for p in paths]
    cpu=[r for m in manifests[2:4] for r in m['records']]
    neural=manifests[4]['records']
    jobs=json.loads((ROOT/'prepared/BASELINE_JOBS.json').read_text())
    validate_job_coverage(jobs,cpu+neural)
    for r in neural:
        if sorted(s['seed'] for s in r['seeds'])!=list(range(20260817,20260827)):raise RuntimeError('Incomplete ten-seed neural final fit')
    artifacts={str(p):sha(p) for p in paths}
    for m in manifests[:2]:
        if m['status']!='DEPLOYMENT_FROZEN' or m['release_sha256']!=PIN:raise RuntimeError('Main deployment/release mismatch')
        artifacts.update(m['artifact_sha256'])
    for r in cpu:
        if r['status']=='FROZEN':
            directory=Path(r['frozen']['checkpoint_dir'])
            artifacts.update({str(p):sha(p) for p in directory.rglob('*') if p.is_file()})
    for r in neural:
        path=Path(r['selection_path']);artifacts[str(path)]=sha(path)
        for s in r['seeds']:
            if sha(s['checkpoint'])!=s['sha256']:raise RuntimeError('Neural checkpoint changed')
            artifacts[s['checkpoint']]=s['sha256']
    verify_release()
    imported=json.loads((ROOT/'BASELINE_IMPORT_RECEIPT.json').read_text())
    if imported['new_main_release_sha256']!=PIN:raise RuntimeError('Baseline import/new release mismatch')
    artifacts.update(imported['artifact_sha256'])
    for p,expected in artifacts.items():
        if sha(p)!=expected:raise RuntimeError('Deployment artifact changed: '+p)
    freeze=dict(status='ALL_MAIN_AND_BASELINES_FROZEN',task_id=TASK,main_release_sha256=PIN,
        baseline_release_sha256=sha(ROOT/'BASELINE_RELEASE.json'),baseline_import_receipt_sha256=sha(ROOT/'BASELINE_IMPORT_RECEIPT.json'),test_used_for_selection=False,
        artifact_sha256=artifacts,baseline_jobs=len(jobs),retained_nonapplicable=[key(r) for r in cpu if r['status']!='FROZEN'])
    return freeze,cpu,neural

def materialize_target(freeze):
    from prism_benchmark import cz_l256_nowcast as d
    raw=Path('/root/autodl-tmp/PRISM_CZ_PRIVATE_RAW_20260915/cz_raw.xlsx')
    if sha(raw)!='c3428966fe006572809156ee5e3f488264b8206b19b20887dcd00840bb26fbc3':raise RuntimeError('Raw source changed')
    global_path=ROOT/'GLOBAL_CROSSROD_FREEZE.json'
    with global_path.open('x') as f:json.dump(freeze,f,indent=2)
    for case in CASES:
        shared=ROOT/'C1'/case
        target='Rod_2' if case=='Rod_1_to_Rod_2' else 'Rod_1'
        pca=json.loads((shared/'JOINT_LIFT_PCA_CONTRACT.json').read_text())
        segments=d._raw_segments(target,d._read_rod(raw,target))
        base=d._base_frame(d._apply_joint_lift(segments,pca))
        destination=shared/'base_data/cz_czochralski/test.parquet'
        if destination.exists():raise RuntimeError('Target base must be a new artifact')
        base.to_parquet(destination,index=False,compression='zstd')
        dump(shared/'TARGET_DATA_SEAL.json',dict(global_freeze_path=str(global_path),global_freeze_sha256=sha(global_path),
            target_base_sha256=sha(destination),raw_sha256=sha(raw),target_PCA_fit=False))

def main_runner(case):
    from canonical.cz_crossrod_l128_20261008.execution import CanonicalExecution
    from canonical.cz_crossrod_l128_20261008.stagewise_engine import StagewisePipeline
    return CanonicalExecution(module=ROOT/'code/canonical/cz_crossrod_l128_20261008',authority=ROOT/'code',
        registration_path=ROOT/'release/REGISTRATION.json',release_seal_path=ROOT/'release/RELEASE_SEAL.json',
        release_sha=PIN,shared=ROOT/'C1'/case,case_id=case,pipeline_factory=StagewisePipeline)

def samples_for(case,information,split):
    import pandas as pd
    from canonical.cz_crossrod_l128_20261008.data import _labels
    shared=ROOT/'C1'/case
    samples=pd.read_parquet(shared/'sample_ids'/TASK/information/'record_time/primary'/(split+'.parquet'))
    paths=[shared/'base_data/cz_czochralski/train.parquet']
    if split=='test':paths.append(shared/'base_data/cz_czochralski/test.parquet')
    return _labels(samples,paths)

def level_export(shared,v,samples,prediction,path):
    from baseline_adapter.accessor import FoldPCAAccessor
    from prism_benchmark.level_reconstruction import metric_bundle_delta_and_level
    split='test' if set(samples.split)=={'test'} else 'validation'
    accessor=FoldPCAAccessor(shared,v.head.dataset,split,[v.head.target])
    levels=accessor.block_means(samples,v.head.target,[(0,1)]).reshape(-1)
    frame=samples.copy();frame['y_pred']=prediction;frame['current_level']=levels
    frame['future_level_true']=levels+frame.y_true;frame['future_level_pred']=levels+prediction
    frame['residual']=frame.y_true-prediction
    path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False,compression='zstd')
    metrics=metric_bundle_delta_and_level(frame.y_true.to_numpy(),prediction,levels)
    for k in ['future_level_true','future_level_pred']:metrics.pop(k,None)
    return dict(prediction_path=str(path),prediction_sha256=sha(path),rows=len(frame),**metrics)

def finalize(frozen):
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from baseline_adapter import representative_baseline_checkpoints as cp
    from prism_benchmark.portable_checkpoints import load_portable_checkpoint,activate_inference_fit_guard
    freeze,cpu,neural=frozen
    materialize_target(freeze)
    summary=[]
    for case in CASES:
        runner=main_runner(case);deployment=ROOT/'formal_actual'/case/'DEPLOYMENT_SEAL.json'
        validation=ROOT/'inference'/case/'validation'
        _,receipt_pin=runner.run_inference('source_validation',output_root=validation,
            deployment_path=deployment,deployment_pin=sha(deployment))
        runner.run_inference('target_test',output_root=ROOT/'inference'/case/'test',deployment_path=deployment,
            deployment_pin=sha(deployment),validation_receipt_path=validation/'INFERENCE_RECEIPT.json',validation_receipt_pin=receipt_pin)
        for split in ['validation','test']:
            for stage in ['K','C','W','A']:
                path=ROOT/'inference'/case/split/(stage+'_pointwise.parquet');frame=pd.read_parquet(path)
                record=level_export(ROOT/'C1'/case,view('dynamic'),frame,frame.deployed_prediction.to_numpy(),
                    ROOT/'pointwise_enriched'/case/split/('PRISM_'+stage+'.parquet'))
                summary.append(dict(direction=case,information_set='dynamic',model='PRISM_'+stage,split=split,seed=-1,**record))
    os.environ['PRISM_FORMAL_INFERENCE_ONLY']='1';activate_inference_fit_guard()
    for r in cpu:
        if r['status']!='FROZEN':continue
        case=r['direction'];v=view(r['information_set']);shared=ROOT/'C1'/case
        checkpoint=Path(r['frozen']['checkpoint_dir']);state,arrays,_=load_portable_checkpoint(checkpoint)
        paths=SimpleNamespace(shared=shared,project=ROOT/'code',run_root=ROOT/'inference')
        samples=samples_for(case,v.information_set,'test')
        x=cp._evaluation_features(paths,v,samples,'test',state,arrays)
        prediction=cp._predict_loaded(checkpoint,x,state,arrays)
        record=level_export(shared,v,samples,prediction,ROOT/'pointwise_enriched'/case/'test'/v.information_set/(r['model']+'.parquet'))
        summary.append(dict(direction=case,information_set=v.information_set,model=r['model'],split='test',seed=-1,**record))
    import torch
    from prism_benchmark import neural3 as n
    from baseline_adapter.accessor import FoldPCAAccessor
    torch.set_num_threads(1);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for r in neural:
        case=r['direction'];v=view(r['information_set']);shared=ROOT/'C1'/case
        contract=json.loads(Path(r['selection_path']).read_text());sel=contract['selected'];dynamic=v.information_set=='dynamic'
        candidate=n.Candidate(r['model'],0,128,sel['capacity'],sel['learning_rate'],'L128')
        sc=contract['scaler'];scaler=n.Scaler(np.array(sc['feature_mean']),np.array(sc['feature_scale']),sc['target_mean'],sc['target_scale'])
        accessor=FoldPCAAccessor(shared,v.head.dataset,'test',[*contract['columns'],v.head.target]).with_pca(contract['joint_lift_pca'])
        n._set_target_column(accessor,v.head.target);samples=samples_for(case,v.information_set,'test')
        for s in r['seeds']:
            model=n.build_model(candidate.model,len(contract['columns'])+int(dynamic),candidate.capacity).to(device)
            model.load_state_dict(torch.load(s['checkpoint'],map_location=device,weights_only=True))
            prediction=n._predict(model,accessor,samples,contract['columns'],128,scaler,dynamic=dynamic,device=device)
            record=level_export(shared,v,samples,prediction,ROOT/'pointwise_enriched'/case/'test'/v.information_set/(r['model']+f"_seed_{s['seed']}.parquet"))
            summary.append(dict(direction=case,information_set=v.information_set,model=r['model'],split='test',seed=s['seed'],**record))
            del model
    pd.DataFrame(summary).to_csv(ROOT/'CROSSROD_E1_BASELINE_METRICS.csv',index=False)
    dump(ROOT/'FINALIZATION_STATUS.json',dict(status='E1_AND_BASELINE_TARGET_INFERENCE_COMPLETED',fit_during_inference=False,
        pointwise_exported=True,remaining_experiments=['E2_SCOPE_REVALIDATION','E3_EQUAL_BUDGET','E4_SENSITIVITY','E5_STRUCTURE_SUMMARY','E6_RAW_NOISE_REIDENTIFICATION']))

def main():
    dump(ROOT/'FINALIZATION_STATUS.json',dict(status='WAITING_FOR_BOTH_MAIN_AND_ALL_BASELINE_FREEZES',target_labels_loaded=False))
    try:
        while True:
            frozen=ready()
            if frozen is not None:break
            # Do not retry a failed main job or invent a deployment.
            path=ROOT/'MAIN_RUN_STATUS.json'
            if path.exists():
                status=json.loads(path.read_text())
                if any(r['exit_code'] not in {None,0} for r in status.get('runs',[])):raise RuntimeError('Main formal run failed; inspect retained logs')
            time.sleep(10)
        finalize(frozen)
    except Exception as e:
        dump(ROOT/'FINALIZATION_STATUS.json',dict(status='FAILED_RETAINED',error=str(e),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
