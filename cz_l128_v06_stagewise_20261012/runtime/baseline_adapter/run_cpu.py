"""Source-only baseline development, final fit, reload, and pointwise exports."""
import argparse
import json
import os
from pathlib import Path
import shutil
import traceback
from types import SimpleNamespace
import numpy as np
import pandas as pd
from prism_benchmark.cpu_data import HeadSpec,ViewSpec,sha256_file
from prism_benchmark.v211_support import load_native_samples
from prism_benchmark.portable_checkpoints import load_portable_checkpoint
from . import v211_public_all_baselines as dev
from . import representative_baseline_checkpoints as checkpoints

ROOT=Path('/root/autodl-tmp/CZ_CROSSROD_L128_V06_20261010')
TASK='CZ_DIAM_RAW2S_CURRENT_L128_H1_CROSSROD_20261008'

def verify_release():
    seal=ROOT/'BASELINE_RELEASE.json'
    release=json.loads(seal.read_text())
    for path,sha in release['file_sha256'].items():
        if sha256_file(Path(path))!=sha:raise RuntimeError('Baseline release changed: '+path)
    return sha256_file(seal)

def view(information):
    head=HeadSpec(TASK,TASK,'cz_czochralski','crystal_diameter',2.,1,1,1,True)
    return ViewSpec(head,information,'record_time','primary')

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)

def run(case,models=None):
    release_sha=verify_release()
    shared=ROOT/'C1'/case;project=ROOT/'code';output=ROOT/'baselines'/case/'cpu_verified'
    output.mkdir(parents=True,exist_ok=False)
    jobs=json.loads((ROOT/'prepared/BASELINE_JOBS.json').read_text())
    jobs=[j for j in jobs if j['direction']==case and j['family']=='CPU_BASELINE' and (models is None or j['model'] in models)]
    # AR must be frozen before its ARX/NARX dependents.
    jobs.sort(key=lambda j:(0 if j['model'] in {'MEAN','PERSISTENCE','SEASONAL_PERSISTENCE','AR'} else 1))
    records=[]
    for job in jobs:
        if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError('NEW_TASK_LOW_DISK')
        information=job['information_set'];model=job['model'];v=view(information)
        item=dict(job,phase='DEVELOPMENT',test_accessed=False)
        dump(output/'RUN_STATUS.json',dict(status='RUNNING',current=item,records=records))
        try:
            if model in dev.SIMPLE_MODELS:r=dev.run_simple_job(shared,project,output,v,model)
            elif model in dev.STATIC_INPUT_MODELS:r=dev.run_static_input_job(shared,project,output,v,model)
            elif model=='DPLS':r=dev.run_dpls_job(shared,project,output,v)
            elif model in {'PARALLEL_HAMMERSTEIN','HAMMERSTEIN_WIENER'}:r=dev.run_hammerstein_job(shared,project,output,v,model)
            else:r={'AR':dev.run_ar_job,'ARX':dev.run_arx_job,'LINEAR_NARX':dev.run_narx_job,'N4SID':dev.run_n4sid_job}[model](shared,project,output,v)
            item.update(status=r['status'],development=r)
            if r['status']=='PASS' and model in checkpoints.FORMAL_COMPATIBLE:
                paths=SimpleNamespace(shared=shared,project=project,run_root=output)
                frozen=checkpoints._fit_one(paths,v,model,r,output/'checkpoints')
                checkpoint=Path(frozen['checkpoint_dir'])
                os.environ['PRISM_FORMAL_INFERENCE_ONLY']='1'
                try:
                    replay=checkpoints.verify_baseline_checkpoint_reload(checkpoint)
                    state,arrays,manifest=load_portable_checkpoint(checkpoint)
                    samples=load_native_samples(shared,v,'validation')
                    x=checkpoints._evaluation_features(paths,v,samples,'validation',state,arrays)
                    prediction=checkpoints._predict_loaded(checkpoint,x,state,arrays)
                    frame=samples.copy();frame['y_pred']=prediction
                    frame['future_level_pred']=frame['current_level']+prediction if 'current_level' in frame else np.nan
                    # Original baseline development output must replay on the
                    # same IDs, including the partition-specific PCA.
                    family='C2' if model in dev.STATIC_INPUT_MODELS or model in dev.SIMPLE_MODELS or model=='DPLS' else 'C3'
                    original=pd.read_parquet(dev._result_root(output,family,model,v)/'validation.parquet')
                    merged=frame[['base_origin_id','y_pred']].merge(original[['base_origin_id','y_pred']],on='base_origin_id',validate='one_to_one',suffixes=('_checkpoint','_development'))
                    if len(merged)!=len(frame):raise RuntimeError('Validation row support drift')
                    error=float(np.max(np.abs((merged.y_pred_checkpoint-merged.y_pred_development).to_numpy()),initial=0.))
                    if error>1e-7:raise RuntimeError(f'Final checkpoint/development replay mismatch {error}')
                    # Anchor levels come from source raw past, never target test.
                    frame['current_level']=checkpoints._current_levels(paths,v,samples,'validation')
                    frame['future_level_pred']=frame.current_level+prediction
                    frame['future_level_true']=frame.current_level+frame.y_true
                    destination=output/'frozen_validation'/information/(model+'.parquet')
                    destination.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(destination,index=False,compression='zstd')
                    item.update(status='FROZEN',frozen=frozen,reload=replay,development_replay_max_abs_error=error,
                        validation_path=str(destination),validation_sha256=sha256_file(destination))
                finally:os.environ.pop('PRISM_FORMAL_INFERENCE_ONLY',None)
            elif r['status']=='PASS':
                item.update(status='NOT_RUN_PROTOCOL_INCOMPATIBLE',reason='Inherited portable final refit unavailable')
        except Exception as e:
            item.update(status='FAILED_RETAINED',error=str(e),traceback=traceback.format_exc())
        records.append(item)
        print(case,information,model,item['status'],flush=True)
        dump(output/'RUN_STATUS.json',dict(status='RUNNING',records=records))
    status='ALL_REGISTERED_CPU_JOBS_TERMINAL' if not any(r['status']=='FAILED_RETAINED' for r in records) else 'FAILED_RETAINED'
    dump(output/'CPU_FREEZE_MANIFEST.json',dict(status=status,case=case,task_id=TASK,history_steps=128,
        final_fit_partition='SOURCE_TRAIN_ONLY',test_used_for_selection=False,baseline_release_sha256=release_sha,records=records))
    dump(output/'RUN_STATUS.json',dict(status=status,records=records))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--models',nargs='+')
    a=p.parse_args();run(a.case,a.models)
