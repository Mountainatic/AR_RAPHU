"""H1/L128 adapters for the original E3 equal-budget and compact E6 audits.

These are the original diagnostic surrogate experiments, explicitly separate
from the full Urysohn PRISM cross-rod E1 deployment. No surrogate is substituted
for an E1 stage. No target-rod test is used by any extension.
"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from . import authority_extension as ext
from canonical.cz_crossrod_l128_20261008.data import _labels,SAMPLE_RELATIVE
from canonical.cz_crossrod_l128_20261008.joint_lift import PCAAccessor,fit_pca
from canonical.cz_crossrod_l128_20261008.fold_geometry import folds
from canonical.cz_crossrod_l128_20261008.invariants import prediction_hash
from prism_benchmark.cpu_data import BaseAccessor
from prism_benchmark.e1e6_phase_b import _e3_candidates,E3_RIDGES,E3_SEEDS,_fit_ridge
from prism_benchmark.strict_oof_selection import strict_nested_oof_select
from prism_benchmark.e1e6_revalidation import _metrics,support_hash
from baseline_adapter.run_cpu import TASK,view

def samples(shared,information,split):
    relative=SAMPLE_RELATIVE.replace('/dynamic/','/'+information+'/')
    T=pd.read_parquet(shared/relative/(split+'.parquet'))
    paths=[shared/'base_data/cz_czochralski/train.parquet']
    if split=='validation':paths.append(shared/'base_data/cz_czochralski/validation.parquet')
    return _labels(T,paths)

def channels_from_freeze(root,case):
    state=json.loads((root/'formal_actual'/case/'deployment_state.json').read_text())
    while state['stage']!='K':state=state['parent']
    if state['routing_status']!='ADMITTED':raise RuntimeError('E3 has no admitted K root; report domain N/A')
    return tuple(r['channel'] for r in state.get('representations') or [])

def design(accessor,T,channels,assignment):
    values=[]
    for channel,history in zip(channels,assignment):
        offsets=np.unique(np.rint(np.linspace(1,int(history),8)).astype(int))
        values.append(accessor.block_means(T,channel,[(int(i)-1,int(i)) for i in offsets]))
    return np.column_stack(values)

def e3_direction(root,case):
    shared=root/'C1'/case;channels=channels_from_freeze(root,case)
    train=samples(shared,'input_only','train');validation=samples(shared,'input_only','validation')
    raw=BaseAccessor(shared,'cz_czochralski','validation',[*channels,'crystal_lift','crucible_lift','crystal_diameter'])
    for column in [*channels,'crystal_lift','crucible_lift','crystal_diameter']:raw.warm_prefixes([column])
    geometry=folds(train,json.loads((root/'code/canonical/cz_crossrod_l128_20261008/CANONICAL_PROTOCOL.json').read_text())['geometry']['inner'])
    local=[PCAAccessor(raw,fit_pca(raw,train.iloc[fi],history=128)) for fi,_ in geometry]
    full=PCAAccessor(raw,fit_pca(raw,train,history=128));target=train.y_true.to_numpy(float)
    current=raw.block_means(validation,'crystal_diameter',[(0,1)]).reshape(-1)
    rows=[]
    for seed in E3_SEEDS:
        for arm in ('UNIFORM_SCALE','CHANNEL_SPECIFIC_MULTISCALE'):
            started=time.time();candidates=_e3_candidates(channels,(32,64,128),seed,arm)
            assert len(candidates)==3*len(E3_RIDGES)
            losses={};specs={};parent=[];weights=[]
            for fi,ei in geometry:
                parent.append(float(np.mean((target[ei]-target[fi].mean())**2)));weights.append(len(ei))
            for assignment,alpha in candidates:
                key=f'history={",".join(map(str,assignment))}|alpha={alpha:.17g}'
                specs[key]=(assignment,alpha);fold_losses=[]
                for accessor,(fi,ei) in zip(local,geometry):
                    fit=train.iloc[fi];ev=train.iloc[ei]
                    pred=_fit_ridge(design(accessor,fit,channels,assignment),target[fi],design(accessor,ev,channels,assignment),alpha)
                    fold_losses.append(float(np.mean((target[ei]-pred)**2)))
                losses[key]=fold_losses
            selection=strict_nested_oof_select(losses,parent,identity='K_ZERO_IDENTITY',fold_weights=weights)
            chosen=str(selection.tuned_nonzero_candidate);assignment,alpha=specs[chosen]
            prediction=_fit_ridge(design(full,train,channels,assignment),target,design(full,validation,channels,assignment),alpha) if selection.active else np.full(len(validation),target.mean())
            metric=_metrics(validation.y_true.to_numpy(float),prediction,current)
            rows.append(dict(task=case,direction=case,seed=seed,arm=arm,candidate_fit_budget=len(candidates),candidate_fits=len(candidates),
                route=selection.routing_status,absolute_oof_gain=selection.incremental_gain,
                relative_admission_margin=selection.incremental_gain/max(selection.parent_oof_risk,np.finfo(float).eps),
                selected_candidate=chosen,selected_scale_assignment=json.dumps(list(assignment)),distinct_selected_scales=len(set(assignment)),
                channels=len(channels),active_channels=json.dumps(channels),candidate_space_degenerate_single_channel=len(channels)==1,
                train_rows=len(train),validation_rows=len(validation),parameter_count=design(full,train.iloc[:1],channels,assignment).shape[1]+1,
                wall_time_seconds=time.time()-started,Delta_RMSE=metric['rmse_delta'],Delta_MAE=metric['mae_delta'],Delta_R2=metric['r2_delta'],
                Level_R2=metric['r2_level_reconstructed'],support_hash=support_hash(validation.base_origin_id.astype(str)),prediction_hash=prediction_hash(prediction),
                test_accessed=False,ood_accessed=False,PCA_fit_scope='EACH_FOLD_TRAIN_ONLY',scope='ORIGINAL_EQUAL_BUDGET_LINEAR_DIAGNOSTIC_NOT_E1_K'))
    return rows

def configure():
    ext.HEAD=TASK;ext.CZ_HISTORIES=(32,64,128);ext.BLOCK_EMBARGO=132
    ext._view=lambda shared,information:view(information)
    ext._run_e3_direction=e3_direction
    # The compact E6 probe consumes observed raw controls. Do not consume a
    # joint-lift PCA estimated outside an E6 outer training fold.
    ext.input_columns=lambda *args: ['main_heater_power','crystal_lift','crucible_lift','crystal_rotation','crucible_rotation']
    ext.load_native_samples=lambda shared,v,split: samples(shared,v.information_set,split)
    return ext
