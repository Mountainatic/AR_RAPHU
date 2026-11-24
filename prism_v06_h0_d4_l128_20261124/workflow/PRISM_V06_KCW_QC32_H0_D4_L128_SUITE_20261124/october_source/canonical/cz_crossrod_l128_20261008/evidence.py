"""Reporting-only paired gains and frozen entity/block bootstrap procedures."""
import numpy as np
from .invariants import ProtocolError
from .risk import paired_route
from .partition import ordered_ids


def _runs(frame):
    result=[]
    for _,group in frame.groupby(['entity_id','fold'],sort=False):
        ordered=group.sort_values('origin')
        positions=ordered.origin.to_numpy(np.int64)
        cuts=np.flatnonzero(np.diff(positions)!=1)+1
        for indices in np.split(np.arange(len(ordered)),cuts):
            if len(indices):
                result.append(ordered.iloc[indices])
    return result


def _sample_run(values,block,rng):
    n=len(values)
    if n<block:
        raise ProtocolError('Bootstrap run is shorter than its frozen block')
    starts=rng.integers(0,n-block+1,size=(n+block-1)//block)
    return np.concatenate([values[s:s+block] for s in starts])[:n]


def report_evidence(pointwise,procedure):
    required={'y_true','parent','MODEL_child','entity_id','origin','fold'}
    if not required.issubset(pointwise) or not len(pointwise):
        raise ProtocolError('Reporter requires canonical paired pointwise records')
    y=pointwise.y_true.to_numpy(float)
    parent=pointwise.parent.to_numpy(float)
    child=pointwise.MODEL_child.to_numpy(float)
    route=paired_route(y,parent,child,score_ids=ordered_ids(pointwise))
    frame=pointwise.copy()
    frame['squared_error_gain']=(y-parent)**2-(y-child)**2
    by_fold=[]
    for fold,g in frame.groupby('fold',sort=False):
        evidence=paired_route(g.y_true,g.parent,g.MODEL_child,score_ids=ordered_ids(g))
        by_fold.append(dict(fold=int(fold),rows=len(g),gain=evidence['incremental_gain'],
            parent_risk=evidence['parent_oof_risk'],child_risk=evidence['child_oof_risk']))
    if procedure.get('selection_authority') is not False:
        raise ProtocolError('Evidence procedures are reporting-only')
    replicates=int(procedure['replicates'])
    quantiles=list(map(float,procedure['quantiles']))
    if replicates<1 or len(quantiles)!=2 or not 0<quantiles[0]<quantiles[1]<1:
        raise ProtocolError('Invalid frozen bootstrap procedure')
    runs=_runs(frame)
    entities=list(frame.entity_id.astype(str).unique())
    per_entity={entity:[g.squared_error_gain.to_numpy(float) for g in runs
        if str(g.entity_id.iloc[0])==entity] for entity in entities}
    bootstrap=[]
    for block in procedure['block_lengths_steps']:
        block=int(block)
        if block<1 or any(len(values)<block for arrays in per_entity.values() for values in arrays):
            bootstrap.append(dict(block_steps=block,status='NOT_APPLICABLE',
                reason='FROZEN_BLOCK_EXCEEDS_CONTIGUOUS_SCORING_RUN',replicates=0))
            continue
        rng=np.random.default_rng(int(procedure['seed']))
        samples=[]
        for _ in range(replicates):
            chosen=rng.choice(entities,size=len(entities),replace=True) if procedure['complete_entity_resampling'] else entities
            selected=[_sample_run(values,block,rng) for entity in chosen for values in per_entity[entity]]
            samples.append(float(np.concatenate(selected).mean()))
        lower,upper=np.quantile(samples,quantiles,method='linear')
        bootstrap.append(dict(block_steps=block,status='COMPLETED',replicates=replicates,
            gain_interval=[float(lower),float(upper)],positive_gain_probability=float(np.mean(np.asarray(samples)>0)),
            quantiles=quantiles,seed=int(procedure['seed']),selection_authority=False))
    return dict(status='COMPLETED',scientific_status='POST_HOC_RETROSPECTIVE_EVIDENCE',
        paired_rows=len(frame),fold_gains=by_fold,paired_risk=route,bootstrap=bootstrap,
        procedure=procedure,reporting_only=True,route_changed_by_report=False,
        NA_identity_rows_retained=True,no_bootstrap_block_crosses_entity_fold_or_origin_gap=True)
