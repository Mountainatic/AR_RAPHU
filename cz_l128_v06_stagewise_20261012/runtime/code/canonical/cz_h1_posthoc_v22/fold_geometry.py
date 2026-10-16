"""User-frozen distributed inner/outer scoring; no outcome-dependent placement."""
import numpy as np
from .invariants import ProtocolError
from .temporal_support import past_fit_positions


def folds(frame,geometry):
    if geometry!={'block_rows':320,'count':4,'end_fractions':[.4,.6,.8,1.]}:
        raise ProtocolError('Unregistered inner/outer fold geometry')
    endpoints=np.floor(len(frame)*np.asarray(geometry['end_fractions'])).astype(int)
    starts=endpoints-geometry['block_rows']
    if np.any(starts<0) or np.any(starts[1:]<endpoints[:-1]):
        return []
    for start,end in zip(starts,endpoints):
        score=frame.iloc[start:end]
        if score.entity_id.nunique()!=1 or np.any(np.diff(score.origin.to_numpy(int))!=1):
            return []
    return [(past_fit_positions(frame,np.arange(start,end)),np.arange(start,end))
            for start,end in zip(starts,endpoints)]


def selection_blocks(frame,geometry,context_rows):
    result=[]
    for j,(_,score) in enumerate(folds(frame,geometry)):
        context=np.arange(max(0,int(score[0])-int(context_rows)),int(score[0]))
        evaluation=np.concatenate([context,score])
        result.append((past_fit_positions(frame,evaluation),evaluation,context,score,j))
    return result


def geometry_audit(frame,geometry):
    endpoints=np.floor(len(frame)*np.asarray(geometry['end_fractions'])).astype(int)
    records=[]
    for j,end in enumerate(endpoints):
        start=int(end)-geometry['block_rows'];score=frame.iloc[max(0,start):int(end)]
        entities=[]
        for entity,g in score.groupby('entity_id',sort=False):
            entities.append(dict(entity_id=str(entity),rows=len(g),origin_min=int(g.origin.min()),
                origin_max=int(g.origin.max()),time_seconds_min=2*int(g.origin.min()),
                time_seconds_max=2*int(g.origin.max())))
        records.append(dict(fold=j,start=start,end=int(end),entities=entities,
            crosses_entity=len(entities)>1,contiguous=bool(len(score) and np.all(np.diff(score.origin.to_numpy(int))==1))))
    return dict(rows=len(frame),available=bool(folds(frame,geometry)),score_blocks=records,
        estimand='MAIN_LONG_GROWTH_SEGMENT_FORWARD_DEVELOPMENT_PREDICTIVE_RISK')
