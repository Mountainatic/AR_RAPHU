"""Joint nearest legal placement: sum displacement, maximum, then lexical order."""
from functools import lru_cache
import numpy as np
from .invariants import ProtocolError
from .partition import validate_partition,support_hash
from .temporal_support import past_fit_positions

PLACEMENT='ENTITY_AWARE_NEAREST_LEGAL_4X320_V1'
UNAVAILABLE='INSUFFICIENT_PARTITION_ROWS_FOR_REGISTERED_GEOMETRY'

@lru_cache(maxsize=32768)
def _optimize(legal,nominal):
    def solve(limit):
        previous=None
        for nominal_end in nominal:
            current=[];cursor=0;best=None
            for end in legal:
                if previous is None:base=(0,())
                else:
                    while cursor<len(legal) and legal[cursor]<=end-320:
                        candidate=previous[cursor]
                        if candidate is not None and (best is None or candidate<best):best=candidate
                        cursor+=1
                    base=best
                distance=abs(end-nominal_end)
                current.append(None if base is None or distance>limit else (base[0]+distance,base[1]+(end,)))
            previous=current
        valid=[p for p in previous if p is not None]
        return min(valid) if valid else None
    if not legal:return None
    optimal=solve(max(max(legal),max(nominal))+320)
    if optimal is None:return None
    low,high=0,max(abs(e-n) for e,n in zip(optimal[1],nominal))
    while low<high:
        middle=(low+high)//2;trial=solve(middle)
        if trial is not None and trial[0]==optimal[0]:high=middle
        else:low=middle+1
    return solve(low)[1]

def placement(frame,geometry):
    validate_partition(frame)
    if geometry!={'block_rows':320,'count':4,'end_fractions':[.4,.6,.8,1.]}:raise ProtocolError('Unregistered fold geometry')
    nominal=tuple(np.floor(len(frame)*np.asarray(geometry['end_fractions'])).astype(int).tolist())
    legal=[];run=0;previous=None;last=None
    for position,(entity,origin) in enumerate(zip(frame.entity_id.astype(str),frame.origin.astype(int)),1):
        run=run+1 if entity==previous and origin==last+1 else 1
        if run>=320:legal.append(position)
        previous,last=entity,origin
    endpoints=_optimize(tuple(legal),nominal)
    return nominal,endpoints,len(legal)

def folds(frame,geometry):
    _,endpoints,_=placement(frame,geometry)
    if endpoints is None:return []
    return [(past_fit_positions(frame,np.arange(end-320,end)),np.arange(end-320,end)) for end in endpoints]

def require_top_level_geometry(frame,geometry):
    if not folds(frame,geometry):raise ProtocolError('HARD_FAIL_PREFLIGHT: SCORING_GEOMETRY_UNAVAILABLE')

def selection_blocks(frame,geometry,context_rows):
    result=[]
    for j,(_,score) in enumerate(folds(frame,geometry)):
        context=np.arange(max(0,int(score[0])-int(context_rows)),int(score[0]))
        evaluation=np.concatenate([context,score])
        result.append((past_fit_positions(frame,evaluation),evaluation,context,score,j))
    return result

def geometry_audit(frame,geometry):
    nominal,endpoints,legal_count=placement(frame,geometry);records=[]
    if endpoints is not None:
        for j,(n,end) in enumerate(zip(nominal,endpoints)):
            start=end-320;score=frame.iloc[start:end];entity=str(score.entity_id.iloc[0])
            records.append(dict(fold=j,start=start,end=end,nominal_endpoint=n,actual_endpoint=end,
                displacement_rows=end-n,displacement_fraction=(end-n)/len(frame),entity_id=entity,
                origin_start=int(score.origin.iloc[0]),origin_end=int(score.origin.iloc[-1]),support_hash=support_hash(score),
                rows=len(score),crosses_entity=score.entity_id.nunique()!=1,
                contiguous=bool(np.all(np.diff(score.origin.to_numpy(int))==1)),entities=[dict(entity_id=entity,rows=320,
                origin_min=int(score.origin.iloc[0]),origin_max=int(score.origin.iloc[-1]),
                time_seconds_min=2*int(score.origin.iloc[0]),time_seconds_max=2*int(score.origin.iloc[-1]))]))
    return dict(rows=len(frame),available=endpoints is not None,placement=PLACEMENT,
        status='AVAILABLE' if endpoints is not None else 'SCORING_GEOMETRY_UNAVAILABLE',
        reason=None if endpoints is not None else UNAVAILABLE,legal_endpoint_count=legal_count,
        nominal_endpoints=nominal,actual_endpoints=endpoints,score_blocks=records,
        cross_entity_scoring_blocks=sum(r['crosses_entity'] for r in records),boundary_induced_geometry_failure=0,
        estimand='REGISTERED_ENTITY_COVERAGE_FORWARD_DEVELOPMENT_PREDICTIVE_RISK')
