"""Past-only raw support and the already frozen 320-row final replay geometry."""
import numpy as np
from .invariants import ProtocolError
from .partition import validate_partition


def past_fit_positions(frame,evaluation_positions):
    validate_partition(frame)
    evaluation=np.asarray(evaluation_positions,dtype=np.int64)
    if not len(evaluation):
        raise ProtocolError('Empty evaluation support')
    first=int(evaluation.min())
    positions=np.arange(first,dtype=np.int64)
    keep=np.ones(first,dtype=bool)
    ev=frame.iloc[evaluation]
    entities=frame.entity_id.astype(str).to_numpy()
    stops=frame.dependency_stop_exclusive.to_numpy(np.int64)
    for entity,g in ev.groupby('entity_id',sort=False):
        same=entities[:first]==str(entity)
        keep[same] &= stops[:first][same]<int(g.dependency_start.min())
    result=positions[keep]
    if len(result) and result[-1]>=first:
        raise ProtocolError('Future partition row in purged prefix')
    return result


def final_replay_blocks(frame,block_rows):
    validate_partition(frame)
    if int(block_rows)!=320:
        raise ProtocolError('The user-frozen final replay block size is 320')
    if set(frame.split.astype(str))!={'train'}:
        raise ProtocolError('CF replay accepts only source-development train rows')
    if frame[['entity_id','origin']].duplicated().any():
        raise ProtocolError('Duplicate CF origins')
    entities=frame.entity_id.astype(str).to_numpy()
    seen=set()
    first=0
    result=[]
    while first<len(frame):
        entity=entities[first]
        if entity in seen:
            raise ProtocolError('Replay entity runs must be contiguous')
        seen.add(entity)
        end=first+1
        while end<len(frame) and entities[end]==entity:
            end+=1
        if np.any(np.diff(frame.origin.iloc[first:end].to_numpy(np.int64))<=0):
            raise ProtocolError('Replay origins must increase within entity')
        for begin in range(first,end,320):
            ev=np.arange(begin,min(begin+320,end),dtype=np.int64)
            result.append((past_fit_positions(frame,ev),ev))
        first=end
    covered=np.concatenate([ev for _,ev in result]) if result else np.array([],dtype=np.int64)
    if not np.array_equal(covered,np.arange(len(frame))):
        raise ProtocolError('Final replay fails complete anchor coverage')
    return result
