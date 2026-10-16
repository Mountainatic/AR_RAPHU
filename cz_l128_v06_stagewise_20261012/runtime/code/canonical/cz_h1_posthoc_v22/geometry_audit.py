"""Label-free recursive call-graph geometry audit (possible domain superset)."""
from collections import Counter
import numpy as np
from .fold_geometry import folds,selection_blocks,geometry_audit
from .temporal_support import final_replay_blocks
from .invariants import digest


def recursive_geometry_audit(frame,protocol):
    pending=[('LEARN','A',np.arange(len(frame)))];seen=set();counts=Counter();examples=[]
    while pending:
        role,stage,positions=pending.pop();key=(role,stage,digest(positions.tolist()))
        if key in seen:continue
        seen.add(key);T=frame.iloc[positions].reset_index(drop=True)
        geometry=protocol['geometry']['outer' if role=='LEARN' else 'inner']
        audit=geometry_audit(T,geometry);available=audit['available']
        counts[role+'_'+stage+'_AVAILABLE' if available else role+'_'+stage+'_UNAVAILABLE']+=1
        crossed=any(b['crosses_entity'] for b in audit['score_blocks'])
        if crossed:
            counts['CROSS_ENTITY_UNAVAILABLE']+=1
            if len(examples)<20:examples.append(dict(role=role,stage=stage,partition_positions_hash=key[-1],audit=audit))
        def add(r,s,local):pending.append((r,s,positions[local]))
        up={'C':'K','W':'C','A':'W'}.get(stage)
        if up:add('LEARN',up,np.arange(len(T)))
        if not available:continue
        if role=='LEARN':
            add('MODEL',stage,np.arange(len(T)))
            for fit,_ in folds(T,geometry):add('MODEL',stage,fit)
        elif stage=='C':
            for fit,_ in folds(T,geometry):add('LEARN','K',fit)
        elif stage in {'W','A'}:
            for fit,*_ in selection_blocks(T,geometry,protocol['geometry']['selection_cf_context_rows_per_block']):add('LEARN',up,fit)
            for fit,_ in final_replay_blocks(T,320):add('LEARN',up,fit)
    return dict(status='COMPLETED',metadata_only=True,target_values_accessed=False,
        enumeration='ALL_POSSIBLE_RECURSIVE_CALLS_BEFORE_NUMERICAL_DOMAIN_PRUNING',
        unique_realizations=len(seen),counts=dict(sorted(counts.items())),examples=examples,
        top_level=geometry_audit(frame,protocol['geometry']['outer']),
        endpoint_moved=False,block_shortened=False)
