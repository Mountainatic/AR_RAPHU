"""Metadata invariants and exact ordered identity support."""
import numpy as np
from .invariants import ProtocolError,digest

def ordered_ids(frame):
    name='sample_id' if 'sample_id' in frame else 'view_sample_id'
    if name not in frame:raise ProtocolError('Explicit sample IDs required')
    return frame[name].astype(str).tolist()

def support_hash(frame):return digest(ordered_ids(frame))

def validate_partition(frame):
    ids=ordered_ids(frame)
    if len(set(ids))!=len(ids):raise ProtocolError('Duplicate sample IDs')
    required={'entity_id','origin','target_stop_exclusive','latest_available_target_index'}
    if not required.issubset(frame):raise ProtocolError('Missing temporal metadata')
    if frame[['entity_id','origin']].duplicated().any():raise ProtocolError('Duplicate entity/origin')
    if np.any(frame.target_stop_exclusive.to_numpy(int)<=frame.origin.to_numpy(int)):raise ProtocolError('Invalid target end')
    if np.any(frame.latest_available_target_index.to_numpy(int)>=frame.origin.to_numpy(int)):raise ProtocolError('Future target available before prediction')
    seen=set();previous=None;last=None
    for entity,origin in zip(frame.entity_id.astype(str),frame.origin.to_numpy(int)):
        if entity!=previous:
            if entity in seen:raise ProtocolError('Entity runs must not recur')
            seen.add(entity);last=None;previous=entity
        if last is not None and origin<=last:raise ProtocolError('Origins must increase within entity')
        last=origin
    return frame
