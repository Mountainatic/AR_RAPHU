"""Maximal H1 anchors from row identity and original chronological split metadata only."""
import hashlib,json
from pathlib import Path
import pandas as pd
from .invariants import ProtocolError,dump,file_sha
from .partition import ordered_ids,support_hash,validate_partition

SAMPLE_RELATIVE='sample_ids/CZ_DIAM_RAW2S_CURRENT_L128_H1_CROSSROD_20261008/dynamic/record_time/primary'
METADATA_COLUMNS={'base_origin_id','view_sample_id','sample_id','dataset','entity_id','task_id','target_head','split','origin',
    'current_start','current_stop_exclusive','history_start','history_stop_exclusive','target_start','target_stop_exclusive',
    'dependency_start','dependency_stop_exclusive','latest_available_target_index','availability_delay_steps','availability_scenario',
    'proxy_policy','information_set','causal_history_floor','anchor_history_steps','sample_support_contract','lmax_steps'}

def metadata(path):
    import pyarrow.parquet as pq
    available=pq.ParquetFile(path).schema.names
    return pq.read_table(path,columns=[c for c in available if c in METADATA_COLUMNS]).to_pandas()

def build_support(old,other,raw,*,split,task,direction):
    if not set(old.columns)<=METADATA_COLUMNS or not set(other.columns)<=METADATA_COLUMNS:
        raise ProtocolError('Support builder accepts metadata columns only')
    if set(raw.columns)!={'entity_id','row_in_entity'}:raise ProtocolError('Support builder accepts raw row identity only')
    raw=raw.drop_duplicates(['entity_id','row_in_entity']);records=[];ranges=[]
    for entity,g in old.groupby('entity_id',sort=False):
        indices=set(raw.loc[raw.entity_id==entity,'row_in_entity'].astype(int))
        if not indices:raise ProtocolError('Missing entity row metadata')
        previous=other.loc[other.entity_id==entity]
        # A within-entity validation boundary is immutable. An entire validation-
        # only entity or target rod has the same split membership from row 1.
        start=int(g.origin.min()) if split=='validation' and len(previous) else 1
        stop=int(g.origin.max()) if split in {'train','validation'} else max(indices)
        template=g.iloc[0].to_dict()
        for origin in range(start,stop+1):
            if origin not in indices or origin-1 not in indices:continue
            match=g.loc[g.origin==origin]
            row=match.iloc[0].to_dict() if len(match) else dict(template)
            base=hashlib.sha256(f'{task}|{direction}|{entity}|{origin}'.encode()).hexdigest()
            row.update(origin=origin,base_origin_id=base,view_sample_id=hashlib.sha256(('dynamic|'+base).encode()).hexdigest(),
                split=split,current_start=origin-1,current_stop_exclusive=origin,history_start=origin-1,history_stop_exclusive=origin,
                target_start=origin,target_stop_exclusive=origin+1,dependency_start=origin-1,dependency_stop_exclusive=origin+1,
                latest_available_target_index=origin-1,causal_history_floor=0,anchor_history_steps=1)
            if 'sample_id' in row:row['sample_id']=row['view_sample_id']
            # Existing identity bytes are authority-bound; a disagreement is not silently repaired.
            if len(match) and str(row['view_sample_id'])!=str(match.iloc[0].view_sample_id):raise ProtocolError('Authority sample ID reconstruction differs')
            records.append(row)
        ranges.append(dict(entity_id=str(entity),origin_start=start,origin_end=stop,
            original_split_origin_start=int(g.origin.min()),original_split_origin_end=int(g.origin.max()),
            restored_early_whole_entity_rows=split=='test' or not len(previous)))
    new=pd.DataFrame(records,columns=old.columns)
    if not len(old):new=old.copy()
    validate_partition(new)
    original=ordered_ids(old);actual=ordered_ids(new)
    if not set(original)<=set(actual):raise ProtocolError('Original split IDs are not a subset')
    return new,dict(old_rows=len(old),maximal_rows=len(new),restored_rows=len(new)-len(old),old_support_is_subset=True,
        split_boundaries_unchanged=True,entity_ranges=ranges,old_support_hash=support_hash(old),maximal_support_hash=support_hash(new),
        support_ID_equivalent=original==actual,target_values_read=False,builder_input_role='ROW_IDENTITIES_AND_FROZEN_SPLIT_METADATA_ONLY')

def materialize_case(source,destination):
    import pyarrow.parquet as pq
    source=Path(source);destination=Path(destination)
    task=json.loads((source/'CZ_TASK_REALIZATION.json').read_text())
    old={s:metadata(source/SAMPLE_RELATIVE/(s+'.parquet')) for s in ('train','validation','test')}
    outputs={}
    for split in ('validation','test'):
        paths=[source/'base_data/cz_czochralski/train.parquet'] if split=='validation' else []
        pieces=[pq.read_table(p,columns=['entity_id','row_in_entity']).to_pandas() for p in paths if p.exists()]
        raw=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['entity_id','row_in_entity'])
        frame=old[split].copy();validate_partition(frame)
        if (frame.origin<128).any():raise ProtocolError('Incomplete L128 anchor')
        audit=dict(old_rows=len(frame),maximal_rows=len(frame),restored_rows=0,old_support_is_subset=True,split_boundaries_unchanged=True,target_values_read=False,policy='EXACT_FROZEN_L128_SUPPORT')
        path=destination/(split+'.parquet');path.parent.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(path,index=False)
        dump(destination/(split+'_AUDIT.json'),audit)
        outputs[split]=dict(relative=path.name,sha256=file_sha(path),support_hash=support_hash(frame),audit=audit)
    return outputs
