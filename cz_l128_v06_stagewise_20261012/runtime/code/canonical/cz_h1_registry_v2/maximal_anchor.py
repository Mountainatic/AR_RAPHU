"""Maximal source-train anchors inside the original authority L256 boundaries.

Only time/ID metadata is used to build this C1 support. Source train labels can
be reconstructed separately on explicit fit-data loading, never in preflight.
The original joint-lift PCA and base inputs are reused without fitting.
"""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from canonical_core import digest, dump, file_sha, folds
from prism_benchmark.v211_support import SUPPORT_CONTRACT, support_id_hash

CONTRACT = 'CZ_H1_MAXIMAL_AUTHORITY_PARTITION_ANCHORS_V1'
TASK = 'CZ_DIAM_RAW2S_CURRENT_L256_H1'
DIRECTIONS = {'Rod_1_to_Rod_2':'Rod_1','Rod_2_to_Rod_1':'Rod_2'}
SAMPLE_RELATIVE = Path('sample_ids')/TASK/'dynamic/record_time/primary/train.parquet'
TRAIN_METADATA = ['base_origin_id','view_sample_id','dataset','entity_id','task_id','split',
    'origin','dependency_start','dependency_stop_exclusive','causal_history_floor','anchor_history_steps',
    'sample_support_contract']


def entity_key(entity):
    match=re.fullmatch(r'Rod_(\d+)_segment_(\d+)',str(entity))
    if not match:raise ValueError('Unexpected authority entity ID')
    return tuple(map(int,match.groups()))


def maximal_metadata(old,base,direction):
    if set(old['split'])!={'train'} or set(old['task_id'])!={TASK}:
        raise ValueError('Expected authority explicit-H1 source-train metadata')
    if not all(str(x).startswith(DIRECTIONS[direction]+'_') for x in base.entity_id):
        raise ValueError('Source-rod metadata mismatch')
    boundaries=old.groupby('entity_id',sort=False).origin.max().astype(int).to_dict()
    last=max(boundaries,key=entity_key)
    rows=[];ranges=[]
    for entity,g in sorted(base.groupby('entity_id',sort=False),key=lambda pair:entity_key(pair[0])):
        if entity_key(entity)>entity_key(last):continue
        indices=np.sort(g.row_in_entity.to_numpy(dtype=int))
        if not np.array_equal(indices,np.arange(len(indices))):raise ValueError('Noncontiguous raw entity metadata')
        # Previously omitted short segments before the frozen train right edge
        # belong to that same chronological train prefix; no boundary is moved.
        end=int(boundaries.get(entity,len(indices)-1))
        if end>=len(indices):raise ValueError('Target crosses raw entity boundary')
        ranges.append(dict(entity_id=entity,origin_start=1,origin_stop_exclusive=end+1,
                           authority_train_right_edge=end,previously_had_anchors=entity in boundaries))
        for origin in range(1,end+1):
            base_id=hashlib.sha256(f'{TASK}|{direction}|{entity}|{origin}'.encode()).hexdigest()
            rows.append(dict(base_origin_id=base_id,view_sample_id=hashlib.sha256(f'dynamic|{base_id}'.encode()).hexdigest(),
                dataset='cz_czochralski',entity_id=entity,task_id=TASK,target_head=TASK,split='train',origin=origin,
                current_start=origin-1,current_stop_exclusive=origin,history_start=origin-1,history_stop_exclusive=origin,
                target_start=origin,target_stop_exclusive=origin+1,
                dependency_start=max(0,origin-256),dependency_stop_exclusive=origin+1,
                latest_available_target_index=origin-1,availability_delay_steps=0,
                availability_scenario='record_time',proxy_policy='primary',information_set='dynamic',
                causal_history_floor=0,anchor_history_steps=1,sample_support_contract=SUPPORT_CONTRACT,lmax_steps=256))
    samples=pd.DataFrame(rows)
    if not set(old.base_origin_id).issubset(set(samples.base_origin_id)):
        raise ValueError('An authority source-train anchor disappeared')
    for block in (8,320):
        old_folds=folds(old,block);new_folds=folds(samples,block)
        for (_,oe),(_,ne) in zip(old_folds,new_folds):
            if support_id_hash(old.iloc[oe])!=support_id_hash(samples.iloc[ne]):
                raise ValueError('Registered tail scoring timestamps changed')
    return samples,ranges


def build(authority_shared,new_shared,direction,access_log):
    authority_shared=Path(authority_shared);new_shared=Path(new_shared)
    if new_shared.exists():raise RuntimeError('Refuse overwrite a C1 support freeze')
    realization=json.loads((authority_shared/'CZ_TASK_REALIZATION.json').read_text())
    if realization.get('lookback_steps')!=256 or realization.get('cadence_seconds')!=2 or realization.get('effective_h_steps')!=1:
        raise ValueError('Use authority raw2s L256 H1 C1, not the old long-history study')
    old_path=authority_shared/SAMPLE_RELATIVE
    base_path=authority_shared/'base_data/cz_czochralski/train.parquet'
    old=pq.ParquetFile(old_path).read(columns=TRAIN_METADATA).to_pandas()
    base=pq.ParquetFile(base_path).read(columns=['entity_id','row_in_entity']).to_pandas()
    access_log.extend([dict(path=str(old_path),columns=TRAIN_METADATA,kind='SOURCE_TRAIN_SUPPORT_METADATA'),
                       dict(path=str(base_path),columns=['entity_id','row_in_entity'],kind='SOURCE_ROD_TIME_INDEX_METADATA')])
    samples,ranges=maximal_metadata(old,base,direction)
    dest=new_shared/SAMPLE_RELATIVE;dest.parent.mkdir(parents=True,exist_ok=True)
    samples.to_parquet(dest,index=False)
    # Input-only is the same anchor universe, with its registered view IDs.
    input_only=samples.copy();input_only['information_set']='input_only'
    input_only['view_sample_id']=[hashlib.sha256(f'input_only|{b}'.encode()).hexdigest() for b in samples.base_origin_id]
    other=new_shared/'sample_ids'/TASK/'input_only/record_time/primary/train.parquet'
    other.parent.mkdir(parents=True,exist_ok=True);input_only.to_parquet(other,index=False)
    link=new_shared/'base_data/cz_czochralski/train.parquet';link.parent.mkdir(parents=True,exist_ok=True)
    link.symlink_to(base_path.resolve())
    pca=authority_shared/'JOINT_LIFT_PCA_CONTRACT.json'
    (new_shared/pca.name).write_bytes(pca.read_bytes())
    manifest=dict(contract_id=CONTRACT,direction=direction,status='MAXIMAL_SOURCE_TRAIN_SUPPORT_FROZEN',
        source_shared=str(authority_shared),source_base_sha256=file_sha(base_path),source_train_metadata_digest=digest(old.to_dict('records')),
        source_raw_time_metadata_digest=digest(base.to_dict('records')),pca_sha256=file_sha(pca),
        anchor_rows=len(samples),authority_anchor_rows=len(old),train_support_hash=support_id_hash(samples),
        train_metadata_parquet_sha256=file_sha(dest),ranges=ranges,target_formula='D[t]-D[t-1]',
        target_partition_accessed=False,target_values_accessed=False,labels_materialized=False,
        split_right_edges_unchanged=True,original_scoring_timestamps_unchanged=True,
        dependency_purge='authority L256 envelope max(0,t-256); candidate-native fit masks use segment floor 0',
        history_does_not_determine_anchor_eligibility=True)
    dump(new_shared/'C1_MAXIMAL_ANCHOR_MANIFEST.json',manifest)
    return samples,manifest


def load(shared,*,for_fit=False):
    """Metadata by default. for_fit is an explicit source-train label operation.

    This function never opens validation/test. Rows returned by the projected
    label read are filtered to the registered training time indices.
    """
    shared=Path(shared);manifest=json.loads((shared/'C1_MAXIMAL_ANCHOR_MANIFEST.json').read_text())
    if manifest['contract_id']!=CONTRACT or file_sha(shared/SAMPLE_RELATIVE)!=manifest['train_metadata_parquet_sha256']:
        raise RuntimeError('C1 maximal anchor freeze changed')
    samples=pd.read_parquet(shared/SAMPLE_RELATIVE)
    if support_id_hash(samples)!=manifest['train_support_hash']:raise RuntimeError('C1 support changed')
    if not for_fit:return samples,manifest
    source=Path(manifest['source_shared'])/'base_data/cz_czochralski/train.parquet'
    if file_sha(source)!=manifest['source_base_sha256']:raise RuntimeError('Frozen source base changed')
    filters=[]
    for entity,g in samples.groupby('entity_id',sort=False):
        filters.append([('entity_id','=',entity),('row_in_entity','>=',0),('row_in_entity','<=',int(g.origin.max()))])
    target=pq.read_table(source,columns=['entity_id','row_in_entity','crystal_diameter'],filters=filters).to_pandas()
    series=target.set_index(['entity_id','row_in_entity']).crystal_diameter
    current=pd.MultiIndex.from_arrays([samples.entity_id,samples.origin-1])
    future=pd.MultiIndex.from_arrays([samples.entity_id,samples.origin])
    samples['y_true']=series.reindex(future).to_numpy(float)-series.reindex(current).to_numpy(float)
    if not np.isfinite(samples.y_true).all():raise RuntimeError('Missing registered source-train label')
    return samples,manifest
