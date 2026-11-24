"""Projected CZ labels; inference readers are reached only after runner barriers."""
from pathlib import Path
import numpy as np
import pandas as pd
from .invariants import ProtocolError,file_sha
from .deployment import require_execution
from .partition import validate_partition

SAMPLE_RELATIVE='sample_ids/CZ_DIAM_RAW2S_CURRENT_L256_H1/dynamic/record_time/primary'


def _labels(samples,paths):
    if not len(samples):
        result=samples.copy();result['y_true']=np.empty(0,dtype=float)
        result['sample_id']=result.view_sample_id.astype(str)
        return result
    import pyarrow.parquet as pq
    filters=[[('entity_id','=',entity),('row_in_entity','>=',0),
              ('row_in_entity','<=',int(g.origin.max()))] for entity,g in samples.groupby('entity_id',sort=False)]
    pieces=[pq.read_table(path,columns=['entity_id','row_in_entity','crystal_diameter'],filters=filters).to_pandas()
            for path in paths if Path(path).is_file()]
    raw=pd.concat(pieces,ignore_index=True)
    if raw[['entity_id','row_in_entity']].duplicated().any():
        groups=raw.groupby(['entity_id','row_in_entity']).crystal_diameter.nunique()
        if (groups>1).any():raise ProtocolError('Conflicting raw observations across partitions')
        raw=raw.drop_duplicates(['entity_id','row_in_entity'])
    series=raw.set_index(['entity_id','row_in_entity']).crystal_diameter
    current=pd.MultiIndex.from_arrays([samples.entity_id,samples.origin-1])
    target=pd.MultiIndex.from_arrays([samples.entity_id,samples.origin])
    result=samples.copy();result['y_true']=series.reindex(target).to_numpy(float)-series.reindex(current).to_numpy(float)
    result['sample_id']=result.view_sample_id.astype(str)
    if not np.isfinite(result.y_true).all():raise ProtocolError('Missing lawful CZ target observations')
    return result


def load_development(context):
    mode=context.get('mode')
    if mode not in {'formal_development','solver_smoke','full_registry_smoke','accelerated_benchmark'}:
        raise ProtocolError('Training labels require an explicitly authorized fitting mode')
    if mode=='formal_development':require_execution(context['protocol'],'formal_development')
    elif mode=='accelerated_benchmark' and context['protocol']['execution_authorization'].get('development_only_benchmark') is not True:
        raise ProtocolError('Development benchmark is not authorized')
    elif context['protocol']['execution_authorization'].get('development_only_solver_smoke') is not True:
        raise ProtocolError('Development solver smoke is not authorized')
    shared=Path(context['shared']);manifest=context['C1_manifest']
    source=shared/'base_data/cz_czochralski/train.parquet'
    if file_sha(source)!=manifest['source_base_sha256']:raise ProtocolError('C1 source base changed')
    samples=pd.read_parquet(shared/SAMPLE_RELATIVE/'train.parquet')
    if set(samples.split)!={'train'}:raise ProtocolError('Expected source-development anchors')
    if mode in {'solver_smoke','full_registry_smoke'}:samples=samples.iloc[:int(context['protocol'][mode]['prefix_rows'])].copy()
    validate_partition(samples)
    return _labels(samples,[source])


def load_inference(context,operation):
    require_execution(context['protocol'],operation)
    if context.get('mode')!='inference_only' or context.get('deployment_checkpoint_verified') is not True or context.get('inference_operation')!=operation:
        raise ProtocolError('Inference labels require the verified deployment/test barrier')
    import pyarrow.parquet as pq
    from prism_benchmark.v211_support import SUPPORT_CONTRACT
    from .runtime_seal import verify_c1
    verify_c1(context['shared'],context['registration']['C1_case_bindings'][context['case_id']])
    source=Path(context['C1_manifest']['source_shared'])
    split={'source_validation':'validation','target_test':'test'}[operation]
    relative=context['registration']['C1_case_bindings'][context['case_id']]['maximal_inference_metadata'][split]['relative']
    path=Path(context['release_dir'])/relative
    binding=context['registration']['C1_case_bindings'][context['case_id']]['maximal_inference_metadata'][split]
    if file_sha(path)!=binding['sha256']:raise ProtocolError('Frozen maximal inference metadata drift')
    columns=[c for c in pq.ParquetFile(path).schema.names if c not in {'y_true','y_pred','residual'}]
    samples=pq.read_table(path,columns=columns).to_pandas()
    samples['target_stop_exclusive']=samples.origin+1
    samples['latest_available_target_index']=samples.origin-1
    samples['causal_history_floor']=0
    samples['sample_support_contract']=SUPPORT_CONTRACT
    samples['anchor_history_steps']=1
    partitions=['train','validation'] if split=='validation' else ['train','validation','test']
    base=[source/'base_data/cz_czochralski'/(s+'.parquet') for s in partitions]
    ev=_labels(samples,base)
    # No pre-score target observations are materialized as A initialization.
    observations=ev.iloc[:0].copy()
    return ev,observations,source,split
