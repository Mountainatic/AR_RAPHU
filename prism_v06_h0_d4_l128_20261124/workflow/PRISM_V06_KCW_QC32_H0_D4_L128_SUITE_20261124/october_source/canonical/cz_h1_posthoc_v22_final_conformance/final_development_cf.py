"""Complete realized-policy prequential replay, including legal N/A identity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .invariants import ProtocolError,NumericallyInvalid,digest,dump,frame_hash,file_sha,prediction_hash
from .temporal_support import final_replay_blocks


def replay(pipe,upstream,frame,selection_freeze):
    if pipe.mode not in {'solver_smoke','full_registry_smoke','formal_development'}:
        raise ProtocolError('This execution mode cannot generate training residuals')
    if upstream not in {'C','W'}:
        raise ProtocolError('Final replay upstream must be C or W')
    if selection_freeze.get('policy_id')!=pipe.policy_id or selection_freeze.get('partition_id')!=pipe.partition_id(frame):
        raise ProtocolError('Selected recipe freeze belongs to another policy/partition')
    if selection_freeze.get('upstream_stage')!=upstream or 'selected_spec' not in selection_freeze:
        raise ProtocolError('Final replay needs the previously selected residual recipe')
    contract=pipe.protocol['final_development_cf']
    if contract['block_rows']!=320 or contract['support_rule']!='PAST_ONLY_REALIZED_ROUTE_WITH_NA_IDENTITY':
        raise ProtocolError('Final replay contract changed')
    binding=dict(role='FINAL_DEVELOPMENT_CF_REPLAY',policy_id=pipe.policy_id,
        partition_id=pipe.partition_id(frame),input_frame_hash=frame_hash(frame),
        selected_recipe=selection_freeze,contract=contract)
    key=digest(binding)
    path=Path(pipe.root)/'final_development_streams'/upstream/key/'prequential.parquet'
    metadata=path.with_suffix('.json')
    if metadata.exists():
        if not path.exists() or not metadata.exists():
            raise ProtocolError('Incomplete final replay cache')
        proof=json.loads(metadata.read_text(encoding='utf-8'))
        if proof['binding']!=json.loads(json.dumps(binding,default=str)) or proof['file_sha256']!=file_sha(path):
            raise ProtocolError('Final replay cache binding/hash changed')
        source=pd.read_parquet(path)
        if frame_hash(source)!=proof['source_frame_hash']:
            raise ProtocolError('Final replay residual/source content changed')
        return source,proof,str(path)
    blocks=final_replay_blocks(frame,contract['block_rows'])
    result=[]
    provenance=[]
    for index,(fi,ei) in enumerate(blocks):
        block_key=digest(dict(binding=binding,index=index,fit_hash=frame_hash(frame.iloc[fi]),eval_hash=frame_hash(frame.iloc[ei])))
        checkpoints=getattr(pipe,'checkpoints',None)
        cached=checkpoints.load('FINAL_CF_BLOCK',block_key) if checkpoints else None
        if cached is not None:
            result.append(pd.read_parquet(cached['path']));provenance.append(cached['provenance']);continue
        fit=frame.iloc[fi].copy()
        ev=frame.iloc[ei].copy()
        parent=pipe.learn(upstream,fit)
        if parent['partition_id']!=pipe.partition_id(fit) or parent['admission_policy_id']!=pipe.policy_id:
            raise ProtocolError('Replay uses another prefix/policy realization')
        if parent['stage']!=upstream or parent['routing_status']=='NUMERICALLY_INVALID':
            raise NumericallyInvalid('Numerically invalid replay parent is never an identity shortcut')
        prediction=np.asarray(pipe.emit(parent,ev),dtype=np.float64)
        if prediction.shape!=(len(ev),) or not np.isfinite(prediction).all():
            raise NumericallyInvalid('Invalid CF upstream predictions')
        source=ev.copy()
        source['parent_prediction']=prediction
        source['residual']=source.y_true.to_numpy(np.float64)-prediction
        source['residual_source_role']='CROSS_FITTED_PREQUENTIAL'
        source['replay_block']=index
        source['upstream_realization_id']=parent.get('partition_model_id',parent.get('deployment_model_id'))
        if upstream=='C' and hasattr(pipe,'response_coordinate'):
            q,native,coordinate=pipe.response_coordinate(parent,ev)
            source['response_coordinate']=q;source['coordinate_available']=native
        overlap,component_domains=pipe.overlap_components(parent,ev)
        for column in overlap:
            source[column]=overlap[column].to_numpy()
        if upstream=='W':
            source['up_w']=prediction-pipe.emit(parent['parent'],ev)
        if not np.isfinite(source[['residual',*[c for c in source if c.startswith('up_')]]].to_numpy(float)).all():
            raise NumericallyInvalid('Invalid CF residual or overlap coordinates')
        provenance.append(dict(block=index,fit_rows=len(fi),replay_rows=len(ei),
            fit_support_hash=pipe.partition_id(fit),replay_support_hash=pipe.partition_id(ev),
            fit_last_partition_position=None if not len(fi) else int(fi[-1]),
            replay_first_partition_position=int(ei[0]),
            upstream_policy_id=pipe.policy_id,upstream_partition_model_id=source.upstream_realization_id.iloc[0],
            upstream_route=parent['routing_status'],identity_parent_used=parent['routing_status']=='NOT_APPLICABLE',
            prediction_hash=prediction_hash(prediction),residual_hash=prediction_hash(source.residual),
            overlap_domains=component_domains,target_used_to_fit_own_predictor=False,
            source_library_lmax_filter=False,strictly_past_raw_support=True))
        result.append(source)
        if checkpoints:
            block_path=path.parent/'blocks'/(str(index)+'.parquet');block_path.parent.mkdir(parents=True,exist_ok=True)
            temporary=block_path.with_suffix('.tmp.parquet');source.to_parquet(temporary,index=False);temporary.replace(block_path)
            checkpoints.save('FINAL_CF_BLOCK',block_key,dict(path=str(block_path.resolve()),provenance=provenance[-1]),[block_path])
    if not result:
        raise ProtocolError('No source-development anchors to replay')
    source=pd.concat(result,ignore_index=True)
    if len(source)!=len(frame) or pipe.partition_id(source)!=pipe.partition_id(frame):
        raise ProtocolError('Realized replay lost or duplicated development anchors')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.parquet')
    source.to_parquet(temporary,index=False)
    temporary.replace(path)
    proof=dict(binding=binding,provenance=provenance,source_frame_hash=frame_hash(source),
        file_sha256=file_sha(path),coverage=dict(partition_rows=len(frame),replay_rows=len(source),
            excluded_rows=0,block_count=len(blocks),block_lengths=[len(ev) for _,ev in blocks],
            first_origin_by_entity={str(entity):int(g.origin.min()) for entity,g in source.groupby('entity_id',sort=False)},
            start_rule='EARLIEST_LEGAL_ANCHOR; EMPTY_OR_NA_PREFIX_PROPAGATES_NEUTRAL_PARENT',
            support_rule=contract['support_rule'],partial_tail_retained=True),
        role='FULL_DEVELOPMENT_CROSS_FITTED_PREQUENTIAL',uses_in_sample_parent_residual=False)
    dump(metadata,proof)
    return source,proof,str(path)
