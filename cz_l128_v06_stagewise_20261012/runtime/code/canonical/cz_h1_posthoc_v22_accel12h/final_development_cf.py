"""Complete realized-policy prequential replay, including legal N/A identity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .invariants import ProtocolError,NumericallyInvalid,digest,dump,frame_hash,file_sha,prediction_hash
from .temporal_support import final_replay_blocks


def replay(pipe,upstream,frame,selection_freeze,*,frozen_parent=None):
    if pipe.mode not in {'solver_smoke','full_registry_smoke','accelerated_benchmark','formal_development'}:
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
    from .replay_policy import ReplaySession,routes_from_state,cadence
    active=pipe.replay_session
    if active is not None:routes=active.routes
    elif frozen_parent is not None:routes=routes_from_state(frozen_parent)
    else:raise ProtocolError('Replay requires already frozen global parent routes')
    needed=['K','C']+(['W'] if upstream=='W' else [])
    routes={k:routes[k] for k in needed}
    binding=dict(role='FINAL_DEVELOPMENT_CF_REPLAY',policy_id=pipe.policy_id,
        partition_id=pipe.partition_id(frame),input_frame_hash=frame_hash(frame),
        selected_recipe=selection_freeze,contract=contract,frozen_routes=routes)
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
    provenance=[];c_history=[]
    with ReplaySession(pipe,routes) as session:
      for index,fi,ei,entity,local_index,reselect in cadence(frame):
          block_key=digest(dict(binding=binding,index=index,fit_hash=frame_hash(frame.iloc[fi]),eval_hash=frame_hash(frame.iloc[ei])))
          checkpoints=getattr(pipe,'checkpoints',None)

          fit=frame.iloc[fi].copy()
          ev=frame.iloc[ei].copy()
          c_source=pd.concat(c_history,ignore_index=True) if c_history else None
          if c_source is not None:c_source=c_source.loc[c_source.sample_id.isin(fit.sample_id)].copy()
          parent=session.at_block(upstream,fit,entity=entity,local_index=local_index,reselect=reselect,refit_source=c_source)
          if parent['partition_id']!=pipe.partition_id(fit) or parent['admission_policy_id']!=pipe.policy_id:
              raise ProtocolError('Replay uses another prefix/policy realization')
          if parent['stage']!=upstream or parent['routing_status']=='NUMERICALLY_INVALID':
              raise NumericallyInvalid('Numerically invalid replay parent is never an identity shortcut')
          if upstream=='W':
              c_parent=parent['parent'];cp=pipe.emit(c_parent,ev)
              cs=ev.copy();cs['parent_prediction']=cp;cs['residual']=cs.y_true.to_numpy(float)-cp
              cs['residual_source_role']='CROSS_FITTED_PREQUENTIAL'
              q,native,_=pipe.response_coordinate(c_parent,ev);cs['response_coordinate']=q;cs['coordinate_available']=native
              overlap,_=pipe.overlap_components(c_parent,ev)
              for column in overlap:cs[column]=overlap[column].to_numpy()
              c_history.append(cs)
          block_key=digest(dict(policy=pipe.policy_id,fit_hash=frame_hash(fit),score_hash=frame_hash(ev),parent_id=parent['partition_model_id'],contract=contract))
          cached=checkpoints.load('FINAL_CF_BLOCK',block_key) if checkpoints else None
          if cached is not None:
              result.append(pd.read_parquet(cached['path']));provenance.append(cached['provenance']);continue
          prediction=np.asarray(pipe.emit(parent,ev.drop(columns=['y_true'])),dtype=np.float64)
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
          provenance.append(dict(block=index,entity_local_block=local_index,reselection_checkpoint=reselect,parameter_refit=True,global_route_re_admitted=False,fit_rows=len(fi),replay_rows=len(ei),
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
              block_path=Path(pipe.root)/'DAG_CF_BLOCKS'/(block_key+'.parquet');block_path.parent.mkdir(parents=True,exist_ok=True)
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
    proof=json.loads(json.dumps(proof,default=str))
    dump(metadata,proof)
    return source,proof,str(path)


