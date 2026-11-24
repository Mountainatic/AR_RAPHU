"""Canonical paired predictions and realization audits; no selection or fitting."""
from pathlib import Path
import numpy as np
from .invariants import ProtocolError,digest,frame_hash,prediction_hash,dump,file_sha


def _prediction(values,n,name):
    values=np.asarray(values,dtype=np.float64)
    if values.shape!=(n,) or not np.isfinite(values).all():
        raise ProtocolError('Incomplete/nonfinite canonical prediction: '+name)
    return values


def paired_pointwise(samples,*,stage,kind,parent,model_child,deployed,row_domain,
                     routing_status,policy_id,partition_id,parameters,
                     state_provenance=None,fold=None,identity_reason=None):
    """Keep the entire registered paired support, including identity/N/A rows.

    MODEL_child is the domain-completed candidate prediction before admission;
    deployed follows the frozen route. N/A is determined by metadata/history,
    never by the prediction gain. Residuals here are reporting outputs only.
    """
    if stage not in {'K','C','W','A'} or kind not in {'outer_oof','source_validation','target_test'}:
        raise ProtocolError('Unknown canonical pointwise stage or role')
    required={'sample_id','entity_id','origin','y_true'}
    if not required.issubset(samples.columns) or samples['sample_id'].duplicated().any():
        raise ProtocolError('Canonical pointwise requires unique explicit sample identities')
    n=len(samples)
    if not n:
        raise ProtocolError('An empty support is not an executed pointwise result')
    parent=_prediction(parent,n,'parent')
    model_child=_prediction(model_child,n,'MODEL child')
    deployed=_prediction(deployed,n,'deployed')
    target=_prediction(samples['y_true'],n,'target')
    mask=np.asarray(row_domain)
    if mask.shape!=(n,) or mask.dtype!=np.dtype(bool):
        raise ProtocolError('Row-domain mask must explicitly cover every paired row')
    if kind=='outer_oof' and routing_status is not None:
        raise ProtocolError('Outer MODEL cannot claim an aggregate stage route')
    if kind!='outer_oof' and routing_status not in {'ADMITTED','BYPASSED','NOT_APPLICABLE'}:
        raise ProtocolError('Numerical failure cannot be exported as a valid route')
    if not np.array_equal(model_child[~mask],parent[~mask]):
        raise ProtocolError('N/A candidate rows must retain their parent exactly')
    if routing_status=='NOT_APPLICABLE' and mask.any():
        raise ProtocolError('A wholly N/A route cannot claim available MODEL rows')
    expected=model_child if kind=='outer_oof' or routing_status=='ADMITTED' else parent
    if not np.array_equal(deployed,expected):
        raise ProtocolError('Pointwise deployment disagrees with the frozen route')
    provenance=state_provenance or {}
    if stage=='A' and not {'state_role','upstream_deployment_model_id',
                            'mature_state_hash','mature_feature_hash'}.issubset(provenance):
        raise ProtocolError('A inference requires deployed-W mature-state provenance')
    if stage=='A' and provenance['state_role']!='A_INFERENCE_DEPLOYED_W_RESIDUAL':
        raise ProtocolError('A inference export cannot be seeded from training CF state')
    support_hash=frame_hash(samples)
    realization=dict(schema='CZ_CANONICAL_PAIRED_REALIZATION_V1',stage=stage,kind=kind,
        policy_id=policy_id,partition_id=partition_id,fold=fold,
        global_routing_status=routing_status,parameters=parameters,
        model_domain_status='MODEL_APPLICABLE' if mask.any() else 'MODEL_NOT_APPLICABLE_IN_REALIZATION',
        identity_reason=identity_reason,
        evaluation_support_hash=support_hash,rows=n,row_domain_hash=digest(mask.tolist()),
        parent_prediction_hash=prediction_hash(parent),
        MODEL_child_prediction_hash=prediction_hash(model_child),
        deployed_prediction_hash=prediction_hash(deployed),state_provenance=provenance)
    realization_id=digest(realization)
    id_name='outer_realization_id' if kind=='outer_oof' else 'inference_realization_id'
    result=samples.copy(deep=True)
    result['parent_prediction']=parent
    result['MODEL_child_prediction']=model_child
    result['deployed_prediction']=deployed
    result['residual']=target-deployed
    result['row_domain']=np.where(mask,'AVAILABLE','NOT_APPLICABLE_IDENTITY')
    result['model_domain_status']=np.where(mask,'MODEL_APPLICABLE','MODEL_NOT_APPLICABLE_IN_REALIZATION')
    result['identity_reason']=np.where(mask,None,identity_reason or 'ROW_MODEL_DOMAIN_UNAVAILABLE')
    for name,value in dict(stage=stage,pointwise_role=kind,global_routing_status=routing_status,
        admission_policy_id=policy_id,partition_id=partition_id,
        scientific_provenance='POST_HOC_PROTOCOL_REVISION').items():
        result[name]=value
    result[id_name]=realization_id
    if kind!='outer_oof':result['deployment_model_id']=parameters['deployment_model_id']
    if fold is not None:
        result['fold']=fold
    return result,dict(**realization,**{id_name:realization_id},pointwise_content_hash=frame_hash(result))


def a_training_prediction_audit(samples,prediction,*,cf_state,feature_audit,
                                policy_id,partition_id,parameters):
    """Bind final A predictions to the exact CF training source and mature state."""
    cf_state.validate(expected_role='A_TRAINING_CROSS_FITTED_W_RESIDUAL',
                      upstream_id=cf_state.upstream_id)
    required={'source_content_hash','mature_feature_hash','complete_rows','sample_support_hash'}
    if not required.issubset(feature_audit):
        raise ProtocolError('A training audit lacks the actual complete-history state')
    if feature_audit['source_content_hash']!=cf_state.content_hash:
        raise ProtocolError('A training audit belongs to another CF residual source')
    if feature_audit['sample_support_hash']!=frame_hash(samples):
        raise ProtocolError('A training features belong to another sample support')
    values=_prediction(prediction,len(samples),'A training prediction')
    body=dict(schema='CZ_CANONICAL_A_TRAINING_PREDICTION_AUDIT_V1',
        admission_policy_id=policy_id,partition_id=partition_id,parameters=parameters,
        training_support_hash=frame_hash(samples),training_rows=len(samples),
        CF_source_hash=cf_state.content_hash,mature_state_id=cf_state.state_id,
        mature_feature_hash=feature_audit['mature_feature_hash'],
        complete_history_rows=feature_audit['complete_rows'],
        training_prediction_hash=prediction_hash(values))
    return dict(**body,audit_id=digest(body))


def export_pointwise(path,frame,audit,*,resume=False):
    """Write only a newly executed canonical result, with an immutable sidecar."""
    path=Path(path)
    sidecar=path.with_suffix(path.suffix+'.manifest.json')
    if sidecar.exists():
        if not resume:raise ProtocolError('Canonical pointwise exports are immutable')
        import json
        previous=json.loads(sidecar.read_text())
        if previous['pointwise_content_hash']!=frame_hash(frame) or file_sha(path)!=previous['output_sha256']:
            raise ProtocolError('Canonical pointwise exports are immutable')
        return previous
    if frame_hash(frame)!=audit['pointwise_content_hash']:
        raise ProtocolError('Canonical pointwise changed after realization audit')
    if path.suffix not in {'.parquet','.csv'}:
        raise ProtocolError('Pointwise output must be parquet or round-trip FP64 CSV')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    if path.suffix=='.parquet':
        frame.to_parquet(temporary,index=False)
    else:
        frame.to_csv(temporary,index=False,float_format='%.17g')
    temporary.replace(path)
    manifest=dict(**audit,output_sha256=file_sha(path),
        current_canonical_result=True,historical_R2_substitution=False,
        selection_authority=False)
    dump(sidecar,manifest)
    return manifest
