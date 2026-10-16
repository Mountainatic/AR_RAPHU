import copy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.pointwise import paired_pointwise,export_pointwise,a_training_prediction_audit
from canonical.cz_h1_posthoc_v22.a_state import build_a_training_state_from_cf_residual,feature_rows,TRAINING_CF
from canonical.cz_h1_posthoc_v22.invariants import ProtocolError,file_sha


def samples(n=12):
    origin=np.arange(1,n+1)
    return pd.DataFrame(dict(sample_id=['s'+str(i) for i in origin],entity_id='e',
        origin=origin,target_stop_exclusive=origin+1,latest_available_target_index=origin-1,
        y_true=origin*.13))


def result(**overrides):
    frame=samples(4)
    arguments=dict(stage='W',kind='outer_oof',parent=np.zeros(4),
        model_child=np.array([0.,.2,.3,0.]),deployed=np.array([0.,.2,.3,0.]),
        row_domain=np.array([False,True,True,False]),routing_status='ADMITTED',
        policy_id='frozen',partition_id='dev',parameters={'contract':{'beta':[.2]}},fold=0)
    arguments.update(overrides)
    return paired_pointwise(frame,**arguments)


def test_na_rows_retained_in_all_predictions_and_support():
    frame,audit=result()
    assert len(frame)==audit['rows']==4
    assert list(frame.row_domain)==['NOT_APPLICABLE_IDENTITY','AVAILABLE','AVAILABLE','NOT_APPLICABLE_IDENTITY']
    assert np.array_equal(frame.residual,frame.y_true-frame.deployed_prediction)
    assert frame.outer_realization_id.nunique()==1
    with pytest.raises(ProtocolError,match='retain their parent'):
        result(model_child=np.ones(4))


def test_realization_id_binds_parameters_route_support_and_predictions():
    _,reference=result()
    variations=[dict(parameters={'contract':{'beta':[.3]}}),dict(policy_id='changed'),
        dict(partition_id='changed'),dict(fold=1),
        dict(model_child=np.array([0.,.1,.3,0.]),deployed=np.array([0.,.1,.3,0.])),
        dict(routing_status='BYPASSED',deployed=np.zeros(4))]
    for variation in variations:
        _,audit=result(**variation)
        assert audit['outer_realization_id']!=reference['outer_realization_id']


def test_route_and_numerical_failure_cannot_be_mislabeled():
    with pytest.raises(ProtocolError,match='frozen route'):
        result(routing_status='BYPASSED')
    with pytest.raises(ProtocolError,match='Numerical failure'):
        result(routing_status='NUMERICALLY_INVALID')
    with pytest.raises(ProtocolError,match='wholly N/A'):
        result(routing_status='NOT_APPLICABLE',deployed=np.zeros(4))


def test_a_inference_export_requires_deployed_w_state_not_cf():
    with pytest.raises(ProtocolError,match='mature-state'):
        result(stage='A')
    state=dict(state_role='A_INFERENCE_DEPLOYED_W_RESIDUAL',
        upstream_deployment_model_id='W-deploy',mature_state_hash='state',mature_feature_hash='features')
    _,audit=result(stage='A',state_provenance=state)
    altered=copy.deepcopy(state);altered['mature_state_hash']='another-state'
    _,second=result(stage='A',state_provenance=altered)
    assert audit['outer_realization_id']!=second['outer_realization_id']
    state['state_role']=TRAINING_CF
    with pytest.raises(ProtocolError,match='training CF'):
        result(stage='A',state_provenance=state)


def test_a_training_prediction_audit_binds_actual_cf_features():
    frame=samples()
    source=frame.assign(residual=frame.y_true,residual_source_role='CROSS_FITTED_PREQUENTIAL')
    state=build_a_training_state_from_cf_residual(source,policy_id='W-learning',h_steps=1)
    _,mask,features=feature_rows(frame,state,(1,8),8,0.,expected_role=TRAINING_CF)
    audit=a_training_prediction_audit(frame,np.zeros(len(frame)),cf_state=state,
        feature_audit=features,policy_id='A-learning',partition_id='dev',parameters={'history':8})
    assert audit['complete_history_rows']==mask.sum()==4
    assert audit['CF_source_hash']==state.content_hash and audit['mature_state_id']==state.state_id
    corrupted=dict(features,source_content_hash='foreign')
    with pytest.raises(ProtocolError,match='another CF'):
        a_training_prediction_audit(frame,np.zeros(len(frame)),cf_state=state,
            feature_audit=corrupted,policy_id='A-learning',partition_id='dev',parameters={})


def test_export_is_current_immutable_and_bound_to_frame(tmp_path):
    frame,audit=result()
    path=tmp_path/'canonical_W.csv'
    manifest=export_pointwise(path,frame,audit)
    restored=pd.read_csv(path,float_precision='round_trip')
    assert np.array_equal(restored.deployed_prediction,frame.deployed_prediction)
    assert manifest['output_sha256']==file_sha(path)
    assert not manifest['historical_R2_substitution'] and not manifest['selection_authority']
    with pytest.raises(ProtocolError,match='immutable'):export_pointwise(path,frame,audit)
    with pytest.raises(ProtocolError,match='changed after'):
        export_pointwise(tmp_path/'changed.csv',frame.assign(residual=0.),audit)
