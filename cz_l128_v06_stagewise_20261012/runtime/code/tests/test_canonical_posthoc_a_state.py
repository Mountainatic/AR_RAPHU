"""Explicit maturity, candidate-native completeness and separate inference state."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.a_state import *
from canonical.cz_h1_posthoc_v22.invariants import ProtocolError


def observations(n=200,h=1):
    origin=np.arange(1,n+1)
    return pd.DataFrame(dict(entity_id='Rod_1_segment_0',origin=origin,
        target_stop_exclusive=origin+h,latest_available_target_index=origin-1,
        y_true=origin.astype(float),residual=origin.astype(float),
        residual_source_role='CROSS_FITTED_PREQUENTIAL'))


def test_h1_newest_residual_is_t_minus_1_and_a8_is_2_to_16_seconds():
    frame=observations();state=build_a_training_state_from_cf_residual(frame,policy_id='policy',h_steps=1)
    row=frame.query('origin==100')
    x,mask,audit=feature_rows(row,state,(1,8),128,0.)
    assert mask.all() and np.array_equal(x[0],np.arange(99,91,-1,dtype=float))
    assert np.array_equal((100-x[0])*2,np.arange(2,18,2))
    assert audit['zero_fill'] is False


def test_own_and_future_residuals_never_change_own_prediction_features():
    frame=observations();row=frame.query('origin==150')
    original=build_a_training_state_from_cf_residual(frame,policy_id='policy',h_steps=1)
    altered=frame.copy();altered.loc[altered.origin>=150,'residual']=1e9
    updated=build_a_training_state_from_cf_residual(altered,policy_id='policy',h_steps=1)
    a,_,_=feature_rows(row,original,(1,128),128,0.)
    b,_,_=feature_rows(row,updated,(1,128),128,0.)
    assert a.shape==(1,128) and np.array_equal(a,b)
    a,_,_=feature_rows(row,original,(1,8),128,0.)
    b,_,_=feature_rows(row,updated,(1,8),128,0.)
    assert np.array_equal(a,b)


def test_missing_lag_removes_native_row_instead_of_centered_zero():
    frame=observations();frame=frame.loc[frame.origin!=95]
    state=build_a_training_state_from_cf_residual(frame,policy_id='p',h_steps=1)
    row=observations().query('origin==100')
    x,mask,_=feature_rows(row,state,(1,8),128,0.)
    assert not mask.any() and x.shape==(0,8)


def test_short_history_has_more_fit_rows_and_scoring_is_exact_intersection():
    frame=observations();state=build_a_training_state_from_cf_residual(frame,policy_id='p',h_steps=1)
    _,short,_=feature_rows(frame,state,(1,8),128,0.)
    _,long,_=feature_rows(frame,state,(1,128),128,0.)
    common=common_scoring_mask(frame,state,[(1,8),(1,128)],128)
    assert short.sum()>long.sum()>0
    assert np.array_equal(common,short&long)
    assert frame.loc[short,'origin'].min()==9
    assert frame.loc[long,'origin'].min()==129


def test_cf_state_cannot_be_used_for_inference_and_source_values_are_deployed_only():
    frame=observations()
    training=build_a_training_state_from_cf_residual(frame,policy_id='cf',h_steps=1)
    with pytest.raises(ProtocolError,match='cannot seed'):
        feature_rows(frame,training,(1,8),128,0.,expected_role=INFERENCE_DEPLOYED)
    inference=build_a_inference_state_from_deployment_residual(frame,np.full(len(frame),10.),deployment_model_id='frozen_W',h_steps=1)
    x,_,_=feature_rows(frame.query('origin==100'),inference,(1,8),128,0.,
        expected_role=INFERENCE_DEPLOYED,upstream_id='frozen_W')
    assert np.array_equal(x[0],np.arange(89,81,-1,dtype=float))
    with pytest.raises(ProtocolError,match='another deployed W'):
        inference.validate(upstream_id='other_W')


def test_entity_reset_does_not_borrow_other_entity_history():
    frame=observations();row=frame.query('origin==100').copy();row.entity_id='Rod_2_segment_0'
    state=build_a_training_state_from_cf_residual(frame,policy_id='p',h_steps=1)
    _,mask,_=feature_rows(row,state,(1,8),128,0.)
    assert not mask.any()


def test_target_end_metadata_and_bound_state_tampering_are_rejected():
    frame=observations();bad=frame.copy();bad.target_stop_exclusive+=1
    with pytest.raises(ProtocolError,match='target end'):
        build_a_training_state_from_cf_residual(bad,policy_id='p',h_steps=1)
    state=build_a_training_state_from_cf_residual(frame,policy_id='p',h_steps=1)
    state.frame.loc[0,'residual']+=1
    with pytest.raises(ProtocolError,match='changed after binding'):state.validate()
