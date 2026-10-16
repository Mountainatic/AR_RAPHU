"""Regression for the real 15k batch-versus-row identity failure."""
import copy
import numpy as np
import pandas as pd
import pytest
from canonical.cz_h1_posthoc_v22_accel12h.canonical_engine import CanonicalPipeline
from canonical.cz_h1_posthoc_v22_accel12h.invariants import ProtocolError
from canonical.cz_h1_posthoc_v22_accel12h.runtime_budget import launch_gate

def test_online_identity_reuses_parent_without_single_row_recomputation():
    p=object.__new__(CanonicalPipeline)
    origins=np.arange(100,110)
    T=pd.DataFrame(dict(entity_id='segment',origin=origins,target_stop_exclusive=origins+1,
        latest_available_target_index=origins-1,y_true=1.,sample_id=[str(x) for x in origins]))
    calls=[]
    def emit(state,rows):
        calls.append(len(rows))
        if len(rows)==1:pytest.fail('Single-row parent recomputed')
        return np.full(len(rows),np.nextafter(1.,2.))
    p.emit=emit
    model=dict(stage='A',domain=False,parent=dict(partition_model_id='W'))
    baseline=emit(model['parent'],T)
    output,mask,proof=p.online_a(model,T,parent_prediction=baseline)
    np.testing.assert_array_equal(output,baseline)
    assert not mask.any() and calls==[10] and proof['parent_prediction_reused_exactly']

def test_bound_parent_is_also_the_mature_residual_baseline():
    p=object.__new__(CanonicalPipeline)
    origins=np.arange(100,106)
    T=pd.DataFrame(dict(entity_id='segment',origin=origins,target_stop_exclusive=origins+1,
        latest_available_target_index=origins-1,y_true=np.arange(6,dtype=float),sample_id=[str(x) for x in origins]))
    p.emit=lambda *args:pytest.fail('Bound upstream baseline was recomputed')
    baseline=np.full(len(T),np.nextafter(1.,2.))
    def predict(state,row,*,a_state,parent_prediction):
        source=a_state.frame
        if len(source):np.testing.assert_array_equal(source.residual,source.y_true.to_numpy()-baseline[:len(source)])
        return parent_prediction.copy(),np.zeros(len(row),bool),{}
    p.predict=predict
    out,_,_=p.online_a(dict(parent=dict(partition_model_id='W')),T,parent_prediction=baseline)
    np.testing.assert_array_equal(out,baseline)

def receipt():
    return dict(schema='CZ_H1_EXPLICIT_USER_LAUNCH_OVERRIDE_V1',authorization_source='HUMAN_USER',
        user_instruction='放宽快点给我开始正式训练，你耽误我太久了',formal_development_training=True,
        source_validation_reporting=False,target_test_inference=False,waive_pending_benchmark_completion=True,
        waive_prelaunch_runtime_forecast=True,scientific_rules_unchanged=True,runtime_stop_limits_unchanged=True)

def test_launch_override_is_explicit_and_preserves_budget():
    protocol=dict(runtime_budget=dict(launch_gate_seconds=36000,soft_stop_seconds=39600,hard_stop_seconds=41400),
        launch_gate_user_override=receipt())
    before=copy.deepcopy(protocol)
    launch_gate({},protocol)
    assert before==protocol
    protocol.pop('launch_gate_user_override')
    with pytest.raises(ProtocolError):launch_gate({},protocol)

@pytest.mark.parametrize('field',['authorization_source','formal_development_training','target_test_inference',
    'runtime_stop_limits_unchanged','scientific_rules_unchanged','user_instruction'])
def test_launch_override_does_not_accept_incomplete_receipt(field):
    override=receipt();override.pop(field)
    with pytest.raises(ProtocolError):launch_gate({},dict(launch_gate_user_override=override))
