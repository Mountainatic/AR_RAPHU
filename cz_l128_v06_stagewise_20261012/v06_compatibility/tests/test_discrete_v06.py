"""Fast read-only math, support, real numerical prediction regressions; no fit."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from dataclasses import FrozenInstanceError
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
AUTHORITY=Path(os.environ.get('CZ_AUTHORITY',ROOT.parents[1]/'deliverables/CZ_H1_POSTHOC_CANONICAL_FINAL_CONFORMANCE_CODE_REVIEW_20261004/code/PRISM_INDUSTRIAL_BENCHMARK_V1'))
sys.path.insert(0,str(AUTHORITY/'src'))
from canonical.cz_crossrod_l128_20261008.discrete_observation import DiscreteObservationSpec,optional_observation_spec
from canonical.cz_crossrod_l128_20261008.discrete_audit import CZHeadAuditConfig,audit_target_contract,observation_rows
from canonical.cz_crossrod_l128_20261008.discrete_diagnostics import (quantize_half_up,spectrum_acf,
    segmented_diagnostics,grid_candidate_evidence,synthetic_scenes)


def spec(**kwargs):
    return DiscreteObservationSpec(sample_period_seconds=2.,head_contract_id='CZ_H1_L128',**kwargs)


def observed(n=32):
    return pd.DataFrame(dict(entity_id='rod',segment_id='s',split_id='development',
        raw_index=np.arange(n),recorded_level=np.arange(n,dtype=float)**2/100))


def test_optional_contract_deep_immutable_json():
    config={'old_key':7};original=copy.deepcopy(config)
    assert optional_observation_spec(config) is None and config==original
    rules={'maximum_gap_seconds':3.,'source':{'notes':['recorded']}}
    s=spec(gap_rule=rules)
    rules['source']['notes'][0]='changed'
    assert s.gap_rule['source']['notes']==('recorded',)
    with pytest.raises(TypeError):s.gap_rule['maximum_gap_seconds']=4
    with pytest.raises(FrozenInstanceError):s.sample_period_seconds=4
    assert DiscreteObservationSpec.from_json(s.to_json()).to_dict()==s.to_dict()
    with pytest.raises(ValueError):spec(output_record_grid_candidate=.01,grid_evidence_status='ADC')
    with pytest.raises(ValueError):spec(grid_evidence_status='instrument_verified')
    with pytest.raises(ValueError):spec(output_record_grid_candidate=-1)


@pytest.mark.parametrize('h',[1,4])
@pytest.mark.parametrize('direction',['Rod_1_to_Rod_2','Rod_2_to_Rod_1'])
def test_target_adapter_matches_authority_builder(h,direction,monkeypatch):
    from prism_benchmark import cz_l256_nowcast as legacy
    monkeypatch.setattr(legacy,'HISTORY_STEPS',128)
    frame=pd.DataFrame({'crystal_diameter':np.linspace(.17,3.21,150)**2})
    raw=legacy.RawSegment('Rod_1','rod',0,len(frame),frame)
    actual=legacy._sample_rows([raw],direction=direction,split='train',h_steps=h)
    samples=actual.loc[actual.information_set=='dynamic'].copy()
    data=pd.DataFrame(dict(entity_id='rod',segment_id='s',split_id='train',raw_index=np.arange(len(frame)),recorded_level=frame.crystal_diameter))
    out=audit_target_contract(data,CZHeadAuditConfig(f'CZ_H{h}_L128',h),spec(),samples=samples)
    assert out.legal_support.all()
    np.testing.assert_array_equal(out.registered_target, samples.y_true)
    np.testing.assert_array_equal(out.target_start_index,samples.target_start)
    np.testing.assert_array_equal(out.target_end_index,samples.target_stop_exclusive-1)
    assert (out.mother_window_h==h-1).all()
    assert (out.label_maturity_index==out.target_end_index).all()
    np.testing.assert_array_equal(out.feedback_available_time,out.target_end_index*2.)
    assert np.array_equal(out.anchor_end_index,samples.origin-1)


def test_manual_example_h1_h4():
    data=observed(12)
    samples=pd.DataFrame({'entity_id':['rod'],'origin':[5]})
    h1=audit_target_contract(data,CZHeadAuditConfig('H1',1,4),spec(),samples=samples).iloc[0]
    h4=audit_target_contract(data,CZHeadAuditConfig('H4',4,4),spec(),samples=samples).iloc[0]
    assert (h1.anchor_end_index,h1.target_start_index,h1.mother_window_h)==(4,5,0)
    assert (h4.anchor_end_index,h4.target_start_index,h4.mother_window_h)==(4,8,3)
    assert h1.registered_target==data.recorded_level[5]-data.recorded_level[4]
    assert h4.registered_target==data.recorded_level[8]-data.recorded_level[4]


def test_gap_short_split_purge_entity_and_scenario_are_report_only():
    data=observed();data['timestamp']=data.raw_index*2.;data.loc[16:,'timestamp']+=10
    data.loc[20:,'split_id']='validation'
    samples=pd.DataFrame({'entity_id':['rod']*5,'origin':[2,17,21,25,27],
        'purge_valid':[True,True,True,False,True],
        'expected_entity_id':['rod']*4+['other'],
        'requires_future_controls':[False]*4+[True], 'controls_available_until':[1,16,20,24,26]})
    untouched=data.copy(deep=True);sample_copy=samples.copy(deep=True)
    out=audit_target_contract(data,CZHeadAuditConfig('H4',4,8),spec(timestamp_field='timestamp'),samples=samples)
    assert len(out)==len(samples)
    for row,reason in [(0,'SHORT_SEGMENT'),(1,'CROSS_SEGMENT_OR_GAP'),(2,'CROSS_SPLIT'),(3,'PURGE'),(4,'UNKNOWN_FUTURE_CONTROL')]:
        assert reason in out.invalid_reason.iloc[row]
    assert 'CROSS_ENTITY' in out.invalid_reason.iloc[4]
    assert (out.audit_status=='POTENTIAL_LEAK_OR_INVALID_SUPPORT').all()
    pd.testing.assert_frame_equal(data,untouched);pd.testing.assert_frame_equal(samples,sample_copy)


def test_future_predictor_and_anchor_maturity_are_detected():
    sample=pd.DataFrame(dict(entity_id=['rod'],origin=[12],last_input_available_index=[12],last_output_available_index=[10]))
    out=audit_target_contract(observed(),CZHeadAuditConfig('H1',1,4),spec(),samples=sample)
    assert 'ANCHOR_NOT_AVAILABLE' in out.invalid_reason[0] and 'FUTURE_PREDICTOR_LEAK' in out.invalid_reason[0]


def test_diagnostics_never_join_timestamp_or_value_gaps():
    data=observed();data['timestamp']=data.raw_index*2.;data.loc[16:,'timestamp']+=10
    data.loc[25,'recorded_level']=np.nan
    out=segmented_diagnostics(data,spec(timestamp_field='timestamp'))
    spans=[(s['raw_index_start'],s['raw_index_end']) for s,_,_ in out]
    assert spans==[(0,15),(16,24),(26,31)]
    assert all(s['N']<=16 for s,_,_ in out)
    assert spectrum_acf([1.,2.,3.],2.)[0]['status']=='SHORT_SEGMENT'


def test_grid_evidence_never_claims_instrument_and_no_prediction_rounding():
    evidence=grid_candidate_evidence([.01,.02,.11])
    assert evidence['fraction_on_candidate_grid']==1.
    assert evidence['grid_evidence_status']=='observed_grid_candidate' and not evidence['instrument_verified']
    np.testing.assert_allclose(quantize_half_up([-.005,.005,.015],.01),[0,.01,.02],atol=1e-15)


@pytest.mark.parametrize('p,q',[(1,7),(2,7),(3,8)])
def test_quantized_ramp_translation_period(p,q):
    ramp=.1234+.01*(p/q)*np.arange(8192)
    diff=np.diff(quantize_half_up(ramp,.01))
    np.testing.assert_allclose(diff[q:],diff[:-q],atol=2e-13,rtol=0)


@pytest.mark.parametrize('seed',[20261010,20261011,20261012])
def test_iid_difference_lag_one_statistical_tolerance(seed):
    v=np.random.default_rng(seed).normal(size=100000)
    empirical=np.corrcoef(np.diff(v)[:-1],np.diff(v)[1:])[0,1]
    assert abs(empirical+.5)<.015 # statistical tolerance; not equality


def test_difference_transfer_and_alias():
    omega=np.random.default_rng(1).uniform(-np.pi,np.pi,1000)
    np.testing.assert_allclose(abs(1-np.exp(-1j*omega))**2,4*np.sin(omega/2)**2,atol=2e-15,rtol=2e-15)
    scenes=synthetic_scenes();params,arrays=scenes['S4']
    assert abs(params['alias_Hz']-1/7)<1e-8
    summary,_,_=spectrum_acf(arrays['sampled'],2.)
    assert abs(summary['peak_frequency_Hz']-params['alias_Hz'])<=summary['frequency_resolution_Hz']


def test_all_original_runtime_files_byte_identical_after_new_modules():
    before=ROOT/'baseline_runtime/canonical/cz_crossrod_l128_20261008'
    after=ROOT/'runtime/canonical/cz_crossrod_l128_20261008'
    for p in before.rglob('*'):
        if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc':
            assert p.read_bytes()==(after/p.relative_to(before)).read_bytes(),str(p)


def numerical_fixture(Pipeline,tmp_path):
    """Hand-frozen nonlinear contracts and real BaseAccessor; no fit or mocks."""
    from prism_benchmark import v2_k,v2_urysohn,v2_c,v211_w,v21_a,v211_support
    from prism_benchmark.cpu_data import BaseAccessor
    from canonical.cz_crossrod_l128_20261008.a_state import build_a_training_state_from_cf_residual
    p=object.__new__(Pipeline)
    p.v2_k=v2_k;p.u=v2_urysohn;p.c=v2_c;p.w=v211_w;p.a=v21_a;p.native=v211_support.native_support_mask;p.A_max=8
    n=180;path=tmp_path/'base_data/cz_czochralski/train.parquet';path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(entity_id='rod',row_in_entity=np.arange(n),main_heater_power=np.sin(np.arange(n)/17))).to_parquet(path,index=False)
    p.accessor=BaseAccessor(tmp_path,'cz_czochralski','train',['main_heater_power'])
    origins=np.arange(128,160)
    T=pd.DataFrame(dict(entity_id='rod',origin=origins,history_start=origins-128,history_stop_exclusive=origins,
        causal_history_floor=origins-128,latest_available_target_index=origins-1,
        target_stop_exclusive=origins+1,anchor_history_steps=1,availability_delay_steps=0,y_true=np.cos(origins/17),
        sample_support_contract=v211_support.SUPPORT_CONTRACT,
        sample_id=[str(i) for i in origins]))
    basis=dict(kind='continuous',mean=0.,scale=1.,knots=[-.5,.5],levels=[],feature_mean=[0.,0.,0.])
    r=dict(channel='main_heater_power',profile=[1,4],m_tau=2,
        contract=dict(family='FULL_FINITE_URYSOHN',basis=basis,theta=[[.2,.03,0.],[-.1,.05,0.]],intercept=.017))
    K=dict(stage='K',routing_status='ADMITTED',domain=True,representations=[r],reference_index=0,reference_channel='main_heater_power')
    C=dict(stage='C',routing_status='ADMITTED',domain=True,parent=K,recipe='ADDITIVE_COMPRESSED',
        contract=dict(mean=[0.],scale=[1.],coefficient=[.2],intercept=.003))
    W=dict(stage='W',routing_status='ADMITTED',domain=True,parent=C,
        contract=dict(family=v211_w.NATURAL_CUBIC,basis=dict(mean=0.,scale=1.,knots=[-1.,-.3,.3,1.]),coefficient=[.03,-.02]))
    A=dict(stage='A',routing_status='ADMITTED',domain=True,parent=W,profile=[1,2],residual_mean=0.,
        contract=dict(family=v21_a.MATURE_RESIDUAL_AR,mean=[0.,0.],scale=[1.,1.],coefficient=[.2,.1],intercept=.002))
    history=T.copy();history['origin']=history.origin-2;history['target_stop_exclusive']=history.origin+1
    history['residual']=np.sin(history.origin/7);history['residual_source_role']='CROSS_FITTED_PREQUENTIAL'
    state=build_a_training_state_from_cf_residual(history,policy_id='frozen-fixture',h_steps=1)
    return p,T,[K,C,W,A],state


@pytest.mark.parametrize('route',['ADMITTED','BYPASSED','NOT_APPLICABLE','UNRESOLVED'])
def test_actual_nonlinear_k_c_w_a_predictions_routes_and_rmse_identical(tmp_path,route):
    from canonical.cz_crossrod_l128_20261008.canonical_engine import CanonicalPipeline
    base=ROOT/'baseline_runtime/canonical/cz_crossrod_l128_20261008'
    loader=importlib.util.spec_from_file_location('v06_baseline',base/'__init__.py',submodule_search_locations=[str(base)])
    module=importlib.util.module_from_spec(loader);sys.modules['v06_baseline']=module;loader.loader.exec_module(module)
    Original=importlib.import_module('v06_baseline.canonical_engine').CanonicalPipeline
    p,T,states,a_state=numerical_fixture(CanonicalPipeline,tmp_path/'new')
    old,old_T,old_states,old_a=numerical_fixture(Original,tmp_path/'old')
    for s,o in zip(states,old_states):s['routing_status']=route;o['routing_status']=route
    before=[old.predict(s,old_T,a_state=old_a)[0] for s in old_states]
    data=observed(180)
    audit_target_contract(data,CZHeadAuditConfig('H1',1),spec(),samples=T)
    segmented_diagnostics(data,spec()) # enabled explicitly; model never sees outputs
    after=[p.predict(s,T,a_state=a_state)[0] for s in states]
    for x,y in zip(before,after):
        np.testing.assert_array_equal(x,y)
        assert np.sqrt(np.mean((T.y_true-x)**2))==np.sqrt(np.mean((T.y_true-y)**2))
    assert [s['routing_status'] for s in states]==[s['routing_status'] for s in old_states]
    np.testing.assert_array_equal(p.emit(states[-1],T,source=a_state),old.emit(old_states[-1],old_T,source=old_a))
    if route=='ADMITTED':assert np.any(after[-1] != np.round(after[-1],2))


def test_new_audit_modules_not_imported_by_any_original_model_file():
    base=ROOT/'baseline_runtime/canonical/cz_crossrod_l128_20261008'
    for file in base.rglob('*.py'):
        text=file.read_text(encoding='utf-8')
        assert 'discrete_observation' not in text and 'discrete_diagnostics' not in text and 'discrete_audit' not in text


def test_c_bypass_retains_nonlinear_q_and_w_bypass_does_not_block_mature_a(tmp_path):
    from canonical.cz_crossrod_l128_20261008.canonical_engine import CanonicalPipeline
    p,T,(K,C,W,A),state=numerical_fixture(CanonicalPipeline,tmp_path)
    C['routing_status']='BYPASSED'
    q,mask,audit=p.response_coordinate(C,T)
    np.testing.assert_array_equal(q,p.predict(K,T)[0])
    assert mask.all() and audit['interface']=='PSI_K_RETAINED_REFERENCE_MODEL_SCALAR'
    assert np.any(p.emit(W,T)!=p.emit(C,T))
    W['routing_status']='BYPASSED'
    a,mask,_=p.predict(A,T,a_state=state)
    assert mask[:-1].all() and not mask[-1] # last origin has missing mature history
    assert np.any(a[mask]!=p.emit(W,T)[mask])
    assert a[-1]==p.emit(W,T)[-1]


def test_a_training_cf_never_silently_seeds_deployment_inference(tmp_path):
    from canonical.cz_crossrod_l128_20261008.canonical_engine import CanonicalPipeline
    from canonical.cz_crossrod_l128_20261008.a_state import INFERENCE_DEPLOYED,feature_rows
    from canonical.cz_crossrod_l128_20261008.invariants import ProtocolError
    p,T,_,state=numerical_fixture(CanonicalPipeline,tmp_path)
    with pytest.raises(ProtocolError,match='cannot seed deployment'):
        feature_rows(T,state,(1,2),8,0.,expected_role=INFERENCE_DEPLOYED)
    future=state.frame.copy();future['origin']+=100;future['target_stop_exclusive']=future.origin+1
    from canonical.cz_crossrod_l128_20261008.a_state import build_a_training_state_from_cf_residual
    ahead=build_a_training_state_from_cf_residual(future,policy_id='frozen-fixture',h_steps=1)
    _,mask,_=feature_rows(T,ahead,(1,2),8,0.)
    assert not mask.any()


def test_float32_grid_storage_uncertainty_does_not_become_adc_claim():
    evidence=grid_candidate_evidence(np.array([317.20,318.13,310.11],dtype=np.float32))
    assert evidence['fraction_on_candidate_grid']==1. and evidence['stored_dtype']=='float32'
    assert evidence['maximum_grid_distance']>1e-8
    assert not evidence['instrument_verified']
    assert grid_candidate_evidence([.012,.028,.033])['grid_evidence_status']=='unknown'
    promoted=grid_candidate_evidence(np.array([317.20,318.13,310.11],dtype=np.float32).astype(np.float64))
    assert promoted['fraction_strict_decimal_grid']==0. and promoted['fraction_on_candidate_grid']==1.
    assert promoted['precision_evidence'].startswith('EMPIRICAL_EXACT_BINARY32')
    assert not promoted['instrument_verified']


def test_declared_target_index_mismatch_is_reported_without_relabeling():
    data=observed(16);samples=pd.DataFrame(dict(entity_id=['rod'],origin=[5],target_start=[8],target_stop_exclusive=[9]))
    out=audit_target_contract(data,CZHeadAuditConfig('H1',1,4),spec(),samples=samples)
    assert 'DECLARED_TARGET_START_MISMATCH' in out.invalid_reason.iloc[0]
    assert 'DECLARED_TARGET_STOP_EXCLUSIVE_MISMATCH' in out.invalid_reason.iloc[0]
    assert samples.target_start.iloc[0]==8 and out.target_start_index.iloc[0]==5
