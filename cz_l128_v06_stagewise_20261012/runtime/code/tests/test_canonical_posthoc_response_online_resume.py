import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from canonical.cz_h1_posthoc_v22.canonical_engine import CanonicalPipeline
from canonical.cz_h1_posthoc_v22.checkpoint import Checkpoints
from canonical.cz_h1_posthoc_v22.fold_geometry import folds,geometry_audit
from canonical.cz_h1_posthoc_v22.invariants import ProtocolError,digest
from canonical.cz_h1_posthoc_v22.a_state import feature_rows

GEOMETRY={'block_rows':320,'count':4,'end_fractions':[.4,.6,.8,1.]}


def samples(n=30,entity='segment'):
    origin=np.arange(100,100+n)
    return pd.DataFrame(dict(entity_id=entity,origin=origin,target_stop_exclusive=origin+1,
        latest_available_target_index=origin-1,y_true=np.arange(n,dtype=float),
        sample_id=[entity+str(i) for i in origin]))


def online_pipe():
    p=object.__new__(CanonicalPipeline)
    p.emit=lambda state,T:np.full(len(T),3.)
    def predict(state,T,*,a_state):
        x,mask,audit=feature_rows(T,a_state,(1,8),128,0.)
        out=np.full(len(T),3.);out[mask]+=x.sum(axis=1)
        return out,mask,audit
    p.predict=predict
    return p


def test_online_cold_start_future_target_is_inaccessible():
    p=online_pipe();T=samples();model={'parent':{'partition_model_id':'W-prefix'}}
    a,mask,proof=p.online_a(model,T)
    assert not mask[:8].any() and mask[8:].all()
    np.testing.assert_array_equal(a[:8],3.)
    changed=T.copy();changed.loc[20:,'y_true']+=10000
    b,_,_=p.online_a(model,changed)
    np.testing.assert_array_equal(a[:21],b[:21])
    assert proof['initialization']=='EMPTY_PER_SCORING_BLOCK_OR_ENTITY'
    assert not proof['training_backcast_used'] and not proof['state_crosses_entity']


def test_independent_block_and_entity_restarts():
    p=online_pipe();model={'parent':{'partition_model_id':'W'}}
    T=pd.concat([samples(entity='A'),samples(entity='B')],ignore_index=True)
    _,mask,_=p.online_a(model,T)
    assert not mask[30:38].any()
    _,again,_=p.online_a(model,T.iloc[10:20].copy())
    assert not again[:8].any()


def test_w_q_reads_retained_k_model_despite_two_bypasses():
    p=object.__new__(CanonicalPipeline)
    K=dict(stage='K',routing_status='BYPASSED',reference_channel='joint_lift')
    C=dict(stage='C',parent=K,routing_status='BYPASSED')
    p.predict=lambda state,T:(np.arange(len(T),dtype=float)+7,np.ones(len(T),bool),{})
    q,native,audit=p.response_coordinate(C,samples())
    assert np.ptp(q)>0 and native.all()
    assert audit['interface']=='PSI_K_RETAINED_REFERENCE_MODEL_SCALAR'


def test_registered_cross_entity_geometry_unavailable_without_moving():
    T=samples(3000);T.loc[:1099,'entity_id']='early'
    assert folds(T,GEOMETRY)==[]
    audit=geometry_audit(T,GEOMETRY)
    assert audit['score_blocks'][0]['start']==880
    assert audit['score_blocks'][0]['crosses_entity']


def test_gap_is_unavailable_and_single_entity_is_available():
    T=samples(3000);T['dependency_start']=T.origin-256;T['dependency_stop_exclusive']=T.origin+1
    assert len(folds(T,GEOMETRY))==4
    T.loc[1000:,'origin']+=1
    assert folds(T,GEOMETRY)==[]


def test_resume_exact_binding_and_artifact_hash(tmp_path):
    binding=dict(release='sealed',environment='same',partition='same',registry='same')
    cp=Checkpoints(tmp_path/'run',binding)
    f=tmp_path/'source';f.write_bytes(b'unchanged')
    cp.save('candidate','key',dict(status='NUMERICALLY_INVALID',error='certificate'),[f])
    again=Checkpoints(tmp_path/'run',binding,resume=True)
    assert again.load('candidate','key')['status']=='NUMERICALLY_INVALID'
    with pytest.raises(ProtocolError):Checkpoints(tmp_path/'run',dict(binding,environment='different'),resume=True)
    f.write_bytes(b'changed')
    with pytest.raises(ProtocolError,match='artifact changed'):again.load('candidate','key')


def test_incomplete_checkpoint_is_not_reused(tmp_path):
    cp=Checkpoints(tmp_path/'run',dict(protocol='same'))
    temporary=cp.path('candidate','k').with_suffix('.tmp');temporary.parent.mkdir(parents=True)
    temporary.write_text('unfinished')
    assert cp.load('candidate','k') is None


def test_protocol_explicitly_freezes_all_cold_starts():
    from pathlib import Path
    p=Path(__file__).parents[1]/'canonical/cz_h1_posthoc_v22/CANONICAL_PROTOCOL.json'
    protocol=json.loads(p.read_text(encoding='utf-8'))
    assert not protocol['A_initialization']['pre_score_warmup']
    assert protocol['A_initialization']['target_test']=='EMPTY_PER_ENTITY_SEGMENT'
    assert protocol['full_registry_smoke']['prefix_rows']==3000
    assert not protocol['execution_authorization']['formal_development_training']


def test_environment_thread_or_backend_change_is_rejected(monkeypatch):
    from canonical.cz_h1_posthoc_v22 import environment
    expected=dict(float_dtype='<f8',float_bits=64,
        thread_environment={k:'1' for k in environment.THREAD_KEYS},blas=[dict(num_threads=1,internal_api='openblas')])
    actual=dict(expected,blas=[dict(num_threads=1,internal_api='changed')])
    monkeypatch.setattr(environment,'fingerprint',lambda:actual)
    with pytest.raises(ProtocolError,match='blas'):environment.verify_environment(expected)


def test_true_authority_w_solver_uses_q_with_zero_parent_baseline(tmp_path):
    pytest.importorskip('pyreadr')
    from pathlib import Path
    from canonical.cz_h1_posthoc_v22.invariants import file_sha
    authority=Path(__file__).parents[1];module=authority/'canonical/cz_h1_posthoc_v22'
    protocol=json.loads((module/'CANONICAL_PROTOCOL.json').read_text())
    registration=dict(canonical_protocol_sha256={'A_TEMPORAL_REGISTRY.json':file_sha(module/'A_TEMPORAL_REGISTRY.json')})
    p=CanonicalPipeline(dict(authority=authority,shared=tmp_path,registration=registration,
        protocol=protocol,policy_id=digest(registration)),tmp_path/'unused',mode='metadata_only')
    p.mode='solver_smoke'
    T=samples(100);q=np.linspace(-2.,2.,100)
    T['parent_prediction']=0.;T['response_coordinate']=q
    T['coordinate_available']=True;T['residual']=q*q+q
    spec=p.w._w_candidates(p.v211,p.v21,p.v2,1,False)[0]
    contract,correction,mask,_=p.fit_residual_recipe('W',spec,T,T,T)
    assert contract is not None and mask.all() and np.isfinite(correction).all()
    assert p.fit_calls==1 and np.ptp(correction)>0
