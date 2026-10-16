"""Actual canonical dispatch and support tests; all labels are synthetic/mocked."""
import ast
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT=Path(__file__).resolve().parents[1]
os.environ['CZ_AUTHORITY_ROOT']=str(PROJECT)
sys.path.insert(0,str(PROJECT/'src'))
sys.path.insert(0,str(PROJECT/'canonical/cz_h1_registry_v2'))
import canonical_engine as engine
from canonical_core import ProtocolSupportError,folds
from maximal_anchor import maximal_metadata,CONTRACT
from prism_benchmark import cz_h1_temporal_registry as registry
from prism_benchmark.cz_l256_nowcast import view
from prism_benchmark.v211_support import SUPPORT_CONTRACT,registered_fold_native_masks,support_id_hash


def samples(n=2400):
    origin=np.arange(1,n+1)
    return pd.DataFrame(dict(base_origin_id=[f'b{i}' for i in origin],view_sample_id=[f'v{i}' for i in origin],
        dataset='cz_czochralski',entity_id='Rod_1_segment_0',task_id='CZ_DIAM_RAW2S_CURRENT_L256_H1',split='train',
        origin=origin,dependency_start=np.maximum(0,origin-256),dependency_stop_exclusive=origin+1,
        causal_history_floor=0,anchor_history_steps=1,sample_support_contract=SUPPORT_CONTRACT))


@pytest.fixture
def pipe(tmp_path):
    reg=dict(k_registry_sha256=registry.file_sha(PROJECT/registry.CONFIG_ROOT/registry.REGISTRY_NAME),
        pipeline_contract_id='CZ_H1_CANONICAL_REGISTRY_SUPPORT_INTEGRATION_V1',anchor_contract=CONTRACT,
        inner_block_rows=8,cf_warmup_rows=640,K_workers=1,
        training_support_hashes={'H1':{'Rod_1_to_Rod_2':'synthetic'}})
    return engine.CanonicalPipeline(tmp_path,tmp_path/'Rod_1_to_Rod_2',
        view('dynamic',h_steps=1,task_id='CZ_DIAM_RAW2S_CURRENT_L256_H1'),'H1',reg,metadata_only=True)


def state(route='BYPASSED',histories=(256,256,256,256)):
    return dict(stage='K',routing_status=route,reference_index=0,
        representations=[dict(channel=ch,profile=[2,h],m_tau=12,contract={}) for ch,h in zip(engine.CHANNELS,histories)])


def test_actual_canonical_runner_grid_is_23_profiles_maximum_512_seconds(pipe):
    for ch in engine.CHANNELS:
        request=pipe.structural_grid_request(samples(),ch,None,pipe.root/ch)
        assert len(request['profiles'])==23
        assert max(p[1] for p in request['profiles'])*2==512
        assert request['m_tau_values']==[4,8,12]
        assert request['local_comparison_history']==256


@pytest.mark.parametrize('channel',registry.CHANNELS)
def test_actual_tune_k_dispatches_registered_grid_to_authority_solver(pipe,channel,monkeypatch):
    class Captured(Exception):pass
    captured=[]
    def spy(**kwargs):captured.append(kwargs);raise Captured()
    pipe.metadata_only=False  # Only the mocked dispatch may run; no accessor IO.
    monkeypatch.setattr(engine,'CHANNELS',[channel])
    monkeypatch.setattr(engine.k,'_run_structural_grid',spy)
    with pytest.raises(Captured):pipe.tune_k(samples(),None,pipe.root/'tune')
    assert captured[0]['channel']==channel and len(captured[0]['profiles'])==23
    assert {h for _,h in captured[0]['profiles']}=={8,16,32,64,128,256}
    assert {d for d,_ in captured[0]['profiles']}=={2,4,8,16}


def test_active_engine_has_no_fixed_2_8_or_heater_growth_registry():
    source=Path(engine.__file__).read_text(encoding='utf-8')
    assert 'self.delta' not in source and 'self.short_history' not in source
    assert '1200' not in source and 'GROWTH' not in source


def test_canonical_growth_arms_are_rejected(pipe):
    with pytest.raises(ProtocolSupportError,match='Growth arms'):pipe.k_grid('main_heater_power',1)


def test_canonical_preflight_cannot_train(pipe):
    for call in [lambda:pipe.learn('K',samples()),lambda:pipe.tune('K',samples(),None),
                 lambda:pipe.tune_k(samples(),None,pipe.root/'tune')]:
        with pytest.raises(ProtocolSupportError,match='NO_TRAINING'):call()


def test_maximal_anchor_c1_restores_real_early_support_and_preserves_scoring():
    # IDs follow the authority task contract, rather than fake b/v IDs.
    from hashlib import sha256
    from maximal_anchor import TASK
    original=samples().query('origin>=256').copy()
    original.base_origin_id=[sha256(f'{TASK}|Rod_1_to_Rod_2|Rod_1_segment_0|{t}'.encode()).hexdigest() for t in original.origin]
    original.view_sample_id=[sha256(f'dynamic|{b}'.encode()).hexdigest() for b in original.base_origin_id]
    base=pd.DataFrame(dict(entity_id=['Rod_1_segment_0']*2500,row_in_entity=np.arange(2500)))
    maximal,ranges=maximal_metadata(original,base,'Rod_1_to_Rod_2')
    assert maximal.origin.min()==1 and maximal.origin.max()==original.origin.max()
    assert len(maximal)==len(original)+255
    assert (maximal.causal_history_floor==0).all()
    assert set(original.base_origin_id).issubset(set(maximal.base_origin_id))
    for (_,oi),(_,ni) in zip(folds(original,320),folds(maximal,320)):
        assert support_id_hash(original.iloc[oi])==support_id_hash(maximal.iloc[ni])


def test_candidate_native_fit_really_has_more_early_rows_in_maximal_c1():
    frame=samples();fs=folds(frame,320)
    def records(history):return registered_fold_native_masks(frame,fs,fit_history_steps=history,
        scoring_history_steps=256,fit_cap=100000,evaluation_cap=50000)
    for short,long in zip(records(8),records(256)):
        assert len(short['fit_native'])>len(long['fit_native'])
        assert support_id_hash(short['evaluation'])==support_id_hash(long['evaluation'])


def test_k_bypass_preserves_parent_support_and_never_reads_basis(pipe,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('K basis accessed during BYPASS')
    monkeypatch.setattr(engine,'profile_values',forbidden)
    frame=samples();bypass=state()
    assert support_id_hash(pipe.support_for(bypass,frame))==support_id_hash(frame)
    assert np.array_equal(pipe.emit(bypass,frame),np.zeros(len(frame)))
    assert len(pipe.support_for(bypass,frame))>len(pipe.support_for(bypass,frame,representations=True))


def test_k_admitted_scalar_support_uses_selected_reference_not_other_bases(pipe,monkeypatch):
    model=state('ADMITTED',(8,256,256,256))
    legal=pipe.support_for(model,samples())
    assert legal.origin.min()==8
    calls=[]
    def values(accessor,frame,ch,profile,m):calls.append(profile);return np.zeros((len(frame),1)),[]
    monkeypatch.setattr(engine,'profile_values',values)
    monkeypatch.setattr(engine,'predict_contract',lambda x,c:np.zeros(len(x)))
    assert len(pipe.emit(model,legal))==len(legal)
    assert calls==[[2,8]]


def test_downstream_bypasses_preserve_the_k_parent_support(pipe,monkeypatch):
    frame=samples();parent=state()
    for stage in ['C','W','A']:
        parent=dict(stage=stage,routing_status='BYPASSED',parent=parent)
        assert support_id_hash(pipe.support_for(parent,frame))==support_id_hash(frame)
        assert np.array_equal(pipe.emit(parent,frame),np.zeros(len(frame)))


def test_c_assembly_view_does_not_permanently_trim_bypassed_parent(pipe):
    frame=samples();parent=state();c=dict(stage='C',routing_status='ADMITTED',parent=parent)
    assert pipe.support_for(c,frame).origin.min()==256
    assert pipe.support_for(parent,frame).origin.min()==1
    c['routing_status']='BYPASSED'
    assert pipe.support_for(c,frame).origin.min()==1


def test_preflight_label_projection_excludes_targets():
    from maximal_anchor import TRAIN_METADATA
    assert not {'y_true','current_level','future_level_true','crystal_diameter'}.intersection(TRAIN_METADATA)


def test_registered_core_and_solver_sources_still_match_authority():
    inventory=json.loads((PROJECT/registry.CONFIG_ROOT/'AUTHORITY_UNCHANGED_BINDINGS.json').read_text())
    registry.verify_bindings(PROJECT,inventory['files'])
    registry.load_frozen_registry(PROJECT)


def test_original_canonical_routing_core_is_unmodified():
    # Verified source hash of the existing canonical_core.py, stored with the
    # baseline engine snapshot. No risk gate or fold algorithm is redesigned.
    provenance=json.loads((PROJECT/'canonical/cz_h1_registry_v2/BASELINE_PROVENANCE.json').read_text())
    assert registry.file_sha(PROJECT/'canonical/cz_h1_registry_v2/canonical_core.py',canonical_lf=True)==provenance['canonical_core_sha256_lf']
