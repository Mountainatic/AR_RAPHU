"""Final-refit integration, actual numerical operators, and held-out replay safety."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from test_canonical_h1_integration_support import pipe, samples, engine
from canonical_core import ProtocolSupportError, NumericallyInvalid, digest, folds, cf_blocks
from prism_benchmark.v211_support import support_id_hash, apply_native_support
from final_development_cf import frozen_contract, require_contract, RecursiveSupport, plan_replay


@pytest.fixture
def replay_pipe(pipe):
    pipe.registration['final_development_cf'] = frozen_contract()
    pipe.policy_id = digest(pipe.registration)
    return pipe


def target_frame(n=5000):
    frame = samples(n)
    frame['y_true'] = np.sin(frame.origin.to_numpy()/17.) + frame.origin.to_numpy()*.003
    frame['latest_available_target_index'] = frame.origin-1
    return frame


def freeze(pipe,frame,upstream='C'):
    return dict(role='FROZEN_RESIDUAL_RECIPE_BEFORE_FINAL_REPLAY',upstream_stage=upstream,
        policy_id=pipe.policy_id,partition_id=support_id_hash(frame),
        selected_spec=['synthetic-test-only'],replay_contract=frozen_contract())


def mock_prefix_learning(pipe,monkeypatch):
    calls = []
    def learn(stage,fit,gid=None):
        calls.append((stage,fit.copy()))
        state = dict(stage='K',parent=None)
        for current in ['C','W'][:1 if stage=='C' else 2]:
            state = dict(stage=current,parent=state)
        state.update(partition_id=support_id_hash(fit),admission_policy_id=pipe.policy_id,
            deployment_model_id=digest([stage,support_id_hash(fit)]),fit_mean=float(fit.y_true.mean()))
        return state
    def emit(state,ev,source=None):
        if state['stage']=='C' and 'fit_mean' not in state:
            return ev.origin.to_numpy(float)*.001
        return ev.origin.to_numpy(float)*.001 + state.get('fit_mean',0.)*.01
    monkeypatch.setattr(pipe,'learn',learn)
    monkeypatch.setattr(pipe,'emit',emit)
    monkeypatch.setattr(pipe,'features',lambda state,ev:(np.column_stack([
        np.sin(ev.origin.to_numpy()/float(13+i)) for i in range(4)]),[]))
    monkeypatch.setattr(pipe,'support_for',lambda state,ev,**kwargs:ev.copy())
    pipe.metadata_only=False
    return calls


def test_protocol_freezes_320_and_keeps_inner_geometry(replay_pipe):
    contract=require_contract(replay_pipe)
    assert contract['block_rows']==320 and contract['include_partial_tail']
    assert replay_pipe.block==8 and replay_pipe.warmup==640
    replay_pipe.registration['final_development_cf']=dict(contract,block_rows=8)
    with pytest.raises(ProtocolSupportError,match='contract'):require_contract(replay_pipe)


def test_missing_contract_stops_final_refit(pipe):
    with pytest.raises(ProtocolSupportError,match='contract'):plan_replay(pipe,'C',samples())


@pytest.mark.parametrize('cf',[False,True])
def test_recursive_support_split_geometry_matches_frozen_core(replay_pipe,cf):
    frame=samples(2400);checker=RecursiveSupport(replay_pipe,frame)
    rows=np.arange(len(frame))
    expected=cf_blocks(frame,8,640) if cf else folds(frame,8)
    for (fit,ev),original in zip(checker.splits(rows,cf=cf),expected):
        assert np.array_equal(fit,original[0]) and np.array_equal(ev,original[1])


@pytest.mark.parametrize('upstream',['C','W'])
def test_full_plan_starts_at_earliest_recursive_support_covers_all_and_tail(replay_pipe,upstream):
    frame=samples(5000)
    blocks,audit=plan_replay(replay_pipe,upstream,frame)
    checker=RecursiveSupport(replay_pipe,frame)
    first=int(blocks[0][1][0]);assert first!=640
    for index in range(first):
        fit=np.flatnonzero(checker.stop[:index]<checker.start[index])
        assert not checker.native[index] or not checker.ready(upstream,fit)
    covered=np.concatenate([ev for _,ev in blocks])
    assert np.array_equal(covered,np.arange(first,len(frame)))
    assert audit['planned_replay_rows']>4*320 and audit['planned_replay_rows']>672
    assert all(len(ev)==320 for _,ev in blocks[:-1])
    assert 0<len(blocks[-1][1])<=320 and blocks[-1][1][-1]==len(frame)-1
    for fi,ei in blocks:
        assert fi.max()<ei.min()
        assert frame.iloc[fi].dependency_stop_exclusive.max()<frame.iloc[ei].dependency_start.min()


def test_entity_reset_never_uses_future_entity_rows(replay_pipe):
    first=samples(2600);second=samples(2600)
    second.entity_id='Rod_1_segment_1'
    second.base_origin_id='next_'+second.base_origin_id
    second.view_sample_id='next_'+second.view_sample_id
    frame=pd.concat([first,second],ignore_index=True)
    blocks,audit=plan_replay(replay_pipe,'C',frame)
    assert audit['planned_replay_rows']>672
    for fi,ei in blocks:
        assert len(set(frame.iloc[ei].entity_id))==1
        assert fi.max()<ei.min()
        if frame.iloc[ei].entity_id.iloc[0]=='Rod_1_segment_0':
            assert set(frame.iloc[fi].entity_id)=={'Rod_1_segment_0'}
    assert np.concatenate([ev for _,ev in blocks])[-1]==len(frame)-1


@pytest.mark.parametrize('upstream',['C','W'])
def test_materialized_replay_is_prefix_fitted_and_matches_provenance(replay_pipe,monkeypatch,upstream):
    frame=target_frame();calls=mock_prefix_learning(replay_pipe,monkeypatch)
    source,audit,path=replay_pipe.final_development_stream(upstream,frame,None,freeze(replay_pipe,frame,upstream))
    assert len(source)>672 and len(source)>1280
    assert 'CF_fold' not in source and audit['binding']['role']=='FINAL_DEVELOPMENT_CF_REPLAY'
    assert len(calls)==len(audit['provenance'])
    for (_,fit),record in zip(calls,audit['provenance']):
        ev=source.loc[source.replay_block==record['block']]
        assert not set(fit.base_origin_id)&set(ev.base_origin_id)
        assert fit.dependency_stop_exclusive.max()<ev.dependency_start.min()
        assert fit.origin.max()<ev.origin.min()
        assert record['fit_support_hash']==support_id_hash(fit)
        assert record['replay_support_hash']==support_id_hash(ev)
        assert not record['target_used_to_fit_own_predictor']
    assert np.array_equal(source.residual.to_numpy(),source.y_true.to_numpy()-source.parent_prediction.to_numpy())
    if upstream=='W':assert 'up_w' in source
    # Disk reuse performs no prefix fitting and retains exact materialized rows.
    calls.clear();replay_pipe.stream_cache.clear()
    again,_,_=replay_pipe.final_development_stream(upstream,frame,None,freeze(replay_pipe,frame,upstream))
    assert not calls and source.equals(again)
    Path(path).write_bytes(b'tampered')
    with pytest.raises(ProtocolSupportError,match='hash mismatch'):
        replay_pipe.final_development_stream(upstream,frame,None,freeze(replay_pipe,frame,upstream))


def test_current_block_targets_cannot_change_own_upstream_prediction(replay_pipe,monkeypatch):
    frame=target_frame();mock_prefix_learning(replay_pipe,monkeypatch)
    blocks,_=plan_replay(replay_pipe,'C',frame);ei=blocks[0][1]
    original,_,_=replay_pipe.final_development_stream('C',frame,None,freeze(replay_pipe,frame))
    changed=frame.copy();changed.loc[ei,'y_true']+=10000
    updated,_,_=replay_pipe.final_development_stream('C',changed,None,freeze(replay_pipe,changed))
    a=original.loc[original.replay_block==0];b=updated.loc[updated.replay_block==0]
    assert np.array_equal(a.parent_prediction,b.parent_prediction)
    assert np.allclose(b.residual-a.residual,10000)


def test_replay_rejects_full_partition_parent(replay_pipe,monkeypatch):
    frame=target_frame();mock_prefix_learning(replay_pipe,monkeypatch)
    monkeypatch.setattr(replay_pipe,'learn',lambda *args:dict(stage='C',partition_id=support_id_hash(frame),admission_policy_id=replay_pipe.policy_id))
    with pytest.raises(ProtocolSupportError,match='another prefix'):
        replay_pipe.final_development_stream('C',frame,None,freeze(replay_pipe,frame))


def test_metadata_preflight_cannot_materialize_residuals(replay_pipe):
    with pytest.raises(ProtocolSupportError,match='NO_TRAINING'):
        replay_pipe.final_development_stream('C',samples(),None,freeze(replay_pipe,samples()))


def test_only_partition_train_rows_are_accepted(replay_pipe):
    frame=samples();frame.loc[100,'split']='validation'
    with pytest.raises(ProtocolSupportError,match='development train'):plan_replay(replay_pipe,'C',frame)


@pytest.mark.parametrize('stage',['W','A'])
def test_actual_tune_residual_selects_on_672_then_final_fits_full_replay(replay_pipe,monkeypatch,stage):
    frame=target_frame();mock_prefix_learning(replay_pipe,monkeypatch)
    monkeypatch.setattr(replay_pipe,'domain',lambda *args:{'pass':True})
    spec=(engine.w.NATURAL_CUBIC,4,1.,0.,1) if stage=='W' else ((1,8),1.,0.)
    monkeypatch.setattr(engine.w,'_w_candidates',lambda *args:[spec])
    if stage=='A':
        replay_pipe.A_profiles=[(1,8)]
        replay_pipe.v2['A_module']['ridge_alpha_grid']=[1.]
        replay_pipe.v21['A']['soft_overlap_mu']=[0.]
    observed=[]
    def fit(stage,spec,fit,ev,source):
        observed.append(dict(spec=spec,fit=fit.copy(),ev=ev.copy(),source=source.copy()))
        return {'numerical_certificate':{'status':'PASS'}},np.zeros(len(ev))
    monkeypatch.setattr(replay_pipe,'fit_residual_recipe',fit)
    destination=replay_pipe.root/'test_tuning';destination.mkdir()
    model=replay_pipe.tune_residual(stage,frame,None,destination)
    assert len(observed)==5
    assert all(len(call['source'])==672 for call in observed[:4])
    final=observed[-1]
    assert len(final['fit'])>1280 and final['fit'].equals(final['ev']) and final['fit'].equals(final['source'])
    assert all(call['spec']==spec for call in observed)
    assert pd.read_parquet(model['inner_selection_source_path']).shape[0]==672
    assert pd.read_parquet(model['source_path']).shape[0]==len(final['fit'])
    assert (destination/'SELECTED_RECIPE_BEFORE_FINAL_REPLAY.json').exists()
    assert model['final_replay_coverage']['planned_replay_rows']==len(final['fit'])
    if stage=='A':
        assert model['residual_mean']==float(apply_native_support(final['source'],8).residual.mean())


@pytest.mark.parametrize('stage',['W','A'])
def test_unchanged_actual_numerical_operator_refits_materialized_full_replay(replay_pipe,monkeypatch,stage):
    frame=target_frame();mock_prefix_learning(replay_pipe,monkeypatch)
    upstream='C' if stage=='W' else 'W'
    source,_,_=replay_pipe.final_development_stream(upstream,frame,None,freeze(replay_pipe,frame,upstream))
    assert len(source)>1280
    spec=(engine.w.NATURAL_CUBIC,4,1.,0.,1) if stage=='W' else ((1,8),1.,0.)
    contract,prediction=replay_pipe.fit_residual_recipe(stage,spec,source,source,source)
    assert engine.numerical_contract_passes(contract)
    assert len(prediction)==len(source) and np.isfinite(prediction).all()


def test_invalid_upstream_prediction_aborts_without_skipping_rows(replay_pipe,monkeypatch):
    frame=target_frame();mock_prefix_learning(replay_pipe,monkeypatch)
    monkeypatch.setattr(replay_pipe,'emit',lambda state,ev,*args:np.full(len(ev),np.nan))
    with pytest.raises(NumericallyInvalid,match='upstream prediction'):
        replay_pipe.final_development_stream('C',frame,None,freeze(replay_pipe,frame))
    assert not list((replay_pipe.root/'final_development_streams').rglob('prequential.parquet'))


def test_replay_requires_selected_recipe_bound_to_this_partition(replay_pipe,monkeypatch):
    frame=target_frame();mock_prefix_learning(replay_pipe,monkeypatch)
    wrong=freeze(replay_pipe,frame);wrong['partition_id']='other'
    with pytest.raises(ProtocolSupportError,match='recipe freeze'):
        replay_pipe.final_development_stream('C',frame,None,wrong)


def test_mutated_registration_cannot_reuse_old_policy_cache(replay_pipe):
    replay_pipe.registration['other_new_parameter']='different'
    with pytest.raises(ProtocolSupportError,match='policy digest is stale'):
        plan_replay(replay_pipe,'C',samples())


def test_operator_registry_support_routing_and_selection_stream_are_ast_identical():
    path=Path(engine.__file__)
    def methods(source):
        cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef))
        return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
    old=methods(path.with_name('canonical_engine.py.support_integration_reference.txt').read_text(encoding='utf-8'))
    new=methods(path.read_text(encoding='utf-8'))
    changed={name for name in old if old[name]!=new[name]}
    assert changed=={'__init__','tune_residual'}
    assert set(new)-set(old)=={'final_development_stream'}
