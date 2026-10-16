import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from canonical.cz_h1_posthoc_v22.fold_geometry import folds,selection_blocks
from canonical.cz_h1_posthoc_v22.canonical_engine import CanonicalPipeline,CHANNELS
from canonical.cz_h1_posthoc_v22.invariants import file_sha,digest,ProtocolError


def frame(n):
    t=np.arange(1,n+1)
    return pd.DataFrame(dict(entity_id='e',origin=t,dependency_start=np.maximum(0,t-256),
        dependency_stop_exclusive=t+1))


def geometry():return dict(block_rows=320,count=4,end_fractions=[.4,.6,.8,1.])


def test_fixed_distributed_blocks_and_raw_purge():
    samples=frame(3000)
    fs=folds(samples,geometry())
    assert [int(e[-1]+1) for _,e in fs]==[1200,1800,2400,3000]
    assert all(len(e)==320 for _,e in fs)
    for f,e in fs:
        assert f[-1]<e[0]
        assert samples.iloc[f].dependency_stop_exclusive.max()<samples.iloc[e].dependency_start.min()
    assert not folds(frame(1580),geometry())


def test_context_and_score_are_separate_supports():
    samples=frame(5000)
    for f,e,context,score,j in selection_blocks(samples,geometry(),640):
        assert len(score)==320 and len(context)==640 and len(e)==960
        assert not set(context)&set(score)
        assert max(f)<min(e)


def test_actual_versioned_engine_loads_full_frozen_k_and_a_registries(tmp_path):
    pytest.importorskip('pyreadr',reason='Authority imports require the sealed server environment')
    benchmark=Path(__file__).resolve().parents[1]
    module=benchmark/'canonical/cz_h1_posthoc_v22'
    protocol=json.loads((module/'CANONICAL_PROTOCOL.json').read_text())
    registration=dict(canonical_protocol_sha256={'A_TEMPORAL_REGISTRY.json':file_sha(module/'A_TEMPORAL_REGISTRY.json')})
    context=dict(authority=benchmark,shared=tmp_path,registration=registration,protocol=protocol,policy_id=digest(registration))
    pipe=CanonicalPipeline(context,tmp_path/'out',mode='metadata_only')
    for ch in CHANNELS:
        profiles,mts,history=pipe.k_grid(ch)
        assert len(profiles)==23 and history==256 and mts==[4,8,12]
    assert pipe.A_profiles==[(1,h) for h in [8,16,32,64,96,128]]
    with pytest.raises(ProtocolError,match='forbidden'):pipe.learn('K',frame(3000))


def test_inference_engine_rejects_learning_without_touching_data():
    pipe=object.__new__(CanonicalPipeline);pipe.mode='inference_only'
    with pytest.raises(ProtocolError,match='forbidden'):pipe.learn('A',None)
    with pytest.raises(ProtocolError,match='forbidden'):pipe.tune_k(None,None)
    with pytest.raises(ProtocolError,match='forbidden'):pipe.fit_residual_recipe('A',None,None,None,None)


def test_actual_a_scoring_api_never_opens_cf_training_seed(monkeypatch):
    from canonical.cz_h1_posthoc_v22.a_state import feature_rows,INFERENCE_DEPLOYED
    pipe=object.__new__(CanonicalPipeline)
    pipe.emit=lambda state,rows:np.full(len(rows),.4)
    t=np.arange(1,31)
    universe=pd.DataFrame(dict(entity_id='e',origin=t,target_stop_exclusive=t+1,
        latest_available_target_index=t-1,y_true=t*.1))
    ev=universe.iloc[20:24].copy()
    def forbidden(*a,**kw):raise AssertionError('CF training seed was opened during outer inference')
    monkeypatch.setattr(pd,'read_parquet',forbidden)
    model=dict(parent={'deployment_model_id':'current-W'},source_path='forbidden-CF-file')
    state,proof=pipe.scoring_a(model,ev,universe)
    assert state.role==INFERENCE_DEPLOYED and proof['upstream_deployment_model_id']=='current-W'
    x,mask,_=feature_rows(ev,state,(1,8),8,0.)
    assert not mask.any() and len(x)==0
    assert state.frame.origin.min()==ev.origin.min()
    assert not proof['pre_score_warmup_used'] and not proof['training_backcast_used']
