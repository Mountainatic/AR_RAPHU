"""Representation-only regression tests; all sample values are synthetic."""
import ast
import builtins
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from prism_benchmark import cz_h1_temporal_registry as registry
from prism_benchmark.cz_l256_nowcast import view, _effective_h_steps
from prism_benchmark.v2_config import load_frozen_config
from prism_benchmark.v2_k import channel_profiles, profile_intervals
from prism_benchmark.v211_config import load_v211_configs
from prism_benchmark.v211_support import registered_fold_native_masks, SUPPORT_CONTRACT

spec = importlib.util.spec_from_file_location("cz_registry_preflight", PROJECT / "scripts/preflight_cz_h1_k_registry_v2.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def task(h):
    return preflight.task_view(h)


@pytest.fixture
def old():
    return load_frozen_config(PROJECT)


@pytest.fixture
def config(old):
    value = json.loads((PROJECT/registry.CONFIG_ROOT/registry.REGISTRY_NAME).read_text(encoding="utf-8"))
    return registry.attach_registry(old, value)


def profiles(config, h=1, channel=registry.CHANNELS[0]):
    return channel_profiles(task(h), channel, config)


def synthetic_support():
    origin = np.arange(8, 1200)
    n = len(origin)
    return pd.DataFrame(dict(base_origin_id=[f"b{i}" for i in origin], view_sample_id=[f"v{i}" for i in origin],
        dataset=["cz_czochralski"]*n, entity_id=["Rod_1_segment_0"]*n,
        task_id=["CZ_DIAM_RAW2S_CURRENT_L256_H1"]*n, split=["train"]*n,
        origin=origin, dependency_start=origin-1, dependency_stop_exclusive=origin+1,
        causal_history_floor=np.zeros(n,dtype=int), anchor_history_steps=np.zeros(n,dtype=int),
        sample_support_contract=[SUPPORT_CONTRACT]*n))


def test_regression_authority_h0_alias_and_explicit_h1_used_different_history_rules(old):
    assert _effective_h_steps(0) == _effective_h_steps(1) == 1
    assert profiles(old,0) != profiles(old,1)
    assert {h for _,h in profiles(old,1)} == {2,4,8}
    assert max(h*2 for _,h in profiles(old,1)) == 16


def test_explicit_h1_registry_extends_beyond_16_seconds(config):
    assert max(h*2 for _,h in profiles(config)) == 512


def test_h1_registered_history_grid_exact(config):
    assert {h for _,h in profiles(config)} == {8,16,32,64,128,256}
    assert profiles(config) == sorted([(d,h) for d in (2,4,8,16) for h in (8,16,32,64,128,256) if d<=h],key=lambda p:(p[1],-p[0]))


def test_h1_multirate_delta_registry_preserved(config):
    assert {d for d,_ in profiles(config)} == {2,4,8,16}
    assert len(profiles(config)) == 23


def test_h1_no_fixed_2_8_only_wrapper_semantics(config):
    assert profiles(config) != [(2,8)]
    assert profiles(config) == profiles(config,channel="main_heater_power")
    assert all(h<=256 for _,h in profiles(config))


def test_cz_h0_alias_and_explicit_h1_have_equivalent_k_temporal_registry(config):
    assert registry.task_contract(task(0).head) == registry.task_contract(task(1).head)
    assert preflight.snapshot(config,0) == preflight.snapshot(config,1)


def test_candidate_native_support_not_global_lmax_trimmed():
    train = synthetic_support()
    folds = [(np.arange(700),np.arange(700,len(train)))]
    counts=[]
    for h in (8,16,32,64,128,256):
        record = registered_fold_native_masks(train,folds,fit_history_steps=h,
            scoring_history_steps=256,fit_cap=10000,evaluation_cap=10000)[0]
        counts.append(len(record["fit_native"]))
        assert len(record["fit"]) == counts[-1]
        assert int(record["fit"].origin.min()) == h
    assert counts[0] > counts[-1] and all(a>b for a,b in zip(counts,counts[1:]))


def test_native_support_masks_precede_caps_and_local_scoring_is_common(config):
    train = synthetic_support()
    folds=[(np.arange(700),np.arange(700,len(train)))]
    rows, available = preflight.support_counts(train,folds,profiles(config),50,30,"Rod_1_to_Rod_2")
    frame=pd.DataFrame(rows)
    assert preflight.support_invariance(frame)
    assert (frame.fit_capped_rows == 50).all()
    assert (frame.local_common_scoring_capped_rows == 30).all()
    assert len(set(frame.local_common_scoring_hash)) == 1


def test_actual_authority_candidate_evaluation_fits_native_rows(monkeypatch, old):
    from prism_benchmark import v211_k
    train=synthetic_support(); train["y_true"]=0.0
    folds=[(np.arange(700),np.arange(700,len(train)))]
    fitted=[]
    monkeypatch.setattr(v211_k,"profile_values",lambda accessor,frame,*args:(np.zeros((len(frame),1)),[]))
    monkeypatch.setattr(v211_k,"fit_contract",lambda x,y,*args,**kwargs: fitted.append(len(y)) or {})
    monkeypatch.setattr(v211_k,"_candidate_valid",lambda *args:True)
    monkeypatch.setattr(v211_k,"predict_contract",lambda x,contract:np.zeros(len(x)))
    old=copy.deepcopy(old)
    old["row_caps"]["single_channel_k_fit"]=10000
    for h in (8,256):
        v211_k.evaluate_candidate(None,train,folds,"main_heater_power",(2,h),12,
             "LINEAR_DISTRIBUTED_LAG",1,(1.,1.,1.),old,scoring_history_steps=256)
    assert fitted == [700,452]


def test_short_lag_representation_survives_long_history_extension(config):
    geometry,audit = preflight.geometry(profiles(config))
    for h in registry.HISTORIES:
        assert audit.loc[audit.history_steps.eq(h),"short_resolution_pass"].any()
    witness=audit.query("delta_steps==2 and history_steps==256 and m_tau==12").iloc[0]
    assert witness.short_resolution_pass
    assert witness.region_0_20_max_bin_seconds == 4
    assert witness.region_0_20_bin_count == 5


def test_profile_intervals_are_causal(config):
    for d,h in profiles(config):
        for m in registry.M_TAU:
            intervals=profile_intervals(d,h,m)
            assert intervals[0][0]==0 and intervals[-1][1]==h
            assert all(0<=a<b<=h for a,b in intervals)
            assert all(b==c for (_,b),(c,_) in zip(intervals,intervals[1:]))
            t=300
            for a,b in intervals:
                indices=np.arange(t-b,t-a)
                assert len(indices)==b-a and indices.max()<t and indices.min()>=t-h


def test_h4_registry_unchanged(old, config):
    assert preflight.snapshot(old,4) == preflight.snapshot(config,4)
    assert {h for _,h in profiles(config,4)} == {8,16,32}


def test_target_partition_not_accessed_during_registry_build(monkeypatch, config):
    def forbidden(*args,**kwargs):raise AssertionError("data access during registry build")
    monkeypatch.setattr(pd,"read_parquet",forbidden)
    monkeypatch.setattr(builtins,"open",forbidden)
    assert len(profiles(config)) == 23


def test_preflight_reads_only_source_train_metadata_projection(monkeypatch, tmp_path):
    opened=[]
    class FakeParquet:
        def __init__(self,path):
            opened.append(path)
            assert path.name=="train.parquet"
        def read(self,columns):
            assert columns == list(preflight.SUPPORT_METADATA_COLUMNS)
            assert not {"y_true","current_level","future_level_true"}.intersection(columns)
            return self
        def to_pandas(self):return synthetic_support()
    monkeypatch.setattr(preflight.pq,"ParquetFile",FakeParquet)
    log=[]
    preflight.load_support(tmp_path,"Rod_1_to_Rod_2",log)
    assert len(opened)==len(log)==1
    assert log[0]["kind"] == "SOURCE_TRAIN_SUPPORT_METADATA_ONLY"


def test_no_residual_or_target_test_statistics_used_to_construct_registry(config):
    for module in (registry,preflight):
        tree=ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported=[n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any("residual" in name or "whitening" in name for name in imported)
        called=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        assert not set(called).intersection({"fit_contract","predict_contract","acf","welch","load_frozen_predictions"})
    registered=config["cz_h1_temporal_registry"]
    assert registered["test_used_for_design"] is False
    assert registered["residual_whitening_used_for_candidate_selection"] is False


def test_authority_numerical_solver_unchanged():
    inventory=json.loads((PROJECT/registry.CONFIG_ROOT/"AUTHORITY_UNCHANGED_BINDINGS.json").read_text(encoding="utf-8"))
    for path in ("src/prism_benchmark/v2_basis.py","src/prism_benchmark/v2_urysohn.py",
                 "src/prism_benchmark/v2_numerics.py","src/prism_benchmark/v211_k.py"):
        assert path in inventory["files"]
    registry.verify_bindings(PROJECT,inventory["files"])


def test_four_cz_channels_share_the_same_registered_family(config):
    assert all(profiles(config,channel=ch)==profiles(config) for ch in registry.CHANNELS)


def test_registry_mutation_after_loading_is_rejected(config):
    config["cz_h1_temporal_registry"]["history_steps"].append(1200)
    with pytest.raises(RuntimeError,match="mutated"):profiles(config)


def test_other_cz_task_does_not_acquire_one_step_semantics(old,config):
    from dataclasses import replace
    other=replace(task(1),head=replace(task(1).head,task_id="CZ_D20"))
    assert channel_profiles(other,registry.CHANNELS[0],old)==channel_profiles(other,registry.CHANNELS[0],config)


def test_preregistered_protocol_cannot_be_loaded_for_training(tmp_path):
    root=tmp_path/registry.CONFIG_ROOT; root.mkdir(parents=True)
    (root/registry.REGISTRY_NAME).write_text(json.dumps({"status":"PREREGISTERED_NO_TRAINING"}))
    (root/registry.SEAL_NAME).write_text(json.dumps({"status":"PREREGISTERED_NO_TRAINING"}))
    with pytest.raises(RuntimeError,match="has not passed"):registry.load_frozen_registry(tmp_path)


def test_frozen_protocol_loader_integration_when_sealed(old,config):
    root=PROJECT/registry.CONFIG_ROOT
    if not (root/registry.SEAL_NAME).exists():
        # Before preflight, refusal is the expected behavior, not a skipped test.
        with pytest.raises(FileNotFoundError):load_v211_configs(PROJECT,protocol=registry.PROTOCOL)
        return
    patch,v21,loaded=load_v211_configs(PROJECT,protocol=registry.PROTOCOL)
    assert profiles(loaded)==profiles(config)
    assert v21["K_C"]["m_tau"]==[4,8,12]
    assert patch["cz_h1_k_temporal_registry_protocol"]==registry.PROTOCOL_ID
    assert profiles(loaded,4)==profiles(old,4)


def test_inherited_algorithm_parameters_match_authority_stage1_merge(monkeypatch):
    from prism_benchmark import v211_representative_stage1_config as authority
    inventory=json.loads((PROJECT/registry.CONFIG_ROOT/"AUTHORITY_UNCHANGED_BINDINGS.json").read_text(encoding="utf-8"))
    def publication_binding(project,record,label):
        # Compare to publication HEAD, not obsolete pre-publication bindings.
        relative=record["path"]
        if relative in inventory["files"]:
            assert registry.file_sha(project/relative,canonical_lf=True)==inventory["files"][relative]
        else:
            assert (project/relative).is_file()
    monkeypatch.setattr(authority,"_require_bound_file",publication_binding)
    assert registry.inherited_algorithm_config(PROJECT)==authority.load_representative_stage1_algorithm_config(PROJECT)
