from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prism_benchmark.c4_prism import channel_class
from prism_benchmark.cpu_data import BaseAccessor
from prism_benchmark.cz_extension import (
    BREAKPOINTS,
    RAW_TO_CANONICAL,
    RodSegment,
    SOURCE_SPLIT_PURGE_STEPS,
    W0_STEPS,
    _aggregate_segments,
    _fit_joint_lift,
    _read_workbook,
    _segment_ranges,
    _split_source_samples,
    build_direction,
)
from prism_benchmark.neural3 import (
    FINAL_SEEDS,
    PARAMETER_BUDGET,
    Candidate,
    ITransformerModel,
    Scaler,
    TimeMixerModel,
    _fit_fixed_epochs,
    _partition_candidate_support,
    _predict,
    build_model,
    materialize_model,
    materialize_sequences,
    native_support,
    parameter_count,
    support_hash,
)
from prism_benchmark.v211_support import SUPPORT_CONTRACT, support_id_hash
from prism_benchmark.six_dataset_package import _small_files
from prism_benchmark.six_dataset_extension import _support_record
from prism_benchmark.six_dataset_reporting import (
    PredictionSpec,
    _bootstrap_pair,
    _old_specs,
    _stable_hash,
)
from run_cz_neural3_extension import _cz_view


def _raw_frame(rows: int, offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float64)
    return pd.DataFrame(
        {
            "主加热功率": offset + 1.0 + index,
            "晶升速度": offset + 2.0 + 0.5 * index,
            "埚升速度": offset + 3.0 + 0.25 * index,
            "晶转速度": offset + 4.0 + 0.1 * index,
            "埚转速度": offset + 5.0 + 0.2 * index,
            "晶体直径": offset + 6.0 + 0.05 * index,
        }
    )


def _native_samples(origins: list[int], latest: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "base_origin_id": [f"b{value}" for value in origins],
            "view_sample_id": [f"v{value}" for value in origins],
            "entity_id": ["segment"] * len(origins),
            "origin": origins,
            "latest_available_target_index": latest,
            "causal_history_floor": [3] * len(origins),
            "anchor_history_steps": [0] * len(origins),
            "sample_support_contract": [SUPPORT_CONTRACT] * len(origins),
            "y_true": np.linspace(0.0, 1.0, len(origins)),
        }
    )


def _accessor(rows: int = 32) -> BaseAccessor:
    accessor = object.__new__(BaseAccessor)
    index = np.arange(rows, dtype=np.int64)
    accessor.entities = {
        "segment": (
            index,
            {
                "u": index.astype(np.float64),
                "target": (100.0 + index).astype(np.float64),
            },
        )
    }
    accessor.columns = frozenset({"u", "target"})
    accessor._target_column = "target"
    accessor._value_prefix = {}
    accessor._count_prefix = {}
    return accessor


def test_workbook_rods_follow_sheet_order_contract(tmp_path: Path) -> None:
    workbook = tmp_path / "cz.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        _raw_frame(8).to_excel(writer, sheet_name="Sheet1", index=False)
        _raw_frame(9, 100.0).to_excel(writer, sheet_name="Sheet2", index=False)
    rods = _read_workbook(workbook)
    assert list(rods) == ["Rod_1", "Rod_2"]
    assert len(rods["Rod_1"]) == 8
    assert len(rods["Rod_2"]) == 9
    assert rods["Rod_2"]["晶体直径"].iloc[0] > rods["Rod_1"]["晶体直径"].iloc[0]


def test_breakpoints_and_causal_aggregation_do_not_cross_segments(
    monkeypatch,
) -> None:
    assert _segment_ranges(15, (5,)) == [(0, 5), (5, 15)]
    monkeypatch.setitem(BREAKPOINTS, "Rod_1", (5,))
    raw = _raw_frame(15)
    segments = _aggregate_segments("Rod_1", raw)
    assert [len(item.frame) for item in segments] == [1, 2]
    expected_first = raw.iloc[:5][list(RAW_TO_CANONICAL)].mean().to_numpy()
    expected_second = raw.iloc[5:10][list(RAW_TO_CANONICAL)].mean().to_numpy()
    observed_first = segments[0].frame[list(RAW_TO_CANONICAL.values())].iloc[0]
    observed_second = segments[1].frame[list(RAW_TO_CANONICAL.values())].iloc[0]
    np.testing.assert_allclose(observed_first, expected_first)
    np.testing.assert_allclose(observed_second, expected_second)


def test_joint_lift_pca_is_train_only_and_sign_deterministic() -> None:
    rows = 20
    frame = pd.DataFrame(
        {
            "row_in_entity": np.arange(rows, dtype=np.int64),
            "crystal_lift": np.linspace(0.0, 4.0, rows),
            "crucible_lift": np.linspace(1.0, 3.0, rows),
        }
    )
    segment = RodSegment("Rod_1", "segment", 0, rows, frame)
    allowed = {"segment": set(range(1, 11))}
    first = _fit_joint_lift([segment], allowed)
    changed = frame.copy()
    changed.loc[10:, ["crystal_lift", "crucible_lift"]] += 10000.0
    second = _fit_joint_lift(
        [RodSegment("Rod_1", "segment", 0, rows, changed)],
        allowed,
    )
    np.testing.assert_allclose(first["mean"], second["mean"])
    np.testing.assert_allclose(first["scale"], second["scale"])
    np.testing.assert_allclose(first["pc1_loading"], second["pc1_loading"])
    assert first["pc1_loading"][0] >= 0.0
    assert first["source_training_rows"] == 10


def test_direction_build_keeps_target_rod_out_of_development(tmp_path: Path) -> None:
    raw_rods = {
        "Rod_1": _raw_frame(3000),
        "Rod_2": _raw_frame(3000, 100.0),
    }
    result = build_direction(
        raw_rods,
        source_rod="Rod_1",
        target_rod="Rod_2",
        output_root=tmp_path,
        direction="Rod_1_to_Rod_2",
    )
    assert result["source_rod"] == "Rod_1"
    root = tmp_path / "Rod_1_to_Rod_2"
    relative = Path("sample_ids/CZ_D20/input_only/record_time/primary")
    train = pd.read_parquet(root / relative / "train.parquet")
    validation = pd.read_parquet(root / relative / "validation.parquet")
    test = pd.read_parquet(root / relative / "test.parquet")
    assert train["entity_id"].str.startswith("Rod_1").all()
    assert validation["entity_id"].str.startswith("Rod_1").all()
    assert test["entity_id"].str.startswith("Rod_2").all()
    assert set(train["base_origin_id"]).isdisjoint(test["base_origin_id"])
    assert set(train["anchor_history_steps"]) == {W0_STEPS}
    assert (
        train["dependency_start"]
        == train["origin"] - W0_STEPS
    ).all()


def test_source_split_has_dependency_safe_purge() -> None:
    rows = 5000
    segment = RodSegment(
        "Rod_1",
        "segment",
        0,
        rows,
        pd.DataFrame(index=np.arange(rows)),
    )
    train, validation = _split_source_samples([segment])
    assert train["segment"]
    assert validation["segment"]
    separation = min(validation["segment"]) - max(train["segment"])
    assert separation >= SOURCE_SPLIT_PURGE_STEPS


def test_cz_channels_use_registered_medium_profile_class() -> None:
    assert channel_class("cz_czochralski", "main_heater_power") == "MEDIUM"
    assert channel_class("cz_czochralski", "joint_lift") == "MEDIUM"


def test_dynamic_native_support_checks_mature_target_history() -> None:
    samples = _native_samples([6, 7, 8], [4, 5, 6])
    input_only = native_support(samples, 3)
    dynamic = native_support(samples, 3, dynamic=True)
    assert input_only["origin"].tolist() == [6, 7, 8]
    assert dynamic["origin"].tolist() == [7, 8]


def test_short_lookback_fit_and_common_scoring_support_are_distinct() -> None:
    samples = _native_samples(list(range(4, 12)), list(range(3, 11)))
    short = native_support(samples, 2)
    long = native_support(samples, 5)
    common = native_support(samples, max(2, 5))
    assert len(short) > len(long)
    assert support_hash(common) == support_hash(long)


def test_unavailable_long_lookback_does_not_remove_short_candidates() -> None:
    train = _native_samples(list(range(4, 12)), list(range(3, 11)))
    validation = _native_samples(list(range(4, 10)), list(range(3, 9)))
    candidates = [
        Candidate("LSTM", 2, 2, "SMALL", 1e-3),
        Candidate("LSTM", 4, 5, "SMALL", 1e-3),
        Candidate("LSTM", 8, 20, "SMALL", 1e-3),
    ]
    available, unavailable = _partition_candidate_support(
        candidates, train, validation, dynamic=False
    )
    assert [candidate.lookback_hours for candidate, _fit in available] == [2, 4]
    assert [item["lookback_hours"] for item in unavailable] == [8]
    assert unavailable[0]["reasons"] == [
        "EMPTY_NATIVE_TRAIN_SUPPORT",
        "EMPTY_NATIVE_VALIDATION_SUPPORT",
    ]


def test_sequence_materialization_is_past_only_for_inputs_and_targets() -> None:
    accessor = _accessor()
    samples = _native_samples([10], [8])
    input_values = materialize_sequences(
        accessor, samples, ["u"], 4, dynamic=False
    )
    dynamic_values = materialize_sequences(
        accessor, samples, ["u"], 4, dynamic=True
    )
    np.testing.assert_allclose(input_values[0, :, 0], [7.5])
    np.testing.assert_allclose(dynamic_values[0, :, 1], [106.5])
    accessor.entities["segment"][1]["u"][10:] = 99999.0
    repeated = materialize_sequences(
        accessor, samples, ["u"], 4, dynamic=False
    )
    np.testing.assert_array_equal(repeated, input_values)


def test_itransformer_uses_variable_tokens_and_timemixer_shapes() -> None:
    itransformer = ITransformerModel(7, "SMALL")
    timemixer = TimeMixerModel(7, "SMALL")
    assert itransformer.variable_embedding.shape[1] == 7
    for tokens in (1, 2, 3, 16):
        values = torch.randn(3, tokens, 7)
        assert itransformer(values).shape == (3,)
        assert timemixer(values).shape == (3,)


def test_all_frozen_capacities_stay_within_parameter_budget() -> None:
    for model_name in ("LSTM", "iTransformer", "TimeMixer"):
        for capacity in ("SMALL", "MEDIUM"):
            model = build_model(model_name, 64, capacity)
            assert parameter_count(model) <= PARAMETER_BUDGET


def test_fixed_epoch_training_is_deterministic_and_prediction_needs_no_y() -> None:
    accessor = _accessor()
    samples = _native_samples(list(range(5, 13)), list(range(4, 12)))
    candidate = Candidate("LSTM", 2, 4, "SMALL", 1e-3)
    scaler = Scaler(np.array([0.0]), np.array([1.0]), 0.0, 1.0)
    first, first_result = _fit_fixed_epochs(
        candidate,
        accessor,
        samples,
        ["u"],
        scaler,
        dynamic=False,
        seed=FINAL_SEEDS[0],
        epochs=1,
        device=torch.device("cpu"),
    )
    second, second_result = _fit_fixed_epochs(
        candidate,
        accessor,
        samples,
        ["u"],
        scaler,
        dynamic=False,
        seed=FINAL_SEEDS[0],
        epochs=1,
        device=torch.device("cpu"),
    )
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key])
    no_target = samples.drop(columns=["y_true"])
    prediction = _predict(
        first,
        accessor,
        no_target,
        ["u"],
        4,
        scaler,
        dynamic=False,
        device=torch.device("cpu"),
    )
    assert prediction.shape == (len(samples),)
    assert first_result["epochs_run"] == second_result["epochs_run"] == 1
    assert FINAL_SEEDS == (20260817, 20260818, 20260819)


def test_cz_view_respects_requested_run_root(tmp_path: Path) -> None:
    shared, view = _cz_view(tmp_path, "Rod_1_to_Rod_2", "dynamic")
    assert shared == tmp_path / "shared" / "Rod_1_to_Rod_2"
    assert view.information_set == "dynamic"
    assert view.relative_root.as_posix().startswith("CZ_D20/")



def test_final_materialization_uses_frozen_epoch_three_seed_ensemble(
    tmp_path: Path,
) -> None:
    build_direction(
        {
            "Rod_1": _raw_frame(3000),
            "Rod_2": _raw_frame(3000, 100.0),
        },
        source_rod="Rod_1",
        target_rod="Rod_2",
        output_root=tmp_path / "shared",
        direction="Rod_1_to_Rod_2",
    )
    shared, view = _cz_view(
        tmp_path, "Rod_1_to_Rod_2", "input_only"
    )
    candidate = Candidate("LSTM", 2, 4, "SMALL", 1e-3)
    count = parameter_count(build_model("LSTM", 4, "SMALL"))
    selection = {
        "status": "PASS",
        "support_contract": SUPPORT_CONTRACT,
        "test_accessed": False,
        "selected_profile": {
            "model": "LSTM",
            "lookback_hours": 2,
            "history_steps": 4,
            "capacity": "SMALL",
            "learning_rate": 1e-3,
            "parameter_count": count,
        },
        "selected_candidate": {
            "candidate_id": candidate.candidate_id,
            "best_epoch": 1,
        },
        "scaler": {
            "feature_mean": [0.0, 0.0, 0.0, 0.0],
            "feature_scale": [10000.0, 10000.0, 10000.0, 10000.0],
            "target_mean": 0.0,
            "target_scale": 100.0,
        },
    }
    result = materialize_model(
        shared=shared,
        view=view,
        selection=selection,
        split="test",
        output=tmp_path / "results",
        device=torch.device("cpu"),
        final_fit=True,
    )
    prediction_path = Path(result["prediction_path"])
    prediction = pd.read_parquet(prediction_path)
    seed_columns = [f"y_pred_seed_{seed}" for seed in FINAL_SEEDS]
    assert result["frozen_best_epoch"] == 1
    assert result["final_seeds"] == list(FINAL_SEEDS)
    assert len(result["seed_results"]) == 3
    assert all(item["epochs_run"] == 1 for item in result["seed_results"])
    assert prediction_path.is_file()
    assert set(seed_columns).issubset(prediction.columns)
    np.testing.assert_allclose(
        prediction["y_pred"],
        prediction[seed_columns].mean(axis=1),
    )
    assert result["test_rows_used_for_fitting"] is False
    assert result["evaluation_targets_used_for_training"] is False


def test_prediction_catalog_parsing_preserves_symlink_relative_paths(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "actual" / "HEAD" / "input_only" / "record_time" / "primary"
    actual.mkdir(parents=True)
    prediction = actual / "MODEL.parquet"
    pd.DataFrame({"sample_id": ["s0"]}).to_parquet(prediction, index=False)
    root = tmp_path / "root"
    root.mkdir()
    (root / "HEAD").symlink_to(
        tmp_path / "actual" / "HEAD",
        target_is_directory=True,
    )

    specs = _old_specs(
        root,
        scope="public5",
        direction=None,
        split="test",
    )

    assert len(specs) == 1
    assert specs[0].target_head == "HEAD"
    assert specs[0].path.relative_to(root).parts[:2] == ("HEAD", "input_only")


def test_support_record_uses_native_support_hash_identity(tmp_path: Path) -> None:
    _, view = _cz_view(tmp_path, "Rod_1_to_Rod_2", "input_only")
    shared = tmp_path / "shared"
    sample_dir = shared / "sample_ids" / view.relative_root
    sample_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "base_origin_id": ["b2", "b1"],
            "view_sample_id": ["v2", "v1"],
            "dataset": "cz_czochralski",
            "entity_id": "Rod_1",
            "task_id": view.head.task_id,
            "target_head": view.head.head_id,
            "split": "train",
            "origin": [2, 1],
            "dependency_start": [1, 0],
            "dependency_stop_exclusive": [3, 2],
            "latest_available_target_index": [1, 0],
            "causal_history_floor": [0, 0],
            "anchor_history_steps": [0, 0],
            "sample_support_contract": SUPPORT_CONTRACT,
        }
    )
    frame.to_parquet(sample_dir / "train.parquet", index=False)
    empty_ood = frame.iloc[0:0].copy()
    empty_ood.to_parquet(sample_dir / "ood.parquet", index=False)

    record = _support_record(
        shared,
        view,
        (),
        scope="cz",
        direction="Rod_1_to_Rod_2",
    )
    observed = record["splits"]["train"]["support_hash"]

    assert observed == support_id_hash(frame)
    assert observed != _stable_hash(frame["view_sample_id"])
    assert support_id_hash(frame.iloc[::-1].reset_index(drop=True)) != observed
    assert record["splits"]["ood"] == {
        "rows": 0,
        "source_rows": 0,
        "support_hash": support_id_hash(empty_ood),
        "support_contract": SUPPORT_CONTRACT,
    }


def test_bootstrap_finite_sample_p_value_is_never_zero() -> None:
    left = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(8)],
            "entity_id": "entity",
            "origin": np.arange(8),
            "y_true": np.zeros(8),
            "y_pred": np.zeros(8),
        }
    )
    right = left.copy()
    right["y_pred"] = 1.0

    result = _bootstrap_pair(
        left,
        right,
        metadata={"seed": 1234},
        block_length=2,
        family="TEST",
    )

    assert result["raw_p_value"] > 0.0
    assert result["probability_candidate_better"] > 0.0


def test_small_file_package_selection_excludes_large_and_raw_payloads(
    tmp_path: Path,
) -> None:
    final = tmp_path / "final"
    final.mkdir()
    keep = final / "summary.json"
    keep.write_text("{}", encoding="utf-8")
    raw = final / "raw.xlsx"
    raw.write_bytes(b"raw")
    prediction = final / "prediction.parquet"
    prediction.write_bytes(b"prediction")
    large = final / "large.json"
    large.write_bytes(b"x" * (2 * 1024 * 1024 + 1))

    selected = _small_files(tmp_path)

    assert keep in selected
    assert raw not in selected
    assert prediction not in selected
    assert large not in selected
