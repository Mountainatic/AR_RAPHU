import json
from pathlib import Path

import numpy as np
import pandas as pd

from prism_benchmark.cpu_data import HeadSpec, ViewSpec, sha256_file
from prism_benchmark.v21_baselines import (
    INVENTORY_NAME,
    REPLAY_MANIFEST_NAME,
    REPLAY_STATUS,
    _baseline_frame,
    baseline_replay_scope,
    freeze_baseline_inventory,
    materialize_test_baselines,
)
from prism_benchmark.v21_config import (
    ACTIVE_HEADS,
    V21Paths,
    load_baseline_replay_amendment,
)
from prism_benchmark.v21_runner import CHAIN_STAGES
from scripts.run_prism_v21_chain import _complete, _marker


def _view(head_id: str = "SRU_H2S__H5__W1", information_set: str = "input_only") -> ViewSpec:
    target = "H2S" if "H2S" in head_id else "SO2"
    return ViewSpec(
        head=HeadSpec(
            head_id=head_id,
            task_id=f"SRU_{target}",
            dataset="sru",
            target=target,
            cadence_seconds=1.0,
            h_steps=5,
            w_steps=1,
            w0_steps=1,
            primary=True,
        ),
        information_set=information_set,
        availability_scenario="record_time",
        proxy_policy="primary",
    )


def _samples(split: str, view: ViewSpec) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "view_sample_id": [f"{view.information_set}-{split}-0", f"{view.information_set}-{split}-1"],
            "base_origin_id": [f"origin-{split}-0", f"origin-{split}-1"],
            "dataset": ["sru", "sru"],
            "entity_id": ["run", "run"],
            "task_id": [view.head.task_id, view.head.task_id],
            "target_head": [view.head.head_id, view.head.head_id],
            "split": [split, split],
            "origin": [0, 1],
            "y_true": [1.0, 2.0],
        }
    )


def _write_prediction(path: Path, split: str, view: ViewSpec) -> pd.DataFrame:
    samples = _samples(split, view)
    frame = _baseline_frame(
        samples,
        view,
        "MEAN",
        np.array([1.1, 1.9]),
        1,
        split,
        6,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def _frozen_internal_inventory(tmp_path: Path, view: ViewSpec) -> V21Paths:
    paths = V21Paths(
        project=tmp_path / "project",
        shared=tmp_path / "shared",
        output=tmp_path / "results_prism_v2_1_sru",
    )
    paths.plan.mkdir(parents=True)
    paths.baseline_amendment_path.write_text("{}", encoding="utf-8")
    configs = paths.project / "configs"
    configs.mkdir(parents=True)
    cpu_freeze = configs / "cpu_model_freeze_v1.json"
    c6_freeze = configs / "c6_full_final_v2.json"
    cpu_freeze.write_text("{}", encoding="utf-8")
    c6_freeze.write_text("{}", encoding="utf-8")
    final_root = paths.output / "BASELINES" / "REPLAY" / "FINAL_FIT"
    validation_path = final_root / "PREDICTIONS" / "validation.parquet"
    test_path = final_root / "PREDICTIONS" / "test.parquet"
    _write_prediction(validation_path, "validation", view)
    _write_prediction(test_path, "test", view)
    inventory = {
        "status": "BASELINE_INCLUSION_FROZEN",
        "baseline_root_relative_to_output": str(final_root.relative_to(paths.output)),
        "entries": [
            {
                "target_head": view.head.head_id,
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "model": "MEAN",
                "validation_path": str(validation_path.relative_to(final_root)),
                "validation_sha256": sha256_file(validation_path),
                "validation_mse": 0.01,
                "test_path": str(test_path.relative_to(final_root)),
                "test_sha256": sha256_file(test_path),
                "test_metrics_computed": False,
                "test_metrics_exposed_to_selection": False,
            }
        ],
        "failures_retained": [],
        "best_by_validation": {"key": "MEAN"},
        "test_metrics_computed": False,
        "test_metrics_exposed_to_selection": False,
    }
    inventory_path = paths.output / "BASELINES" / INVENTORY_NAME
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    manifest = {
        "status": REPLAY_STATUS,
        "historical_prediction_search_performed": False,
        "external_baseline_root_used": False,
        "inventory_sha256": sha256_file(inventory_path),
        "amendment_sha256": sha256_file(paths.baseline_amendment_path),
        "cpu_model_freeze_sha256": sha256_file(cpu_freeze),
        "c6_full_final_freeze_sha256": sha256_file(c6_freeze),
    }
    (paths.output / "BASELINES" / REPLAY_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return paths


def test_baseline_frame_has_pairing_schema_and_exact_sample_order():
    view = _view()
    samples = _samples("validation", view)
    frame = _baseline_frame(
        samples,
        view,
        "MEAN",
        np.array([1.1, 1.9]),
        1,
        "validation",
        6,
    )
    assert {"sample_id", "entity_id", "origin", "y_true", "y_pred"} <= set(frame)
    assert frame["sample_id"].tolist() == samples["view_sample_id"].tolist()
    assert frame["y_true"].tolist() == samples["y_true"].tolist()


def test_internal_inventory_never_searches_for_historical_predictions(tmp_path, monkeypatch):
    paths = _frozen_internal_inventory(tmp_path, _view())
    monkeypatch.setattr(
        "prism_benchmark.v21_baselines.load_baseline_replay_amendment",
        lambda _project: {"status": "frozen"},
    )
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("historical prediction search is forbidden")
        ),
    )
    inventory = freeze_baseline_inventory(paths)
    assert inventory["entries"][0]["model"] == "MEAN"
    assert "baseline_root" not in V21Paths.__dataclass_fields__


def test_baseline_test_rows_are_materialized_from_internal_replay(tmp_path, monkeypatch):
    view = _view()
    paths = _frozen_internal_inventory(tmp_path, view)
    monkeypatch.setattr(
        "prism_benchmark.v21_baselines.load_baseline_replay_amendment",
        lambda _project: {"status": "frozen"},
    )
    monkeypatch.setattr(
        "prism_benchmark.v21_baselines.sru_input_views",
        lambda _shared: [view],
    )
    monkeypatch.setattr(
        "prism_benchmark.v21_baselines.sru_dynamic_views",
        lambda _shared: [],
    )
    monkeypatch.setattr(
        "prism_benchmark.v21_baselines.load_samples",
        lambda _shared, _view, split: _samples(split, view),
    )
    audits = materialize_test_baselines(paths)
    assert audits[0]["test_accessed"] is True
    assert audits[0]["model"] == "MEAN"
    assert (paths.output / audits[0]["prediction_path"]).is_file()


def test_replay_scope_is_exactly_two_sru_h5_heads_and_no_ood(tmp_path, monkeypatch):
    input_views = [_view(head) for head in sorted(ACTIVE_HEADS)]
    dynamic_views = [_view(head, "dynamic") for head in sorted(ACTIVE_HEADS)]
    monkeypatch.setattr(
        "prism_benchmark.v21_baselines.sru_input_views",
        lambda _shared: input_views,
    )
    monkeypatch.setattr(
        "prism_benchmark.v21_baselines.sru_dynamic_views",
        lambda _shared: dynamic_views,
    )
    paths = V21Paths(tmp_path, tmp_path, tmp_path / "results_prism_v2_1_sru")
    scope = baseline_replay_scope(paths)
    assert scope["head_pairs"] == 2
    assert scope["heads"] == sorted(ACTIVE_HEADS)
    assert scope["materialized_splits"] == ["validation", "test"]
    assert scope["ood_splits"] == []
    assert scope["historical_prediction_search"] is False


def test_automatic_chain_starts_with_resumable_b0():
    assert CHAIN_STAGES == (
        "b0",
        "e0",
        "e1",
        "e2k",
        "e2c",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "e8",
    )


def test_b0_resume_marker_requires_frozen_status(tmp_path):
    marker = _marker(tmp_path, "b0")
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"status": REPLAY_STATUS}), encoding="utf-8")
    assert _complete(marker, "b0") is True
    marker.write_text(json.dumps({"status": "FAILED"}), encoding="utf-8")
    assert _complete(marker, "b0") is False


def test_user_authorized_test_generation_override_is_frozen():
    project = Path(__file__).resolve().parents[1]
    amendment = load_baseline_replay_amendment(project)
    assert amendment["historical_baseline_parquet"] == "NOT_AVAILABLE_NOT_SEARCHED_NOT_REQUIRED"
    assert amendment["splits_materialized_by_b0"] == ["validation", "test"]
    assert amendment["baseline_test_access"]["metrics_computed_in_b0"] is False
    assert amendment["baseline_test_access"]["selection_exposure"] is False
