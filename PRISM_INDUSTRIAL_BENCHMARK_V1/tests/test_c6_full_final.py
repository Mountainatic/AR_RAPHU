from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pandas as pd

from prism_benchmark.c3_models import _ridge_block_predict
from prism_benchmark.c6_full_final import (
    _base_partitions,
    _chunked_prediction,
    _linear_fit,
    _linear_predict,
    _metrics_and_statistics,
    _selected,
)


def test_repair_freeze_has_full_two_leaderboards() -> None:
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs/c6_full_final_v2.json").read_text(encoding="utf-8"))
    assert {"RIDGE", "PLS", "DPLS", "RBF_SVR", "XGBOOST", "PRISM_CHANNEL_SPECIFIC"}.issubset(
        config["leaderboards"]["input_only"]
    )
    assert {"AR", "ARX", "LINEAR_NARX", "N4SID", "PRISM_PHYSICS_FIRST", "PRISM_K_JOINT_AR"}.issubset(
        config["leaderboards"]["dynamic"]
    )
    assert "LOCAL_LINEAR_TREND" not in config["leaderboards"]["input_only"]
    assert config["cross_task_primary_rank_requires_complete_coverage"] is True


def test_contract_helpers_are_deterministic() -> None:
    assert _selected({"candidate": [2, 0.03]}) == [2, 0.03]
    assert _selected(4.0) == 4.0
    assert _base_partitions("validation") == ["train", "validation"]
    assert _base_partitions("ood") == ["train", "validation", "test", "ood"]
    assert np.isfinite(float(_selected({"candidate": 0.1})))


def test_chunked_prediction_preserves_order_and_linear_solver_values() -> None:
    rng = np.random.default_rng(7)
    x_fit = rng.normal(size=(80, 5))
    y_fit = rng.normal(size=80)
    x_eval = rng.normal(size=(23, 5))
    expected, _ = _ridge_block_predict(x_fit, y_fit, x_eval, 0.01)
    scaler, coefficient, y_mean, _ = _linear_fit(x_fit, y_fit, 0.01)
    samples = pd.DataFrame({"row": np.arange(len(x_eval))})
    actual = _chunked_prediction(
        samples,
        lambda chunk: x_eval[chunk["row"].to_numpy(dtype=np.int64)],
        lambda matrix: _linear_predict(scaler, coefficient, y_mean, matrix),
        chunk_rows=4,
        workers=3,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_streaming_metrics_pairs_models_without_loading_full_frames(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "PREDICTIONS" / "HEAD" / "test"
    prediction_dir.mkdir(parents=True)
    common = {
        "dataset": ["toy"] * 6,
        "task": ["toy_task"] * 6,
        "sample_id": [f"s{index}" for index in range(6)],
        "profile_id": ["a", "a", "a", "b", "b", "b"],
        "y_true": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        "h_steps": [1] * 6,
        "w_steps": [2] * 6,
        "core_history_steps": [4] * 6,
        "parameter_count": [1] * 6,
    }
    for model, prediction in {
        "PERSISTENCE": [0.0] * 6,
        "RIDGE": [0.0, 0.8, 1.8, 0.0, 0.8, 1.8],
    }.items():
        frame = pd.DataFrame({**common, "model": [model] * 6, "y_pred": prediction})
        frame.to_parquet(prediction_dir / f"input_only__{model}.parquet", index=False)
    config = {
        "leaderboards": {"input_only": ["PERSISTENCE", "RIDGE"]},
        "bootstrap_seed": 19,
        "bootstrap_replicates": 20,
        "holm_alpha": 0.05,
    }
    metrics, bootstrap, ranks, entities = _metrics_and_statistics(tmp_path, config, n_jobs=1)
    assert len(metrics) == 2
    assert len(bootstrap) == 3
    assert len(ranks) == 2
    assert len(entities) == 4
    assert metrics.set_index("model").loc["RIDGE", "mse"] < metrics.set_index("model").loc["PERSISTENCE", "mse"]
