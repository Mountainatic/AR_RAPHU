from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.linear_q import fit_block_ridge, fit_ridge
from src.multiresolution_lags import expand_lag_blocks, lag_block_matrix
from src.resampling import PCA1Transform, causal_block_average, causal_ema
from src.runtime import load_config, task_is_complete
from src.segmentation import Segment, build_segments, detect_breakpoints
from src.targets import build_target_rows
from src.timebase import Timebase
from src.validation import rolling_origin_folds


ROOT = Path(__file__).resolve().parents[1]


def test_timebase_requires_exact_physical_alignment():
    timebase = Timebase(2.0)
    assert timebase.samples_for_minutes(0.5) == 15
    assert timebase.cadence_step(10.0) == 5
    with pytest.raises(ValueError, match="PHYSICAL_TIME_NOT_ALIGNED"):
        timebase.samples_for_seconds(5.0)


def test_breakpoints_and_segments_are_hard_boundaries():
    y = np.array([0.0, 0.0, 1.0, 1.0, -1.0, -1.0])
    assert detect_breakpoints(y, 0.5) == [2, 4]
    segments = build_segments(len(y), [2, 4])
    assert [(item.start, item.stop, item.role) for item in segments] == [
        (0, 2, "sensitivity"),
        (2, 4, "sensitivity"),
        (4, 6, "main"),
    ]


def test_target_exact_formula_and_no_future_input():
    y = np.arange(1000, dtype=np.float64)
    rows = build_target_rows(
        y,
        Segment(100, 900, "main"),
        timebase=Timebase(2.0),
        cadence_sec=10.0,
        horizon_min=1.0,
        target_window_min=0.5,
        history_min=2.0,
    )
    origin = int(rows.origins[0])
    expected = y[origin + 30 : origin + 45].mean() - y[
        origin - 14 : origin + 1
    ].mean()
    assert rows.target[0] == expected
    assert origin < origin + rows.horizon_samples
    assert origin - rows.history_samples + 1 >= 100


def test_causal_block_average_uses_only_past():
    values = np.arange(20, dtype=np.float64)
    origins = np.array([10, 11])
    observed = causal_block_average(
        values, origins, start_samples=0, stop_samples=3
    )
    np.testing.assert_allclose(observed, [9.0, 10.0])


def test_lag_grid_expansion_and_matrix():
    blocks = expand_lag_blocks(
        [[0, 2, 0.5], [2, 10, 2]], history_min=6
    )
    assert len(blocks) == 6
    values = np.arange(1000, dtype=np.float64)
    matrix = lag_block_matrix(
        values,
        np.array([500]),
        blocks,
        timebase=Timebase(2.0),
    )
    assert matrix.shape == (1, 6)


def test_pca_and_ema_are_deterministic():
    values = np.column_stack((np.arange(50.0), 2.0 * np.arange(50.0)))
    pca = PCA1Transform.fit(values[:25])
    np.testing.assert_allclose(pca.transform(values), pca.transform(values))
    ema = causal_ema(np.array([0.0, 1.0, 1.0]), tau_samples=2.0)
    assert ema[0] == 0.0
    assert 0.0 < ema[1] < ema[2] < 1.0


def test_ridge_kkt_and_exact_zero_q_nestedness():
    rng = np.random.default_rng(8)
    ar = rng.normal(size=(200, 3))
    q = rng.normal(size=(200, 2))
    y = 1.2 * ar[:, 0] - 0.3 * ar[:, 1] + 0.01 * rng.normal(size=200)
    ridge = fit_ridge(ar, y, alpha=1.0e-4)
    nested = fit_block_ridge(ar, q, y, alpha_ar=1.0e-4, alpha_q=None)
    np.testing.assert_allclose(ridge.predict(ar), nested.predict(ar, q))
    assert ridge.relative_kkt <= 1.0e-8
    assert nested.relative_kkt <= 1.0e-8
    assert nested.q_coefficients.size == 0


def test_rolling_folds_apply_physical_purge():
    origins = np.arange(100, 1100, 5)
    folds = rolling_origin_folds(
        origins,
        [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9]],
        purge_samples=40,
    )
    assert len(folds) == 4
    for fold in folds:
        assert origins[fold.train_indices].max() < (
            fold.validation_start_origin - 40
        )


def test_rolling_folds_shift_right_using_geometry_only():
    origins = np.arange(0, 330, 2)
    folds = rolling_origin_folds(
        origins,
        [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9]],
        purge_samples=135,
    )
    assert len(folds) == 4
    assert len(folds[0].train_indices) >= 20
    assert folds[-1].validation_indices[-1] < len(origins)


def test_checkpoint_identity_is_strict(tmp_path):
    config_path = ROOT / "configs" / "experiment_v1.yaml"
    _, config_hash = load_config(config_path)
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "status": "COMPLETED",
        "config_sha256": config_hash,
        "data_sha256": "abc",
        "sample_period_sec": 2.0,
    }), encoding="utf-8")
    assert task_is_complete(
        result,
        config_sha256=config_hash,
        data_sha256="abc",
        sample_period_sec=2.0,
    )
    assert not task_is_complete(
        result,
        config_sha256=config_hash,
        data_sha256="different",
        sample_period_sec=2.0,
    )
