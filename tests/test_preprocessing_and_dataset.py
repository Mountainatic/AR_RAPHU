from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.data_protocol import FinalLockboxError  # noqa: E402
from ar_raphu.dataset import CZDirectForecastDataset  # noqa: E402
from ar_raphu.preprocessing import (  # noqa: E402
    TrainOnlyStandardizer,
    v20_grid_ranges_from_train,
)
from ar_raphu.protocol_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    ProtocolNotFrozenError,
    load_protocol_config,
)


def synthetic_cz_arrays() -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(20_103, dtype=np.float64)
    x = np.stack(
        [time * (channel + 1) / 1000.0 for channel in range(9)], axis=1
    )
    x[:, 6] = 17.0
    y = 0.25 * time + np.sin(time / 17.0)
    return x, y


def test_scaler_fit_is_invariant_to_validation_and_test_poisoning() -> None:
    x, y = synthetic_cz_arrays()
    baseline = TrainOnlyStandardizer.fit(x, y, fold_number=1)
    poisoned_x = x.copy()
    poisoned_y = y.copy()
    poisoned_x[10051:] = np.nan
    poisoned_y[10051:] = np.nan
    poisoned = TrainOnlyStandardizer.fit(
        poisoned_x, poisoned_y, fold_number=1
    )

    assert np.array_equal(baseline.x_mean, poisoned.x_mean)
    assert np.array_equal(baseline.x_scale, poisoned.x_scale)
    assert baseline.y_mean == poisoned.y_mean
    assert baseline.y_scale == poisoned.y_scale
    assert baseline.fitted_interval.start == 0
    assert baseline.fitted_interval.stop == 10051


def test_constant_argon_standardizes_to_zero_and_grid_uses_train_only() -> None:
    x, y = synthetic_cz_arrays()
    scaler = TrainOnlyStandardizer.fit(x, y, fold_number=1)
    ranges_x, range_y = v20_grid_ranges_from_train(
        x, y, scaler, fold_number=1
    )

    transformed_argon = scaler.transform_x(x[:10051], fold_number=1)[:, 6]
    assert np.count_nonzero(transformed_argon) == 0
    assert ranges_x[6] == (-0.05, 0.05)
    assert len(ranges_x) == 9
    assert range_y[0] < range_y[1]


def test_scaler_cannot_be_reused_across_folds() -> None:
    x, y = synthetic_cz_arrays()
    scaler = TrainOnlyStandardizer.fit(x, y, fold_number=1)
    with pytest.raises(ValueError, match="belongs to fold 1"):
        scaler.transform_x(x[:2], fold_number=2)


def test_lazy_dataset_alignment_and_namespace() -> None:
    x, y = synthetic_cz_arrays()
    scaler = TrainOnlyStandardizer.fit(x, y, fold_number=1)
    dataset = CZDirectForecastDataset(
        x,
        y,
        track="XAR",
        fold_number=1,
        partition="validation",
        L_x=128,
        L_y=16,
        horizon=30,
        scaler=scaler,
    )
    sample = dataset.sample_indices(0)
    item = dataset[0]

    assert sample.target == 10051
    assert sample.origin == 10021
    assert sample.x_stop - 1 == sample.origin
    assert sample.y_stop - 1 == sample.origin
    assert sample.origin < sample.target
    assert item["x_window"].shape == (9, 128)
    assert item["y_window"].shape == (16,)
    assert item["origin_index"].item() == 10021
    assert item["target_index"].item() == 10051
    assert dataset.identity.cache_key.startswith("PRIVATE_CZ__")


def test_track_dataset_does_not_emit_forbidden_input_branch() -> None:
    x, y = synthetic_cz_arrays()
    scaler = TrainOnlyStandardizer.fit(x, y, fold_number=1)
    x_only = CZDirectForecastDataset(
        x,
        y,
        track="X",
        fold_number=1,
        partition="train",
        L_x=32,
        L_y=4,
        horizon=1,
        scaler=scaler,
    )[0]
    ar_only = CZDirectForecastDataset(
        x,
        y,
        track="AR",
        fold_number=1,
        partition="train",
        L_x=32,
        L_y=4,
        horizon=1,
        scaler=scaler,
    )[0]

    assert "x_window" in x_only and "y_window" not in x_only
    assert "y_window" in ar_only and "x_window" not in ar_only


def test_fold4_lockbox_is_enforced_before_any_dataset_item_exists() -> None:
    x, y = synthetic_cz_arrays()
    scaler = TrainOnlyStandardizer.fit(x, y, fold_number=4)
    with pytest.raises(FinalLockboxError):
        CZDirectForecastDataset(
            x,
            y,
            track="XAR",
            fold_number=4,
            partition="test",
            L_x=32,
            L_y=32,
            horizon=1,
            scaler=scaler,
        )


def test_development_dataset_does_not_inspect_fold4_values() -> None:
    x, y = synthetic_cz_arrays()
    scaler = TrainOnlyStandardizer.fit(x, y, fold_number=1)
    x[18092:] = np.nan
    y[18092:] = np.nan

    dataset = CZDirectForecastDataset(
        x,
        y,
        track="XAR",
        fold_number=1,
        partition="test",
        L_x=32,
        L_y=32,
        horizon=60,
        scaler=scaler,
    )
    assert len(dataset) > 0
    assert dataset.sample_indices(-1).target == 14071


def test_single_config_is_valid_yaml_subset_and_phase1_gate_is_frozen() -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["format"] == "JSON-compatible YAML 1.2"
    config = load_protocol_config()
    assert config["phase1_gate"]["status"] == "FROZEN"
    assert config["phase1_gate"]["unresolved"] == []
    assert load_protocol_config(require_phase1_frozen=True)["phase1_gate"][
        "status"
    ] == "FROZEN"
