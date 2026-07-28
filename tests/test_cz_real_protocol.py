from __future__ import annotations

import numpy as np
import pytest

from ar_raphu.cz_real.linear import TrainScaler, target_indices, window_designs
from ar_raphu.cz_real.protocol import (
    FurnaceBLockedError,
    build_development_folds,
    confirmation_interval,
    load_furnace_b,
    purge_gap,
)


def test_frozen_split_indices_and_purge() -> None:
    folds = build_development_folds(L_x=32, L_y=32)
    assert [fold.nominal_train_stop for fold in folds] == [
        8041,
        10051,
        12061,
        14072,
    ]
    assert [fold.validation_stop for fold in folds] == [
        10051,
        12061,
        14072,
        16082,
    ]
    assert purge_gap(L_x=32, L_y=32, h_max=60) == 91
    assert all(
        fold.effective_train_stop == fold.nominal_train_stop - 91
        for fold in folds
    )
    assert confirmation_interval() == (16082, 20103)


def test_furnace_b_is_rejected_before_r7_without_touching_path() -> None:
    with pytest.raises(FurnaceBLockedError):
        load_furnace_b(
            "/path/does/not/need/to/exist.xlsx",
            protocol_frozen=False,
            stage="R1",
        )
    with pytest.raises(FurnaceBLockedError):
        load_furnace_b(
            "/path/does/not/need/to/exist.xlsx",
            protocol_frozen=True,
            stage="R6",
        )


def test_direct_windows_stop_at_prediction_origin() -> None:
    samples = 200
    x = np.column_stack(
        [np.arange(samples, dtype=np.float64) + offset for offset in range(5)]
    )
    y = np.arange(samples, dtype=np.float64)
    scaler = TrainScaler.fit(x, y, 100)
    targets = target_indices(start=120, stop=140, horizon=15, max_history=32)
    x_design, y_design = window_designs(
        x,
        y,
        targets=targets,
        horizon=15,
        L_x=32,
        L_y=16,
        scaler=scaler,
    )
    assert x_design.shape == (20, 32 * 5)
    assert y_design.shape == (20, 16)
    first_origin = int(targets[0] - 15)
    recovered_current_y = y_design[0, 0] * scaler.y_scale + scaler.y_mean
    assert recovered_current_y == y[first_origin]
    assert first_origin < targets[0]
