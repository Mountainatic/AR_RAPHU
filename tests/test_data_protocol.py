from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.data_protocol import (  # noqa: E402
    CONDITIONAL_L_X_EXTENSION,
    FOLDS,
    FORMAL_L_X_CANDIDATES,
    FORMAL_L_Y_CANDIDATES,
    PREDICTION_HORIZONS,
    FinalLockboxError,
    direct_forecast_sample_count,
    iter_direct_forecast_samples,
)


def test_pre_registered_windows_and_horizons() -> None:
    assert FORMAL_L_Y_CANDIDATES == (1, 4, 8, 16, 32, 64)
    assert FORMAL_L_X_CANDIDATES == (32, 64, 128, 256)
    assert CONDITIONAL_L_X_EXTENSION == 512
    assert PREDICTION_HORIZONS == (1, 5, 10, 30, 60)


def test_exact_four_fold_boundaries() -> None:
    observed = [
        (
            fold.number,
            (fold.train.start, fold.train.stop),
            (fold.validation.start, fold.validation.stop),
            (fold.test.start, fold.test.stop),
            fold.role,
        )
        for fold in FOLDS
    ]
    assert observed == [
        (1, (0, 10051), (10051, 12061), (12061, 14072), "development"),
        (2, (0, 12061), (12061, 14072), (14072, 16082), "development"),
        (3, (0, 14072), (14072, 16082), (16082, 18092), "development"),
        (4, (0, 16082), (16082, 18092), (18092, 20103), "final_lockbox"),
    ]


@pytest.mark.parametrize("horizon", PREDICTION_HORIZONS)
def test_membership_is_by_target_and_no_future_information(horizon: int) -> None:
    first = next(
        iter_direct_forecast_samples(
            1,
            "test",
            L_x=256,
            L_y=64,
            horizon=horizon,
        )
    )

    assert first.target == 12061
    assert first.origin == first.target - horizon
    assert first.x_stop - first.x_start == 256
    assert first.y_stop - first.y_start == 64
    assert first.x_stop - 1 == first.origin
    assert first.y_stop - 1 == first.origin
    assert first.origin < first.target
    assert first.x_start < FOLDS[0].test.start
    assert first.y_start < FOLDS[0].test.start


def test_no_mechanical_boundary_drop_for_validation_or_test() -> None:
    for partition, expected_first_target in [
        ("validation", 10051),
        ("test", 12061),
    ]:
        first = next(
            iter_direct_forecast_samples(
                1,
                partition,
                L_x=256,
                L_y=64,
                horizon=60,
            )
        )
        assert first.target == expected_first_target


def test_training_prefix_is_dropped_only_when_history_is_unavailable() -> None:
    count = direct_forecast_sample_count(
        1,
        "train",
        L_x=256,
        L_y=64,
        horizon=60,
    )
    first = next(
        iter_direct_forecast_samples(
            1,
            "train",
            L_x=256,
            L_y=64,
            horizon=60,
        )
    )

    assert first.x_start == 0
    assert first.origin == 255
    assert first.target == 315
    assert count == 10051 - 315


def test_fold_4_test_is_locked_before_protocol_freeze() -> None:
    with pytest.raises(FinalLockboxError):
        next(
            iter_direct_forecast_samples(
                4,
                "test",
                L_x=32,
                L_y=32,
                horizon=1,
            )
        )

    with pytest.raises(FinalLockboxError):
        direct_forecast_sample_count(
            4,
            "test",
            L_x=32,
            L_y=32,
            horizon=1,
        )


def test_fold_4_test_unlock_is_explicit_and_still_leak_free() -> None:
    first = next(
        iter_direct_forecast_samples(
            4,
            "test",
            L_x=32,
            L_y=32,
            horizon=1,
            protocol_frozen=True,
        )
    )

    assert first.target == 18092
    assert first.origin == 18091
    assert first.x_stop == 18092
    assert first.y_stop == 18092
    assert first.target not in range(first.x_start, first.x_stop)
    assert first.target not in range(first.y_start, first.y_stop)


@pytest.mark.parametrize(
    ("L_x", "L_y", "horizon"),
    [(0, 32, 1), (32, 0, 1), (32, 32, 2)],
)
def test_invalid_protocol_values_are_rejected(
    L_x: int, L_y: int, horizon: int
) -> None:
    with pytest.raises(ValueError):
        next(
            iter_direct_forecast_samples(
                1,
                "train",
                L_x=L_x,
                L_y=L_y,
                horizon=horizon,
            )
        )
