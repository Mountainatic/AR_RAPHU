from __future__ import annotations

import numpy as np

from ar_raphu.baselines.arx_champneys2024 import (
    fit_and_select_arx_history,
    simulate_arx,
)
from ar_raphu.datasets.base import DynamicDataset


def _arx_sequence(seed: int, n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (
            0.65 * y[t - 1]
            - 0.15 * y[t - 2]
            + 0.7 * x[t - 1]
            + 0.05 * rng.normal()
        )
    return x, y


def _dataset() -> DynamicDataset:
    records = [_arx_sequence(seed) for seed in range(4)]
    x = np.concatenate([record[0] for record in records])[:, None]
    y = np.concatenate([record[1] for record in records])[:, None]
    sequence = np.concatenate(
        [
            np.full(len(record[0]), f"{index:04d}:r{index}", dtype=object)
            for index, record in enumerate(records)
        ]
    )
    split = np.concatenate(
        [
            np.full(len(record[0]), "train" if index < 3 else "validation")
            for index, record in enumerate(records)
        ]
    )
    return DynamicDataset(
        x=x,
        y=y,
        timestamps=np.arange(len(x), dtype=np.float64),
        sequence_id=sequence,
        split=split,
        label_mask=np.ones_like(y, dtype=bool),
        quality_mask=np.ones((len(x), 2), dtype=bool),
        feature_names=("input",),
        target_names=("output",),
        metadata={"dataset_id": "fixture"},
    )


def test_lfilter_simulation_matches_manual_arx_recursion() -> None:
    x, y = _arx_sequence(10, n=100)
    ay = np.array([0.65, -0.15])
    bx = np.array([0.7])
    prediction, burn = simulate_arx(
        x, y, coefficients_y=ay, coefficients_x=bx
    )
    manual = y.copy()
    for t in range(burn, len(y)):
        manual[t] = ay @ manual[t - np.arange(1, len(ay) + 1)] + bx @ x[
            t - 1 - np.arange(len(bx))
        ]
    np.testing.assert_allclose(prediction[burn:], manual[burn:])


def test_arx_aic_selection_runs_without_test_and_recovers_small_history() -> None:
    selection = fit_and_select_arx_history(_dataset(), max_nx=4, max_ny=4)
    assert len(selection.candidates) == 16
    assert selection.selected_nx <= 2
    assert selection.selected_ny <= 3
    selected = next(
        candidate
        for candidate in selection.candidates
        if (candidate.nx, candidate.ny)
        == (selection.selected_nx, selection.selected_ny)
    )
    assert selected.stable_simulation is True
    assert np.isfinite(selected.validation_aic_mean)
