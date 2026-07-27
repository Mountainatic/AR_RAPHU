from __future__ import annotations

import numpy as np

from ar_raphu.datasets.base import DynamicDataset
from ar_raphu.datasets.windowing import build_windowed_task
from ar_raphu.spectral.design import build_spectral_design
from ar_raphu.spectral.public_adapter import build_spectral_design_from_windows


def _dataset(x: np.ndarray, y: np.ndarray) -> DynamicDataset:
    n = len(x)
    return DynamicDataset(
        x=x[:, None],
        y=y[:, None],
        timestamps=np.arange(n),
        sequence_id=np.zeros(n, dtype=int),
        split=np.full(n, "train"),
        label_mask=np.ones((n, 1), dtype=bool),
        quality_mask=np.ones((n, 2), dtype=bool),
        feature_names=("x",),
        target_names=("y",),
        metadata={},
    )


def test_window_adapter_matches_contiguous_reference_design() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=64)
    y = rng.normal(size=64)
    dataset = _dataset(x, y)
    task = build_windowed_task(
        dataset,
        target=0,
        horizon=1,
        L_x=8,
        L_y=4,
        include_splits=("train",),
    )
    reference = build_spectral_design(
        dataset.x,
        target_indices=task.target_index,
        train_target_stop=len(dataset.x),
        horizon=1,
        L_x=8,
        lag_basis_count=6,
        amplitude_basis_count=7,
    )
    adapted = build_spectral_design_from_windows(
        task,
        basis_fit_x=dataset.x,
        lag_basis_count=6,
        amplitude_basis_count=7,
    )
    np.testing.assert_allclose(adapted.matrix, reference.matrix, rtol=0, atol=0)
    np.testing.assert_allclose(adapted.lag_gram, reference.lag_gram)
    np.testing.assert_allclose(
        adapted.amplitude_grams[0], reference.amplitude_grams[0]
    )


def test_basis_fit_rows_are_explicit_and_finite() -> None:
    x = np.linspace(-1.0, 1.0, 32)
    task = build_windowed_task(
        _dataset(x, x**2),
        target=0,
        horizon=1,
        L_x=4,
        L_y=2,
    )
    invalid = x[:, None].copy()
    invalid[-1] = np.nan
    try:
        build_spectral_design_from_windows(
            task,
            basis_fit_x=invalid,
            lag_basis_count=4,
            amplitude_basis_count=5,
        )
    except ValueError as error:
        assert "non-finite training value" in str(error)
    else:
        raise AssertionError("Non-finite basis fitting rows were accepted.")
