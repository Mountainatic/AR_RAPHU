from __future__ import annotations

import numpy as np

from ar_raphu.baselines.pnarx_champneys2024 import (
    fit_and_select_pnarx,
    legendre_monomial_design,
    simulate_pnarx,
)
from ar_raphu.datasets.base import DynamicDataset


def _record(seed: int, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (
            0.5 * y[t - 1]
            + 0.3 * x[t - 1]
            + 0.2 * x[t - 1] ** 2
            + 0.01 * rng.normal()
        )
    return x, y


def _dataset() -> DynamicDataset:
    records = [_record(seed) for seed in range(4)]
    n = len(records[0][0])
    return DynamicDataset(
        x=np.concatenate([r[0] for r in records])[:, None],
        y=np.concatenate([r[1] for r in records])[:, None],
        timestamps=np.arange(4 * n, dtype=np.float64),
        sequence_id=np.concatenate(
            [np.full(n, f"{i:04d}:r{i}", dtype=object) for i in range(4)]
        ),
        split=np.concatenate(
            [
                np.full(
                    n,
                    "train" if i < 3 else "validation",
                    dtype=object,
                )
                for i in range(4)
            ]
        ),
        label_mask=np.ones((4 * n, 1), dtype=bool),
        quality_mask=np.ones((4 * n, 2), dtype=bool),
        feature_names=("input",),
        target_names=("output",),
        metadata={"dataset_id": "fixture"},
    )


def test_legendre_design_has_no_cross_terms() -> None:
    history = np.array([[0.2, -0.4]])
    design = legendre_monomial_design(history, order=2)
    expected = np.array(
        [[0.2, 0.5 * (3 * 0.2**2 - 1), -0.4, 0.5 * (3 * 0.4**2 - 1)]]
    )
    np.testing.assert_allclose(design, expected)


def test_free_run_alignment_uses_input_through_previous_origin() -> None:
    x = np.linspace(-0.5, 0.5, 50)
    y = np.zeros(50)
    coefficients = np.array([0.5, 0.3])
    prediction, burn = simulate_pnarx(
        x, y, nx=1, ny=1, order=1, coefficients=coefficients
    )
    manual = y.copy()
    for target in range(burn, len(y)):
        manual[target] = 0.5 * manual[target - 1] + 0.3 * x[target - 1]
    np.testing.assert_allclose(prediction[burn:], manual[burn:])


def test_pnarx_selects_a_finite_low_order_without_test() -> None:
    selection = fit_and_select_pnarx(
        _dataset(), nx=2, ny=2, orders=(2, 3, 4)
    )
    assert selection.order in {2, 3, 4}
    assert all(np.isfinite(row.validation_aic_mean) for row in selection.candidates)
    assert all(row.stable_simulation for row in selection.candidates)


def test_pnarx_accepts_one_contiguous_chronological_split() -> None:
    x, y = _record(31, n=800)
    split = np.full(800, "train", dtype=object)
    split[600:] = "validation"
    dataset = DynamicDataset(
        x=x[:, None],
        y=y[:, None],
        timestamps=np.arange(800, dtype=np.float64),
        sequence_id=np.full(800, "0000:estimation", dtype=object),
        split=split,
        label_mask=np.ones((800, 1), dtype=bool),
        quality_mask=np.ones((800, 2), dtype=bool),
        feature_names=("input",),
        target_names=("output",),
        metadata={"dataset_id": "fixture"},
    )
    selection = fit_and_select_pnarx(
        dataset, nx=2, ny=2, orders=(2, 3)
    )
    assert selection.order in {2, 3}
    assert all(row.stable_simulation for row in selection.candidates)
