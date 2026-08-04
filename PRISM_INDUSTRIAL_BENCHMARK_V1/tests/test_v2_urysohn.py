from __future__ import annotations

import numpy as np

from prism_benchmark.v2_urysohn import fit_contract, predict_contract


def test_neutral_and_linear_are_exactly_nested() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 6))
    y = x @ np.linspace(-0.5, 0.8, 6) + 0.2
    zero = fit_contract(x, y, "EXACT_ZERO", 4, (1e-4, 1e-3, 1e-3))
    linear = fit_contract(x, y, "LINEAR_DISTRIBUTED_LAG", 4, (1e-8, 0.0, 0.0))
    np.testing.assert_array_equal(predict_contract(x, zero), 0.0)
    assert np.mean((predict_contract(x, linear) - y) ** 2) < 1e-8


def test_rank_ladder_returns_rank_bounded_surface() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(700, 5))
    y = np.sin(x).sum(axis=1)
    model = fit_contract(
        x, y, "RANK_2_URYSOHN", 4, (1e-4, 1e-3, 1e-3),
        als_seeds=(11, 12, 13), als_max_iterations=30,
    )
    theta = np.asarray(model["theta"])
    assert np.linalg.matrix_rank(theta, tol=1e-8) <= 2
    assert np.isfinite(predict_contract(x, model)).all()
    assert model["certificate"]["fixed_support_refit"]
