from __future__ import annotations

import numpy as np

from prism_benchmark.v2_urysohn import fit_contract, predict_contract
from prism_benchmark.v2_numerics import (
    centered_sufficient_statistics,
    solve_centered_certified_gram,
    solve_certified,
)


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


def test_streamed_centered_gram_matches_dense_centered_solve() -> None:
    rng = np.random.default_rng(41)
    x = rng.normal(size=(35003, 17))
    y = rng.normal(size=len(x))
    penalty = np.eye(x.shape[1]) * 1e-3
    center = x.mean(axis=0, dtype=np.float64)
    y_mean = float(y.mean(dtype=np.float64))
    dense, _ = solve_certified(x - center, y - y_mean, penalty)
    statistics = centered_sufficient_statistics(x, y, chunk_rows=4096)
    streamed, intercept, certificate = solve_centered_certified_gram(*statistics, penalty)
    np.testing.assert_allclose(streamed, dense, rtol=5e-11, atol=5e-12)
    np.testing.assert_allclose(intercept, y_mean - center @ dense, rtol=5e-11, atol=5e-12)
    assert certificate.status in {"PASS", "PASS_WITH_WARNING"}
