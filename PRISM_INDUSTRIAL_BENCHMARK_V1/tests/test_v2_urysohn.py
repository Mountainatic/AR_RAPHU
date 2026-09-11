from __future__ import annotations

import numpy as np
import pytest

from prism_benchmark.v2_basis import tensor_design
from prism_benchmark.v2_urysohn import (
    DesignStatistics,
    _component_transform,
    _u_transform,
    _v_transform,
    fit_contract,
    fit_prepared_contract,
    predict_contract,
    predict_contract_from_design,
    prepare_contract_fit,
)
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


@pytest.mark.parametrize(
    ("family", "m_x"),
    [
        ("LINEAR_DISTRIBUTED_LAG", 1),
        ("RANK_1_URYSOHN", 4),
        ("RANK_2_URYSOHN", 4),
        ("FULL_FINITE_URYSOHN", 4),
    ],
)
def test_prepared_fit_is_numerically_equivalent_to_direct_fit(
    family: str, m_x: int
) -> None:
    rng = np.random.default_rng(20260911)
    train = rng.normal(size=(240, 5))
    validation = rng.normal(size=(91, 5))
    target = np.sin(train[:, 0]) + 0.25 * train[:, 2] - 0.1 * train[:, 4]
    kwargs = {
        "als_seeds": (17, 19),
        "als_max_iterations": 8,
        "als_tolerance": 1e-8,
        "als_max_increases": 5,
        "als_divergence_factor": 1e6,
    }
    lambdas = (1e-4, 1e-3, 1e-2)
    direct = fit_contract(train, target, family, m_x, lambdas, **kwargs)
    requested = 1 if family == "LINEAR_DISTRIBUTED_LAG" else m_x
    prepared = prepare_contract_fit(train, target, requested)
    cached = fit_prepared_contract(prepared, family, m_x, lambdas, **kwargs)

    assert cached["family"] == direct["family"]
    assert cached["basis"] == direct["basis"]
    assert cached["parameter_count"] == direct["parameter_count"]
    np.testing.assert_allclose(
        cached["theta"], direct["theta"], rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        cached["intercept"], direct["intercept"], rtol=2e-13, atol=2e-13
    )
    direct_prediction = predict_contract(validation, direct)
    prepared_design = tensor_design(validation, prepared.basis)
    cached_prediction = predict_contract_from_design(prepared_design, cached)
    np.testing.assert_allclose(
        cached_prediction, direct_prediction, rtol=2e-13, atol=2e-13
    )


def test_prepared_rank_fits_reuse_full_surface_initialization() -> None:
    rng = np.random.default_rng(29)
    values = rng.normal(size=(180, 4))
    target = rng.normal(size=len(values))
    prepared = prepare_contract_fit(values, target, 4)
    kwargs = {"als_seeds": (3,), "als_max_iterations": 3}
    lambdas = (1e-4, 1e-3, 1e-2)

    fit_prepared_contract(
        prepared, "RANK_1_URYSOHN", 4, lambdas, **kwargs
    )
    fit_prepared_contract(
        prepared, "RANK_2_URYSOHN", 4, lambdas, **kwargs
    )

    assert list(prepared._full_svd_cache) == [lambdas]
    assert list(prepared._full_solution_cache) == [lambdas]


def test_prepared_constant_channel_preserves_exact_zero_semantics() -> None:
    values = np.ones((120, 5), dtype=np.float64)
    target = np.linspace(-1.0, 1.0, len(values), dtype=np.float64)
    for family, m_x in (
        ("LINEAR_DISTRIBUTED_LAG", 1),
        ("RANK_1_URYSOHN", 4),
        ("FULL_FINITE_URYSOHN", 4),
    ):
        requested = 1 if family == "LINEAR_DISTRIBUTED_LAG" else m_x
        prepared = prepare_contract_fit(values, target, requested)
        assert prepared.phi.shape == (120, 5, 0)
        assert prepared.full_statistics.gram.shape == (0, 0)
        assert prepared.linear_statistics.gram.shape == (0, 0)
        contract = fit_prepared_contract(
            prepared, family, m_x, (1e-4, 1e-3, 1e-2)
        )
        assert contract["family"] == "EXACT_ZERO"
        assert contract["certificate"]["reason"] == "CONSTANT_CHANNEL"


def test_projected_statistics_match_materialized_als_designs() -> None:
    rng = np.random.default_rng(20260912)
    phi = rng.normal(size=(137, 5, 4))
    target = rng.normal(size=len(phi))
    full = DesignStatistics.from_design(phi.reshape(len(phi), -1), target)
    u = rng.normal(size=(5, 3))
    v = rng.normal(size=(4, 3))

    cases = (
        (
            full.project(_u_transform(v, 5)),
            np.einsum("tbx,xr->trb", phi, v).reshape(len(phi), -1),
        ),
        (
            full.project(_v_transform(u, 4)),
            np.einsum("tbx,br->trx", phi, u).reshape(len(phi), -1),
        ),
        (
            full.project(_component_transform(u, v.T, 3)),
            np.column_stack(
                [
                    np.einsum("tbx,b,x->t", phi, u[:, index], v[:, index])
                    for index in range(3)
                ]
            ),
        ),
    )
    for projected, materialized in cases:
        explicit = DesignStatistics.from_design(materialized, target)
        np.testing.assert_allclose(projected.gram, explicit.gram, rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(projected.rhs, explicit.rhs, rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(
            projected.feature_sum, explicit.feature_sum, rtol=2e-13, atol=2e-12
        )
