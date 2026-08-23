from __future__ import annotations

import numpy as np

from prism_benchmark.v2_basis import AmplitudeBasis, causal_geometric_intervals
from prism_benchmark.v2_numerics import deterministic_hash_order, residualize, solve_certified
from prism_benchmark.v2_selection import one_se_select, practical_activation


def test_one_se_prefers_neutral_and_uses_ddof_one() -> None:
    losses = {"ZERO": [1.0, 1.0, 1.0, 1.0], "ACTIVE": [0.99, 0.99, 0.99, 0.99]}
    selected = one_se_select(losses, lambda value: (0 if value == "ZERO" else 1,), neutral="ZERO")
    assert selected.selected == "ACTIVE"
    tied = one_se_select({"ZERO": [1.0] * 4, "ACTIVE": [1.0] * 4}, lambda value: (0 if value == "ZERO" else 1,), neutral="ZERO")
    assert tied.selected == "ZERO"
    assert tied.best_se == 0.0


def test_practical_activation_requires_three_of_four_positive() -> None:
    passed = practical_activation([1.0] * 4, [0.95, 0.95, 0.95, 1.01])
    assert passed["pass"]
    failed = practical_activation([1.0] * 4, [0.95, 0.95, 1.01, 1.01])
    assert not failed["pass"]


def test_practical_activation_rejects_unpaired_fold_losses() -> None:
    with np.testing.assert_raises_regex(ValueError, "identical shape"):
        practical_activation([1.0] * 3, [0.9] * 6)


def test_full_sha_order_is_deterministic() -> None:
    values = ["z", "a", "m", "x"]
    np.testing.assert_array_equal(deterministic_hash_order(values), deterministic_hash_order(values))


def test_certified_solver_and_orthogonalization() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(400, 7))
    beta = rng.normal(size=7)
    y = x @ beta
    fitted, certificate = solve_certified(x, y, 1e-10)
    np.testing.assert_allclose(fitted, beta, atol=1e-8)
    assert certificate.status in {"PASS", "PASS_WITH_WARNING"}
    z, audit = residualize(rng.normal(size=(400, 3)), np.column_stack([np.ones(400), x[:, :2]]))
    assert audit["pass"]
    assert np.linalg.norm(np.column_stack([np.ones(400), x[:, :2]]).T @ z) < 1e-8


def test_amplitude_basis_contracts_and_causal_blocks() -> None:
    continuous = np.linspace(-2.0, 3.0, 101)
    basis = AmplitudeBasis.fit(continuous, 6)
    transformed = basis.transform(continuous)
    assert basis.kind == "continuous"
    assert transformed.shape == (101, basis.dimension)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
    assert AmplitudeBasis.fit(np.array([0.0, 1.0, 0.0, 1.0]), 4).kind == "binary"
    intervals = causal_geometric_intervals(64, 8)
    assert intervals[0][0] == 0 and intervals[-1][1] == 64
    assert all(left < right for left, right in intervals)
