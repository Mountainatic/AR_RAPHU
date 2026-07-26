from __future__ import annotations

import numpy as np

from ar_raphu.diagnostics.gate_fista import (
    lambda_maximum,
    solve_gate_fista,
    solve_gate_path,
)


def test_fista_converges_and_produces_exact_zero_coefficients() -> None:
    rng = np.random.default_rng(20260726)
    design = rng.standard_normal((500, 6))
    design -= design.mean(axis=0)
    target = 1.25 + 2.0 * design[:, 0] - 1.5 * design[:, 2]
    maximum = lambda_maximum(design, target)
    solution = solve_gate_fista(
        design,
        target,
        0.10 * maximum,
        max_iterations=10000,
        tolerance=1.0e-9,
    )

    assert solution.converged
    assert set(np.flatnonzero(solution.gates)) == {0, 2}
    assert np.count_nonzero(solution.gates[[1, 3, 4, 5]]) == 0
    assert abs(solution.intercept - 1.25) < 1.0e-8
    assert solution.kkt_residual < 1.0e-6


def test_fista_path_uses_declared_order_and_warm_starts() -> None:
    rng = np.random.default_rng(7)
    design = rng.standard_normal((200, 4))
    target = design[:, 0] - 0.5 * design[:, 1]
    ratios = [0.32, 0.16, 0.08, 0.0]
    solutions = solve_gate_path(design, target, ratios)
    maximum = lambda_maximum(design, target)
    np.testing.assert_allclose(
        [solution.lambda_value for solution in solutions],
        np.asarray(ratios) * maximum,
    )
    assert all(solution.converged for solution in solutions)
