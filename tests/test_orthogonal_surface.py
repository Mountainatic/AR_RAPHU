import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.orthogonal_surface import (
    orthogonal_lag_bases,
    solve_surface_ridge,
    surface_design,
    surface_penalty,
)


def test_lag_basis_is_orthogonal_to_each_anchor() -> None:
    q = torch.rand(3, 32, dtype=torch.float64)
    q /= q.sum(1, keepdim=True)
    residual = orthogonal_lag_bases(q, 8)
    products = torch.einsum("nl,nlm->nm", q, residual)
    torch.testing.assert_close(products, torch.zeros_like(products), atol=1e-12, rtol=0)


def test_surface_design_matches_explicit_contraction() -> None:
    torch.manual_seed(4)
    amplitude = torch.randn(5, 2, 7, 4, dtype=torch.float64)
    lag = torch.randn(2, 7, 3, dtype=torch.float64)
    actual = surface_design(amplitude, lag)
    expected = []
    for observation in range(5):
        row = []
        for variable in range(2):
            row.extend(
                (
                    lag[variable].T @ amplitude[observation, variable]
                ).reshape(-1)
            )
        expected.append(torch.stack(row))
    torch.testing.assert_close(actual, torch.stack(expected))


def test_surface_ridge_recovers_noise_free_prediction() -> None:
    rng = np.random.default_rng(7)
    design = torch.tensor(rng.normal(size=(800, 18)), dtype=torch.float64)
    truth = torch.tensor(rng.normal(size=18), dtype=torch.float64)
    target = design @ truth
    penalty = torch.zeros(18, 18, dtype=torch.float64)
    result = solve_surface_ridge(
        design, target, penalty, smoothness=1.0e-3
    )
    torch.testing.assert_close(
        design @ result.coefficients, target, atol=1e-8, rtol=1e-8
    )
    assert result.kkt_residual < 1.0e-8


def test_surface_penalty_shape_and_psd() -> None:
    penalty = surface_penalty(
        2, 6, 7, device=torch.device("cpu"), dtype=torch.float64
    )
    assert penalty.shape == (84, 84)
    assert torch.linalg.eigvalsh(penalty).min() >= -1.0e-10
