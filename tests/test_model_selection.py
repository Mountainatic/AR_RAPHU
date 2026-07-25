import pytest
import torch

from ar_raphu.model_selection import validation_one_se_select
from tools.run_phase1_m7 import solve_unpenalized_rank_robust


def test_one_se_uses_minimum_configuration_se_and_complexity_order() -> None:
    configs = ["g8_l0", "g8_l1", "g12_l0"]
    losses = {
        "g8_l0": [1.00, 1.10, 0.90],
        "g8_l1": [1.01, 1.11, 0.91],
        "g12_l0": [0.99, 1.09, 0.89],
    }
    rows = [
        {
            "config_id": config_id,
            "unit_id": f"seed_{seed}",
            "validation_loss": loss,
        }
        for config_id, values in losses.items()
        for seed, loss in enumerate(values)
    ]
    complexity = {
        "g8_l0": (8, 0),
        "g8_l1": (8, -1),
        "g12_l0": (12, 0),
    }
    selected = validation_one_se_select(
        rows,
        declared_config_order=configs,
        complexity_key=complexity.__getitem__,
    )
    assert selected["minimum_config_id"] == "g12_l0"
    assert selected["selected_config_id"] == "g8_l1"
    assert selected["test_used"] is False


def test_one_se_rejects_inconsistent_units() -> None:
    rows = [
        {"config_id": "a", "unit_id": "0", "validation_loss": 1.0},
        {"config_id": "a", "unit_id": "1", "validation_loss": 1.1},
        {"config_id": "b", "unit_id": "0", "validation_loss": 0.9},
    ]
    with pytest.raises(ValueError, match="same selection units"):
        validation_one_se_select(
            rows,
            declared_config_order=["a", "b"],
            complexity_key=lambda config_id: (config_id,),
        )


def test_rank_robust_unpenalized_solver_handles_collinearity() -> None:
    torch.manual_seed(11)
    basis = torch.randn(200, 2, 7, 5, dtype=torch.float64)
    basis[:, 1] = basis[:, 0]
    q = torch.rand(2, 7, dtype=torch.float64)
    q /= q.sum(1, keepdim=True)
    q[1] = q[0]
    coefficients = torch.randn(2, 5, dtype=torch.float64)
    target = torch.einsum("bnlm,nm,nl->b", basis, coefficients, q) + 0.4
    result = solve_unpenalized_rank_robust(basis, q, target)
    prediction = (
        torch.einsum("bnlm,nm,nl->b", basis, result.coefficients, q)
        + result.bias
    )
    torch.testing.assert_close(prediction, target, atol=1e-9, rtol=1e-9)
    assert result.converged
    assert result.kkt_residual < 1.0e-10
