from __future__ import annotations

import numpy as np
import torch

from ar_raphu.diagnostics.rank2_model import ARRAPHURank2Diagnostic


def truth_kernels() -> tuple[np.ndarray, np.ndarray]:
    lag = np.arange(64, dtype=np.float32)
    first = np.zeros((10, 64), dtype=np.float32)
    second = np.zeros_like(first)
    for variable in (0, 1, 2):
        q1 = np.exp(-0.5 * ((lag - (5 + variable)) / 2.0) ** 2)
        q2 = np.exp(-0.5 * ((lag - (20 + variable)) / 3.0) ** 2)
        first[variable] = q1 / q1.sum()
        second[variable] = q2 / q2.sum()
    return first, second


def test_rank2_forward_shape_fixed_weights_and_contribution_closure() -> None:
    primary, secondary = truth_kernels()
    model = ARRAPHURank2Diagnostic(
        horizon=1,
        input_grid_ranges_x=[(-3.0, 3.0)] * 10,
        q_primary=primary,
        q_secondary=secondary,
        q_mode="oracle_fixed",
        hidden_kan=4,
        grid_size=5,
        response_execution_mode="legacy",
    )
    x = torch.randn(3, 10, 64)
    prediction, auxiliary = model(x)

    assert prediction.shape == (3, 1)
    assert auxiliary["component_contribution"].shape == (3, 11)
    reconstructed = (
        model.bias + auxiliary["component_contribution"].sum(dim=1)
    )
    torch.testing.assert_close(prediction.squeeze(-1), reconstructed)
    torch.testing.assert_close(
        model.component_weights, torch.tensor([0.6, 0.4])
    )
    assert "component_weights" not in dict(model.named_parameters())
    q_first, q_second = model.lag_kernels()
    torch.testing.assert_close(q_first[:3], torch.from_numpy(primary[:3]))
    torch.testing.assert_close(q_second[:3], torch.from_numpy(secondary[:3]))
    assert torch.count_nonzero(auxiliary["external_contribution"][:, 3:]) == 0


def test_free_truth_initialization_preserves_active_truth_rows() -> None:
    primary, secondary = truth_kernels()
    model = ARRAPHURank2Diagnostic(
        horizon=1,
        input_grid_ranges_x=[(-3.0, 3.0)] * 10,
        q_primary=primary,
        q_secondary=secondary,
        q_mode="free_truth_init",
        hidden_kan=4,
        grid_size=5,
        response_execution_mode="legacy",
    )
    first, second = model.lag_kernels()
    torch.testing.assert_close(
        first[:3], torch.from_numpy(primary[:3]), atol=2.0e-7, rtol=2.0e-6
    )
    torch.testing.assert_close(
        second[:3], torch.from_numpy(secondary[:3]), atol=2.0e-7, rtol=2.0e-6
    )
    assert len(model.lag_parameters()) == 2
