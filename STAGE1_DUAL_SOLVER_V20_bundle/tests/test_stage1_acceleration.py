"""Regression and mathematical-equivalence tests for the v19 acceleration path."""
from __future__ import annotations

import copy
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from stage1.model import Stage1TargetDelayKAN
from stage1.response_kan import UnivariateKANResponse
from stage1.sequence_ops import (
    ContiguousWindowBatch,
    effective_lag_lengths,
    raw_sequence_to_windows,
    static_depthwise_contribution,
    truncate_q_with_error_bound,
    windows_to_raw_sequence,
)
from stage1.state_space import (
    ExponentialKernelApproximation,
    exponential_state_space_filter,
)
from stage1.variational_spline import (
    SplineBasisBank,
    solve_fixed_q_group_lasso,
)


def _model(n=4, lag=8, epsilon=0.0):
    return Stage1TargetDelayKAN(
        n, lag, hidden_kan=4, kan_grid_size=5,
        active_mask=torch.ones(n, dtype=torch.bool), epsilon=epsilon,
    )


def _contiguous(batch=20, n=4, lag=8):
    raw = torch.randn(n, batch + lag - 1)
    return raw, raw_sequence_to_windows(raw, lag)


def test_window_raw_roundtrip():
    raw, windows = _contiguous()
    rebuilt = windows_to_raw_sequence(windows)
    assert torch.equal(raw, rebuilt)


def test_noncontiguous_windows_are_rejected():
    _, windows = _contiguous()
    windows[3, 0, 0] += 1
    with pytest.raises(ValueError, match="not consecutive"):
        windows_to_raw_sequence(windows)


def test_vectorized_kan_matches_legacy_and_gradients():
    torch.manual_seed(1)
    response = UnivariateKANResponse(4, hidden_kan=4, grid_size=5)
    values = torch.randn(6, 4, 8)
    legacy = response.forward_legacy(values)
    vectorized = response.forward_vectorized(values)
    assert torch.allclose(legacy, vectorized, atol=2e-6, rtol=2e-6)

    a = copy.deepcopy(response)
    b = copy.deepcopy(response)
    a.forward_legacy(values).square().mean().backward()
    b.forward_vectorized(values).square().mean().backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=3e-6, rtol=3e-5)


def test_sequence_fast_path_matches_window_forward():
    torch.manual_seed(2)
    raw, windows = _contiguous(batch=25)
    model = _model()
    model.eval()
    with torch.no_grad():
        expected, expected_aux = model(windows)
        actual, actual_aux = model.forward_contiguous(
            ContiguousWindowBatch.from_windows(windows),
            return_aux=True,
            materialize_lag_aux=True,
        )
    assert torch.allclose(expected, actual, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        expected_aux["variable_contribution"],
        actual_aux["variable_contribution"], atol=2e-6, rtol=2e-6)
    assert torch.allclose(expected_aux["contribution"], actual_aux["contribution"],
                          atol=2e-6, rtol=2e-6)


def test_sequence_fast_path_gradient_matches_window_forward():
    torch.manual_seed(3)
    _, windows = _contiguous(batch=12)
    target = torch.randn(12)
    legacy = _model()
    fast = copy.deepcopy(legacy)
    legacy.response_branches.execution_mode = "legacy"
    loss_legacy = (legacy(windows, return_aux=False).squeeze(-1) - target).square().mean()
    loss_fast = (fast.forward_contiguous(
        ContiguousWindowBatch.from_windows(windows), return_aux=False
    ).squeeze(-1) - target).square().mean()
    loss_legacy.backward(); loss_fast.backward()
    assert torch.allclose(loss_legacy, loss_fast, atol=2e-6, rtol=2e-6)
    for (name_a, pa), (name_b, pb) in zip(legacy.named_parameters(), fast.named_parameters()):
        assert name_a == name_b
        if pa.grad is None or pb.grad is None:
            assert pa.grad is None and pb.grad is None
        else:
            assert torch.allclose(pa.grad, pb.grad, atol=5e-6, rtol=5e-5), name_a


def test_dynamic_tilt_falls_back_to_window_path():
    torch.manual_seed(4)
    _, windows = _contiguous(batch=10)
    model = _model(epsilon=0.5)
    with torch.no_grad():
        a = model(windows, return_aux=False)
        b = model.forward_contiguous(windows, return_aux=False)
    assert torch.allclose(a, b)


def test_depthwise_operator_orientation():
    response = torch.tensor([[1., 2., 3., 4., 5.]])
    q = torch.tensor([[0.75, 0.25]])  # current, previous
    out = static_depthwise_contribution(response, q).squeeze(-1)
    expected = torch.tensor([1.75, 2.75, 3.75, 4.75])
    assert torch.allclose(out, expected)


def test_tail_bound_is_valid():
    q = torch.tensor([[.7, .2, .09, .009, .001]])
    truncated, lengths, discarded = truncate_q_with_error_bound(q, 0.011)
    assert lengths.item() == 3
    z = torch.tensor([[1., -2., 3., -4., 2., 1., 0.]])
    exact = static_depthwise_contribution(z, q)
    approx = static_depthwise_contribution(z, truncated)
    bound = z.abs().max() * discarded[0]
    assert (exact - approx).abs().max() <= bound + 1e-7
    assert effective_lag_lengths(q, .011).item() == 3


def test_state_space_exact_for_single_exponential():
    rho = torch.tensor([0.8])
    amp = torch.tensor([[0.2]])
    lag = 12
    tau = torch.arange(lag)
    q = amp @ rho[:, None].pow(tau[None, :])
    approximation = ExponentialKernelApproximation(
        amplitudes=amp, decays=rho, reconstructed_q=q,
        l1_error=torch.zeros(1), max_error=torch.zeros(1),
    )
    z = torch.randn(1, 30)
    state_output = exponential_state_space_filter(z, approximation)
    exact = static_depthwise_contribution(z, q).transpose(0, 1)
    # The state filter has an infinite tail; compare a kernel long enough that
    # the omitted tail is tiny rather than demanding bit equality.
    assert (state_output[:, lag-1:] - exact).abs().max() < 0.25


def test_fista_group_lasso_recovers_single_active_group():
    torch.manual_seed(5)
    batch, n, lag = 120, 3, 6
    raw = torch.randn(n, batch + lag - 1)
    windows = raw_sequence_to_windows(raw, lag)
    basis_bank = SplineBasisBank([(-3, 3)] * n, grid_size=6, spline_order=3)
    basis = basis_bank(torch.flip(windows, dims=(-1,)))
    q = torch.full((n, lag), 1 / lag)
    true_coeff = torch.zeros(n, basis.shape[-1])
    true_coeff[0, :3] = torch.tensor([1.0, -0.7, 0.5])
    phi = torch.einsum("bnlm,nl->bnm", basis, q)
    target = torch.einsum("bnm,nm->b", phi, true_coeff)
    result = solve_fixed_q_group_lasso(
        basis, q, target, lambda_group=1e-4,
        max_iter=1500, tolerance=1e-9,
    )
    assert 0 in result.support
    assert result.coefficients[0].norm() > result.coefficients[1:].norm(dim=-1).max()
    prediction = torch.einsum("bnm,nm->b", phi, result.coefficients) + result.bias
    assert torch.sqrt((prediction - target).square().mean()) < 0.02
