"""Unit and protocol tests for the V20 dual-solver package."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from stage1.experiment_utils import cross_seed_one_se_select
from stage1.independent_path import calibrate_lambda
from stage1.sequence_ops import raw_sequence_to_windows
from stage1.variational_v20 import (
    SplineBasisBankV20,
    VariationalDistributedLagSplineV20,
    group_soft_threshold,
    solve_fixed_q_fista_v20,
)


def test_cross_seed_one_se_uses_within_config_seed_error():
    rows = []
    for seed, value in enumerate([0.10, 0.11, 0.09, 0.10, 0.10]):
        rows.append({"config_id": "a", "seed": seed, "refit_val_rmse": value,
                     "terminal_support": [0, 1, 2, 3], "stable": True})
    for seed, value in enumerate([0.102, 0.101, 0.103, 0.102, 0.101]):
        rows.append({"config_id": "b", "seed": seed, "refit_val_rmse": value,
                     "terminal_support": [0, 1, 2], "stable": True})
    selected = cross_seed_one_se_select(rows)
    assert selected["config_id"] == "b"
    assert selected["one_se_reference_config"] == "a"


def test_cross_seed_selector_is_truth_blind():
    rows = [{"config_id": "a", "refit_val_rmse": 0.1, "terminal_support": [9],
             "stable": True, "seed": seed, "test_rmse": -999,
             "f1": 0.0} for seed in range(5)]
    rows += [{"config_id": "b", "refit_val_rmse": 0.2, "terminal_support": [0, 1, 2],
              "stable": True, "seed": seed, "test_rmse": 0.0,
              "f1": 1.0} for seed in range(5)]
    assert cross_seed_one_se_select(rows)["config_id"] == "a"


def test_lambda_calibration_matches_dimensionless_shrink_definition():
    assert calibrate_lambda(0.006, 2.0, 0.003) == pytest.approx(4.0)


def test_roughness_matrix_is_positive_semidefinite():
    bank = SplineBasisBankV20([(-3, 3)] * 3, grid_size=8)
    eigenvalues = torch.linalg.eigvalsh(bank.roughness)
    assert eigenvalues.min() >= -1e-6


def test_group_soft_threshold_zeros_small_group():
    x = torch.tensor([[3.0, 4.0], [0.1, 0.0]])
    out = group_soft_threshold(x, 0.2)
    assert out[0].norm() == pytest.approx(4.8)
    assert torch.equal(out[1], torch.zeros_like(out[1]))


def test_fista_v20_recovers_sparse_group_and_small_kkt():
    torch.manual_seed(7)
    batch, n, lag = 220, 3, 6
    raw = torch.randn(n, batch + lag - 1)
    windows = raw_sequence_to_windows(raw, lag)
    bank = SplineBasisBankV20([(-4, 4)] * n, grid_size=6)
    basis = bank(torch.flip(windows, dims=(-1,)))
    q = torch.full((n, lag), 1 / lag)
    true_coeff = torch.zeros(n, bank.num_basis)
    true_coeff[0, :4] = torch.tensor([1.2, -0.8, 0.6, 0.3])
    phi = torch.einsum("bnlm,nl->bnm", basis, q)
    target = torch.einsum("bnm,nm->b", phi, true_coeff)
    result = solve_fixed_q_fista_v20(
        basis, q, target, lambda_group=1e-4, lambda_smooth=1e-5,
        roughness_matrix=bank.roughness, max_iter=5000,
        tolerance=1e-8, kkt_tolerance=2e-4,
    )
    prediction = torch.einsum("bnm,nm->b", phi, result.coefficients) + result.bias
    assert 0 in result.support
    assert torch.sqrt((prediction - target).square().mean()) < 0.02
    assert result.kkt_residual < 5e-3
    assert result.history


def test_variational_state_dict_roundtrip_preserves_selection():
    model = VariationalDistributedLagSplineV20(4, 8, [(-3, 3)] * 4, grid_size=6)
    model.selection_mask[:] = torch.tensor([True, False, True, False])
    state = copy.deepcopy(model.state_dict())
    clone = VariationalDistributedLagSplineV20(4, 8, [(-3, 3)] * 4, grid_size=6)
    clone.load_state_dict(state)
    assert torch.equal(clone.selection_mask, model.selection_mask)


def test_v20_runners_do_not_define_nested_homotopy_selection():
    root = Path(__file__).resolve().parents[1]
    text = (root / "run_kan_fast_s0_v20.py").read_text(encoding="utf-8")
    assert "run_independent_fork" in text
    assert "run_group_homotopy_path" not in text


def test_gpu_pool_manifest_schema_is_simple_commands(tmp_path):
    payload = {"jobs": [{"job_id": "a", "command": ["python", "-c", "print(1)"]}]}
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text())
    assert loaded["jobs"][0]["job_id"] == "a"
    assert isinstance(loaded["jobs"][0]["command"], list)


def test_no_placeholder_csv_policy_is_in_shared_writer():
    root = Path(__file__).resolve().parents[1]
    text = (root / "stage1/experiment_utils.py").read_text(encoding="utf-8")
    assert "refusing to write empty CSV" in text


def test_sequence_first_spline_basis_matches_window_basis():
    torch.manual_seed(11)
    n, lag, batch = 3, 7, 20
    raw = torch.randn(n, batch + lag - 1)
    windows = raw_sequence_to_windows(raw, lag)
    bank = SplineBasisBankV20([(-4, 4)] * n, grid_size=6)
    direct = bank(torch.flip(windows, dims=(-1,)))
    sequence = bank.sequence_basis(raw)
    rebuilt = bank.lag_basis_from_sequence(sequence, lag)
    assert torch.allclose(direct, rebuilt, atol=1e-7, rtol=1e-7)
