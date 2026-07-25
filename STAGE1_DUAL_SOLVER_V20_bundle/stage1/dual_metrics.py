"""Common scientific metrics for KAN and variational V20 solvers."""
from __future__ import annotations

import numpy as np
import torch

from .truth_functions import apply_true_response


def rmse(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def correlation(a, b) -> float:
    a, b = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    if a.std() < 1e-14 or b.std() < 1e-14:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def prediction_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    residual = y_pred - y_true
    denom = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.sum(residual ** 2) / denom) if denom > 0 else 0.0,
        "bias": float(residual.mean()),
        "max_abs_error": float(np.abs(residual).max()),
    }


def delay_metric_rows(learned_q, true_q, *, seed: int, variables=(0, 1, 2)) -> list[dict]:
    learned_q = np.asarray(learned_q, dtype=float)
    true_q = np.asarray(true_q, dtype=float)
    tau = np.arange(true_q.shape[-1], dtype=float)
    rows = []
    for j in variables:
        q, h = learned_q[j], true_q[j]
        qm, hm = q @ tau, h @ tau
        qs = np.sqrt(q @ ((tau - qm) ** 2))
        hs = np.sqrt(h @ ((tau - hm) ** 2))
        rows.append({
            "seed": seed,
            "variable": int(j),
            "true_mean": float(hm),
            "learned_mean": float(qm),
            "mean_delay_mae": float(abs(qm - hm)),
            "true_std": float(hs),
            "learned_std": float(qs),
            "std_delay_mae": float(abs(qs - hs)),
            "wasserstein_1": float(np.abs(np.cumsum(q) - np.cumsum(h)).sum()),
            "peak_error": int(abs(int(q.argmax()) - int(h.argmax()))),
            "boundary_mass": float(q[-3:].sum()),
            "q_correlation": correlation(q, h),
        })
    return rows


def _kan_function(model, variable: int, x: torch.Tensor) -> torch.Tensor:
    branch = model.response_branches.branches[int(variable)]
    return branch(x.reshape(-1, 1)).reshape(-1)


def _variational_function(model, variable: int, x: torch.Tensor) -> torch.Tensor:
    return model.function_values(int(variable), x).reshape(-1)


def learned_function(model, variable: int, x: torch.Tensor, solver: str) -> torch.Tensor:
    if solver == "kan":
        return _kan_function(model, variable, x)
    if solver == "variational":
        return _variational_function(model, variable, x)
    raise ValueError(f"unknown solver {solver}")


def train_function_centers(model, raw_train: torch.Tensor, *, solver: str,
                           variables=(0, 1, 2)) -> tuple[dict[int, float], dict[int, float]]:
    learned, truth = {}, {}
    with torch.no_grad():
        for j in variables:
            x = raw_train[j]
            learned[j] = float(learned_function(model, j, x, solver).mean().cpu())
            truth[j] = float(np.mean(apply_true_response(x.detach().cpu().numpy(), j)))
    return learned, truth


def function_metric_rows(model, raw_train: torch.Tensor, input_ranges, *, seed: int,
                         solver: str, variables=(0, 1, 2), grid_points: int = 401):
    learned_centers, truth_centers = train_function_centers(
        model, raw_train, solver=solver, variables=variables)
    rows, arrays = [], {}
    with torch.no_grad():
        for j in variables:
            lo, hi = input_ranges[j]
            x = torch.linspace(lo, hi, grid_points, device=raw_train.device,
                               dtype=raw_train.dtype)
            pred = learned_function(model, j, x, solver).detach().cpu().numpy() - learned_centers[j]
            true = apply_true_response(x.detach().cpu().numpy(), j) - truth_centers[j]
            scale = np.sqrt(np.mean(true ** 2)) + 1e-12
            rows.append({
                "seed": seed,
                "variable": int(j),
                "function_rmse": rmse(pred, true),
                "function_nrmse": rmse(pred, true) / scale,
                "function_correlation": correlation(pred, true),
                "learned_center": learned_centers[j],
                "true_center": truth_centers[j],
            })
            arrays[f"x_grid_v{j}"] = x.cpu().numpy()
            arrays[f"learned_grid_v{j}"] = pred
            arrays[f"true_grid_v{j}"] = true
    return rows, arrays, learned_centers, truth_centers


def _learned_lag_response(model, windows: torch.Tensor, solver: str) -> torch.Tensor:
    x_lag = torch.flip(windows, dims=(-1,))
    if solver == "kan":
        return model.response_branches(x_lag)
    if solver == "variational":
        basis = model.basis(windows)
        return torch.einsum("bnlm,nm->bnl", basis, model.coefficients)
    raise ValueError(solver)


def contribution_metric_rows(model, windows: torch.Tensor, true_q, *, seed: int,
                             solver: str, learned_centers: dict[int, float],
                             truth_centers: dict[int, float], variables=(0, 1, 2)):
    true_q = np.asarray(true_q, dtype=float)
    x_lag_np = torch.flip(windows, dims=(-1,)).detach().cpu().numpy()
    with torch.no_grad():
        learned_response = _learned_lag_response(model, windows, solver).detach().cpu().numpy()
        learned_q = model.delay_prior().detach().cpu().numpy()
    rows, arrays = [], {}
    for j in variables:
        learned_centered = learned_response[:, j] - learned_centers[j]
        true_centered = apply_true_response(x_lag_np[:, j], j) - truth_centers[j]
        learned_lag = learned_q[j][None, :] * learned_centered
        true_lag = true_q[j][None, :] * true_centered
        learned_variable = learned_lag.sum(-1)
        true_variable = true_lag.sum(-1)
        rows.append({
            "seed": seed,
            "variable": int(j),
            "lag_contribution_rmse": rmse(learned_lag, true_lag),
            "lag_contribution_correlation": correlation(learned_lag, true_lag),
            "variable_contribution_rmse": rmse(learned_variable, true_variable),
            "variable_contribution_correlation": correlation(learned_variable, true_variable),
        })
        arrays[f"learned_lag_v{j}"] = learned_lag
        arrays[f"true_lag_v{j}"] = true_lag
        arrays[f"learned_variable_v{j}"] = learned_variable
        arrays[f"true_variable_v{j}"] = true_variable
    return rows, arrays
