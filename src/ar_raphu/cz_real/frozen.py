"""Fit, serialize, and evaluate the frozen CZ ORSS model family."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ar_raphu.orss.operator import build_urysohn_operator
from ar_raphu.orss.penalties import PenaltyWeights, SeparablePenalty
from ar_raphu.orss.sweep import (
    diagonal_spectral_normalization,
    solve_full,
)

from .linear import regression_metrics, target_indices
from .orss_r3 import _free_run_fold
from .protocol import DIRECT_HORIZONS, confirmation_interval, purge_gap


def fit_frozen_horizon(
    x: np.ndarray,
    y: np.ndarray,
    *,
    horizon: int,
    L_x: int,
    L_y: int,
    M_tau: int,
    M_x: int,
    normalized_weights: PenaltyWeights,
    c_rho: float,
    predictive_rank: dict[str, int],
    device: torch.device,
    chunk_time: int,
    maximum_iterations: int,
) -> dict[str, Any]:
    confirmation_start, _ = confirmation_interval(len(y))
    effective_stop = confirmation_start - purge_gap(
        L_x=L_x, L_y=L_y, h_max=max(DIRECT_HORIZONS)
    )
    train_indices = target_indices(
        start=0,
        stop=effective_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    operator, state = build_urysohn_operator(
        x,
        y,
        target_indices=train_indices,
        train_target_stop=effective_stop,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        lag_basis_count=M_tau,
        amplitude_basis_count=M_x,
        continuation_scale_coefficient=c_rho,
        device=device,
        dtype=torch.float64,
        chunk_time=chunk_time,
    )
    target_mean = float(np.mean(y[train_indices]))
    centered_target = torch.as_tensor(
        y[train_indices] - target_mean,
        device=device,
        dtype=torch.float64,
    )
    penalty = SeparablePenalty(
        channels=operator.channels,
        m_tau=operator.m_tau,
        m_x=operator.m_x,
        device=device,
        dtype=torch.float64,
    )
    normalization = diagonal_spectral_normalization(operator, penalty)
    actual = normalization.actual(normalized_weights)
    fitted = solve_full(
        operator,
        centered_target,
        penalty,
        actual,
        relative_tolerance=1.0e-10,
        maximum_iterations=max(2500, maximum_iterations),
    )
    if fitted.relative_kkt_residual > 1.0e-8:
        raise RuntimeError("FINAL_KKT_FAILED")
    return {
        "schema": "CZ_FROZEN_ORSS_HORIZON_V1",
        "horizon": horizon,
        "L_x": L_x,
        "L_y": L_y,
        "M_tau": M_tau,
        "M_x": M_x,
        "normalized_weights": asdict(normalized_weights),
        "actual_weights": asdict(actual),
        "penalty_normalization": asdict(normalization),
        "CONTINUATION_SCALE_COEFFICIENT": c_rho,
        "predictive_rank_by_budget": predictive_rank,
        "coefficients": fitted.coefficients.detach().cpu(),
        "feature_mean": operator.feature_mean.detach().cpu(),
        "basis_state": state,
        "target_mean": target_mean,
        "effective_train_stop": effective_stop,
        "train_target_count": int(len(train_indices)),
        "relative_kkt_residual": fitted.relative_kkt_residual,
        "solver_method": fitted.method,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_access_count": 0,
    }


def save_frozen_horizon(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_frozen_horizon(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_frozen_horizon(
    model: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_start: int,
    target_stop: int,
    device: torch.device,
    chunk_time: int,
    include_predictions: bool = False,
) -> dict[str, object]:
    horizon = int(model["horizon"])
    L_x = int(model["L_x"])
    L_y = int(model["L_y"])
    targets = target_indices(
        start=target_start,
        stop=target_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    operator, _ = build_urysohn_operator(
        x,
        y,
        target_indices=targets,
        train_target_stop=int(model["effective_train_stop"]),
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        lag_basis_count=int(model["M_tau"]),
        amplitude_basis_count=int(model["M_x"]),
        continuation_scale_coefficient=float(
            model["CONTINUATION_SCALE_COEFFICIENT"]
        ),
        device=device,
        dtype=torch.float64,
        chunk_time=chunk_time,
        basis_state=model["basis_state"],
        feature_mean=model["feature_mean"].to(
            device=device, dtype=torch.float64
        ),
    )
    coefficients = model["coefficients"].to(
        device=device, dtype=torch.float64
    )
    prediction = operator.forward(coefficients) + float(model["target_mean"])
    target = np.asarray(y, dtype=np.float64)[targets]
    direct = regression_metrics(target, prediction.detach().cpu().numpy())
    free_run = _free_run_fold(
        operator,
        model["basis_state"],
        coefficients,
        target_mean=float(model["target_mean"]),
        y=y,
        validation_indices=targets,
        horizon=horizon,
        validation_start=target_start,
        c_rho=float(model["CONTINUATION_SCALE_COEFFICIENT"]),
    )
    by_channel = [
        {
            "channel": channel,
            "out_of_domain_fraction": branch.out_of_domain_fraction,
            "total_calls": int(
                branch.amplitude.shape[0] * branch.amplitude.shape[1]
            ),
        }
        for channel, branch in enumerate(operator.branches)
    ]
    result = {
        "horizon": horizon,
        "target_interval": [int(targets[0]), int(targets[-1]) + 1],
        "target_count": int(len(targets)),
        "direct_metrics": direct,
        "free_run_metrics": free_run,
        "continuation_by_channel": by_channel,
    }
    if include_predictions:
        result["direct_prediction"] = prediction.detach().cpu().numpy()
        result["direct_target"] = target
        result["direct_target_indices"] = targets
    return result
