"""Frozen truth-aware metrics used only after diagnostic fitting."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ar_raphu.model import ARRAPHURank1
from ar_raphu.synthetic import second_truth_response, truth_response

from .rank2_model import ACTIVE_SUPPORT, ARRAPHURank2Diagnostic


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - np.sum((target - prediction) ** 2) / denominator)


def wasserstein1_discrete(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Discrete kernels must be equal-length vectors.")
    return float(np.abs(np.cumsum(first) - np.cumsum(second)).sum())


def lag_entropy(q: np.ndarray) -> float:
    values = np.asarray(q, dtype=np.float64)
    values = values[values > 0]
    return float(-(values * np.log(values)).sum())


def _centered_surface_metrics(
    learned: np.ndarray, truth: np.ndarray
) -> tuple[float, float]:
    learned = np.asarray(learned, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    learned = learned - learned.mean()
    truth = truth - truth.mean()
    scale = float(np.sqrt(np.mean(truth**2)))
    nrmse = (
        float(np.sqrt(np.mean((learned - truth) ** 2)) / scale)
        if scale > 0
        else float("nan")
    )
    correlation = (
        float(np.corrcoef(learned.ravel(), truth.ravel())[0, 1])
        if learned.std() > 0 and truth.std() > 0
        else float("nan")
    )
    return nrmse, correlation


@torch.no_grad()
def surface_metrics(
    model: ARRAPHURank1 | ARRAPHURank2Diagnostic,
    sequence,
    data,
    *,
    rank2_truth: bool,
    points: int = 201,
) -> dict[str, Any]:
    """Compare identifiable centered surfaces on train 1--99% amplitude grids."""

    train_stop = sequence.split_target_intervals["train"][1]
    y_scale = float(data.scaler.y_scale)
    per_variable: list[dict[str, Any]] = []
    learned_arrays: dict[str, np.ndarray] = {}
    truth_arrays: dict[str, np.ndarray] = {}

    if isinstance(model, ARRAPHURank2Diagnostic):
        q_first, q_second = (
            item.detach().cpu().numpy() for item in model.lag_kernels()
        )
    else:
        q_single = (
            model.external_branch._static_q().detach().cpu().numpy()
        )

    for variable in ACTIVE_SUPPORT:
        raw = sequence.x[:train_stop, variable]
        lower, upper = np.quantile(raw, [0.01, 0.99])
        amplitude = np.linspace(lower, upper, points, dtype=np.float64)
        standardized = (
            amplitude - data.scaler.x_mean[variable]
        ) / data.scaler.x_scale[variable]
        tensor = torch.as_tensor(
            standardized,
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )[:, None]

        primary_truth = truth_response(variable, amplitude) / y_scale
        if rank2_truth:
            secondary_truth = second_truth_response(variable, amplitude) / y_scale
            true_surface = (
                0.6
                * sequence.truth["q_primary"][variable, :, None]
                * primary_truth[None, :]
                + 0.4
                * sequence.truth["q_secondary"][variable, :, None]
                * secondary_truth[None, :]
            )
        else:
            true_surface = (
                sequence.truth["q_primary"][variable, :, None]
                * primary_truth[None, :]
            )

        if isinstance(model, ARRAPHURank2Diagnostic):
            response_first = (
                model.component_1.external_branch.response_branches.branches[
                    variable
                ](tensor)
                .reshape(-1)
                .cpu()
                .numpy()
            )
            response_second = (
                model.component_2.external_branch.response_branches.branches[
                    variable
                ](tensor)
                .reshape(-1)
                .cpu()
                .numpy()
            )
            learned_surface = (
                0.6 * q_first[variable, :, None] * response_first[None, :]
                + 0.4 * q_second[variable, :, None] * response_second[None, :]
            )
        else:
            response = (
                model.external_branch.response_branches.branches[variable](tensor)
                .reshape(-1)
                .cpu()
                .numpy()
            )
            learned_surface = q_single[variable, :, None] * response[None, :]

        nrmse, correlation = _centered_surface_metrics(
            learned_surface, true_surface
        )
        per_variable.append(
            {
                "variable": variable,
                "surface_nrmse": nrmse,
                "surface_correlation": correlation,
                "amplitude_lower": float(lower),
                "amplitude_upper": float(upper),
            }
        )
        learned_arrays[f"learned_v{variable}"] = learned_surface
        truth_arrays[f"truth_v{variable}"] = true_surface
        learned_arrays[f"amplitude_v{variable}"] = amplitude

    return {
        "per_variable": per_variable,
        "mean_surface_nrmse": float(
            np.mean([item["surface_nrmse"] for item in per_variable])
        ),
        "mean_surface_correlation": float(
            np.mean([item["surface_correlation"] for item in per_variable])
        ),
        "arrays": {**learned_arrays, **truth_arrays},
    }


@torch.no_grad()
def rank2_lag_metrics(
    model: ARRAPHURank2Diagnostic, sequence
) -> dict[str, Any]:
    learned = [item.cpu().numpy() for item in model.lag_kernels()]
    truths = [
        sequence.truth["q_primary"],
        sequence.truth["q_secondary"],
    ]
    rows = []
    for component in range(2):
        for variable in ACTIVE_SUPPORT:
            q = learned[component][variable]
            truth = truths[component][variable]
            rows.append(
                {
                    "component": component + 1,
                    "variable": variable,
                    "w1": wasserstein1_discrete(q, truth),
                    "entropy": lag_entropy(q),
                    "boundary_mass_last3": float(q[-3:].sum()),
                }
            )
    return {
        "per_component_variable": rows,
        "mean_w1": float(np.mean([row["w1"] for row in rows])),
    }


def contribution_energy_metrics(
    contributions: np.ndarray,
    *,
    target_standard_deviation: float = 1.0,
) -> dict[str, Any]:
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 11:
        raise ValueError("Expected ten X contributions and one AR contribution.")
    x = values[:, :10]
    ar = values[:, 10]
    energy = np.mean(x**2, axis=0)
    total = float(energy.sum())
    support_energy = float(energy[list(ACTIVE_SUPPORT)].sum())
    top3 = set(np.argsort(energy)[-3:].tolist())
    active_median = float(np.median(energy[list(ACTIVE_SUPPORT)]))
    inactive = sorted(set(range(10)) - set(ACTIVE_SUPPORT))
    inactive_median = float(np.median(energy[inactive]))
    return {
        "energy_by_variable": energy.tolist(),
        "true_support_energy_fraction": support_energy / max(total, 1.0e-12),
        "top3_contribution_recall": len(top3 & set(ACTIVE_SUPPORT)) / 3.0,
        "active_inactive_median_energy_ratio": active_median
        / max(inactive_median, 1.0e-12),
        "x_contribution_rms_over_target_std": float(
            np.sqrt(np.mean(np.sum(x, axis=1) ** 2))
            / max(target_standard_deviation, 1.0e-12)
        ),
        "ar_contribution_rms_over_target_std": float(
            np.sqrt(np.mean(ar**2)) / max(target_standard_deviation, 1.0e-12)
        ),
    }
