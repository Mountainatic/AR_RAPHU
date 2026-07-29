"""FAST-C conditional Gram/Schur spectrum diagnostics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.linalg

from ar_raphu.spectral.design import (
    build_ar_nuisance_design,
    build_spectral_design,
)

from .residualization import FAST_TASKS, build_fast_folds, target_indices


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (values - mean) / scale, mean, scale


def _ridge_residualize(
    target: np.ndarray, nuisance: np.ndarray, alpha: float
) -> np.ndarray:
    z, _, _ = _standardize(target)
    q, _, _ = _standardize(nuisance)
    gram = q.T @ q / len(q)
    rhs = q.T @ z / len(q)
    system = gram + float(alpha) * np.eye(gram.shape[0])
    try:
        factor = scipy.linalg.cho_factor(
            system, lower=True, check_finite=False
        )
        coefficients = scipy.linalg.cho_solve(
            factor, rhs, check_finite=False
        )
    except np.linalg.LinAlgError:
        coefficients = scipy.linalg.pinvh(
            system, rtol=1.0e-12, check_finite=False
        ) @ rhs
    return z - q @ coefficients


def spectrum_metrics(
    matrix: np.ndarray,
    *,
    coercivity_ratios: Iterable[float],
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(matrix, dtype=np.float64)
    gram = values.T @ values / len(values)
    eigenvalues = scipy.linalg.eigvalsh(
        gram, check_finite=False, driver="evd"
    )[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = float(eigenvalues.sum())
    squared = float(np.sum(eigenvalues**2))
    effective_rank = trace**2 / max(squared, np.finfo(np.float64).eps)
    probabilities = eigenvalues / max(trace, np.finfo(np.float64).eps)
    positive = probabilities[probabilities > 0.0]
    entropy_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    leading = max(float(eigenvalues[0]), np.finfo(np.float64).eps)
    positive_eigenvalues = eigenvalues[
        eigenvalues > leading * 1.0e-12
    ]
    condition = (
        leading / max(float(positive_eigenvalues[-1]), np.finfo(np.float64).eps)
        if len(positive_eigenvalues)
        else float("inf")
    )
    summary = {
        "trace": trace,
        "effective_rank": float(effective_rank),
        "entropy_rank": entropy_rank,
        "lambda1_fraction": float(eigenvalues[0] / max(trace, np.finfo(np.float64).eps)),
        "lambda10_over_lambda1": (
            float(eigenvalues[9] / leading) if len(eigenvalues) >= 10 else 0.0
        ),
        "condition_estimate": float(condition),
        "coercive_dimension": {
            str(ratio): int(np.count_nonzero(eigenvalues / leading >= ratio))
            for ratio in coercivity_ratios
        },
    }
    return eigenvalues, summary


def conditional_gram_audit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    input_names: Iterable[str],
    energy_selection: dict[str, object],
    lag_basis_count: int,
    amplitude_basis_count: int,
    top_eigenvalues: int,
    coercivity_ratios: Iterable[float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    names = tuple(input_names)
    ratios = tuple(float(value) for value in coercivity_ratios)
    csv_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    block_width = lag_basis_count * amplitude_basis_count
    for task in FAST_TASKS:
        task_summary: dict[str, object] = {}
        selected_alphas = [
            float(energy_selection[f"{task.name}:{variable}"]["ridge_alpha"])
            for variable in range(len(names))
        ]
        for fold in build_fast_folds(len(y), task):
            train_targets = target_indices(
                start=0, stop=fold.effective_train_stop, task=task
            )
            external = build_spectral_design(
                x,
                target_indices=train_targets,
                train_target_stop=fold.effective_train_stop,
                horizon=task.horizon,
                L_x=task.L_x,
                lag_basis_count=lag_basis_count,
                amplitude_basis_count=amplitude_basis_count,
                continuation_scale_factor=1.0,
            )
            ar = build_ar_nuisance_design(
                y,
                target_indices=train_targets,
                train_target_stop=fold.effective_train_stop,
                horizon=task.horizon,
                L_y=task.L_y,
                lag_basis_count=lag_basis_count,
                amplitude_basis_count=amplitude_basis_count,
                continuation_scale_factor=1.0,
            )
            blocks: list[tuple[str, int | str, np.ndarray, float]] = []
            for variable, name in enumerate(names):
                start = variable * block_width
                stop = start + block_width
                blocks.append(
                    (
                        name,
                        variable,
                        external.matrix[:, start:stop],
                        selected_alphas[variable],
                    )
                )
            blocks.append(
                (
                    "JOINT_EXTERNAL",
                    "joint",
                    external.matrix,
                    float(np.median(selected_alphas)),
                )
            )
            fold_summary: dict[str, object] = {}
            for block_name, block_index, block, alpha in blocks:
                residual = _ridge_residualize(block, ar, alpha)
                eigenvalues, metrics = spectrum_metrics(
                    residual, coercivity_ratios=ratios
                )
                fold_summary[str(block_index)] = {
                    "block": block_name,
                    "ridge_alpha": alpha,
                    **metrics,
                }
                for eigen_index in range(min(top_eigenvalues, len(eigenvalues))):
                    csv_rows.append(
                        {
                            "task": task.name,
                            "Lx": task.L_x,
                            "Ly": task.L_y,
                            "horizon": task.horizon,
                            "fold": fold.fold,
                            "block": block_name,
                            "block_index": block_index,
                            "ridge_alpha": alpha,
                            "eigen_index": eigen_index + 1,
                            "eigenvalue": float(eigenvalues[eigen_index]),
                            "trace": metrics["trace"],
                            "effective_rank": metrics["effective_rank"],
                            "entropy_rank": metrics["entropy_rank"],
                            "lambda1_fraction": metrics["lambda1_fraction"],
                            "lambda10_over_lambda1": metrics[
                                "lambda10_over_lambda1"
                            ],
                            "condition_estimate": metrics[
                                "condition_estimate"
                            ],
                            "coercive_dimension_1e-2": metrics[
                                "coercive_dimension"
                            ][str(1.0e-2)],
                            "coercive_dimension_1e-3": metrics[
                                "coercive_dimension"
                            ][str(1.0e-3)],
                            "coercive_dimension_1e-4": metrics[
                                "coercive_dimension"
                            ][str(1.0e-4)],
                        }
                    )
            task_summary[str(fold.fold)] = fold_summary
        summaries[task.name] = task_summary
    return csv_rows, summaries
