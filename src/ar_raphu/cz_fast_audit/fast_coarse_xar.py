"""FAST-E coarse dense-batched nonlinear XAR audit."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
import scipy.linalg

from ar_raphu.cz_real.linear import regression_metrics
from ar_raphu.spectral.design import (
    SpectralDesign,
    build_ar_nuisance_design,
    build_spectral_design,
)
from ar_raphu.spectral.penalties import tensor_penalty

from .residualization import FAST_TASKS, FastFold, FastTask, build_fast_folds, target_indices


@dataclass(slots=True)
class PenalizedFit:
    coefficients: np.ndarray
    intercept: float
    relative_kkt_residual: float
    train_mse: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=np.float64) @ self.coefficients + self.intercept


@dataclass(slots=True)
class CoarseModelRecord:
    task: FastTask
    fold: FastFold
    selected_lambda: float
    fit: PenalizedFit
    external_width: int
    train_external: SpectralDesign
    train_ar: np.ndarray


def fit_penalized(
    matrix: np.ndarray,
    target: np.ndarray,
    penalty: np.ndarray,
    *,
    weight: float,
    device: str = "cpu",
) -> PenalizedFit:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc / len(xc)
    rhs = xc.T @ yc / len(xc)
    system = gram + float(weight) * np.asarray(penalty, dtype=np.float64)
    if device.startswith("cuda"):
        import torch

        device_value = torch.device(device)
        gram_tensor = torch.as_tensor(
            gram, dtype=torch.float64, device=device_value
        )
        rhs_tensor = torch.as_tensor(
            rhs, dtype=torch.float64, device=device_value
        )
        penalty_tensor = torch.as_tensor(
            np.asarray(penalty, dtype=np.float64),
            dtype=torch.float64,
            device=device_value,
        )
        if weight == 0.0:
            values_tensor, vectors_tensor = torch.linalg.eigh(gram_tensor)
            threshold = torch.maximum(
                values_tensor.max() * 1.0e-12,
                torch.tensor(
                    np.finfo(np.float64).eps,
                    dtype=torch.float64,
                    device=device_value,
                ),
            )
            inverse_tensor = torch.where(
                values_tensor > threshold,
                1.0 / torch.clamp(values_tensor, min=threshold),
                torch.zeros_like(values_tensor),
            )
            coefficients_tensor = vectors_tensor @ (
                inverse_tensor * (vectors_tensor.T @ rhs_tensor)
            )
        else:
            system_tensor = gram_tensor + float(weight) * penalty_tensor
            try:
                coefficients_tensor = torch.linalg.solve(
                    system_tensor, rhs_tensor
                )
            except RuntimeError:
                coefficients_tensor = (
                    torch.linalg.pinv(system_tensor, rcond=1.0e-12)
                    @ rhs_tensor
                )
        torch.cuda.synchronize(device_value)
        coefficients = coefficients_tensor.cpu().numpy()
    elif weight == 0.0:
        values, vectors = scipy.linalg.eigh(
            gram, check_finite=False, driver="evd"
        )
        threshold = max(
            float(values.max()) * 1.0e-12, np.finfo(np.float64).eps
        )
        inverse = np.where(
            values > threshold,
            1.0 / np.maximum(values, threshold),
            0.0,
        )
        coefficients = vectors @ (inverse * (vectors.T @ rhs))
    else:
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
    intercept = y_mean - float(x_mean @ coefficients)
    prediction = x @ coefficients + intercept
    residual = system @ coefficients - rhs
    relative_kkt = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
    )
    return PenalizedFit(
        coefficients=coefficients,
        intercept=intercept,
        relative_kkt_residual=relative_kkt,
        train_mse=float(np.mean((prediction - y) ** 2)),
    )


def _build_fold(
    x: np.ndarray,
    y: np.ndarray,
    task: FastTask,
    fold: FastFold,
    *,
    lag_basis_count: int,
    amplitude_basis_count: int,
    continuation_scale: float,
) -> dict[str, object]:
    train_targets = target_indices(
        start=0, stop=fold.effective_train_stop, task=task
    )
    validation_targets = target_indices(
        start=fold.validation_start,
        stop=fold.validation_stop,
        task=task,
    )
    kwargs = {
        "train_target_stop": fold.effective_train_stop,
        "horizon": task.horizon,
        "lag_basis_count": lag_basis_count,
        "amplitude_basis_count": amplitude_basis_count,
        "continuation_scale_factor": continuation_scale,
    }
    train_external = build_spectral_design(
        x, target_indices=train_targets, L_x=task.L_x, **kwargs
    )
    validation_external = build_spectral_design(
        x, target_indices=validation_targets, L_x=task.L_x, **kwargs
    )
    train_ar = build_ar_nuisance_design(
        y, target_indices=train_targets, L_y=task.L_y, **kwargs
    )
    validation_ar = build_ar_nuisance_design(
        y, target_indices=validation_targets, L_y=task.L_y, **kwargs
    )
    external_penalty = tensor_penalty(
        train_external.lag_gram,
        train_external.amplitude_grams,
        lag_smoothness=1.0,
        amplitude_smoothness=1.0,
        ridge_weight=1.0,
    )
    ar_penalty = tensor_penalty(
        train_external.lag_gram,
        [np.eye(amplitude_basis_count, dtype=np.float64)],
        lag_smoothness=1.0,
        amplitude_smoothness=1.0,
        ridge_weight=1.0,
    )
    return {
        "fold": fold,
        "train_targets": train_targets,
        "validation_targets": validation_targets,
        "train_external": train_external,
        "validation_external": validation_external,
        "train_ar": train_ar,
        "validation_ar": validation_ar,
        "external_penalty": external_penalty,
        "ar_penalty": ar_penalty,
    }


def coarse_xar_audit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    penalty_path: Iterable[float],
    lag_basis_count: int,
    amplitude_basis_count: int,
    continuation_scale: float,
    linear_rows: list[dict[str, object]],
    device: str = "cpu",
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    list[CoarseModelRecord],
]:
    started = time.perf_counter()
    penalties = tuple(float(value) for value in penalty_path)
    result_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    model_records: list[CoarseModelRecord] = []
    profile: dict[str, object] = {
        "solver": "DENSE_BATCHED_SHARED_GRAM",
        "tasks": {},
    }
    linear_lookup = {
        (str(row["task"]), int(row["fold"])): row for row in linear_rows
    }
    for task in FAST_TASKS:
        task_started = time.perf_counter()
        folds = [
            _build_fold(
                x,
                y,
                task,
                fold,
                lag_basis_count=lag_basis_count,
                amplitude_basis_count=amplitude_basis_count,
                continuation_scale=continuation_scale,
            )
            for fold in build_fast_folds(len(y), task)
        ]
        fit_paths: list[dict[str, list[PenalizedFit]]] = []
        for item in folds:
            target = y[item["train_targets"]]
            train_external = item["train_external"].matrix
            train_ar = item["train_ar"]
            xar_matrix = np.column_stack((train_external, train_ar))
            xar_penalty = scipy.linalg.block_diag(
                item["external_penalty"], item["ar_penalty"]
            )
            fit_paths.append(
                {
                    "AR": [
                        fit_penalized(
                            train_ar,
                            target,
                            item["ar_penalty"],
                            weight=weight,
                            device=device,
                        )
                        for weight in penalties
                    ],
                    "XAR": [
                        fit_penalized(
                            xar_matrix,
                            target,
                            xar_penalty,
                            weight=weight,
                            device=device,
                        )
                        for weight in penalties
                    ],
                }
            )
        mean_losses: dict[str, list[float]] = {"AR": [], "XAR": []}
        for model in ("AR", "XAR"):
            for penalty_index, _ in enumerate(penalties):
                losses = []
                for fold_index, item in enumerate(folds):
                    validation_matrix = (
                        item["validation_ar"]
                        if model == "AR"
                        else np.column_stack(
                            (
                                item["validation_external"].matrix,
                                item["validation_ar"],
                            )
                        )
                    )
                    prediction = fit_paths[fold_index][model][
                        penalty_index
                    ].predict(validation_matrix)
                    losses.append(
                        float(
                            np.mean(
                                (
                                    prediction
                                    - y[item["validation_targets"]]
                                )
                                ** 2
                            )
                        )
                    )
                mean_losses[model].append(float(np.mean(losses)))
        selected = {
            model: min(
                range(len(penalties)),
                key=lambda index: (mean_losses[model][index], index),
            )
            for model in ("AR", "XAR")
        }
        for fold_index, item in enumerate(folds):
            validation_target = y[item["validation_targets"]]
            ar_fit = fit_paths[fold_index]["AR"][selected["AR"]]
            xar_fit = fit_paths[fold_index]["XAR"][selected["XAR"]]
            ar_prediction = ar_fit.predict(item["validation_ar"])
            validation_xar = np.column_stack(
                (item["validation_external"].matrix, item["validation_ar"])
            )
            xar_prediction = xar_fit.predict(validation_xar)
            ar_metrics = regression_metrics(validation_target, ar_prediction)
            xar_metrics = regression_metrics(validation_target, xar_prediction)
            delta = (
                ar_metrics["MSE_mm2"] - xar_metrics["MSE_mm2"]
            ) / max(ar_metrics["MSE_mm2"], np.finfo(np.float64).eps)
            external_width = item["train_external"].matrix.shape[1]
            x_contribution = (
                item["validation_external"].matrix
                @ xar_fit.coefficients[:external_width]
            )
            ar_contribution = (
                item["validation_ar"]
                @ xar_fit.coefficients[external_width:]
            )
            covariance = float(
                np.cov(x_contribution, ar_contribution, ddof=0)[0, 1]
            )
            linear = linear_lookup[(task.name, item["fold"].fold)]
            result_rows.append(
                {
                    "task": task.name,
                    "Lx": task.L_x,
                    "Ly": task.L_y,
                    "horizon": task.horizon,
                    "fold": item["fold"].fold,
                    "selected_AR_penalty": penalties[selected["AR"]],
                    "selected_XAR_penalty": penalties[selected["XAR"]],
                    "FAST_PENALTY_SELECTED_AT_EDGE": bool(
                        selected["XAR"] in {0, len(penalties) - 1}
                    ),
                    "AR_RMSE_mm": ar_metrics["RMSE_mm"],
                    "coarse_XAR_RMSE_mm": xar_metrics["RMSE_mm"],
                    "linear_ARX_RMSE_mm": linear["ARX_RMSE_mm"],
                    "AR_MSE_mm2": ar_metrics["MSE_mm2"],
                    "coarse_XAR_MSE_mm2": xar_metrics["MSE_mm2"],
                    "delta_X_given_AR_coarse": delta,
                    "direction_positive": bool(delta > 0.0),
                    "AR_relative_kkt": ar_fit.relative_kkt_residual,
                    "XAR_relative_kkt": xar_fit.relative_kkt_residual,
                    "exact_zero_XAR_nestedness_margin": (
                        fit_paths[fold_index]["AR"][0].train_mse
                        - fit_paths[fold_index]["XAR"][0].train_mse
                    ),
                }
            )
            contribution_rows.append(
                {
                    "task": task.name,
                    "horizon": task.horizon,
                    "fold": item["fold"].fold,
                    "X_contribution_variance": float(np.var(x_contribution)),
                    "AR_contribution_variance": float(np.var(ar_contribution)),
                    "X_AR_covariance": covariance,
                    "X_contribution_norm": float(
                        np.linalg.norm(x_contribution)
                    ),
                    "AR_contribution_norm": float(
                        np.linalg.norm(ar_contribution)
                    ),
                    "prediction_variance_identity_error": float(
                        abs(
                            np.var(x_contribution + ar_contribution)
                            - (
                                np.var(x_contribution)
                                + np.var(ar_contribution)
                                + 2.0 * covariance
                            )
                        )
                    ),
                    "delta_X_given_AR_coarse": delta,
                }
            )
            model_records.append(
                CoarseModelRecord(
                    task=task,
                    fold=item["fold"],
                    selected_lambda=penalties[selected["XAR"]],
                    fit=xar_fit,
                    external_width=external_width,
                    train_external=item["train_external"],
                    train_ar=item["train_ar"],
                )
            )
        profile["tasks"][task.name] = {
            "elapsed_seconds": time.perf_counter() - task_started,
            "selected_AR_penalty": penalties[selected["AR"]],
            "selected_XAR_penalty": penalties[selected["XAR"]],
            "AR_mean_validation_MSE_by_penalty": dict(
                zip(map(str, penalties), mean_losses["AR"])
            ),
            "XAR_mean_validation_MSE_by_penalty": dict(
                zip(map(str, penalties), mean_losses["XAR"])
            ),
        }
    profile["elapsed_seconds"] = time.perf_counter() - started
    return result_rows, contribution_rows, profile, model_records
