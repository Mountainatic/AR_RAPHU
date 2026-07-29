"""FAST-A conditional input energy and shared leakage-safe index helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterable

import numpy as np
import scipy.linalg


@dataclass(frozen=True, slots=True)
class FastTask:
    name: str
    L_x: int
    L_y: int
    horizon: int
    maximum_correlation_lag: int


@dataclass(frozen=True, slots=True)
class FastFold:
    fold: int
    nominal_train_stop: int
    effective_train_stop: int
    validation_start: int
    validation_stop: int
    purge_gap: int


FAST_TASKS = (
    FastTask("short", 64, 16, 1, 50),
    FastTask("medium", 256, 32, 15, 150),
    FastTask("long", 512, 64, 60, 300),
)


def build_fast_folds(samples: int, task: FastTask) -> tuple[FastFold, ...]:
    """Create the frozen 0-50/50-60 and 0-70/70-80 blocked folds."""

    gap = max(task.L_x - 1, task.L_y - 1) + task.horizon
    rows = []
    for fold_id, (train_fraction, validation_fraction) in enumerate(
        ((0.5, 0.6), (0.7, 0.8)), start=1
    ):
        nominal_stop = floor(samples * train_fraction)
        effective_stop = nominal_stop - gap
        if effective_stop <= max(task.L_x, task.L_y):
            raise ValueError("Purge leaves insufficient training history.")
        rows.append(
            FastFold(
                fold=fold_id,
                nominal_train_stop=nominal_stop,
                effective_train_stop=effective_stop,
                validation_start=nominal_stop,
                validation_stop=floor(samples * validation_fraction),
                purge_gap=gap,
            )
        )
    return tuple(rows)


def target_indices(
    *,
    start: int,
    stop: int,
    task: FastTask,
) -> np.ndarray:
    first = max(start, task.horizon + max(task.L_x, task.L_y) - 1)
    if first >= stop:
        raise ValueError("No eligible target indices.")
    return np.arange(first, stop, dtype=np.int64)


def history_matrix(
    values: np.ndarray,
    *,
    targets: np.ndarray,
    horizon: int,
    length: int,
) -> np.ndarray:
    origins = np.asarray(targets, dtype=np.int64) - int(horizon)
    offsets = np.arange(length, dtype=np.int64)
    if int(origins.min()) - length + 1 < 0:
        raise ValueError("History precedes sequence start.")
    return np.asarray(values, dtype=np.float64)[
        origins[:, None] - offsets[None, :]
    ]


@dataclass(slots=True)
class MultiTargetRidge:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (
            np.asarray(features, dtype=np.float64) - self.feature_mean
        ) / self.feature_scale
        predicted = standardized @ self.coefficients
        return predicted * self.target_scale + self.target_mean


def fit_multi_target_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
) -> MultiTargetRidge:
    x = np.asarray(features, dtype=np.float64)
    z = np.asarray(targets, dtype=np.float64)
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale == 0.0] = 1.0
    z_mean = z.mean(axis=0)
    z_scale = z.std(axis=0)
    z_scale[z_scale == 0.0] = 1.0
    xs = (x - x_mean) / x_scale
    zs = (z - z_mean) / z_scale
    gram = xs.T @ xs / len(xs)
    rhs = xs.T @ zs / len(xs)
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
    return MultiTargetRidge(
        feature_mean=x_mean,
        feature_scale=x_scale,
        target_mean=z_mean,
        target_scale=z_scale,
        coefficients=coefficients,
    )


def moving_block_indices(
    samples: int,
    *,
    replicates: int,
    rng: np.random.Generator,
    block_length: int | None = None,
) -> tuple[np.ndarray, int]:
    length = int(block_length or max(2, round(samples ** (1.0 / 3.0))))
    blocks = int(np.ceil(samples / length))
    starts = rng.integers(0, samples, size=(replicates, blocks))
    offsets = np.arange(length, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % samples
    return indices.reshape(replicates, -1)[:, :samples], length


def _energy_status(ratio: float) -> str:
    if ratio < 0.01:
        return "EXTREMELY_WEAK_INDEPENDENT_EXCITATION"
    if ratio < 0.05:
        return "WEAK_INDEPENDENT_EXCITATION"
    if ratio < 0.15:
        return "MODERATE_INDEPENDENT_EXCITATION"
    return "STRONG_INDEPENDENT_EXCITATION"


def conditional_energy_audit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    input_names: Iterable[str],
    ridge_grid: Iterable[float],
    bootstrap_replicates: int,
    random_seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run FAST-A with alpha selected across the two blocked folds."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    names = tuple(input_names)
    alphas = tuple(float(value) for value in ridge_grid)
    rows: list[dict[str, object]] = []
    selection: dict[str, object] = {}
    for task_index, task in enumerate(FAST_TASKS):
        folds = build_fast_folds(len(y), task)
        fold_cache: dict[tuple[int, int], tuple[np.ndarray, ...]] = {}
        for fold in folds:
            train_targets = target_indices(
                start=0, stop=fold.effective_train_stop, task=task
            )
            validation_targets = target_indices(
                start=fold.validation_start,
                stop=fold.validation_stop,
                task=task,
            )
            train_y = history_matrix(
                y,
                targets=train_targets,
                horizon=task.horizon,
                length=task.L_y,
            )
            validation_y = history_matrix(
                y,
                targets=validation_targets,
                horizon=task.horizon,
                length=task.L_y,
            )
            for variable in range(x.shape[1]):
                train_x = history_matrix(
                    x[:, variable],
                    targets=train_targets,
                    horizon=task.horizon,
                    length=task.L_x,
                )
                validation_x = history_matrix(
                    x[:, variable],
                    targets=validation_targets,
                    horizon=task.horizon,
                    length=task.L_x,
                )
                fold_cache[(fold.fold, variable)] = (
                    train_y,
                    validation_y,
                    train_x,
                    validation_x,
                )
        for variable, name in enumerate(names):
            losses = []
            for alpha in alphas:
                per_fold = []
                for fold in folds:
                    train_y, validation_y, train_x, validation_x = fold_cache[
                        (fold.fold, variable)
                    ]
                    fit = fit_multi_target_ridge(
                        train_y, train_x, alpha=alpha
                    )
                    prediction = fit.predict(validation_y)
                    scale = np.maximum(
                        train_x.std(axis=0), np.finfo(np.float64).eps
                    )
                    per_fold.append(
                        float(np.mean(((prediction - validation_x) / scale) ** 2))
                    )
                losses.append(float(np.mean(per_fold)))
            selected_index = min(
                range(len(alphas)), key=lambda index: (losses[index], index)
            )
            selected_alpha = alphas[selected_index]
            selection[f"{task.name}:{variable}"] = {
                "input": name,
                "ridge_alpha": selected_alpha,
                "mean_validation_losses": {
                    str(alpha): loss for alpha, loss in zip(alphas, losses)
                },
            }
            for fold in folds:
                train_y, validation_y, train_x, validation_x = fold_cache[
                    (fold.fold, variable)
                ]
                fit = fit_multi_target_ridge(
                    train_y, train_x, alpha=selected_alpha
                )
                residual = validation_x - fit.predict(validation_y)
                centered = validation_x - train_x.mean(axis=0)
                row_energy_residual = np.sum(residual**2, axis=1)
                row_energy_total = np.sum(centered**2, axis=1)
                denominator = max(
                    float(row_energy_total.sum()),
                    np.finfo(np.float64).eps,
                )
                ratio = float(row_energy_residual.sum() / denominator)
                rng = np.random.default_rng(
                    random_seed
                    + 10_000 * task_index
                    + 100 * variable
                    + fold.fold
                )
                bootstrap, block_length = moving_block_indices(
                    len(validation_x),
                    replicates=bootstrap_replicates,
                    rng=rng,
                )
                ratios = np.sum(row_energy_residual[bootstrap], axis=1) / np.maximum(
                    np.sum(row_energy_total[bootstrap], axis=1),
                    np.finfo(np.float64).eps,
                )
                rows.append(
                    {
                        "input": name,
                        "input_index": variable,
                        "scale": task.name,
                        "Lx": task.L_x,
                        "Ly": task.L_y,
                        "horizon": task.horizon,
                        "fold": fold.fold,
                        "ridge_alpha": selected_alpha,
                        "conditional_energy_ratio": ratio,
                        "explained_by_y_history_ratio": 1.0 - ratio,
                        "bootstrap_lower": float(np.quantile(ratios, 0.025)),
                        "bootstrap_upper": float(np.quantile(ratios, 0.975)),
                        "bootstrap_block_length": block_length,
                        "status": _energy_status(ratio),
                    }
                )
    return rows, selection
