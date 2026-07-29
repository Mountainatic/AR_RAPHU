"""FAST-B AR-residual/exogenous lag-correlation audit."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ar_raphu.cz_real.linear import TrainScaler, window_designs

from .fast_linear import DenseRidgePath
from .residualization import (
    FAST_TASKS,
    build_fast_folds,
    moving_block_indices,
    target_indices,
)


def _column_correlations(target: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    x = np.asarray(matrix, dtype=np.float64)
    yc = y - y.mean()
    xc = x - x.mean(axis=0)
    denominator = np.sqrt(
        np.sum(yc**2)
        * np.maximum(np.sum(xc**2, axis=0), np.finfo(np.float64).eps)
    )
    return (yc @ xc) / np.maximum(denominator, np.finfo(np.float64).eps)


def _contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            intervals.append((start, index - 1))
            start = None
    return intervals


def _block_permutations(
    samples: int,
    *,
    replicates: int,
    block_length: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    blocks = [
        np.arange(start, min(start + block_length, samples), dtype=np.int64)
        for start in range(0, samples, block_length)
    ]
    output = []
    for _ in range(replicates):
        order = rng.permutation(len(blocks))
        output.append(np.concatenate([blocks[index] for index in order]))
    return output


def lag_correlation_audit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    input_names: Iterable[str],
    ridge_grid: Iterable[float],
    bootstrap_replicates: int,
    random_seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    names = tuple(input_names)
    alphas = tuple(float(value) for value in ridge_grid)
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    direction_by_task_input: dict[tuple[str, int], list[int]] = {}
    for task_index, task in enumerate(FAST_TASKS):
        fold_cache = []
        for fold in build_fast_folds(len(y), task):
            train_targets = target_indices(
                start=0, stop=fold.effective_train_stop, task=task
            )
            validation_targets = target_indices(
                start=fold.validation_start,
                stop=fold.validation_stop,
                task=task,
            )
            scaler = TrainScaler.fit(x, y, fold.effective_train_stop)
            _, train_ar = window_designs(
                x,
                y,
                targets=train_targets,
                horizon=task.horizon,
                L_x=task.L_x,
                L_y=task.L_y,
                scaler=scaler,
            )
            _, validation_ar = window_designs(
                x,
                y,
                targets=validation_targets,
                horizon=task.horizon,
                L_x=task.L_x,
                L_y=task.L_y,
                scaler=scaler,
            )
            path = DenseRidgePath.fit(train_ar, y[train_targets])
            fold_cache.append(
                {
                    "fold": fold,
                    "targets": validation_targets,
                    "validation_ar": validation_ar,
                    "path": path,
                }
            )
        mean_losses = []
        for alpha in alphas:
            mean_losses.append(
                float(
                    np.mean(
                        [
                            np.mean(
                                (
                                    item["path"].predict(
                                        item["validation_ar"], alpha
                                    )
                                    - y[item["targets"]]
                                )
                                ** 2
                            )
                            for item in fold_cache
                        ]
                    )
                )
            )
        selected_index = min(
            range(len(alphas)), key=lambda index: (mean_losses[index], index)
        )
        selected_alpha = alphas[selected_index]
        summary[task.name] = {
            "AR_ridge_alpha": selected_alpha,
            "mean_validation_MSE_by_alpha": dict(
                zip(map(str, alphas), mean_losses)
            ),
        }
        for item in fold_cache:
            fold = item["fold"]
            validation_targets = item["targets"]
            residual = y[validation_targets] - item["path"].predict(
                item["validation_ar"], selected_alpha
            )
            origins = validation_targets - task.horizon
            lags = np.arange(task.maximum_correlation_lag + 1, dtype=np.int64)
            rng = np.random.default_rng(
                random_seed + 100_000 * task_index + fold.fold
            )
            bootstrap, block_length = moving_block_indices(
                len(residual),
                replicates=bootstrap_replicates,
                rng=rng,
            )
            permutations = _block_permutations(
                len(residual),
                replicates=bootstrap_replicates,
                block_length=block_length,
                rng=rng,
            )
            for variable, name in enumerate(names):
                lag_matrix = x[
                    origins[:, None] - lags[None, :], variable
                ]
                correlations = _column_correlations(residual, lag_matrix)
                boot = np.empty(
                    (bootstrap_replicates, len(lags)), dtype=np.float64
                )
                for replicate, indices in enumerate(bootstrap):
                    boot[replicate] = _column_correlations(
                        residual[indices], lag_matrix[indices]
                    )
                lower = np.quantile(boot, 0.025, axis=0)
                upper = np.quantile(boot, 0.975, axis=0)
                significant = (lower > 0.0) | (upper < 0.0)
                intervals = _contiguous_intervals(significant)
                permutation_maxima = np.asarray(
                    [
                        np.max(
                            np.abs(
                                _column_correlations(
                                    residual, lag_matrix[indices]
                                )
                            )
                        )
                        for indices in permutations
                    ],
                    dtype=np.float64,
                )
                maximum_index = int(np.argmax(np.abs(correlations)))
                maximum_sign = int(np.sign(correlations[maximum_index]))
                direction_by_task_input.setdefault(
                    (task.name, variable), []
                ).append(maximum_sign)
                interval_text = ";".join(
                    f"{start}:{stop}" for start, stop in intervals
                )
                for lag, correlation, lo, hi, is_significant in zip(
                    lags,
                    correlations,
                    lower,
                    upper,
                    significant,
                    strict=True,
                ):
                    rows.append(
                        {
                            "task": task.name,
                            "Lx": task.L_x,
                            "Ly": task.L_y,
                            "horizon": task.horizon,
                            "fold": fold.fold,
                            "input": name,
                            "input_index": variable,
                            "lag": int(lag),
                            "correlation": float(correlation),
                            "bootstrap_lower": float(lo),
                            "bootstrap_upper": float(hi),
                            "bootstrap_block_length": block_length,
                            "significant": bool(is_significant),
                            "continuous_significant_intervals": interval_text,
                            "block_permutation_max_abs_p95": float(
                                np.quantile(permutation_maxima, 0.95)
                            ),
                            "is_fold_maximum": bool(lag == maximum_index),
                            "maximum_absolute_correlation": float(
                                abs(correlations[maximum_index])
                            ),
                            "maximum_correlation_lag": maximum_index,
                            "fold_direction_consistent": False,
                        }
                    )
    for row in rows:
        signs = direction_by_task_input[
            (str(row["task"]), int(row["input_index"]))
        ]
        row["fold_direction_consistent"] = bool(
            len(signs) == 2 and signs[0] != 0 and signs[0] == signs[1]
        )
    return rows, summary
