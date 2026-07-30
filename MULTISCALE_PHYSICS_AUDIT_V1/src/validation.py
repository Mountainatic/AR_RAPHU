"""Rolling-origin folds and validation-only ridge selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .linear_q import (
    BlockRidgeFit,
    RidgeFit,
    fit_block_ridge,
    fit_ridge,
)


@dataclass(frozen=True, slots=True)
class Fold:
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_start_origin: int
    purge_samples: int


def rolling_origin_folds(
    origins: np.ndarray,
    fractions: Iterable[Iterable[float]],
    *,
    purge_samples: int,
) -> list[Fold]:
    points = np.asarray(origins, dtype=np.int64)
    specifications = [
        [
            int(np.floor(len(points) * float(start_fraction))),
            int(np.floor(len(points) * float(stop_fraction))),
        ]
        for start_fraction, stop_fraction in fractions
    ]
    # Slow profiles can have a large physical purge relative to their eligible
    # origin span. If only the first requested fold is slightly too early,
    # shift all four blocks right by the minimum deterministic amount. This
    # uses indices only (never target values) and preserves block widths/order.
    minimum_train_rows = 20
    first_start = specifications[0][0]
    while first_start < len(points):
        validation_start_origin = int(points[first_start])
        train_count = int(
            np.sum(points < validation_start_origin - int(purge_samples))
        )
        if train_count >= minimum_train_rows:
            break
        first_start += 1
    shift = first_start - specifications[0][0]
    if shift:
        shifted = [[start + shift, stop + shift] for start, stop in specifications]
        if shifted[-1][1] > len(points):
            raise ValueError(
                "INSUFFICIENT_INNER_FOLD_GEOMETRY:"
                f"n={len(points)}:purge={purge_samples}:shift={shift}"
            )
        specifications = shifted
    folds: list[Fold] = []
    for fold, (validation_start_index, validation_stop_index) in enumerate(
        specifications, start=1
    ):
        validation_start_origin = int(points[validation_start_index])
        train_mask = points < validation_start_origin - int(purge_samples)
        validation = np.arange(
            validation_start_index,
            validation_stop_index,
            dtype=np.int64,
        )
        train = np.flatnonzero(train_mask)
        if len(train) < minimum_train_rows or len(validation) < 5:
            raise ValueError(f"INSUFFICIENT_INNER_FOLD:{fold}")
        if int(points[train].max()) >= validation_start_origin - purge_samples:
            raise AssertionError("PURGE_VIOLATION")
        folds.append(
            Fold(
                fold,
                train,
                validation,
                validation_start_origin,
                int(purge_samples),
            )
        )
    return folds


def select_ridge_alpha(
    matrix: np.ndarray,
    target: np.ndarray,
    folds: Iterable[Fold],
    grid: Iterable[float],
) -> tuple[float, dict[str, float]]:
    losses: dict[str, float] = {}
    fold_list = list(folds)
    candidates = tuple(float(value) for value in grid)
    for alpha in candidates:
        per_fold = []
        for fold in fold_list:
            fit = fit_ridge(
                matrix[fold.train_indices],
                target[fold.train_indices],
                alpha=alpha,
            )
            prediction = fit.predict(matrix[fold.validation_indices])
            per_fold.append(
                float(
                    np.mean(
                        (
                            target[fold.validation_indices] - prediction
                        )
                        ** 2
                    )
                )
            )
        losses[str(alpha)] = float(np.mean(per_fold))
    selected = min(
        candidates,
        key=lambda value: (losses[str(value)], candidates.index(value)),
    )
    return selected, losses


def select_block_alphas(
    ar: np.ndarray,
    q: np.ndarray,
    target: np.ndarray,
    folds: Iterable[Fold],
    grid: Iterable[float],
) -> tuple[tuple[float, float | None], dict[str, float]]:
    candidates = tuple(float(value) for value in grid)
    pairs: list[tuple[float, float | None]] = [
        (alpha_ar, None) for alpha_ar in candidates
    ]
    pairs.extend(
        (alpha_ar, alpha_q)
        for alpha_ar in candidates
        for alpha_q in candidates
    )
    losses: dict[str, float] = {}
    fold_list = list(folds)
    for alpha_ar, alpha_q in pairs:
        key = f"{alpha_ar}|{'Q_ZERO' if alpha_q is None else alpha_q}"
        per_fold = []
        for fold in fold_list:
            fit = fit_block_ridge(
                ar[fold.train_indices],
                q[fold.train_indices],
                target[fold.train_indices],
                alpha_ar=alpha_ar,
                alpha_q=alpha_q,
            )
            prediction = fit.predict(
                ar[fold.validation_indices],
                q[fold.validation_indices],
            )
            per_fold.append(
                float(
                    np.mean(
                        (
                            target[fold.validation_indices] - prediction
                        )
                        ** 2
                    )
                )
            )
        losses[key] = float(np.mean(per_fold))
    selected = min(
        pairs,
        key=lambda pair: (
            losses[
                f"{pair[0]}|{'Q_ZERO' if pair[1] is None else pair[1]}"
            ],
            0 if pair[1] is None else 1,
            pairs.index(pair),
        ),
    )
    return selected, losses


def crossfit_predictions(
    matrix: np.ndarray,
    target: np.ndarray,
    folds: Iterable[Fold],
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    for fold in folds:
        fit = fit_ridge(
            matrix[fold.train_indices],
            target[fold.train_indices],
            alpha=alpha,
        )
        predictions.append(fit.predict(matrix[fold.validation_indices]))
        targets.append(target[fold.validation_indices])
        row_indices.append(fold.validation_indices)
    return (
        np.concatenate(row_indices),
        np.concatenate(targets),
        np.concatenate(predictions),
    )
