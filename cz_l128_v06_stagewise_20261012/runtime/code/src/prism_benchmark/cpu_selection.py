from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class SelectedCandidate:
    candidate: Any
    mean_loss: float
    standard_error_of_best: float
    best_mean_loss: float
    within_one_se: bool


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(np.square(error), dtype=np.float64))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    mse_value = float(np.mean(np.square(error), dtype=np.float64))
    rmse = math.sqrt(mse_value)
    mae = float(np.mean(np.abs(error), dtype=np.float64))
    denominator = float(np.sum(np.square(y_true - np.mean(y_true, dtype=np.float64)), dtype=np.float64))
    r2 = float("nan") if denominator == 0.0 else 1.0 - float(np.sum(np.square(error))) / denominator
    std = float(np.std(y_true, ddof=0))
    nrmse = float("nan") if std == 0.0 else rmse / std
    return {"mse": mse_value, "rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse}


def select_one_se(
    losses: dict[Any, list[float]],
    complexity_key: Callable[[Any], tuple[Any, ...]],
) -> SelectedCandidate:
    if not losses:
        raise ValueError("no candidates")
    fold_counts = {len(values) for values in losses.values()}
    if len(fold_counts) != 1 or next(iter(fold_counts)) < 2:
        raise ValueError(f"one-SE requires equal fold counts >=2, got {fold_counts}")
    means = {candidate: float(np.mean(values, dtype=np.float64)) for candidate, values in losses.items()}
    best = min(means, key=lambda candidate: (means[candidate], complexity_key(candidate)))
    best_values = np.asarray(losses[best], dtype=np.float64)
    standard_error = float(np.std(best_values, ddof=1) / math.sqrt(len(best_values)))
    threshold = means[best] + standard_error
    eligible = [candidate for candidate, mean in means.items() if mean <= threshold]
    selected = min(eligible, key=complexity_key)
    return SelectedCandidate(
        candidate=selected,
        mean_loss=means[selected],
        standard_error_of_best=standard_error,
        best_mean_loss=means[best],
        within_one_se=True,
    )


class Standardizer:
    def fit(self, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=np.float64)
        self.mean_ = np.mean(values, axis=0, dtype=np.float64)
        self.scale_ = np.std(values, axis=0, ddof=0)
        self.scale_ = np.where(self.scale_ > 0.0, self.scale_, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)
