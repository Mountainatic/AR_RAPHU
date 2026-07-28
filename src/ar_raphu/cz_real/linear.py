"""Leakage-safe linear smoke baselines for furnace-A development folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TrainScaler:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, stop: int) -> "TrainScaler":
        if stop <= 0 or stop > len(y):
            raise ValueError("Invalid train-only scaler stop.")
        x_train = np.asarray(x[:stop], dtype=np.float64)
        y_train = np.asarray(y[:stop], dtype=np.float64)
        x_scale = x_train.std(axis=0)
        x_scale[x_scale == 0.0] = 1.0
        y_scale = float(y_train.std())
        if y_scale == 0.0:
            y_scale = 1.0
        return cls(
            x_mean=x_train.mean(axis=0),
            x_scale=x_scale,
            y_mean=float(y_train.mean()),
            y_scale=y_scale,
        )


def target_indices(
    *,
    start: int,
    stop: int,
    horizon: int,
    max_history: int,
) -> np.ndarray:
    first = max(start, horizon + max_history - 1)
    if first >= stop:
        raise ValueError("Target interval contains no eligible samples.")
    return np.arange(first, stop, dtype=np.int64)


def window_designs(
    x: np.ndarray,
    y: np.ndarray,
    *,
    targets: np.ndarray,
    horizon: int,
    L_x: int,
    L_y: int,
    scaler: TrainScaler,
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.asarray(targets, dtype=np.int64) - int(horizon)
    if origins.min() - max(L_x, L_y) + 1 < 0:
        raise ValueError("History precedes sequence start.")
    x_scaled = (np.asarray(x, dtype=np.float64) - scaler.x_mean) / scaler.x_scale
    y_scaled = (np.asarray(y, dtype=np.float64) - scaler.y_mean) / scaler.y_scale
    x_offsets = np.arange(L_x, dtype=np.int64)
    y_offsets = np.arange(L_y, dtype=np.int64)
    x_windows = x_scaled[origins[:, None] - x_offsets[None, :], :]
    y_windows = y_scaled[origins[:, None] - y_offsets[None, :]]
    return x_windows.reshape(len(targets), -1), y_windows


@dataclass(frozen=True, slots=True)
class OLSFit:
    coefficients: np.ndarray
    intercept: float
    rank: int

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=np.float64) @ self.coefficients + self.intercept


def fit_ols(matrix: np.ndarray, target: np.ndarray) -> OLSFit:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    augmented = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    coefficients, _, rank, _ = np.linalg.lstsq(augmented, y, rcond=None)
    return OLSFit(
        coefficients=coefficients[1:],
        intercept=float(coefficients[0]),
        rank=int(rank),
    )


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    residual = pred - y
    absolute = np.abs(residual)
    mse = float(np.mean(residual**2))
    variance = float(np.var(y))
    return {
        "RMSE_mm": float(np.sqrt(mse)),
        "MAE_mm": float(absolute.mean()),
        "R2": float(1.0 - mse / variance) if variance > 0.0 else float("nan"),
        "P50_abs_error_mm": float(np.quantile(absolute, 0.50)),
        "P90_abs_error_mm": float(np.quantile(absolute, 0.90)),
        "P95_abs_error_mm": float(np.quantile(absolute, 0.95)),
        "max_abs_error_mm": float(absolute.max()),
        "NRMSE_by_output_std": (
            float(np.sqrt(mse / variance)) if variance > 0.0 else float("nan")
        ),
        "MSE_mm2": mse,
    }
