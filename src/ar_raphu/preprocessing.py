"""Train-only preprocessing primitives for the frozen CZ rolling protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data_protocol import Interval, get_fold


def _as_numeric_2d(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric.")
    return array


def _as_numeric_1d(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric.")
    return array


@dataclass(frozen=True, slots=True)
class TrainOnlyStandardizer:
    """A fold-bound z-score transform fitted on its raw train interval only.

    This class is an available implementation, not a protocol choice. The
    experiment configuration must explicitly select a scaler before training.
    Constant columns use scale 1 so their transformed value is exactly zero.
    """

    fold_number: int
    fitted_interval: Interval
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        fold_number: int,
    ) -> "TrainOnlyStandardizer":
        x_array = _as_numeric_2d(x, name="x")
        y_array = _as_numeric_1d(y, name="y")
        if len(x_array) != len(y_array):
            raise ValueError("x and y must have the same number of rows.")

        fold = get_fold(fold_number)
        train = fold.train
        if len(x_array) < train.stop:
            raise ValueError("Source arrays do not cover the fold train interval.")
        train_x = x_array[train.start : train.stop].astype(np.float64)
        train_y = y_array[train.start : train.stop].astype(np.float64)
        if not np.isfinite(train_x).all() or not np.isfinite(train_y).all():
            raise ValueError("Fold train values contain non-finite values.")
        x_mean = train_x.mean(axis=0)
        x_scale = train_x.std(axis=0, ddof=0)
        x_scale = np.where(x_scale == 0.0, 1.0, x_scale)
        y_mean = float(train_y.mean())
        y_scale = float(train_y.std(ddof=0))
        if y_scale == 0.0:
            y_scale = 1.0
        return cls(
            fold_number=fold_number,
            fitted_interval=train,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
        )

    def _validate_fold(self, fold_number: int) -> None:
        if fold_number != self.fold_number:
            raise ValueError(
                f"Scaler belongs to fold {self.fold_number}, not fold {fold_number}."
            )
        if self.fitted_interval != get_fold(fold_number).train:
            raise RuntimeError("Scaler provenance is not the frozen train interval.")

    def transform_x(self, values: np.ndarray, *, fold_number: int) -> np.ndarray:
        self._validate_fold(fold_number)
        array = np.asarray(values)
        if array.shape[-1] != self.x_mean.shape[0]:
            raise ValueError("External feature count does not match scaler state.")
        return (array - self.x_mean) / self.x_scale

    def transform_y(self, values: np.ndarray, *, fold_number: int) -> np.ndarray:
        self._validate_fold(fold_number)
        return (np.asarray(values) - self.y_mean) / self.y_scale

    def inverse_y(self, values: np.ndarray, *, fold_number: int) -> np.ndarray:
        self._validate_fold(fold_number)
        return np.asarray(values) * self.y_scale + self.y_mean

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "zscore_population_std",
            "fit_partition": "train",
            "fold_number": self.fold_number,
            "fit_interval": [
                self.fitted_interval.start,
                self.fitted_interval.stop,
            ],
            "x_mean": self.x_mean.tolist(),
            "x_scale": self.x_scale.tolist(),
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
            "zero_variance_policy": "scale_one_transform_to_zero",
        }


def v20_grid_ranges_from_train(
    x: np.ndarray,
    y: np.ndarray,
    scaler: TrainOnlyStandardizer,
    *,
    fold_number: int,
    pad_fraction: float = 0.05,
    minimum_pad: float = 0.05,
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Fit V20 first-layer grid ranges from raw fold-train rows only."""

    if pad_fraction < 0 or minimum_pad <= 0:
        raise ValueError("Grid padding parameters must be nonnegative/positive.")
    scaler._validate_fold(fold_number)
    fold = get_fold(fold_number)
    x_array = _as_numeric_2d(x, name="x")
    y_array = _as_numeric_1d(y, name="y")
    if len(x_array) != len(y_array) or len(x_array) < fold.train.stop:
        raise ValueError("Source arrays do not cover the fold train interval.")

    train_x = scaler.transform_x(
        x_array[fold.train.start : fold.train.stop],
        fold_number=fold_number,
    )
    train_y = scaler.transform_y(
        y_array[fold.train.start : fold.train.stop],
        fold_number=fold_number,
    )
    if not np.isfinite(train_x).all() or not np.isfinite(train_y).all():
        raise ValueError("Fold train values contain non-finite values.")

    def padded_range(values: np.ndarray) -> tuple[float, float]:
        lower = float(np.min(values))
        upper = float(np.max(values))
        pad = max((upper - lower) * pad_fraction, minimum_pad)
        return lower - pad, upper + pad

    x_ranges = [
        padded_range(train_x[:, channel])
        for channel in range(train_x.shape[1])
    ]
    return x_ranges, padded_range(train_y)
