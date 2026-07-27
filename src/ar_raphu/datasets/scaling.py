"""Train-only z-score fitting and explicit OOD range checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import DynamicDataset


@dataclass(frozen=True, slots=True)
class TrainOnlyStandardizer:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    fitted_row_count: int

    @classmethod
    def fit(cls, dataset: DynamicDataset) -> "TrainOnlyStandardizer":
        train = dataset.split == "train"
        if not np.any(train):
            raise ValueError("Training rows are required.")
        x_quality = dataset.quality_mask[:, : dataset.n_features]
        y_quality = dataset.quality_mask[:, dataset.n_features :]
        x_mean = np.empty(dataset.n_features, dtype=np.float64)
        x_scale = np.empty_like(x_mean)
        x_min = np.empty_like(x_mean)
        x_max = np.empty_like(x_mean)
        for column in range(dataset.n_features):
            mask = train & x_quality[:, column]
            if not np.any(mask):
                raise ValueError(f"No valid training values for x column {column}.")
            values = np.asarray(dataset.x[mask, column], dtype=np.float64)
            x_mean[column] = values.mean()
            x_scale[column] = max(values.std(ddof=0), np.finfo(float).eps)
            x_min[column] = values.min()
            x_max[column] = values.max()
        y_mean = np.empty(dataset.n_targets, dtype=np.float64)
        y_scale = np.empty_like(y_mean)
        for column in range(dataset.n_targets):
            mask = (
                train
                & dataset.label_mask[:, column]
                & y_quality[:, column]
            )
            if not np.any(mask):
                raise ValueError(f"No valid training labels for y column {column}.")
            values = np.asarray(dataset.y[mask, column], dtype=np.float64)
            y_mean[column] = values.mean()
            y_scale[column] = max(values.std(ddof=0), np.finfo(float).eps)
        return cls(
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            x_min=x_min,
            x_max=x_max,
            fitted_row_count=int(np.sum(train)),
        )

    def transform(self, dataset: DynamicDataset) -> DynamicDataset:
        if dataset.n_features != len(self.x_mean):
            raise ValueError("Feature count differs from fitted scaler.")
        if dataset.n_targets != len(self.y_mean):
            raise ValueError("Target count differs from fitted scaler.")
        return DynamicDataset(
            x=(np.asarray(dataset.x, dtype=np.float64) - self.x_mean)
            / self.x_scale,
            y=(np.asarray(dataset.y, dtype=np.float64) - self.y_mean)
            / self.y_scale,
            timestamps=dataset.timestamps,
            sequence_id=dataset.sequence_id,
            split=dataset.split,
            label_mask=dataset.label_mask,
            quality_mask=dataset.quality_mask,
            feature_names=dataset.feature_names,
            target_names=dataset.target_names,
            metadata={**dataset.metadata, "scaler_fit_split": "train"},
        )

    def x_ood_mask(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.x_min):
            raise ValueError("Last dimension must match feature count.")
        return np.any((array < self.x_min) | (array > self.x_max), axis=-1)
