"""Leakage-safe direct-forecast persistence and linear baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..sequence_data import PreparedDirectForecastData


LinearKind = Literal["AR", "ARX"]


def target_indices(
    data: PreparedDirectForecastData, partition: str
) -> np.ndarray:
    start, stop = data.split_target_intervals[partition]
    return np.arange(start, stop, dtype=np.int64)


def persistence_predict(
    data: PreparedDirectForecastData, partition: str
) -> np.ndarray:
    """Return y at the forecast origin; no future values are accessed."""

    targets = target_indices(data, partition)
    origins = targets - data.horizon
    return data.y_scaled[origins].astype(np.float64, copy=True)


def linear_design(
    data: PreparedDirectForecastData,
    partition: str,
    *,
    kind: LinearKind,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build current-to-past AR/ARX features using target membership."""

    if kind == "AR" and data.track not in {"AR", "XAR"}:
        raise ValueError("AR features require an output-history data track.")
    if kind == "ARX" and data.track != "XAR":
        raise ValueError("ARX features require Track-XAR data.")
    targets = target_indices(data, partition)
    origins = targets - data.horizon
    rows: list[np.ndarray] = []
    for origin in origins:
        ar = data.y_scaled[origin - data.L_y + 1 : origin + 1][::-1]
        if kind == "AR":
            rows.append(ar)
            continue
        x = data.x_scaled[origin - data.L_x + 1 : origin + 1][::-1]
        rows.append(np.concatenate((ar, x.T.reshape(-1))))
    design = np.asarray(rows, dtype=np.float64)
    target = data.y_scaled[targets].astype(np.float64, copy=False)
    return design, target, targets


@dataclass(frozen=True, slots=True)
class LinearDirectForecaster:
    kind: LinearKind
    coefficients: np.ndarray
    intercept: float
    rank: int
    singular_values: np.ndarray

    @classmethod
    def fit(
        cls,
        data: PreparedDirectForecastData,
        *,
        kind: LinearKind,
        rcond: float | None = None,
    ) -> "LinearDirectForecaster":
        """Fit unregularized least squares on the training targets only."""

        design, target, _ = linear_design(data, "train", kind=kind)
        augmented = np.column_stack((np.ones(len(design)), design))
        solution, _, rank, singular_values = np.linalg.lstsq(
            augmented, target, rcond=rcond
        )
        return cls(
            kind=kind,
            coefficients=solution[1:].copy(),
            intercept=float(solution[0]),
            rank=int(rank),
            singular_values=singular_values.copy(),
        )

    def predict(
        self, data: PreparedDirectForecastData, partition: str
    ) -> tuple[np.ndarray, np.ndarray]:
        design, _, indices = linear_design(data, partition, kind=self.kind)
        return self.intercept + design @ self.coefficients, indices
