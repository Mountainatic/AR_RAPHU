"""Immutable semantic contract shared by public dynamic benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SPLIT_VALUES = frozenset({"warmup", "train", "validation", "test"})


@dataclass(frozen=True, slots=True)
class DynamicDataset:
    """A collection of one or more independent ordered records.

    Arrays retain the source-record row order.  A sequence identifier must
    occupy one contiguous block so causal windows can never join records.
    """

    x: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray | None
    sequence_id: np.ndarray
    split: np.ndarray
    label_mask: np.ndarray
    quality_mask: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        sequence_id = np.asarray(self.sequence_id)
        split = np.asarray(self.split)
        label_mask = np.asarray(self.label_mask)
        quality_mask = np.asarray(self.quality_mask)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
            raise ValueError("x and y must be non-empty 2D arrays of equal length.")
        n_time = len(x)
        if sequence_id.shape != (n_time,) or split.shape != (n_time,):
            raise ValueError("sequence_id and split must have shape [n_time].")
        if label_mask.shape != y.shape:
            raise ValueError("label_mask must match y.")
        if quality_mask.shape != (n_time, x.shape[1] + y.shape[1]):
            raise ValueError("quality_mask must cover every x and y channel.")
        if len(self.feature_names) != x.shape[1]:
            raise ValueError("feature_names do not match x.")
        if len(self.target_names) != y.shape[1]:
            raise ValueError("target_names do not match y.")
        if self.timestamps is not None and np.asarray(self.timestamps).shape != (
            n_time,
        ):
            raise ValueError("timestamps must be None or have shape [n_time].")
        unknown = set(map(str, np.unique(split))) - SPLIT_VALUES
        if unknown:
            raise ValueError(f"Unknown split values: {sorted(unknown)}")
        if label_mask.dtype != np.bool_ or quality_mask.dtype != np.bool_:
            raise TypeError("label_mask and quality_mask must be boolean.")
        for value in np.unique(sequence_id):
            positions = np.flatnonzero(sequence_id == value)
            if positions[-1] - positions[0] + 1 != len(positions):
                raise ValueError(f"sequence_id {value!r} is not contiguous.")
        x_valid = quality_mask[:, : x.shape[1]]
        y_valid = quality_mask[:, x.shape[1] :]
        if not np.isfinite(x[x_valid]).all():
            raise ValueError("A quality-valid x value is not finite.")
        if not np.isfinite(y[y_valid & label_mask]).all():
            raise ValueError("A quality-valid observed label is not finite.")

    @property
    def n_time(self) -> int:
        return int(len(self.x))

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_targets(self) -> int:
        return int(self.y.shape[1])

    def indices(self, split: str) -> np.ndarray:
        if split not in SPLIT_VALUES:
            raise ValueError(f"Unknown split {split!r}.")
        return np.flatnonzero(self.split == split)
