"""Sequence-safe direct-forecast windows with target-index split ownership."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import DynamicDataset
from .scaling import TrainOnlyStandardizer


@dataclass(frozen=True, slots=True)
class WindowedTask:
    x_history: np.ndarray
    y_history: np.ndarray
    target: np.ndarray
    origin_index: np.ndarray
    target_index: np.ndarray
    sequence_id: np.ndarray
    split: np.ndarray
    ood_mask: np.ndarray
    target_name: str
    horizon: int
    L_x: int
    L_y: int


def build_windowed_task(
    dataset: DynamicDataset,
    *,
    target: int,
    horizon: int,
    L_x: int,
    L_y: int,
    include_splits: tuple[str, ...] = ("train", "validation", "test"),
    standardizer: TrainOnlyStandardizer | None = None,
) -> WindowedTask:
    """Build current-to-past histories ending at origin t for target t+h."""

    if not 0 <= target < dataset.n_targets:
        raise ValueError("Invalid target column.")
    if horizon <= 0 or L_x <= 0 or L_y <= 0:
        raise ValueError("horizon, L_x and L_y must be positive.")
    target_indices: list[int] = []
    origin_indices: list[int] = []
    x_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    targets: list[float] = []
    sequence_ids: list[object] = []
    splits: list[str] = []
    ood: list[bool] = []
    x_offsets = np.arange(L_x, dtype=np.int64)
    y_offsets = np.arange(L_y, dtype=np.int64)
    x_quality = dataset.quality_mask[:, : dataset.n_features]
    y_quality = dataset.quality_mask[:, dataset.n_features + target]
    allowed = set(include_splits)
    for target_index in range(dataset.n_time):
        if str(dataset.split[target_index]) not in allowed:
            continue
        if not dataset.label_mask[target_index, target] or not y_quality[target_index]:
            continue
        origin = target_index - horizon
        if origin < 0 or origin - max(L_x, L_y) + 1 < 0:
            continue
        x_indices = origin - x_offsets
        y_indices = origin - y_offsets
        sequence = dataset.sequence_id[target_index]
        if (
            np.any(dataset.sequence_id[x_indices] != sequence)
            or np.any(dataset.sequence_id[y_indices] != sequence)
        ):
            continue
        if not np.all(x_quality[x_indices]) or not np.all(y_quality[y_indices]):
            continue
        x_window = np.asarray(dataset.x[x_indices], dtype=np.float64).T
        y_window = np.asarray(dataset.y[y_indices, target], dtype=np.float64)
        target_indices.append(target_index)
        origin_indices.append(origin)
        x_windows.append(x_window)
        y_windows.append(y_window)
        targets.append(float(dataset.y[target_index, target]))
        sequence_ids.append(sequence)
        splits.append(str(dataset.split[target_index]))
        if standardizer is None:
            ood.append(False)
        else:
            chronological = x_window[:, ::-1].T
            ood.append(bool(np.any(standardizer.x_ood_mask(chronological))))
    if not target_indices:
        raise ValueError("No eligible causal windows.")
    return WindowedTask(
        x_history=np.stack(x_windows),
        y_history=np.stack(y_windows),
        target=np.asarray(targets, dtype=np.float64),
        origin_index=np.asarray(origin_indices, dtype=np.int64),
        target_index=np.asarray(target_indices, dtype=np.int64),
        sequence_id=np.asarray(sequence_ids),
        split=np.asarray(splits),
        ood_mask=np.asarray(ood, dtype=bool),
        target_name=dataset.target_names[target],
        horizon=int(horizon),
        L_x=int(L_x),
        L_y=int(L_y),
    )
