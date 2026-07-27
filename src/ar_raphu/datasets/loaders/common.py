"""Adapters from official nonlinear-benchmarks records to DynamicDataset."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..base import DynamicDataset


def array_record(
    *,
    u: np.ndarray,
    y: np.ndarray,
    name: str,
    sampling_time: float | None,
) -> SimpleNamespace:
    """Construct the minimal immutable-like record contract used by adapters."""
    return SimpleNamespace(
        u=np.asarray(u, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        name=name,
        sampling_time=sampling_time,
    )


def _column(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    raise ValueError(f"{name} must be one- or two-dimensional, got {array.shape}.")


def records_to_dynamic_dataset(
    records: Iterable[tuple[Any, str, str]],
    *,
    dataset_id: str,
    source_metadata: dict[str, Any],
    feature_names: tuple[str, ...] = ("input",),
    target_names: tuple[str, ...] = ("output",),
) -> DynamicDataset:
    """Concatenate independent official records without inventing validation data.

    ``records`` entries are ``(official_record, split, stable_record_name)``.
    Official estimation records remain ``train`` and official confirmation
    records remain ``test``. A later, preregistered development partitioner is
    responsible for creating validation data from training records.
    """

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sequence_ids: list[np.ndarray] = []
    splits: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    sampling_times: dict[str, float | None] = {}
    offset = 0.0

    for sequence_number, (record, split, record_name) in enumerate(records):
        x = _column(record.u, name=f"{record_name}.u")
        y = _column(record.y, name=f"{record_name}.y")
        if len(x) != len(y):
            raise ValueError(f"{record_name}: input/output lengths differ.")
        if not len(x):
            raise ValueError(f"{record_name}: empty official record.")
        dt_value = getattr(record, "sampling_time", None)
        dt = None if dt_value is None else float(dt_value)
        sampling_times[record_name] = dt
        local_time = np.arange(len(x), dtype=np.float64)
        if dt is not None:
            local_time *= dt
        timestamps.append(local_time + offset)
        offset = float(timestamps[-1][-1] + (dt if dt is not None else 1.0))
        xs.append(x)
        ys.append(y)
        sequence_ids.append(
            np.full(len(x), f"{sequence_number:04d}:{record_name}", dtype=object)
        )
        splits.append(np.full(len(x), split, dtype=object))

    if not xs:
        raise ValueError("At least one official record is required.")
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    if x_all.shape[1] != len(feature_names):
        raise ValueError("Official input width does not match feature_names.")
    if y_all.shape[1] != len(target_names):
        raise ValueError("Official output width does not match target_names.")
    metadata = {
        **source_metadata,
        "dataset_id": dataset_id,
        "sequence_count": len(xs),
        "sampling_time_by_sequence": sampling_times,
        "validation_status": "NOT_YET_DEFINED",
        "official_test_locked": True,
    }
    quality_mask = np.isfinite(np.concatenate((x_all, y_all), axis=1))
    label_mask = np.isfinite(y_all)
    return DynamicDataset(
        x=x_all,
        y=y_all,
        timestamps=np.concatenate(timestamps),
        sequence_id=np.concatenate(sequence_ids),
        split=np.concatenate(splits),
        label_mask=label_mask,
        quality_mask=quality_mask,
        feature_names=feature_names,
        target_names=target_names,
        metadata=metadata,
    )
