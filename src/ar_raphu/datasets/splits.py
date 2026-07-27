"""Chronology and official-split integrity checks."""

from __future__ import annotations

import numpy as np

from .base import DynamicDataset


_ORDER = {"warmup": 0, "train": 1, "validation": 2, "test": 3}


def validate_split_integrity(dataset: DynamicDataset) -> dict[str, int | bool]:
    """Reject time reversal and within-record split regression."""

    for sequence in np.unique(dataset.sequence_id):
        indices = np.flatnonzero(dataset.sequence_id == sequence)
        codes = np.array([_ORDER[str(value)] for value in dataset.split[indices]])
        if np.any(np.diff(codes) < 0):
            raise ValueError(f"Split order regresses inside sequence {sequence!r}.")
        if dataset.timestamps is not None:
            timestamps = np.asarray(dataset.timestamps)[indices]
            if np.any(timestamps[1:] < timestamps[:-1]):
                raise ValueError(
                    f"Timestamps regress inside sequence {sequence!r}."
                )
    return {
        "sequence_count": int(len(np.unique(dataset.sequence_id))),
        "train_count": int(np.sum(dataset.split == "train")),
        "validation_count": int(np.sum(dataset.split == "validation")),
        "test_count": int(np.sum(dataset.split == "test")),
        "time_order_verified": True,
        "split_verified": True,
    }


def development_tail_split(
    n_time: int,
    *,
    validation_fraction: float,
) -> np.ndarray:
    """Create an ordered train/validation split for an estimation record."""

    if n_time < 3 or not 0.0 < validation_fraction < 0.5:
        raise ValueError("Need at least three rows and validation fraction < 0.5.")
    stop = int(np.floor(n_time * (1.0 - validation_fraction)))
    split = np.full(n_time, "train", dtype="U10")
    split[stop:] = "validation"
    return split
