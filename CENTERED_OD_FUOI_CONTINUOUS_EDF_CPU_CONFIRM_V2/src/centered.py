from __future__ import annotations

import numpy as np


def centered_increment(sequence_u: np.ndarray) -> np.ndarray:
    """Return x(t-l)-x(t) for a current-to-past sequence tensor."""
    sequence = np.asarray(sequence_u, dtype=np.float64)
    if sequence.ndim != 3:
        raise ValueError(f"INVALID_SEQUENCE_SHAPE:{sequence.shape}")
    delta = sequence - sequence[:, :1, :]
    delta[:, 0, :] = 0.0
    if not np.all(np.isfinite(delta)):
        raise RuntimeError("CENTERED_INCREMENT_NONFINITE")
    if not np.array_equal(delta[:, 0, :], np.zeros_like(delta[:, 0, :])):
        raise RuntimeError("CENTERED_INCREMENT_LAG0_NOT_ZERO")
    return delta


def support_audit(train_delta: np.ndarray, other_delta: np.ndarray) -> list[dict[str, float | int]]:
    train = np.asarray(train_delta, dtype=np.float64)
    other = np.asarray(other_delta, dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for channel in range(train.shape[2]):
        lower = float(np.min(train[:, :, channel]))
        upper = float(np.max(train[:, :, channel]))
        values = other[:, :, channel]
        outside = (values < lower) | (values > upper)
        rows.append({
            "channel_index": channel,
            "train_lower": lower,
            "train_upper": upper,
            "other_lower": float(np.min(values)),
            "other_upper": float(np.max(values)),
            "extension_ratio": float(np.mean(outside)),
            "rows_with_extension_ratio": float(np.mean(np.any(outside, axis=1))),
        })
    return rows
