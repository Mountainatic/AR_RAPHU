from __future__ import annotations

import os

import numpy as np

try:
    from _prism_rust import block_means_prefix as _block_means_prefix
except ImportError:
    _block_means_prefix = None


def available() -> bool:
    return _block_means_prefix is not None


def enabled() -> bool:
    return available() and os.environ.get("PRISM_RUST_KERNELS", "1") != "0"


def block_means_prefix(
    origins: np.ndarray,
    dense_min: int,
    value_prefix: np.ndarray,
    count_prefix: np.ndarray,
    intervals: list[tuple[int, int]],
) -> np.ndarray | None:
    if not enabled():
        return None
    return np.asarray(
        _block_means_prefix(
            np.ascontiguousarray(origins, dtype=np.int64),
            int(dense_min),
            np.ascontiguousarray(value_prefix, dtype=np.float64),
            np.ascontiguousarray(count_prefix, dtype=np.int64),
            np.ascontiguousarray(intervals, dtype=np.int64),
        ),
        dtype=np.float64,
    )
