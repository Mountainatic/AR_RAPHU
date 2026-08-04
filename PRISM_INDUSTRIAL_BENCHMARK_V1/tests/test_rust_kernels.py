from __future__ import annotations

import numpy as np
import pytest

from prism_benchmark import rust_kernels


@pytest.mark.skipif(not rust_kernels.available(), reason="optional Rust extension not built")
def test_rust_prefix_kernel_is_bitwise_equal_to_reference() -> None:
    values = np.linspace(-2.0, 3.0, 200, dtype=np.float64)
    prefix = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    count = np.arange(len(values) + 1, dtype=np.int64)
    origins = np.asarray([50, 90, 150, 199], dtype=np.int64)
    intervals = [(0, 3), (3, 11), (11, 37)]
    actual = rust_kernels.block_means_prefix(origins, 0, prefix, count, intervals)
    expected = np.column_stack(
        [
            (prefix[origins - near] - prefix[origins - far]) / (far - near)
            for near, far in intervals
        ]
    )
    np.testing.assert_array_equal(actual, expected)
