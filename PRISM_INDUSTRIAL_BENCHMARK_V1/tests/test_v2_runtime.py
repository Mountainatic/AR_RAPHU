from __future__ import annotations

import numpy as np
import pandas as pd

from prism_benchmark.cpu_data import BaseAccessor
from prism_benchmark.v2_runtime import resolve_worker_count


def test_worker_budget_never_exceeds_requested_or_tasks() -> None:
    assert resolve_worker_count(31, 3, per_worker_gib=1.0) in {1, 2, 3}


def test_block_prefix_cache_is_numerically_invisible(tmp_path) -> None:
    root = tmp_path / "base_data" / "toy"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "entity_id": ["a"] * 8,
            "row_in_entity": np.arange(8),
            "x": np.arange(8, dtype=np.float64),
        }
    ).to_parquet(root / "train.parquet", index=False)
    accessor = BaseAccessor(tmp_path, "toy", "train", ["x"])
    samples = pd.DataFrame({"entity_id": ["a", "a"], "origin": [4, 8]})
    expected = np.asarray([[2.5, 0.5], [6.5, 4.5]])
    intervals = [(0, 2), (2, 4)]
    np.testing.assert_array_equal(accessor.block_means(samples, "x", intervals), expected)
    accessor.warm_prefixes(["x"])
    np.testing.assert_array_equal(accessor.block_means(samples, "x", intervals), expected)
