from __future__ import annotations

import numpy as np
import pandas as pd

from prism_benchmark.cpu_data import BaseAccessor, deterministic_subsample, geometric_intervals, inner_folds
from prism_benchmark.cpu_selection import select_one_se


def test_geometric_intervals_are_contiguous_and_cover_history() -> None:
    intervals = geometric_intervals(64, 8)
    assert intervals[0][0] == 0
    assert intervals[-1][1] == 64
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    assert all(stop > start for start, stop in intervals)


def test_accessor_uses_strict_past_and_block_means() -> None:
    accessor = object.__new__(BaseAccessor)
    accessor.dataset = "demo"
    accessor.entities = {
        "one": (
            np.arange(10, dtype=np.int64),
            {"u": np.arange(10, dtype=np.float64), "y": np.arange(10, dtype=np.float64)},
        )
    }
    samples = pd.DataFrame(
        {
            "entity_id": ["one", "one"],
            "origin": [5, 7],
            "latest_available_target_index": [4, 6],
        }
    )
    np.testing.assert_array_equal(accessor.snapshot(samples, ["u"]).reshape(-1), [4.0, 6.0])
    np.testing.assert_allclose(accessor.block_means(samples, "u", [(0, 2), (2, 5)]), [[3.5, 1.0], [5.5, 3.0]])


def test_accessor_grouped_fast_path_preserves_interleaved_row_order() -> None:
    accessor = object.__new__(BaseAccessor)
    accessor.dataset = "demo"
    accessor.entities = {
        "one": (np.arange(8, dtype=np.int64), {"u": np.arange(8, dtype=np.float64)}),
        "two": (np.arange(8, dtype=np.int64), {"u": 100.0 + np.arange(8, dtype=np.float64)}),
    }
    samples = pd.DataFrame(
        {
            "entity_id": ["two", "one", "two", "one"],
            "origin": [5, 4, 7, 6],
            "latest_available_target_index": [4, 3, 6, 5],
        }
    )
    np.testing.assert_array_equal(
        accessor.snapshot(samples, ["u"]).reshape(-1),
        [104.0, 3.0, 106.0, 5.0],
    )
    np.testing.assert_allclose(
        accessor.block_means(samples, "u", [(0, 2)]).reshape(-1),
        [103.5, 2.5, 105.5, 4.5],
    )


def test_tep_inner_folds_keep_same_run_across_faults() -> None:
    rows = []
    for run in range(1, 6):
        for fault in range(2):
            rows.append(
                {
                    "dataset": "tep",
                    "entity_id": f"Training|fault={fault}|run={run}",
                    "dependency_start": 0,
                    "dependency_stop_exclusive": 1,
                }
            )
    samples = pd.DataFrame(rows)
    folds = inner_folds(samples, count=4)
    for train, validation in folds:
        train_runs = set(samples.iloc[train].entity_id.str.extract(r"run=(\d+)$", expand=False))
        validation_runs = set(samples.iloc[validation].entity_id.str.extract(r"run=(\d+)$", expand=False))
        assert train_runs.isdisjoint(validation_runs)
        for run in validation_runs:
            assert sum(samples.iloc[validation].entity_id.str.endswith(f"run={run}")) == 2


def test_one_se_prefers_declared_simpler_candidate() -> None:
    losses = {1: [1.0, 1.2, 0.8, 1.0], 2: [0.99, 1.19, 0.79, 0.99]}
    selected = select_one_se(losses, lambda value: (value,))
    assert selected.candidate == 1


def test_hash_subsample_is_deterministic_and_bounded() -> None:
    frame = pd.DataFrame({"base_origin_id": [f"id-{index}" for index in range(100)]})
    first = deterministic_subsample(frame, 10)
    second = deterministic_subsample(frame, 10)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 10
