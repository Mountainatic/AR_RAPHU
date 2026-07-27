from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ar_raphu.datasets.loaders.common import records_to_dynamic_dataset


def _record(name: str, n: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        u=np.arange(n, dtype=np.float64),
        y=np.arange(n, dtype=np.float64) + 10.0,
        sampling_time=0.25,
    )


def test_official_records_remain_independent_and_test_is_locked() -> None:
    dataset = records_to_dynamic_dataset(
        [
            (_record("estimation", 5), "train", "estimation"),
            (_record("confirmation", 3), "test", "confirmation"),
        ],
        dataset_id="fixture",
        source_metadata={"doi": "fixture"},
    )
    assert dataset.n_time == 8
    assert len(np.unique(dataset.sequence_id)) == 2
    assert dataset.split.tolist() == ["train"] * 5 + ["test"] * 3
    assert dataset.metadata["validation_status"] == "NOT_YET_DEFINED"
    assert dataset.metadata["official_test_locked"] is True


def test_adapter_never_uses_current_or_future_target_as_input() -> None:
    dataset = records_to_dynamic_dataset(
        [(_record("estimation", 9), "train", "estimation")],
        dataset_id="fixture",
        source_metadata={},
    )
    assert dataset.x.shape == (9, 1)
    assert not np.shares_memory(dataset.x, dataset.y)
