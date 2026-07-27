from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ar_raphu.datasets.registry import get_public_dataset


RAW_ROOT = os.environ.get("AR_RAPHU_PUBLIC_RAW_ROOT")
pytestmark = pytest.mark.skipif(
    RAW_ROOT is None,
    reason="Set AR_RAPHU_PUBLIC_RAW_ROOT for official-source integration tests.",
)


@pytest.mark.parametrize(
    ("dataset_id", "train_count", "train_sequences"),
    [
        ("pwh", 3_276_800, 100),
        ("whpn", 81_920, 10),
        ("cascaded_tanks", 1_024, 1),
        ("silverbox", 65_062, 1),
    ],
)
def test_default_loader_does_not_expose_official_test(
    dataset_id: str, train_count: int, train_sequences: int
) -> None:
    dataset = get_public_dataset(dataset_id).load(Path(RAW_ROOT))
    assert dataset.n_time == train_count
    assert len(np.unique(dataset.sequence_id)) == train_sequences
    assert np.all(dataset.split == "train")
    assert dataset.metadata["test_records_exposed"] is False


@pytest.mark.parametrize(
    ("dataset_id", "test_count", "test_sequences"),
    [
        ("pwh", 196_608, 6),
        ("whpn", 32_768, 2),
        ("cascaded_tanks", 1_024, 1),
        ("silverbox", 94_163, 3),
    ],
)
def test_audit_mode_preserves_official_test_records(
    dataset_id: str, test_count: int, test_sequences: int
) -> None:
    dataset = get_public_dataset(dataset_id).load(
        Path(RAW_ROOT), include_test=True
    )
    assert int(np.count_nonzero(dataset.split == "test")) == test_count
    test_ids = np.unique(dataset.sequence_id[dataset.split == "test"])
    assert len(test_ids) == test_sequences
    assert dataset.metadata["test_records_exposed"] is True
    assert np.all(dataset.quality_mask)
    assert np.all(dataset.label_mask)


def test_whpn_preserves_raw_alignment_and_process_noise() -> None:
    dataset = get_public_dataset("whpn").load(Path(RAW_ROOT))
    assert dataset.metadata["process_noise_preserved"] is True
    assert dataset.metadata["outlier_cleaning"] == "NONE"
    assert dataset.metadata["channel_alignment"] == "RAW_UNSHIFTED"
