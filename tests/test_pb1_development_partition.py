from __future__ import annotations

import copy

import numpy as np
import pytest

from ar_raphu.datasets.base import DynamicDataset
from ar_raphu.datasets.pb1_protocol import (
    apply_pb1_development_partition,
    verify_whpn_freeze_audit,
)


def _dataset(dataset_id: str, records: list[str], *, rows: int = 4):
    n = len(records) * rows
    sequence_id = np.concatenate(
        [
            np.full(rows, f"{index:04d}:{record}", dtype=object)
            for index, record in enumerate(records)
        ]
    )
    x = np.arange(n, dtype=np.float64)[:, None]
    y = (1000.0 + np.arange(n, dtype=np.float64))[:, None]
    return DynamicDataset(
        x=x,
        y=y,
        timestamps=np.arange(n, dtype=np.float64),
        sequence_id=sequence_id,
        split=np.full(n, "train", dtype=object),
        label_mask=np.ones_like(y, dtype=bool),
        quality_mask=np.ones((n, 2), dtype=bool),
        feature_names=("input",),
        target_names=("output",),
        metadata={
            "dataset_id": dataset_id,
            "test_records_exposed": False,
            "official_test_locked": True,
        },
    )


def _freeze() -> dict:
    validation_ids = [
        "WH_EstimationExample.mat::dataMeas[realization_zero_based=8]",
        "WH_EstimationExample.mat::dataMeas[realization_zero_based=9]",
    ]
    hashes = ["hash-8", "hash-9"]
    return {
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "freeze_status": "PARTIALLY_FROZEN",
        "datasets": {
            "pwh": {
                "status": "FROZEN",
                "periods_per_record": 2,
                "periods_are_atomic": True,
                "validation_record_count": 20,
            },
            "whpn": {
                "status": "FROZEN",
                "validation_record_ids": validation_ids,
                "validation_record_sha256": hashes,
            },
            "cascaded_tanks": {"status": "PENDING_SPLIT_ADEQUACY_AUDIT"},
        },
    }


def _whpn_audit() -> dict:
    return {
        "dataset": "whpn",
        "status": "COMPLETED",
        "validation_record_ids": [
            "WH_EstimationExample.mat::dataMeas[realization_zero_based=8]",
            "WH_EstimationExample.mat::dataMeas[realization_zero_based=9]",
        ],
        "validation_record_sha256": ["hash-8", "hash-9"],
        "protocol_consistency": {
            "same_mat_file": True,
            "sample_count_consistent": True,
            "sampling_rate_consistent": True,
        },
    }


def test_pwh_partition_keeps_each_phase_amplitude_record_atomic() -> None:
    records = [
        f"Est-phase-{phase}-amp-{amplitude}"
        for phase in range(20)
        for amplitude in range(5)
    ]
    partitioned = apply_pb1_development_partition(
        _dataset("pwh", records), _freeze()
    )
    assert np.count_nonzero(partitioned.split == "train") == 80 * 4
    assert np.count_nonzero(partitioned.split == "validation") == 20 * 4
    for sequence in np.unique(partitioned.sequence_id):
        indices = np.flatnonzero(partitioned.sequence_id == sequence)
        assert len(np.unique(partitioned.split[indices])) == 1
    assert partitioned.metadata["official_test_locked"] is True


def test_whpn_partition_requires_matching_ids_and_hashes() -> None:
    dataset = _dataset("whpn", [f"estimation-{i}" for i in range(10)])
    partitioned = apply_pb1_development_partition(
        dataset, _freeze(), whpn_audit=_whpn_audit()
    )
    assert np.count_nonzero(partitioned.split == "train") == 8 * 4
    assert np.count_nonzero(partitioned.split == "validation") == 2 * 4
    bad_audit = copy.deepcopy(_whpn_audit())
    bad_audit["validation_record_sha256"][1] = "changed"
    with pytest.raises(ValueError, match="freeze/audit mismatch"):
        verify_whpn_freeze_audit(_freeze(), bad_audit)


def test_partitioner_refuses_exposed_test_and_pending_datasets() -> None:
    pwh = _dataset(
        "pwh",
        [
            f"Est-phase-{phase}-amp-{amplitude}"
            for phase in range(20)
            for amplitude in range(5)
        ],
    )
    pwh.metadata["test_records_exposed"] = True
    with pytest.raises(PermissionError, match="exposed test"):
        apply_pb1_development_partition(pwh, _freeze())
    tanks = _dataset("cascaded_tanks", ["estimation"])
    with pytest.raises(PermissionError, match="not unlocked"):
        apply_pb1_development_partition(tanks, _freeze())
