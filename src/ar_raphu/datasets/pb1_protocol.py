"""Apply the preregistered PB1 development partitions without opening test data."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .base import DynamicDataset


_PWH_RECORD = re.compile(r"^Est-phase-(\d+)-amp-(\d+)$")
_WHPN_RECORD = re.compile(r"^estimation-(\d+)$")


def load_pb1_protocol_freeze(path: str | Path) -> dict[str, Any]:
    """Read a partial PB1 freeze and reject unsupported schema/status values."""

    with Path(path).open("r", encoding="utf-8") as stream:
        freeze = json.load(stream)
    if freeze.get("suite") != "OPS_UOI_PUBLIC_BENCHMARK_PB1":
        raise ValueError("Not a PB1 protocol freeze.")
    if freeze.get("freeze_status") not in {"PARTIALLY_FROZEN", "FROZEN"}:
        raise ValueError("PB1 protocol is not frozen enough for development.")
    return freeze


def verify_whpn_freeze_audit(
    freeze: dict[str, Any], audit: dict[str, Any]
) -> None:
    """Bind the two validation IDs and hashes to the completed pure-data audit."""

    specification = freeze["datasets"]["whpn"]
    if specification.get("status") != "FROZEN":
        raise ValueError("WHPN is not frozen.")
    if audit.get("dataset") != "whpn" or audit.get("status") != "COMPLETED":
        raise ValueError("WHPN pure-data audit is not completed.")
    for field in ("validation_record_ids", "validation_record_sha256"):
        if specification.get(field) != audit.get(field):
            raise ValueError(f"WHPN freeze/audit mismatch for {field}.")
    consistency = audit.get("protocol_consistency", {})
    if not consistency.get("same_mat_file"):
        raise ValueError("WHPN validation records are not from the audited MAT file.")
    if not consistency.get("sample_count_consistent"):
        raise ValueError("WHPN realization lengths are inconsistent.")
    if not consistency.get("sampling_rate_consistent"):
        raise ValueError("WHPN realization sampling rates are inconsistent.")


def _record_name(sequence_id: object) -> str:
    value = str(sequence_id)
    if ":" not in value:
        raise ValueError(f"Sequence ID lacks stable record prefix: {value!r}.")
    return value.split(":", 1)[1]


def _reject_exposed_test_data(dataset: DynamicDataset) -> None:
    if bool(dataset.metadata.get("test_records_exposed", False)):
        raise PermissionError("PB1 development partitioner refuses exposed test data.")
    if np.any(dataset.split == "test"):
        raise PermissionError("PB1 development partitioner refuses test rows.")


def _apply_record_partition(
    dataset: DynamicDataset,
    *,
    expected_records: set[str],
    validation_records: set[str],
    dataset_id: str,
) -> DynamicDataset:
    observed_records = {
        _record_name(sequence) for sequence in np.unique(dataset.sequence_id)
    }
    if observed_records != expected_records:
        missing = sorted(expected_records - observed_records)
        unexpected = sorted(observed_records - expected_records)
        raise ValueError(
            f"{dataset_id} estimation record identity mismatch; "
            f"missing={missing}, unexpected={unexpected}."
        )
    split = np.full(dataset.n_time, "train", dtype=object)
    for sequence in np.unique(dataset.sequence_id):
        indices = np.flatnonzero(dataset.sequence_id == sequence)
        record = _record_name(sequence)
        if record in validation_records:
            split[indices] = "validation"
        if len(set(map(str, split[indices]))) != 1:
            raise AssertionError(f"{record}: record was not assigned atomically.")
    metadata = {
        **dataset.metadata,
        "validation_status": "FROZEN",
        "pb1_protocol_applied": True,
        "pb1_dataset_status": "FROZEN",
        "official_test_locked": True,
    }
    return replace(dataset, split=split, metadata=metadata)


def apply_pb1_development_partition(
    dataset: DynamicDataset,
    freeze: dict[str, Any],
    *,
    whpn_audit: dict[str, Any] | None = None,
) -> DynamicDataset:
    """Create only the frozen development train/validation partition.

    The function accepts estimation-only datasets. Official confirmation data
    must not have been loaded, even if the caller intends to exclude it later.
    """

    _reject_exposed_test_data(dataset)
    dataset_id = str(dataset.metadata.get("dataset_id", ""))
    specification = freeze.get("datasets", {}).get(dataset_id)
    if not specification or specification.get("status") != "FROZEN":
        raise PermissionError(f"{dataset_id!r} is not unlocked by PB1 freeze.")

    if dataset_id == "pwh":
        expected = {
            f"Est-phase-{phase}-amp-{amplitude}"
            for phase in range(20)
            for amplitude in range(5)
        }
        validation = {
            f"Est-phase-{phase}-amp-{amplitude}"
            for phase in range(16, 20)
            for amplitude in range(5)
        }
        if specification.get("periods_per_record") != 2:
            raise ValueError("PWH freeze must preserve both periods.")
        if specification.get("periods_are_atomic") is not True:
            raise ValueError("PWH freeze does not declare atomic periods.")
        if specification.get("validation_record_count") != len(validation):
            raise ValueError("PWH validation count differs from frozen partition.")
        return _apply_record_partition(
            dataset,
            expected_records=expected,
            validation_records=validation,
            dataset_id=dataset_id,
        )

    if dataset_id == "whpn":
        if whpn_audit is None:
            raise ValueError("WHPN requires its completed pure-data audit.")
        verify_whpn_freeze_audit(freeze, whpn_audit)
        expected = {f"estimation-{index}" for index in range(10)}
        validation = {"estimation-8", "estimation-9"}
        return _apply_record_partition(
            dataset,
            expected_records=expected,
            validation_records=validation,
            dataset_id=dataset_id,
        )

    raise PermissionError(f"{dataset_id!r} has no implemented PB1 partitioner.")


def apply_pb1_repair_v2_partition(
    dataset: DynamicDataset,
    config: dict[str, Any],
) -> DynamicDataset:
    """Apply schema-7 Tanks/Silverbox literature development partitions."""

    _reject_exposed_test_data(dataset)
    if config.get("schema_version") != 7:
        raise ValueError("PB1 Repair V2 requires schema_version 7.")
    dataset_id = str(dataset.metadata.get("dataset_id", ""))
    configured_id = str(config.get("dataset", {}).get("id", ""))
    if dataset_id != configured_id:
        raise ValueError("PB1 Repair V2 dataset/config identity mismatch.")
    specification = config["dataset"]["development_split"]
    if specification.get("status") != "FROZEN_LITERATURE_SPLIT":
        raise PermissionError("Literature development split is not frozen.")
    sequences = np.unique(dataset.sequence_id)
    if len(sequences) != 1:
        raise ValueError("Tanks/Silverbox development expects one estimation record.")
    indices = np.flatnonzero(dataset.sequence_id == sequences[0])
    split = np.full(dataset.n_time, "train", dtype=object)
    if dataset_id == "cascaded_tanks":
        if specification.get("source") != "Champneys2024_associated_code":
            raise ValueError("Tanks split source is not the frozen literature code.")
        if specification.get("train_rows") != [0, 700]:
            raise ValueError("Tanks train split must be [0,700).")
        if specification.get("validation_rows") != [700, "end"]:
            raise ValueError("Tanks validation split must be [700,end).")
        if len(indices) <= 700:
            raise ValueError("Tanks estimation record is shorter than split boundary.")
        split[indices[700:]] = "validation"
        split_label = "CHAMPNEYS_ASSOCIATED_CODE_700_REST"
    elif dataset_id == "silverbox":
        if specification.get("source") != "Champneys2024_associated_code":
            raise ValueError("Silverbox split source is not frozen literature code.")
        if specification.get("train_fraction") != [0.0, 0.5]:
            raise ValueError("Silverbox train fraction must be [0,0.5).")
        if specification.get("validation_fraction") != [0.5, 1.0]:
            raise ValueError("Silverbox validation fraction must be [0.5,1].")
        boundary = len(indices) // 2
        if boundary <= 0 or boundary >= len(indices):
            raise ValueError("Silverbox estimation record cannot be split in half.")
        split[indices[boundary:]] = "validation"
        split_label = "CHAMPNEYS_ASSOCIATED_CODE_HALF_HALF"
    else:
        raise PermissionError(f"{dataset_id!r} is not a Repair V2 literature split.")
    metadata = {
        **dataset.metadata,
        "validation_status": "FROZEN",
        "pb1_protocol_applied": True,
        "pb1_repair_v2": True,
        "pb1_dataset_status": "FROZEN_LITERATURE_SPLIT",
        "development_split_label": split_label,
        "official_test_locked": True,
        "official_test_access_count": 0,
    }
    return replace(dataset, split=split, metadata=metadata)
