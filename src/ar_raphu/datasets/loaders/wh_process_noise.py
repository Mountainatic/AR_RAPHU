"""Source inspection utilities for the WH process-noise archive.

The official ``nonlinear-benchmarks`` package does not expose this benchmark.
The loader remains audit-only until the archive's own documentation has been
used to freeze exact estimation/test record mapping.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from ..base import DynamicDataset
from .common import array_record, records_to_dynamic_dataset


def inspect_whpn_archive(path: str | Path) -> dict[str, Any]:
    archive = Path(path)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as handle:
        members = sorted(
            {
                item.filename
                for item in handle.infolist()
                if not item.is_dir()
            }
        )
    return {
        "archive": str(archive),
        "member_count": len(members),
        "members": members,
        "loader_status": "BLOCKED_PENDING_OFFICIAL_RECORD_MAPPING",
    }


def _load_records(path: Path, *, split: str, prefix: str) -> list[tuple[Any, str, str]]:
    source = loadmat(path, squeeze_me=True, struct_as_record=False)["dataMeas"]
    u = np.asarray(source.u, dtype=np.float64)
    y = np.asarray(source.y, dtype=np.float64)
    if u.shape != y.shape or u.ndim != 2:
        raise ValueError(f"{path.name}: expected matching [time, record] u/y.")
    fs = float(np.asarray(source.fs).squeeze())
    return [
        (
            array_record(
                u=u[:, index],
                y=y[:, index],
                name=f"{prefix}-{index}",
                sampling_time=1.0 / fs,
            ),
            split,
            f"{prefix}-{index}",
        )
        for index in range(u.shape[1])
    ]


def load_whpn(
    raw_root: str | Path, *, include_test: bool = False
) -> DynamicDataset:
    """Load the documented example estimation and noiseless official test sets.

    The archive README reports a possible one-sample inter-channel shift. Raw
    alignment is preserved here; no compensating shift is applied before a
    protocol choice is explicitly frozen.
    """

    root = Path(raw_root) / "WHPN" / "WienerHammersteinFiles"
    records = _load_records(
        root / "WH_EstimationExample.mat",
        split="train",
        prefix="estimation",
    )
    if include_test:
        records.extend(
            _load_records(
                root / "WH_TestDataset.mat",
                split="test",
                prefix="official_noiseless_test",
            )
        )
    return records_to_dynamic_dataset(
        records,
        dataset_id="whpn",
        source_metadata={
            "doi": "10.4121/12952124.v2",
            "process_noise_preserved": True,
            "outlier_cleaning": "NONE",
            "channel_alignment": "RAW_UNSHIFTED",
            "channel_alignment_warning": (
                "Archive README reports a possible one-sample shift."
            ),
            "test_records_exposed": include_test,
        },
    )
