#!/usr/bin/env python3
"""Audit PB1 source semantics without fitting or evaluating any model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ar_raphu.datasets.loaders import (
    inspect_whpn_archive,
    load_cascaded_tanks,
    load_pwh,
    load_silverbox,
    load_whpn,
)


LOADERS = {
    "pwh": load_pwh,
    "cascaded_tanks": load_cascaded_tanks,
    "silverbox": load_silverbox,
    "whpn": load_whpn,
}


def _dataset_audit(dataset_id: str, raw_root: Path) -> dict[str, object]:
    dataset = LOADERS[dataset_id](raw_root, include_test=True)
    sequence_values = np.unique(dataset.sequence_id)
    records: list[dict[str, object]] = []
    for sequence in sequence_values:
        indices = np.flatnonzero(dataset.sequence_id == sequence)
        split_values = np.unique(dataset.split[indices])
        if len(split_values) != 1:
            raise AssertionError(f"{sequence}: official record crosses splits.")
        if not np.all(np.diff(indices) == 1):
            raise AssertionError(f"{sequence}: non-contiguous source record.")
        if dataset.timestamps is not None:
            local_time = dataset.timestamps[indices]
            if not np.all(np.diff(local_time) > 0):
                raise AssertionError(f"{sequence}: time is not strictly ordered.")
        records.append(
            {
                "sequence_id": str(sequence),
                "split": str(split_values[0]),
                "n_time": int(len(indices)),
                "start_row": int(indices[0]),
                "stop_row_exclusive": int(indices[-1] + 1),
            }
        )
    counts = {
        name: int(np.count_nonzero(dataset.split == name))
        for name in ("warmup", "train", "validation", "test")
    }
    if counts["train"] == 0 or counts["test"] == 0:
        raise AssertionError("Official estimation/test records were not preserved.")
    if counts["validation"] != 0:
        raise AssertionError("Source audit must not invent a validation split.")
    return {
        "schema_version": 6,
        "dataset_id": dataset_id,
        "audit_scope": "SOURCE_AND_SPLIT_ONLY_NO_MODEL_EVALUATION",
        "shape": {
            "n_time": dataset.n_time,
            "n_features": dataset.n_features,
            "n_targets": dataset.n_targets,
            "n_sequences": int(len(sequence_values)),
        },
        "split_counts": counts,
        "records": records,
        "validation_status": "NOT_YET_DEFINED",
        "official_test_status": "LOCKED_NOT_EVALUATED",
        "quality": {
            "invalid_x_or_y_cells": int(np.count_nonzero(~dataset.quality_mask)),
            "unobserved_labels": int(np.count_nonzero(~dataset.label_mask)),
        },
        "status": {
            "OFFICIAL_SPLIT_VERIFIED": True,
            "TIME_ORDER_VERIFIED": True,
        },
        "metadata": dataset.metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/OPS_UOI_WORKSPACE/data/raw"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/public_benchmarks/pb0"),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(LOADERS),
        required=True,
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset_id in args.dataset:
        payload = _dataset_audit(dataset_id, args.raw_root)
        if dataset_id == "whpn":
            payload["archive_audit"] = inspect_whpn_archive(
                args.raw_root / "WHPN" / "WienerHammersteinFiles.zip"
            )
        output = args.output_root / f"{dataset_id}_source_audit.json"
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"{dataset_id}: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
