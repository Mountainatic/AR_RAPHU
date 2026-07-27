#!/usr/bin/env python3
"""Validate frozen PB1 development inputs without fitting any model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ar_raphu.datasets.loaders import load_pwh, load_whpn
from ar_raphu.datasets.pb1_preflight import (
    development_preregistration_gaps,
    development_preflight_status,
)
from ar_raphu.datasets.pb1_protocol import (
    apply_pb1_development_partition,
    load_pb1_protocol_freeze,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _split_summary(dataset) -> dict[str, Any]:
    by_record = {"train": 0, "validation": 0, "test": 0}
    for sequence in np.unique(dataset.sequence_id):
        indices = np.flatnonzero(dataset.sequence_id == sequence)
        values = np.unique(dataset.split[indices])
        if len(values) != 1:
            raise AssertionError(f"{sequence}: record crosses PB1 splits.")
        split = str(values[0])
        if split in by_record:
            by_record[split] += 1
    return {
        "time_rows": {
            split: int(np.count_nonzero(dataset.split == split))
            for split in by_record
        },
        "whole_records": by_record,
        "official_test_rows_loaded": int(
            np.count_nonzero(dataset.split == "test")
        ),
        "official_test_locked": bool(
            dataset.metadata.get("official_test_locked", False)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/OPS_UOI_WORKSPACE/data/raw"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/public_benchmarks/pb1/PB1_DEVELOPMENT_PREFLIGHT.json",
    )
    args = parser.parse_args()

    freeze_path = ROOT / "configs/public_benchmarks/PB1_PROTOCOL_FREEZE.json"
    freeze = load_pb1_protocol_freeze(freeze_path)
    audit_path = (
        ROOT
        / "results/public_benchmarks/pb1/protocol_audit"
        / "whpn_realization_audit.json"
    )
    whpn_audit = _read_json(audit_path)
    entries: dict[str, Any] = {}
    for dataset_id, loader, audit in (
        ("pwh", load_pwh, None),
        ("whpn", load_whpn, whpn_audit),
    ):
        config_path = (
            ROOT / f"configs/public_benchmarks/pb1_{dataset_id}.yaml"
        )
        config = _read_json(config_path)
        raw = loader(args.raw_root, include_test=False)
        partitioned = apply_pb1_development_partition(
            raw, freeze, whpn_audit=audit
        )
        entries[dataset_id] = {
            "status": development_preflight_status(config),
            "missing_preregistration": development_preregistration_gaps(config),
            "split": _split_summary(partitioned),
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256(config_path),
        }
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "scope": "DEVELOPMENT_PREFLIGHT_NO_MODEL_FIT_NO_TEST_ACCESS",
        "source_commit": _source_commit(),
        "protocol_freeze_sha256": _sha256(freeze_path),
        "datasets": entries,
        "overall_status": (
            "READY_FOR_DEVELOPMENT"
            if all(
                value["status"] == "READY_FOR_DEVELOPMENT"
                for value in entries.values()
            )
            else "BLOCKED_BY_MISSING_PREREGISTRATION"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(payload["overall_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
