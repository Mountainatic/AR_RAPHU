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
from ar_raphu.spectral.penalty_interval import (
    automatic_penalty_interval,
    normalize_penalty_relative_to_gram,
    penalty_boundary_status,
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
    literature_profile_path = (
        ROOT / "configs/public_benchmarks/PB1_LITERATURE_PROFILES.json"
    )
    literature_profile = _read_json(literature_profile_path)
    literature_audit_path = (
        ROOT
        / "results/public_benchmarks/pb1/protocol_audit"
        / "literature_profile_audit.json"
    )
    literature_audit = (
        _read_json(literature_audit_path)
        if literature_audit_path.is_file()
        else {"status": "NOT_YET_RUN", "gates": {}}
    )
    normalization = normalize_penalty_relative_to_gram(
        np.diag([0.0, 1.0, 2.0]), np.eye(3)
    )
    interval = automatic_penalty_interval(
        normalization.normalized, np.eye(3)
    )
    penalty_algorithm_gates = {
        "PENALTY_NORMALIZATION_FROZEN": bool(
            np.isclose(
                np.median(
                    normalization.positive_generalized_eigenvalues
                ),
                1.0,
            )
        ),
        "PENALTY_INTERVAL_ALGORITHM_FROZEN": interval.lower < interval.upper,
        "PENALTY_BOUNDARY_POLICY_FROZEN": (
            penalty_boundary_status(
                selected_index=0, grid_size=7, expansion_count=0
            )
            == "PENALTY_INTERVAL_EXPANSION_REQUIRED"
        ),
    }
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
        if (
            config["literature_profiles"]["companion_code_commit"]
            != literature_profile["companion_code"]["commit"]
        ):
            entries[dataset_id]["status"] = (
                "BLOCKED_BY_LITERATURE_PROFILE_MISMATCH"
            )
            entries[dataset_id]["missing_preregistration"].append(
                "literature_profiles.companion_code_commit_mismatch"
            )
    literature_ready = (
        literature_audit.get("status") == "COMPLETED"
        and literature_audit.get("gates", {}).get(
            "LITERATURE_PROFILE_PINNED"
        )
        is True
    )
    penalty_ready = all(penalty_algorithm_gates.values())
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB1",
        "scope": "DEVELOPMENT_PREFLIGHT_NO_MODEL_FIT_NO_TEST_ACCESS",
        "source_commit": _source_commit(),
        "protocol_freeze_sha256": _sha256(freeze_path),
        "literature_profile_sha256": _sha256(literature_profile_path),
        "literature_profile_audit": {
            "path": str(literature_audit_path.relative_to(ROOT)),
            "status": literature_audit.get("status"),
            "gates": literature_audit.get("gates", {}),
        },
        "penalty_algorithm_gates": penalty_algorithm_gates,
        "datasets": entries,
        "overall_status": (
            "READY_FOR_DEVELOPMENT"
            if literature_ready
            and penalty_ready
            and all(
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
