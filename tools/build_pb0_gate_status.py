#!/usr/bin/env python3
"""Combine immutable lineage and source audits into the PB0 gate table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("pwh", "whpn", "cascaded_tanks", "silverbox")
GATES = (
    "SOURCE_VERIFIED",
    "HASH_VERIFIED",
    "LICENSE_RECORDED",
    "OFFICIAL_SPLIT_VERIFIED",
    "TIME_ORDER_VERIFIED",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lineage-root",
        type=Path,
        default=Path("data_manifests/public_benchmarks"),
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("results/public_benchmarks/pb0"),
    )
    args = parser.parse_args()
    rows: dict[str, dict[str, object]] = {}
    for dataset_id in DATASETS:
        lineage_path = args.lineage_root / f"{dataset_id}.json"
        audit_path = args.audit_root / f"{dataset_id}_source_audit.json"
        lineage = _read(lineage_path) if lineage_path.is_file() else {}
        audit = _read(audit_path) if audit_path.is_file() else {}
        lineage_status = lineage.get("status", {})
        audit_status = audit.get("status", {})
        gates = {
            name: bool(audit_status.get(name, lineage_status.get(name, False)))
            for name in GATES
        }
        if all(gates.values()):
            status = "COMPLETED"
        elif not lineage_path.is_file() or not audit_path.is_file():
            status = "NOT_YET_RUN"
        else:
            status = "BLOCKED_BY_MISSING_METADATA"
        rows[dataset_id] = {
            "status": status,
            "gates": gates,
            "lineage": str(lineage_path),
            "audit": str(audit_path),
            "blockers": [name for name, passed in gates.items() if not passed],
        }
    payload = {
        "schema_version": 6,
        "suite": "OPS_UOI_PUBLIC_BENCHMARK_PB0",
        "private_cz_access": "FORBIDDEN",
        "datasets": rows,
        "overall_status": (
            "COMPLETED"
            if all(row["status"] == "COMPLETED" for row in rows.values())
            else "BLOCKED_BY_MISSING_METADATA"
        ),
    }
    output = args.audit_root / "PB0_GATE_STATUS.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
