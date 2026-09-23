"""Certify four-way CPU use for bounded CZ authority tiny pilots only."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ALLOWED_COMPLETION = {"PASS", "COMPLETED_WITH_RETAINED_FAILURES"}
MAXIMUM_UNIT_GIB = 0.25
PRIVATE_REQUIRED_GIB = 9.0
SCRATCH_REQUIRED_GIB = 8.0


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def certify(
    parent_run: Path,
    completed_units: Iterable[Path],
    scratch_root: Path,
) -> dict[str, Any]:
    gate_path = parent_run / "STREAMING_STORAGE_GATE.json"
    worker_path = parent_run / "WORKER_EQUIVALENCE.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or worker.get("status") != "PASS":
        raise RuntimeError("STOP_TINY_PILOT_PREREQUISITE_NOT_PASS")
    units = [path.resolve() for path in completed_units]
    if len(units) < 4 or len(set(units)) != len(units):
        raise RuntimeError("STOP_TINY_PILOT_RESOURCE_SAMPLE_INSUFFICIENT")
    evidence = []
    for unit in units:
        status = json.loads(
            (unit / "AUTHORITY_REFIT_STATUS.json").read_text(encoding="utf-8")
        )
        if status.get("status") not in ALLOWED_COMPLETION:
            raise RuntimeError(f"STOP_TINY_PILOT_UNIT_NOT_COMPLETE:{unit}")
        if status.get("role") != "P1_TINY_PILOT_NON_SELECTION_AUTHORITY":
            raise RuntimeError(f"STOP_TINY_PILOT_ROLE_MISMATCH:{unit}")
        if status.get("formal_target_or_ood_accessed") is not False:
            raise RuntimeError(f"STOP_TINY_PILOT_EVIDENCE_BOUNDARY:{unit}")
        size_gib = _tree_bytes(unit) / (1024**3)
        if size_gib > MAXIMUM_UNIT_GIB:
            raise RuntimeError(f"STOP_TINY_PILOT_UNIT_TOO_LARGE:{unit}")
        evidence.append(
            {
                "unit": str(unit),
                "status": status["status"],
                "inner_workers": int(status.get("inner_workers", -1)),
                "persistent_size_gib": size_gib,
            }
        )
    private_free = shutil.disk_usage(parent_run).free / (1024**3)
    scratch_free = shutil.disk_usage(scratch_root).free / (1024**3)
    passed = private_free >= PRIVATE_REQUIRED_GIB and scratch_free >= SCRATCH_REQUIRED_GIB
    result = {
        "status": "PASS" if passed else "BLOCKED",
        "scope": "P1_TINY_PILOT_NON_SELECTION_AUTHORITY_ONLY",
        "checked_utc": _utc(),
        "completed_unit_evidence": evidence,
        "maximum_observed_persistent_unit_gib": max(
            item["persistent_size_gib"] for item in evidence
        ),
        "maximum_allowed_persistent_unit_gib": MAXIMUM_UNIT_GIB,
        "private_free_gib": private_free,
        "private_required_gib": PRIVATE_REQUIRED_GIB,
        "scratch_free_gib": scratch_free,
        "scratch_required_gib": SCRATCH_REQUIRED_GIB,
        "maximum_concurrent_outer_units": 4,
        "inner_workers_per_unit": 2,
        "formal_target_or_ood_access_forbidden": True,
        "statistical_claims_allowed": False,
    }
    destination = parent_run / "TINY_PILOT_PARALLELISM_GATE.json"
    _write_json(destination, result)
    if not passed:
        raise RuntimeError("STOP_TINY_PILOT_RESOURCE_GATE_BLOCKED")
    gate["tiny_pilot_parallelism"] = "PASS_RESOURCE_CERTIFICATE"
    gate["tiny_pilot_maximum_concurrent_outer_units"] = 4
    gate["tiny_pilot_inner_workers_per_unit"] = 2
    gate["tiny_pilot_parallelism_gate_path"] = str(destination)
    _write_json(gate_path, gate)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--completed-unit", type=Path, action="append", required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/dev/shm"))
    args = parser.parse_args()
    result = certify(
        args.parent_run_root.resolve(),
        args.completed_unit,
        args.scratch_root.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
