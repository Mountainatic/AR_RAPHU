"""Certify one-worker/two-worker equality before CZ outer parallelism."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DECISION_KEYS = (
    "status",
    "active",
    "selected_family",
    "selected_profile",
    "selected_candidate",
    "selection_status",
    "routing_status",
    "final_selected_candidate",
    "selected_k_representation",
    "selected_contract_candidate_id",
)


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


def parquet_content_hash(path: Path) -> str:
    frame = pd.read_parquet(path)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": list(frame.columns),
                "dtypes": [str(value) for value in frame.dtypes],
                "rows": len(frame),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def _decisions(root: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "results").rglob("RESULT.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        values[path.relative_to(root / "results").as_posix()] = {
            key: payload.get(key) for key in DECISION_KEYS if key in payload
        }
    return values


def _parquets(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root / "results").as_posix(): parquet_content_hash(path)
        for path in sorted((root / "results").rglob("*.parquet"))
    }


def verify(reference: Path, candidate: Path, parent_run: Path) -> dict[str, Any]:
    for root, workers in ((reference, 1), (candidate, 2)):
        status_path = root / "AUTHORITY_REFIT_STATUS.json"
        if not status_path.is_file():
            raise RuntimeError(f"STOP_WORKER_EQUIVALENCE_STATUS_MISSING:{root}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "PASS" or int(status.get("inner_workers", -1)) != workers:
            raise RuntimeError(f"STOP_WORKER_EQUIVALENCE_STATUS_INVALID:{root}")
    reference_decisions = _decisions(reference)
    candidate_decisions = _decisions(candidate)
    reference_parquets = _parquets(reference)
    candidate_parquets = _parquets(candidate)
    decision_equal = reference_decisions == candidate_decisions
    prediction_equal = reference_parquets == candidate_parquets
    result = {
        "status": "PASS" if decision_equal and prediction_equal else "FAILED",
        "checked_utc": _utc(),
        "reference_inner_workers": 1,
        "candidate_inner_workers": 2,
        "decision_files": len(reference_decisions),
        "parquet_files": len(reference_parquets),
        "decision_equality": decision_equal,
        "prediction_content_hash_equality": prediction_equal,
        "missing_or_changed_decisions": sorted(
            key
            for key in set(reference_decisions) | set(candidate_decisions)
            if reference_decisions.get(key) != candidate_decisions.get(key)
        ),
        "missing_or_changed_parquets": sorted(
            key
            for key in set(reference_parquets) | set(candidate_parquets)
            if reference_parquets.get(key) != candidate_parquets.get(key)
        ),
    }
    destination = parent_run / "WORKER_EQUIVALENCE.json"
    _write_json(destination, result)
    if result["status"] != "PASS":
        raise RuntimeError("STOP_CZ_WORKER_EQUIVALENCE_FAILED")
    gate_path = parent_run / "STREAMING_STORAGE_GATE.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["two_way_parallelism"] = "PASS_WORKER_EQUIVALENCE"
    gate["maximum_concurrent_outer_units"] = 2
    gate["worker_equivalence_path"] = str(destination)
    _write_json(gate_path, gate)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-unit", type=Path, required=True)
    parser.add_argument("--candidate-unit", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.reference_unit.resolve(),
        args.candidate_unit.resolve(),
        args.parent_run_root.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
