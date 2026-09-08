"""Build and audit the frozen task H/W registry used by TIM E3--E6.

This module reads committed protocol artifacts only.  It does not inspect target
values, model predictions, or any outer-test metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "TEP_H0": ("tep", "TEP_G_NOWCAST_H0__H0__W1", 0, 1, 180.0),
    "DEB_C4": ("debutanizer", "DEB_C4__H5__W1", 5, 1, 360.0),
    "SRU_H2S": ("sru", "SRU_H2S_REP_H1__H1__W1", 1, 1, 60.0),
    "SRU_SO2": ("sru", "SRU_SO2_REP_H1__H1__W1", 1, 1, 60.0),
    "PMSM_PM5": ("pmsm", "PMSM_PM5__H600__W60", 600, 60, 0.5),
    "METRO_P60": ("metropt", "METRO_P60__H6__W1", 6, 1, 10.0),
    "METRO_OIL20": ("metropt", "METRO_OIL20__H120__W12", 120, 12, 10.0),
    "CZ_H4": ("cz_czochralski", "CZ_DIAM_RAW2S_CURRENT_L256_H4", 4, 1, 2.0),
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_registry(project: Path, output: Path) -> dict[str, Any]:
    config = project / "configs" / "stagewise_ablation_hybrid_hw_20260904.json"
    active = project / "configs" / "active3_neural3_parallel_cache_20260826.json"
    legacy = project / "artifacts" / "c1" / "TASK_REGISTRY.json"
    for path in (config, active, legacy):
        if not path.is_file():
            raise FileNotFoundError(path)

    stagewise = _read(config)
    if not str(stagewise.get("status", "")).startswith("FROZEN_"):
        raise RuntimeError("stagewise H/W source is not frozen")
    by_head = {row["head_id"]: row for row in stagewise["task_matrix"]}
    active_value = _read(active)
    legacy_value = _read(legacy)

    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for task, (dataset, head, h_steps, w_steps, cadence) in EXPECTED.items():
        row = by_head.get(head)
        if row is None:
            unresolved.append({"task": task, "reason": "HEAD_ABSENT", "head": head})
            continue
        observed = (
            str(row["dataset"]),
            str(row["head_id"]),
            int(row["h_steps"]),
            int(row["w_steps"]),
            float(row.get("cadence_seconds", cadence)),
        )
        expected = (dataset, head, h_steps, w_steps, cadence)
        if observed != expected:
            unresolved.append(
                {"task": task, "reason": "VALUE_CONFLICT", "expected": expected, "observed": observed}
            )
            continue
        physical_horizon_seconds = h_steps * cadence
        if task == "TEP_H0":
            physical_horizon_seconds = 0.0
        rows.append(
            {
                "task": task,
                "dataset": dataset,
                "head": head,
                "H_steps": h_steps,
                "W_steps": w_steps,
                "cadence_seconds": cadence,
                "physical_horizon_seconds": physical_horizon_seconds,
                "task_type": row.get("task_type", "FUTURE_FORECAST"),
                "history_steps": json.dumps(row.get("history_steps", []), separators=(",", ":")),
                "main_view": json.dumps(row.get("main_view", []), separators=(",", ":")),
                "source": config.relative_to(project).as_posix(),
                "source_sha256": _sha256(config),
            }
        )

    # These are historical differences, not unresolved conflicts: the later
    # stagewise configuration explicitly amends the frozen release task matrix.
    legacy_primary = [
        {
            "task_id": row["task_id"],
            "head_id": row["head_id"],
            "H_steps": row["h_steps"],
            "W_steps": row["w_steps"],
        }
        for row in legacy_value["heads"]
        if row.get("primary") and row.get("task_id") in {"TEP_G12", "SRU_H2S", "SRU_SO2"}
    ]
    amendment = active_value["protocol_amendment"]
    superseded = {
        "status": "SUPERSEDED_NOT_ACTIVE_CONFLICT",
        "legacy_primary_heads": legacy_primary,
        "superseding_amendment": amendment,
        "superseding_stagewise_protocol": stagewise["protocol_id"],
    }

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "FROZEN_TASK_HW_REGISTRY.csv"
    fields = list(rows[0]) if rows else ["status"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "schema_version": 1,
        "status": "PASS" if len(rows) == len(EXPECTED) and not unresolved else "PROTOCOL_BLOCKED",
        "selection_data_accessed": False,
        "outer_test_accessed": False,
        "authoritative_base_commit": "be7557933f4a2ad11c7655c04ce031f524611cb0",
        "active_source": {
            "path": config.relative_to(project).as_posix(),
            "sha256": _sha256(config),
            "protocol_id": stagewise["protocol_id"],
            "status": stagewise["status"],
        },
        "supporting_source": {
            "path": active.relative_to(project).as_posix(),
            "sha256": _sha256(active),
            "protocol_id": active_value["protocol_id"],
            "amendment": amendment,
        },
        "legacy_source": {
            "path": legacy.relative_to(project).as_posix(),
            "sha256": _sha256(legacy),
        },
        "cz_resolution": {
            "H_steps": 4,
            "W_steps": 1,
            "source": "stagewise frozen task_matrix",
            "conflict": False,
        },
        "superseded_differences": superseded,
        "unresolved_conflicts": unresolved,
        "registry_rows": rows,
        "registry_hash": _canonical_hash(rows),
    }
    (output / "HW_PROTOCOL_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = [
        "# H/W Protocol Audit",
        "",
        f"Status: `{audit['status']}`.",
        "",
        "The active H/W values were read from the committed, frozen stagewise task matrix. No target values, predictions, or outer-test metrics were read.",
        "",
        "CZ is resolved as `H=4, W=1` from the frozen task matrix; W was not inferred or guessed.",
        "",
        "The older C1 registry contains earlier TEP/SRU primary heads. They are recorded as superseded differences because the committed TEP-nowcast amendment and later stagewise protocol explicitly replace them for this validation.",
    ]
    if unresolved:
        markdown.extend(["", "Unresolved conflicts:", "", "```json", json.dumps(unresolved, indent=2), "```"])
    (output / "HW_PROTOCOL_AUDIT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_registry(args.project.resolve(), args.output.resolve())
    print(json.dumps({"status": audit["status"], "registry_hash": audit["registry_hash"]}))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

