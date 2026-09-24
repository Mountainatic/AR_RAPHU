"""Aggregate the development-only CZ E2 S1_K screening pilots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ALLOWED_COMPLETION = {"PASS", "COMPLETED_WITH_RETAINED_FAILURES"}
DIRECTIONS = ("Rod_1_to_Rod_2", "Rod_2_to_Rod_1")
SEEDS = tuple(range(10))


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


def _unit_path(
    initial_rod1_unit: Path, parallel_root: Path, direction: str, seed: int
) -> Path:
    if direction == "Rod_1_to_Rod_2" and seed == 0:
        return initial_rod1_unit
    return parallel_root / direction / f"seed_{seed}"


def collect(initial_rod1_unit: Path, parallel_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for seed in SEEDS:
            unit = _unit_path(initial_rod1_unit, parallel_root, direction, seed)
            status = json.loads(
                (unit / "AUTHORITY_REFIT_STATUS.json").read_text(encoding="utf-8")
            )
            if status.get("status") not in ALLOWED_COMPLETION:
                raise RuntimeError(f"STOP_S1_SCREENING_UNIT_INCOMPLETE:{unit}")
            if status.get("formal_target_or_ood_accessed") is not False:
                raise RuntimeError(f"STOP_S1_SCREENING_EVIDENCE_BOUNDARY:{unit}")
            results = list((unit / "results" / "DEVELOPMENT").rglob("RESULT.json"))
            if len(results) != 8:
                raise RuntimeError(f"STOP_S1_SCREENING_STAGE_COUNT:{unit}:{len(results)}")
            joint_paths = [path for path in results if path.parent.name == "joint_lift"]
            if len(joint_paths) != 1:
                raise RuntimeError(f"STOP_S1_SCREENING_JOINT_K_COUNT:{unit}")
            joint = json.loads(joint_paths[0].read_text(encoding="utf-8"))
            other_k = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in results
                if "/K/" in path.as_posix() and path not in joint_paths
            ]
            nontruth_exact_zero = bool(other_k) and all(
                item.get("status") == "PASS"
                and item.get("selected_family") == "EXACT_ZERO"
                for item in other_k
            )
            recovered = bool(
                joint.get("status") == "PASS"
                and joint.get("active") is True
                and nontruth_exact_zero
            )
            rmse = joint.get("rmse") if recovered else None
            r2 = joint.get("r2") if recovered else None
            if recovered and not all(
                math.isfinite(float(value)) for value in (rmse, r2)
            ):
                raise RuntimeError(f"STOP_S1_SCREENING_NONFINITE_METRIC:{unit}")
            rows.append(
                {
                    "direction": direction,
                    "seed": seed,
                    "unit_status": status["status"],
                    "inner_workers": int(status.get("inner_workers", -1)),
                    "stage_count": len(results),
                    "true_k_recovered": recovered,
                    "nontruth_k_exact_zero": nontruth_exact_zero,
                    "joint_k_status": joint.get("status"),
                    "selected_family": joint.get("selected_family"),
                    "selected_profile": joint.get("selected_profile"),
                    "rmse": rmse,
                    "r2": r2,
                    "formal_target_or_ood_accessed": False,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    directions = []
    for direction in DIRECTIONS:
        group = [row for row in rows if row["direction"] == direction]
        recovered = [row for row in group if row["true_k_recovered"]]
        rate = len(recovered) / len(group)
        directions.append(
            {
                "direction": direction,
                "seeds": len(group),
                "true_k_recovered": len(recovered),
                "recovery_rate": rate,
                "retained_failures": sum(
                    row["unit_status"] == "COMPLETED_WITH_RETAINED_FAILURES"
                    for row in group
                ),
                "mean_rmse_among_recovered": mean(
                    float(row["rmse"]) for row in recovered
                ),
                "mean_r2_among_recovered": mean(
                    float(row["r2"]) for row in recovered
                ),
                "phase_a_gate_authority": False,
            }
        )
    total_recovered = sum(row["true_k_recovered"] for row in rows)
    pooled_rate = total_recovered / len(rows)
    return {
        "status": "PASS_SCREENING_COMPLETE",
        "role": "P1_TINY_PILOT_NON_SELECTION_AUTHORITY",
        "regime": "S1_K",
        "selection_authority": False,
        "statistical_claims_allowed": False,
        "screening_seeds": list(SEEDS),
        "directions": directions,
        "pooled": {
            "units": len(rows),
            "true_k_recovered": total_recovered,
            "recovery_rate": pooled_rate,
            "retained_failures": sum(
                row["unit_status"] == "COMPLETED_WITH_RETAINED_FAILURES"
                for row in rows
            ),
            "phase_a_gate_authority": False,
        },
        "phase_a_gate_applicability": "NONE_S1_K_SCREENING_ONLY",
        "authority_phase_a_recovery_scope": ["S2:C", "S3:W", "S4:A"],
        "authority_phase_a_evidence": "formal_30seed_only",
        "next_stage": "S2_KC_SCREENING_ADAPTER_REQUIRED",
        "formal_target_or_ood_accessed": False,
        "completed_utc": _utc(),
    }


def write_outputs(rows: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    summary = summarize(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "seed_results.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "selected_profile": json.dumps(row["selected_profile"]),
                }
            )
    os.replace(temporary, csv_path)
    summary["seed_results_path"] = str(csv_path)
    _write_json(output_root / "SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-rod1-unit", type=Path, required=True)
    parser.add_argument("--parallel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.initial_rod1_unit.resolve(), args.parallel_root.resolve())
    result = write_outputs(rows, args.output_root.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
