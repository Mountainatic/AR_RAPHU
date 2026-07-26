#!/usr/bin/env python3
"""Generate only the pre-registered Spectral v0.3 decision fields."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "spectral_v03"


def status(experiment: str) -> str:
    path = RESULT_ROOT / experiment / "summary.json"
    if not path.exists():
        return "NOT_YET_RUN"
    return str(json.loads(path.read_text(encoding="utf-8"))["status"])


def main() -> None:
    statuses = {f"E{index}": status(f"E{index}") for index in range(9)}
    e0_pass = statuses["E0"] == "E0_COMPONENT_IDENTITY_PASS"
    e1_pass = statuses["E1"] == "E1_PROJECTION_CAPACITY_PASS"
    if not e0_pass:
        next_stage = "STOP_E0_GENERATOR_REPLAY"
    elif not e1_pass:
        next_stage = "STOP_E1_PROJECTION_CAPACITY"
    else:
        next_stage = "E2_DEVELOPMENT"
    fields = {
        **{f"E{index}_STATUS": statuses[f"E{index}"] for index in range(9)},
        "FULL_KERNEL_CAPACITY": "NOT_YET_RUN",
        "DOUBLE_RESIDUALIZATION_VALID": "NOT_YET_RUN",
        "SUPPORT_RECOVERY_VALID": "NOT_YET_RUN",
        "RANK_ADAPTATION_VALID": "NOT_YET_RUN",
        "ADAPTIVE_WEIGHTING_ADOPTED": "NOT_YET_RUN",
        "PREDICTION_RECOMBINATION_VALID": "NOT_YET_RUN",
        "RECURSIVE_DEPLOYMENT_VALID": "NOT_YET_RUN",
        "NEXT_ALLOWED_STAGE": next_stage,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    decision = "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n"
    (RESULT_ROOT / "DEVELOPMENT_DECISION.md").write_text(
        decision, encoding="utf-8"
    )
    with (RESULT_ROOT / "spectral_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["field", "value"])
        writer.writerows(fields.items())


if __name__ == "__main__":
    main()
