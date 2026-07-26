#!/usr/bin/env python3
"""Generate only the pre-registered Spectral v0.3/v0.3.1 decision fields."""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "spectral_v03"


def status(experiment: str) -> str:
    path = RESULT_ROOT / experiment / "summary.json"
    if not path.exists():
        return "NOT_YET_RUN"
    return str(json.loads(path.read_text(encoding="utf-8"))["status"])


def summarize_v03() -> None:
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


def summarize_v031() -> None:
    result_root = ROOT / "results" / "spectral_v031"

    def v031_status(experiment: str) -> str:
        path = result_root / experiment / "summary.json"
        if not path.exists():
            return "NOT_YET_RUN"
        return str(json.loads(path.read_text(encoding="utf-8"))["status"])

    e1r = v031_status("E1R")
    e2a = v031_status("E2A")
    e2b = v031_status("E2B")
    e3 = v031_status("E3")
    if e1r != "E1R_REPRESENTATION_CERTIFIED_32x16":
        next_stage = "STOP_REPRESENTATION"
    elif e2a != "E2A_SINGLE_KERNEL_CAPACITY_PASS":
        next_stage = "STOP_SINGLE_KERNEL_CAPACITY"
    elif e2b != "E2B_JOINT_EXTERNAL_CAPACITY_PASS":
        next_stage = "STOP_JOINT_IDENTIFIABILITY"
    elif e3 != "E3_DOUBLE_RESIDUALIZATION_PASS":
        next_stage = "STOP_NUISANCE_ORTHOGONALIZATION"
    else:
        next_stage = "ALLOW_E4_SUPPORT_VALIDATION"
    fields = {
        "E0_STATUS": "REUSED_E0_COMPONENT_IDENTITY_PASS_FROM_V03",
        "OLD_E1_STATUS": "E1_COMPRESSED_LAG_BASIS_UNDERSPECIFIED",
        "E1R_STATUS": e1r,
        "E2A_STATUS": e2a,
        "E2B_STATUS": e2b,
        "E3_STATUS": e3,
        "REPRESENTATION_CERTIFIED": (
            "TRUE"
            if e1r == "E1R_REPRESENTATION_CERTIFIED_32x16"
            else "FALSE"
        ),
        "SINGLE_KERNEL_CAPACITY_VALID": (
            "TRUE" if e2a == "E2A_SINGLE_KERNEL_CAPACITY_PASS" else "FALSE"
        ),
        "JOINT_EXTERNAL_CAPACITY_VALID": (
            "NOT_YET_RUN"
            if e2b == "NOT_YET_RUN"
            else (
                "TRUE"
                if e2b == "E2B_JOINT_EXTERNAL_CAPACITY_PASS"
                else "FALSE"
            )
        ),
        "DOUBLE_RESIDUALIZATION_VALID": (
            "NOT_YET_RUN"
            if e3 == "NOT_YET_RUN"
            else (
                "TRUE"
                if e3 == "E3_DOUBLE_RESIDUALIZATION_PASS"
                else "FALSE"
            )
        ),
        "NEXT_ALLOWED_STAGE": next_stage,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    decision = "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n"
    (result_root / "V031_CORE_DECISION.md").write_text(
        decision, encoding="utf-8"
    )
    with (result_root / "spectral_v031_core_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["field", "value"])
        writer.writerows(fields.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "spectral_v03.yaml",
    )
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 1)) == 2:
        summarize_v031()
    else:
        summarize_v03()


if __name__ == "__main__":
    main()
