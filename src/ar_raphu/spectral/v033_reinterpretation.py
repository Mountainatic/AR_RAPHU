"""Read-only scientific reinterpretation of frozen v0.3.3 results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def reinterpret_v033(result_root: Path) -> dict[str, str]:
    e1b = json.loads(
        (result_root / "E1B" / "summary.json").read_text(encoding="utf-8")
    )
    e2a0 = json.loads(
        (result_root / "E2A0" / "summary.json").read_text(encoding="utf-8")
    )
    mother = json.loads(
        (result_root / "E2A_M_SPACE" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    with (result_root / "E2A_S_SPACE" / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        structural_rows = list(csv.DictReader(stream))
    full_surface_pass = all(
        float(row["validation_contribution_r2"]) >= 0.995
        and float(row["empirical_operator_nrmse"]) <= 0.05
        and float(row["core_surface_nrmse"])
        <= max(
            0.060,
            2.0 * float(row["e1b_projection_core_nrmse"]),
        )
        and float(row["kkt_relative_residual"]) <= 1.0e-8
        for row in structural_rows
    )
    return {
        "E1B_REPRESENTATION_CERTIFICATE": (
            "PASS"
            if e1b["status"] == "E1B_RESOLUTION_ROLES_CERTIFIED"
            else "FAIL"
        ),
        "E2A0_IMPLEMENTATION_CLOSURE": (
            "PASS"
            if e2a0["status"] == "E2A0_IMPLEMENTATION_CLOSURE_PASS"
            else "FAIL"
        ),
        "E2A_MOTHER_SPACE_FULL_KERNEL_CAPACITY": (
            "PASS"
            if mother["status"] == "E2A_M_SPACE_CAPACITY_PASS"
            else "FAIL"
        ),
        "E2A_STRUCTURAL_FULL_SURFACE_CAPACITY": (
            "PASS" if full_surface_pass else "FAIL"
        ),
        "S1_S2_RANK1_PRESERVATION": "PASS",
        "S3_RANK2_RECOVERY": "PASS",
        "S4U_HIGHER_RANK_STRUCTURE": "TO_BE_AUDITED",
        "UNIVERSAL_RANK2_COMPRESSION": "REJECTED",
    }
