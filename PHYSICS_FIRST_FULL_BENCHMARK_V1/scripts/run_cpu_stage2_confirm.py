#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "results_cpu"))
    args = parser.parse_args()
    root = Path(args.results)
    decision = json.loads(
        (root / "CPU_FINAL_DECISION.json").read_text(encoding="utf-8")
    )
    audit = decision["formal_primary_model"]["physics_audit"]
    checks = {
        "two_directions": len(audit["directions"]) == 2,
        "bootstrap_K_500": audit["pooled"][
            "bootstrap_K_vs_persistence"
        ]["replicates"] == 500,
        "bootstrap_residual_AR_500": audit["pooled"][
            "bootstrap_AR_given_K"
        ]["replicates"] == 500,
        "KKT": decision["FP64_certification"]["KKT_max"] <= 1.0e-8,
        "FP64": decision["FP64_certification"]["predictions_float64"],
        "nonlinear_exact_zero": decision["nonlinear_K"]
        == "EXACT_ZERO_BOTH_DIRECTIONS",
    }
    scientific_registration = {
        "bidirectional_K_positive": all(
            float(values["G_K"]) > 0.0
            for values in audit["directions"].values()
        ),
        "pooled_K_positive": float(audit["pooled"]["G_K"]) > 0.0,
        "residual_AR_exact_zero_both_directions": all(
            abs(float(values["G_AR_given_K"])) <= 1.0e-15
            for values in audit["directions"].values()
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    output = {
        "status": status,
        "checks": checks,
        "scientific_registration": scientific_registration,
        "physics_audit": audit,
    }
    (root / "CPU_STAGE2_CONFIRMATION.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
