#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.common import atomic_json, environment_snapshot, load_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "protocol_frozen_l6.yaml")
    )
    parser.add_argument("--output", default=str(ROOT / "results_cpu" / "preflight"))
    args = parser.parse_args()
    protocol, config_hash = load_json(args.config)
    actual_hash = sha256_file(args.data)
    checks = {
        "data_sha256": actual_hash == protocol["data_sha256"],
        "frozen_L6": (
            protocol["cadence_sec"] == 10.0
            and protocol["history_min"] == 40.0
            and protocol["horizon_min"] == 20.0
            and protocol["target_window_min"] == 2.0
        ),
        "no_future_input": protocol["allow_future_input"] is False,
        "hard_boundaries": protocol["allow_boundary_crossing"] is False,
        "two_directions": len(protocol["outer_directions"]) == 2,
        "four_inner_folds": len(protocol["inner_folds"]) == 4,
        "bootstrap_500": protocol["bootstrap_replicates"] == 500,
        "fp64": protocol["primary_dtype"] == "float64",
    }
    l6_path = (
        REPO
        / "MULTISCALE_PHYSICS_AUDIT_V1"
        / "results"
        / "stage2"
        / "profiles"
        / "L6__pc1"
        / "result.json"
    )
    if l6_path.is_file():
        l6 = json.loads(l6_path.read_text(encoding="utf-8"))
        checks["source_L6_structure_pass"] = bool(
            l6["gates"]["S2_stable_structure"]
        )
    else:
        checks["source_L6_structure_pass"] = False
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": protocol["schema"],
        "status": status,
        "checks": checks,
        "config_sha256": config_hash,
        "data_sha256": actual_hash,
        "environment": environment_snapshot(REPO),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "preflight.json", report)
    (output / "PREFLIGHT_REPORT.md").write_text(
        "\n".join(
            [
                "# Physics-First CPU Preflight",
                "",
                f"- Status: **{status}**",
                f"- Data SHA256: `{actual_hash}`",
                f"- Protocol SHA256: `{config_hash}`",
                "",
                *[
                    f"- `{name}`: {'PASS' if value else 'FAIL'}"
                    for name, value in checks.items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
