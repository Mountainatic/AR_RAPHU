#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_workbook_data
from src.preflight import run_preflight
from src.runtime import atomic_json, load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-period-sec", required=True, type=float)
    parser.add_argument("--output", default="results/preflight")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "experiment_v1.yaml")
    )
    args = parser.parse_args()
    config, config_sha = load_config(args.config)
    workbook = load_workbook_data(
        args.data,
        required_sheets=config["data"]["required_sheets"],
        required_columns=config["data"]["required_columns"],
    )
    report = run_preflight(
        workbook, config, sample_period_sec=args.sample_period_sec
    )
    report["config_sha256"] = config_sha
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "preflight.json", report)
    atomic_json(ROOT / "results" / "breakpoint_manifest.json", {
        "data_sha256": workbook.sha256,
        "threshold_mm": config["data"]["diameter_break_threshold_mm"],
        "sheets": {
            name: {
                "detected": details["detected_breakpoints"],
                "frozen": details["frozen_breakpoints"],
                "verified": details["frozen_breakpoints_verified"],
                "main_stable_segment": details["main_stable_segment"],
            }
            for name, details in report["sheets"].items()
        },
    })
    lines = [
        "# MULTISCALE-PHYSICS-AUDIT V1 Preflight",
        "",
        f"- Status: **{'PASS' if report['preflight_pass'] else 'FAIL'}**",
        f"- Data SHA256: `{workbook.sha256}`",
        f"- Sample period: `{args.sample_period_sec:g} s`",
        "- Timestamp audit: `NOT_AVAILABLE` (sample period supplied explicitly)",
        "",
        "## Stable segments and breakpoints",
        "",
        "| Sheet | Samples | Detected | Frozen | Verified | Stable hours |",
        "|---|---:|---|---|---|---:|",
    ]
    for sheet, details in report["sheets"].items():
        lines.append(
            f"| {sheet} | {details['samples']} | "
            f"`{details['detected_breakpoints']}` | "
            f"`{details['frozen_breakpoints']}` | "
            f"{details['frozen_breakpoints_verified']} | "
            f"{details['main_stable_duration_hours']:.2f} |"
        )
    lines += [
        "",
        "This stage audits time scales and data boundaries only. It does not "
        "declare model evidence.",
        "",
    ]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "PREFLIGHT_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({
        "PREFLIGHT_STATUS": "PASS" if report["preflight_pass"] else "FAIL",
        "DATA_SHA256": workbook.sha256,
        "CONFIG_SHA256": config_sha,
    }))
    return 0 if report["preflight_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
