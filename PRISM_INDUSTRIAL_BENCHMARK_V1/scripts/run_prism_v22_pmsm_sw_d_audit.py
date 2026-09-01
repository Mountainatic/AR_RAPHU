#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.pmsm_sw_d_audit import run_primary_d_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered PMSM stator-winding D-only implementation audit"
    )
    parser.add_argument("--shared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="Optional preregistered channel subset for implementation smoke; default runs all eight.",
    )
    args = parser.parse_args()
    summary = run_primary_d_audit(
        args.shared.resolve(),
        PROJECT_ROOT.resolve(),
        args.output.resolve(),
        channels=args.channels,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
