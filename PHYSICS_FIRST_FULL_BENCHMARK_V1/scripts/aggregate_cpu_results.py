#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting import build_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "results_cpu"))
    args = parser.parse_args()
    report = build_report(Path(args.results))
    print(json.dumps({"status": "PASS", "report": str(report.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
