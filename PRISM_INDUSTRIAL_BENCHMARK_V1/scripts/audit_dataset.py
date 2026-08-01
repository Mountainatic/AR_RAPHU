#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.stage0 import DATASET_NAMES, run_stage0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict PRISM V1 Stage-0 dataset audit")
    parser.add_argument("--dataset", choices=["all", *DATASET_NAMES], default="all")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    summary = run_stage0(args.raw_root, args.registry, args.dataset)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

