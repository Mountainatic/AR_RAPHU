#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage3 import run_stage3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-period-sec", required=True, type=float)
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment_v1.yaml"))
    args = parser.parse_args()
    summary = run_stage3(
        root=ROOT,
        config_path=Path(args.config),
        data_path=Path(args.data),
        sample_period_sec=args.sample_period_sec,
    )
    print("STAGE3_SUMMARY=" + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
