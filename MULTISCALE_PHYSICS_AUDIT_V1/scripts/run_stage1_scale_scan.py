#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage1 import run_stage1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-period-sec", required=True, type=float)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "experiment_v1.yaml")
    )
    args = parser.parse_args()
    summary = run_stage1(
        root=ROOT,
        config_path=Path(args.config).resolve(),
        data_path=Path(args.data).resolve(),
        sample_period_sec=args.sample_period_sec,
        n_jobs=args.n_jobs,
        task_ids=set(args.task_id) or None,
    )
    print("STAGE1_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["failed_tasks"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
