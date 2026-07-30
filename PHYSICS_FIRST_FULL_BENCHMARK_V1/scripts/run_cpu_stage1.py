#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.benchmark import run_cpu_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data", default=str(ROOT / "shared"))
    parser.add_argument(
        "--protocol", default=str(ROOT / "configs" / "protocol_frozen_l6.yaml")
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "cpu_models.yaml")
    )
    parser.add_argument("--results", default=str(ROOT / "results_cpu"))
    parser.add_argument("--n-jobs", type=int, default=20)
    args = parser.parse_args()
    result = run_cpu_benchmark(
        repo_root=REPO,
        project_root=ROOT,
        shared_root=Path(args.shared_data),
        protocol_path=Path(args.protocol),
        cpu_path=Path(args.config),
        results_root=Path(args.results),
        n_jobs=args.n_jobs,
    )
    print("CPU_STAGE1_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"].startswith("COMPLETED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
