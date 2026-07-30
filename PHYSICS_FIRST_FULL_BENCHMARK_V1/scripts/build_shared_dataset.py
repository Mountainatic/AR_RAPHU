#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.shared_dataset import build_shared_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "protocol_frozen_l6.yaml")
    )
    parser.add_argument("--output", default=str(ROOT / "shared"))
    parser.add_argument(
        "--package",
        default=str(ROOT / "return" / "SHARED_BENCHMARK_DATASET_bundle.zip"),
    )
    args = parser.parse_args()
    result = build_shared_dataset(
        repo_root=REPO,
        project_root=ROOT,
        data_path=Path(args.data),
        config_path=Path(args.config),
        shared_root=Path(args.output),
        package_path=Path(args.package),
    )
    print("SHARED_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
