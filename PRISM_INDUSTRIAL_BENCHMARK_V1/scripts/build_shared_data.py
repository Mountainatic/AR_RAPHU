from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prism_benchmark.c1_builder import build_shared_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Build immutable PRISM C1 shared data")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/c1_tasks.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_shared_data(args.raw_root, args.registry_root, args.config, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
