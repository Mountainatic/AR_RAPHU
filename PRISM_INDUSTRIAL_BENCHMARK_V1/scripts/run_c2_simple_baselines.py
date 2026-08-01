from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_benchmark.cpu_simple_baselines import run_simple_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_simple_baselines(args.shared, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
