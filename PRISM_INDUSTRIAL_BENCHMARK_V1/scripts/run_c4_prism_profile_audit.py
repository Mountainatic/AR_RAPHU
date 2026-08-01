from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_benchmark.c4_prism import run_c4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run_c4(args.shared, args.project, args.output, args.n_jobs), sort_keys=True))


if __name__ == "__main__":
    main()
