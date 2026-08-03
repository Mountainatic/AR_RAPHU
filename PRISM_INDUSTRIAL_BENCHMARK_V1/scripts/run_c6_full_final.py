from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_benchmark.c6_full_final import run_full_c6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--c2-output", type=Path, required=True)
    parser.add_argument("--c3-output", type=Path, required=True)
    parser.add_argument("--c4-output", type=Path, required=True)
    parser.add_argument("--c5-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=20)
    args = parser.parse_args()
    decision = run_full_c6(
        args.shared.resolve(),
        args.project.resolve(),
        args.c2_output.resolve(),
        args.c3_output.resolve(),
        args.c4_output.resolve(),
        args.c5_output.resolve(),
        args.output.resolve(),
        args.n_jobs,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
