from __future__ import annotations

import argparse
from pathlib import Path

from prism_benchmark.c5_models import run_c5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--c3-output", type=Path, required=True)
    parser.add_argument("--c4-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=2)
    arguments = parser.parse_args()
    manifest = run_c5(arguments.shared, arguments.project, arguments.c3_output, arguments.c4_output, arguments.output, arguments.n_jobs)
    print(f"C5_STATUS={manifest['status']}")
    print(f"C5_JOBS={manifest['jobs']}")


if __name__ == "__main__":
    main()
