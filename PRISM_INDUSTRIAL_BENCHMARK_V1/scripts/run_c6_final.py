from __future__ import annotations

import argparse
from pathlib import Path

from prism_benchmark.c6_final import run_c6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--c2-output", required=True, type=Path)
    parser.add_argument("--c3-output", required=True, type=Path)
    parser.add_argument("--c4-output", required=True, type=Path)
    parser.add_argument("--c5-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = run_c6(arguments.shared, arguments.project, arguments.c2_output, arguments.c3_output, arguments.c4_output, arguments.c5_output, arguments.output)
    print(f"C6_STATUS={result['status']}")
    print(f"C6_METRIC_ROWS={result['metric_rows']}")
    print(f"C6_BOOTSTRAP_ROWS={result['bootstrap_rows']}")


if __name__ == "__main__":
    main()
