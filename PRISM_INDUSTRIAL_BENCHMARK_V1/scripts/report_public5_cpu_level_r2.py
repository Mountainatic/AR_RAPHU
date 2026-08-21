"""Generate the reporting-only public-five CPU/PRISM Level-R2 deliverable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from level_r2_reporting import collect_level_r2
from prism_benchmark.public5_level_r2_reporting import (
    generate_public5_cpu_level_r2_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--pytest-log", type=Path, required=True)
    args = parser.parse_args()
    result = generate_public5_cpu_level_r2_report(
        args.public_root,
        args.output_root,
        args.repository_root,
        collect_level_r2,
        pytest_log=args.pytest_log,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
