from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_benchmark.v211_public_all_config import PublicAllPaths
from prism_benchmark.v211_public_all_reporting import report_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Report and package the PRISM v2.1.1 public-all rerun.")
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reporting-commit")
    args = parser.parse_args()
    paths = PublicAllPaths(
        project=args.project.resolve(),
        shared=args.shared.resolve(),
        run_root=args.run_root.resolve(),
    )
    result = report_all(paths, reporting_commit=args.reporting_commit)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
