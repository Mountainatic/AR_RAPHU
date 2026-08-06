from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_benchmark.v21_config import V21Paths
from prism_benchmark.v21_runner import run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen PRISM v2.1 SRU stage")
    parser.add_argument("stage", choices=["e0", "e1", "e2k", "e2c", "e3", "e4", "e5", "e6", "e7", "e8"])
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Frozen prediction root; required by E6 and reused by E7/E8.",
    )
    args = parser.parse_args()
    project = args.project.resolve()
    output = (args.output or project / "results_prism_v2_1_sru").resolve()
    paths = V21Paths(
        project=project,
        shared=args.shared.resolve(),
        output=output,
        baseline_root=(
            None if args.baseline_root is None else args.baseline_root.resolve()
        ),
    )
    result = run_stage(args.stage, paths)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
