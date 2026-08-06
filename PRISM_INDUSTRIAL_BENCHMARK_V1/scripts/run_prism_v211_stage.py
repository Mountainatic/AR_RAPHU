from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_benchmark.v211_config import V211Paths
from prism_benchmark.v211_runner import CHAIN_STAGES, run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen PRISM v2.1.1 SRU stage")
    parser.add_argument("stage", choices=CHAIN_STAGES)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-source", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    paths = V211Paths(
        project=project,
        shared=args.shared.resolve(),
        output=(
            args.output or project / "results_prism_v2_1_1_sru"
        ).resolve(),
        baseline_source=args.baseline_source.resolve(),
    )
    result = run_stage(args.stage, paths)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
