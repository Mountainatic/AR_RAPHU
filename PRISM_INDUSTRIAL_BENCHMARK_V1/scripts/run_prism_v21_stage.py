from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_benchmark.v21_config import V21Paths
from prism_benchmark.v21_runner import CHAIN_STAGES, run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen PRISM v2.1 SRU stage")
    parser.add_argument("stage", choices=CHAIN_STAGES)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output = (args.output or project / "results_prism_v2_1_sru").resolve()
    paths = V21Paths(
        project=project,
        shared=args.shared.resolve(),
        output=output,
    )
    result = run_stage(args.stage, paths)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
