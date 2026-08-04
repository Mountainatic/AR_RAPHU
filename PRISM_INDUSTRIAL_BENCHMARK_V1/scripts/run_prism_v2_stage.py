from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prism_benchmark.v2_config import V2Paths
from prism_benchmark.v2_c import run_v3_c
from prism_benchmark.v2_a import run_v5_a
from prism_benchmark.v2_assembly import freeze_g3, run_v6_assembly
from prism_benchmark.v2_freeze import run_v0_audit
from prism_benchmark.v2_k import run_v2_channels
from prism_benchmark.v2_j import run_v7_j
from prism_benchmark.v2_state import run_v1_state
from prism_benchmark.v2_w import run_v4_w


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a numerically frozen PRISM V2 stage")
    parser.add_argument("stage", choices=["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "g3"])
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c6-summary", type=Path)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--quick-registry-check", action="store_true")
    args = parser.parse_args()
    if args.n_jobs < 1 or args.n_jobs > 31:
        raise SystemExit("--n-jobs must be between 1 and 31")
    paths = V2Paths(args.project.resolve(), args.shared.resolve(), args.output.resolve())
    if args.stage == "v0":
        if args.c6_summary is None:
            raise SystemExit("v0 requires --c6-summary")
        result = run_v0_audit(paths, args.c6_summary.resolve(), full_registry_check=not args.quick_registry_check)
    elif args.stage == "v1":
        result = run_v1_state(paths.shared, paths.project, paths.output, args.n_jobs)
    elif args.stage == "v2":
        result = run_v2_channels(paths.shared, paths.project, paths.output, args.n_jobs)
    elif args.stage == "v3":
        result = run_v3_c(paths.shared, paths.project, paths.output, args.n_jobs)
    elif args.stage == "v4":
        result = run_v4_w(paths.shared, paths.project, paths.output, args.n_jobs)
    elif args.stage == "v5":
        result = run_v5_a(paths.shared, paths.project, paths.output, args.n_jobs)
    elif args.stage == "v6":
        result = run_v6_assembly(paths.shared, paths.project, paths.output)
    elif args.stage == "v7":
        result = run_v7_j(paths.shared, paths.project, paths.output, args.n_jobs)
    else:
        result = freeze_g3(paths)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
