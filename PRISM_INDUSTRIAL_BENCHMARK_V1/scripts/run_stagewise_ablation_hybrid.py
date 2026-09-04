from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from prism_benchmark.stagewise_ablation_runner import (
    fit_checkpoints,
    freeze_bootstrap_block_lengths,
    infer_checkpoints,
    matching_views,
    run_development,
)
from prism_benchmark.v211_public_all_config import PublicAllPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen hybrid-h/w stagewise ablation.")
    parser.add_argument(
        "stage", choices=("development", "freeze-bootstrap", "checkpoint", "infer")
    )
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--selection-run-root", type=Path, required=True)
    parser.add_argument("--head-id", required=True)
    parser.add_argument("--information-set", choices=("input_only", "dynamic"), default="dynamic")
    parser.add_argument("--availability-scenario", default="record_time")
    parser.add_argument("--proxy-policy", default="primary")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--destination-run-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-worker-gib", type=float, default=4.0)
    args = parser.parse_args()
    paths = PublicAllPaths(
        args.project.resolve(), args.shared.resolve(), args.selection_run_root.resolve()
    )
    if args.stage == "development":
        result = run_development(
            paths,
            args.head_id,
            workers=args.workers,
            per_worker_gib=args.per_worker_gib,
        )
    else:
        views = matching_views(
            paths,
            args.head_id,
            information_set=args.information_set,
            availability_scenario=args.availability_scenario,
            proxy_policy=args.proxy_policy,
        )
        if args.stage == "freeze-bootstrap":
            result = freeze_bootstrap_block_lengths(paths, views)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return
        if args.checkpoint_root is None:
            raise RuntimeError("--checkpoint-root is required")
        if args.stage == "checkpoint":
            result = fit_checkpoints(paths, args.checkpoint_root.resolve(), views)
        else:
            if args.destination_run_root is None:
                raise RuntimeError("--destination-run-root is required")
            result = infer_checkpoints(
                paths,
                args.checkpoint_root.resolve(),
                args.destination_run_root.resolve(),
                views,
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
