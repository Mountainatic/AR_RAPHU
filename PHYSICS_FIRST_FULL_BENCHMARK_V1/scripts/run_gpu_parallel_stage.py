#!/usr/bin/env python3
"""Run one GPU benchmark stage as independent MPS-backed model shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


RUNNERS = {
    "core": "run_gpu_stage1_core.py",
    "frontier": "run_gpu_stage2_frontier.py",
    "finalists": "run_gpu_stage3_finalists.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=sorted(RUNNERS))
    parser.add_argument("--shared", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--cpu-results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--models")
    parser.add_argument("--directions")
    parser.add_argument("--parallel-workers", type=int, default=6)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--log-prefix")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_model_ids(config_path: Path, stage: str, selected: set[str] | None) -> list[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_ids = [
        str(model["id"])
        for model in config["models"]
        if str(model["stage"]) == stage
        and (selected is None or str(model["id"]) in selected)
    ]
    if selected is not None:
        missing = sorted(selected.difference(model_ids))
        if missing:
            raise RuntimeError(f"MODELS_NOT_IN_STAGE:{stage}:{','.join(missing)}")
    if not model_ids:
        raise RuntimeError(f"NO_MODELS_FOR_STAGE:{stage}")
    return model_ids


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    results = Path(args.results).resolve()
    config_path = Path(args.config).resolve()
    selected = set(args.models.split(",")) if args.models else None
    model_ids = load_model_ids(config_path, args.stage, selected)
    worker_count = max(1, min(args.parallel_workers, len(model_ids)))
    groups = [model_ids[index::worker_count] for index in range(worker_count)]
    log_prefix = args.log_prefix or args.stage
    log_root = results / "logs"
    checkpoint_root = results / "checkpoints"
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    cpu_threads = max(1, min(4, (os.cpu_count() or worker_count) // worker_count))
    mps_share = max(10, 100 // worker_count)
    processes: list[tuple[int, list[str], subprocess.Popen[bytes], object]] = []
    runner = root / "scripts" / RUNNERS[args.stage]
    for index, group in enumerate(groups):
        log_path = log_root / f"{log_prefix}_shard_{index}.log"
        command = [
            args.python_bin,
            str(runner),
            "--shared",
            str(Path(args.shared).resolve()),
            "--config",
            str(config_path),
            "--results",
            str(results),
            "--device",
            args.device,
            "--models",
            ",".join(group),
            "--seeds",
            args.seeds,
            "--strict-folds",
            "--workers",
            str(args.loader_workers),
            "--train-fraction",
            str(args.train_fraction),
            "--checkpoint-name",
            f"{log_prefix}_shard_{index}.json",
            "--skip-aggregate",
        ]
        if args.cpu_results:
            command.extend(["--cpu-results", str(Path(args.cpu_results).resolve())])
        if args.directions:
            command.extend(["--directions", args.directions])
        if args.force:
            command.append("--force")
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": str(cpu_threads),
                "MKL_NUM_THREADS": str(cpu_threads),
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": str(cpu_threads),
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(mps_share),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        handle = log_path.open("wb")
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        (log_root / f"{log_prefix}_shard_{index}.pid").write_text(
            f"{process.pid}\n", encoding="utf-8"
        )
        processes.append((index, group, process, handle))
        print(
            f"SHARD_STARTED index={index} pid={process.pid} "
            f"models={','.join(group)} log={log_path}",
            flush=True,
        )

    return_codes: dict[int, int] = {}
    for index, _group, process, handle in processes:
        return_codes[index] = process.wait()
        handle.close()
        print(f"SHARD_FINISHED index={index} rc={return_codes[index]}", flush=True)

    aggregate = subprocess.run(
        [
            args.python_bin,
            str(root / "scripts" / "aggregate_gpu_results.py"),
            "--results",
            str(results),
        ],
        check=False,
    )
    status = "PASS" if all(code == 0 for code in return_codes.values()) and aggregate.returncode == 0 else "PARTIAL"
    payload = {
        "status": status,
        "stage": args.stage,
        "models": model_ids,
        "seeds": [int(value) for value in args.seeds.split(",") if value],
        "parallel_shards": worker_count,
        "cpu_threads_per_shard": cpu_threads,
        "mps_active_thread_percentage_per_shard": mps_share,
        "train_fraction": args.train_fraction,
        "directions": args.directions,
        "return_codes": return_codes,
    }
    (checkpoint_root / "latest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PARALLEL_STAGE_RESULT=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
