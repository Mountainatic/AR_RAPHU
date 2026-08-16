from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

from prism_benchmark.v2_runtime import run_parallel
from prism_benchmark.v211_a import run_a_view
from prism_benchmark.v211_c import run_c_view
from prism_benchmark.v211_config import PUBLIC_ALL_PROTOCOL
from prism_benchmark.v211_joint_stability import run_joint_stability_view
from prism_benchmark.v211_k import run_k_channel
from prism_benchmark.v211_public_all_baselines import (
    run_public_all_baseline_development,
)
from prism_benchmark.v211_public_all_config import (
    PublicAllPaths,
    load_public_all_descriptor,
)
from prism_benchmark.v211_public_all_views import (
    public_all_dynamic_views,
    public_all_input_views,
    public_all_k_jobs,
)
from prism_benchmark.v211_w import run_w_view


DEFAULT_RUN_ROOT = Path("/root/autodl-tmp/PRISM_V211_NATIVE_PUBLIC_ALL_20260815")
DEFAULT_WORKERS = 6
DEFAULT_PER_WORKER_GIB = 4.0


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _workers(args: argparse.Namespace, descriptor: dict[str, Any]) -> int:
    if args.workers is not None:
        return max(1, int(args.workers))
    return max(
        1,
        int(
            os.environ.get(
                "PRISM_PUBLIC_ALL_WORKERS",
                descriptor["resource"].get("outer_workers", DEFAULT_WORKERS),
            )
        ),
    )


def _per_worker_gib(args: argparse.Namespace) -> float:
    if args.per_worker_gib is not None:
        return float(args.per_worker_gib)
    return float(
        os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", DEFAULT_PER_WORKER_GIB)
    )


def _stage_summary(stage: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [str(item.get("status")) for item in results]
    return {
        "status": "PASS" if all(status == "PASS" for status in statuses) else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": stage,
        "jobs": len(results),
        "pass": sum(status == "PASS" for status in statuses),
        "retained_failures": sum(status != "PASS" for status in statuses),
        "test_accessed": False,
        "ood_accessed": False,
    }


def _run_jobs(
    *,
    stage: str,
    function: Callable[..., Any],
    jobs: list[tuple[Any, ...]],
    output: Path,
    workers: int,
    per_worker_gib: float,
) -> dict[str, Any]:
    results = run_parallel(
        function,
        jobs,
        workers,
        per_worker_gib=per_worker_gib,
        label=f"PRISM_PUBLIC_ALL_{stage}",
    )
    results = [dict(result) for result in results]
    summary = _stage_summary(stage, results)
    _write_json(output / "DEVELOPMENT" / stage / "SUMMARY.json", summary)
    return summary


def run_scope(paths: PublicAllPaths) -> dict[str, Any]:
    descriptor = load_public_all_descriptor(paths.project)
    input_views = public_all_input_views(paths.shared)
    dynamic_views = public_all_dynamic_views(paths.shared)
    k_jobs = public_all_k_jobs(paths.shared)
    heads = sorted(
        {
            view.head.task_id
            for view in [*input_views, *dynamic_views]
        }
    )
    result = {
        "status": "PASS",
        "stage": "G6_PUBLIC_ALL_SCOPE",
        "protocol_id": descriptor["protocol_id"],
        "support_contract": descriptor["support_contract"],
        "source_commit": descriptor["source_commit"],
        "input_views": len(input_views),
        "dynamic_views": len(dynamic_views),
        "k_channel_jobs": len(k_jobs),
        "primary_heads": heads,
        "primary_head_count": len(heads),
        "expected_primary_head_count": 7,
        "test_accessed": False,
        "ood_accessed": False,
    }
    if heads != sorted(descriptor["primary_tasks"]):
        raise RuntimeError("public-all primary heads do not match frozen descriptor")
    if result["input_views"] != 9 or result["dynamic_views"] != 11:
        raise RuntimeError("public-all primary view count mismatch")
    if result["k_channel_jobs"] != 124:
        raise RuntimeError("public-all K job count mismatch")
    _write_json(paths.logs / "G6_PUBLIC_ALL_SCOPE.json", result)
    return result


def run_k(paths: PublicAllPaths, workers: int, per_worker_gib: float) -> dict[str, Any]:
    jobs = [
        (
            paths.shared,
            paths.project,
            paths.output,
            view,
            channel,
            PUBLIC_ALL_PROTOCOL,
        )
        for view, channel in public_all_k_jobs(paths.shared)
    ]
    return _run_jobs(
        stage="K",
        function=run_k_channel,
        jobs=jobs,
        output=paths.output,
        workers=workers,
        per_worker_gib=per_worker_gib,
    )


def run_c(paths: PublicAllPaths, workers: int, per_worker_gib: float) -> dict[str, Any]:
    views = public_all_input_views(paths.shared)
    jobs = [
        (paths.shared, paths.project, paths.output, view, PUBLIC_ALL_PROTOCOL)
        for view in views
    ]
    return _run_jobs(
        stage="C",
        function=run_c_view,
        jobs=jobs,
        output=paths.output,
        workers=workers,
        per_worker_gib=per_worker_gib,
    )


def run_w(paths: PublicAllPaths, workers: int, per_worker_gib: float) -> dict[str, Any]:
    views = public_all_input_views(paths.shared)
    jobs = [
        (paths.shared, paths.project, paths.output, view, PUBLIC_ALL_PROTOCOL)
        for view in views
    ]
    return _run_jobs(
        stage="W",
        function=run_w_view,
        jobs=jobs,
        output=paths.output,
        workers=workers,
        per_worker_gib=per_worker_gib,
    )


def run_a(paths: PublicAllPaths, workers: int, per_worker_gib: float) -> dict[str, Any]:
    views = public_all_dynamic_views(paths.shared)
    jobs = [
        (paths.shared, paths.project, paths.output, view, PUBLIC_ALL_PROTOCOL)
        for view in views
    ]
    return _run_jobs(
        stage="A",
        function=run_a_view,
        jobs=jobs,
        output=paths.output,
        workers=workers,
        per_worker_gib=per_worker_gib,
    )


def run_joint(
    paths: PublicAllPaths,
    workers: int,
    per_worker_gib: float,
    legacy_results_root: Path | None,
) -> dict[str, Any]:
    views = public_all_dynamic_views(paths.shared)
    jobs = [
        (
            paths.shared,
            paths.project,
            paths.output,
            legacy_results_root,
            view,
            PUBLIC_ALL_PROTOCOL,
        )
        for view in views
    ]
    return _run_jobs(
        stage="JOINT",
        function=run_joint_stability_view,
        jobs=jobs,
        output=paths.output,
        workers=workers,
        per_worker_gib=per_worker_gib,
    )


def run_baselines(paths: PublicAllPaths, workers: int) -> dict[str, Any]:
    return run_public_all_baseline_development(
        paths.shared,
        paths.project,
        paths.output,
        workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PRISM v2.1.1 native-support public-all development stages."
    )
    parser.add_argument(
        "stage",
        choices=("scope", "k", "c", "w", "a", "joint", "baselines", "all"),
    )
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--legacy-results-root", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--per-worker-gib", type=float)
    args = parser.parse_args()

    project = args.project.resolve()
    paths = PublicAllPaths(
        project=project,
        shared=args.shared.resolve(),
        run_root=args.run_root.resolve(),
    )
    descriptor = load_public_all_descriptor(project)
    workers = _workers(args, descriptor)
    per_worker_gib = _per_worker_gib(args)
    paths.output.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)

    if args.stage == "scope":
        print(json.dumps(run_scope(paths), sort_keys=True))
        return

    run_scope(paths)
    runners = {
        "k": lambda: run_k(paths, workers, per_worker_gib),
        "c": lambda: run_c(paths, workers, per_worker_gib),
        "w": lambda: run_w(paths, workers, per_worker_gib),
        "a": lambda: run_a(paths, workers, per_worker_gib),
        "joint": lambda: run_joint(
            paths,
            workers,
            per_worker_gib,
            args.legacy_results_root.resolve()
            if args.legacy_results_root is not None
            else None,
        ),
        "baselines": lambda: run_baselines(paths, workers),
    }
    stages = (
        ("k", "c", "w", "a", "joint", "baselines")
        if args.stage == "all"
        else (args.stage,)
    )
    results = {stage: runners[stage]() for stage in stages}
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
