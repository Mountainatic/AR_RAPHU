from __future__ import annotations

import argparse
import json
import os
import sys
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
from prism_benchmark.v211_config import REPRESENTATIVE_STAGE1_PROTOCOL
from prism_benchmark.v211_joint_stability import run_joint_stability_view
from prism_benchmark.v211_k import run_k_channel
from prism_benchmark.v211_public_all_baselines import (
    run_public_all_baseline_development,
)
from prism_benchmark.v211_representative_stage1_config import (
    PRIMARY_TASKS,
    RepresentativeStage1Paths,
    load_representative_stage1_descriptor,
)
from prism_benchmark.v211_representative_stage1_views import (
    representative_stage1_dynamic_views,
    representative_stage1_input_views,
    representative_stage1_k_jobs,
)
from prism_benchmark.v211_w import run_w_view


DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/PRISM_V211_REPRESENTATIVE_STAGE1_TEP_SRU_CPU_20260823_R1"
)


def _require_uv_runtime() -> None:
    if os.environ.get("AR_RAPHU_RUNTIME_MANAGER") != "uv":
        raise RuntimeError("AR_RAPHU_RUNTIME_MANAGER must be exactly 'uv'")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            f"representative Stage1 requires Python 3.10, got {sys.version.split()[0]}"
        )
    virtual_environment = os.environ.get("VIRTUAL_ENV")
    if not virtual_environment:
        raise RuntimeError("VIRTUAL_ENV is missing; launch through uv run --frozen")
    if Path(sys.prefix).resolve() != Path(virtual_environment).resolve():
        raise RuntimeError("interpreter prefix does not match the uv-managed VIRTUAL_ENV")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _completed(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        value.get("status") == "PASS"
        and value.get("test_accessed") is False
        and value.get("ood_accessed", False) is False
        and value.get("global_freeze_created", False) is False
    ):
        return value
    return None


def _run_jobs(
    *,
    stage: str,
    function: Callable[..., Any],
    jobs: list[tuple[Path, tuple[Any, ...]]],
    output: Path,
    workers: int,
    per_worker_gib: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    pending: list[tuple[Any, ...]] = []
    for result_path, arguments in jobs:
        previous = _completed(result_path)
        if previous is None:
            pending.append(arguments)
        else:
            results.append(previous)
    if pending:
        results.extend(
            dict(result)
            for result in run_parallel(
                function,
                pending,
                max(1, min(int(workers), len(pending))),
                per_worker_gib=float(per_worker_gib),
                label=f"REPRESENTATIVE_STAGE1_{stage}",
            )
        )
    statuses = [str(item.get("status")) for item in results]
    summary = {
        "status": (
            "PASS"
            if all(status == "PASS" for status in statuses)
            else "COMPLETED_WITH_RETAINED_FAILURES"
        ),
        "stage": stage,
        "jobs": len(results),
        "new_jobs": len(pending),
        "reused_jobs": len(results) - len(pending),
        "pass": sum(status == "PASS" for status in statuses),
        "retained_failures": sum(status != "PASS" for status in statuses),
        "test_accessed": False,
        "ood_accessed": False,
        "global_freeze_created": False,
    }
    _write_json(output / "DEVELOPMENT" / stage / "SUMMARY.json", summary)
    return summary


def run_scope(paths: RepresentativeStage1Paths) -> dict[str, Any]:
    descriptor = load_representative_stage1_descriptor(paths.project)
    input_views = representative_stage1_input_views(paths.shared)
    dynamic_views = representative_stage1_dynamic_views(paths.shared)
    k_jobs = representative_stage1_k_jobs(paths.shared)
    tasks = sorted(
        {view.head.task_id for view in [*input_views, *dynamic_views]}
    )
    if tasks != sorted(PRIMARY_TASKS):
        raise RuntimeError("representative Stage-1 task scope changed")
    result = {
        "status": "PASS",
        "stage": "REPRESENTATIVE_STAGE1_TEP_SRU_CPU_SCOPE",
        "protocol_id": descriptor["protocol_id"],
        "config_sha256": descriptor["config_sha256"],
        "base_commit": descriptor["base_commit"],
        "primary_tasks": tasks,
        "input_views": len(input_views),
        "dynamic_views": len(dynamic_views),
        "k_channel_jobs": len(k_jobs),
        "development_only": True,
        "neural_in_scope": False,
        "test_accessed": False,
        "ood_accessed": False,
        "global_freeze_created": False,
    }
    _write_json(paths.logs / "SCOPE.json", result)
    return result


def _result_path(
    paths: RepresentativeStage1Paths,
    stage: str,
    view: Any,
    channel: str | None = None,
) -> Path:
    root = paths.output / "DEVELOPMENT" / stage / view.head.head_id
    if stage == "K":
        assert channel is not None
        return root / view.proxy_policy / channel / "RESULT.json"
    if stage in {"C", "W"}:
        return root / view.proxy_policy / "RESULT.json"
    return (
        root
        / view.availability_scenario
        / view.proxy_policy
        / "RESULT.json"
    )


def run_k(
    paths: RepresentativeStage1Paths, workers: int, per_worker_gib: float
) -> dict[str, Any]:
    jobs = [
        (
            _result_path(paths, "K", view, channel),
            (
                paths.shared,
                paths.project,
                paths.output,
                view,
                channel,
                REPRESENTATIVE_STAGE1_PROTOCOL,
            ),
        )
        for view, channel in representative_stage1_k_jobs(paths.shared)
    ]
    return _run_jobs(
        stage="K",
        function=run_k_channel,
        jobs=jobs,
        output=paths.output,
        workers=workers,
        per_worker_gib=per_worker_gib,
    )


def _view_stage(
    paths: RepresentativeStage1Paths,
    stage: str,
    function: Callable[..., Any],
    views: list[Any],
    workers: int,
    per_worker_gib: float,
) -> dict[str, Any]:
    jobs = [
        (
            _result_path(paths, stage, view),
            (
                paths.shared,
                paths.project,
                paths.output,
                view,
                REPRESENTATIVE_STAGE1_PROTOCOL,
            ),
        )
        for view in views
    ]
    return _run_jobs(
        stage=stage,
        function=function,
        jobs=jobs,
        output=paths.output,
        workers=workers,
        per_worker_gib=per_worker_gib,
    )


def run_joint(
    paths: RepresentativeStage1Paths, workers: int, per_worker_gib: float
) -> dict[str, Any]:
    views = representative_stage1_dynamic_views(paths.shared)
    jobs = [
        (
            _result_path(paths, "JOINT", view),
            (
                paths.shared,
                paths.project,
                paths.output,
                None,
                view,
                REPRESENTATIVE_STAGE1_PROTOCOL,
            ),
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


def run_baselines(
    paths: RepresentativeStage1Paths, workers: int
) -> dict[str, Any]:
    return run_public_all_baseline_development(
        paths.shared,
        paths.project,
        paths.output,
        workers,
        input_views_override=representative_stage1_input_views(paths.shared),
        dynamic_views_override=representative_stage1_dynamic_views(paths.shared),
        stage_name="REPRESENTATIVE_STAGE1_TEP_SRU_CPU_BASELINE_DEVELOPMENT",
    )


def main() -> None:
    _require_uv_runtime()
    parser = argparse.ArgumentParser(
        description="Run development-only TEP/SRU representative H1 CPU stages."
    )
    parser.add_argument(
        "stage",
        choices=(
            "scope",
            "k",
            "c",
            "w",
            "a",
            "joint",
            "baselines",
            "prism",
            "all",
        ),
    )
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--per-worker-gib", type=float, default=4.0)
    args = parser.parse_args()

    paths = RepresentativeStage1Paths(
        project=args.project.resolve(),
        shared=args.shared.resolve(),
        run_root=args.run_root.resolve(),
    )
    descriptor = load_representative_stage1_descriptor(paths.project)
    workers = max(
        1,
        int(
            args.workers
            if args.workers is not None
            else descriptor["resource"]["outer_workers"]
        ),
    )
    os.environ.setdefault(
        "PRISM_V211_K_INNER_WORKERS",
        str(descriptor["resource"]["k_inner_workers"]),
    )
    for name in (
        "PRISM_V211_C_INNER_WORKERS",
        "PRISM_V211_W_INNER_WORKERS",
        "PRISM_V211_A_INNER_WORKERS",
        "PRISM_V211_JOINT_STABILITY_J_INNER_WORKERS",
    ):
        os.environ.setdefault(
            name, str(descriptor["resource"]["stage_inner_workers"])
        )
    paths.output.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    scope = run_scope(paths)
    if args.stage == "scope":
        print(json.dumps(scope, sort_keys=True))
        return

    input_views = representative_stage1_input_views(paths.shared)
    dynamic_views = representative_stage1_dynamic_views(paths.shared)
    runners = {
        "k": lambda: run_k(paths, workers, args.per_worker_gib),
        "c": lambda: _view_stage(
            paths, "C", run_c_view, input_views, workers, args.per_worker_gib
        ),
        "w": lambda: _view_stage(
            paths, "W", run_w_view, input_views, workers, args.per_worker_gib
        ),
        "a": lambda: _view_stage(
            paths, "A", run_a_view, dynamic_views, workers, args.per_worker_gib
        ),
        "joint": lambda: run_joint(paths, workers, args.per_worker_gib),
        "baselines": lambda: run_baselines(paths, workers),
    }
    stages = {
        "prism": ("k", "c", "w", "a", "joint"),
        "all": ("k", "c", "w", "a", "joint", "baselines"),
    }.get(args.stage, (args.stage,))
    result = {stage: runners[stage]() for stage in stages}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
