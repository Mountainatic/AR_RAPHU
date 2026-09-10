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

from prism_benchmark.cpu_data import input_columns
from prism_benchmark.v2_runtime import run_parallel
from prism_benchmark.v2_views import development_dynamic_views, development_input_views
from prism_benchmark.v211_a import run_a_view
from prism_benchmark.v211_c import run_c_view
from prism_benchmark.v211_config import PUBLIC_ALL_PROTOCOL
from prism_benchmark.v211_joint_stability import run_joint_stability_view
from prism_benchmark.v211_k import run_k_channel
from prism_benchmark.v211_w import run_w_view


TEP_HEAD = "TEP_G_NOWCAST_H0__H0__W1"
SRU_HEADS = {
    "SRU_H2S_REP_H1__H1__W1",
    "SRU_SO2_REP_H1__H1__W1",
}
PUBLIC_HEADS = {
    "DEB_C4__H5__W1",
    "PMSM_PM5__H600__W60",
    "METRO_P60__H6__W1",
    "METRO_OIL20__H120__W12",
}
EXPECTED_HEADS = {TEP_HEAD, *SRU_HEADS, *PUBLIC_HEADS}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scoped_views(
    roots: tuple[tuple[Path, set[str]], ...], *, dynamic: bool
) -> list[tuple[Path, Any]]:
    loader = development_dynamic_views if dynamic else development_input_views
    result = [
        (root, view)
        for root, allowed in roots
        for view in loader(root)
        if view.head.head_id in allowed
    ]
    return sorted(
        result,
        key=lambda item: (
            item[1].head.dataset,
            item[1].head.head_id,
            item[1].availability_scenario,
            item[1].proxy_policy,
        ),
    )


def _summary(stage: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [str(item.get("status")) for item in results]
    return {
        "status": "PASS" if all(value == "PASS" for value in statuses) else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": stage,
        "jobs": len(results),
        "pass": sum(value == "PASS" for value in statuses),
        "retained_failures": sum(value != "PASS" for value in statuses),
        "test_accessed": False,
        "ood_accessed": False,
    }


def _run(
    stage: str,
    function: Callable[..., Any],
    jobs: list[tuple[Any, ...]],
    output: Path,
    workers: int,
    per_worker_gib: float,
) -> dict[str, Any]:
    results = [
        dict(value)
        for value in run_parallel(
            function,
            jobs,
            workers,
            per_worker_gib=per_worker_gib,
            label=f"STRICT_OOF_PUBLIC5_HYBRID_HW_{stage}",
        )
    ]
    value = _summary(stage, results)
    _write_json(output / "DEVELOPMENT" / stage / "SUMMARY.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the strict-OOF five-public-dataset hybrid H/W development matrix."
    )
    parser.add_argument("stage", choices=("scope", "k", "c", "w", "a", "joint", "all"))
    parser.add_argument("--tep-shared", type=Path, required=True)
    parser.add_argument("--sru-shared", type=Path, required=True)
    parser.add_argument("--public-shared", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--per-worker-gib", type=float, default=4.0)
    args = parser.parse_args()

    roots = (
        (args.tep_shared.resolve(), {TEP_HEAD}),
        (args.sru_shared.resolve(), SRU_HEADS),
        (args.public_shared.resolve(), PUBLIC_HEADS),
    )
    project = args.project.resolve()
    output = args.run_root.resolve() / "results"
    output.mkdir(parents=True, exist_ok=True)
    input_views = _scoped_views(roots, dynamic=False)
    dynamic_views = _scoped_views(roots, dynamic=True)
    heads = {view.head.head_id for _, view in [*input_views, *dynamic_views]}
    k_jobs = [
        (root, project, output, view, channel, PUBLIC_ALL_PROTOCOL)
        for root, view in input_views
        for channel in input_columns(root, view.head.task_id, view.proxy_policy)
    ]
    scope = {
        "status": "PASS",
        "protocol": "STRICT_NESTED_OOF_PUBLIC5_HYBRID_HW_DEVELOPMENT_V1",
        "heads": sorted(heads),
        "input_views": len(input_views),
        "dynamic_views": len(dynamic_views),
        "k_jobs": len(k_jobs),
        "test_accessed": False,
        "ood_accessed": False,
    }
    if heads != EXPECTED_HEADS or len(input_views) != 9 or len(dynamic_views) != 11:
        raise RuntimeError(f"hybrid H/W scope mismatch: {scope}")
    _write_json(args.run_root.resolve() / "logs" / "SCOPE.json", scope)
    if args.stage == "scope":
        print(json.dumps(scope, sort_keys=True))
        return

    jobs = {
        "k": (run_k_channel, k_jobs),
        "c": (
            run_c_view,
            [(root, project, output, view, PUBLIC_ALL_PROTOCOL) for root, view in input_views],
        ),
        "w": (
            run_w_view,
            [(root, project, output, view, PUBLIC_ALL_PROTOCOL) for root, view in input_views],
        ),
        "a": (
            run_a_view,
            [(root, project, output, view, PUBLIC_ALL_PROTOCOL) for root, view in dynamic_views],
        ),
        "joint": (
            run_joint_stability_view,
            [(root, project, output, None, view, PUBLIC_ALL_PROTOCOL) for root, view in dynamic_views],
        ),
    }
    stages = ("k", "c", "w", "a", "joint") if args.stage == "all" else (args.stage,)
    results = {
        stage: _run(
            stage.upper(), jobs[stage][0], jobs[stage][1], output, args.workers, args.per_worker_gib
        )
        for stage in stages
    }
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
