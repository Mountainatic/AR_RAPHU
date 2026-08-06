from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from prism_benchmark.v21_runner import CHAIN_STAGES

STAGES = CHAIN_STAGES
GIB = 1024**3


def _validate_runtime_budget(
    *,
    cpu_count: int,
    memory_limit_bytes: int | None,
    workers: int,
    threads_per_worker: int,
    memory_gib_per_worker: int,
    memory_reserve_gib: int,
) -> dict[str, Any]:
    if min(cpu_count, workers, threads_per_worker, memory_gib_per_worker) < 1:
        raise ValueError("runtime CPU/worker budgets must be positive")
    if memory_reserve_gib < 0:
        raise ValueError("runtime memory reserve must be non-negative")
    thread_budget = workers * threads_per_worker
    if thread_budget > cpu_count:
        raise ValueError(
            f"runtime oversubscribes CPUs: {workers}x{threads_per_worker}>{cpu_count}"
        )
    required_memory_bytes = (
        workers * memory_gib_per_worker + memory_reserve_gib
    ) * GIB
    if (
        memory_limit_bytes is not None
        and required_memory_bytes > memory_limit_bytes
    ):
        raise ValueError(
            "runtime memory budget exceeds the cgroup limit: "
            f"required={required_memory_bytes} limit={memory_limit_bytes}"
        )
    return {
        "status": "PASS",
        "cpu_count": cpu_count,
        "baseline_workers": workers,
        "threads_per_worker": threads_per_worker,
        "total_thread_budget": thread_budget,
        "memory_limit_bytes": memory_limit_bytes,
        "memory_gib_per_worker": memory_gib_per_worker,
        "memory_reserve_gib": memory_reserve_gib,
        "required_memory_budget_bytes": required_memory_bytes,
        "blas_thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "RAYON_NUM_THREADS",
            )
        },
    }


def _runtime_profile() -> dict[str, Any]:
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = int(os.cpu_count() or 1)
    memory_limit_path = Path("/sys/fs/cgroup/memory.max")
    memory_limit_bytes = None
    if memory_limit_path.is_file():
        raw = memory_limit_path.read_text(encoding="utf-8").strip()
        if raw != "max":
            memory_limit_bytes = int(raw)
    return _validate_runtime_budget(
        cpu_count=cpu_count,
        memory_limit_bytes=memory_limit_bytes,
        workers=int(os.environ.get("PRISM_V21_BASELINE_WORKERS", "8")),
        threads_per_worker=int(
            os.environ.get("PRISM_V21_THREADS_PER_WORKER", "4")
        ),
        memory_gib_per_worker=int(
            os.environ.get("PRISM_V21_MEMORY_GIB_PER_WORKER", "4")
        ),
        memory_reserve_gib=int(
            os.environ.get("PRISM_V21_MEMORY_RESERVE_GIB", "8")
        ),
    )


def _marker(output: Path, stage: str) -> Path:
    return {
        "b0": output / "BASELINES" / "BASELINE_REPLAY_MANIFEST.json",
        "e0": output / "FREEZE" / "E0_INHERITANCE_AUDIT.json",
        "e1": output / "FREEZE" / "E1_REGRESSION_TESTS.json",
        "e2k": output / "DEVELOPMENT" / "K" / "SUMMARY.json",
        "e2c": output / "DEVELOPMENT" / "C" / "SUMMARY.json",
        "e3": output / "DEVELOPMENT" / "W" / "SUMMARY.json",
        "e4": output / "DEVELOPMENT" / "A" / "SUMMARY.json",
        "e5": output / "DEVELOPMENT" / "JOINT" / "SUMMARY.json",
        "e6": output / "FREEZE" / "V21_SRU_FINAL_FREEZE_MANIFEST.json",
        "e7": output / "FINAL" / "TEST_ACCESS_AUDIT.json",
        "e8": output / "REPORTS" / "E8_SUMMARY.json",
    }[stage]


def _complete(marker: Path, stage: str) -> bool:
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "b0": {"BASELINE_REPLAY_FROZEN"},
        "e6": {"ASSEMBLY_FROZEN"},
        "e8": {"COMPLETED"},
    }.get(stage, {"PASS"})
    return value.get("status") in expected


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete frozen PRISM v2.1 SRU B0-E8 chain",
    )
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--through", choices=STAGES, default="e8")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Fail when a completed stage marker exists instead of resuming.",
    )
    args = parser.parse_args()
    project = args.project.resolve()
    shared = args.shared.resolve()
    output = (args.output or project / "results_prism_v2_1_sru").resolve()
    if output.name != "results_prism_v2_1_sru":
        raise SystemExit("output directory must be named results_prism_v2_1_sru")
    if not shared.is_dir():
        raise SystemExit(f"shared data root does not exist: {shared}")
    runtime_profile = _runtime_profile()
    runtime_profile["time_unix"] = time.time()
    runtime_path = output / "RUN_LOG" / "V21_RUNTIME_PROFILE.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(
        json.dumps(runtime_profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stop = STAGES.index(args.through)
    selected = STAGES[: stop + 1]
    log = output / "RUN_LOG" / "V21_CHAIN_EVENTS.jsonl"
    stage_script = project / "scripts" / "run_prism_v21_stage.py"
    for stage in selected:
        marker = _marker(output, stage)
        if _complete(marker, stage):
            if args.no_resume:
                raise SystemExit(f"completed stage already exists: {stage}: {marker}")
            _append_event(
                log,
                {
                    "event": "SKIP_COMPLETED",
                    "stage": stage,
                    "marker": str(marker),
                    "time_unix": time.time(),
                },
            )
            continue
        if marker.exists():
            raise SystemExit(f"non-PASS stage marker requires audit: {stage}: {marker}")
        if stage == "e7" and not _complete(_marker(output, "e6"), "e6"):
            raise SystemExit("E7 test access refused because E6 freeze is absent")
        command = [
            sys.executable,
            str(stage_script),
            stage,
            "--project",
            str(project),
            "--shared",
            str(shared),
            "--output",
            str(output),
        ]
        started = time.time()
        _append_event(
            log,
            {
                "event": "START",
                "stage": stage,
                "command": command,
                "time_unix": started,
            },
        )
        completed = subprocess.run(command, cwd=project, text=True)
        _append_event(
            log,
            {
                "event": "FINISH",
                "stage": stage,
                "returncode": completed.returncode,
                "elapsed_seconds": time.time() - started,
                "time_unix": time.time(),
            },
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)
        if not _complete(marker, stage):
            raise SystemExit(f"stage returned success without a valid marker: {stage}")
    summary = {
        "status": "PASS",
        "stages": list(selected),
        "through": args.through,
        "output": str(output),
        "baseline_replay_test_accessed": STAGES.index(args.through) >= STAGES.index("b0"),
        "v21_candidate_test_accessed": STAGES.index(args.through) >= STAGES.index("e7"),
        "test_accessed": STAGES.index(args.through) >= STAGES.index("e7"),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
