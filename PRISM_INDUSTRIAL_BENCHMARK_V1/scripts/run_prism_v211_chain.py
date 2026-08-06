from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


STAGES = (
    "e0r",
    "e1r",
    "e2rk",
    "e2rc",
    "e3r",
    "e4r",
    "e5r",
    "e55",
    "e6r",
    "e7r",
    "e8r",
)
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
    if memory_limit_bytes is not None and required_memory_bytes > memory_limit_bytes:
        raise ValueError(
            "runtime memory budget exceeds the cgroup limit: "
            f"required={required_memory_bytes} limit={memory_limit_bytes}"
        )
    return {
        "status": "PASS",
        "runtime_manager": os.environ.get("AR_RAPHU_RUNTIME_MANAGER"),
        "python_executable": sys.executable,
        "cpu_count": cpu_count,
        "workers": workers,
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
    memory_limit = None
    memory_path = Path("/sys/fs/cgroup/memory.max")
    if memory_path.is_file():
        value = memory_path.read_text(encoding="utf-8").strip()
        if value != "max":
            memory_limit = int(value)
    workers = int(os.environ.get("PRISM_V211_WORKERS", "8"))
    threads = int(os.environ.get("PRISM_V211_THREADS_PER_WORKER", "4"))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    os.environ["PRISM_V211_WORKERS"] = str(workers)
    os.environ.setdefault("PRISM_MEMORY_RESERVE_GIB", "8")
    profile = _validate_runtime_budget(
        cpu_count=cpu_count,
        memory_limit_bytes=memory_limit,
        workers=workers,
        threads_per_worker=threads,
        memory_gib_per_worker=int(
            os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "4")
        ),
        memory_reserve_gib=int(
            os.environ.get("PRISM_V211_MEMORY_RESERVE_GIB", "8")
        ),
    )
    if profile["runtime_manager"] != "uv":
        raise RuntimeError("server execution requires AR_RAPHU_RUNTIME_MANAGER=uv")
    return profile


def _marker(output: Path, stage: str) -> Path:
    return {
        "e0r": output / "FREEZE" / "E0R_HASH_AUDIT.json",
        "e1r": output / "FREEZE" / "E1R_REGRESSION_TESTS.json",
        "e2rk": output / "DEVELOPMENT" / "K" / "SUMMARY.json",
        "e2rc": output / "DEVELOPMENT" / "C" / "SUMMARY.json",
        "e3r": output / "DEVELOPMENT" / "W" / "SUMMARY.json",
        "e4r": output / "DEVELOPMENT" / "A" / "SUMMARY.json",
        "e5r": output / "DEVELOPMENT" / "JOINT" / "SUMMARY.json",
        "e55": output / "FREEZE" / "V211_DEVELOPMENT_DECISION.json",
        "e6r": output / "FREEZE" / "V211_SRU_FINAL_FREEZE_MANIFEST.json",
        "e7r": output / "FINAL" / "TEST_ACCESS_AUDIT.json",
        "e8r": output / "REPORTS" / "E8R_SUMMARY.json",
    }[stage]


def _status(marker: Path) -> str | None:
    if not marker.is_file():
        return None
    return str(json.loads(marker.read_text(encoding="utf-8")).get("status"))


def _complete(marker: Path, stage: str) -> bool:
    status = _status(marker)
    expected = {
        "e5r": {"PASS", "COMPLETED_WITH_RETAINED_FAILURES"},
        "e55": {"PASS", "V2_1_1_DEVELOPMENT_STOP"},
        "e6r": {"V2_1_1_ASSEMBLY_FROZEN"},
        "e8r": {"COMPLETED"},
    }.get(stage, {"PASS"})
    if status not in expected:
        return False
    if stage == "e55" and status == "V2_1_1_DEVELOPMENT_STOP":
        output = marker.parents[1]
        summary_path = output / "REPORTS" / "E55_STOP_PACKAGE_SUMMARY.json"
        if not summary_path.is_file():
            return False
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        package = summary.get("package", {})
        return (
            summary.get("status") == "V2_1_1_DEVELOPMENT_STOP"
            and Path(str(package.get("zip_path", ""))).is_file()
            and Path(str(package.get("sidecar_path", ""))).is_file()
        )
    return True


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen PRISM v2.1.1 SRU E0R-E8R chain"
    )
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--through", choices=STAGES, default="e8r")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    shared = args.shared.resolve()
    baseline_source = args.baseline_source.resolve()
    output = (args.output or project / "results_prism_v2_1_1_sru").resolve()
    if output.name != "results_prism_v2_1_1_sru":
        raise SystemExit("output directory must be named results_prism_v2_1_1_sru")
    if not shared.is_dir():
        raise SystemExit(f"shared data root does not exist: {shared}")
    if not baseline_source.is_dir():
        raise SystemExit(f"baseline source does not exist: {baseline_source}")
    profile = _runtime_profile()
    profile["time_unix"] = time.time()
    runtime_path = output / "RUN_LOG" / "V211_RUNTIME_PROFILE.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    if runtime_path.is_file():
        existing_profile = json.loads(runtime_path.read_text(encoding="utf-8"))
        for key in (
            "runtime_manager",
            "cpu_count",
            "workers",
            "threads_per_worker",
            "memory_gib_per_worker",
            "memory_reserve_gib",
        ):
            if existing_profile.get(key) != profile.get(key):
                raise SystemExit(f"runtime profile changed on resume: {key}")
    else:
        runtime_path.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    stop = STAGES.index(args.through)
    selected = STAGES[: stop + 1]
    log = output / "RUN_LOG" / "V211_CHAIN_EVENTS.jsonl"
    stage_script = project / "scripts" / "run_prism_v211_stage.py"
    terminal_status = "PASS"
    completed_stages = []
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
            completed_stages.append(stage)
            if stage == "e55" and _status(marker) == "V2_1_1_DEVELOPMENT_STOP":
                terminal_status = "V2_1_1_DEVELOPMENT_STOP"
                break
            continue
        resumable_incomplete_stop = (
            stage == "e55"
            and marker.is_file()
            and _status(marker) == "V2_1_1_DEVELOPMENT_STOP"
        )
        if marker.exists() and not resumable_incomplete_stop:
            raise SystemExit(f"non-complete stage marker requires audit: {stage}: {marker}")
        if stage == "e7r" and not _complete(_marker(output, "e6r"), "e6r"):
            raise SystemExit("E7R test access refused because E6R freeze is absent")
        command = [
            sys.executable,
            str(stage_script),
            stage,
            "--project",
            str(project),
            "--shared",
            str(shared),
            "--baseline-source",
            str(baseline_source),
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
        completed = subprocess.run(command, cwd=project, text=True, env=os.environ.copy())
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
        completed_stages.append(stage)
        if stage == "e55" and _status(marker) == "V2_1_1_DEVELOPMENT_STOP":
            terminal_status = "V2_1_1_DEVELOPMENT_STOP"
            break
    summary = {
        "status": terminal_status,
        "stages_requested": list(selected),
        "stages_completed": completed_stages,
        "through": args.through,
        "output": str(output),
        "baseline_replay_test_accessed": True,
        "v211_candidate_test_accessed": "e7r" in completed_stages,
        "test_accessed": "e7r" in completed_stages,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
