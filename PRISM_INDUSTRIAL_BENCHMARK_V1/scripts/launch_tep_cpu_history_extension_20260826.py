"""Sequential low-memory launcher for the TEP CPU L256 extension."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "scripts" / "run_tep_cpu_history_extension_20260826.py"
CONFIG = PROJECT / "configs" / "tep_cpu_history_extension_20260826.json"


def _read_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _rss_bytes(pid: int) -> int:
    total = 0
    pending = [pid]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        status = Path(f"/proc/{current}/status")
        try:
            lines = status.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("VmRSS:"):
                total += int(line.split()[1]) * 1024
            elif line.startswith("PPid:"):
                pass
        for child_status in Path("/proc").glob("[0-9]*/status"):
            try:
                child_lines = child_status.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                ppid = next(
                    int(line.split()[1])
                    for line in child_lines
                    if line.startswith("PPid:")
                )
                if ppid == current:
                    pending.append(int(child_status.parent.name))
            except (OSError, StopIteration, ValueError):
                continue
    return total


def _terminate_own_stage(process: subprocess.Popen, reason: str) -> None:
    marker = f"STOPPING_OWN_STAGE:{reason}\n"
    sys.stderr.write(marker)
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    raise RuntimeError(reason)


def _cgroup_memory_bytes() -> int:
    for path in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            return int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
    raise RuntimeError("STOP_CGROUP_MEMORY_ACCOUNTING_UNAVAILABLE")


def _run_stage(stage: str, shared: Path, run_root: Path, config: dict) -> None:
    logs = (
        run_root.parent / f".{run_root.name}_launcher_logs"
        if stage == "scope"
        else run_root / "launcher_logs"
    )
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{stage}.log"
    command = [
        sys.executable,
        str(RUNNER),
        stage,
        "--shared",
        str(shared),
        "--run-root",
        str(run_root),
    ]
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PRISM_V211_K_INNER_WORKERS",
        "PRISM_V211_C_INNER_WORKERS",
        "PRISM_V211_W_INNER_WORKERS",
        "PRISM_V211_A_INNER_WORKERS",
        "PRISM_V211_JOINT_STABILITY_J_INNER_WORKERS",
    ):
        environment[name] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        recommended = float(config["resources"]["recommended_memory_limit_gib"])
        stopline = float(config["resources"]["minimum_runtime_free_gib"])
        while process.poll() is None:
            rss_gib = _rss_bytes(process.pid) / (1024**3)
            try:
                cgroup_gib = _cgroup_memory_bytes() / (1024**3)
            except RuntimeError as error:
                _terminate_own_stage(process, str(error))
            free_gib = shutil.disk_usage(run_root.parent).free / (1024**3)
            if cgroup_gib >= recommended:
                _terminate_own_stage(
                    process,
                    f"STOP_CGROUP_MEMORY_LIMIT:{cgroup_gib:.3f}>={recommended:.3f}GiB"
                )
            if free_gib < stopline:
                _terminate_own_stage(
                    process, f"STOP_LOW_STORAGE:{free_gib:.3f}<{stopline:.3f}GiB"
                )
            log.write(
                f"WATCHDOG stage={stage} cgroup_gib={cgroup_gib:.3f} "
                f"process_tree_rss_gib={rss_gib:.3f} free_gib={free_gib:.3f}\n"
            )
            log.flush()
            time.sleep(2)
        if process.returncode != 0:
            raise RuntimeError(
                f"TEP CPU stage failed: {stage}; inspect {log_path}"
            )
    if stage == "scope":
        final_logs = run_root / "launcher_logs"
        final_logs.mkdir(parents=True, exist_ok=True)
        log_path.replace(final_logs / log_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot-only", action="store_true")
    mode.add_argument("--skip-pilot", action="store_true")
    args = parser.parse_args()
    config = _read_config()
    stages = ["scope"]
    if args.pilot_only:
        stages.append("pilot")
    else:
        if not args.skip_pilot:
            stages.append("pilot")
        stages.extend(["k", "c", "w", "a", "joint", "baselines"])
    for stage in stages:
        _run_stage(stage, args.shared.resolve(), args.run_root.resolve(), config)
    print(
        json.dumps(
            {
                "status": "DEVELOPMENT_COMPLETE",
                "stages": stages,
                "run_root": str(args.run_root.resolve()),
                "test_accessed": False,
                "ood_accessed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
