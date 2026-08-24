from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "scripts" / "run_cz_only_pilot.py"
STAGES = ("scope", "preflight", "development", "freeze", "checkpoints", "test")
PILOT_CODE_FILES = (
    PROJECT / "src" / "prism_benchmark" / "cz_only_pilot.py",
    PROJECT / "src" / "prism_benchmark" / "representative_formal.py",
    PROJECT / "src" / "prism_benchmark" / "cz_l256_nowcast.py",
    PROJECT / "src" / "prism_benchmark" / "cz_k_support.py",
    PROJECT / "src" / "prism_benchmark" / "portable_checkpoints.py",
    PROJECT / "src" / "prism_benchmark" / "representative_baseline_checkpoints.py",
    PROJECT / "src" / "prism_benchmark" / "representative_prism_checkpoints.py",
    PROJECT / "scripts" / "run_cz_only_pilot.py",
    Path(__file__).resolve(),
    PROJECT / "configs" / "representative_horizon_stage1_tep_sru_cpu.json",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _old_recovery_running() -> bool:
    if os.name == "nt":
        return False
    completed = subprocess.run(
        ["pgrep", "-f", "parallel_final_recovery.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _input_manifest(args: argparse.Namespace) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(PROJECT.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(PROJECT.parent), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("STOP_CZ_PILOT_PROJECT_WORKTREE_NOT_CLEAN")
    dependencies = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    code_inventory = [
        {
            "path": path.relative_to(PROJECT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in PILOT_CODE_FILES
    ]
    return {
        "status": "CZ_PILOT_INPUTS_SEALED_BEFORE_RUN",
        "created_utc": _utc(),
        "source_commit": commit,
        "worktree_clean": True,
        "python": sys.version,
        "dependency_lock": dependencies,
        "dependency_lock_sha256": hashlib.sha256(
            "\n".join(dependencies).encode("utf-8")
        ).hexdigest(),
        "code_inventory": code_inventory,
        "code_inventory_hash": hashlib.sha256(
            "".join(item["sha256"] for item in code_inventory).encode("ascii")
        ).hexdigest(),
        "cz_raw": {
            "path": str(args.raw_cz),
            "bytes": args.raw_cz.stat().st_size,
            "sha256": _sha256(args.raw_cz),
        },
        "old_recovery_running_at_launch": bool(args.old_recovery_running_at_launch),
        "concurrent_old_recovery_explicitly_authorized": bool(
            args.allow_concurrent_old_recovery
        ),
        "excluded_from_full_formal_selection": True,
    }


def _run_stage(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RUNNER),
        stage,
        "--project",
        str(PROJECT),
        "--run-root",
        str(args.run_root),
        "--raw-cz",
        str(args.raw_cz),
    ]
    environment = os.environ.copy()
    if stage == "test":
        environment["PRISM_FORMAL_INFERENCE_ONLY"] = "1"
    log = args.run_root / "logs" / f"CZ_PILOT_{stage.upper()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        while process.poll() is None:
            free_gib = shutil.disk_usage(args.run_root.parent).free / (1024**3)
            if free_gib < 5.0:
                _terminate(process)
                raise RuntimeError(
                    f"STOP_LOW_STORAGE_SAFE_CZ_PILOT_STAGE_TERMINATION:{stage}:{free_gib:.3f}GiB"
                )
            time.sleep(30)
    if process.returncode != 0:
        raise RuntimeError(f"CZ_PILOT_STAGE_FAILED:{stage}:exit={process.returncode}:log={log}")
    return {"stage": stage, "status": "PASS", "completed_utc": _utc(), "log": str(log)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential isolated CZ L256 pilot runner.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-cz", type=Path, required=True)
    parser.add_argument("--allow-concurrent-old-recovery", action="store_true")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.raw_cz = args.raw_cz.resolve()
    if not args.raw_cz.is_file():
        raise RuntimeError("--raw-cz must identify the frozen CZ workbook")
    args.old_recovery_running_at_launch = _old_recovery_running()
    if args.old_recovery_running_at_launch and not args.allow_concurrent_old_recovery:
        raise RuntimeError("STOP_OLD_CZ_NEURAL3_RECOVERY_STILL_RUNNING")
    if args.run_root.exists() and any(args.run_root.iterdir()):
        raise RuntimeError(f"refusing nonempty CZ pilot namespace: {args.run_root}")
    args.run_root.mkdir(parents=True, exist_ok=True)
    _write(args.run_root / "freeze" / "CZ_PILOT_INPUT_MANIFEST.json", _input_manifest(args))
    status_path = args.run_root / "logs" / "CZ_PILOT_LAUNCH_STATUS.json"
    state: dict[str, Any] = {
        "status": "RUNNING",
        "started_utc": _utc(),
        "project": str(PROJECT),
        "run_root": str(args.run_root),
        "stages": [],
        "old_recovery_running_at_launch": bool(args.old_recovery_running_at_launch),
        "concurrent_old_recovery_explicitly_authorized": bool(
            args.allow_concurrent_old_recovery
        ),
        "excluded_from_full_formal_selection": True,
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "stage2_status": "NOT_RUN_BY_USER_SCOPE",
    }
    _write(status_path, state)
    try:
        for stage in STAGES:
            state["active_stage"] = stage
            _write(status_path, state)
            state["stages"].append(_run_stage(args, stage))
        state.update(
            {
                "status": "PASS",
                "active_stage": None,
                "completed_utc": _utc(),
                "test_accessed": True,
                "ood_accessed": False,
            }
        )
    except BaseException as error:
        state.update(
            {
                "status": "FAILED_SAFE_STOP",
                "failed_utc": _utc(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _write(status_path, state)
        raise
    _write(status_path, state)
    print(json.dumps(state, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
