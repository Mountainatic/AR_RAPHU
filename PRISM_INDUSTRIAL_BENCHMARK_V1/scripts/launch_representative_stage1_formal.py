from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
RUNNER = PROJECT / "scripts" / "run_representative_stage1_formal.py"
STAGES = (
    "scope",
    "preflight",
    "public-development",
    "cz-development",
    "freeze",
    "checkpoints",
    "test",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    if root.is_file():
        return [{"path": root.name, "bytes": root.stat().st_size, "sha256": _sha256(root)}]
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _installed_dependency_inventory() -> list[str]:
    """Return a deterministic environment snapshot without requiring pip.

    uv-managed environments intentionally need not contain the pip module.  The
    authoritative resolver state is still sealed by pyproject.toml + uv.lock;
    this inventory records the distributions actually importable at launch.
    """
    installed = {
        f"{name}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if (name := distribution.metadata.get("Name"))
    }
    return sorted(installed, key=str.casefold)


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
        raise RuntimeError("STOP_FORMAL_PROJECT_WORKTREE_NOT_CLEAN")
    dependencies = _installed_dependency_inventory()
    repository_root = PROJECT.parent
    pyproject_path = repository_root / "pyproject.toml"
    uv_lock_path = repository_root / "uv.lock"
    protocol_files = [
        PROJECT / "configs" / "representative_horizon_stage1_tep_sru_cpu.json",
        PROJECT / "configs" / "representative_horizon_stage1_tep_sru_c1_tasks.json",
        PROJECT / "configs" / "REPRESENTATIVE_HORIZON_STAGE1_REGISTRY.json",
        PROJECT
        / "PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_PACKAGE"
        / "PRISM_V2_1_2_JOINT_OOF_PROTOCOL_CORRECTION_CONFIG.json",
        PROJECT
        / "PRISM_V2_1_1_JOINT_PREDICTIVE_STABILITY_PRACTICE_PACKAGE"
        / "PRISM_V2_1_1_JOINT_PREDICTIVE_STABILITY_PRACTICE_CONFIG.json",
        PROJECT
        / "PRISM_V2_1_1_SRU_IMPLEMENTATION_CORRECTION"
        / "PRISM_V2_1_1_SRU_CONFIG_PATCH.json",
        PROJECT / "configs" / "cpu_model_freeze_v1.json",
    ]
    return {
        "status": "SEALED_BEFORE_RUN",
        "created_utc": _utc(),
        "source_commit": commit,
        "worktree_clean": True,
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "runtime_manager": os.environ.get("AR_RAPHU_RUNTIME_MANAGER"),
        "virtual_environment": os.environ.get("VIRTUAL_ENV"),
        "uv_executable": os.environ.get("AR_RAPHU_UV_EXECUTABLE"),
        "pyproject_sha256": _sha256(pyproject_path),
        "uv_lock_sha256": _sha256(uv_lock_path),
        "dependency_lock": dependencies,
        "dependency_lock_sha256": hashlib.sha256(
            "\n".join(dependencies).encode("utf-8")
        ).hexdigest(),
        "protocol_files": [
            {
                "path": path.relative_to(PROJECT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in protocol_files
        ],
        "protocol_hash": hashlib.sha256(
            "".join(_sha256(path) for path in protocol_files).encode("ascii")
        ).hexdigest(),
        "public_raw_inventory": {
            "tep": _file_inventory(args.raw_public_root / "tep_rieth"),
            "sru": _file_inventory(args.raw_public_root / "sru"),
        },
        "registry_inventory": {
            "tep": _file_inventory(args.registry_root / "tep"),
            "sru": _file_inventory(args.registry_root / "sru"),
        },
        "cz_raw_inventory": _file_inventory(args.raw_cz),
        "old_recovery_running_at_launch": bool(args.old_recovery_running_at_launch),
        "concurrent_old_recovery_explicitly_authorized": bool(
            args.allow_concurrent_old_recovery
        ),
    }


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


def _run_stage(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RUNNER),
        stage,
        "--project",
        str(PROJECT),
        "--run-root",
        str(args.run_root),
        "--workers",
        str(args.workers),
        "--per-worker-gib",
        str(args.per_worker_gib),
    ]
    if args.raw_public_root is not None:
        command.extend(["--raw-public-root", str(args.raw_public_root)])
    if args.registry_root is not None:
        command.extend(["--registry-root", str(args.registry_root)])
    if args.raw_cz is not None:
        command.extend(["--raw-cz", str(args.raw_cz)])
    environment = os.environ.copy()
    if stage == "test":
        environment["PRISM_FORMAL_INFERENCE_ONLY"] = "1"
    log = args.run_root / "logs" / f"FORMAL_{stage.upper().replace('-', '_')}.log"
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
                    f"STOP_LOW_STORAGE_SAFE_STAGE_TERMINATION:{stage}:{free_gib:.3f}GiB"
                )
            time.sleep(30)
    if process.returncode != 0:
        raise RuntimeError(f"FORMAL_STAGE_FAILED:{stage}:exit={process.returncode}:log={log}")
    return {
        "stage": stage,
        "status": "PASS",
        "completed_utc": _utc(),
        "log": str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential formal TEP/SRU/CZ runner.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-public-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--raw-cz", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--per-worker-gib", type=float, default=4.0)
    parser.add_argument(
        "--allow-concurrent-old-recovery",
        action="store_true",
        help=(
            "Allow this formal run to start beside the unrelated legacy CZ/Neural3 "
            "recovery when the user has explicitly authorized sufficient compute."
        ),
    )
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.raw_public_root = args.raw_public_root.resolve()
    args.registry_root = args.registry_root.resolve()
    args.raw_cz = args.raw_cz.resolve()
    args.old_recovery_running_at_launch = _old_recovery_running()
    if args.old_recovery_running_at_launch and not args.allow_concurrent_old_recovery:
        raise RuntimeError("STOP_OLD_CZ_NEURAL3_RECOVERY_STILL_RUNNING")
    if args.run_root.exists() and any(args.run_root.iterdir()):
        raise RuntimeError(f"refusing nonempty formal run namespace: {args.run_root}")
    args.run_root.mkdir(parents=True, exist_ok=True)
    _write(args.run_root / "freeze" / "RUN_INPUT_MANIFEST.json", _input_manifest(args))
    status_path = args.run_root / "logs" / "FORMAL_LAUNCH_STATUS.json"
    state: dict[str, Any] = {
        "status": "RUNNING",
        "started_utc": _utc(),
        "project": str(PROJECT),
        "run_root": str(args.run_root),
        "stages": [],
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "stage2_status": "NOT_RUN_BY_USER_SCOPE",
        "old_recovery_running_at_launch": bool(args.old_recovery_running_at_launch),
        "concurrent_old_recovery_explicitly_authorized": bool(
            args.allow_concurrent_old_recovery
        ),
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
