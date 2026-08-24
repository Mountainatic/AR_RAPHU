from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
C1_CONFIG = PROJECT / "configs/representative_horizon_stage1_tep_sru_c1_tasks.json"
REPOSITORY_ROOT = PROJECT.parent

RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "pyarrow",
    "pyreadr",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "xgboost",
    "nfoursid",
    "torch",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_audit() -> dict[str, Any]:
    if os.environ.get("AR_RAPHU_RUNTIME_MANAGER") != "uv":
        raise RuntimeError("AR_RAPHU_RUNTIME_MANAGER must be exactly 'uv'")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            f"representative Stage1 requires Python 3.10, got {sys.version.split()[0]}"
        )

    virtual_environment = os.environ.get("VIRTUAL_ENV")
    if not virtual_environment:
        raise RuntimeError("VIRTUAL_ENV is missing; launch through uv run --frozen")
    executable = Path(sys.executable).absolute()
    resolved_executable = Path(sys.executable).resolve()
    environment_root = Path(virtual_environment).resolve()
    interpreter_prefix = Path(sys.prefix).resolve()
    if interpreter_prefix != environment_root:
        raise RuntimeError(
            "interpreter prefix does not match VIRTUAL_ENV: "
            f"{interpreter_prefix} vs {environment_root}"
        )

    uv_executable_value = os.environ.get("AR_RAPHU_UV_EXECUTABLE")
    if not uv_executable_value:
        raise RuntimeError("AR_RAPHU_UV_EXECUTABLE is required")
    uv_executable = Path(uv_executable_value).resolve()
    if not uv_executable.is_file():
        raise RuntimeError(f"uv executable is missing: {uv_executable}")
    uv_version = subprocess.run(
        [str(uv_executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    lock_path = REPOSITORY_ROOT / "uv.lock"
    project_path = REPOSITORY_ROOT / "pyproject.toml"
    if not lock_path.is_file() or not project_path.is_file():
        raise RuntimeError("pyproject.toml/uv.lock is missing from repository root")
    packages = {
        name: importlib.metadata.version(name) for name in RUNTIME_DISTRIBUTIONS
    }
    commit = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "status": "PASS",
        "runtime_manager": "uv",
        "python_executable": str(executable),
        "python_executable_resolved": str(resolved_executable),
        "python_prefix": str(interpreter_prefix),
        "python_version": sys.version,
        "virtual_environment": str(environment_root),
        "uv_executable": str(uv_executable),
        "uv_version": uv_version,
        "uv_sha256": _sha256(uv_executable),
        "pyproject_sha256": _sha256(project_path),
        "uv_lock_sha256": _sha256(lock_path),
        "git_commit": commit,
        "package_versions": packages,
    }


def _run(
    command: list[str], log_path: Path, env: dict[str, str]
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc()}] RUN {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fresh TEP/SRU representative C1 and run the formal development phase."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--per-worker-gib", type=float, default=4.0)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    shared = run_root / "shared"
    status_path = run_root / "logs" / "LAUNCH_STATUS.json"
    try:
        runtime_audit = _runtime_audit()
    except Exception as error:
        _write_json(
            status_path,
            {
                "schema_version": 1,
                "protocol": "REPRESENTATIVE_STAGE1_TEP_SRU_CZ_L256_FORMAL_V1",
                "status": "FAILED_RUNTIME_PREFLIGHT",
                "failed_utc": _utc(),
                "run_root": str(run_root),
                "formal_protocol_development_phase": True,
                "model_fitting_started": False,
                "test_accessed": False,
                "ood_accessed": False,
                "global_freeze_created": False,
                "error": str(error),
                "python_executable": sys.executable,
                "python_version": sys.version,
            },
        )
        raise
    status: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "REPRESENTATIVE_STAGE1_TEP_SRU_CZ_L256_FORMAL_V1",
        "status": "RUNNING",
        "started_utc": _utc(),
        "run_root": str(run_root),
        "shared": str(shared),
        "formal_protocol_development_phase": True,
        "neural_in_scope": False,
        "test_accessed": False,
        "ood_accessed": False,
        "global_freeze_created": False,
        "runtime_environment": runtime_audit,
        "stages": {},
    }
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("test_accessed") is not False or previous.get(
            "ood_accessed"
        ) is not False:
            raise RuntimeError("existing partial run reports test/OOD access")
        status["first_started_utc"] = previous.get(
            "first_started_utc", previous.get("started_utc")
        )
        status["resumed_utc"] = _utc()
    else:
        status["first_started_utc"] = status["started_utc"]

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT / "src"), existing_pythonpath]
    ).rstrip(os.pathsep)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    runtime_audit["thread_environment"] = {
        name: env[name]
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    _write_json(run_root / "logs" / "RUNTIME_ENVIRONMENT.json", runtime_audit)
    _write_json(status_path, status)

    try:
        if not (shared / "TASK_REGISTRY.json").is_file():
            if shared.exists():
                raise RuntimeError(
                    "incomplete fresh C1 output exists; use a new run root"
                )
            status["stages"]["C1_BUILD"] = {
                "status": "RUNNING",
                "started_utc": _utc(),
            }
            _write_json(status_path, status)
            _run(
                [
                    sys.executable,
                    str(PROJECT / "scripts/build_shared_data.py"),
                    "--raw-root",
                    str(args.raw_root.resolve()),
                    "--registry-root",
                    str(args.registry_root.resolve()),
                    "--config",
                    str(C1_CONFIG),
                    "--output",
                    str(shared),
                ],
                run_root / "logs" / "C1_BUILD.log",
                env,
            )
            status["stages"]["C1_BUILD"]["status"] = "PASS"
            status["stages"]["C1_BUILD"]["completed_utc"] = _utc()
            _write_json(status_path, status)

        status["stages"]["C1_VALIDATE"] = {
            "status": "RUNNING",
            "started_utc": _utc(),
        }
        _write_json(status_path, status)
        _run(
            [
                sys.executable,
                str(PROJECT / "scripts/validate_shared_data.py"),
                "--shared",
                str(shared),
                "--output",
                str(run_root / "logs" / "C1_VALIDATION.json"),
            ],
            run_root / "logs" / "C1_VALIDATE.log",
            env,
        )
        status["stages"]["C1_VALIDATE"]["status"] = "PASS"
        status["stages"]["C1_VALIDATE"]["completed_utc"] = _utc()
        _write_json(status_path, status)

        status["stages"]["CPU_DEVELOPMENT"] = {
            "status": "RUNNING",
            "started_utc": _utc(),
        }
        _write_json(status_path, status)
        _run(
            [
                sys.executable,
                str(
                    PROJECT
                    / "scripts/run_representative_stage1_tep_sru_cpu.py"
                ),
                "all",
                "--shared",
                str(shared),
                "--project",
                str(PROJECT),
                "--run-root",
                str(run_root),
                "--workers",
                str(max(1, args.workers)),
                "--per-worker-gib",
                str(args.per_worker_gib),
            ],
            run_root / "logs" / "CPU_DEVELOPMENT.log",
            env,
        )
        status["stages"]["CPU_DEVELOPMENT"]["status"] = "PASS"
        status["stages"]["CPU_DEVELOPMENT"]["completed_utc"] = _utc()
        status["status"] = "PARTIAL_DEVELOPMENT_CPU_ONLY"
        status["completed_utc"] = _utc()
        _write_json(status_path, status)
    except Exception as error:
        status["status"] = "FAILED"
        status["error"] = str(error)
        status["failed_utc"] = _utc()
        _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
