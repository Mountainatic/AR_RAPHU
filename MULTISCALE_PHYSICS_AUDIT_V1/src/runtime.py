"""Configuration, provenance, atomic I/O, and checkpoint helpers."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path).resolve()
    raw = source.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    return config, sha256(raw).hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def git_commit(directory: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "NOT_AVAILABLE"


def environment_snapshot(project_root: str | Path) -> dict[str, Any]:
    try:
        import numpy
        import openpyxl
        import scipy

        packages = {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "openpyxl": openpyxl.__version__,
        }
    except ImportError:
        packages = {}
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "git_commit": git_commit(project_root),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def task_is_complete(
    path: str | Path,
    *,
    config_sha256: str,
    data_sha256: str,
    sample_period_sec: float,
) -> bool:
    target = Path(path)
    if not target.is_file():
        return False
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("status") in {"COMPLETED", "NOT_APPLICABLE"}
        and payload.get("config_sha256") == config_sha256
        and payload.get("data_sha256") == data_sha256
        and float(payload.get("sample_period_sec", -1.0))
        == float(sample_period_sec)
    )
