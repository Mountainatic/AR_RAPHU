"""Shared provenance, hashing, atomic output, metrics, and bootstrap helpers."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def load_json(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8")), sha256(raw).hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, target)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("status\nNO_ROWS\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    residual = y - p
    mse = float(np.mean(residual**2))
    variance = float(np.sum((y - y.mean()) ** 2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": float(1.0 - np.sum(residual**2) / max(variance, np.finfo(float).eps)),
    }


def improvement(reference_loss: np.ndarray, model_loss: np.ndarray) -> float:
    return float(
        1.0
        - np.mean(np.asarray(model_loss, dtype=np.float64))
        / max(np.mean(np.asarray(reference_loss, dtype=np.float64)), np.finfo(float).eps)
    )


def circular_block_bootstrap(
    reference_loss: np.ndarray,
    model_loss: np.ndarray,
    *,
    replicates: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    ref = np.asarray(reference_loss, dtype=np.float64)
    mod = np.asarray(model_loss, dtype=np.float64)
    if len(ref) != len(mod) or len(ref) == 0:
        raise ValueError("INVALID_BOOTSTRAP_LOSSES")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    blocks = int(np.ceil(len(ref) / block_length))
    offsets = np.arange(block_length)
    for replicate in range(replicates):
        starts = rng.integers(0, len(ref), size=blocks)
        index = ((starts[:, None] + offsets) % len(ref)).ravel()[: len(ref)]
        values[replicate] = improvement(ref[index], mod[index])
    return {
        "replicates": int(replicates),
        "block_length": int(block_length),
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "upper_95": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
    }


def pooled_two_direction_bootstrap(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    replicates: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    offsets = np.arange(block_length)
    for replicate in range(replicates):
        ref_sum = 0.0
        model_sum = 0.0
        for reference, model in pairs:
            reference = np.asarray(reference, dtype=np.float64)
            model = np.asarray(model, dtype=np.float64)
            blocks = int(np.ceil(len(reference) / block_length))
            starts = rng.integers(0, len(reference), size=blocks)
            index = (
                (starts[:, None] + offsets) % len(reference)
            ).ravel()[: len(reference)]
            ref_sum += float(np.sum(reference[index]))
            model_sum += float(np.sum(model[index]))
        values[replicate] = 1.0 - model_sum / max(ref_sum, np.finfo(float).eps)
    return {
        "replicates": int(replicates),
        "block_length": int(block_length),
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "upper_95": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
    }


def environment_snapshot(repo_root: str | Path) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "sklearn", "statsmodels", "xgboost", "psutil"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "UNKNOWN")
        except ImportError:
            packages[name] = "NOT_AVAILABLE"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "NOT_AVAILABLE"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "git_commit": commit,
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "timestamp_unix": time.time(),
    }
