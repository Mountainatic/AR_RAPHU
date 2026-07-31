"""Immutable shared-dataset validation and causal view construction."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .gpu_common import sha256_array, sha256_file


@dataclass(frozen=True)
class SplitArrays:
    sample_id: np.ndarray
    sequence_u: np.ndarray
    sequence_y: np.ndarray
    sequence_y_centered: np.ndarray
    target_z: np.ndarray
    evaluation_mask: np.ndarray


@dataclass(frozen=True)
class DirectionData:
    name: str
    train: SplitArrays
    test: SplitArrays
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, axes: tuple[int, ...]) -> "Standardizer":
        mean = np.mean(values, axis=axes, keepdims=True, dtype=np.float64)
        scale = np.std(values, axis=axes, keepdims=True, dtype=np.float64)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "TargetScaler":
        mean = float(np.mean(values, dtype=np.float64))
        scale = float(np.std(values, dtype=np.float64))
        if scale < 1e-8:
            scale = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean


def _load_split(path: Path) -> SplitArrays:
    with np.load(path, allow_pickle=False) as stored:
        required = {
            "sample_id",
            "sequence_u",
            "sequence_y",
            "sequence_y_centered",
            "target_z",
            "evaluation_mask",
        }
        missing = sorted(required.difference(stored.files))
        if missing:
            raise RuntimeError(f"MISSING_SEQUENCE_ARRAYS:{path}:{missing}")
        arrays = {name: stored[name].copy() for name in required}
    n = len(arrays["sample_id"])
    for name, value in arrays.items():
        if len(value) != n:
            raise RuntimeError(f"ROW_COUNT_MISMATCH:{path}:{name}:{len(value)}:{n}")
    if arrays["sequence_u"].ndim != 3 or arrays["sequence_u"].shape[2] != 4:
        raise RuntimeError(f"INVALID_SEQUENCE_U:{arrays['sequence_u'].shape}")
    if arrays["sequence_y"].shape != arrays["sequence_y_centered"].shape:
        raise RuntimeError("SEQUENCE_Y_SHAPE_MISMATCH")
    if arrays["sequence_y"].shape[:2] != arrays["sequence_u"].shape[:2]:
        raise RuntimeError("U_Y_SEQUENCE_LENGTH_MISMATCH")
    return SplitArrays(
        sample_id=arrays["sample_id"],
        sequence_u=arrays["sequence_u"].astype(np.float32, copy=False),
        sequence_y=arrays["sequence_y"].astype(np.float32, copy=False),
        sequence_y_centered=arrays["sequence_y_centered"].astype(np.float32, copy=False),
        target_z=arrays["target_z"].astype(np.float64, copy=False),
        evaluation_mask=arrays["evaluation_mask"].astype(bool, copy=False),
    )


def resolve_shared_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = [root, root / "shared", root / "SHARED_BENCHMARK_DATASET"]
    for candidate in candidates:
        if (candidate / "DATA_AND_SPLIT_HASHES.json").is_file():
            return candidate
    raise FileNotFoundError(f"SHARED_ROOT_NOT_FOUND:{root}")


def validate_shared_dataset(path: str | Path) -> dict[str, Any]:
    root = resolve_shared_root(path)
    manifest_path = root / "DATA_AND_SPLIT_HASHES.json"
    protocol_path = root / "BENCHMARK_PROTOCOL.json"
    if not protocol_path.is_file():
        raise RuntimeError("BENCHMARK_PROTOCOL_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if not protocol.get("frozen", False):
        problems.append("PROTOCOL_NOT_FROZEN")
    if not protocol.get("regeneration_by_gpu_forbidden", False):
        problems.append("GPU_REGENERATION_GUARD_MISSING")
    for record in manifest.get("files", []):
        file_path = root / record["file"]
        if not file_path.is_file():
            problems.append(f"MISSING:{record['file']}")
            continue
        if sha256_file(file_path) != record["file_sha256"]:
            problems.append(f"FILE_HASH:{record['file']}")
            continue
        with np.load(file_path, allow_pickle=False) as stored:
            for name, expected in record.get("arrays", {}).items():
                if name not in stored.files:
                    problems.append(f"MISSING_ARRAY:{record['file']}:{name}")
                    continue
                value = stored[name]
                if list(value.shape) != expected["shape"]:
                    problems.append(f"SHAPE:{record['file']}:{name}")
                if str(value.dtype) != expected["dtype"]:
                    problems.append(f"DTYPE:{record['file']}:{name}")
                if sha256_array(value) != expected["sha256"]:
                    problems.append(f"ARRAY_HASH:{record['file']}:{name}")
    forbidden = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".xlsx", ".xls"}
    ]
    problems.extend(f"FORBIDDEN_RAW_DATA:{name}" for name in forbidden)
    directions: dict[str, Any] = {}
    for direction_name in sorted(manifest.get("directions", {})):
        try:
            direction = load_direction(root, direction_name, validate=False)
            directions[direction_name] = {
                "train_samples": len(direction.train.sample_id),
                "test_samples": len(direction.test.sample_id),
                "sequence_shape": list(direction.train.sequence_u.shape[1:]),
                "evaluation_samples": int(direction.test.evaluation_mask.sum()),
            }
        except Exception as exc:  # fail closed, report all problems
            problems.append(f"DIRECTION_LOAD:{direction_name}:{type(exc).__name__}:{exc}")
    return {
        "status": "PASS" if not problems else "FAIL",
        "shared_root": str(root),
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_file(manifest_path),
        "files_checked": len(manifest.get("files", [])),
        "directions": directions,
        "problems": problems,
    }


def load_direction(
    shared_root: str | Path,
    direction_name: str,
    *,
    validate: bool = True,
) -> DirectionData:
    root = resolve_shared_root(shared_root)
    if validate:
        report = validate_shared_dataset(root)
        if report["status"] != "PASS":
            raise RuntimeError(f"SHARED_VALIDATION_FAILED:{report['problems']}")
    direction_root = root / direction_name
    metadata = json.loads((direction_root / "metadata.json").read_text(encoding="utf-8"))
    train = _load_split(direction_root / "sequence_view" / "train.npz")
    test = _load_split(direction_root / "sequence_view" / "test.npz")
    if len(np.unique(train.sample_id)) != len(train.sample_id):
        raise RuntimeError(f"DUPLICATE_TRAIN_SAMPLE_ID:{direction_name}")
    if len(np.unique(test.sample_id)) != len(test.sample_id):
        raise RuntimeError(f"DUPLICATE_TEST_SAMPLE_ID:{direction_name}")
    return DirectionData(direction_name, train, test, metadata)


def list_directions(shared_root: str | Path) -> list[str]:
    root = resolve_shared_root(shared_root)
    manifest = json.loads((root / "DATA_AND_SPLIT_HASHES.json").read_text(encoding="utf-8"))
    return sorted(manifest["directions"])


def make_model_view(split: SplitArrays, mode: str) -> np.ndarray:
    if mode == "input":
        return split.sequence_u.astype(np.float32, copy=False)
    if mode == "dynamic":
        return np.concatenate(
            (split.sequence_u, split.sequence_y_centered[..., None]), axis=-1
        ).astype(np.float32, copy=False)
    raise ValueError(f"UNKNOWN_MODE:{mode}")


def chronological_folds(
    n_rows: int,
    inner_folds: list[list[float]],
    purge_rows: int,
    *,
    minimum_train: int = 64,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_rows, dtype=np.int64)
    for train_fraction, validation_stop in inner_folds:
        train_stop = int(np.floor(n_rows * float(train_fraction))) - purge_rows
        valid_start = int(np.floor(n_rows * float(train_fraction)))
        valid_stop = int(np.floor(n_rows * float(validation_stop)))
        if train_stop < minimum_train or valid_stop <= valid_start:
            continue
        yield indices[:train_stop], indices[valid_start:valid_stop]


def simple_train_validation_split(
    n_rows: int,
    *,
    validation_fraction: float,
    purge_rows: int,
    minimum_train: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    valid_start = int(np.floor(n_rows * (1.0 - validation_fraction)))
    train_stop = valid_start - purge_rows
    if train_stop < minimum_train:
        raise RuntimeError(
            f"INSUFFICIENT_TRAIN_ROWS:{n_rows}:{train_stop}:{minimum_train}"
        )
    return (
        np.arange(train_stop, dtype=np.int64),
        np.arange(valid_start, n_rows, dtype=np.int64),
    )


def _first_existing(stored: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in stored.files:
            return stored[key].copy()
    return None


def load_k_predictions(
    cpu_results_root: str | Path,
    direction: DirectionData,
) -> tuple[np.ndarray, np.ndarray]:
    """Load frozen K train-OOF and test predictions without fitting K on GPU.

    The loader accepts several historical CPU artifact schemas. It fails closed
    when a train OOF prediction is unavailable; using in-sample K residuals
    would violate the frozen protocol.
    """
    root = Path(cpu_results_root).expanduser().resolve()
    candidates = [
        root / "CPU_MODEL_PREDICTIONS" / direction.name / "K-only.npz",
        root / direction.name / "K-only.npz",
        root / direction.name / "K_ONLY.npz",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"K_PREDICTION_ARTIFACT_NOT_FOUND:{candidates}")
    with np.load(path, allow_pickle=False) as stored:
        train_id = _first_existing(
            stored,
            ("train_sample_id", "oof_sample_id", "sample_id_train"),
        )
        train_prediction = _first_existing(
            stored,
            (
                "train_oof_prediction",
                "oof_prediction",
                "train_y_pred_oof",
                "y_pred_train_oof",
            ),
        )
        test_id = _first_existing(
            stored,
            ("test_sample_id", "sample_id_test", "sample_id"),
        )
        test_prediction = _first_existing(
            stored,
            ("test_prediction", "y_pred_test", "prediction", "y_pred"),
        )
    if train_id is None or train_prediction is None:
        raise RuntimeError(
            "K_TRAIN_OOF_MISSING:GPU residual models are skipped rather than "
            "using leakage-prone in-sample K residuals"
        )
    if test_id is None or test_prediction is None:
        raise RuntimeError("K_TEST_PREDICTION_MISSING")
    train_map = {str(k): float(v) for k, v in zip(train_id, train_prediction)}
    test_map = {str(k): float(v) for k, v in zip(test_id, test_prediction)}
    missing_train = [str(k) for k in direction.train.sample_id if str(k) not in train_map]
    missing_test = [str(k) for k in direction.test.sample_id if str(k) not in test_map]
    if missing_train or missing_test:
        raise RuntimeError(
            f"K_SAMPLE_ID_MISMATCH:train={len(missing_train)}:test={len(missing_test)}"
        )
    return (
        np.asarray([train_map[str(k)] for k in direction.train.sample_id], dtype=np.float64),
        np.asarray([test_map[str(k)] for k in direction.test.sample_id], dtype=np.float64),
    )


def matured_residual_history(
    target: np.ndarray,
    k_prediction: np.ndarray,
    *,
    maturity_rows: int,
    history_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(k_prediction, dtype=np.float64)
    n = len(residual)
    history = np.zeros((n, history_rows, 1), dtype=np.float32)
    available = np.zeros(n, dtype=bool)
    for row in range(n):
        right = row - maturity_rows + 1
        left = right - history_rows
        if left < 0 or right <= 0:
            continue
        window = residual[left:right]
        # Rolling OOF predictions are intentionally unavailable before the
        # first expanding validation block.  A residual row is usable only
        # when both its target correction and every matured history value are
        # finite; silently replacing unavailable OOF values would leak.
        if not np.isfinite(residual[row]) or not np.all(np.isfinite(window)):
            continue
        history[row, :, 0] = window.astype(np.float32)
        available[row] = True
    return history, available
