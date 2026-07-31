from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DIRECTIONS = ("sheet1_to_sheet2", "sheet2_to_sheet1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def safe_extract(bundle: Path, destination: Path) -> Path:
    marker = destination / ".complete"
    digest = sha256_file(bundle)
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == digest:
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"UNSAFE_ZIP_MEMBER:{member.filename}")
        archive.extractall(destination)
    marker.write_text(digest + "\n", encoding="utf-8")
    return destination


def find_named_root(extracted: Path, name: str) -> Path:
    direct = extracted / name
    if direct.is_dir():
        return direct
    matches = [path for path in extracted.rglob(name) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"ROOT_NOT_UNIQUE:{name}:{len(matches)}")
    return matches[0]


@dataclass(frozen=True)
class DirectionData:
    name: str
    train: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def load_direction(shared_root: Path, direction: str) -> DirectionData:
    root = shared_root / direction
    train = _load_npz(root / "sequence_view" / "train.npz")
    test = _load_npz(root / "sequence_view" / "test.npz")
    train.update(_load_npz(root / "multiresolution_tabular_view" / "train.npz"))
    test.update(_load_npz(root / "multiresolution_tabular_view" / "test.npz"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return DirectionData(direction, train, test, metadata)


def load_protocol(shared_root: Path) -> dict[str, Any]:
    return json.loads((shared_root / "BENCHMARK_PROTOCOL.json").read_text(encoding="utf-8"))


def inner_folds(
    origin_raw_index: np.ndarray,
    fold_specs: list[list[float]],
    *,
    purge_raw_samples: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    origins = np.asarray(origin_raw_index, dtype=np.int64)
    n_rows = len(origins)
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for train_fraction, validation_end_fraction in fold_specs:
        start = int(np.floor(float(train_fraction) * n_rows))
        stop = int(np.floor(float(validation_end_fraction) * n_rows))
        first_validation = int(origins[start])
        training = np.flatnonzero(origins < first_validation - purge_raw_samples)
        validation = np.arange(start, stop, dtype=np.int64)
        if not len(training) or not len(validation):
            raise RuntimeError("EMPTY_INNER_FOLD")
        if int(origins[training].max()) + purge_raw_samples >= int(origins[validation].min()):
            raise RuntimeError("PURGE_FAILED")
        output.append((training, validation))
    return output


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = target - prediction
    mse = float(np.mean(error * error))
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(error))),
        "R2": 1.0 - float(np.sum(error * error)) / denominator if denominator else float("nan"),
        "rows": int(len(target)),
    }


def pooled_metrics(payloads: list[dict[str, np.ndarray]]) -> dict[str, float | int]:
    target = np.concatenate([item["target_z"][item["evaluation_mask"]] for item in payloads])
    prediction = np.concatenate([item["prediction"][item["evaluation_mask"]] for item in payloads])
    return metrics(target, prediction)


def load_cpu_prediction(cpu_root: Path, direction: str, model: str) -> dict[str, np.ndarray]:
    path = cpu_root / "results_cpu/CPU_MODEL_PREDICTIONS" / direction / f"{model}.npz"
    payload = _load_npz(path)
    return {
        "sample_id": np.asarray(payload["sample_id"]).astype("U"),
        "prediction": np.asarray(payload["prediction"], dtype=np.float64),
        "target_z": np.asarray(payload["target_z"], dtype=np.float64),
        "evaluation_mask": np.asarray(payload["evaluation_mask"], dtype=bool),
    }


def _gpu_id(model: str) -> str:
    return {
        "NLinear-U": "final__nlinear_u",
        "Temporal Autoencoder": "final__temporal_autoencoder_uxy",
    }[model]


def load_gpu_ensemble(
    gpu_root: Path, direction: str, model: str
) -> tuple[dict[str, np.ndarray], list[float]]:
    root = gpu_root / "results_gpu/tasks/finalists" / direction / _gpu_id(model)
    paths = sorted(root.glob("seed_*/predictions.npz"))
    if not paths:
        raise RuntimeError(f"GPU_PREDICTIONS_MISSING:{direction}:{model}")
    sample_id = target = mask = None
    predictions: list[np.ndarray] = []
    seed_mse: list[float] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            current_id = np.asarray(stored["sample_id"]).astype("U")
            current_target = np.asarray(stored["y_true"], dtype=np.float64)
            current_prediction = np.asarray(stored["y_pred"], dtype=np.float64)
            current_mask = np.asarray(stored["evaluation_mask"], dtype=bool)
        if sample_id is None:
            sample_id, target, mask = current_id, current_target, current_mask
        elif not (
            np.array_equal(sample_id, current_id)
            and np.array_equal(target, current_target)
            and np.array_equal(mask, current_mask)
        ):
            raise RuntimeError(f"GPU_SEED_ALIGNMENT_MISMATCH:{path}")
        predictions.append(current_prediction)
        seed_mse.append(float(np.mean((current_target[current_mask] - current_prediction[current_mask]) ** 2)))
    return {
        "sample_id": sample_id,
        "prediction": np.median(np.stack(predictions), axis=0).astype(np.float64),
        "target_z": target,
        "evaluation_mask": mask,
    }, seed_mse


def _normal(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def published_mse(root: Path, model: str) -> float:
    gpu_ids = {
        "NLinear-U": "final__nlinear_u",
        "Temporal Autoencoder": "final__temporal_autoencoder_uxy",
    }
    if model in gpu_ids:
        files = [path for path in root.rglob("GPU_FINALISTS.csv") if "ABLATIONS" not in path.parts]
        if len(files) != 1:
            raise RuntimeError(f"GPU_FINALISTS_NOT_UNIQUE:{len(files)}")
        with files[0].open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("model_id") == gpu_ids[model]:
                    return float(row["pooled_MSE_seed_median"])
        raise RuntimeError(f"PUBLISHED_MSE_NOT_FOUND:{model}")
    wanted = _normal(model)
    candidates: list[float] = []
    for path in root.rglob("*.csv"):
        if "ABLATIONS" in path.parts:
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (UnicodeDecodeError, csv.Error):
            continue
        for row in rows:
            direction = next((value for key, value in row.items() if key.lower() == "direction"), "")
            if direction and direction.upper() != "POOLED":
                continue
            name = next((value for key, value in row.items() if key.lower() in {"name", "model", "model_name", "model_id"}), "")
            if not name or not (_normal(name) == wanted or wanted in _normal(name) or _normal(name) in wanted):
                continue
            for key, value in row.items():
                if key.lower() in {"pooled_mse", "mse", "pooled_mse_seed_median"} and value:
                    try:
                        candidates.append(float(value))
                    except ValueError:
                        pass
    if not candidates:
        raise RuntimeError(f"PUBLISHED_MSE_NOT_FOUND:{model}")
    rounded = sorted(set(round(value, 15) for value in candidates))
    counts = {value: sum(abs(candidate - value) <= 1e-14 for candidate in candidates) for value in rounded}
    return float(max(counts, key=counts.get))


def gpu_published_summary(root: Path, model: str) -> dict[str, Any]:
    model_id = _gpu_id(model)
    files = [path for path in root.rglob("GPU_FINALISTS.csv") if "ABLATIONS" not in path.parts]
    if len(files) != 1:
        raise RuntimeError(f"GPU_FINALISTS_NOT_UNIQUE:{len(files)}")
    with files[0].open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("model_id") == model_id:
                return {
                    "pooled_mse": float(row["pooled_MSE_seed_median"]),
                    "pooled_rmse": float(row["pooled_RMSE_seed_median"]),
                    "direction_mse": {key: float(value) for key, value in json.loads(row["direction_MSE_json"]).items()},
                    "metric_basis": "published median pooled MSE across 10 seeds",
                }
    raise RuntimeError(f"GPU_FINALISTS_MODEL_MISSING:{model}")


def validate_alignment(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], label: str) -> None:
    for key in ("sample_id", "target_z", "evaluation_mask"):
        if not np.array_equal(reference[key], candidate[key]):
            raise RuntimeError(f"BASELINE_ALIGNMENT_FAILED:{label}:{key}")


def moving_block_bootstrap(
    baseline_errors: list[np.ndarray],
    candidate_errors: list[np.ndarray],
    *,
    block_rows: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    observed_base = float(np.mean(np.concatenate([value * value for value in baseline_errors])))
    observed_candidate = float(np.mean(np.concatenate([value * value for value in candidate_errors])))
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        base_sum = candidate_sum = 0.0
        total = 0
        for base, candidate in zip(baseline_errors, candidate_errors):
            n_rows = len(base)
            starts = rng.integers(0, max(n_rows - block_rows + 1, 1), size=int(np.ceil(n_rows / block_rows)))
            index = np.concatenate([np.arange(start, min(start + block_rows, n_rows)) for start in starts])[:n_rows]
            base_sum += float(np.sum(base[index] ** 2))
            candidate_sum += float(np.sum(candidate[index] ** 2))
            total += len(index)
        base_mse = base_sum / total
        candidate_mse = candidate_sum / total
        draws[replicate] = (base_mse - candidate_mse) / max(base_mse, 1e-30)
    return {
        "observed_relative_improvement": (observed_base - observed_candidate) / max(observed_base, 1e-30),
        "median_relative_improvement": float(np.median(draws)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "positive_probability": float(np.mean(draws > 0)),
    }
