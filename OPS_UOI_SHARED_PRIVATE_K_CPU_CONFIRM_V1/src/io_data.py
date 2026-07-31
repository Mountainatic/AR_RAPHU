from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def safe_extract(bundle: Path, destination: Path) -> Path:
    marker = destination / ".complete"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"UNSAFE_ZIP_MEMBER:{member.filename}")
        archive.extractall(destination)
    marker.write_text(sha256_file(bundle) + "\n", encoding="utf-8")
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
    train_tab = _load_npz(root / "multiresolution_tabular_view" / "train.npz")
    test_tab = _load_npz(root / "multiresolution_tabular_view" / "test.npz")
    train.update(train_tab)
    test.update(test_tab)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return DirectionData(direction, train, test, metadata)


def load_protocol(shared_root: Path) -> dict[str, Any]:
    return json.loads(
        (shared_root / "BENCHMARK_PROTOCOL.json").read_text(encoding="utf-8")
    )


def inner_folds(
    origin_raw_index: np.ndarray,
    fold_specs: list[list[float]],
    *,
    purge_raw_samples: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    origins = np.asarray(origin_raw_index, dtype=np.int64)
    n_rows = len(origins)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_fraction, validation_end_fraction in fold_specs:
        validation_start = int(np.floor(float(train_fraction) * n_rows))
        validation_end = int(np.floor(float(validation_end_fraction) * n_rows))
        if not (0 < validation_start < validation_end <= n_rows):
            raise RuntimeError("INVALID_INNER_FOLD")
        first_validation_origin = int(origins[validation_start])
        training = np.flatnonzero(
            origins < first_validation_origin - int(purge_raw_samples)
        )
        validation = np.arange(validation_start, validation_end, dtype=np.int64)
        if len(training) == 0 or len(validation) == 0:
            raise RuntimeError("EMPTY_INNER_FOLD")
        if int(origins[training].max()) + purge_raw_samples >= int(
            origins[validation].min()
        ):
            raise RuntimeError("PURGE_FAILED")
        folds.append((training, validation))
    return folds


def prediction_files(root: Path) -> dict[tuple[str, str], Path]:
    output: dict[tuple[str, str], Path] = {}
    for path in root.rglob("*.npz"):
        direction = next(
            (name for name in DIRECTIONS if name in path.parts), None
        )
        if direction is None:
            continue
        try:
            with np.load(path, allow_pickle=False) as stored:
                keys = set(stored.files)
        except Exception:
            continue
        if {"sample_id", "prediction", "target_z", "evaluation_mask"} <= keys:
            output[(direction, path.stem)] = path
    return output


def _normal_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def resolve_prediction(
    index: dict[tuple[str, str], Path],
    direction: str,
    requested_name: str,
) -> Path:
    exact = index.get((direction, requested_name))
    if exact is not None:
        return exact
    wanted = _normal_name(requested_name)
    candidates = [
        path
        for (candidate_direction, name), path in index.items()
        if candidate_direction == direction
        and (
            _normal_name(name) == wanted
            or wanted in _normal_name(name)
            or _normal_name(name) in wanted
        )
    ]
    if len(candidates) != 1:
        available = sorted(
            name for candidate_direction, name in index if candidate_direction == direction
        )
        raise RuntimeError(
            f"PREDICTION_NOT_UNIQUE:{direction}:{requested_name}:"
            f"{[str(path) for path in candidates]}:AVAILABLE={available}"
        )
    return candidates[0]


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    payload = _load_npz(path)
    return {
        "sample_id": np.asarray(payload["sample_id"]).astype("U"),
        "prediction": np.asarray(payload["prediction"], dtype=np.float64),
        "target_z": np.asarray(payload["target_z"], dtype=np.float64),
        "evaluation_mask": np.asarray(payload["evaluation_mask"], dtype=bool),
    }


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = target - prediction
    mse = float(np.mean(error * error))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - float(np.sum(error * error)) / denominator if denominator else float("nan")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "rows": int(len(target))}


def prediction_metrics(payload: dict[str, np.ndarray]) -> dict[str, float]:
    mask = payload["evaluation_mask"]
    return metrics(payload["target_z"][mask], payload["prediction"][mask])


def pooled_metrics(payloads: list[dict[str, np.ndarray]]) -> dict[str, float]:
    targets = np.concatenate(
        [payload["target_z"][payload["evaluation_mask"]] for payload in payloads]
    )
    predictions = np.concatenate(
        [payload["prediction"][payload["evaluation_mask"]] for payload in payloads]
    )
    return metrics(targets, predictions)


def _read_leaderboards(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in root.rglob("*.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    row = dict(row)
                    row["__source__"] = str(path)
                    rows.append(row)
        except (UnicodeDecodeError, csv.Error):
            continue
    return rows


def published_mse(root: Path, model_name: str) -> float | None:
    wanted = _normal_name(model_name)
    candidates: list[float] = []
    for row in _read_leaderboards(root):
        source_parts = Path(row.get("__source__", "")).parts
        if any(key.lower() == "model_id" for key in row):
            if Path(row.get("__source__", "")).name != "GPU_FINALISTS.csv":
                continue
            if "ABLATIONS" in source_parts:
                continue
        direction_value = next(
            (
                row[key]
                for key in row
                if key.lower() == "direction" and row.get(key)
            ),
            None,
        )
        if direction_value is not None and direction_value.upper() != "POOLED":
            continue
        name_value = next(
            (
                row[key]
                for key in row
                if key.lower() in {"name", "model", "model_name", "model_id"}
                and row.get(key)
            ),
            None,
        )
        if name_value is None:
            continue
        normalized = _normal_name(name_value)
        if not (
            normalized == wanted
            or wanted in normalized
            or normalized in wanted
        ):
            continue
        for key, value in row.items():
            if key.lower() in {
                "pooled_mse",
                "mse",
                "pooled_mse_seed_median",
            } and value not in (None, ""):
                try:
                    candidates.append(float(value))
                except ValueError:
                    pass
    if not candidates:
        return None
    unique = sorted(set(round(value, 15) for value in candidates))
    if len(unique) > 1:
        # Prefer the value repeated across final leaderboard files.
        counts = {
            value: sum(abs(candidate - value) <= 1e-14 for candidate in candidates)
            for value in unique
        }
        return float(max(counts, key=counts.get))
    return float(unique[0])


def _gpu_model_id(model_name: str) -> str:
    normalized = _normal_name(model_name)
    if normalized == _normal_name("NLinear-U"):
        return "final__nlinear_u"
    if normalized == _normal_name("Temporal Autoencoder"):
        return "final__temporal_autoencoder_uxy"
    raise KeyError(f"UNKNOWN_GPU_REFERENCE:{model_name}")


def load_gpu_seed_ensemble(
    gpu_root: Path,
    direction: str,
    model_name: str,
) -> tuple[dict[str, np.ndarray], list[float]]:
    model_id = _gpu_model_id(model_name)
    root = gpu_root / "results_gpu/tasks/finalists" / direction / model_id
    paths = sorted(root.glob("seed_*/predictions.npz"))
    if not paths:
        raise RuntimeError(
            f"GPU_FINALIST_PREDICTIONS_MISSING:{direction}:{model_id}"
        )
    sample_id = None
    target = None
    mask = None
    predictions = []
    seed_mse = []
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            current_id = np.asarray(stored["sample_id"]).astype("U")
            current_target = np.asarray(stored["y_true"], dtype=np.float64)
            current_prediction = np.asarray(stored["y_pred"], dtype=np.float64)
            current_mask = np.asarray(stored["evaluation_mask"], dtype=bool)
        if sample_id is None:
            sample_id = current_id
            target = current_target
            mask = current_mask
        elif (
            not np.array_equal(sample_id, current_id)
            or not np.array_equal(target, current_target)
            or not np.array_equal(mask, current_mask)
        ):
            raise RuntimeError(
                f"GPU_SEED_ALIGNMENT_MISMATCH:{direction}:{model_id}:{path}"
            )
        predictions.append(current_prediction)
        error = current_target[current_mask] - current_prediction[current_mask]
        seed_mse.append(float(np.mean(error * error)))
    ensemble_prediction = np.median(
        np.stack(predictions, axis=0), axis=0
    ).astype(np.float64)
    return (
        {
            "sample_id": sample_id,
            "prediction": ensemble_prediction,
            "target_z": target,
            "evaluation_mask": mask,
        },
        seed_mse,
    )


def copy_plan(plan_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan_path, destination)
