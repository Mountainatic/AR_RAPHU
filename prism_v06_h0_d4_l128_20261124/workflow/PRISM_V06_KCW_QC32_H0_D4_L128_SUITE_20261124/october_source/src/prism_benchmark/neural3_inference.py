"""Inference-only loader and reporter for sealed cached Neural-3 models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from . import neural3 as historical_neural3
from . import neural3_cached as active3_cached
from .cpu_data import BaseAccessor, ViewSpec, input_columns as registered_input_columns
from .level_reconstruction import metric_bundle_delta_and_level
from .neural3 import Scaler, _predict, _set_target_column, build_model, native_support
from .neural_candidate_cache import (
    NeuralCandidateCache,
    atomic_write_json,
    sha256_file,
)
from .v211_support import SUPPORT_CONTRACT, load_native_samples, support_id_hash


INFERENCE_ENVIRONMENT = "PRISM_NEURAL3_INFERENCE_ONLY"


def _runtime_code_binding() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    paths = [
        root / "neural3.py",
        root / "neural3_cached.py",
        root / "neural_candidate_cache.py",
        root / "level_reconstruction.py",
        root / "v211_support.py",
    ]
    return {
        "format": "PRISM_NEURAL_RUNTIME_CODE_BINDING_V1",
        "files": [
            {
                "name": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }


def _forbidden(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise RuntimeError("STOP_INFERENCE_ONLY_REFUSES_FIT_REFIT_SELECT")


def install_inference_only_guard() -> None:
    if os.environ.get(INFERENCE_ENVIRONMENT) != "1":
        raise RuntimeError("STOP_NEURAL3_TEST_REQUIRES_INFERENCE_ONLY_ENVIRONMENT")
    for name in (
        "_fit_one",
        "_fit_fixed_epochs",
        "select_candidate",
        "select_candidate_histories",
        "materialize_model",
    ):
        if hasattr(historical_neural3, name):
            setattr(historical_neural3, name, _forbidden)
    # Also close the active namespace's training/materialization entry points;
    # inference must not be able to fit, refit, or select in-process.
    for name in (
        "prepare_candidate",
        "train_selection_candidate",
        "train_frozen_selection_checkpoint",
    ):
        if hasattr(active3_cached, name):
            setattr(active3_cached, name, _forbidden)
    # A test worker never creates an optimizer.  Guard the concrete optimizer
    # as defense in depth so an accidental local training loop fails closed.
    torch.optim.AdamW.step = _forbidden  # type: ignore[method-assign]


def require_sealed_artifact(path: Path, expected_status: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("status") != expected_status or value.get("sealed") is not True:
        raise RuntimeError(f"STOP_REQUIRED_SEALED_ARTIFACT_INVALID:{path}")
    return value


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prediction_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_verified_artifact(
    *,
    cache: NeuralCandidateCache,
    logical_candidate_id: str,
    candidate_dir: Path,
    expected: dict[str, Any],
    device: torch.device,
    code_commit: str,
) -> tuple[torch.nn.Module, Scaler, dict[str, Any], dict[str, Any]]:
    observed_dir = cache.candidate_dir(logical_candidate_id).resolve()
    candidate_dir = Path(candidate_dir).resolve()
    if observed_dir != candidate_dir:
        raise RuntimeError("STOP_SELECTED_CHECKPOINT_DIRECTORY_NOT_CACHE_BOUND")
    record = cache.validate_candidate(logical_candidate_id)
    if record.get("sealed") is not True or record.get("deletion_forbidden") is not True:
        raise RuntimeError("STOP_SELECTED_CANDIDATE_NOT_IMMUTABLY_SEALED")
    if record.get("record_hash") != expected.get("candidate_record_hash"):
        raise RuntimeError("STOP_SELECTED_CANDIDATE_RECORD_HASH_MISMATCH")
    candidate_manifest_path = candidate_dir / "CANDIDATE_MANIFEST.json"
    if sha256_file(candidate_manifest_path) != expected.get("candidate_manifest_sha256"):
        raise RuntimeError("STOP_SELECTED_CANDIDATE_MANIFEST_SHA_MISMATCH")
    if record.get("hashes", {}).get("config_hash") != expected.get("config_hash"):
        raise RuntimeError("STOP_SELECTED_CANDIDATE_CONFIG_HASH_MISMATCH")
    config = json.loads(
        (candidate_dir / "model_config.json").read_text(encoding="utf-8")
    )
    preprocessing = json.loads(
        (candidate_dir / "preprocessing.json").read_text(encoding="utf-8")
    )
    if config.get("code_commit") != code_commit:
        raise RuntimeError("STOP_CHECKPOINT_CODE_COMMIT_MISMATCH")
    if config.get("code_binding") != _runtime_code_binding():
        raise RuntimeError("STOP_CHECKPOINT_RUNTIME_CODE_BINDING_MISMATCH")
    with np.load(candidate_dir / "model_weights.npz", allow_pickle=False) as archive:
        state = {
            name: torch.from_numpy(np.array(archive[name], copy=True))
            for name in archive.files
        }
    with np.load(candidate_dir / "scaler.npz", allow_pickle=False) as archive:
        scaler = Scaler(
            np.asarray(archive["feature_mean"], dtype=np.float64),
            np.asarray(archive["feature_scale"], dtype=np.float64),
            float(np.asarray(archive["target_mean"]).reshape(-1)[0]),
            float(np.asarray(archive["target_scale"]).reshape(-1)[0]),
        )
    model = build_model(
        str(config["model"]), int(config["input_dim"]), str(config["capacity"])
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, config, preprocessing


def _serializable_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in bundle.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = f"NOT_DEFINED_{str(value).upper()}"
        result[key] = value
    return result


def predict_cached_ensemble(
    *,
    shared: Path,
    view: ViewSpec,
    checkpoint_dirs: Sequence[Path],
    split: str,
    output: Path,
    device: torch.device,
    global_selection_freeze_path: Path,
    candidate_manifest_path: Path,
    selected_checkpoint_manifest_path: Path,
    code_commit: str,
    direction: str | None = None,
    horizon_steps: int | None = None,
) -> dict[str, Any]:
    """Load one manifest-bound cached best weight and materialize test metrics."""

    if split != "test":
        raise ValueError("active3 formal inference is limited to test")
    install_inference_only_guard()
    global_freeze = require_sealed_artifact(
        global_selection_freeze_path, "GLOBAL_SELECTION_FROZEN"
    )
    candidate_manifest = require_sealed_artifact(
        candidate_manifest_path, "CANDIDATE_CHECKPOINTS_SEALED"
    )
    selected_manifest = require_sealed_artifact(
        selected_checkpoint_manifest_path, "SELECTED_CHECKPOINTS_SEALED"
    )
    if selected_manifest.get("global_selection_freeze_sha256") != sha256_file(
        global_selection_freeze_path
    ):
        raise RuntimeError("STOP_SELECTED_MANIFEST_GLOBAL_FREEZE_BINDING_MISMATCH")
    if selected_manifest.get("candidate_checkpoint_manifest_sha256") != sha256_file(
        candidate_manifest_path
    ):
        raise RuntimeError("STOP_SELECTED_MANIFEST_CANDIDATE_CACHE_BINDING_MISMATCH")
    if len(checkpoint_dirs) != 1:
        raise RuntimeError("STOP_CURRENT_NO_RETRAIN_PROTOCOL_REQUIRES_ONE_CHECKPOINT")
    requested_dir = Path(checkpoint_dirs[0]).resolve()
    entries = selected_manifest.get("selected_checkpoints")
    if not isinstance(entries, list) or len(entries) != 81:
        raise RuntimeError("STOP_SELECTED_CHECKPOINT_MANIFEST_SCOPE_COUNT_MISMATCH")
    matching = [
        item
        for item in entries
        if Path(str(item.get("candidate_dir", ""))).resolve() == requested_dir
    ]
    if len(matching) != 1:
        raise RuntimeError("STOP_CHECKPOINT_NOT_UNIQUELY_ALLOWED_BY_SELECTED_MANIFEST")
    expected = matching[0]
    expected_binding = {
        "dataset": view.head.dataset,
        "task_id": view.head.task_id,
        "head_id": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "direction": direction,
        "horizon_steps": view.head.h_steps if horizon_steps is None else horizon_steps,
    }
    for field, value in expected_binding.items():
        if expected.get(field) != value:
            raise RuntimeError(f"STOP_SELECTED_CHECKPOINT_{field.upper()}_MISMATCH")
    if int(expected.get("seed", -1)) != 20260817:
        raise RuntimeError("STOP_SELECTED_CHECKPOINT_SCREENING_SEED_MISMATCH")
    cache = NeuralCandidateCache(Path(candidate_manifest_path).parent).initialize()
    model, scaler, config, preprocessing = _load_verified_artifact(
        cache=cache,
        logical_candidate_id=str(expected["logical_candidate_id"]),
        candidate_dir=requested_dir,
        expected=expected,
        device=device,
        code_commit=code_commit,
    )
    for field in ("model", "candidate_id"):
        if config.get(field) != expected.get(field):
            raise RuntimeError(f"STOP_SELECTED_MODEL_{field.upper()}_MISMATCH")
    if record_hash := expected.get("candidate_record_hash"):
        if not isinstance(record_hash, str) or len(record_hash) != 64:
            raise RuntimeError("STOP_SELECTED_CANDIDATE_RECORD_HASH_INVALID")
    history_steps = int(config["history_steps"])
    dynamic = view.information_set == "dynamic"
    evaluation = native_support(
        load_native_samples(Path(shared), view, split),
        history_steps,
        dynamic=dynamic,
    ).reset_index(drop=True)
    if evaluation.empty:
        raise RuntimeError("STOP_EMPTY_TEST_SUPPORT")
    columns = registered_input_columns(Path(shared), view.head.task_id, view.proxy_policy)
    if list(columns) != preprocessing["feature_order"]:
        raise RuntimeError("STOP_REGISTERED_TEST_FEATURE_ORDER_CHANGED")
    accessor = BaseAccessor(
        Path(shared),
        view.head.dataset,
        split,
        [*columns, view.head.target],
    )
    _set_target_column(accessor, view.head.target)
    prediction = _predict(
        model,
        accessor,
        evaluation,
        columns,
        history_steps,
        scaler,
        dynamic=dynamic,
        device=device,
    )
    if not np.isfinite(prediction).all():
        raise RuntimeError("STOP_NONFINITE_TEST_PREDICTION")
    delta_true = evaluation["y_true"].to_numpy(dtype=np.float64)
    current_level = accessor.block_means(
        evaluation,
        view.head.target,
        [(0, max(1, int(view.head.w0_steps)))],
    ).reshape(-1)
    bundle = metric_bundle_delta_and_level(delta_true, prediction, current_level)
    metrics = _serializable_metrics(bundle)
    if float(metrics["residual_identity_max_abs_error"]) > 1e-10:
        raise AssertionError("STOP_TEST_RESIDUAL_IDENTITY_FAILED")
    future_level_true = current_level + delta_true
    future_level_pred = current_level + prediction
    frame = pd.DataFrame(
        {
            "sample_id": evaluation["view_sample_id"].astype(str),
            "base_origin_id": evaluation["base_origin_id"].astype(str),
            "dataset": view.head.dataset,
            "task_id": view.head.task_id,
            "target_head": view.head.head_id,
            "split": split,
            "model": config["model"],
            "information_set": view.information_set,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "entity_id": evaluation["entity_id"].astype(str),
            "origin": evaluation["origin"].to_numpy(dtype=np.int64),
            "current_level": current_level,
            "y_true_delta": delta_true,
            "y_pred_delta": prediction,
            "y_true_level": future_level_true,
            "y_pred_level": future_level_pred,
        }
    )
    destination = Path(output)
    prediction_path = destination / "test.parquet"
    _atomic_parquet(frame, prediction_path)
    result = {
        "status": "PASS",
        "stage": "NEURAL3_SEALED_CHECKPOINT_TEST_INFERENCE_ONLY",
        "dataset": view.head.dataset,
        "task_id": view.head.task_id,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "direction": direction,
        "horizon_steps": view.head.h_steps if horizon_steps is None else horizon_steps,
        "model": config["model"],
        "selected_candidate_id": config["candidate_id"],
        "history_steps": history_steps,
        "checkpoint_seed": int(config["seed"]),
        "checkpoint_contract": "CACHED_DEVELOPMENT_BEST_WEIGHT_NO_RETRAIN",
        "metrics": metrics,
        "evaluation_rows": int(len(evaluation)),
        "evaluation_support_hash": support_id_hash(evaluation),
        "prediction_value_sha256": _prediction_hash(prediction),
        "prediction_path": str(prediction_path),
        "prediction_file_sha256": sha256_file(prediction_path),
        "checkpoint_dirs": [str(Path(path)) for path in checkpoint_dirs],
        "global_selection_freeze_sha256": sha256_file(global_selection_freeze_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "selected_checkpoint_manifest_sha256": sha256_file(
            selected_checkpoint_manifest_path
        ),
        "support_contract": SUPPORT_CONTRACT,
        "fit_called": False,
        "refit_called": False,
        "select_called": False,
        "test_rows_used_for_fitting": False,
        "test_accessed": True,
        "deletion_forbidden": True,
    }
    atomic_write_json(destination / "TEST_RESULT.json", result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


__all__ = [
    "INFERENCE_ENVIRONMENT",
    "install_inference_only_guard",
    "predict_cached_ensemble",
    "require_sealed_artifact",
]
