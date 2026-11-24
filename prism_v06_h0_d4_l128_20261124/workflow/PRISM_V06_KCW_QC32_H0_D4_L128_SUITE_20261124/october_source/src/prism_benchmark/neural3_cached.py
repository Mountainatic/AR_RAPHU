"""Permanent, resumable Neural-3 training and inference primitives.

This module is intentionally separate from the historical ``neural3`` final
materializer.  Development candidates and frozen-selection final fits are
written immediately to the portable candidate cache.  Test inference only
loads those artifacts; it has no optimizer or fit entry point.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .cpu_data import BaseAccessor, ViewSpec, input_columns as registered_input_columns
from .level_reconstruction import metric_bundle_delta_and_level
from .neural3 import (
    EFFECTIVE_BATCH_SIZE,
    GRADIENT_CLIP_NORM,
    MAX_EPOCHS,
    MAX_SEQUENCE_TOKENS,
    ITRANSFORMER_TEMPORAL_TOKENS,
    NEURAL_FIT_ROW_CAP,
    NEURAL_VALIDATION_ROW_CAP,
    PATIENCE,
    PHYSICAL_BATCH_SIZE,
    SCREENING_SEED,
    WEIGHT_DECAY,
    Candidate,
    Scaler,
    _cap_after_native_support,
    _history_candidate_grid,
    _partition_candidate_support,
    _predict,
    _scaled_batch,
    _set_target_column,
    build_model,
    fit_scaler,
    native_support,
    parameter_count,
    set_seed,
)
from .neural_candidate_cache import (
    CandidateConflictError,
    CandidateHashes,
    CandidateIntegrityError,
    NeuralCandidateCache,
    atomic_write_json,
    atomic_write_npz,
    file_record,
    sha256_file,
    stable_hash,
)
from .v211_support import (
    SUPPORT_CONTRACT,
    base_origin_support_hash,
    load_native_samples,
    require_native_support_contract,
    support_id_hash,
)

try:  # pragma: no cover - the formal worker is Linux
    import resource as _resource
except ImportError:  # pragma: no cover - Windows development hosts
    _resource = None


MISSING_VALUE_POLICY = "FAIL_ON_NONFINITE_REQUIRED_VALUE"
FINAL_FIT_PHASE = "FROZEN_SELECTION_FINAL_FIT"
SELECTION_PHASE = "DEVELOPMENT_CANDIDATE_SELECTION"


def runtime_code_binding() -> dict[str, Any]:
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


def _resolved_model_architecture(candidate: Candidate, input_dim: int) -> dict[str, Any]:
    if candidate.model == "LSTM":
        hidden, layers, dropout = (
            (64, 1, 0.0) if candidate.capacity == "SMALL" else (96, 2, 0.1)
        )
        return {
            "family": "LSTM",
            "input_dim": input_dim,
            "hidden_size": hidden,
            "num_layers": layers,
            "dropout": dropout,
            "batch_first": True,
            "head": "Linear(hidden_size,1)",
        }
    if candidate.model == "iTransformer":
        d_model, heads, layers, d_ff, dropout = (
            (64, 4, 2, 128, 0.1)
            if candidate.capacity == "SMALL"
            else (96, 4, 2, 192, 0.1)
        )
        return {
            "family": "iTransformer",
            "input_dim": input_dim,
            "variables_are_tokens": True,
            "temporal_tokens": ITRANSFORMER_TEMPORAL_TOKENS,
            "temporal_projection": [ITRANSFORMER_TEMPORAL_TOKENS, d_model],
            "d_model": d_model,
            "nhead": heads,
            "encoder_layers": layers,
            "dim_feedforward": d_ff,
            "dropout": dropout,
            "activation": "gelu",
            "norm_first": True,
            "batch_first": True,
            "pooling": "mean_over_variable_tokens",
        }
    if candidate.model == "TimeMixer":
        d_model = 64 if candidate.capacity == "SMALL" else 96
        return {
            "family": "TimeMixer",
            "input_dim": input_dim,
            "d_model": d_model,
            "past_mixer_blocks": 3,
            "past_mixer": "Linear-GELU-Linear",
            "causal_pooling_strides": [1, 2, 4],
            "scale_summary": "mean_over_past_tokens",
            "future_mixer": "LayerNorm-Linear-GELU",
            "head": "Linear(d_model,1)",
        }
    raise ValueError(candidate.model)


@dataclass(frozen=True)
class PreparedCandidate:
    shared: Path
    view: ViewSpec
    candidate: Candidate
    accessor: BaseAccessor
    fit_samples: pd.DataFrame
    validation_samples: pd.DataFrame
    common_fit_samples: pd.DataFrame
    scaler: Scaler
    columns: tuple[str, ...]
    dynamic: bool
    data_hash: str
    data_details: Mapping[str, Any]
    support_details: Mapping[str, Any]
    sample_order_details: Mapping[str, Any]
    fit_row_cap: int
    validation_row_cap: int


@dataclass(frozen=True)
class LoadedArtifact:
    model: nn.Module
    scaler: Scaler
    model_config: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    candidate_dir: Path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return f"NOT_DEFINED_{str(value).upper()}"
    if isinstance(value, Path):
        return str(value)
    return value


def _ordered_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _prediction_hash(prediction: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(prediction, dtype="<f8"))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _state_dict_arrays(state: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().contiguous().numpy()
        for name, tensor in sorted(state.items())
    }


def _load_state_dict_npz(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: torch.from_numpy(np.array(archive[name], copy=True))
            for name in archive.files
        }


def _pack_tree(value: Any) -> tuple[dict[str, np.ndarray], Any]:
    arrays: dict[str, np.ndarray] = {}

    def visit(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            key = f"array_{len(arrays):06d}"
            arrays[key] = item.detach().cpu().contiguous().numpy()
            return {"type": "torch_tensor", "array": key}
        if isinstance(item, np.ndarray):
            key = f"array_{len(arrays):06d}"
            if item.dtype.hasobject:
                raise ValueError("object arrays are forbidden in training state")
            arrays[key] = np.ascontiguousarray(item)
            return {"type": "numpy_array", "array": key}
        if isinstance(item, Mapping):
            return {
                "type": "mapping",
                "items": [[visit(key), visit(child)] for key, child in item.items()],
            }
        if isinstance(item, tuple):
            return {"type": "tuple", "items": [visit(child) for child in item]}
        if isinstance(item, list):
            return {"type": "list", "items": [visit(child) for child in item]}
        if isinstance(item, np.generic):
            return visit(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return {"type": "nonfinite_float", "value": repr(item)}
        if item is None or isinstance(item, (str, int, float, bool)):
            return {"type": "scalar", "value": _json_safe(item)}
        raise TypeError(f"unsupported portable training-state value: {type(item)!r}")

    return arrays, visit(value)


def _unpack_tree(arrays: Mapping[str, np.ndarray], schema: Any) -> Any:
    kind = schema["type"]
    if kind == "torch_tensor":
        return torch.from_numpy(np.array(arrays[schema["array"]], copy=True))
    if kind == "numpy_array":
        return np.array(arrays[schema["array"]], copy=True)
    if kind == "mapping":
        return {
            _unpack_tree(arrays, key): _unpack_tree(arrays, value)
            for key, value in schema["items"]
        }
    if kind == "tuple":
        return tuple(_unpack_tree(arrays, item) for item in schema["items"])
    if kind == "list":
        return [_unpack_tree(arrays, item) for item in schema["items"]]
    if kind == "scalar":
        return schema["value"]
    if kind == "nonfinite_float":
        return float(schema["value"])
    raise ValueError(f"unknown portable tree node: {kind}")


def _peak_process_rss_bytes() -> int:
    if _resource is None:
        return 0
    value = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.  Formal execution is Linux, but
    # keeping the conversion explicit makes local QA deterministic.
    return value if sys.platform == "darwin" else value * 1024


def _candidate_by_id(
    model_name: str,
    histories: Sequence[int],
    candidate_id: str,
    history_labels: Mapping[int, str] | None,
) -> Candidate:
    matches = [
        candidate
        for candidate in _history_candidate_grid(model_name, histories, history_labels)
        if candidate.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise KeyError(f"candidate id is not unique in frozen grid: {candidate_id}")
    return matches[0]


def _data_binding_paths(shared: Path, view: ViewSpec) -> list[Path]:
    common = [
        shared / "TASK_REGISTRY.json",
        shared / "PROTOCOL.json",
        shared / "dataset_views" / "VIEW_REGISTRY.json",
        shared / "base_data" / view.head.dataset / "train.parquet",
        shared / "sample_ids" / view.relative_root / "train.parquet",
        shared / "sample_ids" / view.relative_root / "validation.parquet",
    ]
    if view.head.dataset == "cz_czochralski":
        common.extend(
            [
                shared / "CZ_TASK_REALIZATION.json",
                shared / "JOINT_LIFT_PCA_CONTRACT.json",
                shared / "C1_NATIVE_SUPPORT_AUDIT.json",
            ]
        )
    else:
        common.extend(
            [
                shared / "base_data" / view.head.dataset / "validation.parquet",
                shared / "DATASET_HASHES.json",
            ]
        )
    return common


def build_data_binding(shared: Path, view: ViewSpec) -> dict[str, Any]:
    shared = Path(shared).resolve()
    files = []
    for path in _data_binding_paths(shared, view):
        if not path.is_file():
            raise FileNotFoundError(f"STOP_REQUIRED_NEURAL_DATA_BINDING_FILE_MISSING:{path}")
        if path.is_symlink():
            raise RuntimeError(f"STOP_NEURAL_DATA_BINDING_SYMLINK:{path}")
        info = path.stat()
        files.append(
            {
                "path": path.relative_to(shared).as_posix(),
                "bytes": int(info.st_size),
                "mtime_ns": int(info.st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )
    return {
        "format": "PRISM_NEURAL_DATA_BINDING_V1",
        "shared_root": str(shared),
        "view": view.relative_root.as_posix(),
        "files": files,
    }


def validate_data_binding(
    shared: Path,
    view: ViewSpec,
    binding: Mapping[str, Any],
    *,
    verify_sha256: bool,
) -> None:
    shared = Path(shared).resolve()
    if binding.get("format") != "PRISM_NEURAL_DATA_BINDING_V1":
        raise ValueError("invalid precomputed neural data binding")
    if binding.get("shared_root") != str(shared):
        raise ValueError("precomputed neural data binding shared root mismatch")
    if binding.get("view") != view.relative_root.as_posix():
        raise ValueError("precomputed neural data binding view mismatch")
    expected_paths = {
        path.relative_to(shared).as_posix() for path in _data_binding_paths(shared, view)
    }
    records = binding.get("files")
    if (
        not isinstance(records, list)
        or len(records) != len(expected_paths)
        or {str(item.get("path")) for item in records if isinstance(item, Mapping)}
        != expected_paths
    ):
        raise RuntimeError("STOP_NEURAL_DATA_BINDING_FILE_SET_MISMATCH")
    for item in records:
        if not isinstance(item, Mapping):
            raise RuntimeError("STOP_NEURAL_DATA_BINDING_RECORD_INVALID")
        relative = str(item.get("path", ""))
        relative_path = PurePosixPath(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or any(part in ("", ".", "..") for part in relative_path.parts)
        ):
            raise RuntimeError("STOP_NEURAL_DATA_BINDING_PATH_INVALID")
        path = shared.joinpath(*relative_path.parts)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"STOP_NEURAL_DATA_BINDING_SOURCE_UNSAFE:{path}")
        info = path.stat()
        try:
            expected_bytes = int(item["bytes"])
            expected_mtime_ns = int(item["mtime_ns"])
            expected_sha = str(item["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("STOP_NEURAL_DATA_BINDING_RECORD_INVALID") from error
        if expected_bytes < 0 or expected_mtime_ns < 0 or len(expected_sha) != 64:
            raise RuntimeError("STOP_NEURAL_DATA_BINDING_RECORD_INVALID")
        if int(info.st_size) != expected_bytes:
            raise RuntimeError(f"STOP_PRECOMPUTED_DATA_BINDING_SIZE_CHANGED:{path}")
        if int(info.st_mtime_ns) != expected_mtime_ns:
            raise RuntimeError(f"STOP_PRECOMPUTED_DATA_BINDING_MTIME_CHANGED:{path}")
        # ``verify_sha256=False`` is retained for diagnostics/API compatibility,
        # but all execution call sites use the strict path below.  A caller that
        # needs provenance safety must never rely on size and mtime alone.
        if verify_sha256 and sha256_file(path) != expected_sha:
            raise RuntimeError(f"STOP_PRECOMPUTED_DATA_BINDING_SHA256_CHANGED:{path}")


def prepare_candidate(
    *,
    shared: Path,
    view: ViewSpec,
    model_name: str,
    histories: Sequence[int],
    candidate_id: str,
    history_labels: Mapping[int, str] | None = None,
    fit_row_cap: int = NEURAL_FIT_ROW_CAP,
    validation_row_cap: int = NEURAL_VALIDATION_ROW_CAP,
    data_binding: Mapping[str, Any] | None = None,
) -> PreparedCandidate:
    """Load train/validation only and realize frozen native/common support."""

    shared = Path(shared).resolve()
    if fit_row_cap < 1 or validation_row_cap < 1:
        raise ValueError("row caps must be positive")
    data_details = (
        build_data_binding(shared, view)
        if data_binding is None
        else dict(data_binding)
    )
    # The runner prevalidates a shared binding once per view before dispatch.
    # Per-candidate workers still recheck path, size, and mtime, while avoiding
    # hashing multi-gigabyte parquet files once for every candidate process.
    validate_data_binding(shared, view, data_details, verify_sha256=True)
    candidate = _candidate_by_id(
        model_name, histories, candidate_id, history_labels
    )
    train = load_native_samples(shared, view, "train")
    validation = load_native_samples(shared, view, "validation")
    require_native_support_contract(train)
    require_native_support_contract(validation)
    dynamic = view.information_set == "dynamic"
    grid = _history_candidate_grid(model_name, histories, history_labels)
    available, unavailable = _partition_candidate_support(
        grid, train, validation, dynamic=dynamic
    )
    available_by_id = {item.candidate_id: support for item, support in available}
    if candidate.candidate_id not in available_by_id:
        raise RuntimeError(
            f"candidate has no native support: {candidate.candidate_id}: {unavailable}"
        )
    max_history = max(item.history_steps for item, _ in available)
    common_fit = native_support(train, max_history, dynamic=dynamic)
    common_validation = native_support(validation, max_history, dynamic=dynamic)
    candidate_fit = available_by_id[candidate.candidate_id]
    candidate_fit = _cap_after_native_support(candidate_fit, int(fit_row_cap))
    common_validation = _cap_after_native_support(
        common_validation, int(validation_row_cap)
    )
    if candidate_fit.empty or common_fit.empty or common_validation.empty:
        raise RuntimeError("candidate support became empty after frozen row caps")
    columns = tuple(
        registered_input_columns(shared, view.head.task_id, view.proxy_policy)
    )
    accessor = BaseAccessor(
        shared,
        view.head.dataset,
        "validation",
        [*columns, view.head.target],
    )
    _set_target_column(accessor, view.head.target)
    scaler = fit_scaler(
        accessor,
        common_fit,
        list(columns),
        view.head.target,
        max_history,
    )
    if not (
        np.isfinite(scaler.feature_mean).all()
        and np.isfinite(scaler.feature_scale).all()
        and math.isfinite(scaler.target_mean)
        and math.isfinite(scaler.target_scale)
    ):
        raise RuntimeError("STOP_NONFINITE_FROZEN_NEURAL_SCALER")
    # Detect a concurrent mutation during data/sample materialization.  Full
    # SHA was established before the load; bytes+mtime are rechecked here and
    # again immediately before candidate finalize.
    validate_data_binding(shared, view, data_details, verify_sha256=True)
    support_details = {
        "contract": SUPPORT_CONTRACT,
        "train_rows": int(len(candidate_fit)),
        "validation_rows": int(len(common_validation)),
        "common_scaler_fit_rows": int(len(common_fit)),
        "train_base_origin_support_hash": base_origin_support_hash(candidate_fit),
        "validation_base_origin_support_hash": base_origin_support_hash(
            common_validation
        ),
        "common_scaler_fit_base_origin_support_hash": base_origin_support_hash(
            common_fit
        ),
    }
    sample_order_details = {
        "train_support_id_hash": support_id_hash(candidate_fit),
        "validation_support_id_hash": support_id_hash(common_validation),
        "common_scaler_fit_support_id_hash": support_id_hash(common_fit),
        "train_base_origin_order_hash": _ordered_hash(
            candidate_fit["base_origin_id"].astype(str).tolist()
        ),
        "validation_base_origin_order_hash": _ordered_hash(
            common_validation["base_origin_id"].astype(str).tolist()
        ),
    }
    data_hash = stable_hash(data_details)
    return PreparedCandidate(
        shared=shared,
        view=view,
        candidate=candidate,
        accessor=accessor,
        fit_samples=candidate_fit,
        validation_samples=common_validation,
        common_fit_samples=common_fit.reset_index(drop=True),
        scaler=scaler,
        columns=columns,
        dynamic=dynamic,
        data_hash=data_hash,
        data_details=data_details,
        support_details=support_details,
        sample_order_details=sample_order_details,
        fit_row_cap=int(fit_row_cap),
        validation_row_cap=int(validation_row_cap),
    )


def logical_candidate_id(
    prepared: PreparedCandidate,
    *,
    phase: str,
    seed: int,
    direction: str | None = None,
    horizon_steps: int | None = None,
) -> str:
    parts = [
        phase,
        prepared.view.head.dataset,
        prepared.view.head.task_id,
        prepared.view.information_set,
        prepared.view.availability_scenario,
        prepared.view.proxy_policy,
        direction or "NO_DIRECTION",
        f"H{prepared.view.head.h_steps if horizon_steps is None else horizon_steps}",
        prepared.candidate.model,
        prepared.candidate.candidate_id,
        f"seed{int(seed)}",
    ]
    return "__".join(parts)


def candidate_provenance(
    prepared: PreparedCandidate,
    *,
    phase: str,
    seed: int,
    code_commit: str,
    epochs: int | None = None,
    direction: str | None = None,
    horizon_steps: int | None = None,
) -> tuple[CandidateHashes, dict[str, Any]]:
    input_dim = len(prepared.columns) + (1 if prepared.dynamic else 0)
    config = {
        "phase": phase,
        "dataset": prepared.view.head.dataset,
        "task_id": prepared.view.head.task_id,
        "head_id": prepared.view.head.head_id,
        "target": prepared.view.head.target,
        "information_set": prepared.view.information_set,
        "availability_scenario": prepared.view.availability_scenario,
        "proxy_policy": prepared.view.proxy_policy,
        "direction": direction,
        "horizon_steps": (
            prepared.view.head.h_steps if horizon_steps is None else horizon_steps
        ),
        "model": prepared.candidate.model,
        "candidate_id": prepared.candidate.candidate_id,
        "history_steps": prepared.candidate.history_steps,
        "history_label": prepared.candidate.history_label,
        "capacity": prepared.candidate.capacity,
        "learning_rate": prepared.candidate.learning_rate,
        "seed": int(seed),
        "epochs": None if epochs is None else int(epochs),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "resolved_model_architecture": _resolved_model_architecture(
            prepared.candidate, input_dim
        ),
        "resolved_optimizer": {
            "name": "AdamW",
            "learning_rate": prepared.candidate.learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
        },
        "early_stopping": {
            "metric": "validation_mse",
            "comparator": "strict_less_than",
            "min_delta": 0.0,
            "patience": PATIENCE,
            "restore_best_validation_state": True,
        },
        "sequence_tokenization": {
            "maximum_tokens": MAX_SEQUENCE_TOKENS,
            "history_compression": "contiguous_past_block_mean",
            "causal": True,
        },
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "fit_row_cap": prepared.fit_row_cap,
        "validation_row_cap": prepared.validation_row_cap,
        "feature_order": list(prepared.columns),
        "dynamic_target_history": prepared.dynamic,
        "missing_value_policy": MISSING_VALUE_POLICY,
        "support_contract": SUPPORT_CONTRACT,
        "code_commit": code_commit,
        "code_binding": runtime_code_binding(),
        "tf32_allowed": False,
        "precision": "FP32",
        "torch_determinism": {
            "use_deterministic_algorithms": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
    }
    hashes = CandidateHashes(
        config_hash=stable_hash(config),
        data_hash=prepared.data_hash,
        support_hash=stable_hash(prepared.support_details),
        sample_order_hash=stable_hash(prepared.sample_order_details),
    )
    metadata = {
        "config": config,
        "support": dict(prepared.support_details),
        "sample_order": dict(prepared.sample_order_details),
        "hashes": hashes.as_dict(),
        "data_binding": dict(prepared.data_details),
        "test_accessed": False,
        "ood_accessed": False,
        "deletion_forbidden": True,
    }
    return hashes, metadata


def _rng_payload(permutation_rng: np.random.Generator) -> dict[str, Any]:
    return {
        "numpy_permutation_rng": permutation_rng.bit_generator.state,
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng(payload: Mapping[str, Any]) -> np.random.Generator:
    generator = np.random.default_rng()
    generator.bit_generator.state = payload["numpy_permutation_rng"]
    torch.set_rng_state(payload["torch_cpu_rng"])
    if torch.cuda.is_available() and payload.get("torch_cuda_rng"):
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng"])
    return generator


_TRAINING_STATE_FORMAT = "PRISM_NEURAL_PORTABLE_TRAINING_STATE_V1"


def _snapshot_pointer_paths(
    writer: Any, pointer: Mapping[str, Any]
) -> tuple[str, Path, Path]:
    """Validate a resume pointer and every file record it references."""

    if pointer.get("format") != _TRAINING_STATE_FORMAT:
        raise CandidateIntegrityError("STOP_TRAINING_STATE_POINTER_FORMAT_INVALID")
    slot = pointer.get("slot")
    if slot not in {"A", "B"}:
        raise CandidateIntegrityError("STOP_TRAINING_STATE_POINTER_SLOT_INVALID")
    expected_names = {
        "npz": f"resume/training_state_{slot}.npz",
        "schema": f"resume/training_state_{slot}.json",
    }
    paths: dict[str, Path] = {}
    staging_root = Path(writer.staging_dir).resolve()
    for kind, expected_name in expected_names.items():
        record = pointer.get(kind)
        if not isinstance(record, Mapping) or record.get("name") != expected_name:
            raise CandidateIntegrityError(
                f"STOP_TRAINING_STATE_POINTER_{kind.upper()}_RECORD_INVALID"
            )
        relative = PurePosixPath(str(record["name"]))
        if relative.is_absolute() or any(
            part in ("", ".", "..") for part in relative.parts
        ):
            raise CandidateIntegrityError("STOP_TRAINING_STATE_POINTER_PATH_INVALID")
        path = Path(writer.staging_dir).joinpath(*relative.parts)
        try:
            path.resolve().relative_to(staging_root)
        except ValueError as error:
            raise CandidateIntegrityError(
                "STOP_TRAINING_STATE_POINTER_PATH_OUTSIDE_STAGING"
            ) from error
        if path.is_symlink() or not path.is_file():
            raise CandidateIntegrityError(
                f"STOP_TRAINING_STATE_POINTER_FILE_MISSING:{path}"
            )
        try:
            expected_bytes = int(record["bytes"])
            expected_mtime_ns = int(record["mtime_ns"])
            expected_sha = str(record["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise CandidateIntegrityError(
                f"STOP_TRAINING_STATE_POINTER_{kind.upper()}_RECORD_INVALID"
            ) from error
        if (
            expected_bytes < 0
            or expected_mtime_ns < 0
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
        ):
            raise CandidateIntegrityError(
                f"STOP_TRAINING_STATE_POINTER_{kind.upper()}_RECORD_INVALID"
            )
        observed = file_record(path, relative_to=writer.staging_dir)
        for field, expected in (
            ("bytes", expected_bytes),
            ("mtime_ns", expected_mtime_ns),
            ("sha256", expected_sha),
        ):
            if observed[field] != expected:
                raise CandidateIntegrityError(
                    f"STOP_TRAINING_STATE_{kind.upper()}_{field.upper()}_MISMATCH:{path}"
                )
        paths[kind] = path
    for field in ("epoch", "next_batch_number"):
        value = pointer.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CandidateIntegrityError(
                f"STOP_TRAINING_STATE_POINTER_{field.upper()}_INVALID"
            )
    if not isinstance(pointer.get("epoch_in_progress"), bool):
        raise CandidateIntegrityError(
            "STOP_TRAINING_STATE_POINTER_EPOCH_IN_PROGRESS_INVALID"
        )
    return slot, paths["npz"], paths["schema"]


def _schema_array_names(schema: Any) -> set[str]:
    """Collect array references while rejecting malformed portable schemas."""

    if not isinstance(schema, Mapping):
        raise CandidateIntegrityError("STOP_TRAINING_STATE_SCHEMA_NODE_INVALID")
    kind = schema.get("type")
    if kind in {"torch_tensor", "numpy_array"}:
        name = schema.get("array")
        if not isinstance(name, str) or not name:
            raise CandidateIntegrityError("STOP_TRAINING_STATE_SCHEMA_ARRAY_INVALID")
        return {name}
    if kind in {"mapping", "tuple", "list"}:
        items = schema.get("items")
        if not isinstance(items, list):
            raise CandidateIntegrityError("STOP_TRAINING_STATE_SCHEMA_ITEMS_INVALID")
        if kind == "mapping":
            if any(not isinstance(item, list) or len(item) != 2 for item in items):
                raise CandidateIntegrityError(
                    "STOP_TRAINING_STATE_SCHEMA_MAPPING_ITEMS_INVALID"
                )
            children = [child for item in items for child in item]
        else:
            children = items
        names: set[str] = set()
        for child in children:
            names.update(_schema_array_names(child))
        return names
    if kind in {"scalar", "nonfinite_float"}:
        return set()
    raise CandidateIntegrityError(f"STOP_TRAINING_STATE_SCHEMA_TYPE_INVALID:{kind}")


def _validate_snapshot_payload(
    pointer: Mapping[str, Any], value: Any, archive_names: set[str], schema: Any
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateIntegrityError("STOP_TRAINING_STATE_ROOT_IS_NOT_MAPPING")
    referenced = _schema_array_names(schema)
    if referenced != archive_names:
        raise CandidateIntegrityError(
            "STOP_TRAINING_STATE_SCHEMA_ARRAY_SET_MISMATCH"
        )
    if value.get("format") not in (None, _TRAINING_STATE_FORMAT):
        raise CandidateIntegrityError("STOP_TRAINING_STATE_PAYLOAD_FORMAT_INVALID")
    for field in ("epoch", "next_batch_number"):
        if field in value and value[field] != pointer[field]:
            raise CandidateIntegrityError(
                f"STOP_TRAINING_STATE_POINTER_{field.upper()}_PAYLOAD_MISMATCH"
            )
    if "epoch_in_progress" in value and value["epoch_in_progress"] != pointer[
        "epoch_in_progress"
    ]:
        raise CandidateIntegrityError(
            "STOP_TRAINING_STATE_POINTER_EPOCH_IN_PROGRESS_PAYLOAD_MISMATCH"
        )
    if "epoch" in value and (
        isinstance(value["epoch"], bool)
        or not isinstance(value["epoch"], int)
        or value["epoch"] < 0
    ):
        raise CandidateIntegrityError("STOP_TRAINING_STATE_EPOCH_INVALID")
    if "next_batch_number" in value and (
        isinstance(value["next_batch_number"], bool)
        or not isinstance(value["next_batch_number"], int)
        or value["next_batch_number"] < 0
    ):
        raise CandidateIntegrityError("STOP_TRAINING_STATE_NEXT_BATCH_INVALID")
    if "epoch_in_progress" in value and not isinstance(
        value["epoch_in_progress"], bool
    ):
        raise CandidateIntegrityError("STOP_TRAINING_STATE_EPOCH_IN_PROGRESS_INVALID")
    return value


def _write_training_snapshot(writer: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    pointer_path = writer.staging_dir / "resume" / "LATEST.json"
    active_slot = None
    if pointer_path.is_file():
        try:
            previous_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            active_slot, _, _ = _snapshot_pointer_paths(writer, previous_pointer)
        except CandidateIntegrityError:
            raise
        except (OSError, ValueError, TypeError) as error:
            raise CandidateIntegrityError(
                f"STOP_TRAINING_STATE_POINTER_UNREADABLE:{pointer_path}"
            ) from error
    slot = "B" if active_slot == "A" else "A"
    arrays, schema = _pack_tree(dict(payload))
    npz_name = f"resume/training_state_{slot}.npz"
    schema_name = f"resume/training_state_{slot}.json"
    npz_path = writer.write_npz(npz_name, arrays)
    schema_path = writer.write_json(schema_name, schema)
    pointer = {
        "format": _TRAINING_STATE_FORMAT,
        "slot": slot,
        "npz": file_record(npz_path, relative_to=writer.staging_dir),
        "schema": file_record(schema_path, relative_to=writer.staging_dir),
        "epoch": int(payload["epoch"]),
        "epoch_in_progress": bool(payload["epoch_in_progress"]),
        "next_batch_number": int(payload["next_batch_number"]),
        "updated_unix_seconds": time.time(),
    }
    writer.write_json("resume/LATEST.json", pointer)
    writer.checkpoint_state(
        latest_training_state="resume/LATEST.json",
        latest_training_state_record=file_record(
            pointer_path, relative_to=writer.staging_dir
        ),
        epoch=pointer["epoch"],
        epoch_in_progress=pointer["epoch_in_progress"],
        next_batch_number=pointer["next_batch_number"],
    )
    return pointer


def _read_training_snapshot(writer: Any) -> dict[str, Any] | None:
    pointer_path = writer.staging_dir / "resume" / "LATEST.json"
    if not pointer_path.is_file():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        _, npz_path, schema_path = _snapshot_pointer_paths(writer, pointer)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        value = _unpack_tree(arrays, schema)
        return _validate_snapshot_payload(pointer, value, set(arrays), schema)
    except CandidateIntegrityError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CandidateIntegrityError(
            f"STOP_INCOMPLETE_TRAINING_STATE_UNREADABLE:{pointer_path}"
        ) from exc


def _portable_preprocessing(prepared: PreparedCandidate) -> dict[str, Any]:
    pca_path = prepared.shared / "JOINT_LIFT_PCA_CONTRACT.json"
    pca: Any = (
        json.loads(pca_path.read_text(encoding="utf-8"))
        if pca_path.is_file()
        else {"status": "NOT_APPLICABLE"}
    )
    return {
        "format": "PRISM_NEURAL_PREPROCESSING_V1",
        "feature_order": list(prepared.columns),
        "dynamic_target_history_appended": prepared.dynamic,
        "dynamic_target_column": (
            prepared.view.head.target if prepared.dynamic else None
        ),
        "missing_value_policy": MISSING_VALUE_POLICY,
        "scaler_fit_scope": "OUTER_TRAINING_COMMON_MAX_HISTORY_SUPPORT_ONLY",
        "pca": pca,
        "support_contract": SUPPORT_CONTRACT,
    }


def _training_state_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_state: Mapping[str, torch.Tensor] | None,
    permutation_rng: np.random.Generator,
    epoch: int,
    epoch_in_progress: bool,
    next_batch_number: int,
    order: np.ndarray | None,
    best_validation_mse: float,
    best_epoch: int,
    stale_epochs: int,
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        "format": "PRISM_NEURAL_PORTABLE_TRAINING_STATE_V1",
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "best_model_state": (
            None
            if best_state is None
            else {
                key: value.detach().cpu().clone()
                for key, value in best_state.items()
            }
        ),
        "rng": _rng_payload(permutation_rng),
        "epoch": int(epoch),
        "epoch_in_progress": bool(epoch_in_progress),
        "next_batch_number": int(next_batch_number),
        "order": None if order is None else np.asarray(order, dtype=np.int64),
        "best_validation_mse": float(best_validation_mse),
        "best_epoch": int(best_epoch),
        "stale_epochs": int(stale_epochs),
        "stop_reason": stop_reason,
    }


def _restore_training_state(
    payload: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[
    np.random.Generator,
    dict[str, torch.Tensor] | None,
    int,
    bool,
    int,
    np.ndarray | None,
    float,
    int,
    int,
]:
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    permutation_rng = _restore_rng(payload["rng"])
    best_state = payload.get("best_model_state")
    order = payload.get("order")
    return (
        permutation_rng,
        best_state,
        int(payload["epoch"]),
        bool(payload["epoch_in_progress"]),
        int(payload["next_batch_number"]),
        None if order is None else np.asarray(order, dtype=np.int64),
        float(payload["best_validation_mse"]),
        int(payload["best_epoch"]),
        int(payload["stale_epochs"]),
    )


def _scalar_metric_bundle(
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    current_level: np.ndarray,
) -> dict[str, Any]:
    bundle = metric_bundle_delta_and_level(delta_true, delta_pred, current_level)
    result = {
        key: value
        for key, value in bundle.items()
        if not isinstance(value, np.ndarray)
    }
    residual = delta_true - delta_pred
    level_residual = (current_level + delta_true) - (current_level + delta_pred)
    identity = float(np.max(np.abs(residual - level_residual), initial=0.0))
    if identity > 1e-10:
        raise AssertionError("STOP_LEVEL_RECONSTRUCTION_IDENTITY_FAILED")
    result["residual_identity_max_abs_error"] = identity
    result["residual_identity_tolerance"] = 1e-10
    result["residual_identity_status"] = "PASS"
    return _json_safe(result)


def train_selection_candidate(
    *,
    cache: NeuralCandidateCache,
    prepared: PreparedCandidate,
    code_commit: str,
    device: torch.device,
    direction: str | None = None,
    horizon_steps: int | None = None,
    seed: int = SCREENING_SEED,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    stop_requested: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    """Train or exactly reuse one development candidate under its own lock."""

    if max_epochs < 1 or max_epochs > MAX_EPOCHS or patience < 1:
        raise ValueError("invalid epoch or patience limit")
    hashes, metadata = candidate_provenance(
        prepared,
        phase=SELECTION_PHASE,
        seed=seed,
        code_commit=code_commit,
        direction=direction,
        horizon_steps=horizon_steps,
    )
    metadata = dict(metadata)
    metadata["runtime_limits"] = {
        "max_epochs": int(max_epochs),
        "patience": int(patience),
    }
    # Pilot candidates use the formal limits so their artifacts are reusable.
    if max_epochs != MAX_EPOCHS or patience != PATIENCE:
        hashes = CandidateHashes(
            config_hash=stable_hash({**metadata["config"], **metadata["runtime_limits"]}),
            data_hash=hashes.data_hash,
            support_hash=hashes.support_hash,
            sample_order_hash=hashes.sample_order_hash,
        )
        metadata["hashes"] = hashes.as_dict()
    logical_id = logical_candidate_id(
        prepared,
        phase=SELECTION_PHASE,
        seed=seed,
        direction=direction,
        horizon_steps=horizon_steps,
    )
    with cache.lock(logical_id, {"phase": SELECTION_PHASE, "model": prepared.candidate.model}):
        decision = cache.reuse_status(logical_id, hashes)
        if decision.reusable:
            return {
                "status": "REUSED",
                "candidate_id": logical_id,
                "candidate_dir": str(decision.candidate_dir),
                "record": dict(decision.record or {}),
            }
        if decision.status in {"CORRUPT", "CORRUPT_QUARANTINED"}:
            raise CandidateIntegrityError(
                f"STOP_CORRUPT_CANDIDATE_REQUIRES_REPORT_AND_APPROVAL:{logical_id}:"
                f"{decision.reason}"
            )
        if decision.status == "HASH_MISMATCH":
            raise CandidateConflictError(
                f"STOP_EXISTING_CANDIDATE_HASH_MISMATCH:{logical_id}"
            )
        writer = cache.begin_candidate(logical_id, hashes, metadata, resume=True)
        try:
            set_seed(seed)
            model = build_model(
                prepared.candidate.model,
                len(prepared.columns) + (1 if prepared.dynamic else 0),
                prepared.candidate.capacity,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=prepared.candidate.learning_rate,
                weight_decay=WEIGHT_DECAY,
            )
            accumulation = max(1, EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE)
            if accumulation != 1:
                raise RuntimeError(
                    "mid-epoch portable resume currently requires accumulation=1"
                )
            snapshot = _read_training_snapshot(writer) if writer.resumed else None
            if snapshot is None:
                permutation_rng = np.random.default_rng(seed)
                best_state: dict[str, torch.Tensor] | None = None
                restored_epoch = 0
                epoch_in_progress = False
                next_batch_number = 0
                restored_order: np.ndarray | None = None
                best_validation_mse = float("inf")
                best_epoch = 0
                stale_epochs = 0
            else:
                (
                    permutation_rng,
                    best_state,
                    restored_epoch,
                    epoch_in_progress,
                    next_batch_number,
                    restored_order,
                    best_validation_mse,
                    best_epoch,
                    stale_epochs,
                ) = _restore_training_state(snapshot, model=model, optimizer=optimizer)
            started = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            first_epoch = restored_epoch if epoch_in_progress else restored_epoch + 1
            epochs_run = restored_epoch
            if snapshot is not None and not epoch_in_progress and stale_epochs >= patience:
                # The prior process may have exited after the atomic epoch
                # checkpoint but before executing the deterministic patience
                # break.  Finalize the already-selected best state directly.
                first_epoch = max_epochs + 1
            for epoch in range(max(1, first_epoch), max_epochs + 1):
                model.train()
                if epoch_in_progress and epoch == restored_epoch:
                    if restored_order is None:
                        raise CandidateIntegrityError(
                            "STOP_RESUME_ORDER_MISSING_FOR_IN_PROGRESS_EPOCH"
                        )
                    order = restored_order
                    batch_start_number = next_batch_number
                else:
                    order = permutation_rng.permutation(len(prepared.fit_samples))
                    batch_start_number = 0
                optimizer.zero_grad(set_to_none=True)
                starts = list(range(0, len(order), PHYSICAL_BATCH_SIZE))
                for batch_number in range(batch_start_number, len(starts)):
                    start = starts[batch_number]
                    batch = prepared.fit_samples.iloc[
                        order[start : start + PHYSICAL_BATCH_SIZE]
                    ]
                    values, target_values = _scaled_batch(
                        prepared.accessor,
                        batch,
                        list(prepared.columns),
                        prepared.candidate.history_steps,
                        prepared.scaler,
                        dynamic=prepared.dynamic,
                        device=device,
                    )
                    loss = F.mse_loss(model(values), target_values)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRADIENT_CLIP_NORM
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    reason = None if stop_requested is None else stop_requested()
                    if reason:
                        state = _training_state_payload(
                            model=model,
                            optimizer=optimizer,
                            best_state=best_state,
                            permutation_rng=permutation_rng,
                            epoch=epoch,
                            epoch_in_progress=True,
                            next_batch_number=batch_number + 1,
                            order=order,
                            best_validation_mse=best_validation_mse,
                            best_epoch=best_epoch,
                            stale_epochs=stale_epochs,
                            stop_reason=reason,
                        )
                        _write_training_snapshot(writer, state)
                        writer.abort(reason)
                        return {
                            "status": "SAFE_PAUSED",
                            "candidate_id": logical_id,
                            "reason": reason,
                            "resumable": True,
                            "staging_dir": str(writer.staging_dir),
                        }
                reason = None if stop_requested is None else stop_requested()
                if reason:
                    state = _training_state_payload(
                        model=model,
                        optimizer=optimizer,
                        best_state=best_state,
                        permutation_rng=permutation_rng,
                        epoch=epoch,
                        epoch_in_progress=True,
                        next_batch_number=len(starts),
                        order=order,
                        best_validation_mse=best_validation_mse,
                        best_epoch=best_epoch,
                        stale_epochs=stale_epochs,
                        stop_reason=reason,
                    )
                    _write_training_snapshot(writer, state)
                    writer.abort(reason)
                    return {
                        "status": "SAFE_PAUSED",
                        "candidate_id": logical_id,
                        "reason": reason,
                        "resumable": True,
                        "staging_dir": str(writer.staging_dir),
                    }
                validation_prediction = _predict(
                    model,
                    prepared.accessor,
                    prepared.validation_samples,
                    list(prepared.columns),
                    prepared.candidate.history_steps,
                    prepared.scaler,
                    dynamic=prepared.dynamic,
                    device=device,
                )
                validation_target = prepared.validation_samples["y_true"].to_numpy(
                    dtype=np.float64
                )
                if not np.isfinite(validation_prediction).all():
                    raise RuntimeError("STOP_NONFINITE_VALIDATION_PREDICTION")
                validation_mse = float(
                    np.mean(
                        np.square(validation_prediction - validation_target),
                        dtype=np.float64,
                    )
                )
                if validation_mse < best_validation_mse:
                    best_validation_mse = validation_mse
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                epochs_run = epoch
                state = _training_state_payload(
                    model=model,
                    optimizer=optimizer,
                    best_state=best_state,
                    permutation_rng=permutation_rng,
                    epoch=epoch,
                    epoch_in_progress=False,
                    next_batch_number=0,
                    order=None,
                    best_validation_mse=best_validation_mse,
                    best_epoch=best_epoch,
                    stale_epochs=stale_epochs,
                    stop_reason=None,
                )
                _write_training_snapshot(writer, state)
                epoch_in_progress = False
                reason = None if stop_requested is None else stop_requested()
                if reason:
                    writer.abort(reason)
                    return {
                        "status": "SAFE_PAUSED",
                        "candidate_id": logical_id,
                        "reason": reason,
                        "resumable": True,
                        "staging_dir": str(writer.staging_dir),
                    }
                if stale_epochs >= patience:
                    break
            if best_state is None or best_epoch < 1:
                raise RuntimeError("STOP_NEURAL_CANDIDATE_HAS_NO_BEST_STATE")
            completion_state = _read_training_snapshot(writer)
            if completion_state is None:
                raise CandidateIntegrityError(
                    "STOP_COMPLETE_CANDIDATE_LATEST_RESUME_STATE_MISSING"
                )
            model.load_state_dict(best_state)
            validation_prediction = _predict(
                model,
                prepared.accessor,
                prepared.validation_samples,
                list(prepared.columns),
                prepared.candidate.history_steps,
                prepared.scaler,
                dynamic=prepared.dynamic,
                device=device,
            )
            validation_target = prepared.validation_samples["y_true"].to_numpy(
                dtype=np.float64
            )
            current_level = prepared.accessor.block_means(
                prepared.validation_samples,
                prepared.view.head.target,
                [(0, max(1, int(prepared.view.head.w0_steps)))],
            ).reshape(-1)
            metrics = _scalar_metric_bundle(
                validation_target, validation_prediction, current_level
            )
            metrics.update(
                {
                    "validation_mse": float(best_validation_mse),
                    "validation_rmse": float(math.sqrt(best_validation_mse)),
                    "best_epoch": int(best_epoch),
                    "epochs_run": int(epochs_run),
                    "training_seconds_this_process": float(time.time() - started),
                    "peak_vram_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    ),
                    "peak_process_rss_bytes": _peak_process_rss_bytes(),
                    "parameter_count": parameter_count(model),
                    "seed": int(seed),
                }
            )
            prediction_hash = _prediction_hash(validation_prediction)
            model_config = {
                **metadata["config"],
                "input_dim": len(prepared.columns) + (1 if prepared.dynamic else 0),
                "parameter_count": parameter_count(model),
                "best_epoch": int(best_epoch),
                "epochs_run": int(epochs_run),
                "model_weights_role": "BEST_VALIDATION_INFERENCE_STATE",
                "training_state_role": "LATEST_CONSISTENT_RESUMABLE_MODEL_OPTIMIZER_STATE",
            }
            writer.write_npz("model_weights.npz", _state_dict_arrays(best_state))
            final_arrays, final_schema = _pack_tree(completion_state)
            writer.write_npz("training_state.npz", final_arrays)
            writer.write_json("training_state.json", final_schema)
            writer.write_npz(
                "scaler.npz",
                {
                    "feature_mean": prepared.scaler.feature_mean,
                    "feature_scale": prepared.scaler.feature_scale,
                    "target_mean": np.asarray([prepared.scaler.target_mean]),
                    "target_scale": np.asarray([prepared.scaler.target_scale]),
                },
            )
            writer.write_json("scaler.json", prepared.scaler.to_json())
            writer.write_json("model_config.json", model_config)
            writer.write_json(
                "preprocessing.json", _portable_preprocessing(prepared)
            )
            writer.write_json("support_and_order.json", {
                "support": dict(prepared.support_details),
                "sample_order": dict(prepared.sample_order_details),
            })
            writer.write_json("validation_metrics.json", metrics)
            validation_sample_ids = prepared.validation_samples[
                "view_sample_id"
            ].astype(str).tolist()
            validation_base_ids = prepared.validation_samples[
                "base_origin_id"
            ].astype(str).tolist()
            sample_id_width = max(1, max(map(len, validation_sample_ids)))
            base_id_width = max(1, max(map(len, validation_base_ids)))
            validation_archive_path = writer.write_npz(
                "validation_predictions.npz",
                {
                    "view_sample_id": np.asarray(
                        validation_sample_ids, dtype=f"U{sample_id_width}"
                    ),
                    "base_origin_id": np.asarray(
                        validation_base_ids, dtype=f"U{base_id_width}"
                    ),
                    "y_true_delta": validation_target,
                    "y_pred_delta": validation_prediction,
                    "current_level": current_level,
                },
            )
            with np.load(validation_archive_path, allow_pickle=False) as archive:
                if archive["view_sample_id"].astype(str).tolist() != validation_sample_ids:
                    raise CandidateIntegrityError(
                        "STOP_SAVED_VALIDATION_SAMPLE_ID_ORDER_CHANGED"
                    )
                if archive["base_origin_id"].astype(str).tolist() != validation_base_ids:
                    raise CandidateIntegrityError(
                        "STOP_SAVED_VALIDATION_BASE_ID_ORDER_CHANGED"
                    )
            reason = None if stop_requested is None else stop_requested()
            if reason:
                writer.abort(reason)
                return {
                    "status": "SAFE_PAUSED",
                    "candidate_id": logical_id,
                    "reason": reason,
                    "resumable": True,
                    "staging_dir": str(writer.staging_dir),
                }
            validate_data_binding(
                prepared.shared,
                prepared.view,
                prepared.data_details,
                verify_sha256=True,
            )
            record = writer.finalize(
                validation_metrics=metrics,
                validation_prediction_hash=prediction_hash,
                seal=False,
            )
            return {
                "status": "TRAINED_AND_CACHED",
                "candidate_id": logical_id,
                "candidate_dir": str(cache.candidate_dir(logical_id)),
                "record": record,
                "metrics": metrics,
            }
        except CandidateIntegrityError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if writer.resumed and writer.staging_dir.is_dir():
                # A corrupt resume payload must never be silently overwritten;
                # preserve it in quarantine so an operator can inspect it.
                cache.quarantine_staging(logical_id, writer.staging_dir, reason)
                writer.finalized = True
            else:
                writer.abort(reason)
            raise
        except BaseException as exc:
            writer.abort(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if "model" in locals():
                del model
            if "optimizer" in locals():
                del optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


def _read_scaler(candidate_dir: Path) -> Scaler:
    path = Path(candidate_dir) / "scaler.npz"
    with np.load(path, allow_pickle=False) as archive:
        return Scaler(
            np.asarray(archive["feature_mean"], dtype=np.float64),
            np.asarray(archive["feature_scale"], dtype=np.float64),
            float(np.asarray(archive["target_mean"]).reshape(-1)[0]),
            float(np.asarray(archive["target_scale"]).reshape(-1)[0]),
        )


def _final_fit_prepared(
    prepared: PreparedCandidate,
    frozen_scaler: Scaler,
) -> PreparedCandidate:
    train = native_support(
        load_native_samples(prepared.shared, prepared.view, "train"),
        prepared.candidate.history_steps,
        dynamic=prepared.dynamic,
    ).reset_index(drop=True)
    validation = native_support(
        load_native_samples(prepared.shared, prepared.view, "validation"),
        prepared.candidate.history_steps,
        dynamic=prepared.dynamic,
    ).reset_index(drop=True)
    combined = pd.concat([train, validation], ignore_index=True)
    if combined.empty:
        raise RuntimeError("STOP_EMPTY_FROZEN_SELECTION_FINAL_FIT_SUPPORT")
    if combined["base_origin_id"].astype(str).duplicated().any():
        raise RuntimeError("STOP_DUPLICATE_TRAIN_VALIDATION_FINAL_FIT_SUPPORT")
    support_details = {
        "contract": SUPPORT_CONTRACT,
        "fit_scope": "TRAIN_PLUS_VALIDATION_NATIVE_SELECTED_HISTORY",
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "combined_rows": int(len(combined)),
        "train_base_origin_support_hash": base_origin_support_hash(train),
        "validation_base_origin_support_hash": base_origin_support_hash(validation),
        "combined_base_origin_support_hash": base_origin_support_hash(combined),
    }
    order_details = {
        "train_support_id_hash": support_id_hash(train),
        "validation_support_id_hash": support_id_hash(validation),
        "combined_support_id_hash": support_id_hash(combined),
        "combined_base_origin_order_hash": _ordered_hash(
            combined["base_origin_id"].astype(str).tolist()
        ),
    }
    return replace(
        prepared,
        fit_samples=combined,
        common_fit_samples=combined,
        scaler=frozen_scaler,
        support_details=support_details,
        sample_order_details=order_details,
        fit_row_cap=int(len(combined)),
    )


def train_frozen_selection_checkpoint(
    *,
    cache: NeuralCandidateCache,
    prepared: PreparedCandidate,
    selected_candidate_dir: Path,
    global_selection_freeze_path: Path,
    code_commit: str,
    device: torch.device,
    seed: int,
    frozen_best_epoch: int,
    direction: str | None = None,
    horizon_steps: int | None = None,
    stop_requested: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    """Historical final-fit entry, disabled by the current no-retrain rule."""

    raise RuntimeError("STOP_FINAL_FIT_RETRAIN_FORBIDDEN_BY_CURRENT_USER_RULE")

    if frozen_best_epoch < 1 or frozen_best_epoch > MAX_EPOCHS:
        raise ValueError("invalid frozen best epoch")
    freeze_path = Path(global_selection_freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "GLOBAL_SELECTION_FROZEN" or freeze.get("sealed") is not True:
        raise RuntimeError("STOP_FINAL_FIT_REQUIRES_GLOBAL_SELECTION_FREEZE")
    selected_candidate_dir = Path(selected_candidate_dir)
    selected_record = json.loads(
        (selected_candidate_dir / "CANDIDATE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    if selected_record.get("status") != "COMPLETE":
        raise RuntimeError("STOP_SELECTED_DEVELOPMENT_CANDIDATE_NOT_COMPLETE")
    model_config = json.loads(
        (selected_candidate_dir / "model_config.json").read_text(encoding="utf-8")
    )
    if (
        model_config.get("candidate_id") != prepared.candidate.candidate_id
        or model_config.get("model") != prepared.candidate.model
    ):
        raise RuntimeError("STOP_SELECTION_AND_FINAL_FIT_PROFILE_DISAGREE")
    frozen_scaler = _read_scaler(selected_candidate_dir)
    if len(frozen_scaler.feature_mean) != len(prepared.columns):
        raise RuntimeError("STOP_FROZEN_SCALER_FEATURE_ORDER_MISMATCH")
    final_prepared = _final_fit_prepared(prepared, frozen_scaler)
    hashes, metadata = candidate_provenance(
        final_prepared,
        phase=FINAL_FIT_PHASE,
        seed=seed,
        code_commit=code_commit,
        epochs=frozen_best_epoch,
        direction=direction,
        horizon_steps=horizon_steps,
    )
    metadata = dict(metadata)
    metadata.update(
        {
            "global_selection_freeze": file_record(freeze_path),
            "source_development_candidate_manifest": file_record(
                selected_candidate_dir / "CANDIDATE_MANIFEST.json"
            ),
            "source_development_candidate_id": selected_record["candidate_id"],
            "source_validation_prediction_hash": selected_record.get(
                "validation_prediction_hash"
            ),
            "source_validation_metrics": selected_record.get(
                "validation_metrics", {}
            ),
            "frozen_best_epoch": int(frozen_best_epoch),
        }
    )
    logical_id = logical_candidate_id(
        final_prepared,
        phase=FINAL_FIT_PHASE,
        seed=seed,
        direction=direction,
        horizon_steps=horizon_steps,
    )
    with cache.lock(logical_id, {"phase": FINAL_FIT_PHASE, "seed": seed}):
        decision = cache.reuse_status(logical_id, hashes)
        if decision.reusable:
            return {
                "status": "REUSED",
                "candidate_id": logical_id,
                "candidate_dir": str(decision.candidate_dir),
                "record": dict(decision.record or {}),
            }
        if decision.status in {"CORRUPT", "CORRUPT_QUARANTINED"}:
            raise CandidateIntegrityError(
                f"STOP_CORRUPT_FINAL_CHECKPOINT_REQUIRES_APPROVAL:{logical_id}"
            )
        if decision.status == "HASH_MISMATCH":
            raise CandidateConflictError(
                f"STOP_EXISTING_FINAL_CHECKPOINT_HASH_MISMATCH:{logical_id}"
            )
        writer = cache.begin_candidate(logical_id, hashes, metadata, resume=True)
        try:
            set_seed(seed)
            model = build_model(
                final_prepared.candidate.model,
                len(final_prepared.columns) + (1 if final_prepared.dynamic else 0),
                final_prepared.candidate.capacity,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=final_prepared.candidate.learning_rate,
                weight_decay=WEIGHT_DECAY,
            )
            if EFFECTIVE_BATCH_SIZE != PHYSICAL_BATCH_SIZE:
                raise RuntimeError(
                    "mid-epoch portable resume currently requires accumulation=1"
                )
            snapshot = _read_training_snapshot(writer) if writer.resumed else None
            if snapshot is None:
                permutation_rng = np.random.default_rng(seed)
                restored_epoch = 0
                epoch_in_progress = False
                next_batch_number = 0
                restored_order = None
            else:
                (
                    permutation_rng,
                    _unused_best_state,
                    restored_epoch,
                    epoch_in_progress,
                    next_batch_number,
                    restored_order,
                    _unused_best_mse,
                    _unused_best_epoch,
                    _unused_stale,
                ) = _restore_training_state(snapshot, model=model, optimizer=optimizer)
            started = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            first_epoch = restored_epoch if epoch_in_progress else restored_epoch + 1
            epochs_run = restored_epoch
            for epoch in range(max(1, first_epoch), frozen_best_epoch + 1):
                model.train()
                if epoch_in_progress and epoch == restored_epoch:
                    if restored_order is None:
                        raise CandidateIntegrityError(
                            "STOP_FINAL_RESUME_ORDER_MISSING"
                        )
                    order = restored_order
                    batch_start_number = next_batch_number
                else:
                    order = permutation_rng.permutation(len(final_prepared.fit_samples))
                    batch_start_number = 0
                starts = list(
                    range(0, len(order), PHYSICAL_BATCH_SIZE)
                )
                optimizer.zero_grad(set_to_none=True)
                for batch_number in range(batch_start_number, len(starts)):
                    start = starts[batch_number]
                    batch = final_prepared.fit_samples.iloc[
                        order[start : start + PHYSICAL_BATCH_SIZE]
                    ]
                    values, target_values = _scaled_batch(
                        final_prepared.accessor,
                        batch,
                        list(final_prepared.columns),
                        final_prepared.candidate.history_steps,
                        final_prepared.scaler,
                        dynamic=final_prepared.dynamic,
                        device=device,
                    )
                    loss = F.mse_loss(model(values), target_values)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRADIENT_CLIP_NORM
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    reason = None if stop_requested is None else stop_requested()
                    if reason:
                        state = _training_state_payload(
                            model=model,
                            optimizer=optimizer,
                            best_state=None,
                            permutation_rng=permutation_rng,
                            epoch=epoch,
                            epoch_in_progress=True,
                            next_batch_number=batch_number + 1,
                            order=order,
                            best_validation_mse=float("inf"),
                            best_epoch=frozen_best_epoch,
                            stale_epochs=0,
                            stop_reason=reason,
                        )
                        _write_training_snapshot(writer, state)
                        writer.abort(reason)
                        return {
                            "status": "SAFE_PAUSED",
                            "candidate_id": logical_id,
                            "reason": reason,
                            "resumable": True,
                            "staging_dir": str(writer.staging_dir),
                        }
                epochs_run = epoch
                state = _training_state_payload(
                    model=model,
                    optimizer=optimizer,
                    best_state=None,
                    permutation_rng=permutation_rng,
                    epoch=epoch,
                    epoch_in_progress=False,
                    next_batch_number=0,
                    order=None,
                    best_validation_mse=float("inf"),
                    best_epoch=frozen_best_epoch,
                    stale_epochs=0,
                    stop_reason=None,
                )
                _write_training_snapshot(writer, state)
                epoch_in_progress = False
            final_state = _training_state_payload(
                model=model,
                optimizer=optimizer,
                best_state=None,
                permutation_rng=permutation_rng,
                epoch=epochs_run,
                epoch_in_progress=False,
                next_batch_number=0,
                order=None,
                best_validation_mse=float("inf"),
                best_epoch=frozen_best_epoch,
                stale_epochs=0,
                stop_reason=None,
            )
            final_arrays, final_schema = _pack_tree(final_state)
            writer.write_npz("model_weights.npz", _state_dict_arrays(model.state_dict()))
            writer.write_npz("training_state.npz", final_arrays)
            writer.write_json("training_state.json", final_schema)
            writer.write_npz(
                "scaler.npz",
                {
                    "feature_mean": final_prepared.scaler.feature_mean,
                    "feature_scale": final_prepared.scaler.feature_scale,
                    "target_mean": np.asarray([final_prepared.scaler.target_mean]),
                    "target_scale": np.asarray([final_prepared.scaler.target_scale]),
                },
            )
            final_config = {
                **metadata["config"],
                "input_dim": len(final_prepared.columns)
                + (1 if final_prepared.dynamic else 0),
                "parameter_count": parameter_count(model),
                "frozen_best_epoch": int(frozen_best_epoch),
                "epochs_run": int(epochs_run),
            }
            writer.write_json("model_config.json", final_config)
            writer.write_json(
                "preprocessing.json", _portable_preprocessing(final_prepared)
            )
            writer.write_json(
                "support_and_order.json",
                {
                    "support": dict(final_prepared.support_details),
                    "sample_order": dict(final_prepared.sample_order_details),
                },
            )
            runtime = {
                "status": "PASS",
                "phase": FINAL_FIT_PHASE,
                "seed": int(seed),
                "frozen_best_epoch": int(frozen_best_epoch),
                "epochs_run": int(epochs_run),
                "training_seconds_this_process": float(time.time() - started),
                "peak_vram_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
                "peak_process_rss_bytes": _peak_process_rss_bytes(),
                "parameter_count": parameter_count(model),
                "test_accessed": False,
            }
            writer.write_json("final_fit_metrics.json", runtime)
            source_metrics = _json_safe(
                selected_record.get("validation_metrics", {})
            )
            record = writer.finalize(
                validation_metrics=source_metrics,
                validation_prediction_hash=selected_record.get(
                    "validation_prediction_hash"
                ),
                seal=False,
            )
            return {
                "status": "TRAINED_AND_CACHED",
                "candidate_id": logical_id,
                "candidate_dir": str(cache.candidate_dir(logical_id)),
                "record": record,
                "runtime": runtime,
            }
        except BaseException as exc:
            writer.abort(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if "model" in locals():
                del model
            if "optimizer" in locals():
                del optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


def load_cached_artifact(candidate_dir: Path, device: torch.device) -> LoadedArtifact:
    candidate_dir = Path(candidate_dir)
    config = json.loads(
        (candidate_dir / "model_config.json").read_text(encoding="utf-8")
    )
    preprocessing = json.loads(
        (candidate_dir / "preprocessing.json").read_text(encoding="utf-8")
    )
    model = build_model(
        str(config["model"]),
        int(config["input_dim"]),
        str(config["capacity"]),
    ).to(device)
    model.load_state_dict(_load_state_dict_npz(candidate_dir / "model_weights.npz"))
    model.eval()
    return LoadedArtifact(
        model=model,
        scaler=_read_scaler(candidate_dir),
        model_config=config,
        preprocessing=preprocessing,
        candidate_dir=candidate_dir,
    )


__all__ = [
    "FINAL_FIT_PHASE",
    "MISSING_VALUE_POLICY",
    "SELECTION_PHASE",
    "LoadedArtifact",
    "PreparedCandidate",
    "candidate_provenance",
    "build_data_binding",
    "load_cached_artifact",
    "logical_candidate_id",
    "prepare_candidate",
    "train_selection_candidate",
    "runtime_code_binding",
    "validate_data_binding",
]
