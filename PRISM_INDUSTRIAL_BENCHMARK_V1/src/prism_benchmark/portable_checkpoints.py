"""Portable, hash-addressed final-model checkpoint primitives.

The formal representative run deliberately keeps fitting and scoring in two
different processes.  This module contains the small numerical codecs shared
by those processes; none of the persisted formats require pickle.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .cpu_data import sha256_file
from .stage0 import write_json


FORMAT_VERSION = "PRISM_PORTABLE_CHECKPOINT_V1"
INFERENCE_ONLY_ENV = "PRISM_FORMAL_INFERENCE_ONLY"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_fitting_allowed() -> None:
    if os.environ.get(INFERENCE_ONLY_ENV) == "1":
        raise RuntimeError("STOP_FIT_REFIT_SELECT_FORBIDDEN_IN_INFERENCE_PROCESS")


def assert_inference_only() -> None:
    if os.environ.get(INFERENCE_ONLY_ENV) != "1":
        raise RuntimeError("STOP_TEST_PROCESS_MUST_SET_PRISM_FORMAL_INFERENCE_ONLY=1")


def _forbidden_fit(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError("STOP_FIT_REFIT_SELECT_FORBIDDEN_IN_INFERENCE_PROCESS")


def activate_inference_fit_guard() -> None:
    """Fail closed if legacy materialization accidentally attempts fitting."""

    assert_inference_only()
    from . import c2_models, c3_models, cpu_selection, v2_c, v211_joint_stability, v211_w, v21_a

    for module, names in (
        (c2_models, ("_ridge_predictions", "_pls_predictions", "_svr_fit_predict", "_xgb_fit_predict")),
        (c3_models, ("_ridge_block_predict",)),
        (cpu_selection, ("select_one_se",)),
        (v2_c, ("fit_physical_features", "_ridge_fit")),
        (v211_w, ("fit_w_correction", "_fit_c_routed")),
        (v21_a, ("fit_mature_residual_ar",)),
        (v211_joint_stability, ("fit_joint_candidate_stability", "select_predictive_eta", "select_k_representation")),
    ):
        for name in names:
            setattr(module, name, _forbidden_fit)
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import Ridge
        from sklearn.svm import SVR

        for cls in (PLSRegression, IsotonicRegression, Ridge, SVR):
            cls.fit = _forbidden_fit  # type: ignore[method-assign]
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor

        XGBRegressor.fit = _forbidden_fit  # type: ignore[method-assign]
    except ImportError:
        pass


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_key(*parts: str) -> str:
    cleaned = [str(part).replace("\\", "_").replace("/", "_") for part in parts]
    return "__".join(cleaned)


def write_portable_checkpoint(
    root: Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray] | None = None,
    *,
    native_files: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Write one checkpoint directory and return its content manifest."""

    assert_fitting_allowed()
    if root.exists():
        raise RuntimeError(f"refusing to overwrite final checkpoint: {root}")
    root.mkdir(parents=True, exist_ok=False)
    array_path = root / "arrays.npz"
    normalized = {
        key: np.asarray(value)
        for key, value in sorted((arrays or {}).items())
    }
    np.savez_compressed(array_path, **normalized)
    state = {
        "format": FORMAT_VERSION,
        "created_utc": _utc(),
        "deletion_forbidden": True,
        **dict(metadata),
        "array_names": sorted(normalized),
    }
    write_json(root / "checkpoint.json", state)
    for name, source in sorted((native_files or {}).items()):
        if Path(name).name != name:
            raise ValueError(f"native checkpoint name must be a basename: {name}")
        destination = root / name
        destination.write_bytes(Path(source).read_bytes())
    files = [path for path in sorted(root.iterdir()) if path.is_file()]
    result = {
        "format": FORMAT_VERSION,
        "checkpoint_dir": str(root),
        "deletion_forbidden": True,
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    result["checkpoint_hash"] = stable_hash(result["files"])
    write_json(root / "MANIFEST.json", result)
    # MANIFEST is self-excluding so it can be verified without a recursive hash.
    return result


def load_portable_checkpoint(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    assert_inference_only()
    state_path = root / "checkpoint.json"
    array_path = root / "arrays.npz"
    manifest_path = root / "MANIFEST.json"
    if not (state_path.is_file() and array_path.is_file() and manifest_path.is_file()):
        raise FileNotFoundError(f"incomplete portable checkpoint: {root}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if state.get("format") != FORMAT_VERSION or state.get("deletion_forbidden") is not True:
        raise RuntimeError(f"invalid portable checkpoint state: {root}")
    expected = {item["name"]: item for item in manifest.get("files", [])}
    for name, record in expected.items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"STOP_CHECKPOINT_HASH_MISMATCH:{path}")
    if stable_hash(manifest.get("files", [])) != manifest.get("checkpoint_hash"):
        raise RuntimeError(f"STOP_CHECKPOINT_MANIFEST_HASH_MISMATCH:{root}")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if sorted(arrays) != sorted(state.get("array_names", [])):
        raise RuntimeError(f"STOP_CHECKPOINT_ARRAY_SCHEMA_MISMATCH:{root}")
    return state, arrays, manifest


def seal_checkpoint_tree(root: Path) -> None:
    """Best-effort cross-platform read-only seal after the manifest is complete."""

    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def fit_standardized_ridge(
    x: np.ndarray, y: np.ndarray, penalties: float | np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    assert_fitting_allowed()
    matrix = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = matrix.mean(axis=0, dtype=np.float64)
    scale = matrix.std(axis=0, dtype=np.float64)
    scale[scale * scale < 1e-12] = 1.0
    standardized = (matrix - mean) / scale
    intercept = float(target.mean(dtype=np.float64))
    centered = target - intercept
    penalty = np.broadcast_to(np.asarray(penalties, dtype=np.float64), matrix.shape[1])
    system = standardized.T @ standardized + np.diag(penalty)
    rhs = standardized.T @ centered
    try:
        coefficient = np.linalg.solve(system, rhs)
        solver = "solve"
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(system, rhs, rcond=1e-12)[0]
        solver = "svd_rescue"
    return (
        {"codec": "STANDARDIZED_RIDGE", "intercept": intercept, "solver": solver},
        {"mean": mean, "scale": scale, "coefficient": coefficient},
    )


def predict_standardized_ridge(
    x: np.ndarray, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> np.ndarray:
    matrix = np.asarray(x, dtype=np.float64)
    return (
        (matrix - arrays["mean"]) / arrays["scale"]
    ) @ arrays["coefficient"] + float(metadata["intercept"])


def fit_pls_codec(
    x: np.ndarray, y: np.ndarray, components: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    assert_fitting_allowed()
    from sklearn.cross_decomposition import PLSRegression

    matrix = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    model = PLSRegression(
        n_components=int(components), scale=True, max_iter=1000, tol=1e-8
    ).fit(matrix, target)
    coefficient = np.asarray(model.coef_, dtype=np.float64)
    if coefficient.shape == (matrix.shape[1], 1):
        coefficient = coefficient.T
    arrays = {
        "x_mean": np.asarray(model._x_mean, dtype=np.float64),
        "x_std": np.asarray(model._x_std, dtype=np.float64),
        "coefficient": coefficient,
        "intercept": np.asarray(model.intercept_, dtype=np.float64).reshape(-1),
    }
    reference = np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1)
    centered = (matrix - arrays["x_mean"]) @ coefficient.T + arrays["intercept"]
    standardized = (
        (matrix - arrays["x_mean"]) / arrays["x_std"]
    ) @ coefficient.T + arrays["intercept"]
    if np.allclose(centered.reshape(-1), reference, rtol=1e-10, atol=1e-10):
        transform = "CENTER_ONLY_COEFFICIENT_ALREADY_SCALED"
    elif np.allclose(standardized.reshape(-1), reference, rtol=1e-10, atol=1e-10):
        transform = "STANDARDIZE_X"
    else:
        raise RuntimeError("STOP_PLS_PORTABLE_REPLAY_MISMATCH")
    metadata = {
        "codec": "PLS_REGRESSION",
        "components": int(components),
        "x_transform": transform,
    }
    return metadata, arrays


def predict_pls_codec(
    x: np.ndarray, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> np.ndarray:
    del metadata
    matrix = np.asarray(x, dtype=np.float64) - arrays["x_mean"]
    if metadata["x_transform"] == "STANDARDIZE_X":
        matrix = matrix / arrays["x_std"]
    elif metadata["x_transform"] != "CENTER_ONLY_COEFFICIENT_ALREADY_SCALED":
        raise ValueError(f"unsupported PLS transform: {metadata['x_transform']}")
    return (matrix @ arrays["coefficient"].T + arrays["intercept"]).reshape(-1)


def fit_rbf_svr_codec(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
    gamma: float,
    epsilon: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    assert_fitting_allowed()
    from sklearn.svm import SVR

    matrix = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    x_mean = matrix.mean(axis=0, dtype=np.float64)
    x_scale = matrix.std(axis=0, dtype=np.float64)
    x_scale[x_scale == 0.0] = 1.0
    y_mean = float(target.mean(dtype=np.float64))
    y_scale = float(target.std(dtype=np.float64)) or 1.0
    model = SVR(
        C=float(c_value), gamma=float(gamma), epsilon=float(epsilon),
        kernel="rbf", cache_size=4096,
    ).fit((matrix - x_mean) / x_scale, (target - y_mean) / y_scale)
    metadata = {
        "codec": "RBF_SVR",
        "gamma": float(gamma),
        "y_mean": y_mean,
        "y_scale": y_scale,
        "c": float(c_value),
        "epsilon": float(epsilon),
    }
    arrays = {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "support_vectors": np.asarray(model.support_vectors_, dtype=np.float64),
        "dual_coef": np.asarray(model.dual_coef_, dtype=np.float64).reshape(-1),
        "intercept": np.asarray(model.intercept_, dtype=np.float64).reshape(-1),
    }
    replay = predict_rbf_svr_codec(matrix, metadata, arrays)
    reference = np.asarray(model.predict((matrix - x_mean) / x_scale) * y_scale + y_mean)
    if not np.allclose(replay, reference, rtol=1e-10, atol=1e-10):
        raise RuntimeError("STOP_SVR_PORTABLE_REPLAY_MISMATCH")
    return metadata, arrays


def predict_rbf_svr_codec(
    x: np.ndarray, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> np.ndarray:
    matrix = (np.asarray(x, dtype=np.float64) - arrays["x_mean"]) / arrays["x_scale"]
    support = arrays["support_vectors"]
    squared = (
        np.sum(matrix * matrix, axis=1)[:, None]
        + np.sum(support * support, axis=1)[None, :]
        - 2.0 * matrix @ support.T
    )
    kernel = np.exp(-float(metadata["gamma"]) * np.maximum(squared, 0.0))
    standardized = kernel @ arrays["dual_coef"] + float(arrays["intercept"][0])
    return standardized * float(metadata["y_scale"]) + float(metadata["y_mean"])


def predict_codec(
    x: np.ndarray, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> np.ndarray:
    codec = str(metadata["codec"])
    if codec == "CONSTANT":
        return np.full(len(x), float(metadata["value"]), dtype=np.float64)
    if codec == "STANDARDIZED_RIDGE":
        return predict_standardized_ridge(x, metadata, arrays)
    if codec == "PLS_REGRESSION":
        return predict_pls_codec(x, metadata, arrays)
    if codec == "RBF_SVR":
        return predict_rbf_svr_codec(x, metadata, arrays)
    raise ValueError(f"unsupported portable codec: {codec}")
