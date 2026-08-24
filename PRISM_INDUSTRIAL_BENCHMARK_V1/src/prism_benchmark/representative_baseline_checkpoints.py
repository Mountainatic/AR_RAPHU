"""Final-fit and inference-only materialization for registered CPU baselines."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .c3_models import _arx_features
from .cpu_data import BaseAccessor, ViewSpec, input_columns, sha256_file
from .level_reconstruction import metric_bundle_delta_and_level, support_hash
from .portable_checkpoints import (
    assert_fitting_allowed,
    assert_inference_only,
    checkpoint_key,
    fit_pls_codec,
    fit_rbf_svr_codec,
    fit_standardized_ridge,
    load_portable_checkpoint,
    predict_codec,
    predict_standardized_ridge,
    stable_hash,
    write_portable_checkpoint,
)
from .stage0 import write_json
from .v211_public_all_baseline_materialization import (
    _common_test,
    _development,
    _freeze,
    _result,
    _static_native,
    baseline_candidates,
)
from .v211_public_all_baselines import SupportRequirement, _cap_after_support
from .v211_public_all_config import PublicAllPaths
from .v211_public_all_materialization import _prediction_frame, _prediction_root
from .v211_support import support_id_hash


FORMAL_COMPATIBLE = {
    "MEAN",
    "PERSISTENCE",
    "RIDGE",
    "PLS",
    "RBF_SVR",
    "XGBOOST",
    "DPLS",
    "PARALLEL_HAMMERSTEIN",
    "HAMMERSTEIN_WIENER",
    "AR",
    "ARX",
    "LINEAR_NARX",
}


def _checkpoint_dir(root: Path, view: ViewSpec, model: str) -> Path:
    return root / "baseline" / checkpoint_key(
        view.head.dataset,
        view.head.head_id,
        view.information_set,
        view.availability_scenario,
        view.proxy_policy,
        model,
    )


def _fit_samples(paths: PublicAllPaths, view: ViewSpec, model: str, selection: Mapping[str, Any]):
    freeze = _freeze(paths.project)
    default_cap = int(freeze["selection"]["fit_row_cap_default"])
    if model in {"MEAN", "PERSISTENCE"}:
        return _development(paths.shared, view, [SupportRequirement()])
    if model in {"RIDGE", "PLS", "RBF_SVR", "XGBOOST"}:
        return _static_native(paths, view, model)
    if model == "DPLS":
        requirement = SupportRequirement(input_history_steps=int(selection["selected_history"]))
    elif model in {"PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"}:
        profile = tuple(int(value) for value in selection["selected_profile"])
        requirement = SupportRequirement(input_history_steps=profile[1])
    elif model in {"AR", "ARX", "LINEAR_NARX"}:
        profile = tuple(int(value) for value in selection["selected_profile"])
        requirement = SupportRequirement(
            input_history_steps=profile[1] if model != "AR" else 0,
            target_delta_steps=profile[0],
            target_history_steps=profile[1],
        )
    else:
        raise KeyError(model)
    return _cap_after_support(_development(paths.shared, view, [requirement]), default_cap)


def _nonlinear_fit(raw: np.ndarray, kind: str) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    mean = raw.mean(axis=0, dtype=np.float64)
    scale = raw.std(axis=0, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    standard = (raw - mean) / scale
    pieces = [standard]
    arrays: dict[str, np.ndarray] = {"raw_mean": mean, "raw_scale": scale}
    if kind in {"quadratic_centered", "cubic_centered"}:
        arrays["square_mean"] = np.square(standard).mean(axis=0, dtype=np.float64)
        pieces.append(np.square(standard) - arrays["square_mean"])
    if kind == "cubic_centered":
        arrays["cube_mean"] = np.power(standard, 3).mean(axis=0, dtype=np.float64)
        pieces.append(np.power(standard, 3) - arrays["cube_mean"])
    if kind == "piecewise_linear_5_quantile_knots":
        arrays["knots"] = np.quantile(
            standard, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6], axis=0
        )
        pieces.extend(np.maximum(standard - knot, 0.0) for knot in arrays["knots"])
    return np.concatenate(pieces, axis=1), {"kind": kind}, arrays


def _nonlinear_predict(raw: np.ndarray, feature: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    standard = (np.asarray(raw, dtype=np.float64) - arrays["raw_mean"]) / arrays["raw_scale"]
    pieces = [standard]
    kind = str(feature["kind"])
    if kind in {"quadratic_centered", "cubic_centered"}:
        pieces.append(np.square(standard) - arrays["square_mean"])
    if kind == "cubic_centered":
        pieces.append(np.power(standard, 3) - arrays["cube_mean"])
    if kind == "piecewise_linear_5_quantile_knots":
        pieces.extend(np.maximum(standard - knot, 0.0) for knot in arrays["knots"])
    return np.concatenate(pieces, axis=1)


def _merge_arrays(*groups: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for prefix_index, group in enumerate(groups):
        prefix = "" if prefix_index == len(groups) - 1 else f"p{prefix_index}__"
        for key, value in group.items():
            merged[f"{prefix}{key}"] = np.asarray(value)
    return merged


def _prefixed(arrays: Mapping[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {key[len(prefix) :]: value for key, value in arrays.items() if key.startswith(prefix)}


def _fit_xgboost(
    x: np.ndarray, y: np.ndarray, config: Mapping[str, Any], selection: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, Any, np.ndarray]:
    assert_fitting_allowed()
    from xgboost import XGBRegressor

    pair = selection["selected_pair"]["candidate"]
    model = XGBRegressor(
        max_depth=int(pair[0]),
        learning_rate=float(pair[1]),
        n_estimators=int(selection["selected_estimators"]["candidate"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        min_child_weight=float(config["min_child_weight"]),
        reg_lambda=float(config["reg_lambda"]),
        tree_method=str(config["tree_method"]),
        n_jobs=int(config["nthread_per_fit"]),
        objective="reg:squarederror",
        random_state=20260802,
    ).fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), verbose=False)
    temporary = tempfile.TemporaryDirectory(prefix="prism-xgb-")
    path = Path(temporary.name) / "xgboost.json"
    model.save_model(path)
    leaf_count = int(sum(tree.count("leaf") for tree in model.get_booster().get_dump()))
    replay = np.asarray(model.predict(np.asarray(x, dtype=np.float64)[:32]), dtype=np.float64)
    return {"codec": "XGBOOST_NATIVE_JSON", "parameter_count": leaf_count}, path, temporary, replay


def _fit_one(
    paths: PublicAllPaths,
    view: ViewSpec,
    model: str,
    result: Mapping[str, Any],
    checkpoint_root: Path,
) -> dict[str, Any]:
    assert_fitting_allowed()
    selection = dict(result.get("selection", {}))
    fit = _fit_samples(paths, view, model, selection)
    y = fit["y_true"].to_numpy(dtype=np.float64)
    freeze = _freeze(paths.project)
    columns = input_columns(paths.shared, view.head.task_id, view.proxy_policy)
    accessor_columns = [view.head.target, *columns] if model in {"ARX", "LINEAR_NARX"} else columns
    accessor = BaseAccessor(paths.shared, view.head.dataset, "validation", accessor_columns or [view.head.target])
    feature: dict[str, Any] = {"columns": columns}
    arrays: dict[str, np.ndarray] = {}
    native: dict[str, Path] = {}
    temporary: tempfile.TemporaryDirectory[str] | None = None

    if model in {"MEAN", "PERSISTENCE"}:
        codec = {"codec": "CONSTANT", "value": float(y.mean()) if model == "MEAN" else 0.0}
        parameter_count = 1 if model == "MEAN" else 0
        feature["family"] = "NONE"
        x = np.zeros((len(fit), 0), dtype=np.float64)
    elif model in {"RIDGE", "PLS", "RBF_SVR", "XGBOOST"}:
        x = accessor.snapshot(fit, columns)
        feature["family"] = "SNAPSHOT_T_MINUS_1"
        if model == "RIDGE":
            codec, arrays = fit_standardized_ridge(x, y, float(selection["selected_alpha"]))
            parameter_count = x.shape[1] + 1
        elif model == "PLS":
            codec, arrays = fit_pls_codec(x, y, int(selection["selected_components"]))
            parameter_count = int(selection["selected_components"]) * (x.shape[1] + 2) + 1
        elif model == "RBF_SVR":
            codec, arrays = fit_rbf_svr_codec(
                x,
                y,
                c_value=float(selection["selected_c"]["candidate"]),
                gamma=float(selection["selected_gamma"]["candidate"]),
                epsilon=float(freeze["c2"]["rbf_svr"]["epsilon_standardized"]),
            )
            parameter_count = int(len(arrays["support_vectors"]) * (x.shape[1] + 1))
        else:
            codec, native_path, temporary, replay_prediction = _fit_xgboost(
                x, y, freeze["c2"]["xgboost"], selection
            )
            native = {"xgboost.json": native_path}
            parameter_count = int(codec["parameter_count"])
    elif model == "DPLS":
        history = int(selection["selected_history"])
        maximum = int(freeze["c2"]["dpls"]["maximum_lags_per_channel"])
        x = accessor.input_lags(fit, columns, history, maximum)
        feature.update({"family": "INPUT_LAGS", "history": history, "maximum_lags": maximum})
        codec, arrays = fit_pls_codec(x, y, int(selection["selected_components"]))
        parameter_count = int(selection["selected_components"]) * (x.shape[1] + 2) + 1
    elif model in {"PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"}:
        profile = tuple(int(value) for value in selection["selected_profile"])
        raw = accessor.input_regular_lags(fit, columns, *profile, 8)
        expanded, nonlinear, preprocessing = _nonlinear_fit(
            raw, str(selection["selected_input_nonlinearity"])
        )
        codec, ridge_arrays = fit_standardized_ridge(expanded, y, 0.001)
        x = expanded
        arrays = _merge_arrays(preprocessing, ridge_arrays)
        feature.update({"family": "HAMMERSTEIN", "profile": list(profile), "maximum_lags": 8, "nonlinear": nonlinear})
        if selection["selected_output_map"] == "isotonic_train_only":
            from sklearn.isotonic import IsotonicRegression

            train_prediction = predict_standardized_ridge(expanded, codec, ridge_arrays)
            isotonic = IsotonicRegression(out_of_bounds="clip").fit(train_prediction, y)
            arrays["isotonic_x"] = np.asarray(isotonic.X_thresholds_, dtype=np.float64)
            arrays["isotonic_y"] = np.asarray(isotonic.y_thresholds_, dtype=np.float64)
            codec["output_map"] = "ISOTONIC_CLIP"
        else:
            codec["output_map"] = "IDENTITY"
        parameter_count = expanded.shape[1] + 1
    elif model == "AR":
        profile = tuple(int(value) for value in selection["selected_profile"])
        state_accessor = BaseAccessor(paths.shared, view.head.dataset, "validation", [view.head.target])
        x = state_accessor.target_state(fit, view.head.target, *profile)
        feature.update({"family": "TARGET_STATE", "profile": list(profile)})
        codec, arrays = fit_standardized_ridge(x, y, float(selection["selected_alpha"]))
        parameter_count = x.shape[1] + 1
    elif model in {"ARX", "LINEAR_NARX"}:
        profile = tuple(int(value) for value in selection["selected_profile"])
        maximum = int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"])
        raw, ar_width = _arx_features(accessor, fit, view, columns, profile, maximum)
        feature.update({"family": model, "profile": list(profile), "maximum_input_lags": maximum})
        if model == "ARX":
            alpha = float(selection["ar_alpha"])
            ratio = selection["selected_x_penalty_ratio"]
            if ratio == "EXACT_X_ZERO":
                raw = raw[:, :ar_width]
                penalties: float | np.ndarray = alpha
                feature["feature_width"] = ar_width
            else:
                penalties = np.concatenate(
                    [np.full(ar_width, alpha), np.full(raw.shape[1] - ar_width, alpha * float(ratio))]
                )
            codec, arrays = fit_standardized_ridge(raw, y, penalties)
            x = raw
            parameter_count = raw.shape[1] + 1
        else:
            raw_mean = raw.mean(axis=0, dtype=np.float64)
            raw_scale = raw.std(axis=0, dtype=np.float64)
            raw_scale[raw_scale == 0.0] = 1.0
            standardized = (raw - raw_mean) / raw_scale
            centered_y = y - y.mean(dtype=np.float64)
            denominator = np.sqrt(np.sum(np.square(standardized), axis=0) * np.sum(np.square(centered_y)))
            correlations = np.abs((standardized.T @ centered_y) / np.where(denominator > 0, denominator, np.inf))
            maximum_features = int(
                freeze["c3"]["linear_narx"]["maximum_linear_state_features_before_expansion"]
            )
            order = np.lexsort((np.arange(len(correlations)), -correlations))[:maximum_features]
            selected = standardized[:, order]
            expanded = np.concatenate([selected, np.square(selected)], axis=1)
            x = expanded
            codec, ridge_arrays = fit_standardized_ridge(
                expanded, y, float(selection["selected_alpha"])
            )
            arrays = {
                "raw_mean": raw_mean,
                "raw_scale": raw_scale,
                "selected_features": order,
                **ridge_arrays,
            }
            parameter_count = expanded.shape[1] + 1
    else:
        raise KeyError(model)

    replay_rows = min(32, len(fit))
    if model != "XGBOOST":
        if feature["family"] == "HAMMERSTEIN":
            replay_prediction = predict_standardized_ridge(
                x[:replay_rows], codec, _prefixed(arrays, "")
            )
            if codec.get("output_map") == "ISOTONIC_CLIP":
                replay_prediction = np.interp(
                    replay_prediction,
                    arrays["isotonic_x"],
                    arrays["isotonic_y"],
                    left=float(arrays["isotonic_y"][0]),
                    right=float(arrays["isotonic_y"][-1]),
                )
        else:
            replay_prediction = predict_codec(x[:replay_rows], codec, arrays)
    arrays["__replay_x"] = np.asarray(x[:replay_rows], dtype=np.float64)
    arrays["__replay_y"] = np.asarray(replay_prediction, dtype=np.float64)
    metadata = {
        **codec,
        "artifact_type": "FORMAL_FINAL_MODEL",
        "family": "CPU_BASELINE",
        "model": model,
        "dataset": view.head.dataset,
        "task": view.head.task_id,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "selection": selection,
        "selection_hash": stable_hash(selection),
        "feature": feature,
        "feature_order": columns,
        "missing_value_policy": "REJECT_NONFINITE_C1",
        "fit_partition": "train_plus_validation",
        "fit_rows": int(len(fit)),
        "fit_support_hash": support_id_hash(fit),
        "parameter_count": int(parameter_count),
        "reload_prediction_tolerance": 1e-10,
    }
    try:
        manifest = write_portable_checkpoint(
            _checkpoint_dir(checkpoint_root, view, model), metadata, arrays, native_files=native
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    return {
        "status": "PASS",
        "model": model,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "checkpoint_hash": manifest["checkpoint_hash"],
        "checkpoint_dir": str(_checkpoint_dir(checkpoint_root, view, model)),
        "fit_rows": int(len(fit)),
        "fit_support_hash": support_id_hash(fit),
    }


def fit_baseline_checkpoints_for_view(
    paths: PublicAllPaths, view: ViewSpec, checkpoint_root: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family, model, output_model in baseline_candidates(view):
        result = _result(paths, family, model, view)
        development_status = None if result is None else str(result.get("status"))
        if model not in FORMAL_COMPATIBLE:
            records.append({
                "status": "NOT_RUN_PROTOCOL_INCOMPATIBLE",
                "model": output_model,
                "reason": "FINAL_PORTABLE_REFIT_NOT_REGISTERED",
                "test_accessed": False,
            })
        elif development_status != "PASS":
            records.append({
                "status": development_status or "FAILED_RETAINED",
                "model": output_model,
                "reason": "DEVELOPMENT_SELECTION_NOT_PASS",
                "test_accessed": False,
            })
        else:
            records.append(_fit_one(paths, view, output_model, result, checkpoint_root))
    return records


def _evaluation_features(
    paths: PublicAllPaths,
    view: ViewSpec,
    samples: pd.DataFrame,
    split: str,
    state: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    feature = state["feature"]
    family = str(feature["family"])
    columns = list(feature.get("columns", []))
    if family == "NONE":
        return np.zeros((len(samples), 0), dtype=np.float64)
    accessor_columns = [view.head.target, *columns] if family in {"ARX", "LINEAR_NARX"} else columns
    accessor = BaseAccessor(paths.shared, view.head.dataset, split, accessor_columns or [view.head.target])
    if family == "SNAPSHOT_T_MINUS_1":
        return accessor.snapshot(samples, columns)
    if family == "INPUT_LAGS":
        return accessor.input_lags(samples, columns, int(feature["history"]), int(feature["maximum_lags"]))
    if family == "HAMMERSTEIN":
        raw = accessor.input_regular_lags(samples, columns, *tuple(feature["profile"]), int(feature["maximum_lags"]))
        return _nonlinear_predict(raw, feature["nonlinear"], _prefixed(arrays, "p0__"))
    if family == "TARGET_STATE":
        return accessor.target_state(samples, view.head.target, *tuple(feature["profile"]))
    if family in {"ARX", "LINEAR_NARX"}:
        raw, _ = _arx_features(
            accessor, samples, view, columns, tuple(feature["profile"]), int(feature["maximum_input_lags"])
        )
        if family == "ARX":
            return raw[:, : int(feature["feature_width"])] if "feature_width" in feature else raw
        standardized = (raw - arrays["raw_mean"]) / arrays["raw_scale"]
        selected = standardized[:, arrays["selected_features"].astype(np.int64)]
        return np.concatenate([selected, np.square(selected)], axis=1)
    raise ValueError(f"unknown feature family: {family}")


def _current_levels(paths: PublicAllPaths, view: ViewSpec, samples: pd.DataFrame, split: str) -> np.ndarray:
    if "current_level" in samples.columns:
        return samples["current_level"].to_numpy(dtype=np.float64)
    accessor = BaseAccessor(paths.shared, view.head.dataset, split, [view.head.target])
    return accessor.block_means(samples, view.head.target, [(0, int(view.head.w0_steps))]).reshape(-1)


def _predict_loaded(
    checkpoint: Path,
    x: np.ndarray,
    state: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    if state["codec"] == "XGBOOST_NATIVE_JSON":
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(checkpoint / "xgboost.json")
        return np.asarray(model.predict(x), dtype=np.float64)
    if state["feature"]["family"] == "HAMMERSTEIN":
        prediction = predict_standardized_ridge(x, state, _prefixed(arrays, ""))
        # Ridge arrays are unprefixed because they are the last merged group.
        if state.get("output_map") == "ISOTONIC_CLIP":
            prediction = np.interp(
                prediction,
                arrays["isotonic_x"],
                arrays["isotonic_y"],
                left=float(arrays["isotonic_y"][0]),
                right=float(arrays["isotonic_y"][-1]),
            )
        return prediction
    return predict_codec(x, state, arrays)


def verify_baseline_checkpoint_reload(checkpoint: Path) -> dict[str, Any]:
    """Numerically replay the frozen calibration rows after reloading from disk."""

    assert_inference_only()
    state, arrays, manifest = load_portable_checkpoint(checkpoint)
    observed = _predict_loaded(
        checkpoint, arrays["__replay_x"], state, arrays
    )
    expected = arrays["__replay_y"]
    maximum = float(np.max(np.abs(observed - expected), initial=0.0))
    tolerance = float(state.get("reload_prediction_tolerance", 1e-10))
    if maximum > tolerance:
        raise RuntimeError(f"STOP_BASELINE_CHECKPOINT_RELOAD_MISMATCH:{checkpoint}:{maximum}")
    return {
        "status": "PASS",
        "checkpoint_dir": str(checkpoint),
        "checkpoint_hash": manifest["checkpoint_hash"],
        "rows": int(len(expected)),
        "maximum_absolute_prediction_error": maximum,
        "tolerance": tolerance,
    }


def predict_baseline_checkpoints_for_view(
    paths: PublicAllPaths,
    view: ViewSpec,
    checkpoint_root: Path,
    *,
    split: str = "test",
) -> list[dict[str, Any]]:
    assert_inference_only()
    samples = _common_test(paths, view, split)
    records: list[dict[str, Any]] = []
    for _, model, output_model in baseline_candidates(view):
        checkpoint = _checkpoint_dir(checkpoint_root, view, output_model)
        if not checkpoint.is_dir():
            records.append({
                "status": "NOT_RUN_PROTOCOL_INCOMPATIBLE" if model not in FORMAL_COMPATIBLE else "FAILED_RETAINED",
                "model": output_model,
                "reason": "SEALED_CHECKPOINT_ABSENT",
                "test_accessed": False,
            })
            continue
        started = time.time()
        state, arrays, manifest = load_portable_checkpoint(checkpoint)
        x = _evaluation_features(paths, view, samples, split, state, arrays)
        prediction = _predict_loaded(checkpoint, x, state, arrays)
        frame = _prediction_frame(samples, view, output_model, prediction, int(state["parameter_count"]))
        frame["split"] = split
        destination = _prediction_root(paths, split, baseline=True) / view.relative_root / f"{output_model}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False, compression="zstd")
        metrics = metric_bundle_delta_and_level(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
            _current_levels(paths, view, samples, split),
        )
        metrics.pop("future_level_true")
        metrics.pop("future_level_pred")
        row_ids = frame["sample_id"].astype(str).tolist()
        records.append({
            "status": "PASS",
            "dataset": view.head.dataset,
            "target_head": view.head.head_id,
            "information_set": view.information_set,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "model": output_model,
            "split": split,
            "rows": int(len(frame)),
            "sample_id_order_hash": support_hash(row_ids),
            "scoring_support_hash": support_id_hash(samples),
            "fit_support_hash": state["fit_support_hash"],
            "checkpoint_hash": manifest["checkpoint_hash"],
            "checkpoint_dir": str(checkpoint),
            "prediction_path": str(destination.relative_to(paths.run_root)),
            "prediction_sha256": sha256_file(destination),
            "test_accessed": split == "test",
            "ood_accessed": split == "ood",
            "fit_called_in_inference": False,
            "elapsed_seconds": time.time() - started,
            **metrics,
        })
    write_json(
        _prediction_root(paths, split, baseline=True) / view.relative_root / "INFERENCE_RESULT.json",
        {"status": "PASS", "models": records, "inference_only": True},
    )
    return records
