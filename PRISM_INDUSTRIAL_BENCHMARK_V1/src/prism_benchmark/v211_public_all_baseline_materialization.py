from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .c2_models import (
    _pls_predictions,
    _ridge_predictions,
    _svr_fit_predict,
    _xgb_fit_predict,
)
from .c3_models import (
    _arx_features,
    _narx_expand,
    _nonlinear_features,
    _ridge_block_predict,
)
from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    input_columns,
    sha256_file,
)
from .cpu_selection import regression_metrics
from .v211_public_all_baselines import (
    SupportRequirement,
    _cap_after_support,
)
from .v211_public_all_closure import common_support_record
from .v211_public_all_config import PublicAllPaths
from .v211_public_all_materialization import (
    _development,
    _filtered_split,
    _prediction_frame,
    _prediction_root,
)
from .v211_support import support_id_hash


def _freeze(project: Path) -> dict[str, Any]:
    return json.loads(
        (project / "configs/cpu_model_freeze_v1.json").read_text(encoding="utf-8")
    )


def _result(paths: PublicAllPaths, family: str, model: str, view: ViewSpec) -> dict[str, Any] | None:
    path = (
        paths.output
        / "BASELINE_DEVELOPMENT"
        / family
        / "PREDICTIONS"
        / model
        / view.relative_root
        / "RESULT.json"
    )
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _common_test(paths: PublicAllPaths, view: ViewSpec, split: str):
    record = common_support_record(paths, view)
    requirements = tuple(
        SupportRequirement(**item) for item in record.get("requirements", ())
    ) or (SupportRequirement(),)
    return _filtered_split(paths.shared, view, split, requirements)


def _write(
    paths: PublicAllPaths,
    view: ViewSpec,
    model: str,
    samples,
    prediction: np.ndarray,
    native_fit,
    started: float,
    *,
    selection: dict[str, Any],
    parameter_count: int,
    split: str,
) -> dict[str, Any]:
    destination = (
        _prediction_root(paths, split, baseline=True)
        / view.relative_root
        / f"{model}.parquet"
    )
    if set(samples["split"].astype(str).unique()) != {split}:
        raise RuntimeError(f"baseline samples are not isolated to {split}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = _prediction_frame(
        samples, view, model, prediction, int(parameter_count)
    )
    frame.to_parquet(destination, index=False, compression="zstd")
    return {
        "status": "PASS",
        "dataset": view.head.dataset,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "model": model,
        "split": split,
        "rows": len(frame),
        "native_fit_rows": len(native_fit),
        "native_fit_support_hash": support_id_hash(native_fit),
        "scoring_support_hash": support_id_hash(samples),
        "parameter_count": int(parameter_count),
        "selection": selection,
        "prediction_path": str(destination.relative_to(paths.run_root)),
        "prediction_sha256": sha256_file(destination),
        "test_accessed": split == "test",
        "ood_accessed": split == "ood",
        "elapsed_seconds": time.time() - started,
        **regression_metrics(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        ),
    }


def _not_run(
    view: ViewSpec,
    model: str,
    reason: str,
    *,
    status: str = "NOT_RUN_PROTOCOL_INCOMPATIBLE",
    split: str = "test",
) -> dict[str, Any]:
    return {
        "status": status,
        "dataset": view.head.dataset,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "model": model,
        "split": split,
        "reason": reason,
        "test_accessed": False,
        "ood_accessed": False,
    }


def _static_native(paths: PublicAllPaths, view: ViewSpec):
    requirement = SupportRequirement(input_history_steps=1)
    fit = _development(paths.shared, view, [requirement])
    cap = int(_freeze(paths.project)["selection"]["fit_row_cap_default"])
    return _cap_after_support(fit, cap)


def _simple_model(
    paths: PublicAllPaths,
    view: ViewSpec,
    model: str,
    result: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    fit = _development(paths.shared, view, [SupportRequirement()])
    evaluation = _common_test(paths, view, split)
    if model == "MEAN":
        value = float(np.mean(fit["y_true"].to_numpy(dtype=np.float64)))
        parameter_count = 1
    elif model == "PERSISTENCE":
        value = 0.0
        parameter_count = 0
    else:
        raise KeyError(model)
    prediction = np.full(len(evaluation), value, dtype=np.float64)
    selection = {
        **result.get("selection", {}),
        "final_refit_partition": "train_plus_validation",
    }
    return prediction, fit, selection, parameter_count


def _static_model(
    paths: PublicAllPaths,
    view: ViewSpec,
    model: str,
    result: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    freeze = _freeze(paths.project)
    fit = _static_native(paths, view)
    test = _common_test(paths, view, split)
    columns = input_columns(paths.shared, view.head.task_id, view.proxy_policy)
    fit_accessor = BaseAccessor(paths.shared, view.head.dataset, "validation", columns)
    test_accessor = BaseAccessor(paths.shared, view.head.dataset, split, columns)
    x_fit = fit_accessor.snapshot(fit, columns)
    x_test = test_accessor.snapshot(test, columns)
    y_fit = fit["y_true"].to_numpy(dtype=np.float64)
    selection = result["selection"]
    if model == "RIDGE":
        prediction, parameter_count = _ridge_predictions(
            x_fit, y_fit, x_test, float(selection["selected_alpha"])
        )
    elif model == "PLS":
        prediction, parameter_count = _pls_predictions(
            x_fit, y_fit, x_test, int(selection["selected_components"])
        )
    elif model == "RBF_SVR":
        prediction, parameter_count = _svr_fit_predict(
            x_fit,
            y_fit,
            x_test,
            float(selection["selected_c"]["candidate"]),
            float(selection["selected_gamma"]["candidate"]),
            float(freeze["c2"]["rbf_svr"]["epsilon_standardized"]),
        )
    elif model == "XGBOOST":
        config = freeze["c2"]["xgboost"]
        pair = selection["selected_pair"]["candidate"]
        prediction, parameter_count = _xgb_fit_predict(
            x_fit,
            y_fit,
            x_test,
            config,
            int(pair[0]),
            float(pair[1]),
            int(selection["selected_estimators"]["candidate"]),
        )
    else:
        raise KeyError(model)
    return prediction, fit, selection, parameter_count


def _dpls_model(
    paths: PublicAllPaths,
    view: ViewSpec,
    result: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    freeze = _freeze(paths.project)
    selection = result["selection"]
    history = int(selection["selected_history"])
    components = int(selection["selected_components"])
    requirement = SupportRequirement(input_history_steps=history)
    fit = _cap_after_support(
        _development(paths.shared, view, [requirement]),
        int(freeze["selection"]["fit_row_cap_default"]),
    )
    test = _common_test(paths, view, split)
    columns = input_columns(paths.shared, view.head.task_id, view.proxy_policy)
    fit_accessor = BaseAccessor(paths.shared, view.head.dataset, "validation", columns)
    test_accessor = BaseAccessor(paths.shared, view.head.dataset, split, columns)
    x_fit = fit_accessor.input_lags(
        fit,
        columns,
        history,
        int(freeze["c2"]["dpls"]["maximum_lags_per_channel"]),
    )
    x_test = test_accessor.input_lags(
        test,
        columns,
        history,
        int(freeze["c2"]["dpls"]["maximum_lags_per_channel"]),
    )
    prediction, parameter_count = _pls_predictions(
        x_fit, fit["y_true"].to_numpy(dtype=np.float64), x_test, components
    )
    return prediction, fit, selection, parameter_count


def _hammerstein_model(
    paths: PublicAllPaths,
    view: ViewSpec,
    model: str,
    result: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    freeze = _freeze(paths.project)
    selection = result["selection"]
    profile = tuple(int(value) for value in selection["selected_profile"])
    requirement = SupportRequirement(input_history_steps=int(profile[1]))
    fit = _cap_after_support(
        _development(paths.shared, view, [requirement]),
        int(freeze["selection"]["fit_row_cap_default"]),
    )
    test = _common_test(paths, view, split)
    columns = input_columns(paths.shared, view.head.task_id, view.proxy_policy)
    fit_accessor = BaseAccessor(paths.shared, view.head.dataset, "validation", columns)
    test_accessor = BaseAccessor(paths.shared, view.head.dataset, split, columns)
    raw_fit = fit_accessor.input_regular_lags(fit, columns, *profile, 8)
    raw_test = test_accessor.input_regular_lags(test, columns, *profile, 8)
    nonlinearity = selection["selected_input_nonlinearity"]
    x_fit, x_test = _nonlinear_features(raw_fit, raw_test, nonlinearity)
    prediction, _ = _ridge_block_predict(
        x_fit,
        fit["y_true"].to_numpy(dtype=np.float64),
        x_test,
        0.001,
    )
    if selection["selected_output_map"] == "isotonic_train_only":
        from sklearn.isotonic import IsotonicRegression

        train_prediction, _ = _ridge_block_predict(
            x_fit, fit["y_true"].to_numpy(dtype=np.float64), x_fit, 0.001
        )
        prediction = IsotonicRegression(out_of_bounds="clip").fit(
            train_prediction, fit["y_true"].to_numpy(dtype=np.float64)
        ).predict(prediction)
    return prediction, fit, selection, int(x_fit.shape[1] + 1)


def _ar_model(
    paths: PublicAllPaths,
    view: ViewSpec,
    model: str,
    result: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    selection = result["selection"]
    profile = tuple(int(value) for value in selection["selected_profile"])
    requirement = SupportRequirement(
        target_delta_steps=profile[0], target_history_steps=profile[1]
    )
    freeze = _freeze(paths.project)
    fit = _cap_after_support(
        _development(paths.shared, view, [requirement]),
        int(freeze["selection"]["fit_row_cap_default"]),
    )
    test = _common_test(paths, view, split)
    accessor_fit = BaseAccessor(paths.shared, view.head.dataset, "validation", [view.head.target])
    accessor_test = BaseAccessor(paths.shared, view.head.dataset, split, [view.head.target])
    x_fit = accessor_fit.target_state(fit, view.head.target, *profile)
    x_test = accessor_test.target_state(test, view.head.target, *profile)
    prediction, _ = _ridge_block_predict(
        x_fit,
        fit["y_true"].to_numpy(dtype=np.float64),
        x_test,
        float(selection["selected_alpha"]),
    )
    return prediction, fit, selection, int(x_fit.shape[1] + 1)


def _arx_model(
    paths: PublicAllPaths,
    view: ViewSpec,
    model: str,
    result: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    selection = result["selection"]
    profile = tuple(int(value) for value in selection["selected_profile"])
    requirement = SupportRequirement(
        input_history_steps=profile[1],
        target_delta_steps=profile[0],
        target_history_steps=profile[1],
    )
    freeze = _freeze(paths.project)
    fit = _cap_after_support(
        _development(paths.shared, view, [requirement]),
        int(freeze["selection"]["fit_row_cap_default"]),
    )
    test = _common_test(paths, view, split)
    columns = input_columns(paths.shared, view.head.task_id, view.proxy_policy)
    all_columns = [view.head.target, *columns]
    fit_accessor = BaseAccessor(paths.shared, view.head.dataset, "validation", all_columns)
    test_accessor = BaseAccessor(paths.shared, view.head.dataset, split, all_columns)
    x_fit, ar_width = _arx_features(
        fit_accessor,
        fit,
        view,
        columns,
        profile,
        int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"]),
    )
    x_test, _ = _arx_features(
        test_accessor,
        test,
        view,
        columns,
        profile,
        int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"]),
    )
    alpha = float(selection["ar_alpha"])
    selected_ratio = selection["selected_x_penalty_ratio"]
    if selected_ratio == "EXACT_X_ZERO":
        x_fit = x_fit[:, :ar_width]
        x_test = x_test[:, :ar_width]
        penalties: float | np.ndarray = alpha
    else:
        ratio = float(selected_ratio)
        penalties = np.concatenate(
            [
                np.full(ar_width, alpha),
                np.full(x_fit.shape[1] - ar_width, alpha * ratio),
            ]
        )
    prediction, _ = _ridge_block_predict(
        x_fit, fit["y_true"].to_numpy(dtype=np.float64), x_test, penalties
    )
    return prediction, fit, selection, int(x_fit.shape[1] + 1)


def _narx_model(
    paths: PublicAllPaths,
    view: ViewSpec,
    result: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    selection = result["selection"]
    profile = tuple(int(value) for value in selection["selected_profile"])
    requirement = SupportRequirement(
        input_history_steps=profile[1],
        target_delta_steps=profile[0],
        target_history_steps=profile[1],
    )
    freeze = _freeze(paths.project)
    fit = _cap_after_support(
        _development(paths.shared, view, [requirement]),
        int(freeze["selection"]["fit_row_cap_default"]),
    )
    test = _common_test(paths, view, split)
    columns = input_columns(paths.shared, view.head.task_id, view.proxy_policy)
    all_columns = [view.head.target, *columns]
    fit_accessor = BaseAccessor(paths.shared, view.head.dataset, "validation", all_columns)
    test_accessor = BaseAccessor(paths.shared, view.head.dataset, split, all_columns)
    raw_fit, _ = _arx_features(
        fit_accessor,
        fit,
        view,
        columns,
        profile,
        int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"]),
    )
    raw_test, _ = _arx_features(
        test_accessor,
        test,
        view,
        columns,
        profile,
        int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"]),
    )
    x_fit, _, selected_features = _narx_expand(
        raw_fit,
        fit["y_true"].to_numpy(dtype=np.float64),
        raw_fit[:1],
        int(freeze["c3"]["linear_narx"]["maximum_linear_state_features_before_expansion"]),
    )
    _, x_test, observed_features = _narx_expand(
        raw_fit,
        fit["y_true"].to_numpy(dtype=np.float64),
        raw_test,
        int(freeze["c3"]["linear_narx"]["maximum_linear_state_features_before_expansion"]),
    )
    if not np.array_equal(selected_features, observed_features):
        raise RuntimeError("frozen NARX feature selection drifted")
    prediction, _ = _ridge_block_predict(
        x_fit,
        fit["y_true"].to_numpy(dtype=np.float64),
        x_test,
        float(selection["selected_alpha"]),
    )
    return prediction, fit, selection, int(x_fit.shape[1] + 1)


def materialize_baseline_view(
    paths: PublicAllPaths, view: ViewSpec, *, split: str = "test"
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    candidates: list[tuple[str, str, str]] = [
        ("C2", "MEAN", "MEAN"),
        ("C2", "PERSISTENCE", "PERSISTENCE"),
        ("C2", "SEASONAL_PERSISTENCE", "SEASONAL_PERSISTENCE"),
    ]
    if view.information_set == "input_only":
        candidates.extend(
            [
                ("C2", "RIDGE", "RIDGE"),
                ("C2", "PLS", "PLS"),
                ("C2", "RBF_SVR", "RBF_SVR"),
                ("C2", "XGBOOST", "XGBOOST"),
                ("C2", "DPLS", "DPLS"),
                ("C3", "PARALLEL_HAMMERSTEIN", "PARALLEL_HAMMERSTEIN"),
                ("C3", "HAMMERSTEIN_WIENER", "HAMMERSTEIN_WIENER"),
            ]
        )
    else:
        candidates.extend(
            [
                ("C3", "AR", "AR"),
                ("C3", "ARX", "ARX"),
                ("C3", "LINEAR_NARX", "LINEAR_NARX"),
                ("C3", "N4SID", "N4SID"),
            ]
        )
    for family, model, output_model in candidates:
        result = _result(paths, family, model, view)
        if result is None:
            audits.append(
                _not_run(
                    view,
                    output_model,
                    "DEVELOPMENT_RESULT_ABSENT",
                    status="FAILED_RETAINED",
                    split=split,
                )
            )
            continue
        development_status = str(result.get("status"))
        if development_status in {
            "NOT_RUN_IMPLEMENTATION_ABSENT",
            "NOT_RUN_PROTOCOL_INCOMPATIBLE",
        }:
            audits.append(
                _not_run(
                    view,
                    output_model,
                    str(result.get("reason", f"DEVELOPMENT_STATUS_{development_status}")),
                    status=development_status,
                    split=split,
                )
            )
            continue
        if development_status != "PASS":
            audits.append(
                _not_run(
                    view,
                    output_model,
                    f"DEVELOPMENT_STATUS_{development_status}",
                    status="FAILED_RETAINED",
                    split=split,
                )
            )
            continue
        started = time.time()
        if model in {"MEAN", "PERSISTENCE"}:
            prediction, native_fit, selection, parameter_count = _simple_model(
                paths, view, model, result, split=split
            )
        elif model in {"RIDGE", "PLS", "RBF_SVR", "XGBOOST"}:
            prediction, native_fit, selection, parameter_count = _static_model(
                paths, view, model, result, split=split
            )
        elif model == "DPLS":
            prediction, native_fit, selection, parameter_count = _dpls_model(
                paths, view, result, split=split
            )
        elif model in {"PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"}:
            prediction, native_fit, selection, parameter_count = _hammerstein_model(
                paths, view, model, result, split=split
            )
        elif model == "AR":
            prediction, native_fit, selection, parameter_count = _ar_model(
                paths, view, model, result, split=split
            )
        elif model == "ARX":
            prediction, native_fit, selection, parameter_count = _arx_model(
                paths, view, model, result, split=split
            )
        elif model == "LINEAR_NARX":
            prediction, native_fit, selection, parameter_count = _narx_model(
                paths, view, result, split=split
            )
        else:
            audits.append(
                _not_run(
                    view,
                    output_model,
                    "N4SID_FINAL_REFIT_NOT_IMPLEMENTED",
                    split=split,
                )
            )
            continue
        audits.append(
            _write(
                paths,
                view,
                output_model,
                _common_test(paths, view, split),
                prediction,
                native_fit,
                started,
                selection=selection,
                parameter_count=parameter_count,
                split=split,
            )
        )
    return audits
