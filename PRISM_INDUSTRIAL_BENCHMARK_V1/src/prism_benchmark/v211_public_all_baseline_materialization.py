from __future__ import annotations

import json
import time
from collections.abc import Iterator
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
    _nonlinear_features,
    _ridge_block_predict,
)
from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    input_columns,
    sha256_file,
)
from .cpu_selection import Standardizer, regression_metrics
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
from .v2_runtime import release_process_memory


_MATERIALIZATION_PREDICTION_BLOCK_ROWS = 100_000


def _fit_ridge_block_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    penalties: np.ndarray | float,
) -> tuple[Standardizer, np.ndarray, float, dict[str, Any]]:
    """Fit the frozen ridge model without materializing an evaluation matrix."""
    scaler = Standardizer().fit(x_train)
    train = scaler.transform(x_train)
    y_mean = float(np.mean(y_train, dtype=np.float64))
    centered_y = np.asarray(y_train, dtype=np.float64) - y_mean
    penalty = np.broadcast_to(np.asarray(penalties, dtype=np.float64), train.shape[1])
    gram = train.T @ train
    rhs = train.T @ centered_y
    system = gram + np.diag(penalty)
    try:
        coefficient = np.linalg.solve(system, rhs)
        solver = "solve"
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(system, rhs, rcond=1e-12)[0]
        solver = "svd_rescue"
    residual = system @ coefficient - rhs
    relative_kkt = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0))
    condition = float(np.linalg.cond(system))
    certificate = {
        "solver": solver,
        "relative_kkt": relative_kkt,
        "condition_number": condition,
    }
    return scaler, np.asarray(coefficient, dtype=np.float64), y_mean, certificate


def _ridge_blockwise_predict(
    blocks: Iterator[np.ndarray],
    rows: int,
    scaler: Standardizer,
    coefficient: np.ndarray,
    y_mean: float,
) -> np.ndarray:
    prediction = np.empty(rows, dtype=np.float64)
    cursor = 0
    for block in blocks:
        block_rows = len(block)
        if cursor + block_rows > rows:
            raise RuntimeError(
                "blockwise ridge prediction exceeded frozen support rows"
            )
        evaluation = scaler.transform(block)
        prediction[cursor : cursor + block_rows] = evaluation @ coefficient + y_mean
        cursor += block_rows
        del evaluation, block
        release_process_memory()
    if cursor != rows:
        raise RuntimeError(
            f"blockwise ridge prediction row mismatch: expected={rows} observed={cursor}"
        )
    return prediction


def _iter_arx_feature_blocks(
    accessor: BaseAccessor,
    samples,
    view: ViewSpec,
    columns: list[str],
    profile: tuple[int, int],
    maximum_input_lags: int,
    *,
    feature_width: int | None = None,
    block_rows: int | None = None,
) -> Iterator[np.ndarray]:
    rows_per_block = (
        _MATERIALIZATION_PREDICTION_BLOCK_ROWS
        if block_rows is None
        else int(block_rows)
    )
    if rows_per_block <= 0:
        raise ValueError("materialization block_rows must be positive")
    for start in range(0, len(samples), rows_per_block):
        subset = samples.iloc[start : start + rows_per_block]
        features, _ = _arx_features(
            accessor,
            subset,
            view,
            columns,
            profile,
            maximum_input_lags,
        )
        if feature_width is not None:
            features = features[:, :feature_width]
        yield features


def _fit_narx_expansion(
    raw_train: np.ndarray,
    y_train: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, Standardizer, np.ndarray]:
    """Freeze NARX feature order on fit data for blockwise evaluation."""
    scaler = Standardizer().fit(raw_train)
    standardized = scaler.transform(raw_train)
    centered_y = np.asarray(y_train, dtype=np.float64) - np.mean(
        y_train, dtype=np.float64
    )
    denominator = np.sqrt(
        np.sum(np.square(standardized), axis=0) * np.sum(np.square(centered_y))
    )
    correlations = np.abs(
        (standardized.T @ centered_y) / np.where(denominator > 0, denominator, np.inf)
    )
    order = np.lexsort((np.arange(len(correlations)), -correlations))[:maximum]
    selected_train = standardized[:, order]
    expanded_train = np.concatenate([selected_train, np.square(selected_train)], axis=1)
    return expanded_train, scaler, order


def _expand_narx_block(
    raw_block: np.ndarray,
    scaler: Standardizer,
    order: np.ndarray,
) -> np.ndarray:
    selected = scaler.transform(raw_block)[:, order]
    return np.concatenate([selected, np.square(selected)], axis=1)


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


def _static_native(paths: PublicAllPaths, view: ViewSpec, model: str):
    requirement = SupportRequirement(input_history_steps=1)
    fit = _development(paths.shared, view, [requirement])
    cap_key = {
        "RBF_SVR": "fit_row_cap_svr",
        "XGBOOST": "fit_row_cap_xgboost",
    }.get(model, "fit_row_cap_default")
    cap = int(_freeze(paths.project)["selection"][cap_key])
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
    fit = _static_native(paths, view, model)
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
    alpha = float(selection["ar_alpha"])
    selected_ratio = selection["selected_x_penalty_ratio"]
    if selected_ratio == "EXACT_X_ZERO":
        x_fit = x_fit[:, :ar_width]
        penalties: float | np.ndarray = alpha
        feature_width: int | None = ar_width
    else:
        ratio = float(selected_ratio)
        penalties = np.concatenate(
            [
                np.full(ar_width, alpha),
                np.full(x_fit.shape[1] - ar_width, alpha * ratio),
            ]
        )
        feature_width = None
    parameter_count = int(x_fit.shape[1] + 1)
    scaler, coefficient, y_mean, _ = _fit_ridge_block_model(
        x_fit,
        fit["y_true"].to_numpy(dtype=np.float64),
        penalties,
    )
    del x_fit
    release_process_memory()
    prediction = _ridge_blockwise_predict(
        _iter_arx_feature_blocks(
            test_accessor,
            test,
            view,
            columns,
            profile,
            int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"]),
            feature_width=feature_width,
        ),
        len(test),
        scaler,
        coefficient,
        y_mean,
    )
    return prediction, fit, selection, parameter_count


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
    y_fit = fit["y_true"].to_numpy(dtype=np.float64)
    x_fit, raw_scaler, selected_features = _fit_narx_expansion(
        raw_fit,
        y_fit,
        int(
            freeze["c3"]["linear_narx"][
                "maximum_linear_state_features_before_expansion"
            ]
        ),
    )
    del raw_fit
    release_process_memory()
    parameter_count = int(x_fit.shape[1] + 1)
    scaler, coefficient, y_mean, _ = _fit_ridge_block_model(
        x_fit,
        y_fit,
        float(selection["selected_alpha"]),
    )
    del x_fit
    release_process_memory()
    raw_blocks = _iter_arx_feature_blocks(
        test_accessor,
        test,
        view,
        columns,
        profile,
        int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"]),
    )
    expanded_blocks = (
        _expand_narx_block(raw_block, raw_scaler, selected_features)
        for raw_block in raw_blocks
    )
    prediction = _ridge_blockwise_predict(
        expanded_blocks,
        len(test),
        scaler,
        coefficient,
        y_mean,
    )
    return prediction, fit, selection, parameter_count


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
