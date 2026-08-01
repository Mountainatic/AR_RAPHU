from __future__ import annotations

import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    deterministic_subsample,
    inner_folds,
    input_columns,
    load_samples,
    main_views,
    sha256_file,
)
from .cpu_selection import Standardizer, mse, regression_metrics, select_one_se
from .stage0 import write_json


def _load_freeze(project: Path) -> dict[str, Any]:
    return json.loads((project / "configs/cpu_model_freeze_v1.json").read_text(encoding="utf-8"))


def _capped(samples: pd.DataFrame, indices: np.ndarray, cap: int) -> np.ndarray:
    subset = samples.iloc[indices]
    return indices[deterministic_subsample(subset, cap)]


def _folds(samples: pd.DataFrame, view: ViewSpec) -> list[tuple[np.ndarray, np.ndarray]]:
    buffer_steps = int(math.ceil(600.0 / view.head.cadence_seconds))
    return inner_folds(samples, count=4, extra_buffer_steps=buffer_steps)


def _prediction_frame(
    samples: pd.DataFrame,
    view: ViewSpec,
    model: str,
    prediction: np.ndarray,
    parameter_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": samples["view_sample_id"].astype(str),
            "base_origin_id": samples["base_origin_id"].astype(str),
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "split": "validation",
            "model": model,
            "y_true": samples["y_true"].to_numpy(dtype=np.float64),
            "y_pred": np.asarray(prediction, dtype=np.float64),
            "information_set": view.information_set,
            "profile_id": samples["entity_id"].astype(str),
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "seed": -1,
            "dtype": "float64",
            "parameter_count": parameter_count,
        }
    )


def _ridge_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, int]:
    from sklearn.linear_model import Ridge

    scaler = Standardizer().fit(x_train)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-10)
    model.fit(scaler.transform(x_train), y_train)
    return np.asarray(model.predict(scaler.transform(x_eval)), dtype=np.float64), x_train.shape[1] + 1


def _pls_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    components: int,
) -> tuple[np.ndarray, int]:
    from sklearn.cross_decomposition import PLSRegression

    model = PLSRegression(n_components=components, scale=True, max_iter=1000, tol=1e-08)
    model.fit(x_train, y_train)
    prediction = model.predict(x_eval).reshape(-1)
    parameter_count = components * (x_train.shape[1] + 2) + 1
    return np.asarray(prediction, dtype=np.float64), parameter_count


def _static_matrices(
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    columns: list[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_subset = samples.iloc[train_indices]
    validation_subset = samples.iloc[validation_indices]
    return (
        accessor.snapshot(train_subset, columns),
        train_subset["y_true"].to_numpy(dtype=np.float64),
        accessor.snapshot(validation_subset, columns),
        validation_subset["y_true"].to_numpy(dtype=np.float64),
    )


def _run_ridge(shared: Path, project: Path, view: ViewSpec) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _load_freeze(project)
    config = freeze["c2"]["ridge"]
    train = load_samples(shared, view, "train")
    validation = load_samples(shared, view, "validation")
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    accessor = BaseAccessor(shared, view.head.dataset, "train", columns)
    losses = {float(alpha): [] for alpha in config["alphas"]}
    for train_index, validation_index in _folds(train, view):
        train_index = _capped(train, train_index, freeze["selection"]["fit_row_cap_default"])
        validation_index = _capped(train, validation_index, freeze["selection"]["selection_validation_row_cap_default"])
        x_train, y_train, x_validation, y_validation = _static_matrices(accessor, train, columns, train_index, validation_index)
        for alpha in losses:
            prediction, _ = _ridge_predictions(x_train, y_train, x_validation, alpha)
            losses[alpha].append(mse(y_validation, prediction))
    selected = select_one_se(losses, lambda alpha: (-alpha,))
    final_index = deterministic_subsample(train, freeze["selection"]["fit_row_cap_default"])
    x_train = accessor.snapshot(train.iloc[final_index], columns)
    y_train = train.iloc[final_index]["y_true"].to_numpy(dtype=np.float64)
    validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", columns)
    x_validation = validation_accessor.snapshot(validation, columns)
    prediction, parameter_count = _ridge_predictions(x_train, y_train, x_validation, float(selected.candidate))
    return prediction, {"selected_alpha": selected.candidate, "fold_losses": losses, "one_se": selected.__dict__}, parameter_count


def _run_pls(shared: Path, project: Path, view: ViewSpec, dynamic: bool = False) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _load_freeze(project)
    section = "dpls" if dynamic else "pls"
    config = freeze["c2"][section]
    train = load_samples(shared, view, "train")
    validation = load_samples(shared, view, "validation")
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    accessor = BaseAccessor(shared, view.head.dataset, "train", columns)
    fit_cap = freeze["selection"]["fit_row_cap_default"]
    validation_cap = freeze["selection"]["selection_validation_row_cap_default"]
    if not dynamic:
        candidates = [int(value) for value in config["components"] if value <= len(columns)]
        losses = {value: [] for value in candidates}
        for train_index, validation_index in _folds(train, view):
            train_index = _capped(train, train_index, fit_cap)
            validation_index = _capped(train, validation_index, validation_cap)
            x_train, y_train, x_validation, y_validation = _static_matrices(accessor, train, columns, train_index, validation_index)
            for components in candidates:
                prediction, _ = _pls_predictions(x_train, y_train, x_validation, components)
                losses[components].append(mse(y_validation, prediction))
        selected = select_one_se(losses, lambda value: (value,))
        final_index = deterministic_subsample(train, fit_cap)
        x_train = accessor.snapshot(train.iloc[final_index], columns)
        y_train = train.iloc[final_index]["y_true"].to_numpy(dtype=np.float64)
        validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", columns)
        x_validation = validation_accessor.snapshot(validation, columns)
        prediction, parameter_count = _pls_predictions(x_train, y_train, x_validation, int(selected.candidate))
        return prediction, {"selected_components": selected.candidate, "fold_losses": losses, "one_se": selected.__dict__}, parameter_count

    histories = [max(1, multiplier * view.head.h_steps) for multiplier in config["lag_coverage_for_positive_h"]]
    candidates = [(history, int(component)) for history in histories for component in config["components"]]
    losses = {candidate: [] for candidate in candidates}
    for train_index, validation_index in _folds(train, view):
        train_index = _capped(train, train_index, fit_cap)
        validation_index = _capped(train, validation_index, validation_cap)
        train_subset = train.iloc[train_index]
        validation_subset = train.iloc[validation_index]
        for history in histories:
            x_train = accessor.input_lags(train_subset, columns, history, config["maximum_lags_per_channel"])
            x_validation = accessor.input_lags(validation_subset, columns, history, config["maximum_lags_per_channel"])
            y_train = train_subset["y_true"].to_numpy(dtype=np.float64)
            y_validation = validation_subset["y_true"].to_numpy(dtype=np.float64)
            for _, components in [candidate for candidate in candidates if candidate[0] == history]:
                if components > min(x_train.shape[1], len(x_train) - 1):
                    losses[(history, components)].append(float("inf"))
                    continue
                prediction, _ = _pls_predictions(x_train, y_train, x_validation, components)
                losses[(history, components)].append(mse(y_validation, prediction))
    selected = select_one_se(losses, lambda value: (value[0], value[1]))
    history, components = selected.candidate
    final_index = deterministic_subsample(train, fit_cap)
    train_subset = train.iloc[final_index]
    x_train = accessor.input_lags(train_subset, columns, history, config["maximum_lags_per_channel"])
    y_train = train_subset["y_true"].to_numpy(dtype=np.float64)
    validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", columns)
    x_validation = validation_accessor.input_lags(validation, columns, history, config["maximum_lags_per_channel"])
    prediction, parameter_count = _pls_predictions(x_train, y_train, x_validation, components)
    return prediction, {"selected_history": history, "selected_components": components, "fold_losses": {str(k): v for k, v in losses.items()}, "one_se": selected.__dict__}, parameter_count


def _svr_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    c_value: float,
    gamma: float,
    epsilon: float,
) -> tuple[np.ndarray, int]:
    from sklearn.svm import SVR

    scaler = Standardizer().fit(x_train)
    y_mean = float(np.mean(y_train, dtype=np.float64))
    y_scale = float(np.std(y_train, ddof=0)) or 1.0
    model = SVR(C=c_value, gamma=gamma, epsilon=epsilon, kernel="rbf", cache_size=4096)
    model.fit(scaler.transform(x_train), (y_train - y_mean) / y_scale)
    prediction = model.predict(scaler.transform(x_eval)) * y_scale + y_mean
    return np.asarray(prediction, dtype=np.float64), int(len(model.support_) * (x_train.shape[1] + 1))


def _run_svr(shared: Path, project: Path, view: ViewSpec) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _load_freeze(project)
    config = freeze["c2"]["rbf_svr"]
    train = load_samples(shared, view, "train")
    validation = load_samples(shared, view, "validation")
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    accessor = BaseAccessor(shared, view.head.dataset, "train", columns)
    fold_data = []
    for train_index, validation_index in _folds(train, view):
        train_index = _capped(train, train_index, freeze["selection"]["fit_row_cap_svr"])
        validation_index = _capped(train, validation_index, freeze["selection"]["selection_validation_row_cap_svr"])
        fold_data.append(_static_matrices(accessor, train, columns, train_index, validation_index))
    pilot_gamma = 0.0078125
    c_losses = {float(value): [] for value in config["c"]}
    for x_train, y_train, x_validation, y_validation in fold_data:
        for c_value in c_losses:
            prediction, _ = _svr_fit_predict(x_train, y_train, x_validation, c_value, pilot_gamma, config["epsilon_standardized"])
            c_losses[c_value].append(mse(y_validation, prediction))
    selected_c = select_one_se(c_losses, lambda value: (value,))
    gamma_losses = {float(value): [] for value in config["gamma"]}
    for x_train, y_train, x_validation, y_validation in fold_data:
        for gamma in gamma_losses:
            prediction, _ = _svr_fit_predict(x_train, y_train, x_validation, float(selected_c.candidate), gamma, config["epsilon_standardized"])
            gamma_losses[gamma].append(mse(y_validation, prediction))
    selected_gamma = select_one_se(gamma_losses, lambda value: (value,))
    final_index = deterministic_subsample(train, freeze["selection"]["fit_row_cap_svr"])
    x_train = accessor.snapshot(train.iloc[final_index], columns)
    y_train = train.iloc[final_index]["y_true"].to_numpy(dtype=np.float64)
    validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", columns)
    x_validation = validation_accessor.snapshot(validation, columns)
    prediction, parameter_count = _svr_fit_predict(
        x_train,
        y_train,
        x_validation,
        float(selected_c.candidate),
        float(selected_gamma.candidate),
        config["epsilon_standardized"],
    )
    return prediction, {
        "selected_c": selected_c.__dict__,
        "selected_gamma": selected_gamma.__dict__,
        "c_fold_losses": c_losses,
        "gamma_fold_losses": gamma_losses,
    }, parameter_count


def _xgb_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    config: dict[str, Any],
    depth: int,
    learning_rate: float,
    estimators: int,
) -> tuple[np.ndarray, int]:
    from xgboost import XGBRegressor

    model = XGBRegressor(
        max_depth=depth,
        learning_rate=learning_rate,
        n_estimators=estimators,
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        min_child_weight=config["min_child_weight"],
        reg_lambda=config["reg_lambda"],
        tree_method=config["tree_method"],
        n_jobs=config["nthread_per_fit"],
        objective="reg:squarederror",
        random_state=20260802,
    )
    model.fit(x_train, y_train, verbose=False)
    prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
    return prediction, int(sum(tree.count("leaf") for tree in model.get_booster().get_dump()))


def _run_xgboost(shared: Path, project: Path, view: ViewSpec) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _load_freeze(project)
    config = freeze["c2"]["xgboost"]
    train = load_samples(shared, view, "train")
    validation = load_samples(shared, view, "validation")
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    accessor = BaseAccessor(shared, view.head.dataset, "train", columns)
    fold_data = []
    for train_index, validation_index in _folds(train, view):
        train_index = _capped(train, train_index, freeze["selection"]["fit_row_cap_xgboost"])
        validation_index = _capped(train, validation_index, freeze["selection"]["selection_validation_row_cap_default"])
        fold_data.append(_static_matrices(accessor, train, columns, train_index, validation_index))
    pair_losses = {(int(depth), float(rate)): [] for depth in config["max_depth"] for rate in config["learning_rate"]}
    for x_train, y_train, x_validation, y_validation in fold_data:
        for depth, rate in pair_losses:
            prediction, _ = _xgb_fit_predict(x_train, y_train, x_validation, config, depth, rate, 400)
            pair_losses[(depth, rate)].append(mse(y_validation, prediction))
    selected_pair = select_one_se(pair_losses, lambda value: (value[0], value[1]))
    estimator_losses = {int(value): [] for value in config["n_estimators"]}
    for x_train, y_train, x_validation, y_validation in fold_data:
        for estimators in estimator_losses:
            prediction, _ = _xgb_fit_predict(
                x_train,
                y_train,
                x_validation,
                config,
                selected_pair.candidate[0],
                selected_pair.candidate[1],
                estimators,
            )
            estimator_losses[estimators].append(mse(y_validation, prediction))
    selected_estimators = select_one_se(estimator_losses, lambda value: (value,))
    final_index = deterministic_subsample(train, freeze["selection"]["fit_row_cap_xgboost"])
    x_train = accessor.snapshot(train.iloc[final_index], columns)
    y_train = train.iloc[final_index]["y_true"].to_numpy(dtype=np.float64)
    validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", columns)
    x_validation = validation_accessor.snapshot(validation, columns)
    prediction, parameter_count = _xgb_fit_predict(
        x_train,
        y_train,
        x_validation,
        config,
        selected_pair.candidate[0],
        selected_pair.candidate[1],
        int(selected_estimators.candidate),
    )
    return prediction, {
        "selected_pair": selected_pair.__dict__,
        "selected_estimators": selected_estimators.__dict__,
        "pair_fold_losses": {str(k): v for k, v in pair_losses.items()},
        "estimator_fold_losses": estimator_losses,
    }, parameter_count


def _trend_prediction(accessor: BaseAccessor, samples: pd.DataFrame, target: str, history: int, head: Any) -> np.ndarray:
    offsets = np.arange(history, 0, -1, dtype=np.int64)
    indices = samples["origin"].to_numpy(dtype=np.int64)[:, None] - offsets[None, :]
    levels = accessor.gather(samples, [target], indices)
    time_axis = -offsets.astype(np.float64)
    centered_time = time_axis - np.mean(time_axis)
    denominator = float(np.sum(np.square(centered_time)))
    slopes = ((levels - levels.mean(axis=1, keepdims=True)) @ centered_time) / denominator
    current_time = np.arange(-head.w0_steps, 0, dtype=np.float64).mean()
    future_time = np.arange(head.h_steps, head.h_steps + head.w_steps, dtype=np.float64).mean()
    return slopes * (future_time - current_time)


def _run_trend(shared: Path, project: Path, view: ViewSpec) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _load_freeze(project)
    histories = freeze["c2"]["local_linear_trend"]["history_steps"]
    train = load_samples(shared, view, "train")
    validation = load_samples(shared, view, "validation")
    accessor = BaseAccessor(shared, view.head.dataset, "train", [view.head.target])
    losses = {int(history): [] for history in histories}
    for _, validation_index in _folds(train, view):
        validation_index = _capped(train, validation_index, freeze["selection"]["selection_validation_row_cap_default"])
        subset = train.iloc[validation_index]
        for history in losses:
            prediction = _trend_prediction(accessor, subset, view.head.target, history, view.head)
            losses[history].append(mse(subset["y_true"].to_numpy(dtype=np.float64), prediction))
    selected = select_one_se(losses, lambda value: (value,))
    validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", [view.head.target])
    prediction = _trend_prediction(validation_accessor, validation, view.head.target, int(selected.candidate), view.head)
    return prediction, {"selected_history": selected.candidate, "fold_losses": losses, "one_se": selected.__dict__}, 2


RUNNERS: dict[str, Callable[[Path, Path, ViewSpec], tuple[np.ndarray, dict[str, Any], int]]] = {
    "RIDGE": _run_ridge,
    "PLS": _run_pls,
    "DPLS": lambda shared, project, view: _run_pls(shared, project, view, dynamic=True),
    "RBF_SVR": _run_svr,
    "XGBOOST": _run_xgboost,
    "LOCAL_LINEAR_TREND": _run_trend,
}


def run_job(shared: Path, project: Path, output: Path, view: ViewSpec, model: str) -> dict[str, Any]:
    started = time.time()
    destination_root = output / "PREDICTIONS" / model / view.relative_root
    destination_root.mkdir(parents=True, exist_ok=True)
    try:
        prediction, selection, parameter_count = RUNNERS[model](shared, project, view)
        validation = load_samples(shared, view, "validation")
        frame = _prediction_frame(validation, view, model, prediction, parameter_count)
        prediction_path = destination_root / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        result = {
            "status": "PASS",
            "stage": "C2_CLASSICAL_DEVELOPMENT",
            "model": model,
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "information_set": view.information_set,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "split": "validation",
            "rows": len(frame),
            "parameter_count": parameter_count,
            "dtype": "float64",
            "test_accessed": False,
            "selection": selection,
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "elapsed_seconds": time.time() - started,
            **regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()),
        }
    except Exception as error:
        result = {
            "status": "FAILED_RETAINED",
            "stage": "C2_CLASSICAL_DEVELOPMENT",
            "model": model,
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "information_set": view.information_set,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "test_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    write_json(destination_root / "RESULT.json", result)
    return result


def run_c2(shared: Path, project: Path, output: Path, n_jobs: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    input_views = main_views(shared, "input_only")
    dynamic_views = main_views(shared, "dynamic")
    jobs = [(view, model) for view in input_views for model in ("RIDGE", "PLS", "DPLS", "RBF_SVR", "XGBOOST")]
    jobs.extend((view, "LOCAL_LINEAR_TREND") for view in dynamic_views)
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(run_job, shared, project, output, view, model): (view, model)
            for view, model in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
    flat = []
    for result in results:
        row = {key: value for key, value in result.items() if key not in {"selection", "traceback"}}
        flat.append(row)
    pd.DataFrame(flat).sort_values(["task", "model"]).to_csv(output / "CLASSICAL_SOFT_SENSOR_DEVELOPMENT.csv", index=False)
    manifest = {
        "status": "PASS" if all(result["status"] == "PASS" for result in results) else "PASS_WITH_RETAINED_FAILURES",
        "jobs": len(results),
        "passed": sum(result["status"] == "PASS" for result in results),
        "failed_retained": sum(result["status"] != "PASS" for result in results),
        "test_accessed": False,
        "freeze_sha256": sha256_file(project / "configs/cpu_model_freeze_v1.json"),
    }
    write_json(output / "C2_CLASSICAL_MANIFEST.json", manifest)
    return manifest
