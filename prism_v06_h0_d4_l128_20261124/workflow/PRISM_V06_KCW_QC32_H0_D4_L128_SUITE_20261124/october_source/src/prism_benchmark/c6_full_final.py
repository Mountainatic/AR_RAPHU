from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .c2_models import (
    _pls_predictions,
    _ridge_predictions,
    _svr_fit_predict,
    _trend_prediction,
    _xgb_fit_predict,
)
from .c3_models import (
    _arx_features,
    _fit_n4sid,
    _n4sid_predict,
    _nonlinear_features,
    _ridge_block_predict,
)
from .c6_final import (
    _c4_result,
    _c5_result,
    _entity_groups,
    _execute_bootstrap_tasks,
    _fit_ar_final,
    _fit_k_final,
    _frame,
    _holm,
    _k_joint_final,
    _markdown_table,
    _paired_frames,
    _physics_first_final,
)
from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    deterministic_subsample,
    input_columns,
    load_samples,
    main_views,
    sha256_file,
)
from .cpu_selection import Standardizer, regression_metrics
from .stage0 import write_json


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _main_freeze(project: Path) -> dict[str, Any]:
    return _load_json(project / "configs/cpu_model_freeze_v1.json")


def _repair_freeze(project: Path) -> dict[str, Any]:
    return _load_json(project / "configs/c6_full_final_v2.json")


def _development_samples(shared: Path, view: ViewSpec) -> pd.DataFrame:
    return pd.concat(
        [load_samples(shared, view, "train"), load_samples(shared, view, "validation")],
        ignore_index=True,
    )


def _result_path(root: Path, model: str, view: ViewSpec) -> Path:
    return root / "PREDICTIONS" / model / view.relative_root / "RESULT.json"


def _contract(root: Path, model: str, view: ViewSpec) -> dict[str, Any]:
    path = _result_path(root, model, view)
    result = _load_json(path)
    if result.get("status") != "PASS":
        raise RuntimeError(
            f"development contract retained as unavailable: {model} {view.head.head_id} "
            f"status={result.get('status')} reason={result.get('error', result.get('reason', 'unspecified'))}"
        )
    return result


def _selected(value: Any) -> Any:
    return value.get("candidate") if isinstance(value, dict) and "candidate" in value else value


def _fit_subset(samples: pd.DataFrame, cap: int) -> pd.DataFrame:
    return samples.iloc[deterministic_subsample(samples, cap)]


def _chunked_prediction(
    samples: pd.DataFrame,
    feature_function: Any,
    prediction_function: Any,
    chunk_rows: int = 100_000,
    workers: int | None = None,
) -> np.ndarray:
    prediction = np.empty(len(samples), dtype=np.float64)
    starts = list(range(0, len(samples), chunk_rows))
    workers = (12 if len(samples) > 1_000_000 else 1) if workers is None else max(1, workers)

    def evaluate(start: int) -> tuple[int, np.ndarray]:
        stop = min(len(samples), start + chunk_rows)
        values = np.asarray(
            prediction_function(feature_function(samples.iloc[start:stop])), dtype=np.float64
        ).reshape(-1)
        return start, values

    if workers == 1:
        completed = map(evaluate, starts)
    else:
        executor = ThreadPoolExecutor(max_workers=min(workers, len(starts)))
        futures = [executor.submit(evaluate, start) for start in starts]
        completed = (future.result() for future in as_completed(futures))
    try:
        for start, values in completed:
            prediction[start : start + len(values)] = values
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return prediction


def _linear_fit(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    penalties: np.ndarray | float,
) -> tuple[Standardizer, np.ndarray, float, dict[str, float]]:
    scaler = Standardizer().fit(x_fit)
    standardized = scaler.transform(x_fit)
    y_mean = float(np.mean(y_fit, dtype=np.float64))
    centered_y = np.asarray(y_fit, dtype=np.float64) - y_mean
    penalty = np.broadcast_to(np.asarray(penalties, dtype=np.float64), standardized.shape[1])
    gram = standardized.T @ standardized
    rhs = standardized.T @ centered_y
    system = gram + np.diag(penalty)
    try:
        coefficient = np.linalg.solve(system, rhs)
        solver = "solve"
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(system, rhs, rcond=1e-12)[0]
        solver = "svd_rescue"
    residual = system @ coefficient - rhs
    certificate = {
        "solver": solver,
        "relative_kkt": float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0)),
        "condition_number": float(np.linalg.cond(system)),
    }
    return scaler, coefficient, y_mean, certificate


def _linear_predict(
    scaler: Standardizer,
    coefficient: np.ndarray,
    y_mean: float,
    x_evaluation: np.ndarray,
) -> np.ndarray:
    return np.asarray(scaler.transform(x_evaluation) @ coefficient + y_mean, dtype=np.float64)


def _fit_c2_model(
    shared: Path,
    project: Path,
    c2_output: Path,
    view: ViewSpec,
    model: str,
    evaluation: pd.DataFrame,
    split: str,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    freeze = _main_freeze(project)
    development = _development_samples(shared, view)
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    fit_accessor = BaseAccessor(shared, view.head.dataset, "validation", columns)
    eval_accessor = BaseAccessor(shared, view.head.dataset, split, columns)
    result = _contract(c2_output, model, view)
    selection = result["selection"]

    if model == "RIDGE":
        from sklearn.linear_model import Ridge

        fit = _fit_subset(development, int(freeze["selection"]["fit_row_cap_default"]))
        x_fit = fit_accessor.snapshot(fit, columns)
        scaler = Standardizer().fit(x_fit)
        estimator = Ridge(alpha=float(selection["selected_alpha"]), fit_intercept=True, solver="lsqr", tol=1e-10)
        estimator.fit(scaler.transform(x_fit), fit["y_true"].to_numpy(dtype=np.float64))
        prediction = _chunked_prediction(
            evaluation,
            lambda chunk: eval_accessor.snapshot(chunk, columns),
            lambda matrix: estimator.predict(scaler.transform(matrix)),
        )
        parameters = x_fit.shape[1] + 1
    elif model == "PLS":
        from sklearn.cross_decomposition import PLSRegression

        fit = _fit_subset(development, int(freeze["selection"]["fit_row_cap_default"]))
        x_fit = fit_accessor.snapshot(fit, columns)
        components = int(selection["selected_components"])
        estimator = PLSRegression(n_components=components, scale=True, max_iter=1000, tol=1e-8)
        estimator.fit(x_fit, fit["y_true"].to_numpy(dtype=np.float64))
        prediction = _chunked_prediction(
            evaluation,
            lambda chunk: eval_accessor.snapshot(chunk, columns),
            lambda matrix: estimator.predict(matrix),
        )
        parameters = components * (x_fit.shape[1] + 2) + 1
    elif model == "DPLS":
        from sklearn.cross_decomposition import PLSRegression

        fit = _fit_subset(development, int(freeze["selection"]["fit_row_cap_default"]))
        history = int(selection["selected_history"])
        components = int(selection["selected_components"])
        maximum = int(freeze["c2"]["dpls"]["maximum_lags_per_channel"])
        x_fit = fit_accessor.input_lags(fit, columns, history, maximum)
        estimator = PLSRegression(n_components=components, scale=True, max_iter=1000, tol=1e-8)
        estimator.fit(x_fit, fit["y_true"].to_numpy(dtype=np.float64))
        prediction = _chunked_prediction(
            evaluation,
            lambda chunk: eval_accessor.input_lags(chunk, columns, history, maximum),
            lambda matrix: estimator.predict(matrix),
        )
        parameters = components * (x_fit.shape[1] + 2) + 1
    elif model == "RBF_SVR":
        from sklearn.svm import SVR

        fit = _fit_subset(development, int(freeze["selection"]["fit_row_cap_svr"]))
        x_fit = fit_accessor.snapshot(fit, columns)
        y_fit = fit["y_true"].to_numpy(dtype=np.float64)
        scaler = Standardizer().fit(x_fit)
        y_mean = float(np.mean(y_fit, dtype=np.float64))
        y_scale = float(np.std(y_fit, ddof=0)) or 1.0
        estimator = SVR(
            C=float(_selected(selection["selected_c"])),
            gamma=float(_selected(selection["selected_gamma"])),
            epsilon=float(freeze["c2"]["rbf_svr"]["epsilon_standardized"]),
            kernel="rbf",
            cache_size=4096,
        )
        estimator.fit(scaler.transform(x_fit), (y_fit - y_mean) / y_scale)
        prediction = _chunked_prediction(
            evaluation,
            lambda chunk: eval_accessor.snapshot(chunk, columns),
            lambda matrix: estimator.predict(scaler.transform(matrix)) * y_scale + y_mean,
        )
        parameters = int(len(estimator.support_) * (x_fit.shape[1] + 1))
    elif model == "XGBOOST":
        from xgboost import XGBRegressor

        fit = _fit_subset(development, int(freeze["selection"]["fit_row_cap_xgboost"]))
        depth, rate = _selected(selection["selected_pair"])
        config = freeze["c2"]["xgboost"]
        estimator = XGBRegressor(
            max_depth=int(depth),
            learning_rate=float(rate),
            n_estimators=int(_selected(selection["selected_estimators"])),
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            min_child_weight=config["min_child_weight"],
            reg_lambda=config["reg_lambda"],
            tree_method=config["tree_method"],
            n_jobs=config["nthread_per_fit"],
            objective="reg:squarederror",
            random_state=20260802,
        )
        estimator.fit(
            fit_accessor.snapshot(fit, columns), fit["y_true"].to_numpy(dtype=np.float64), verbose=False
        )
        prediction = _chunked_prediction(
            evaluation,
            lambda chunk: eval_accessor.snapshot(chunk, columns),
            estimator.predict,
        )
        parameters = int(sum(tree.count("leaf") for tree in estimator.get_booster().get_dump()))
    else:
        raise KeyError(model)
    return prediction, parameters, {"development_result_sha256": sha256_file(_result_path(c2_output, model, view))}


def _fit_hammerstein_final(
    shared: Path,
    project: Path,
    c3_output: Path,
    input_view: ViewSpec,
    model: str,
    evaluation: pd.DataFrame,
    split: str,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    from sklearn.isotonic import IsotonicRegression

    dynamic_view = ViewSpec(
        input_view.head,
        "dynamic",
        input_view.availability_scenario,
        input_view.proxy_policy,
    )
    result = _contract(c3_output, model, dynamic_view)
    selection = result["selection"]
    freeze = _main_freeze(project)
    development = _development_samples(shared, input_view)
    fit = _fit_subset(development, int(freeze["selection"]["fit_row_cap_default"]))
    columns = input_columns(shared, input_view.head.task_id, input_view.proxy_policy)
    fit_accessor = BaseAccessor(shared, input_view.head.dataset, "validation", columns)
    eval_accessor = BaseAccessor(shared, input_view.head.dataset, split, columns)
    profile = tuple(selection["selected_profile"])
    raw_fit = fit_accessor.input_regular_lags(fit, columns, *profile, 8)
    x_fit, _ = _nonlinear_features(raw_fit, raw_fit[:1], selection["selected_input_nonlinearity"])
    y_fit = fit["y_true"].to_numpy(dtype=np.float64)
    scaler, coefficient, y_mean, certificate = _linear_fit(x_fit, y_fit, 0.001)
    raw_to_features = lambda raw: _nonlinear_features(
        raw_fit, raw, selection["selected_input_nonlinearity"]
    )[1]
    prediction = _chunked_prediction(
        evaluation,
        lambda chunk: raw_to_features(eval_accessor.input_regular_lags(chunk, columns, *profile, 8)),
        lambda matrix: _linear_predict(scaler, coefficient, y_mean, matrix),
    )
    if selection["selected_output_map"] == "isotonic_train_only":
        train_prediction = _linear_predict(scaler, coefficient, y_mean, x_fit)
        prediction = IsotonicRegression(out_of_bounds="clip").fit(train_prediction, y_fit).predict(prediction)
    return np.asarray(prediction, dtype=np.float64), x_fit.shape[1] + 1, {
        "development_result_sha256": sha256_file(_result_path(c3_output, model, dynamic_view)),
        "numerical_certificate": certificate,
    }


def _dynamic_features(
    shared: Path,
    view: ViewSpec,
    samples: pd.DataFrame,
    split: str,
    profile: tuple[int, int],
) -> np.ndarray:
    accessor = BaseAccessor(shared, view.head.dataset, split, [view.head.target])
    return accessor.target_state(samples, view.head.target, *profile)


def _fit_dynamic_model(
    shared: Path,
    project: Path,
    c2_output: Path,
    c3_output: Path,
    view: ViewSpec,
    model: str,
    evaluation: pd.DataFrame,
    split: str,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    freeze = _main_freeze(project)
    development = _development_samples(shared, view)
    fit = _fit_subset(development, int(freeze["selection"]["fit_row_cap_default"]))

    if model == "LOCAL_LINEAR_TREND":
        result = _contract(c2_output, model, view)
        history = int(result["selection"]["selected_history"])
        accessor = BaseAccessor(shared, view.head.dataset, split, [view.head.target])
        return (
            _chunked_prediction(
                evaluation,
                lambda chunk: chunk,
                lambda chunk: _trend_prediction(accessor, chunk, view.head.target, history, view.head),
            ),
            2,
            {"development_result_sha256": sha256_file(_result_path(c2_output, model, view))},
        )
    if model == "AR":
        result = _contract(c3_output, model, view)
        profile = tuple(result["selection"]["selected_profile"])
        fit_target_accessor = BaseAccessor(shared, view.head.dataset, "validation", [view.head.target])
        evaluation_target_accessor = BaseAccessor(shared, view.head.dataset, split, [view.head.target])
        x_fit = fit_target_accessor.target_state(fit, view.head.target, *profile)
        scaler, coefficient, y_mean, certificate = _linear_fit(
            x_fit,
            fit["y_true"].to_numpy(dtype=np.float64),
            float(result["selection"]["selected_alpha"]),
        )
        prediction = _chunked_prediction(
            evaluation,
            lambda chunk: evaluation_target_accessor.target_state(chunk, view.head.target, *profile),
            lambda matrix: _linear_predict(scaler, coefficient, y_mean, matrix),
        )
        parameters = x_fit.shape[1] + 1
        return prediction, parameters, {
            "development_result_sha256": sha256_file(_result_path(c3_output, model, view)),
            "numerical_certificate": certificate,
        }

    result = _contract(c3_output, model, view)
    selection = result["selection"]
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    all_columns = [view.head.target, *columns]
    fit_accessor = BaseAccessor(shared, view.head.dataset, "validation", all_columns)
    eval_accessor = BaseAccessor(shared, view.head.dataset, split, all_columns)
    profile = tuple(selection.get("selected_profile", (1, 1)))

    if model == "ARX":
        x_fit, ar_width = _arx_features(
            fit_accessor,
            fit,
            view,
            columns,
            profile,
            int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"]),
        )
        alpha = float(selection["ar_alpha"])
        ratio = selection["selected_x_penalty_ratio"]
        if ratio == "EXACT_X_ZERO":
            linear_fit = _linear_fit(
                x_fit[:, :ar_width], fit["y_true"].to_numpy(dtype=np.float64), alpha
            )
            parameters = ar_width + 1
        else:
            penalties = np.concatenate(
                [np.full(ar_width, alpha), np.full(x_fit.shape[1] - ar_width, alpha * float(ratio))]
            )
            linear_fit = _linear_fit(x_fit, fit["y_true"].to_numpy(dtype=np.float64), penalties)
            parameters = x_fit.shape[1] + 1
        scaler, coefficient, y_mean, certificate = linear_fit
        def arx_chunk(chunk: pd.DataFrame) -> np.ndarray:
            matrix, _ = _arx_features(
                eval_accessor,
                chunk,
                view,
                columns,
                profile,
                int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"]),
            )
            return matrix[:, :ar_width] if ratio == "EXACT_X_ZERO" else matrix
        prediction = _chunked_prediction(
            evaluation, arx_chunk, lambda matrix: _linear_predict(scaler, coefficient, y_mean, matrix)
        )
    elif model == "LINEAR_NARX":
        raw_fit, _ = _arx_features(
            fit_accessor,
            fit,
            view,
            columns,
            profile,
            int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"]),
        )
        scaler = Standardizer().fit(raw_fit)
        selected = np.asarray(selection["selected_linear_feature_indices"], dtype=np.int64)
        standardized_fit = scaler.transform(raw_fit)[:, selected]
        x_fit = np.concatenate([standardized_fit, np.square(standardized_fit)], axis=1)
        linear_scaler, coefficient, y_mean, certificate = _linear_fit(
            x_fit,
            fit["y_true"].to_numpy(dtype=np.float64),
            float(selection["selected_alpha"]),
        )
        def narx_chunk(chunk: pd.DataFrame) -> np.ndarray:
            raw, _ = _arx_features(
                eval_accessor,
                chunk,
                view,
                columns,
                profile,
                int(freeze["c3"]["arx"]["maximum_input_lags_per_channel"]),
            )
            standardized = scaler.transform(raw)[:, selected]
            return np.concatenate([standardized, np.square(standardized)], axis=1)
        prediction = _chunked_prediction(
            evaluation,
            narx_chunk,
            lambda matrix: _linear_predict(linear_scaler, coefficient, y_mean, matrix),
        )
        parameters = x_fit.shape[1] + 1
    elif model == "N4SID":
        return _fit_n4sid_final(shared, c3_output, view, result, evaluation, split)
    else:
        raise KeyError(model)
    return prediction, parameters, {
        "development_result_sha256": sha256_file(_result_path(c3_output, model, view)),
        "numerical_certificate": certificate,
    }


def _base_partitions(split: str) -> list[str]:
    order = ["train", "validation", "test", "ood"]
    return order[: order.index(split) + 1]


def _base_frame(shared: Path, view: ViewSpec, split: str, columns: list[str]) -> pd.DataFrame:
    frames = []
    for partition in _base_partitions(split):
        path = shared / "base_data" / view.head.dataset / f"{partition}.parquet"
        if path.is_file():
            frames.append(pd.read_parquet(path, columns=[*columns, view.head.target, "entity_id", "row_in_entity"]))
    if not frames:
        raise FileNotFoundError((view.head.dataset, split))
    return pd.concat(frames, ignore_index=True).sort_values(["entity_id", "row_in_entity"])


def _fit_n4sid_final(
    shared: Path,
    c3_output: Path,
    view: ViewSpec,
    result: dict[str, Any],
    evaluation: pd.DataFrame,
    split: str,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    development_base = _base_frame(shared, view, "validation", columns)
    if development_base["entity_id"].nunique() != 1:
        raise ValueError("N4SID final fit requires one continuous entity; development failure should have been retained")
    system = _fit_n4sid(
        development_base.sort_values("row_in_entity"),
        view.head.target,
        columns,
        int(result["selection"]["selected_order"]),
    )
    evaluation_base = _base_frame(shared, view, split, columns)
    if evaluation_base["entity_id"].nunique() != 1:
        raise ValueError("N4SID evaluation requires one continuous entity")
    prediction = _n4sid_predict(
        system,
        evaluation_base.sort_values("row_in_entity"),
        evaluation,
        view.head.target,
        columns,
        view.head,
    )
    parameters = sum(matrix.size for matrix in (system.a, system.b, system.c, system.d, system.k))
    return prediction, parameters, {
        "development_result_sha256": sha256_file(_result_path(c3_output, "N4SID", view)),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(system.a)))) if system.a.size else 0.0,
    }


def _core_history(
    c2_output: Path,
    c3_output: Path,
    c4_output: Path,
    c5_output: Path,
    input_view: ViewSpec,
    dynamic_view: ViewSpec,
) -> int:
    histories = [input_view.head.h_steps + input_view.head.w_steps]
    for model in ("DPLS",):
        result = _contract(c2_output, model, input_view)
        histories.append(int(result["selection"]["selected_history"]))
    for model in ("LOCAL_LINEAR_TREND",):
        result = _contract(c2_output, model, dynamic_view)
        histories.append(int(result["selection"]["selected_history"]))
    for model in ("AR", "ARX", "LINEAR_NARX"):
        result = _contract(c3_output, model, dynamic_view)
        histories.append(int(result["selection"]["selected_profile"][1]))
    for model in ("PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"):
        result = _contract(c3_output, model, dynamic_view)
        histories.append(int(result["selection"]["selected_profile"][1]))
    c4 = _c4_result(c4_output, input_view)
    histories.extend(int(item["selected_profile"][1]) for item in c4["selected_channel_contracts"])
    c5 = _c5_result(c5_output, dynamic_view)
    if c5["physics_first"].get("selected_profile") is not None:
        histories.append(int(c5["physics_first"]["selected_profile"][1]))
    histories.append(int(c5["k_joint_ar"]["state_profile"][1]))
    return max(histories)


def _prediction_path(output: Path, view: ViewSpec, split: str, model: str) -> Path:
    return output / "PREDICTIONS" / view.head.head_id / split / f"{view.information_set}__{model}.parquet"


def _audit_path(output: Path, view: ViewSpec, split: str, model: str) -> Path:
    return output / "MODEL_AUDIT" / view.head.head_id / split / f"{view.information_set}__{model}.json"


def _validate_cached(path: Path, shared: Path, view: ViewSpec, split: str, model: str) -> bool:
    if not path.is_file():
        return False
    output = path.parents[3]
    audit_path = _audit_path(output, view, split, model)
    if audit_path.is_file():
        audit = _load_json(audit_path)
        if (
            audit.get("status") == "PASS"
            and audit.get("model") == model
            and audit.get("target_head") == view.head.head_id
            and audit.get("split") == split
            and audit.get("information_set") == view.information_set
            and audit.get("prediction_sha256") == sha256_file(path)
        ):
            import pyarrow.parquet as pq

            return int(audit.get("rows", -1)) == pq.ParquetFile(path).metadata.num_rows
    samples = load_samples(shared, view, split)
    frame = pd.read_parquet(path, columns=["sample_id", "base_origin_id", "model", "information_set", "y_true"])
    return (
        len(frame) == len(samples)
        and set(frame["model"].astype(str)) == {model}
        and set(frame["information_set"].astype(str)) == {view.information_set}
        and np.array_equal(frame["sample_id"].to_numpy(), samples["view_sample_id"].to_numpy())
        and np.array_equal(frame["base_origin_id"].to_numpy(), samples["base_origin_id"].to_numpy())
        and np.array_equal(frame["y_true"].to_numpy(dtype=np.float64), samples["y_true"].to_numpy(dtype=np.float64))
    )


def _evaluate_baseline_job(args: tuple[Any, ...]) -> dict[str, Any]:
    shared, project, c2_output, c3_output, output, view, split, model, core_history_steps = args
    started = time.time()
    path = _prediction_path(output, view, split, model)
    try:
        if _validate_cached(path, shared, view, split, model):
            return {"status": "PASS", "model": model, "target_head": view.head.head_id, "split": split, "information_set": view.information_set, "prediction_path": str(path), "resumed": True}
        samples = load_samples(shared, view, split)
        if model == "PERSISTENCE":
            prediction, parameters, audit = np.zeros(len(samples), dtype=np.float64), 0, {"formula": "registered_change_equals_zero"}
        elif model == "MEAN":
            development = _development_samples(shared, view)
            prediction = np.full(len(samples), float(development["y_true"].mean()), dtype=np.float64)
            parameters, audit = 1, {"fit_partition": "train_plus_validation"}
        elif model in {"RIDGE", "PLS", "DPLS", "RBF_SVR", "XGBOOST"}:
            prediction, parameters, audit = _fit_c2_model(shared, project, c2_output, view, model, samples, split)
        elif model in {"PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"}:
            prediction, parameters, audit = _fit_hammerstein_final(shared, project, c3_output, view, model, samples, split)
        else:
            prediction, parameters, audit = _fit_dynamic_model(shared, project, c2_output, c3_output, view, model, samples, split)
        frame = _frame(samples, view, model, prediction, parameters, split, core_history_steps)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")
        audit.update(
            {
                "status": "PASS",
                "model": model,
                "target_head": view.head.head_id,
                "split": split,
                "information_set": view.information_set,
                "fit_partition": "train_plus_validation_only",
                "evaluation_partition": split,
                "rows": len(frame),
                "parameter_count": int(parameters),
                "elapsed_seconds": time.time() - started,
                "prediction_sha256": sha256_file(path),
            }
        )
        write_json(_audit_path(output, view, split, model), audit)
        return {**audit, "prediction_path": str(path), "resumed": False}
    except Exception as error:
        failure = {
            "status": "FAILED_RETAINED",
            "model": model,
            "target_head": view.head.head_id,
            "split": split,
            "information_set": view.information_set,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        write_json(_audit_path(output, view, split, model), failure)
        return failure


def _evaluate_prism_job(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    shared, project, c3_output, c4_output, c5_output, output, input_view, split, core_history_steps = args
    dynamic_view = ViewSpec(input_view.head, "dynamic", input_view.availability_scenario, input_view.proxy_policy)
    models = [
        (input_view, "PRISM_CHANNEL_SPECIFIC"),
        (dynamic_view, "PRISM_PHYSICS_FIRST"),
        (dynamic_view, "PRISM_K_JOINT_AR"),
    ]
    if all(_validate_cached(_prediction_path(output, view, split, model), shared, view, split, model) for view, model in models):
        return [{"status": "PASS", "model": model, "target_head": view.head.head_id, "split": split, "information_set": view.information_set, "prediction_path": str(_prediction_path(output, view, split, model)), "resumed": True} for view, model in models]
    started = time.time()
    try:
        input_samples = load_samples(shared, input_view, split)
        dynamic_samples = load_samples(shared, dynamic_view, split)
        if not np.array_equal(input_samples["base_origin_id"].to_numpy(), dynamic_samples["base_origin_id"].to_numpy()):
            raise AssertionError("input-only and dynamic PRISM samples do not share base origins")
        contracts = list(_c4_result(c4_output, input_view)["selected_channel_contracts"])
        k_prediction, train_shapes, evaluation_shapes, fit = _fit_k_final(
            shared, project, input_view, input_samples, contracts
        )
        physics_prediction, physics_parameters = _physics_first_final(
            shared, project, c4_output, c5_output, dynamic_view, dynamic_samples, split, k_prediction
        )
        joint_prediction, joint_parameters = _k_joint_final(
            shared,
            project,
            c3_output,
            c5_output,
            dynamic_view,
            dynamic_samples,
            split,
            train_shapes,
            evaluation_shapes,
            fit,
        )
        payloads = [
            (input_view, "PRISM_CHANNEL_SPECIFIC", input_samples, k_prediction, len(contracts) + 1),
            (dynamic_view, "PRISM_PHYSICS_FIRST", dynamic_samples, physics_prediction, physics_parameters),
            (dynamic_view, "PRISM_K_JOINT_AR", dynamic_samples, joint_prediction, joint_parameters),
        ]
        results = []
        elapsed = time.time() - started
        for view, model, samples, prediction, parameters in payloads:
            path = _prediction_path(output, view, split, model)
            path.parent.mkdir(parents=True, exist_ok=True)
            _frame(samples, view, model, prediction, parameters, split, core_history_steps).to_parquet(
                path, index=False, compression="zstd"
            )
            audit = {
                "status": "PASS",
                "model": model,
                "target_head": view.head.head_id,
                "split": split,
                "information_set": view.information_set,
                "fit_partition": "train_plus_validation_only",
                "evaluation_partition": split,
                "rows": len(samples),
                "parameter_count": int(parameters),
                "elapsed_seconds_shared_prism_bundle": elapsed,
                "prediction_sha256": sha256_file(path),
            }
            write_json(_audit_path(output, view, split, model), audit)
            results.append({**audit, "prediction_path": str(path), "resumed": False})
        return results
    except Exception as error:
        return [
            {
                "status": "FAILED_RETAINED",
                "model": model,
                "target_head": view.head.head_id,
                "split": split,
                "information_set": view.information_set,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
            }
            for view, model in models
        ]


def write_full_final_freeze_manifest(
    shared: Path,
    project: Path,
    c2_output: Path,
    c3_output: Path,
    c4_output: Path,
    c5_output: Path,
    output: Path,
) -> Path:
    existing = output / "FULL_FINAL_FREEZE_MANIFEST.json"
    if existing.is_file():
        payload = _load_json(existing)
        if payload.get("status") == "FROZEN" and payload.get("protocol") == "PRISM_CPU_C6_FULL_FINAL_V2":
            return existing
    repair_config = project / "configs/c6_full_final_v2.json"
    prerequisites = [
        c2_output / "C2_CLASSICAL_MANIFEST.json",
        c3_output / "C3_MANIFEST.json",
        c4_output / "C4_MANIFEST.json",
        c5_output / "C5_MANIFEST.json",
    ]
    for path in prerequisites:
        status = _load_json(path).get("status")
        if status not in {"PASS", "PASS_WITH_RETAINED_FAILURES"}:
            raise RuntimeError(f"incomplete prerequisite: {path} status={status}")
    frozen = [
        repair_config,
        project / "configs/cpu_model_freeze_v1.json",
        shared / "TASK_REGISTRY.json",
        shared / "PROTOCOL.json",
        shared / "dataset_views/VIEW_REGISTRY.json",
        *prerequisites,
        *sorted(c2_output.rglob("RESULT.json")),
        *sorted(c3_output.rglob("RESULT.json")),
        *sorted(c4_output.rglob("RESULT.json")),
        *sorted(c5_output.rglob("RESULT.json")),
    ]
    manifest = {
        "status": "FROZEN",
        "protocol": "PRISM_CPU_C6_FULL_FINAL_V2",
        "repair_reason": "C6 V1 evaluated only a predeclared subset and omitted formal CPU baselines",
        "selection_policy": "reuse_C2_C5_development_contracts_without_reselection",
        "prior_subset_test_access_disclosed": True,
        "result_dependent_model_inclusion": False,
        "test_access_authorized": True,
        "test_accessed_at_manifest_creation": False,
        "repair_config": _repair_freeze(project),
        "frozen_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in frozen
        ],
    }
    path = existing
    write_json(path, manifest)
    return path


def _prediction_inventory(output: Path) -> list[Path]:
    return sorted((output / "PREDICTIONS").rglob("*.parquet"))


def _parquet_first_row(path: Path, columns: list[str]) -> dict[str, Any]:
    """Read scalar prediction metadata without materialising repeated columns."""
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows < 1:
        raise AssertionError(f"empty prediction file: {path}")
    return parquet.read_row_group(0, columns=columns).slice(0, 1).to_pylist()[0]


def _identity_hash(path: Path, column: str) -> np.ndarray:
    """Compact ordered identity used while streaming multi-million-row files."""
    values = pd.read_parquet(path, columns=[column])[column]
    return pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64, copy=True)


def _metrics_and_statistics(
    output: Path,
    config: dict[str, Any],
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = _prediction_inventory(output)
    groups: dict[tuple[str, str, str], list[Path]] = {}
    for path in inventory:
        head, split = path.parent.parts[-2:]
        information_set, _ = path.stem.split("__", 1)
        groups.setdefault((head, split, information_set), []).append(path)

    metric_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    model_order = {
        info: {model: index for index, model in enumerate(models)}
        for info, models in config["leaderboards"].items()
    }
    for key, paths in sorted(groups.items()):
        ordered_paths = sorted(
            paths,
            key=lambda path: (
                model_order.get(key[2], {}).get(path.stem.split("__", 1)[1], 10_000),
                path.stem,
            ),
        )
        reference_path = ordered_paths[0]
        reference_meta = _parquet_first_row(
            reference_path,
            ["dataset", "task", "h_steps", "w_steps", "core_history_steps"],
        )
        y_true = pd.read_parquet(reference_path, columns=["y_true"])["y_true"].to_numpy(
            dtype=np.float64, copy=True
        )
        sample_hash = _identity_hash(reference_path, "sample_id")
        entities_raw = pd.read_parquet(reference_path, columns=["profile_id"])["profile_id"].astype(str).to_numpy()
        labels, entity_groups = _entity_groups(entities_raw)
        del entities_raw

        predictions: dict[str, np.ndarray] = {}
        for path in ordered_paths:
            model = path.stem.split("__", 1)[1]
            meta = _parquet_first_row(path, ["model", "parameter_count"])
            if str(meta["model"]) != model:
                raise AssertionError(f"prediction model identity mismatch: {path}")
            candidate_hash = _identity_hash(path, "sample_id")
            if not np.array_equal(candidate_hash, sample_hash):
                raise AssertionError(f"unpaired finalist sample IDs: {path}")
            del candidate_hash
            candidate_true = pd.read_parquet(path, columns=["y_true"])["y_true"].to_numpy(
                dtype=np.float64, copy=False
            )
            if not np.array_equal(candidate_true, y_true):
                raise AssertionError(f"prediction targets changed: {path}")
            del candidate_true
            prediction = pd.read_parquet(path, columns=["y_pred"])["y_pred"].to_numpy(
                dtype=np.float64, copy=True
            )
            predictions[model] = prediction
            row = {
                "dataset": str(reference_meta["dataset"]),
                "task": str(reference_meta["task"]),
                "target_head": key[0],
                "split": key[1],
                "information_set": key[2],
                "model": model,
                "rows": len(y_true),
                "parameter_count": int(meta["parameter_count"]),
                **regression_metrics(y_true, prediction),
            }
            metric_rows.append(row)
            for entity, index in zip(labels, entity_groups, strict=True):
                entity_rows.append(
                    {
                        "target_head": key[0],
                        "split": key[1],
                        "information_set": key[2],
                        "model": model,
                        "entity_id": str(entity),
                        "rows": len(index),
                        **regression_metrics(y_true[index], prediction[index]),
                    }
                )

        ordered = sorted(predictions, key=lambda model: (model_order.get(key[2], {}).get(model, 10_000), model))
        pair_batch: list[tuple[Any, ...]] = []
        for reference, model in itertools.combinations(ordered, 2):
            squared_model = np.square(predictions[model] - y_true)
            squared_reference = np.square(predictions[reference] - y_true)
            diff = squared_reference - squared_model
            base_block = max(1, int(reference_meta["h_steps"]) + int(reference_meta["w_steps"]))
            history_block = max(1, int(math.ceil(int(reference_meta["core_history_steps"]) / 4)))
            for block in sorted({base_block, 2 * base_block, history_block}):
                seed_text = f"{key}|{model}|{reference}|{block}|{config['bootstrap_seed']}"
                seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
                pair_batch.append(
                    (
                        diff,
                        labels,
                        entity_groups,
                        block,
                        int(config["bootstrap_replicates"]),
                        seed,
                        {
                            "target_head": key[0],
                            "split": key[1],
                            "information_set": key[2],
                            "model": model,
                            "reference": reference,
                            "block_length": block,
                        },
                    )
                )
            if len(pair_batch) >= 15:
                bootstrap_rows.extend(_execute_bootstrap_tasks(pair_batch, n_jobs))
                pair_batch = []
        if pair_batch:
            bootstrap_rows.extend(_execute_bootstrap_tasks(pair_batch, n_jobs))

    metrics = pd.DataFrame(metric_rows)
    entities = pd.DataFrame(entity_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    if metrics.empty:
        raise RuntimeError("no repaired C6 metrics")
    for _, indices in bootstrap.groupby(
        ["target_head", "split", "information_set", "block_length"], sort=False
    ).groups.items():
        family = [bootstrap.loc[index].to_dict() for index in indices]
        _holm(family, float(config["holm_alpha"]))
        for index, row in zip(indices, family, strict=True):
            bootstrap.loc[index, "holm_adjusted_p"] = row["holm_adjusted_p"]
            bootstrap.loc[index, "holm_reject"] = row["holm_reject"]

    reference = metrics[["target_head", "split", "information_set", "model", "mse"]].copy()
    persistence = reference[reference.model == "PERSISTENCE"].drop(columns="model").rename(columns={"mse": "persistence_mse"})
    ar = reference[reference.model == "AR"].drop(columns="model").rename(columns={"mse": "ar_mse"})
    metrics = metrics.merge(persistence, on=["target_head", "split", "information_set"], how="left")
    metrics = metrics.merge(ar, on=["target_head", "split", "information_set"], how="left")
    metrics["relative_persistence_skill"] = (metrics["persistence_mse"] - metrics["mse"]) / metrics["persistence_mse"]
    metrics["relative_ar_skill"] = (metrics["ar_mse"] - metrics["mse"]) / metrics["ar_mse"]
    metrics["rank"] = metrics.groupby(["target_head", "split", "information_set"])["mse"].rank(method="average")
    expected = metrics.groupby(["split", "information_set"])["target_head"].nunique().rename("expected_tasks")
    ranks = metrics.groupby(["split", "information_set", "model"], as_index=False).agg(
        mean_rank=("rank", "mean"), median_rank=("rank", "median"), tasks=("target_head", "nunique")
    )
    ranks = ranks.merge(expected, on=["split", "information_set"], how="left")
    ranks["coverage"] = ranks["tasks"] / ranks["expected_tasks"]
    ranks["complete_coverage"] = ranks["tasks"] == ranks["expected_tasks"]
    return metrics, bootstrap, ranks, entities


def _failure_registry(
    repair_config: dict[str, Any],
    results: list[dict[str, Any]],
    heads: list[ViewSpec],
) -> list[dict[str, Any]]:
    failures = [result for result in results if result.get("status") != "PASS"]
    for view in heads:
        for split in ("test", "ood"):
            failures.append(
                {
                    "status": "NOT_APPLICABLE",
                    "model": "SEASONAL_PERSISTENCE",
                    "target_head": view.head.head_id,
                    "split": split,
                    "information_set": "dynamic",
                    "reason": repair_config["not_applicable_models"]["SEASONAL_PERSISTENCE"],
                }
            )
    return failures


def _build_report(
    output: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ranks: pd.DataFrame,
    failures: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    complete_ranks = ranks[ranks["complete_coverage"]].sort_values(
        ["split", "information_set", "mean_rank", "model"]
    )
    winners = (
        metrics.sort_values(["target_head", "split", "information_set", "mse", "model"])
        .groupby(["target_head", "split", "information_set"], as_index=False)
        .first()[["target_head", "split", "information_set", "model", "mse", "relative_persistence_skill", "relative_ar_skill"]]
    )
    failure_frame = pd.DataFrame(failures)
    failure_summary = (
        failure_frame.groupby(["status", "model"], dropna=False).size().rename("records").reset_index()
        if not failure_frame.empty
        else pd.DataFrame(columns=["status", "model", "records"])
    )
    lines = [
        "# PRISM Industrial CPU Benchmark — Full C6 Repair Report",
        "",
        "Status: `PASS_WITH_RETAINED_FAILURES`",
        "",
        "## What was repaired",
        "",
        "The earlier C6 V1 release evaluated only Persistence, AR and three PRISM routes. It did not final-fit the other CPU baselines that had already completed C2/C3 development. This V2 layer preserves C1–C5 selections, adds every pre-registered successful CPU baseline to formal test/OOD evaluation, and retains unavailable methods as explicit failures or not-applicable records.",
        "",
        "The repaired inclusion list and statistics policy were frozen before these new baseline test predictions were read. The earlier subset test access is disclosed in `FULL_FINAL_FREEZE_MANIFEST.json`; no C2–C5 contract was reselected from test results.",
        "",
        "## Protocol-complete cross-task ranks",
        "",
        _markdown_table(complete_ranks),
        "",
        "## Per-task winners",
        "",
        _markdown_table(winners),
        "",
        "## Retained failures and not-applicable methods",
        "",
        _markdown_table(failure_summary),
        "",
        "N4SID development failures remain unavailable for the affected heads; no test score was fabricated. Seasonal Persistence remains not applicable because no train-only season was registered.",
        "",
        "## Statistical contract",
        "",
        f"All successful models retain per-sample predictions. There are {len(bootstrap)} paired block-bootstrap rows with {config['bootstrap_replicates']} replicates per comparison. Holm correction is applied within each target/split/information-set/block-length family. Cross-task primary ranks include only methods with complete task coverage; partial methods remain visible with coverage in `CROSS_TASK_RANKS.csv`.",
        "",
        "## Files",
        "",
        "- `CPU_FINAL_METRICS.csv`: MSE/RMSE/MAE/R2/NRMSE, relative Persistence skill, dynamic AR skill and task rank.",
        "- `ENTITY_METRICS.csv`: per-realization/profile error distributions.",
        "- `BOOTSTRAP_ALL_PAIRS.csv`: paired block-bootstrap comparisons and Holm decisions.",
        "- `CROSS_TASK_RANKS.csv`: rank and coverage audit.",
        "- `C6_FAILURES.json`: every retained failure and not-applicable record.",
        "- `MODEL_AUDIT/`: final fit source contract hashes, fit partitions, parameter counts and numerical certificates.",
    ]
    (output / "CPU_FULL_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_c6(
    shared: Path,
    project: Path,
    c2_output: Path,
    c3_output: Path,
    c4_output: Path,
    c5_output: Path,
    output: Path,
    n_jobs: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = write_full_final_freeze_manifest(
        shared, project, c2_output, c3_output, c4_output, c5_output, output
    )
    config = _repair_freeze(project)
    input_views = main_views(shared, "input_only")
    jobs: list[tuple[Any, ...]] = []
    prism_jobs: list[tuple[Any, ...]] = []
    skipped_absent_splits: list[dict[str, Any]] = []
    for input_view in input_views:
        dynamic_view = ViewSpec(
            input_view.head,
            "dynamic",
            input_view.availability_scenario,
            input_view.proxy_policy,
        )
        core_history_steps = _core_history(
            c2_output, c3_output, c4_output, c5_output, input_view, dynamic_view
        )
        for split in ("test", "ood"):
            sample_path = shared / "sample_ids" / input_view.relative_root / f"{split}.parquet"
            if not sample_path.is_file():
                skipped_absent_splits.append(
                    {"status": "NOT_APPLICABLE", "target_head": input_view.head.head_id, "split": split, "reason": "SPLIT_NOT_REGISTERED"}
                )
                continue
            for model in config["leaderboards"]["input_only"]:
                if model != "PRISM_CHANNEL_SPECIFIC":
                    jobs.append((shared, project, c2_output, c3_output, output, input_view, split, model, core_history_steps))
            for model in config["leaderboards"]["dynamic"]:
                if model not in {"PRISM_PHYSICS_FIRST", "PRISM_K_JOINT_AR"}:
                    jobs.append((shared, project, c2_output, c3_output, output, dynamic_view, split, model, core_history_steps))
            prism_jobs.append((shared, project, c3_output, c4_output, c5_output, output, input_view, split, core_history_steps))

    results: list[dict[str, Any]] = []
    large_jobs = [job for job in jobs if job[5].head.dataset == "tep"]
    regular_jobs = [job for job in jobs if job[5].head.dataset != "tep"]
    large_prism = [job for job in prism_jobs if job[6].head.dataset == "tep"]
    regular_prism = [job for job in prism_jobs if job[6].head.dataset != "tep"]

    regular_workers = max(1, min(n_jobs, 10, len(regular_jobs) + len(regular_prism)))
    with ProcessPoolExecutor(max_workers=regular_workers) as executor:
        futures = [executor.submit(_evaluate_baseline_job, job) for job in regular_jobs]
        futures.extend(executor.submit(_evaluate_prism_job, job) for job in regular_prism)
        for future in as_completed(futures):
            value = future.result()
            results.extend(value if isinstance(value, list) else [value])

    # A TEP primary split contains up to 7.6 million immutable origins. Each
    # full feature/evaluation frame needs roughly 12-16 GiB, so large jobs use
    # a fresh one-worker pool. This preserves all rows and FP64 while ensuring
    # native-library allocations are released between models.
    for job in large_jobs:
        with ProcessPoolExecutor(max_workers=1) as executor:
            value = executor.submit(_evaluate_baseline_job, job).result()
        results.extend(value if isinstance(value, list) else [value])
    for job in large_prism:
        with ProcessPoolExecutor(max_workers=1) as executor:
            value = executor.submit(_evaluate_prism_job, job).result()
        results.extend(value if isinstance(value, list) else [value])

    failures = _failure_registry(config, results, input_views)
    failures.extend(skipped_absent_splits)
    write_json(output / "C6_FAILURES.json", failures)
    pd.DataFrame([{key: value for key, value in row.items() if key != "traceback"} for row in results]).to_csv(
        output / "FINAL_FIT_REGISTRY.csv", index=False
    )

    metrics, bootstrap, ranks, entities = _metrics_and_statistics(output, config, n_jobs)
    metrics.to_csv(output / "CPU_FINAL_METRICS.csv", index=False)
    bootstrap.to_csv(output / "BOOTSTRAP_ALL_PAIRS.csv", index=False)
    ranks.to_csv(output / "CROSS_TASK_RANKS.csv", index=False)
    entities.to_csv(output / "ENTITY_METRICS.csv", index=False)
    _build_report(output, metrics, bootstrap, ranks, failures, config)
    decision = {
        "status": "PASS" if not failures else "PASS_WITH_RETAINED_FAILURES",
        "stage": "C6_FULL_FINAL_V2",
        "test_accessed": True,
        "successful_prediction_files": len(_prediction_inventory(output)),
        "fit_registry_rows": len(results),
        "metric_rows": len(metrics),
        "bootstrap_rows": len(bootstrap),
        "entity_metric_rows": len(entities),
        "failure_and_not_applicable_records": len(failures),
        "full_final_freeze_sha256": sha256_file(manifest),
        "repair_config_sha256": sha256_file(project / "configs/c6_full_final_v2.json"),
        "n_jobs": int(n_jobs),
    }
    write_json(output / "CPU_FULL_FINAL_DECISION.json", decision)
    return decision
