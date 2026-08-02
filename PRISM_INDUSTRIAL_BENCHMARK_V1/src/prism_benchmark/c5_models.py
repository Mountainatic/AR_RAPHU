from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .c2_models import _prediction_frame
from .c3_models import _load_ar_contract, _ridge_block_predict
from .c4_prism import _contract_prediction, _fit_candidate, profile_values
from .cpu_data import BaseAccessor, ViewSpec, deterministic_subsample, load_samples, main_views, realized_state_profiles, sha256_file
from .cpu_selection import Standardizer, mse, regression_metrics, select_one_se
from .stage0 import write_json


def _freeze(project: Path) -> dict[str, Any]:
    return json.loads((project / "configs/cpu_model_freeze_v1.json").read_text(encoding="utf-8"))


def rolling_oof_folds(samples: pd.DataFrame, count: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding folds inside every entity, with dependency-interval separation."""
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    entity_series = samples["entity_id"].astype(str)
    entity_groups = entity_series.groupby(entity_series, sort=False).indices
    dependency_start = samples["dependency_start"].to_numpy(dtype=np.int64)
    dependency_stop = samples["dependency_stop_exclusive"].to_numpy(dtype=np.int64)
    for fold in range(count):
        train_parts: list[np.ndarray] = []
        validation_parts: list[np.ndarray] = []
        for entity_index in entity_groups.values():
            entity_index = np.asarray(entity_index, dtype=np.int64)
            ordered = entity_index[np.argsort(samples.iloc[entity_index]["origin"].to_numpy(dtype=np.int64))]
            if len(ordered) < count + 1:
                continue
            chunks = np.array_split(ordered, count + 1)
            train_candidate = np.concatenate(chunks[: fold + 1])
            validation_candidate = chunks[fold + 1]
            if not len(train_candidate) or not len(validation_candidate):
                continue
            boundary = int(np.min(dependency_start[validation_candidate]))
            train_index = train_candidate[dependency_stop[train_candidate] <= boundary]
            validation_index = validation_candidate[dependency_start[validation_candidate] >= boundary]
            if len(train_index) and len(validation_index):
                train_parts.append(train_index)
                validation_parts.append(validation_index)
        if not train_parts or not validation_parts:
            raise ValueError(f"rolling OOF fold {fold} is empty")
        folds.append((np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(validation_parts))))
    return folds


def _residual_offsets(delta_steps: int, history_steps: int, maximum_lags: int) -> np.ndarray:
    offsets = np.arange(0, history_steps + 1, max(1, delta_steps), dtype=np.int64)
    if len(offsets) > maximum_lags:
        offsets = offsets[np.unique(np.rint(np.linspace(0, len(offsets) - 1, maximum_lags)).astype(np.int64))]
    return offsets


def mature_residual_features(
    samples: pd.DataFrame,
    residuals: pd.DataFrame,
    horizon_steps: int,
    window_steps: int,
    delta_steps: int,
    history_steps: int,
    maximum_lags: int,
    residual_mean: float,
) -> np.ndarray:
    offsets = _residual_offsets(delta_steps, history_steps, maximum_lags)
    result = np.zeros((len(samples), len(offsets)), dtype=np.float64)
    sample_entity_series = samples["entity_id"].astype(str)
    sample_groups = sample_entity_series.groupby(sample_entity_series, sort=False).indices
    origins = samples["origin"].to_numpy(dtype=np.int64)
    residual_entity_series = residuals["entity_id"].astype(str)
    residual_groups = residual_entity_series.groupby(residual_entity_series, sort=False).indices
    for entity, sample_index in sample_groups.items():
        sample_index = np.asarray(sample_index, dtype=np.int64)
        residual_index = residual_groups.get(entity)
        if residual_index is None or not len(residual_index):
            continue
        residual_index = np.asarray(residual_index, dtype=np.int64)
        raw_origins = residuals.iloc[residual_index]["origin"].to_numpy(dtype=np.int64)
        raw_values = residuals.iloc[residual_index]["residual"].to_numpy(dtype=np.float64) - residual_mean
        # Preserve the legacy dict rule for duplicate origins (last value wins),
        # then perform every lag lookup for the entity in compiled NumPy code.
        lookup = {int(origin): float(value) for origin, value in zip(raw_origins, raw_values, strict=True)}
        entity_origins = np.asarray(sorted(lookup), dtype=np.int64)
        entity_values = np.asarray([lookup[int(origin)] for origin in entity_origins], dtype=np.float64)
        latest_mature = origins[sample_index] - horizon_steps - window_steps
        queries = latest_mature[:, None] - offsets[None, :]
        positions = np.searchsorted(entity_origins, queries)
        valid = positions < len(entity_origins)
        safe_positions = np.minimum(positions, len(entity_origins) - 1)
        valid &= entity_origins[safe_positions] == queries
        values = np.zeros(queries.shape, dtype=np.float64)
        values[valid] = entity_values[safe_positions[valid]]
        result[sample_index] = values
    return result


def _ridge_path_losses(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    alphas: list[float],
) -> list[float]:
    """Evaluate the frozen ridge path with one FP64 sufficient-statistics pass."""
    scaler = Standardizer().fit(x_train)
    train = scaler.transform(x_train)
    evaluation = scaler.transform(x_evaluation)
    y_mean = float(np.mean(y_train, dtype=np.float64))
    centered_y = np.asarray(y_train, dtype=np.float64) - y_mean
    gram = train.T @ train
    rhs = train.T @ centered_y
    identity = np.eye(train.shape[1], dtype=np.float64)
    losses = []
    for alpha in alphas:
        system = gram + identity * float(alpha)
        try:
            coefficient = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            coefficient = np.linalg.lstsq(system, rhs, rcond=1e-12)[0]
        prediction = evaluation @ coefficient + y_mean
        losses.append(mse(y_evaluation, prediction))
    return losses


def _residual_profile_fold_path(
    fold: int,
    profile: tuple[int, int],
    fit_rows: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    oof: pd.DataFrame,
    view: ViewSpec,
    maximum_lags: int,
    residual_mean: float,
    alpha_grid: list[float],
) -> tuple[int, tuple[int, int], list[float]]:
    fit_x = mature_residual_features(
        fit_rows, oof, view.head.h_steps, view.head.w_steps, *profile, maximum_lags, residual_mean
    )
    evaluation_x = mature_residual_features(
        evaluation_rows, oof, view.head.h_steps, view.head.w_steps, *profile, maximum_lags, residual_mean
    )
    path_losses = _ridge_path_losses(
        fit_x,
        fit_rows["residual"].to_numpy(dtype=np.float64),
        evaluation_x,
        evaluation_rows["residual"].to_numpy(dtype=np.float64),
        alpha_grid,
    )
    return fold, profile, path_losses


def _load_c4_joint(c4_output: Path, view: ViewSpec) -> dict[str, Any]:
    path = c4_output / "JOINT" / view.head.head_id / view.proxy_policy / "RESULT.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError(f"C4 joint prerequisite is not PASS: {path}")
    return result


def _fit_frozen_channel_shapes(
    shared: Path,
    project: Path,
    view: ViewSpec,
    train_samples: pd.DataFrame,
    evaluation_samples: pd.DataFrame,
    contracts: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    if not contracts:
        return np.zeros((len(train_samples), 0), dtype=np.float64), np.zeros((len(evaluation_samples), 0), dtype=np.float64)
    solver = _freeze(project)["c4"]["solver"]
    train_parts = []
    evaluation_parts = []
    for contract in contracts:
        channel = contract["channel"]
        fit_split = "validation" if "split" in train_samples and bool((train_samples["split"] == "validation").any()) else "train"
        accessor = BaseAccessor(shared, view.head.dataset, fit_split, [channel])
        evaluation_split = str(evaluation_samples["split"].iloc[0]) if "split" in evaluation_samples else "train"
        evaluation_accessor = BaseAccessor(shared, view.head.dataset, evaluation_split, [channel])
        train_values, _ = profile_values(accessor, train_samples, channel, tuple(contract["selected_profile"]), int(contract["selected_m_tau"]))
        evaluation_values, _ = profile_values(evaluation_accessor, evaluation_samples, channel, tuple(contract["selected_profile"]), int(contract["selected_m_tau"]))
        combined = np.concatenate([train_values, evaluation_values], axis=0)
        prediction, _, _ = _fit_candidate(
            train_values,
            train_samples["y_true"].to_numpy(dtype=np.float64),
            combined,
            str(contract["selected_kind"]),
            int(contract["selected_m_x"]),
            tuple(float(value) for value in contract["selected_lambdas"]),
            solver,
        )
        train_parts.append(prediction[: len(train_samples)])
        evaluation_parts.append(prediction[len(train_samples) :])
    return np.column_stack(train_parts), np.column_stack(evaluation_parts)


def _oof_k_predictions(
    shared: Path,
    project: Path,
    view: ViewSpec,
    train: pd.DataFrame,
    contracts: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    freeze = _freeze(project)
    fit_cap = int(freeze["c4"]["fit_row_cap"])
    pieces = []
    audit = []
    for fold, (train_index, validation_index) in enumerate(rolling_oof_folds(train, int(freeze["c5"]["physics_first"]["oof_folds"]))):
        fold_train = train.iloc[train_index]
        fold_train = fold_train.iloc[deterministic_subsample(fold_train, fit_cap)]
        fold_validation = train.iloc[validation_index]
        train_shapes, validation_shapes = _fit_frozen_channel_shapes(shared, project, view, fold_train, fold_validation, contracts)
        if train_shapes.shape[1]:
            prediction, certificate = _ridge_block_predict(
                train_shapes,
                fold_train["y_true"].to_numpy(dtype=np.float64),
                validation_shapes,
                0.0001,
            )
        else:
            prediction = np.zeros(len(fold_validation), dtype=np.float64)
            certificate = {"solver": "exact_zero", "relative_kkt": 0.0, "condition_number": 1.0}
        frame = fold_validation[["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]].copy()
        frame["k_oof"] = prediction
        for index in range(validation_shapes.shape[1]):
            frame[f"k_component_{index:03d}"] = validation_shapes[:, index]
        frame["residual"] = frame["y_true"].to_numpy(dtype=np.float64) - prediction
        frame["oof_fold"] = fold
        pieces.append(frame)
        audit.append({"fold": fold, "fit_rows": len(fold_train), "validation_rows": len(fold_validation), "certificate": certificate})
    oof = pd.concat(pieces, ignore_index=True).sort_values(["entity_id", "origin"]).reset_index(drop=True)
    if oof["base_origin_id"].duplicated().any():
        raise AssertionError("OOF physical residuals are not unique")
    return oof, audit


def _residual_model(
    shared: Path,
    project: Path,
    view: ViewSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    oof: pd.DataFrame,
    validation_k: np.ndarray,
    inner_jobs: int = 1,
) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _freeze(project)
    config = freeze["c5"]["physics_first"]
    alpha_grid = [float(value) for value in freeze["c3"]["state_profile"]["penalty_scan_after_profile_freeze"]]
    profiles = realized_state_profiles(view.head)
    residual_mean = float(oof["residual"].mean())
    candidates: list[Any] = ["EXACT_ZERO", *[(profile, alpha) for profile in profiles for alpha in alpha_grid]]
    losses: dict[Any, list[float]] = {candidate: [] for candidate in candidates}
    usable_folds = sorted(int(value) for value in oof["oof_fold"].unique())[1:]
    fold_rows: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold in usable_folds:
        fit_rows = oof[oof["oof_fold"] < fold]
        evaluation_rows = oof[oof["oof_fold"] == fold]
        fold_rows[fold] = (fit_rows, evaluation_rows)
        losses["EXACT_ZERO"].append(float(np.mean(np.square(evaluation_rows["residual"].to_numpy(dtype=np.float64)), dtype=np.float64)))
    tasks = [(fold, profile, *fold_rows[fold]) for fold in usable_folds for profile in profiles]
    path_results: dict[tuple[int, tuple[int, int]], list[float]] = {}

    def execute(task: tuple[int, tuple[int, int], pd.DataFrame, pd.DataFrame]) -> tuple[int, tuple[int, int], list[float]]:
        fold, profile, fit_rows, evaluation_rows = task
        return _residual_profile_fold_path(
            fold,
            profile,
            fit_rows,
            evaluation_rows,
            oof,
            view,
            int(config["maximum_residual_lags"]),
            residual_mean,
            alpha_grid,
        )

    if inner_jobs <= 1:
        for fold, profile, path in map(execute, tasks):
            path_results[(fold, profile)] = path
    else:
        with ThreadPoolExecutor(max_workers=min(inner_jobs, len(tasks))) as executor:
            futures = [executor.submit(execute, task) for task in tasks]
            for future in as_completed(futures):
                fold, profile, path = future.result()
                path_results[(fold, profile)] = path

    # Completion order must not affect one-SE. Restore legacy fold/profile/alpha order.
    for fold in usable_folds:
        for profile in profiles:
            for alpha, value in zip(alpha_grid, path_results[(fold, profile)], strict=True):
                losses[(profile, alpha)].append(value)
    selection = select_one_se(losses, lambda value: (0,) if value == "EXACT_ZERO" else (1, value[0][1], -value[0][0], -value[1]))
    if selection.candidate == "EXACT_ZERO":
        residual_prediction = np.zeros(len(validation), dtype=np.float64)
        certificate = {"solver": "exact_zero", "relative_kkt": 0.0, "condition_number": 1.0}
        parameter_count = 0
        selected_profile = None
        selected_alpha = None
    else:
        selected_profile, selected_alpha = selection.candidate
        oof_x = mature_residual_features(oof, oof, view.head.h_steps, view.head.w_steps, *selected_profile, int(config["maximum_residual_lags"]), residual_mean)
        validation_residuals = validation[["entity_id", "origin"]].copy()
        validation_residuals["residual"] = validation["y_true"].to_numpy(dtype=np.float64) - validation_k
        combined_residuals = pd.concat([oof[["entity_id", "origin", "residual"]], validation_residuals], ignore_index=True)
        validation_x = mature_residual_features(validation, combined_residuals, view.head.h_steps, view.head.w_steps, *selected_profile, int(config["maximum_residual_lags"]), residual_mean)
        final_index = deterministic_subsample(oof, int(freeze["selection"]["fit_row_cap_default"]))
        residual_prediction, certificate = _ridge_block_predict(
            oof_x[final_index],
            oof.iloc[final_index]["residual"].to_numpy(dtype=np.float64),
            validation_x,
            float(selected_alpha),
        )
        parameter_count = oof_x.shape[1] + 1
    return residual_prediction, {
        "selected_candidate": selection.candidate,
        "selected_profile": None if selected_profile is None else list(selected_profile),
        "selected_alpha": selected_alpha,
        "one_se": selection.__dict__,
        "fold_losses": {str(key): value for key, value in losses.items()},
        "residual_mean": residual_mean,
        "usable_residual_selection_folds": len(usable_folds),
        "maturity_rule": "s+h+W<=t",
        "numerical_certificate": certificate,
        "runtime_parallelism": {
            "inner_jobs": int(inner_jobs),
            "path_tasks": len(tasks),
            "ridge_sufficient_statistics_reused": True,
        },
    }, parameter_count


def _joint_ar(
    shared: Path,
    project: Path,
    c3_output: Path,
    view: ViewSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    oof: pd.DataFrame,
    contracts: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _freeze(project)
    config = freeze["c5"]["k_joint_ar"]
    ar_contract = _load_ar_contract(c3_output, view)
    profile = tuple(ar_contract["selected_profile"])
    alpha = float(ar_contract["selected_alpha"])
    train_accessor = BaseAccessor(shared, view.head.dataset, "train", [view.head.target])
    oof_samples = train.set_index("base_origin_id").loc[oof["base_origin_id"]].reset_index()
    oof_ar = train_accessor.target_state(oof_samples, view.head.target, *profile)
    component_columns = sorted(column for column in oof if column.startswith("k_component_"))
    oof_k = oof[component_columns].to_numpy(dtype=np.float64)
    candidates: list[Any] = ["EXACT_BOTH_ZERO", "EXACT_K_ZERO", "EXACT_AR_ZERO", *[float(value) for value in config["block_penalty_ratios_k_over_ar"]]]
    losses = {candidate: [] for candidate in candidates}
    for fold in sorted(int(value) for value in oof["oof_fold"].unique())[1:]:
        fit = np.flatnonzero(oof["oof_fold"].to_numpy(dtype=np.int64) < fold)
        evaluate = np.flatnonzero(oof["oof_fold"].to_numpy(dtype=np.int64) == fold)
        y_fit = oof.iloc[fit]["y_true"].to_numpy(dtype=np.float64)
        y_evaluate = oof.iloc[evaluate]["y_true"].to_numpy(dtype=np.float64)
        losses["EXACT_BOTH_ZERO"].append(float(np.mean(np.square(y_evaluate), dtype=np.float64)))
        ar_prediction, _ = _ridge_block_predict(oof_ar[fit], y_fit, oof_ar[evaluate], alpha)
        losses["EXACT_K_ZERO"].append(mse(y_evaluate, ar_prediction))
        if oof_k.shape[1]:
            k_prediction, _ = _ridge_block_predict(oof_k[fit], y_fit, oof_k[evaluate], alpha)
        else:
            k_prediction = np.zeros(len(evaluate), dtype=np.float64)
        losses["EXACT_AR_ZERO"].append(mse(y_evaluate, k_prediction))
        joint = np.concatenate([oof_ar, oof_k], axis=1)
        for ratio in candidates[3:]:
            penalties = np.concatenate([np.full(oof_ar.shape[1], alpha), np.full(oof_k.shape[1], alpha * float(ratio))])
            prediction, _ = _ridge_block_predict(joint[fit], y_fit, joint[evaluate], penalties)
            losses[ratio].append(mse(y_evaluate, prediction))
    selection = select_one_se(losses, lambda value: ({"EXACT_BOTH_ZERO": 0, "EXACT_K_ZERO": 1, "EXACT_AR_ZERO": 1}.get(value, 2), 0 if isinstance(value, str) else -float(value)))

    fit_index = deterministic_subsample(train, int(freeze["selection"]["fit_row_cap_default"]))
    fit_samples = train.iloc[fit_index]
    train_shapes, validation_shapes = _fit_frozen_channel_shapes(shared, project, view, fit_samples, validation, contracts)
    fit_ar = train_accessor.target_state(fit_samples, view.head.target, *profile)
    validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", [view.head.target])
    validation_ar = validation_accessor.target_state(validation, view.head.target, *profile)
    y_fit = fit_samples["y_true"].to_numpy(dtype=np.float64)
    selected = selection.candidate
    if selected == "EXACT_BOTH_ZERO":
        prediction = np.zeros(len(validation), dtype=np.float64)
        certificate = {"solver": "exact_zero", "relative_kkt": 0.0, "condition_number": 1.0}
        parameter_count = 0
    elif selected == "EXACT_K_ZERO":
        prediction, certificate = _ridge_block_predict(fit_ar, y_fit, validation_ar, alpha)
        parameter_count = fit_ar.shape[1] + 1
    elif selected == "EXACT_AR_ZERO":
        if train_shapes.shape[1]:
            prediction, certificate = _ridge_block_predict(train_shapes, y_fit, validation_shapes, alpha)
            parameter_count = train_shapes.shape[1] + 1
        else:
            prediction = np.zeros(len(validation), dtype=np.float64)
            certificate = {"solver": "exact_zero", "relative_kkt": 0.0, "condition_number": 1.0}
            parameter_count = 0
    else:
        train_joint = np.concatenate([fit_ar, train_shapes], axis=1)
        validation_joint = np.concatenate([validation_ar, validation_shapes], axis=1)
        penalties = np.concatenate([np.full(fit_ar.shape[1], alpha), np.full(train_shapes.shape[1], alpha * float(selected))])
        prediction, certificate = _ridge_block_predict(train_joint, y_fit, validation_joint, penalties)
        parameter_count = train_joint.shape[1] + 1
    return prediction, {
        "selected_candidate": selected,
        "one_se": selection.__dict__,
        "fold_losses": {str(key): value for key, value in losses.items()},
        "state_profile": list(profile),
        "ar_alpha": alpha,
        "k_family": "c4_selected_true_urysohn_fixed_shape_subspace",
        "numerical_certificate": certificate,
    }, parameter_count


def _run_view(
    shared: Path,
    project: Path,
    c3_output: Path,
    c4_output: Path,
    output: Path,
    view: ViewSpec,
    inner_jobs: int,
) -> dict[str, Any]:
    started = time.time()
    destination = output / view.head.head_id / view.proxy_policy
    destination.mkdir(parents=True, exist_ok=True)
    try:
        c4 = _load_c4_joint(c4_output, view)
        contracts = list(c4["selected_channel_contracts"])
        train = load_samples(shared, view, "train")
        validation = load_samples(shared, view, "validation")
        oof, oof_audit = _oof_k_predictions(shared, project, view, train, contracts)
        oof_path = destination / "PHYSICS_K_OOF_RESIDUALS.parquet"
        oof.to_parquet(oof_path, index=False, compression="zstd")
        validation_k = pd.read_parquet(c4_output / "JOINT" / view.head.head_id / view.proxy_policy / "PRISM_CHANNEL_SPECIFIC.validation.parquet")["y_pred"].to_numpy(dtype=np.float64)
        residual_prediction, physics_selection, physics_parameters = _residual_model(
            shared, project, view, train, validation, oof, validation_k, inner_jobs
        )
        physics_prediction = validation_k + residual_prediction
        joint_prediction, joint_selection, joint_parameters = _joint_ar(shared, project, c3_output, view, train, validation, oof, contracts)
        rows = []
        files = []
        for model, prediction, parameters in (
            ("PRISM_PHYSICS_FIRST", physics_prediction, len(contracts) + physics_parameters),
            ("PRISM_K_JOINT_AR", joint_prediction, joint_parameters),
        ):
            frame = _prediction_frame(validation, view, model, prediction, parameters)
            path = destination / f"{model}.validation.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            files.append({"model": model, "path": str(path.relative_to(output)), "sha256": sha256_file(path)})
            rows.append({"model": model, "rows": len(frame), "parameter_count": parameters, **regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())})
        result = {
            "status": "PASS",
            "stage": "C5_PRISM_DYNAMIC",
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "selected_channels": [item["channel"] for item in contracts],
            "physics_first": physics_selection,
            "k_joint_ar": joint_selection,
            "oof_audit": oof_audit,
            "oof_path": str(oof_path.relative_to(output)),
            "oof_sha256": sha256_file(oof_path),
            "metrics": rows,
            "prediction_files": files,
            "test_accessed": False,
            "elapsed_seconds": time.time() - started,
        }
    except Exception as error:
        result = {
            "status": "FAILED_RETAINED",
            "stage": "C5_PRISM_DYNAMIC",
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "test_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    write_json(destination / "RESULT.json", result)
    return result


def run_c5(
    shared: Path,
    project: Path,
    c3_output: Path,
    c4_output: Path,
    output: Path,
    n_jobs: int,
    inner_jobs: int = 1,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    results = []
    pending = []
    for view in main_views(shared, "dynamic"):
        path = output / view.head.head_id / view.proxy_policy / "RESULT.json"
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("status") in {"PASS", "FAILED_RETAINED"}:
                results.append(previous)
                continue
        pending.append(view)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(_run_view, shared, project, c3_output, c4_output, output, view, inner_jobs): view
            for view in pending
        }
        for future in as_completed(futures):
            results.append(future.result())
    manifest = {
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "PASS_WITH_RETAINED_FAILURES",
        "jobs": len(results),
        "passed": sum(item["status"] == "PASS" for item in results),
        "failed_retained": sum(item["status"] != "PASS" for item in results),
        "outer_jobs": int(n_jobs),
        "residual_inner_jobs": int(inner_jobs),
        "test_accessed": False,
        "freeze_sha256": sha256_file(project / "configs/cpu_model_freeze_v1.json"),
    }
    write_json(output / "C5_MANIFEST.json", manifest)
    return manifest
