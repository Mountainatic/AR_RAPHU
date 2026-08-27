from __future__ import annotations

import copy
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    deterministic_subsample,
    input_columns,
    load_samples,
    main_views,
    realized_state_profiles,
    sha256_file,
)
from .cpu_selection import Standardizer, mse, regression_metrics, select_one_se
from .c2_models import _capped, _folds, _prediction_frame
from .stage0 import write_json
from .v2_runtime import release_process_memory


def _freeze(project: Path) -> dict[str, Any]:
    return json.loads((project / "configs/cpu_model_freeze_v1.json").read_text(encoding="utf-8"))


def _ridge_block_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    penalties: np.ndarray | float,
) -> tuple[np.ndarray, dict[str, float]]:
    scaler = Standardizer().fit(x_train)
    train = scaler.transform(x_train)
    evaluation = scaler.transform(x_eval)
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
    prediction = evaluation @ coefficient + y_mean
    return np.asarray(prediction, dtype=np.float64), {
        "solver": solver,
        "relative_kkt": relative_kkt,
        "condition_number": condition,
    }


def _select_ar(
    shared: Path,
    project: Path,
    view: ViewSpec,
) -> tuple[tuple[int, int], float, dict[str, Any]]:
    freeze = _freeze(project)
    config = freeze["c3"]["state_profile"]
    train = load_samples(shared, view, "train")
    accessor = BaseAccessor(shared, view.head.dataset, "train", [view.head.target])
    fit_cap = freeze["selection"]["fit_row_cap_default"]
    validation_cap = freeze["selection"]["selection_validation_row_cap_default"]
    profiles = realized_state_profiles(view.head)
    profile_losses = {profile: [] for profile in profiles}
    fold_indices = []
    for train_index, validation_index in _folds(train, view):
        train_index = _capped(train, train_index, fit_cap)
        validation_index = _capped(train, validation_index, validation_cap)
        fold_indices.append((train_index, validation_index))
        train_subset = train.iloc[train_index]
        validation_subset = train.iloc[validation_index]
        for delta, history in profiles:
            x_train = accessor.target_state(train_subset, view.head.target, delta, history)
            x_validation = accessor.target_state(validation_subset, view.head.target, delta, history)
            prediction, _ = _ridge_block_predict(
                x_train,
                train_subset["y_true"].to_numpy(dtype=np.float64),
                x_validation,
                float(config["pilot_alpha"]),
            )
            profile_losses[(delta, history)].append(
                mse(validation_subset["y_true"].to_numpy(dtype=np.float64), prediction)
            )
    selected_profile = select_one_se(profile_losses, lambda value: (value[1], -value[0]))
    delta, history = selected_profile.candidate
    penalty_losses = {float(alpha): [] for alpha in config["penalty_scan_after_profile_freeze"]}
    for train_index, validation_index in fold_indices:
        train_subset = train.iloc[train_index]
        validation_subset = train.iloc[validation_index]
        x_train = accessor.target_state(train_subset, view.head.target, delta, history)
        x_validation = accessor.target_state(validation_subset, view.head.target, delta, history)
        for alpha in penalty_losses:
            prediction, _ = _ridge_block_predict(
                x_train,
                train_subset["y_true"].to_numpy(dtype=np.float64),
                x_validation,
                alpha,
            )
            penalty_losses[alpha].append(
                mse(validation_subset["y_true"].to_numpy(dtype=np.float64), prediction)
            )
    selected_penalty = select_one_se(penalty_losses, lambda value: (-value,))
    audit = {
        "profile": selected_profile.__dict__,
        "penalty": selected_penalty.__dict__,
        "profile_fold_losses": {str(key): value for key, value in profile_losses.items()},
        "penalty_fold_losses": penalty_losses,
    }
    return (delta, history), float(selected_penalty.candidate), audit


def _run_ar(shared: Path, project: Path, view: ViewSpec) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _freeze(project)
    profile, alpha, audit = _select_ar(shared, project, view)
    train = load_samples(shared, view, "train")
    validation = load_samples(shared, view, "validation")
    final_index = deterministic_subsample(train, freeze["selection"]["fit_row_cap_default"])
    accessor = BaseAccessor(shared, view.head.dataset, "train", [view.head.target])
    validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", [view.head.target])
    x_train = accessor.target_state(train.iloc[final_index], view.head.target, *profile)
    x_validation = validation_accessor.target_state(validation, view.head.target, *profile)
    prediction, certificate = _ridge_block_predict(
        x_train,
        train.iloc[final_index]["y_true"].to_numpy(dtype=np.float64),
        x_validation,
        alpha,
    )
    audit.update({"selected_profile": list(profile), "selected_alpha": alpha, "numerical_certificate": certificate})
    return prediction, audit, x_train.shape[1] + 1


def _load_ar_contract(c3_output: Path, view: ViewSpec) -> dict[str, Any]:
    path = c3_output / "PREDICTIONS/AR" / view.relative_root / "RESULT.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError(f"AR prerequisite is not PASS: {path}")
    return result["selection"]


def _arx_features(
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    view: ViewSpec,
    columns: list[str],
    profile: tuple[int, int],
    maximum_input_lags: int,
) -> tuple[np.ndarray, int]:
    delta, history = profile
    ar_state = accessor.target_state(samples, view.head.target, delta, history)
    input_state = accessor.input_regular_lags(samples, columns, delta, history, maximum_input_lags)
    return np.concatenate([ar_state, input_state], axis=1), ar_state.shape[1]


def _run_arx(
    shared: Path,
    project: Path,
    c3_output: Path,
    view: ViewSpec,
) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _freeze(project)
    config = freeze["c3"]["arx"]
    ar = _load_ar_contract(c3_output, view)
    profile = tuple(ar["selected_profile"])
    alpha = float(ar["selected_alpha"])
    train = load_samples(shared, view, "train")
    validation = load_samples(shared, view, "validation")
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    all_columns = [view.head.target, *columns]
    accessor = BaseAccessor(shared, view.head.dataset, "train", all_columns)
    fit_cap = freeze["selection"]["fit_row_cap_default"]
    validation_cap = freeze["selection"]["selection_validation_row_cap_default"]
    candidates: list[Any] = ["EXACT_X_ZERO", *[float(value) for value in config["block_penalty_ratios_x_over_ar"]]]
    losses = {candidate: [] for candidate in candidates}
    for train_index, validation_index in _folds(train, view):
        train_index = _capped(train, train_index, fit_cap)
        validation_index = _capped(train, validation_index, validation_cap)
        train_subset = train.iloc[train_index]
        validation_subset = train.iloc[validation_index]
        x_train, ar_width = _arx_features(accessor, train_subset, view, columns, profile, config["maximum_input_lags_per_channel"])
        x_validation, _ = _arx_features(accessor, validation_subset, view, columns, profile, config["maximum_input_lags_per_channel"])
        y_train = train_subset["y_true"].to_numpy(dtype=np.float64)
        y_validation = validation_subset["y_true"].to_numpy(dtype=np.float64)
        zero_prediction, _ = _ridge_block_predict(x_train[:, :ar_width], y_train, x_validation[:, :ar_width], alpha)
        losses["EXACT_X_ZERO"].append(mse(y_validation, zero_prediction))
        for ratio in candidates[1:]:
            penalties = np.concatenate(
                [np.full(ar_width, alpha), np.full(x_train.shape[1] - ar_width, alpha * ratio)]
            )
            prediction, _ = _ridge_block_predict(x_train, y_train, x_validation, penalties)
            losses[ratio].append(mse(y_validation, prediction))
        del train_subset, validation_subset, x_train, x_validation, y_train, y_validation, zero_prediction
        release_process_memory()
    selected = select_one_se(losses, lambda value: (0,) if value == "EXACT_X_ZERO" else (1, -float(value)))
    final_index = deterministic_subsample(train, fit_cap)
    train_subset = train.iloc[final_index]
    validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", all_columns)
    x_train, ar_width = _arx_features(accessor, train_subset, view, columns, profile, config["maximum_input_lags_per_channel"])
    x_validation, _ = _arx_features(validation_accessor, validation, view, columns, profile, config["maximum_input_lags_per_channel"])
    y_train = train_subset["y_true"].to_numpy(dtype=np.float64)
    if selected.candidate == "EXACT_X_ZERO":
        prediction, certificate = _ridge_block_predict(x_train[:, :ar_width], y_train, x_validation[:, :ar_width], alpha)
        parameter_count = ar_width + 1
    else:
        ratio = float(selected.candidate)
        penalties = np.concatenate(
            [np.full(ar_width, alpha), np.full(x_train.shape[1] - ar_width, alpha * ratio)]
        )
        prediction, certificate = _ridge_block_predict(x_train, y_train, x_validation, penalties)
        parameter_count = x_train.shape[1] + 1
    return prediction, {
        "selected_profile": list(profile),
        "ar_alpha": alpha,
        "selected_x_penalty_ratio": selected.candidate,
        "fold_losses": {str(key): value for key, value in losses.items()},
        "one_se": selected.__dict__,
        "numerical_certificate": certificate,
    }, parameter_count


def _narx_expand(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = Standardizer().fit(x_train)
    standardized = scaler.transform(x_train)
    evaluation = scaler.transform(x_eval)
    centered_y = y_train - np.mean(y_train, dtype=np.float64)
    denominator = np.sqrt(np.sum(np.square(standardized), axis=0) * np.sum(np.square(centered_y)))
    correlations = np.abs((standardized.T @ centered_y) / np.where(denominator > 0, denominator, np.inf))
    order = np.lexsort((np.arange(len(correlations)), -correlations))[:maximum]
    selected_train = standardized[:, order]
    selected_eval = evaluation[:, order]
    return (
        np.concatenate([selected_train, np.square(selected_train)], axis=1),
        np.concatenate([selected_eval, np.square(selected_eval)], axis=1),
        order,
    )


def _run_narx(
    shared: Path,
    project: Path,
    c3_output: Path,
    view: ViewSpec,
) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _freeze(project)
    config = freeze["c3"]["linear_narx"]
    arx_config = freeze["c3"]["arx"]
    state_config = freeze["c3"]["state_profile"]
    ar = _load_ar_contract(c3_output, view)
    profile = tuple(ar["selected_profile"])
    train = load_samples(shared, view, "train")
    validation = load_samples(shared, view, "validation")
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    all_columns = [view.head.target, *columns]
    accessor = BaseAccessor(shared, view.head.dataset, "train", all_columns)
    fit_cap = freeze["selection"]["fit_row_cap_default"]
    validation_cap = freeze["selection"]["selection_validation_row_cap_default"]
    alphas = [float(value) for value in state_config["penalty_scan_after_profile_freeze"]]
    losses = {alpha: [] for alpha in alphas}
    for train_index, validation_index in _folds(train, view):
        train_index = _capped(train, train_index, fit_cap)
        validation_index = _capped(train, validation_index, validation_cap)
        train_subset = train.iloc[train_index]
        validation_subset = train.iloc[validation_index]
        raw_train, _ = _arx_features(accessor, train_subset, view, columns, profile, arx_config["maximum_input_lags_per_channel"])
        raw_validation, _ = _arx_features(accessor, validation_subset, view, columns, profile, arx_config["maximum_input_lags_per_channel"])
        x_train, x_validation, _ = _narx_expand(
            raw_train,
            train_subset["y_true"].to_numpy(dtype=np.float64),
            raw_validation,
            config["maximum_linear_state_features_before_expansion"],
        )
        for alpha in alphas:
            prediction, _ = _ridge_block_predict(x_train, train_subset["y_true"].to_numpy(dtype=np.float64), x_validation, alpha)
            losses[alpha].append(mse(validation_subset["y_true"].to_numpy(dtype=np.float64), prediction))
        del train_subset, validation_subset, raw_train, raw_validation, x_train, x_validation
        release_process_memory()
    selected = select_one_se(losses, lambda value: (-value,))
    final_index = deterministic_subsample(train, fit_cap)
    train_subset = train.iloc[final_index]
    validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", all_columns)
    raw_train, _ = _arx_features(accessor, train_subset, view, columns, profile, arx_config["maximum_input_lags_per_channel"])
    raw_validation, _ = _arx_features(validation_accessor, validation, view, columns, profile, arx_config["maximum_input_lags_per_channel"])
    x_train, x_validation, selected_features = _narx_expand(
        raw_train,
        train_subset["y_true"].to_numpy(dtype=np.float64),
        raw_validation,
        config["maximum_linear_state_features_before_expansion"],
    )
    prediction, certificate = _ridge_block_predict(
        x_train,
        train_subset["y_true"].to_numpy(dtype=np.float64),
        x_validation,
        float(selected.candidate),
    )
    return prediction, {
        "selected_profile": list(profile),
        "selected_alpha": selected.candidate,
        "selected_linear_feature_indices": selected_features.tolist(),
        "fold_losses": losses,
        "one_se": selected.__dict__,
        "numerical_certificate": certificate,
    }, x_train.shape[1] + 1


def _nonlinear_features(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = Standardizer().fit(x_train)
    train = scaler.transform(x_train)
    evaluation = scaler.transform(x_eval)
    if kind == "linear":
        return train, evaluation
    pieces_train = [train]
    pieces_eval = [evaluation]
    if kind in {"quadratic_centered", "cubic_centered"}:
        square_mean = np.mean(np.square(train), axis=0, dtype=np.float64)
        pieces_train.append(np.square(train) - square_mean)
        pieces_eval.append(np.square(evaluation) - square_mean)
    if kind == "cubic_centered":
        cube_mean = np.mean(np.power(train, 3), axis=0, dtype=np.float64)
        pieces_train.append(np.power(train, 3) - cube_mean)
        pieces_eval.append(np.power(evaluation, 3) - cube_mean)
    if kind == "piecewise_linear_5_quantile_knots":
        knots = np.quantile(train, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6], axis=0)
        for knot in knots:
            pieces_train.append(np.maximum(train - knot, 0.0))
            pieces_eval.append(np.maximum(evaluation - knot, 0.0))
    return np.concatenate(pieces_train, axis=1), np.concatenate(pieces_eval, axis=1)


def _hammerstein_profiles(
    view: ViewSpec,
    positive_h_history_multipliers: Sequence[int] | None = None,
    delta_steps_override: Sequence[int] | None = None,
    profile_cap: int = 6,
    require_every_registered_history: bool = False,
) -> list[tuple[int, int]]:
    profiles = sorted(
        realized_state_profiles(
            view.head,
            positive_h_history_multipliers=positive_h_history_multipliers,
            delta_steps_override=delta_steps_override,
        ),
        key=lambda value: (value[1], -value[0]),
    )
    cap = int(profile_cap)
    if cap <= 0:
        raise ValueError("Hammerstein profile cap must be positive")
    if len(profiles) <= cap:
        return profiles
    if not require_every_registered_history:
        indices = np.rint(np.linspace(0, len(profiles) - 1, cap)).astype(int)
        return [profiles[index] for index in indices]

    # Select one deterministic (largest-delta) profile for every registered
    # history, then fill remaining slots from the same registered grid.  The
    # extension contract is a strict history override: the legacy default
    # profiles must never be reintroduced implicitly.
    histories = list(dict.fromkeys(int(profile[1]) for profile in profiles))
    selected: list[tuple[int, int]] = []
    selected.extend(
        next(profile for profile in profiles if int(profile[1]) == history)
        for history in histories
    )
    if len(selected) > cap:
        raise ValueError(
            "Hammerstein profile cap cannot cover every registered history"
        )
    remaining = [profile for profile in profiles if profile not in selected]
    slots = cap - len(selected)
    if slots and remaining:
        indices = np.rint(
            np.linspace(0, len(remaining) - 1, min(slots, len(remaining)))
        ).astype(int)
        selected.extend(remaining[index] for index in dict.fromkeys(indices))
    return sorted(
        dict.fromkeys(selected), key=lambda value: (value[1], -value[0])
    )


def _run_hammerstein(
    shared: Path,
    project: Path,
    view: ViewSpec,
    wiener: bool,
) -> tuple[np.ndarray, dict[str, Any], int]:
    from sklearn.isotonic import IsotonicRegression

    freeze = _freeze(project)
    section = "hammerstein_wiener" if wiener else "parallel_hammerstein"
    config = freeze["c3"][section]
    train = load_samples(shared, view, "train")
    validation = load_samples(shared, view, "validation")
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    accessor = BaseAccessor(shared, view.head.dataset, "train", columns)
    fit_cap = freeze["selection"]["fit_row_cap_default"]
    validation_cap = freeze["selection"]["selection_validation_row_cap_default"]
    profiles = _hammerstein_profiles(view)
    if wiener:
        candidates = [
            (profile, nonlinearity, output_map)
            for profile in profiles
            for nonlinearity in config["input_nonlinearities"]
            for output_map in config["output_map"]
        ]
    else:
        candidates = [(profile, nonlinearity, "identity") for profile in profiles for nonlinearity in config["nonlinearities"]]
    losses = {candidate: [] for candidate in candidates}
    for train_index, validation_index in _folds(train, view):
        train_index = _capped(train, train_index, fit_cap)
        validation_index = _capped(train, validation_index, validation_cap)
        train_subset = train.iloc[train_index]
        validation_subset = train.iloc[validation_index]
        for profile in profiles:
            delta, history = profile
            raw_train = accessor.input_regular_lags(train_subset, columns, delta, history, 8)
            raw_validation = accessor.input_regular_lags(validation_subset, columns, delta, history, 8)
            nonlinearities = config["input_nonlinearities"] if wiener else config["nonlinearities"]
            output_maps = config["output_map"] if wiener else ["identity"]
            for nonlinearity in nonlinearities:
                x_train, x_validation = _nonlinear_features(raw_train, raw_validation, nonlinearity)
                prediction, _ = _ridge_block_predict(
                    x_train,
                    train_subset["y_true"].to_numpy(dtype=np.float64),
                    x_validation,
                    0.001,
                )
                train_prediction = None
                for output_map in output_maps:
                    current_prediction = prediction
                    if output_map == "isotonic_train_only":
                        if train_prediction is None:
                            train_prediction, _ = _ridge_block_predict(
                                x_train,
                                train_subset["y_true"].to_numpy(dtype=np.float64),
                                x_train,
                                0.001,
                            )
                        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                            train_prediction,
                            train_subset["y_true"].to_numpy(dtype=np.float64),
                        )
                        current_prediction = calibrator.predict(prediction)
                    losses[(profile, nonlinearity, output_map)].append(
                        mse(validation_subset["y_true"].to_numpy(dtype=np.float64), current_prediction)
                    )
                del x_train, x_validation, prediction, train_prediction
            del raw_train, raw_validation
            release_process_memory()
    selected = select_one_se(
        losses,
        lambda value: (value[0][1], -value[0][0], 0 if value[1] == "linear" else 1, 0 if value[2] == "identity" else 1),
    )
    profile, nonlinearity, output_map = selected.candidate
    final_index = deterministic_subsample(train, fit_cap)
    train_subset = train.iloc[final_index]
    validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", columns)
    raw_train = accessor.input_regular_lags(train_subset, columns, *profile, 8)
    raw_validation = validation_accessor.input_regular_lags(validation, columns, *profile, 8)
    x_train, x_validation = _nonlinear_features(raw_train, raw_validation, nonlinearity)
    prediction, certificate = _ridge_block_predict(
        x_train,
        train_subset["y_true"].to_numpy(dtype=np.float64),
        x_validation,
        0.001,
    )
    if output_map == "isotonic_train_only":
        train_prediction, _ = _ridge_block_predict(
            x_train,
            train_subset["y_true"].to_numpy(dtype=np.float64),
            x_train,
            0.001,
        )
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            train_prediction,
            train_subset["y_true"].to_numpy(dtype=np.float64),
        )
        prediction = calibrator.predict(prediction)
    return np.asarray(prediction, dtype=np.float64), {
        "selected_profile": list(profile),
        "selected_input_nonlinearity": nonlinearity,
        "selected_output_map": output_map,
        "fold_losses": {str(key): value for key, value in losses.items()},
        "one_se": selected.__dict__,
        "numerical_certificate": certificate,
    }, x_train.shape[1] + 1


def _fit_n4sid(frame: pd.DataFrame, target: str, columns: list[str], order: int) -> Any:
    from nfoursid.nfoursid import NFourSID

    identifier = NFourSID(
        frame[[*columns, target]],
        output_columns=[target],
        input_columns=columns,
        num_block_rows=2 * order,
    )
    identifier.subspace_identification()
    system, _ = identifier.system_identification(rank=order)
    radius = float(np.max(np.abs(np.linalg.eigvals(system.a)))) if system.a.size else 0.0
    if radius >= 0.999:
        raise ValueError(f"unstable N4SID radius={radius}")
    return system


def _n4sid_predict(
    system: Any,
    base: pd.DataFrame,
    samples: pd.DataFrame,
    target: str,
    columns: list[str],
    head: Any,
) -> np.ndarray:
    rows = base["row_in_entity"].to_numpy(dtype=np.int64)
    if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
        raise ValueError("N4SID sequence has gaps")
    inputs = base[columns].to_numpy(dtype=np.float64)
    outputs = base[target].to_numpy(dtype=np.float64)
    states: dict[int, np.ndarray] = {}
    x = np.zeros((system.a.shape[0], 1), dtype=np.float64)
    for position, row in enumerate(rows):
        states[int(row)] = x.copy()
        u = inputs[position][:, None]
        predicted = system.c @ x + system.d @ u
        innovation = np.asarray([[outputs[position] - predicted.item()]], dtype=np.float64)
        x = system.a @ x + system.b @ u + system.k @ innovation
    predictions = np.empty(len(samples), dtype=np.float64)
    lookup = {int(row): position for position, row in enumerate(rows)}
    for index, sample in enumerate(samples.itertuples(index=False)):
        origin = int(sample.origin)
        x = states[origin].copy()
        u = inputs[lookup[origin - 1]][:, None]
        future = []
        for step in range(head.h_steps + head.w_steps):
            y_hat = system.c @ x + system.d @ u
            if step >= head.h_steps:
                future.append(float(y_hat.item()))
            x = system.a @ x + system.b @ u
        current_positions = [lookup[row] for row in range(origin - head.w0_steps, origin)]
        current = float(np.mean(outputs[current_positions], dtype=np.float64))
        predictions[index] = float(np.mean(future, dtype=np.float64)) - current
    return predictions


def _run_n4sid(shared: Path, project: Path, view: ViewSpec) -> tuple[np.ndarray, dict[str, Any], int]:
    freeze = _freeze(project)
    config = freeze["c3"]["n4sid"]
    train_samples = load_samples(shared, view, "train")
    validation_samples = load_samples(shared, view, "validation")
    if train_samples["entity_id"].nunique() != 1:
        raise ValueError("N4SID explicit grouped-state reset not available; retained as not applicable")
    columns = input_columns(shared, view.head.task_id, view.proxy_policy)
    train_base = pd.read_parquet(
        shared / "base_data" / view.head.dataset / "train.parquet",
        columns=[*columns, view.head.target, "entity_id", "row_in_entity"],
    ).sort_values("row_in_entity")
    if not np.array_equal(
        train_base["row_in_entity"].to_numpy(dtype=np.int64),
        np.arange(int(train_base.row_in_entity.min()), int(train_base.row_in_entity.max()) + 1),
    ):
        raise ValueError("N4SID train sequence contains gaps")
    losses = {int(order): [] for order in config["state_orders"]}
    folds = _folds(train_samples, view)
    for train_index, validation_index in folds:
        fold_train_samples = train_samples.iloc[train_index]
        fold_validation_samples = train_samples.iloc[validation_index]
        stop = int(fold_train_samples["dependency_stop_exclusive"].max())
        fold_base = train_base[train_base["row_in_entity"] < stop]
        evaluation_stop = int(fold_validation_samples["dependency_stop_exclusive"].max())
        evaluation_base = train_base[train_base["row_in_entity"] < evaluation_stop]
        for order in losses:
            try:
                system = _fit_n4sid(fold_base, view.head.target, columns, order)
                prediction = _n4sid_predict(system, evaluation_base, fold_validation_samples, view.head.target, columns, view.head)
                losses[order].append(mse(fold_validation_samples["y_true"].to_numpy(dtype=np.float64), prediction))
            except Exception:
                losses[order].append(float("inf"))
    finite = {order: values for order, values in losses.items() if np.all(np.isfinite(values))}
    selected = select_one_se(finite, lambda value: (value,))
    system = _fit_n4sid(train_base, view.head.target, columns, int(selected.candidate))
    validation_base = pd.concat(
        [
            train_base,
            pd.read_parquet(
                shared / "base_data" / view.head.dataset / "validation.parquet",
                columns=[*columns, view.head.target, "entity_id", "row_in_entity"],
            ),
        ],
        ignore_index=True,
    ).sort_values("row_in_entity")
    prediction = _n4sid_predict(system, validation_base, validation_samples, view.head.target, columns, view.head)
    parameter_count = sum(matrix.size for matrix in (system.a, system.b, system.c, system.d, system.k))
    return prediction, {
        "selected_order": selected.candidate,
        "fold_losses": losses,
        "one_se": selected.__dict__,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(system.a)))),
        "future_input_policy": config["unknown_future_input_policy"],
    }, parameter_count


def _run_model(
    shared: Path,
    project: Path,
    c3_output: Path,
    view: ViewSpec,
    model: str,
) -> tuple[np.ndarray, dict[str, Any], int]:
    if model == "AR":
        return _run_ar(shared, project, view)
    if model == "ARX":
        return _run_arx(shared, project, c3_output, view)
    if model == "LINEAR_NARX":
        return _run_narx(shared, project, c3_output, view)
    if model == "N4SID":
        return _run_n4sid(shared, project, view)
    if model == "PARALLEL_HAMMERSTEIN":
        return _run_hammerstein(shared, project, view, wiener=False)
    if model == "HAMMERSTEIN_WIENER":
        return _run_hammerstein(shared, project, view, wiener=True)
    raise KeyError(model)


def run_job(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    model: str,
) -> dict[str, Any]:
    started = time.time()
    destination = output / "PREDICTIONS" / model / view.relative_root
    destination.mkdir(parents=True, exist_ok=True)
    try:
        prediction, selection, parameter_count = _run_model(shared, project, output, view, model)
        validation = load_samples(shared, view, "validation")
        frame = _prediction_frame(validation, view, model, prediction, parameter_count)
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        result = {
            "status": "PASS",
            "stage": "C3_SYSTEM_IDENTIFICATION_DEVELOPMENT",
            "model": model,
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "information_set": "dynamic",
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
            "stage": "C3_SYSTEM_IDENTIFICATION_DEVELOPMENT",
            "model": model,
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "information_set": "dynamic",
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "test_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    write_json(destination / "RESULT.json", result)
    return result


def _run_jobs(
    shared: Path,
    project: Path,
    output: Path,
    jobs: list[tuple[ViewSpec, str]],
    n_jobs: int,
) -> list[dict[str, Any]]:
    results = []
    pending = []
    for view, model in jobs:
        result_path = output / "PREDICTIONS" / model / view.relative_root / "RESULT.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") in {"PASS", "FAILED_RETAINED"}:
                results.append(result)
                continue
        pending.append((view, model))
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(run_job, shared, project, output, view, model): (view, model)
            for view, model in pending
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def run_c3(shared: Path, project: Path, output: Path, n_jobs: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    views = main_views(shared, "dynamic")
    results = _run_jobs(shared, project, output, [(view, "AR") for view in views], n_jobs)
    results.extend(
        _run_jobs(
            shared,
            project,
            output,
            [(view, model) for view in views for model in ("ARX", "LINEAR_NARX", "N4SID", "PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER")],
            n_jobs,
        )
    )
    flat = [{key: value for key, value in result.items() if key not in {"selection", "traceback"}} for result in results]
    pd.DataFrame(flat).sort_values(["task", "model"]).to_csv(output / "SYSTEM_IDENTIFICATION_DEVELOPMENT.csv", index=False)
    manifest = {
        "status": "PASS" if all(result["status"] == "PASS" for result in results) else "PASS_WITH_RETAINED_FAILURES",
        "jobs": len(results),
        "passed": sum(result["status"] == "PASS" for result in results),
        "failed_retained": sum(result["status"] != "PASS" for result in results),
        "test_accessed": False,
        "freeze_sha256": sha256_file(project / "configs/cpu_model_freeze_v1.json"),
    }
    write_json(output / "C3_MANIFEST.json", manifest)
    return manifest
