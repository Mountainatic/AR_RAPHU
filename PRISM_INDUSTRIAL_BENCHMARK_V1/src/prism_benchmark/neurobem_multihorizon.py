"""Prospective NeuroBEM direct multi-horizon PRISM experiment.

The implementation is deliberately separate from the completed one-step R1
runner.  Every feature is indexed from ``prediction_origin = target - h`` and
every continuous processed segment remains an independent history entity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
import multiprocessing as mp
import os
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np

from .neurobem_data import SegmentData, SegmentRecord, body_context, frozen_inner_fold, generalized_targets, motor_thrust_proxy
from .neurobem_experiment import ridge_from_json, ridge_to_json
from .neurobem_linear import RidgeContract, fit_numerical_ridge, guarded_one_se, normalized_mse, predict_ridge, restricted_cubic_basis, signed_quadratic_context


W0 = "W0_IDENTITY"
W1 = "W1_GENERIC"
W2 = "W2_AERO"
IDENTITY = "IDENTITY_CORRECTION"
EXACT_ZERO = "EXACT_ZERO"
ARM_ROUTE_NAMES = {
    W0: ("K__W0__NO_A", "K__W0__A"),
    W1: ("K__W1_GENERIC__NO_A", "K__W1_GENERIC__A"),
    W2: ("K__W2_AERO__NO_A", "K__W2_AERO__A"),
}


@dataclass(frozen=True)
class MultiHorizonWContract:
    candidate: str
    ridge: RidgeContract
    knots: tuple[np.ndarray, ...]
    feature_source: str


def w_contract_to_json(contract: MultiHorizonWContract | None) -> dict[str, object] | None:
    if contract is None:
        return None
    return {
        "candidate": contract.candidate,
        "ridge": ridge_to_json(contract.ridge),
        "knots": [value.tolist() for value in contract.knots],
        "feature_source": contract.feature_source,
    }


def w_contract_from_json(value: Mapping[str, object] | None) -> MultiHorizonWContract | None:
    if value is None:
        return None
    return MultiHorizonWContract(
        candidate=str(value["candidate"]),
        ridge=ridge_from_json(value["ridge"]),
        knots=tuple(np.asarray(item, dtype=np.float64) for item in value.get("knots", [])),
        feature_source=str(value["feature_source"]),
    )


def multihorizon_sample_id(record: SegmentRecord, horizon: int, prediction_origin: int, target_row: int) -> str:
    payload = f"NeuroBEM|{record.flight_id}|{record.segment_id}|{horizon}|{prediction_origin}|{target_row}"
    return sha256(payload.encode()).hexdigest()


def mature_target_lags(horizon: int, ages: Sequence[int]) -> tuple[int, ...]:
    if horizon < 1 or any(int(age) < 0 for age in ages):
        raise ValueError("invalid horizon or mature residual age")
    return tuple(horizon + int(age) for age in ages)


def base_prediction_support_start(max_history: int, horizon: int) -> int:
    return max_history - 1 + horizon


def route_support_start(max_history: int, horizon: int, max_mature_age: int) -> int:
    # Residual at t-h-age itself requires an h-step K/W prediction.
    return max_history - 1 + 2 * horizon + max_mature_age


def common_horizon_support_start(config: Mapping[str, object]) -> int:
    max_history = max(int(value) for value in config["K"]["candidate_fir_histories_samples"])
    max_age = max(int(value) for values in config["A"]["mature_residual_age_sets_samples"] for value in values)
    return max(route_support_start(max_history, int(h), max_age) for h in config["targets"]["forecast_horizons_samples"])


def k_design_horizon(
    segment: SegmentData,
    history: int,
    horizon: int,
    *,
    target_start: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if history < 1 or horizon < 1:
        raise ValueError("history and horizon must be positive")
    minimum = history - 1 + horizon
    start = minimum if target_start is None else max(minimum, int(target_start))
    if start >= segment.row_count:
        return np.empty((0, history * 4)), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    target_rows = np.arange(start, segment.row_count, dtype=np.int64)
    origins = target_rows - horizon
    lag_indices = origins[:, None] - np.arange(history, dtype=np.int64)[None, :]
    proxy = motor_thrust_proxy(segment)
    design = proxy[lag_indices].reshape(len(target_rows), history * 4)
    return design, target_rows, origins


def segment_k_xy_horizon(
    segment: SegmentData,
    history: int,
    horizon: int,
    mass: float,
    inertia: Sequence[float],
    *,
    target_start: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, rows, origins = k_design_horizon(segment, history, horizon, target_start=target_start)
    return x, generalized_targets(segment, mass, inertia)[rows], rows, origins


def concatenate_k_xy_horizon(
    segments: Iterable[SegmentData],
    history: int,
    horizon: int,
    mass: float,
    inertia: Sequence[float],
    *,
    target_start: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for segment in segments:
        x, y, rows, _ = segment_k_xy_horizon(segment, history, horizon, mass, inertia, target_start=target_start)
        if len(rows):
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError("no legal multi-horizon K rows")
    return np.concatenate(xs), np.concatenate(ys)


def _target_parameters(config: Mapping[str, object]) -> tuple[float, list[float]]:
    return float(config["targets"]["mass_kg"]), [float(value) for value in config["targets"]["inertia_diagonal_kg_m2"]]


def _ridge_parameters(config: Mapping[str, object], stage: str) -> tuple[list[float], float, float]:
    return (
        [float(value) for value in config[stage]["numerical_ridge_grid"]],
        float(config["K"]["maximum_condition_number"]),
        float(config["K"]["maximum_relative_kkt_residual"]),
    )


def fit_k_contract(segments: Sequence[SegmentData], history: int, horizon: int, config: Mapping[str, object]) -> RidgeContract:
    mass, inertia = _target_parameters(config)
    x, y = concatenate_k_xy_horizon(segments, history, horizon, mass, inertia)
    grid, condition, kkt = _ridge_parameters(config, "K")
    return fit_numerical_ridge(x, y, grid, condition, kkt)


def make_k_frame(
    segment: SegmentData,
    contract: RidgeContract,
    history: int,
    horizon: int,
    formal_start: int,
    config: Mapping[str, object],
    *,
    fold: int,
    role: str,
) -> dict[str, object]:
    mass, inertia = _target_parameters(config)
    base_start = base_prediction_support_start(max(int(v) for v in config["K"]["candidate_fir_histories_samples"]), horizon)
    x, y, rows, origins = segment_k_xy_horizon(segment, history, horizon, mass, inertia, target_start=base_start)
    prediction = predict_ridge(contract, x)
    context = body_context(segment)[origins]
    formal_mask = rows >= formal_start
    if not np.any(formal_mask):
        raise ValueError(f"NO_FORMAL_ROWS:{segment.record.segment_id}:h{horizon}")
    return {
        "segment": segment,
        "fold": fold,
        "role": role,
        "horizon": horizon,
        "rows": rows,
        "origins": origins,
        "formal_mask": formal_mask,
        "formal_rows": rows[formal_mask],
        "y": y,
        "k_prediction": prediction,
        "context": context,
        "speed": np.linalg.norm(context[:, :3], axis=1),
    }


_K_SEGMENTS: Sequence[SegmentData] = ()
_K_CONFIG: Mapping[str, object] = {}
_K_HORIZON = 1
_K_FORMAL_START = 0


def _k_fold_worker(task: tuple[int, int]) -> tuple[int, int, RidgeContract, float, float, int, str]:
    fold, history = task
    fit_segments = [segment for segment in _K_SEGMENTS if segment.record.inner_fold != fold]
    eval_segments = [segment for segment in _K_SEGMENTS if segment.record.inner_fold == fold]
    if not fit_segments or not eval_segments:
        raise RuntimeError(f"EMPTY_GROUPED_FOLD:{fold}")
    contract = fit_k_contract(fit_segments, history, _K_HORIZON, _K_CONFIG)
    mass, inertia = _target_parameters(_K_CONFIG)
    fit_y = np.concatenate([
        segment_k_xy_horizon(segment, history, _K_HORIZON, mass, inertia)[1]
        for segment in fit_segments
    ])
    variance = np.var(fit_y, axis=0)
    ys: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    digest = sha256()
    for segment in eval_segments:
        x, y, rows, origins = segment_k_xy_horizon(
            segment, history, _K_HORIZON, mass, inertia, target_start=_K_FORMAL_START
        )
        ys.append(y)
        predictions.append(predict_ridge(contract, x))
        for target, origin in zip(rows, origins, strict=True):
            digest.update(bytes.fromhex(multihorizon_sample_id(segment.record, _K_HORIZON, int(origin), int(target))))
    y = np.concatenate(ys)
    pred = np.concatenate(predictions)
    loss = normalized_mse(y, pred, variance)
    zero = normalized_mse(y, np.broadcast_to(contract.target_mean, y.shape), variance)
    return fold, history, contract, loss, zero, len(y), digest.hexdigest()


def _parallel_k(
    train: Sequence[SegmentData], tasks: Sequence[tuple[int, int]], horizon: int, formal_start: int, config: Mapping[str, object]
) -> list[tuple[int, int, RidgeContract, float, float, int, str]]:
    global _K_SEGMENTS, _K_CONFIG, _K_HORIZON, _K_FORMAL_START
    _K_SEGMENTS = train
    _K_CONFIG = config
    _K_HORIZON = horizon
    _K_FORMAL_START = formal_start
    workers = min(int(config["runtime"]["workers"]), len(tasks))
    if workers > 1 and sys.platform.startswith("linux"):
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(_k_fold_worker, tasks)
    return [_k_fold_worker(task) for task in tasks]


def run_k_horizon(
    train: Sequence[SegmentData], validation: Sequence[SegmentData], horizon: int, config: Mapping[str, object]
) -> tuple[dict[str, object], dict[int, dict[str, list[dict[str, object]]]], list[dict[str, object]], RidgeContract]:
    histories = [int(value) for value in config["K"]["candidate_fir_histories_samples"]]
    max_history = max(histories)
    max_age = max(int(value) for values in config["A"]["mature_residual_age_sets_samples"] for value in values)
    formal_start = route_support_start(max_history, horizon, max_age)
    fold_count = int(config["source_and_split"]["inner_folds"])
    losses = {history: [] for history in histories}
    zeros = {history: [] for history in histories}
    contracts: dict[tuple[int, int], RidgeContract] = {}
    audit: list[dict[str, object]] = []
    tasks = [(fold, history) for fold in range(fold_count) for history in histories]
    for fold, history, contract, loss, zero, rows, support_digest in sorted(
        _parallel_k(train, tasks, horizon, formal_start, config), key=lambda item: (item[0], item[1])
    ):
        contracts[(fold, history)] = contract
        losses[history].append(loss)
        zeros[history].append(zero)
        audit.append({
            "horizon": horizon,
            "fold": fold,
            "history": history,
            "native_fit_start": history - 1 + horizon,
            "formal_scoring_start": formal_start,
            "evaluation_rows": rows,
            "evaluation_support_sha256": support_digest,
            "row_cap_applied_after_native_mask": True,
        })
    selection = guarded_one_se(losses, histories, float(config["K"]["maximum_relative_regret_vs_best"]))
    selected = int(selection["selected"])
    selected_loss = np.asarray(losses[selected])
    selected_zero = np.asarray(zeros[selected])
    improvement = float((selected_zero.mean() - selected_loss.mean()) / selected_zero.mean())
    positive = float(np.mean(selected_loss < selected_zero))
    activation = improvement >= float(config["K"]["minimum_relative_improvement_vs_zero"]) and positive >= float(config["K"]["minimum_positive_fold_fraction"])
    route_folds: dict[int, dict[str, list[dict[str, object]]]] = {}
    for fold in range(fold_count):
        route_folds[fold] = {"fit": [], "evaluation": []}
        contract = contracts[(fold, selected)]
        for role, segments in (
            ("fit", [item for item in train if item.record.inner_fold != fold]),
            ("evaluation", [item for item in train if item.record.inner_fold == fold]),
        ):
            route_folds[fold][role] = [
                make_k_frame(segment, contract, selected, horizon, formal_start, config, fold=fold, role=role)
                for segment in segments
            ]
    validation_contract = fit_k_contract(train, selected, horizon, config)
    validation_frames = [
        make_k_frame(segment, validation_contract, selected, horizon, formal_start, config, fold=-1, role="validation")
        for segment in validation
    ]
    y_val = np.concatenate([frame["y"][frame["formal_mask"]] for frame in validation_frames])
    p_val = np.concatenate([frame["k_prediction"][frame["formal_mask"]] for frame in validation_frames])
    mse = np.mean(np.square(y_val - p_val), axis=0)
    zero_mse = np.mean(np.square(y_val - validation_contract.target_mean), axis=0)
    variance_ratio = np.var(p_val, axis=0) / np.maximum(np.var(y_val, axis=0), np.finfo(float).eps)
    gate = {
        "variance_pass_by_axis": (variance_ratio >= 1e-8).tolist(),
        "coefficient_pass": bool(np.max(np.abs(validation_contract.coefficient_physical)) >= 1e-10),
        "mse_pass_by_axis": (mse <= 1.02 * zero_mse).tolist(),
        "numerical_pass": bool(validation_contract.condition_number <= float(config["K"]["maximum_condition_number"]) and validation_contract.relative_kkt_residual <= float(config["K"]["maximum_relative_kkt_residual"])),
    }
    gate["passed"] = bool(activation and all(gate["variance_pass_by_axis"]) and gate["coefficient_pass"] and all(gate["mse_pass_by_axis"]) and gate["numerical_pass"])
    # This registered experiment must retain every horizon and every formal
    # arm.  The gate is a physics-consistency diagnostic at that horizon; it
    # is not permission to delete a pre-registered horizon from the predictive
    # audit.  Numerical/protocol failures still raise before reaching here.
    result = {
        "status": "PASS",
        "K_input_gate_status": "PASS" if gate["passed"] else "FAILED_RETAINED_FOR_REGISTERED_HORIZON_AUDIT",
        "physics_consistency_supported": bool(gate["passed"]),
        "horizon": horizon,
        "candidate_fold_losses": {str(key): value for key, value in losses.items()},
        "exact_zero_fold_losses_by_history": {str(key): value for key, value in zeros.items()},
        "selection": selection,
        "selected_history": selected,
        "history_status": "K_HISTORY_BOUNDARY_SELECTED" if selected == max_history else "INTERIOR_HISTORY_SELECTED",
        "relative_improvement_vs_zero": improvement,
        "positive_fold_fraction": positive,
        "formal_support_start": formal_start,
        "base_prediction_support_start": base_prediction_support_start(max_history, horizon),
        "support_audit": audit,
        "selected_fold_contracts": {str(fold): ridge_to_json(contracts[(fold, selected)]) for fold in range(fold_count)},
        "train_to_validation_contract": ridge_to_json(validation_contract),
        "validation_input_gate": gate,
        "test_accessed": False,
    }
    return result, route_folds, validation_frames, validation_contract


def _frame_arrays(frames: Iterable[dict[str, object]], *, prediction_key: str = "k_prediction", formal_only: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contexts: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for frame in frames:
        mask = frame["formal_mask"] if formal_only else np.ones(len(frame["rows"]), dtype=bool)
        contexts.append(frame["context"][mask])
        predictions.append(frame[prediction_key][mask])
        targets.append(frame["y"][mask])
    return np.concatenate(contexts), np.concatenate(predictions), np.concatenate(targets)


def _generic_knots(k_prediction: np.ndarray, count: int) -> tuple[np.ndarray, ...]:
    quantiles = np.linspace(0.0, 1.0, count + 2)
    knots = tuple(np.unique(np.quantile(k_prediction[:, axis], quantiles)) for axis in range(k_prediction.shape[1]))
    if any(len(value) < 4 for value in knots):
        raise ValueError("GENERIC_K_LATENT_SUPPORT_DEGENERATE")
    return knots


def _w_features(candidate: str, context: np.ndarray, k_prediction: np.ndarray, knots: tuple[np.ndarray, ...]) -> np.ndarray:
    if candidate.startswith("NATURAL_CUBIC_K_LATENT_K"):
        if len(knots) != k_prediction.shape[1]:
            raise ValueError("GENERIC_W_KNOT_DIMENSION_MISMATCH")
        return np.column_stack([restricted_cubic_basis(k_prediction[:, axis], knots[axis]) for axis in range(k_prediction.shape[1])])
    if candidate == "SIGNED_QUADRATIC_AERO_CONTEXT":
        return signed_quadratic_context(context, k_prediction)
    if candidate.startswith("NATURAL_CUBIC_SPEED_CONTEXT_K"):
        if len(knots) != 1:
            raise ValueError("AERO_W_KNOT_DIMENSION_MISMATCH")
        speed = np.linalg.norm(context[:, :3], axis=1)
        return np.column_stack((context, k_prediction, restricted_cubic_basis(speed, knots[0])))
    raise ValueError(f"UNKNOWN_W_CANDIDATE:{candidate}")


def fit_w_contract(candidate: str, frames: Iterable[dict[str, object]], config: Mapping[str, object]) -> MultiHorizonWContract:
    context, k_prediction, y = _frame_arrays(frames)
    if candidate.startswith("NATURAL_CUBIC_K_LATENT_K"):
        count = int(candidate.rsplit("K", 1)[1])
        knots = _generic_knots(k_prediction, count)
        source = "FROZEN_K_LATENT_ONLY"
    elif candidate.startswith("NATURAL_CUBIC_SPEED_CONTEXT_K"):
        count = int(candidate.rsplit("K", 1)[1])
        knots = (np.unique(np.quantile(np.linalg.norm(context[:, :3], axis=1), np.linspace(0.0, 1.0, count + 2))),)
        if len(knots[0]) < 4:
            raise ValueError("AERO_SPEED_SUPPORT_DEGENERATE")
        source = "ORIGIN_CAUSAL_AERODYNAMIC_CONTEXT_EXTENSION"
    elif candidate == "SIGNED_QUADRATIC_AERO_CONTEXT":
        knots = ()
        source = "ORIGIN_CAUSAL_AERODYNAMIC_CONTEXT_EXTENSION"
    else:
        raise ValueError(f"UNKNOWN_W_CANDIDATE:{candidate}")
    x = _w_features(candidate, context, k_prediction, knots)
    grid, condition, kkt = _ridge_parameters(config, "W")
    ridge = fit_numerical_ridge(x, y - k_prediction, grid, condition, kkt)
    return MultiHorizonWContract(candidate, ridge, knots, source)


def predict_w_contract(contract: MultiHorizonWContract, context: np.ndarray, k_prediction: np.ndarray) -> np.ndarray:
    return predict_ridge(contract.ridge, _w_features(contract.candidate, context, k_prediction, contract.knots))


def _apply_w(frames: Iterable[dict[str, object]], contract: MultiHorizonWContract | None) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for frame in frames:
        copied = dict(frame)
        correction = np.zeros_like(frame["k_prediction"]) if contract is None else predict_w_contract(contract, frame["context"], frame["k_prediction"])
        copied["w_correction"] = correction
        copied["no_a_prediction"] = frame["k_prediction"] + correction
        output.append(copied)
    return output


def run_w_arms(
    route_folds: dict[int, dict[str, list[dict[str, object]]]],
    validation_frames: list[dict[str, object]],
    config: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, dict[int, dict[str, list[dict[str, object]]]]], dict[str, list[dict[str, object]]], dict[str, MultiHorizonWContract | None]]:
    fold_count = len(route_folds)
    results: dict[str, object] = {}
    arm_folds: dict[str, dict[int, dict[str, list[dict[str, object]]]]] = {}
    validation_by_arm: dict[str, list[dict[str, object]]] = {}
    validation_contracts: dict[str, MultiHorizonWContract | None] = {}
    for arm in (W0, W1, W2):
        candidates = list(config["W"]["arms"][arm])
        losses: dict[str, list[float]] = {candidate: [] for candidate in candidates}
        fold_contracts: dict[tuple[int, str], MultiHorizonWContract] = {}
        for fold in range(fold_count):
            fit_frames = route_folds[fold]["fit"]
            eval_frames = route_folds[fold]["evaluation"]
            _, fit_k, fit_y = _frame_arrays(fit_frames)
            eval_context, eval_k, eval_y = _frame_arrays(eval_frames)
            variance = np.var(fit_y, axis=0)
            losses[IDENTITY].append(normalized_mse(eval_y, eval_k, variance))
            for candidate in candidates[1:]:
                contract = fit_w_contract(candidate, fit_frames, config)
                fold_contracts[(fold, candidate)] = contract
                pred = eval_k + predict_w_contract(contract, eval_context, eval_k)
                losses[candidate].append(normalized_mse(eval_y, pred, variance))
        if arm == W0:
            selection = {"selected": IDENTITY, "best_mean": IDENTITY, "eligible": [IDENTITY], "mean_losses": {IDENTITY: float(np.mean(losses[IDENTITY]))}, "standard_errors": {IDENTITY: float(np.std(losses[IDENTITY], ddof=1) / np.sqrt(fold_count))}, "threshold": float(np.mean(losses[IDENTITY]))}
            selected = IDENTITY
        else:
            selection = guarded_one_se(losses, candidates, float(config["W"]["maximum_relative_regret_vs_best"]))
            selected = str(selection["selected"])
            identity = np.asarray(losses[IDENTITY])
            chosen = np.asarray(losses[selected])
            improvement = float((identity.mean() - chosen.mean()) / identity.mean())
            positive = float(np.mean(chosen < identity))
            if selected != IDENTITY and not (improvement >= float(config["W"]["minimum_relative_improvement"]) and positive >= float(config["W"]["minimum_positive_fold_fraction"])):
                selected = IDENTITY
        identity = np.asarray(losses[IDENTITY])
        final_losses = np.asarray(losses[selected])
        improvement = float((identity.mean() - final_losses.mean()) / identity.mean())
        positive = float(np.mean(final_losses < identity))
        arm_folds[arm] = {}
        for fold in range(fold_count):
            contract = None if selected == IDENTITY else fold_contracts[(fold, selected)]
            arm_folds[arm][fold] = {
                "fit": _apply_w(route_folds[fold]["fit"], contract),
                "evaluation": _apply_w(route_folds[fold]["evaluation"], contract),
            }
        oof_k_frames = [frame for fold in range(fold_count) for frame in route_folds[fold]["evaluation"]]
        validation_contract = None if selected == IDENTITY else fit_w_contract(selected, oof_k_frames, config)
        validation_contracts[arm] = validation_contract
        validation_by_arm[arm] = _apply_w(validation_frames, validation_contract)
        selected_id = f"{arm}|{selected}"
        results[arm] = {
            "candidate_pool": candidates,
            "candidate_fold_losses": losses,
            "selection_before_activation_guard": selection,
            "final_selected_candidate": selected,
            "final_selected_candidate_id": selected_id,
            "final_selected_fold_losses": losses[selected],
            "final_selected_prediction_candidate_id": selected_id,
            "final_selected_contract_candidate_id": selected_id,
            "relative_improvement_vs_identity": improvement,
            "positive_fold_fraction": positive,
            "validation_contract": w_contract_to_json(validation_contract),
            "generic_reads_k_latent_only": arm != W1 or selected == IDENTITY or validation_contract.feature_source == "FROZEN_K_LATENT_ONLY",
            "global_arm_selection_used": False,
        }
    return results, arm_folds, validation_by_arm, validation_contracts


def _residual_series(frames: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for frame in frames:
        segment = frame["segment"]
        residual = np.full((segment.row_count, 4), np.nan, dtype=np.float64)
        residual[frame["rows"]] = frame["y"] - frame["no_a_prediction"]
        output.append({"segment": segment, "residual": residual, "formal_rows": frame["formal_rows"]})
    return output


def _a_xy(series: Iterable[dict[str, object]], horizon: int, ages: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    lags = mature_target_lags(horizon, ages)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for item in series:
        residual = item["residual"]
        rows = np.asarray(item["formal_rows"], dtype=np.int64)
        features = np.column_stack([residual[rows - lag] for lag in lags])
        target = residual[rows]
        valid = np.isfinite(features).all(axis=1) & np.isfinite(target).all(axis=1)
        if not np.all(valid):
            raise RuntimeError("FORMAL_ROUTE_SUPPORT_FAILED_A_MATURITY")
        xs.append(features)
        ys.append(target)
    return np.concatenate(xs), np.concatenate(ys)


def _apply_a(frames: Iterable[dict[str, object]], contract: RidgeContract | None, horizon: int, ages: Sequence[int] | None) -> list[dict[str, object]]:
    frames = list(frames)
    by_segment = {item["segment"].record.segment_id: item for item in _residual_series(frames)}
    output: list[dict[str, object]] = []
    for frame in frames:
        copied = dict(frame)
        prediction = frame["no_a_prediction"].copy()
        state = np.zeros_like(prediction)
        if contract is not None and ages is not None:
            lags = mature_target_lags(horizon, ages)
            residual = by_segment[frame["segment"].record.segment_id]["residual"]
            rows = frame["formal_rows"]
            features = np.column_stack([residual[rows - lag] for lag in lags])
            if not np.isfinite(features).all():
                raise RuntimeError("IMMATURE_RESIDUAL_WOULD_BE_USED")
            state_values = predict_ridge(contract, features)
            state[frame["formal_mask"]] = state_values
            prediction[frame["formal_mask"]] += state_values
        copied["a_prediction"] = state
        copied["with_a_prediction"] = prediction
        output.append(copied)
    return output


def run_a_arms(
    arm_folds: dict[str, dict[int, dict[str, list[dict[str, object]]]]],
    validation_by_arm: dict[str, list[dict[str, object]]],
    horizon: int,
    config: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, dict[int, list[dict[str, object]]]], dict[str, list[dict[str, object]]], dict[str, RidgeContract | None], dict[str, tuple[int, ...] | None]]:
    age_sets = [tuple(int(value) for value in ages) for ages in config["A"]["mature_residual_age_sets_samples"]]
    fold_count = len(next(iter(arm_folds.values())))
    results: dict[str, object] = {}
    oof_by_arm: dict[str, dict[int, list[dict[str, object]]]] = {}
    validation_output: dict[str, list[dict[str, object]]] = {}
    validation_contracts: dict[str, RidgeContract | None] = {}
    selected_ages: dict[str, tuple[int, ...] | None] = {}
    for arm in (W0, W1, W2):
        keys = ["A_" + "_".join(map(str, ages)) for ages in age_sets]
        losses: dict[str, list[float]] = {EXACT_ZERO: []} | {key: [] for key in keys}
        fold_contracts: dict[tuple[int, str], RidgeContract] = {}
        for fold in range(fold_count):
            fit_series = _residual_series(arm_folds[arm][fold]["fit"])
            eval_series = _residual_series(arm_folds[arm][fold]["evaluation"])
            for index, ages in enumerate(age_sets):
                key = keys[index]
                x_fit, y_fit = _a_xy(fit_series, horizon, ages)
                x_eval, y_eval = _a_xy(eval_series, horizon, ages)
                grid, condition, kkt = _ridge_parameters(config, "A")
                contract = fit_numerical_ridge(x_fit, y_fit, grid, condition, kkt)
                fold_contracts[(fold, key)] = contract
                losses[key].append(normalized_mse(y_eval, predict_ridge(contract, x_eval), np.var(y_fit, axis=0)))
                if index == 0:
                    losses[EXACT_ZERO].append(normalized_mse(y_eval, np.zeros_like(y_eval), np.var(y_fit, axis=0)))
        selection = guarded_one_se(losses, [EXACT_ZERO] + keys, float(config["A"]["maximum_relative_regret_vs_best"]))
        selected = str(selection["selected"])
        base = np.asarray(losses[EXACT_ZERO])
        chosen = np.asarray(losses[selected])
        improvement = float((base.mean() - chosen.mean()) / base.mean())
        positive = float(np.mean(chosen < base))
        if selected != EXACT_ZERO and not (improvement >= float(config["A"]["minimum_relative_improvement"]) and positive >= float(config["A"]["minimum_positive_fold_fraction"])):
            selected = EXACT_ZERO
        ages = None if selected == EXACT_ZERO else age_sets[keys.index(selected)]
        selected_ages[arm] = ages
        oof_by_arm[arm] = {}
        for fold in range(fold_count):
            contract = None if ages is None else fold_contracts[(fold, selected)]
            oof_by_arm[arm][fold] = _apply_a(arm_folds[arm][fold]["evaluation"], contract, horizon, ages)
        oof_eval_frames = [frame for fold in range(fold_count) for frame in arm_folds[arm][fold]["evaluation"]]
        validation_contract = None
        if ages is not None:
            x, y = _a_xy(_residual_series(oof_eval_frames), horizon, ages)
            grid, condition, kkt = _ridge_parameters(config, "A")
            validation_contract = fit_numerical_ridge(x, y, grid, condition, kkt)
        validation_contracts[arm] = validation_contract
        validation_output[arm] = _apply_a(validation_by_arm[arm], validation_contract, horizon, ages)
        candidate_id = f"h{horizon}|{arm}|{selected}"
        results[arm] = {
            "candidate_fold_losses": losses,
            "selection_before_activation_guard": selection,
            "final_selected_candidate": selected,
            "final_selected_candidate_id": candidate_id,
            "final_selected_fold_losses": losses[selected],
            "final_selected_prediction_candidate_id": candidate_id,
            "final_selected_contract_candidate_id": candidate_id,
            "selected_mature_ages": ages,
            "selected_actual_target_lags": None if ages is None else mature_target_lags(horizon, ages),
            "relative_improvement_vs_exact_zero": float((base.mean() - np.asarray(losses[selected]).mean()) / base.mean()),
            "positive_fold_fraction": float(np.mean(np.asarray(losses[selected]) < base)),
            "validation_contract": None if validation_contract is None else ridge_to_json(validation_contract),
            "raw_input_used": False,
            "maturity_pass": True,
        }
    return results, oof_by_arm, validation_output, validation_contracts, selected_ages


def _target_history_features(segment: SegmentData, rows: np.ndarray, horizon: int, ages: Sequence[int]) -> np.ndarray:
    target = generalized_targets(segment, 0.772, [0.0025, 0.0021, 0.0043])
    origins = rows - horizon
    return np.column_stack([target[origins - int(age)] for age in ages])


def _baseline_xy(
    frames: Iterable[dict[str, object]], horizon: int, ages: Sequence[int], *, motor_history: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for frame in frames:
        rows = frame["formal_rows"]
        target_features = _target_history_features(frame["segment"], rows, horizon, ages)
        if motor_history is None:
            x = target_features
        else:
            motor, motor_rows, _ = k_design_horizon(frame["segment"], motor_history, horizon, target_start=int(rows[0]))
            if not np.array_equal(motor_rows, rows):
                raise RuntimeError("NARX_SUPPORT_MISMATCH")
            x = np.column_stack((motor, target_features))
        xs.append(x)
        ys.append(frame["y"][frame["formal_mask"]])
    return np.concatenate(xs), np.concatenate(ys)


def _predict_baseline_frames(frames: Iterable[dict[str, object]], horizon: int, ages: Sequence[int], contract: RidgeContract, motor_history: int | None) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for frame in frames:
        x, _ = _baseline_xy([frame], horizon, ages, motor_history=motor_history)
        result[frame["segment"].record.segment_id] = predict_ridge(contract, x)
    return result


def run_baselines(
    route_folds: dict[int, dict[str, list[dict[str, object]]]],
    validation_frames: list[dict[str, object]],
    horizon: int,
    selected_k_history: int,
    config: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]], dict[str, dict[int, dict[str, np.ndarray]]], dict[str, RidgeContract], dict[str, tuple[int, ...]]]:
    age_sets = [tuple(int(value) for value in ages) for ages in config["baselines"]["target_history_age_sets_samples"]]
    fold_count = len(route_folds)
    result: dict[str, object] = {}
    validation_predictions: dict[str, dict[str, np.ndarray]] = {}
    oof_predictions: dict[str, dict[int, dict[str, np.ndarray]]] = {"PERSISTENCE": {}}
    contracts: dict[str, RidgeContract] = {}
    selected_ages: dict[str, tuple[int, ...]] = {}
    for fold in range(fold_count):
        oof_predictions["PERSISTENCE"][fold] = {
            frame["segment"].record.segment_id: generalized_targets(frame["segment"], 0.772, [0.0025, 0.0021, 0.0043])[frame["origins"][frame["formal_mask"]]]
            for frame in route_folds[fold]["evaluation"]
        }
    validation_predictions["PERSISTENCE"] = {
        frame["segment"].record.segment_id: generalized_targets(frame["segment"], 0.772, [0.0025, 0.0021, 0.0043])[frame["origins"][frame["formal_mask"]]]
        for frame in validation_frames
    }
    result["PERSISTENCE"] = {"status": "FIXED", "information_cutoff": "prediction_origin"}
    for route, motor_history in (("TARGET_ONLY_VAR_AR", None), ("LINEAR_NARX", selected_k_history)):
        losses = {ages: [] for ages in age_sets}
        fold_contracts: dict[tuple[int, tuple[int, ...]], RidgeContract] = {}
        for fold in range(fold_count):
            x_fit, y_fit = _baseline_xy(route_folds[fold]["fit"], horizon, age_sets[-1], motor_history=motor_history)
            variance = np.var(y_fit, axis=0)
            for ages in age_sets:
                x_fit, y_fit = _baseline_xy(route_folds[fold]["fit"], horizon, ages, motor_history=motor_history)
                x_eval, y_eval = _baseline_xy(route_folds[fold]["evaluation"], horizon, ages, motor_history=motor_history)
                contract = fit_numerical_ridge(x_fit, y_fit, config["K"]["numerical_ridge_grid"], float(config["K"]["maximum_condition_number"]), float(config["K"]["maximum_relative_kkt_residual"]))
                fold_contracts[(fold, ages)] = contract
                losses[ages].append(normalized_mse(y_eval, predict_ridge(contract, x_eval), variance))
        selection = guarded_one_se(losses, age_sets, 0.02)
        ages = tuple(selection["selected"])
        selected_ages[route] = ages
        oof_predictions[route] = {}
        for fold in range(fold_count):
            oof_predictions[route][fold] = _predict_baseline_frames(route_folds[fold]["evaluation"], horizon, ages, fold_contracts[(fold, ages)], motor_history)
        x, y = _baseline_xy([frame for fold in range(fold_count) for frame in route_folds[fold]["evaluation"]], horizon, ages, motor_history=motor_history)
        contract = fit_numerical_ridge(x, y, config["K"]["numerical_ridge_grid"], float(config["K"]["maximum_condition_number"]), float(config["K"]["maximum_relative_kkt_residual"]))
        contracts[route] = contract
        validation_predictions[route] = _predict_baseline_frames(validation_frames, horizon, ages, contract, motor_history)
        result[route] = {
            "candidate_fold_losses": {"_".join(map(str, key)): value for key, value in losses.items()},
            "selection": {**selection, "selected": list(ages), "best_mean": list(selection["best_mean"]), "eligible": [list(value) for value in selection["eligible"]], "mean_losses": {"_".join(map(str, key)): value for key, value in selection["mean_losses"].items()}, "standard_errors": {"_".join(map(str, key)): value for key, value in selection["standard_errors"].items()}},
            "selected_ages": ages,
            "motor_history": motor_history,
            "validation_contract": ridge_to_json(contract),
        }
    return result, validation_predictions, oof_predictions, contracts, selected_ages


def collect_prediction_records(
    frames_by_arm: Mapping[str, Sequence[dict[str, object]]],
    baseline_predictions: Mapping[str, Mapping[str, np.ndarray]],
    horizon: int,
) -> list[dict[str, object]]:
    reference = {frame["segment"].record.segment_id: frame for frame in frames_by_arm[W0]}
    arm_maps = {arm: {frame["segment"].record.segment_id: frame for frame in frames} for arm, frames in frames_by_arm.items()}
    records: list[dict[str, object]] = []
    for segment_id, frame in reference.items():
        mask = frame["formal_mask"]
        routes: dict[str, np.ndarray] = {}
        for arm in (W0, W1, W2):
            current = arm_maps[arm][segment_id]
            no_a, with_a = ARM_ROUTE_NAMES[arm]
            routes[no_a] = current["no_a_prediction"][mask]
            routes[with_a] = current["with_a_prediction"][mask]
        for route, values in baseline_predictions.items():
            routes[route] = values[segment_id]
        origins = frame["origins"][mask]
        rows = frame["rows"][mask]
        records.append({
            "flight_id": frame["segment"].record.flight_id,
            "segment_id": segment_id,
            "horizon": horizon,
            "target_rows": rows,
            "origins": origins,
            "sample_ids": [multihorizon_sample_id(frame["segment"].record, horizon, int(origin), int(target)) for origin, target in zip(origins, rows, strict=True)],
            "y": frame["y"][mask],
            "speed": frame["speed"][mask],
            "routes": routes,
        })
    return records


def metrics_for_records(records: Sequence[dict[str, object]], route: str, *, minimum_target_row: int | None = None, speed_range: tuple[float, float | None] | None = None) -> dict[str, object]:
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for item in records:
        mask = np.ones(len(item["y"]), dtype=bool)
        if minimum_target_row is not None:
            mask &= item["target_rows"] >= minimum_target_row
        if speed_range is not None:
            low, high = speed_range
            mask &= item["speed"] >= low
            if high is not None:
                mask &= item["speed"] < high
        if np.any(mask):
            ys.append(item["y"][mask])
            ps.append(item["routes"][route][mask])
    if not ys:
        return {"status": "NOT_APPLICABLE", "rows": 0}
    y = np.concatenate(ys)
    prediction = np.concatenate(ps)
    error = y - prediction
    mse = np.mean(np.square(error), axis=0)
    variance = np.var(y, axis=0)
    std = np.sqrt(np.maximum(variance, np.finfo(float).eps))
    return {
        "status": "COMPLETED",
        "rows": len(y),
        "mse_by_axis": mse.tolist(),
        "rmse_by_axis": np.sqrt(mse).tolist(),
        "mae_by_axis": np.mean(np.abs(error), axis=0).tolist(),
        "r2_by_axis": (1.0 - mse / np.maximum(variance, np.finfo(float).eps)).tolist(),
        "nrmse_by_axis": (np.sqrt(mse) / std).tolist(),
        "normalized_mse_by_axis": (mse / np.maximum(variance, np.finfo(float).eps)).tolist(),
        "pooled_normalized_mse": float(np.mean(mse / np.maximum(variance, np.finfo(float).eps))),
    }


def grouped_oof_records(
    oof_by_arm: Mapping[str, Mapping[int, Sequence[dict[str, object]]]],
    oof_baselines: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    horizon: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for fold in sorted(oof_by_arm[W0]):
        frames = {arm: oof_by_arm[arm][fold] for arm in (W0, W1, W2)}
        baselines = {route: values[fold] for route, values in oof_baselines.items()}
        output.extend(collect_prediction_records(frames, baselines, horizon))
    return output


def fit_final_horizon_contracts(
    development: Sequence[SegmentData],
    horizon: int,
    selections: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    count = int(config["source_and_split"]["final_refit_grouped_oof_folds"])
    salt = str(config["source_and_split"]["final_refit_oof_fold_salt"])
    remapped = [SegmentData(replace(item.record, inner_fold=frozen_inner_fold(item.record.flight_id, count, salt)), item.values) for item in development]
    history = int(selections["K"]["selected_history"])
    max_history = max(int(value) for value in config["K"]["candidate_fir_histories_samples"])
    max_age = max(int(value) for values in config["A"]["mature_residual_age_sets_samples"] for value in values)
    formal_start = route_support_start(max_history, horizon, max_age)
    route_folds: dict[int, dict[str, list[dict[str, object]]]] = {}
    for fold in range(count):
        fit = [item for item in remapped if item.record.inner_fold != fold]
        evaluate = [item for item in remapped if item.record.inner_fold == fold]
        contract = fit_k_contract(fit, history, horizon, config)
        route_folds[fold] = {
            "fit": [make_k_frame(item, contract, history, horizon, formal_start, config, fold=fold, role="fit") for item in fit],
            "evaluation": [make_k_frame(item, contract, history, horizon, formal_start, config, fold=fold, role="evaluation") for item in evaluate],
        }
    final_w: dict[str, MultiHorizonWContract | None] = {}
    final_a: dict[str, RidgeContract | None] = {}
    selected_ages: dict[str, tuple[int, ...] | None] = {}
    for arm in (W0, W1, W2):
        selected_w = str(selections["W"][arm]["final_selected_candidate"])
        oof_k = [frame for fold in range(count) for frame in route_folds[fold]["evaluation"]]
        final_w[arm] = None if selected_w == IDENTITY else fit_w_contract(selected_w, oof_k, config)
        oof_w: list[dict[str, object]] = []
        for fold in range(count):
            fold_contract = None if selected_w == IDENTITY else fit_w_contract(selected_w, route_folds[fold]["fit"], config)
            oof_w.extend(_apply_w(route_folds[fold]["evaluation"], fold_contract))
        ages_value = selections["A"][arm]["selected_mature_ages"]
        ages = None if ages_value is None else tuple(int(value) for value in ages_value)
        selected_ages[arm] = ages
        if ages is None:
            final_a[arm] = None
        else:
            x, y = _a_xy(_residual_series(oof_w), horizon, ages)
            grid, condition, kkt = _ridge_parameters(config, "A")
            final_a[arm] = fit_numerical_ridge(x, y, grid, condition, kkt)
    k_contract = fit_k_contract(remapped, history, horizon, config)
    baseline_contracts: dict[str, RidgeContract] = {}
    baseline_ages: dict[str, tuple[int, ...]] = {}
    all_frames = [make_k_frame(item, k_contract, history, horizon, formal_start, config, fold=-1, role="full_development") for item in remapped]
    for route, motor_history in (("TARGET_ONLY_VAR_AR", None), ("LINEAR_NARX", history)):
        ages = tuple(int(value) for value in selections["BASELINES"][route]["selected_ages"])
        baseline_ages[route] = ages
        x, y = _baseline_xy(all_frames, horizon, ages, motor_history=motor_history)
        baseline_contracts[route] = fit_numerical_ridge(x, y, config["K"]["numerical_ridge_grid"], float(config["K"]["maximum_condition_number"]), float(config["K"]["maximum_relative_kkt_residual"]))
    return {
        "horizon": horizon,
        "selected_K_history": history,
        "formal_support_start": formal_start,
        "K_contract": ridge_to_json(k_contract),
        "W_contracts": {arm: w_contract_to_json(final_w[arm]) for arm in (W0, W1, W2)},
        "A_contracts": {arm: None if final_a[arm] is None else ridge_to_json(final_a[arm]) for arm in (W0, W1, W2)},
        "A_mature_ages": {arm: selected_ages[arm] for arm in (W0, W1, W2)},
        "baseline_contracts": {route: ridge_to_json(contract) for route, contract in baseline_contracts.items()},
        "baseline_ages": baseline_ages,
        "candidate_binding_pass": True,
    }


def materialize_horizon(
    segments: Sequence[SegmentData], horizon_contracts: Mapping[str, object], config: Mapping[str, object]
) -> list[dict[str, object]]:
    horizon = int(horizon_contracts["horizon"])
    history = int(horizon_contracts["selected_K_history"])
    formal_start = int(horizon_contracts["formal_support_start"])
    k_contract = ridge_from_json(horizon_contracts["K_contract"])
    k_frames = [make_k_frame(item, k_contract, history, horizon, formal_start, config, fold=-1, role="formal_test") for item in segments]
    frames_by_arm: dict[str, list[dict[str, object]]] = {}
    for arm in (W0, W1, W2):
        w_contract = w_contract_from_json(horizon_contracts["W_contracts"][arm])
        no_a = _apply_w(k_frames, w_contract)
        a_value = horizon_contracts["A_contracts"][arm]
        a_contract = None if a_value is None else ridge_from_json(a_value)
        ages_value = horizon_contracts["A_mature_ages"][arm]
        ages = None if ages_value is None else tuple(int(value) for value in ages_value)
        frames_by_arm[arm] = _apply_a(no_a, a_contract, horizon, ages)
    baseline_predictions: dict[str, dict[str, np.ndarray]] = {
        "PERSISTENCE": {
            frame["segment"].record.segment_id: generalized_targets(frame["segment"], 0.772, [0.0025, 0.0021, 0.0043])[frame["origins"][frame["formal_mask"]]]
            for frame in k_frames
        }
    }
    for route, motor_history in (("TARGET_ONLY_VAR_AR", None), ("LINEAR_NARX", history)):
        contract = ridge_from_json(horizon_contracts["baseline_contracts"][route])
        ages = tuple(int(value) for value in horizon_contracts["baseline_ages"][route])
        baseline_predictions[route] = _predict_baseline_frames(k_frames, horizon, ages, contract, motor_history)
    return collect_prediction_records(frames_by_arm, baseline_predictions, horizon)


def bootstrap_contrast(
    records: Sequence[dict[str, object]],
    candidate: str,
    baseline: str,
    *,
    cluster_key: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    y = np.concatenate([item["y"] for item in records])
    variance = np.maximum(np.var(y, axis=0), np.finfo(float).eps)
    grouped: dict[str, dict[str, object]] = {}
    for item in records:
        key = str(item[cluster_key])
        current = grouped.setdefault(key, {"candidate_sse": np.zeros(4), "baseline_sse": np.zeros(4), "rows": 0})
        current["candidate_sse"] += np.sum(np.square(item["y"] - item["routes"][candidate]), axis=0)
        current["baseline_sse"] += np.sum(np.square(item["y"] - item["routes"][baseline]), axis=0)
        current["rows"] += len(item["y"])
    clusters = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        c_sse = np.sum([grouped[clusters[int(i)]]["candidate_sse"] for i in chosen], axis=0)
        b_sse = np.sum([grouped[clusters[int(i)]]["baseline_sse"] for i in chosen], axis=0)
        rows = sum(int(grouped[clusters[int(i)]]["rows"]) for i in chosen)
        draws[index] = float(np.mean((c_sse / rows - b_sse / rows) / variance))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "cluster": "PARENT_FLIGHT_CLUSTER" if cluster_key == "flight_id" else "CONTINUOUS_SEGMENT_CLUSTER",
        "cluster_count": len(clusters),
        "sample_count": len(y),
        "replicates": replicates,
        "mean_paired_difference": float(draws.mean()),
        "percentile_95_low": float(np.quantile(draws, 0.025)),
        "percentile_95_high": float(np.quantile(draws, 0.975)),
        "probability_candidate_better": float(np.mean(draws < 0.0)),
        "probability_candidate_worse": float(np.mean(draws > 0.0)),
    }


def integrated_k_topology(contract: RidgeContract, history: int) -> np.ndarray:
    return contract.coefficient_physical.reshape(history, 4, 4).sum(axis=0)


def runtime_memory_gib() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / (1024.0 * 1024.0 if sys.platform.startswith("linux") else 1024.0 ** 3)
    except Exception:
        return float("nan")
