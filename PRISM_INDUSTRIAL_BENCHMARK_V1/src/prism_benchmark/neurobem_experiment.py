"""Stagewise NeuroBEM experiment implementation under the frozen v2.1.1 contract."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import multiprocessing as mp
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np

from .neurobem_data import (
    SegmentData,
    SegmentRecord,
    body_context,
    frozen_inner_fold,
    generalized_targets,
    load_segment,
    motor_thrust_proxy,
    registry_from_zip,
    registry_json,
    sha256_file,
    write_json,
)
from .neurobem_linear import (
    ContextContract,
    EraContract,
    RidgeContract,
    concatenate_k_xy,
    context_at_targets,
    era_from_markov,
    fit_numerical_ridge,
    fit_w_candidate,
    guarded_one_se,
    markov_parameters,
    normalized_mse,
    predict_ridge,
    predict_w,
    segment_k_xy,
    simulate_era,
)


IDENTITY_W = "IDENTITY_CORRECTION"
EXACT_ZERO_A = "EXACT_ZERO"


def _array(value: np.ndarray | None) -> list[object] | None:
    return None if value is None else value.tolist()


def ridge_to_json(contract: RidgeContract) -> dict[str, object]:
    return {
        "alpha": contract.alpha,
        "feature_mean": _array(contract.feature_mean),
        "feature_scale": _array(contract.feature_scale),
        "target_mean": _array(contract.target_mean),
        "coefficient_standardized": _array(contract.coefficient_standardized),
        "coefficient_physical": _array(contract.coefficient_physical),
        "intercept_physical": _array(contract.intercept_physical),
        "condition_number": contract.condition_number,
        "relative_kkt_residual": contract.relative_kkt_residual,
        "fit_rows": contract.fit_rows,
    }


def ridge_from_json(value: Mapping[str, object]) -> RidgeContract:
    return RidgeContract(
        alpha=float(value["alpha"]),
        feature_mean=np.asarray(value["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(value["feature_scale"], dtype=np.float64),
        target_mean=np.asarray(value["target_mean"], dtype=np.float64),
        coefficient_standardized=np.asarray(value["coefficient_standardized"], dtype=np.float64),
        condition_number=float(value["condition_number"]),
        relative_kkt_residual=float(value["relative_kkt_residual"]),
        fit_rows=int(value["fit_rows"]),
    )


def context_to_json(contract: ContextContract | None) -> dict[str, object] | None:
    if contract is None:
        return None
    return {
        "candidate": contract.candidate,
        "ridge": ridge_to_json(contract.ridge),
        "speed_knots": _array(contract.speed_knots),
    }


def context_from_json(value: Mapping[str, object] | None) -> ContextContract | None:
    if value is None:
        return None
    knots = value.get("speed_knots")
    return ContextContract(
        candidate=str(value["candidate"]),
        ridge=ridge_from_json(value["ridge"]),
        speed_knots=None if knots is None else np.asarray(knots, dtype=np.float64),
    )


def era_to_json(contract: EraContract | None) -> dict[str, object] | None:
    if contract is None:
        return None
    return {
        "order": contract.order,
        "A": _array(contract.A),
        "B": _array(contract.B),
        "C": _array(contract.C),
        "D": _array(contract.D),
        "singular_values": _array(contract.singular_values),
        "spectral_radius": contract.spectral_radius,
    }


def era_from_json(value: Mapping[str, object] | None) -> EraContract | None:
    if value is None:
        return None
    return EraContract(
        order=int(value["order"]),
        A=np.asarray(value["A"], dtype=np.float64),
        B=np.asarray(value["B"], dtype=np.float64),
        C=np.asarray(value["C"], dtype=np.float64),
        D=np.asarray(value["D"], dtype=np.float64),
        singular_values=np.asarray(value["singular_values"], dtype=np.float64),
        spectral_radius=float(value["spectral_radius"]),
    )


def records_from_json(values: Iterable[Mapping[str, object]]) -> list[SegmentRecord]:
    return [
        SegmentRecord(
            flight_id=str(value["flight_id"]),
            segment_id=str(value["segment_id"]),
            filename=str(value["filename"]),
            partition=str(value["partition"]),
            inner_fold=None if value.get("inner_fold") is None else int(value["inner_fold"]),
            zip_uncompressed_bytes=int(value["zip_uncompressed_bytes"]),
            zip_crc32=str(value["zip_crc32"]),
        )
        for value in values
    ]


def _numeric(config: Mapping[str, object]) -> tuple[list[float], float, float]:
    k = config["K"]
    return (
        [float(value) for value in k["numerical_ridge_grid"]],
        float(k["maximum_condition_number"]),
        float(k["maximum_relative_kkt_residual"]),
    )


def _target_parameters(config: Mapping[str, object]) -> tuple[float, list[float]]:
    target = config["rigid_body_targets"]
    return float(target["mass_kg"]), [float(value) for value in target["inertia_diagonal_kg_m2"]]


def load_partition(
    records: Iterable[SegmentRecord],
    extracted_root: Path,
    partition: str,
    *,
    allow_locked_test: bool = False,
) -> list[SegmentData]:
    return [
        load_segment(extracted_root, record, allow_locked_test=allow_locked_test)
        for record in records
        if record.partition == partition
    ]


def _fit_k(segments: Sequence[SegmentData], history: int, config: Mapping[str, object]) -> RidgeContract:
    mass, inertia = _target_parameters(config)
    x, y, _ = concatenate_k_xy(segments, history, mass, inertia)
    alpha_grid, condition, kkt = _numeric(config)
    return fit_numerical_ridge(x, y, alpha_grid, condition, kkt)


def _predict_k_segment(
    segment: SegmentData,
    contract: RidgeContract,
    history: int,
    common_support: int,
    config: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mass, inertia = _target_parameters(config)
    x, y, rows = segment_k_xy(segment, history, mass, inertia, support_start=common_support)
    prediction = predict_ridge(contract, x)
    context = context_at_targets(segment, rows)
    return rows, y, prediction, context


_K_WORKER_SEGMENTS: Sequence[SegmentData] = ()
_K_WORKER_CONFIG: Mapping[str, object] = {}


def _k_candidate_fold_worker(task: tuple[int, int]) -> tuple[int, int, RidgeContract, float, float, int]:
    fold, history = task
    train = _K_WORKER_SEGMENTS
    config = _K_WORKER_CONFIG
    common = int(config["K"]["local_common_scoring_history_samples"])
    fit_segments = [segment for segment in train if segment.record.inner_fold != fold]
    eval_segments = [segment for segment in train if segment.record.inner_fold == fold]
    if not fit_segments or not eval_segments:
        raise RuntimeError(f"EMPTY_K_FOLD:{fold}")
    mass, inertia = _target_parameters(config)
    contract = _fit_k(fit_segments, history, config)
    fit_variance = np.var(
        np.concatenate([generalized_targets(segment, mass, inertia)[history:] for segment in fit_segments]),
        axis=0,
    )
    y_eval: list[np.ndarray] = []
    prediction_eval: list[np.ndarray] = []
    for segment in eval_segments:
        _, y, prediction, _ = _predict_k_segment(segment, contract, history, common, config)
        y_eval.append(y)
        prediction_eval.append(prediction)
    y = np.concatenate(y_eval)
    prediction = np.concatenate(prediction_eval)
    loss = normalized_mse(y, prediction, fit_variance)
    zero = normalized_mse(y, np.broadcast_to(contract.target_mean, y.shape), fit_variance)
    return fold, history, contract, loss, zero, int(len(y))


def _parallel_k_candidate_folds(
    train: Sequence[SegmentData], tasks: Sequence[tuple[int, int]], config: Mapping[str, object]
) -> list[tuple[int, int, RidgeContract, float, float, int]]:
    global _K_WORKER_SEGMENTS, _K_WORKER_CONFIG
    _K_WORKER_SEGMENTS = train
    _K_WORKER_CONFIG = config
    workers = min(int(config["runtime"]["workers"]), len(tasks))
    if workers > 1 and sys.platform.startswith("linux"):
        with mp.get_context("fork").Pool(processes=workers) as pool:
            return pool.map(_k_candidate_fold_worker, tasks)
    return [_k_candidate_fold_worker(task) for task in tasks]


def run_k_development(train: Sequence[SegmentData], validation: Sequence[SegmentData], config: Mapping[str, object]) -> tuple[dict[str, object], RidgeContract, dict[int, dict[str, list[dict[str, object]]]], dict[int, RidgeContract]]:
    kcfg = config["K"]
    histories = [int(value) for value in kcfg["candidate_fir_histories_samples"]]
    common = int(kcfg["local_common_scoring_history_samples"])
    folds = int(config["entity_contract"]["inner_folds"])
    mass, inertia = _target_parameters(config)
    losses: dict[int, list[float]] = {history: [] for history in histories}
    zero_losses_by_history: dict[int, list[float]] = {history: [] for history in histories}
    contracts: dict[tuple[int, int], RidgeContract] = {}
    fold_rows: dict[int, dict[str, int]] = defaultdict(dict)
    tasks = [(fold, history) for fold in range(folds) for history in histories]
    for fold, history, contract, loss, zero_loss, row_count in sorted(
        _parallel_k_candidate_folds(train, tasks, config), key=lambda value: (value[0], value[1])
    ):
        contracts[(fold, history)] = contract
        losses[history].append(loss)
        zero_losses_by_history[history].append(zero_loss)
        fold_rows[fold][str(history)] = row_count
    selection = guarded_one_se(losses, histories, float(kcfg["maximum_relative_regret_vs_best"]))
    selected = int(selection["selected"])
    zero_losses = zero_losses_by_history[selected]
    relative_improvement = float((np.mean(zero_losses) - np.mean(losses[selected])) / np.mean(zero_losses))
    positive_fraction = float(np.mean(np.asarray(losses[selected]) < np.asarray(zero_losses)))
    activation_pass = (
        relative_improvement >= float(kcfg["minimum_relative_improvement_vs_zero"])
        and positive_fraction >= float(kcfg["minimum_positive_fold_fraction"])
    )
    route_folds: dict[int, dict[str, list[dict[str, object]]]] = {}
    selected_fold_contracts: dict[int, RidgeContract] = {}
    for fold in range(folds):
        contract = contracts[(fold, selected)]
        selected_fold_contracts[fold] = contract
        route_folds[fold] = {"fit": [], "evaluation": []}
        for role, segments in (
            ("fit", [value for value in train if value.record.inner_fold != fold]),
            ("evaluation", [value for value in train if value.record.inner_fold == fold]),
        ):
            for segment in segments:
                rows, y, prediction, context = _predict_k_segment(segment, contract, selected, common, config)
                route_folds[fold][role].append(
                    {
                        "segment": segment,
                        "fold": fold,
                        "role": role,
                        "rows": rows,
                        "y": y,
                        "k_prediction": prediction,
                        "context": context,
                    }
                )
    validation_contract = _fit_k(train, selected, config)
    validation_y: list[np.ndarray] = []
    validation_prediction: list[np.ndarray] = []
    validation_frames: list[dict[str, object]] = []
    for segment in validation:
        rows, y, prediction, context = _predict_k_segment(segment, validation_contract, selected, common, config)
        validation_y.append(y)
        validation_prediction.append(prediction)
        validation_frames.append({"segment": segment, "rows": rows, "y": y, "k_prediction": prediction, "context": context})
    y_val = np.concatenate(validation_y)
    pred_val = np.concatenate(validation_prediction)
    variance_ratio = np.var(pred_val, axis=0) / np.maximum(np.var(y_val, axis=0), np.finfo(float).eps)
    axis_mse = np.mean(np.square(y_val - pred_val), axis=0)
    zero_axis_mse = np.mean(np.square(y_val - validation_contract.target_mean), axis=0)
    gatecfg = kcfg["input_gate"]
    gate = {
        "variance_pass_by_axis": (variance_ratio >= float(gatecfg["minimum_prediction_variance_to_target_variance_ratio"])).tolist(),
        "coefficient_pass": bool(np.max(np.abs(validation_contract.coefficient_physical)) >= float(gatecfg["minimum_nonintercept_coefficient_abs"])),
        "mse_pass_by_axis": (axis_mse <= float(gatecfg["maximum_mse_ratio_vs_axis_intercept_baseline"]) * zero_axis_mse).tolist(),
        "numerical_pass": bool(validation_contract.condition_number <= float(kcfg["maximum_condition_number"]) and validation_contract.relative_kkt_residual <= float(kcfg["maximum_relative_kkt_residual"])),
    }
    gate["passed"] = bool(activation_pass and all(gate["variance_pass_by_axis"]) and gate["coefficient_pass"] and all(gate["mse_pass_by_axis"]) and gate["numerical_pass"])
    result = {
        "status": "PASS" if gate["passed"] else "PHYSICS_ROUTE_NOT_SUPPORTED",
        "candidate_histories": histories,
        "common_scoring_history": common,
        "candidate_fold_losses": {str(key): value for key, value in losses.items()},
        "exact_zero_fold_losses": zero_losses,
        "exact_zero_fold_losses_by_native_history": {str(key): value for key, value in zero_losses_by_history.items()},
        "selection": selection,
        "selected_history": selected,
        "relative_improvement_vs_zero": relative_improvement,
        "positive_fold_fraction": positive_fraction,
        "activation_pass": activation_pass,
        "fold_evaluation_rows": fold_rows,
        "selected_fold_contracts": {str(key): ridge_to_json(value) for key, value in selected_fold_contracts.items()},
        "train_to_validation_contract": ridge_to_json(validation_contract),
        "validation_axis_mse": axis_mse.tolist(),
        "validation_axis_zero_mse": zero_axis_mse.tolist(),
        "validation_input_gate": gate,
        "test_accessed": False,
        "ood_accessed": False,
    }
    return result, validation_contract, route_folds, selected_fold_contracts


def _concat_frames(frames: Iterable[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = list(frames)
    return (
        np.concatenate([frame["context"] for frame in frames]),
        np.concatenate([frame["k_prediction"] for frame in frames]),
        np.concatenate([frame["y"] for frame in frames]),
    )


def _w_candidates(config: Mapping[str, object]) -> list[str]:
    return [IDENTITY_W, "SIGNED_QUADRATIC_AERO_CONTEXT"] + [
        f"NATURAL_CUBIC_SPEED_CONTEXT_K{int(count)}" for count in config["W"]["natural_cubic_internal_knot_counts"]
    ]


def _fit_w(candidate: str, frames: Iterable[dict[str, object]], config: Mapping[str, object]) -> ContextContract:
    context, k_prediction, y = _concat_frames(frames)
    wcfg = config["W"]
    kcfg = config["K"]
    return fit_w_candidate(
        candidate,
        context,
        k_prediction,
        y - k_prediction,
        [float(value) for value in wcfg["numerical_ridge_grid"]],
        float(kcfg["maximum_condition_number"]),
        float(kcfg["maximum_relative_kkt_residual"]),
    )


def run_w_development(route_folds: dict[int, dict[str, list[dict[str, object]]]], config: Mapping[str, object]) -> tuple[dict[str, object], ContextContract | None, dict[int, dict[str, list[dict[str, object]]]]]:
    folds = int(config["entity_contract"]["inner_folds"])
    candidates = _w_candidates(config)
    losses: dict[str, list[float]] = {candidate: [] for candidate in candidates}
    fold_contracts: dict[tuple[int, str], ContextContract] = {}
    for fold in range(folds):
        fit_frames = route_folds[fold]["fit"]
        eval_frames = route_folds[fold]["evaluation"]
        fit_context, fit_k, fit_y = _concat_frames(fit_frames)
        eval_context, eval_k, eval_y = _concat_frames(eval_frames)
        fit_variance = np.var(fit_y, axis=0)
        losses[IDENTITY_W].append(normalized_mse(eval_y, eval_k, fit_variance))
        for candidate in candidates[1:]:
            contract = _fit_w(candidate, fit_frames, config)
            fold_contracts[(fold, candidate)] = contract
            prediction = eval_k + predict_w(contract, eval_context, eval_k)
            losses[candidate].append(normalized_mse(eval_y, prediction, fit_variance))
    selection = guarded_one_se(losses, candidates, 0.02)
    selected = str(selection["selected"])
    identity = np.asarray(losses[IDENTITY_W])
    selected_losses = np.asarray(losses[selected])
    improvement = float((identity.mean() - selected_losses.mean()) / identity.mean())
    positive = float(np.mean(selected_losses < identity))
    wcfg = config["W"]
    if selected != IDENTITY_W and not (
        improvement >= float(wcfg["minimum_relative_improvement"])
        and positive >= float(wcfg["minimum_positive_fold_fraction"])
    ):
        selected = IDENTITY_W
    selected_route_folds: dict[int, dict[str, list[dict[str, object]]]] = {}
    for fold in range(folds):
        selected_route_folds[fold] = {"fit": [], "evaluation": []}
        for role in ["fit", "evaluation"]:
            for frame in route_folds[fold][role]:
                copied = dict(frame)
                if selected == IDENTITY_W:
                    copied["pf_prediction"] = frame["k_prediction"]
                else:
                    contract = fold_contracts[(fold, selected)]
                    copied["pf_prediction"] = frame["k_prediction"] + predict_w(contract, frame["context"], frame["k_prediction"])
                selected_route_folds[fold][role].append(copied)
    evaluation_frames = [frame for fold in range(folds) for frame in route_folds[fold]["evaluation"]]
    final_contract = None if selected == IDENTITY_W else _fit_w(selected, evaluation_frames, config)
    result = {
        "status": "PASS",
        "candidates": candidates,
        "candidate_fold_losses": losses,
        "selection_before_activation_guard": selection,
        "selected_W": selected,
        "relative_improvement_vs_identity": improvement,
        "positive_fold_fraction": positive,
        "identity_equivalence_pass": bool(selected != IDENTITY_W or all(np.array_equal(frame["pf_prediction"], frame["k_prediction"]) for fold in selected_route_folds.values() for role in fold.values() for frame in role)),
        "final_contract": context_to_json(final_contract),
        "airflow_observed": False,
        "causal_aerodynamic_claim_allowed": False,
        "test_accessed": False,
        "ood_accessed": False,
    }
    return result, final_contract, selected_route_folds


def _residual_series(frames: Iterable[dict[str, object]]) -> dict[str, tuple[SegmentData, int, np.ndarray]]:
    result: dict[str, tuple[SegmentData, int, np.ndarray]] = {}
    for frame in frames:
        segment = frame["segment"]
        residual = np.full((segment.row_count, 4), np.nan, dtype=np.float64)
        residual[frame["rows"]] = frame["y"] - frame["pf_prediction"]
        result[segment.record.segment_id] = (segment, int(frame["fold"]), residual)
    return result


def _a_xy(series: Iterable[tuple[SegmentData, int, np.ndarray]], lags: Sequence[int]) -> tuple[np.ndarray, np.ndarray, list[tuple[SegmentData, np.ndarray]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    provenance: list[tuple[SegmentData, np.ndarray]] = []
    maximum = max(lags)
    for segment, _, residual in series:
        rows = np.arange(maximum, segment.row_count, dtype=np.int64)
        features = np.column_stack([residual[rows - lag] for lag in lags])
        target = residual[rows]
        valid = np.isfinite(features).all(axis=1) & np.isfinite(target).all(axis=1)
        if np.any(valid):
            xs.append(features[valid])
            ys.append(target[valid])
            provenance.append((segment, rows[valid]))
    if not xs:
        raise ValueError("no mature residual rows")
    return np.concatenate(xs), np.concatenate(ys), provenance


def run_a_development(route_folds: dict[int, dict[str, list[dict[str, object]]]], config: Mapping[str, object]) -> tuple[dict[str, object], RidgeContract | None, Sequence[int] | None]:
    fold_count = int(config["entity_contract"]["inner_folds"])
    lag_sets = [tuple(int(value) for value in values) for values in config["A"]["lag_sets_samples"]]
    losses: dict[str, list[float]] = {EXACT_ZERO_A: []}
    for lags in lag_sets:
        losses["L" + "_".join(map(str, lags))] = []
    for fold in range(fold_count):
        fit_series = list(_residual_series(route_folds[fold]["fit"]).values())
        eval_series = list(_residual_series(route_folds[fold]["evaluation"]).values())
        for lags in lag_sets:
            key = "L" + "_".join(map(str, lags))
            x_fit, y_fit, _ = _a_xy(fit_series, lags)
            x_eval, y_eval, _ = _a_xy(eval_series, lags)
            alpha, condition, kkt = _numeric(config)
            contract = fit_numerical_ridge(x_fit, y_fit, [float(v) for v in config["A"]["numerical_ridge_grid"]], condition, kkt)
            prediction = predict_ridge(contract, x_eval)
            losses[key].append(normalized_mse(y_eval, prediction, np.var(y_fit, axis=0)))
            if lags == lag_sets[0]:
                losses[EXACT_ZERO_A].append(normalized_mse(y_eval, np.zeros_like(y_eval), np.var(y_fit, axis=0)))
    complexity = [EXACT_ZERO_A] + ["L" + "_".join(map(str, lags)) for lags in lag_sets]
    selection = guarded_one_se(losses, complexity, 0.02)
    selected = str(selection["selected"])
    baseline = np.asarray(losses[EXACT_ZERO_A])
    chosen = np.asarray(losses[selected])
    improvement = float((baseline.mean() - chosen.mean()) / baseline.mean())
    positive = float(np.mean(chosen < baseline))
    acfg = config["A"]
    if selected != EXACT_ZERO_A and not (
        improvement >= float(acfg["minimum_relative_improvement"])
        and positive >= float(acfg["minimum_positive_fold_fraction"])
    ):
        selected = EXACT_ZERO_A
    selected_lags = None if selected == EXACT_ZERO_A else tuple(int(value) for value in selected[1:].split("_"))
    final_contract = None
    if selected_lags is not None:
        evaluation_frames = [frame for fold in range(fold_count) for frame in route_folds[fold]["evaluation"]]
        x, y, _ = _a_xy(_residual_series(evaluation_frames).values(), selected_lags)
        _, condition, kkt = _numeric(config)
        final_contract = fit_numerical_ridge(x, y, [float(v) for v in acfg["numerical_ridge_grid"]], condition, kkt)
    return {
        "status": "PASS",
        "candidate_fold_losses": losses,
        "selection_before_activation_guard": selection,
        "selected_A": selected,
        "selected_lags": selected_lags,
        "relative_improvement_vs_exact_zero": improvement,
        "positive_fold_fraction": positive,
        "final_contract": None if final_contract is None else ridge_to_json(final_contract),
        "causal_wind_or_vortex_claim_allowed": False,
        "test_accessed": False,
        "ood_accessed": False,
    }, final_contract, selected_lags


def _center_for_state_space(k_contract: RidgeContract, history: int) -> tuple[np.ndarray, np.ndarray]:
    markov = markov_parameters(k_contract, history)
    center = k_contract.feature_mean.reshape(history, 4).mean(axis=0)
    intercept = k_contract.intercept_physical + np.sum(markov @ center, axis=0)
    return center, intercept


def run_era_development(k_contract: RidgeContract, history: int, validation: Sequence[SegmentData], config: Mapping[str, object]) -> tuple[dict[str, object], EraContract | None]:
    ecfg = config["mimo_realization"]
    markov = markov_parameters(k_contract, history)
    block_rows = min(20, history // 2)
    center, intercept = _center_for_state_space(k_contract, history)
    mass, inertia = _target_parameters(config)
    per_order_by_flight: dict[int, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    k_by_flight: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    common = int(config["K"]["local_common_scoring_history_samples"])
    for segment in validation:
        _, y, prediction, _ = _predict_k_segment(segment, k_contract, history, common, config)
        k_by_flight[segment.record.flight_id].append((y, prediction))
    contracts: dict[int, EraContract] = {}
    invalid: dict[str, str] = {}
    for order in [int(value) for value in ecfg["candidate_orders"]]:
        try:
            contract = era_from_markov(markov, order, block_rows)
            if not contract.spectral_radius < float(ecfg["maximum_spectral_radius"]):
                invalid[str(order)] = f"UNSTABLE_SPECTRAL_RADIUS:{contract.spectral_radius}"
                continue
        except (ValueError, np.linalg.LinAlgError) as exc:
            invalid[str(order)] = str(exc)
            continue
        contracts[order] = contract
        by_flight: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        for segment in validation:
            u = motor_thrust_proxy(segment) - center
            prediction = simulate_era(contract, u) + intercept
            y = generalized_targets(segment, mass, inertia)
            rows = np.arange(int(config["K"]["local_common_scoring_history_samples"]), segment.row_count)
            by_flight[segment.record.flight_id].append((y[rows], prediction[rows]))
        per_order_by_flight[order] = by_flight
    if not contracts:
        return {
            "status": "MIMO_REALIZATION_NOT_SUPPORTED",
            "invalid_orders": invalid,
            "markov_parameters": _array(markov),
            "test_accessed": False,
            "ood_accessed": False,
        }, None
    losses: dict[int, list[float]] = {}
    for order, by_flight in per_order_by_flight.items():
        losses[order] = []
        for pairs in by_flight.values():
            y = np.concatenate([pair[0] for pair in pairs])
            prediction = np.concatenate([pair[1] for pair in pairs])
            losses[order].append(normalized_mse(y, prediction, np.var(y, axis=0)))
    selection = guarded_one_se(losses, sorted(contracts), 0.02)
    selected = int(selection["selected"])
    contract = contracts[selected]
    k_losses: list[float] = []
    for pairs in k_by_flight.values():
        y = np.concatenate([pair[0] for pair in pairs])
        prediction = np.concatenate([pair[1] for pair in pairs])
        k_losses.append(normalized_mse(y, prediction, np.var(y, axis=0)))
    selected_mean = float(np.mean(losses[selected]))
    k_mean = float(np.mean(k_losses))
    allowed = float(config["assembly"]["maximum_mse_ratio_vs_best_K_for_input_preservation"]) * k_mean
    preservation_pass = bool(selected_mean <= allowed)
    status = "PASS" if preservation_pass else "MIMO_REALIZATION_STABLE_BUT_NOT_PREDICTIVELY_PRESERVED"
    return {
        "status": status,
        "block_rows": block_rows,
        "candidate_flight_losses": {str(key): value for key, value in losses.items()},
        "invalid_orders": invalid,
        "selection": selection,
        "selected_order": selected,
        "selected_spectral_radius": contract.spectral_radius,
        "frozen_K_flight_losses": k_losses,
        "frozen_K_mean_loss": k_mean,
        "selected_ERA_mean_loss": selected_mean,
        "maximum_allowed_ERA_mean_loss": allowed,
        "K_prediction_preservation_pass": preservation_pass,
        "formal_test_eligible": preservation_pass,
        "singular_values": _array(contract.singular_values),
        "markov_parameters": _array(markov),
        "test_accessed": False,
        "ood_accessed": False,
    }, contract


def final_crossfit_contracts(
    development: Sequence[SegmentData],
    history: int,
    selected_w: str,
    selected_a_lags: Sequence[int] | None,
    config: Mapping[str, object],
) -> tuple[RidgeContract, ContextContract | None, RidgeContract | None, list[dict[str, object]]]:
    freeze = config["development_freeze"]
    count = int(freeze["final_refit_grouped_oof_folds"])
    salt = str(freeze["final_refit_oof_fold_salt"])
    remapped = [
        SegmentData(replace(segment.record, inner_fold=frozen_inner_fold(segment.record.flight_id, count, salt)), segment.values)
        for segment in development
    ]
    route_folds: dict[int, dict[str, list[dict[str, object]]]] = {}
    common = int(config["K"]["local_common_scoring_history_samples"])
    for fold in range(count):
        fit = [segment for segment in remapped if segment.record.inner_fold != fold]
        evaluate = [segment for segment in remapped if segment.record.inner_fold == fold]
        if not fit or not evaluate:
            raise RuntimeError(f"EMPTY_FINAL_OOF_FOLD:{fold}")
        contract = _fit_k(fit, history, config)
        route_folds[fold] = {"fit": [], "evaluation": []}
        for role, segments in (("fit", fit), ("evaluation", evaluate)):
            for segment in segments:
                rows, y, prediction, context = _predict_k_segment(segment, contract, history, common, config)
                route_folds[fold][role].append(
                    {"segment": segment, "fold": fold, "role": role, "rows": rows, "y": y, "k_prediction": prediction, "context": context}
                )
    pf_evaluation_frames: list[dict[str, object]] = []
    for fold in range(count):
        fit_frames = route_folds[fold]["fit"]
        contract = None if selected_w == IDENTITY_W else _fit_w(selected_w, fit_frames, config)
        for frame in route_folds[fold]["evaluation"]:
            copied = dict(frame)
            copied["pf_prediction"] = frame["k_prediction"] if contract is None else frame["k_prediction"] + predict_w(contract, frame["context"], frame["k_prediction"])
            pf_evaluation_frames.append(copied)
    a_contract = None
    if selected_a_lags is not None:
        series = _residual_series(pf_evaluation_frames)
        x, y, _ = _a_xy(series.values(), selected_a_lags)
        _, condition, kkt = _numeric(config)
        a_contract = fit_numerical_ridge(x, y, [float(v) for v in config["A"]["numerical_ridge_grid"]], condition, kkt)
    k_contract = _fit_k(remapped, history, config)
    full_frames: list[dict[str, object]] = []
    for segment in remapped:
        rows, y, prediction, context = _predict_k_segment(segment, k_contract, history, common, config)
        full_frames.append({"segment": segment, "fold": -1, "role": "full_fit", "rows": rows, "y": y, "k_prediction": prediction, "context": context})
    w_contract = None if selected_w == IDENTITY_W else _fit_w(selected_w, full_frames, config)
    return k_contract, w_contract, a_contract, pf_evaluation_frames


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    error = y - prediction
    mse = np.mean(np.square(error), axis=0)
    variance = np.var(y, axis=0)
    return {
        "rows": int(len(y)),
        "rmse_by_axis": np.sqrt(mse).tolist(),
        "mae_by_axis": np.mean(np.abs(error), axis=0).tolist(),
        "r2_by_axis": (1.0 - mse / np.maximum(variance, np.finfo(float).eps)).tolist(),
        "normalized_mse": float(np.mean(mse / np.maximum(variance, np.finfo(float).eps))),
    }


def predict_test_segment(
    segment: SegmentData,
    history: int,
    k_contract: RidgeContract,
    w_contract: ContextContract | None,
    a_contract: RidgeContract | None,
    a_lags: Sequence[int] | None,
    era_contract: EraContract | None,
    config: Mapping[str, object],
) -> dict[str, object]:
    common = int(config["K"]["local_common_scoring_history_samples"])
    rows, y, k_prediction, context = _predict_k_segment(segment, k_contract, history, common, config)
    kw_prediction = k_prediction if w_contract is None else k_prediction + predict_w(w_contract, context, k_prediction)
    pf_prediction = kw_prediction.copy()
    if a_contract is not None and a_lags is not None:
        residual = np.full((segment.row_count, 4), np.nan, dtype=np.float64)
        residual[rows] = y - kw_prediction
        maximum = max(a_lags)
        for local_index, target_row in enumerate(rows):
            if target_row < common + maximum:
                continue
            features = np.concatenate([residual[target_row - lag] for lag in a_lags])[None, :]
            if np.isfinite(features).all():
                pf_prediction[local_index] += predict_ridge(a_contract, features)[0]
    era_prediction = None
    if era_contract is not None:
        center, intercept = _center_for_state_space(k_contract, history)
        full_era = simulate_era(era_contract, motor_thrust_proxy(segment) - center) + intercept
        era_prediction = full_era[rows]
    speed = np.linalg.norm(body_context(segment)[rows - 1, :3], axis=1)
    return {
        "rows": rows,
        "y": y,
        "K": k_prediction,
        "KW": kw_prediction,
        "PF_SELECTED": pf_prediction,
        "KWA": pf_prediction,
        "ERA_K": era_prediction,
        "speed": speed,
    }


def aggregate_predictions(values: Iterable[dict[str, object]], mask_name: str | None = None, threshold: float = 15.0) -> dict[str, object]:
    values = list(values)
    result: dict[str, object] = {}
    for route in ["K", "KW", "KWA", "PF_SELECTED", "ERA_K"]:
        pairs = []
        for value in values:
            if value[route] is None:
                continue
            mask = np.ones(len(value["y"]), dtype=bool) if mask_name is None else value["speed"] >= threshold
            if np.any(mask):
                pairs.append((value["y"][mask], value[route][mask]))
        result[route] = "NOT_APPLICABLE" if not pairs else _metrics(
            np.concatenate([pair[0] for pair in pairs]), np.concatenate([pair[1] for pair in pairs])
        )
    return result


def bootstrap_segment_differences(
    predictions: Sequence[dict[str, object]],
    replicates: int,
    seed: int = 211,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    comparisons = [("KW", "K"), ("PF_SELECTED", "KW"), ("PF_SELECTED", "K")]
    output: dict[str, object] = {}
    count = len(predictions)
    for candidate, baseline in comparisons:
        draws = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            chosen = rng.integers(0, count, size=count)
            candidate_error: list[np.ndarray] = []
            baseline_error: list[np.ndarray] = []
            scale: list[np.ndarray] = []
            for index in chosen:
                value = predictions[int(index)]
                candidate_error.append(np.square(value["y"] - value[candidate]))
                baseline_error.append(np.square(value["y"] - value[baseline]))
                scale.append(np.square(value["y"] - np.mean(value["y"], axis=0)))
            denominator = np.maximum(np.mean(np.concatenate(scale), axis=0), np.finfo(float).eps)
            draws[replicate] = float(np.mean((np.mean(np.concatenate(candidate_error), axis=0) - np.mean(np.concatenate(baseline_error), axis=0)) / denominator))
        output[f"{candidate}_vs_{baseline}"] = {
            "mean_normalized_mse_difference_candidate_minus_baseline": float(np.mean(draws)),
            "percentile_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
            "replicates": replicates,
            "cluster": "CONTIGUOUS_PROCESSED_SEGMENT_ID",
        }
    return output
