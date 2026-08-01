from __future__ import annotations

import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .c2_models import _capped, _folds, _prediction_frame
from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    deterministic_subsample,
    geometric_intervals,
    input_columns,
    load_samples,
    main_views,
    sha256_file,
)
from .cpu_selection import mse, regression_metrics, select_one_se
from .stage0 import write_json
from .urysohn import NaturalCubicBasis, fit_full, fit_rank_als, solve_penalized, tensor_design


def _freeze(project: Path) -> dict[str, Any]:
    return json.loads((project / "configs/cpu_model_freeze_v1.json").read_text(encoding="utf-8"))


def channel_class(dataset: str, channel: str) -> str:
    lower = channel.lower()
    if dataset == "tep":
        return "FAST" if lower.startswith("xmv_") else "MEDIUM"
    if dataset in {"debutanizer", "sru"}:
        return "MEDIUM"
    if dataset == "pmsm":
        return "SLOW" if lower in {"ambient", "coolant", "stator_winding", "stator_tooth", "stator_yoke"} else "FAST"
    if dataset == "metropt":
        if "temperature" in lower:
            return "SLOW"
        if any(token in lower for token in ("comp", "switch", "current", "caudal", "dv_", "lps", "mpg")):
            return "FAST"
        return "MEDIUM"
    raise KeyError(dataset)


def channel_profiles(view: ViewSpec, channel: str, config: dict[str, Any]) -> list[tuple[int, int]]:
    category = channel_class(view.head.dataset, channel)
    deltas = [int(value) for value in config["delta_ratio_by_class"][category]]
    maximum_delta = max(1, view.head.h_steps + view.head.w_steps)
    deltas = [delta for delta in deltas if delta <= maximum_delta] or [min(config["delta_ratio_by_class"][category])]
    if view.head.h_steps > 0:
        histories = [max(1, int(multiplier * view.head.h_steps)) for multiplier in config["history_for_positive_h"]]
        return sorted({(delta, history) for delta in deltas for history in histories if delta <= history}, key=lambda value: (value[1], -value[0]))
    return sorted(
        {(delta, int(multiplier * delta)) for delta in deltas for multiplier in config["history_for_zero_h_in_delta"]},
        key=lambda value: (value[1], -value[0]),
    )


def profile_intervals(delta: int, history: int, m_tau: int) -> list[tuple[int, int]]:
    units = max(1, int(math.ceil(history / delta)))
    intervals = geometric_intervals(units, min(m_tau, units))
    converted = [(near * delta, min(far * delta, history)) for near, far in intervals]
    converted[-1] = (converted[-1][0], history)
    return [(near, far) for near, far in converted if far > near]


def profile_values(
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    channel: str,
    profile: tuple[int, int],
    m_tau: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    intervals = profile_intervals(profile[0], profile[1], m_tau)
    return accessor.block_means(samples, channel, intervals), intervals


def _fit_candidate(
    values_train: np.ndarray,
    target: np.ndarray,
    values_eval: np.ndarray,
    kind: str,
    m_x: int,
    lambdas: tuple[float, float, float],
    solver_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], int]:
    lambda_0, lambda_tau, lambda_x = lambdas
    if kind == "exact_zero":
        return np.zeros(len(values_eval), dtype=np.float64), {
            "kind": kind,
            "intercept": 0.0,
            "theta": [],
            "basis": None,
            "numerical_certificate": {"relative_kkt": 0.0, "condition_number": 1.0},
        }, 0
    requested_dimension = 1 if kind == "linear" else m_x
    basis = NaturalCubicBasis.fit(values_train.reshape(-1), requested_dimension)
    phi_train = tensor_design(values_train, basis)
    phi_eval = tensor_design(values_eval, basis)
    if kind == "linear":
        design = phi_train[:, :, 0]
        evaluation = phi_eval[:, :, 0]
        prediction, coefficient, certificate = solve_penalized(
            design,
            target,
            evaluation,
            lambda_0 * np.eye(design.shape[1], dtype=np.float64),
        )
        theta = coefficient[:, None]
        parameter_count = len(coefficient) + 1
    elif kind == "full":
        prediction, theta, certificate = fit_full(
            phi_train,
            target,
            phi_eval,
            lambda_0,
            lambda_tau,
            lambda_x,
        )
        parameter_count = theta.size + 1
    elif kind.startswith("rank_"):
        rank = int(kind.split("_")[1])
        candidates = []
        for seed in range(int(solver_config["als_initializations"])):
            candidate = fit_rank_als(
                phi_train,
                target,
                phi_eval,
                rank,
                lambda_0,
                lambda_tau,
                lambda_x,
                int(solver_config["als_max_iterations"]),
                float(solver_config["als_relative_tolerance"]),
                seed,
            )
            candidates.append(candidate)
        prediction, theta, certificate = min(candidates, key=lambda value: value[2]["train_mse"])
        certificate["initializations"] = len(candidates)
        parameter_count = rank * (theta.shape[0] + theta.shape[1]) + 1
    else:
        raise KeyError(kind)
    return prediction, {
        "kind": kind,
        "theta": np.asarray(theta, dtype=np.float64).tolist(),
        "basis": basis.metadata(),
        "intercept": float(certificate["intercept"]),
        "numerical_certificate": certificate,
    }, parameter_count


def _finite(losses: dict[Any, list[float]]) -> dict[Any, list[float]]:
    result = {key: value for key, value in losses.items() if len(value) >= 2 and np.all(np.isfinite(value))}
    if not result:
        raise ValueError("all validation candidates failed")
    return result


def _numerical_status(contract: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, float, float]:
    certificate = contract["numerical_certificate"]
    if "relative_kkt" in certificate:
        kkt = float(certificate["relative_kkt"])
        condition = float(certificate["condition_number"])
    else:
        subproblems = certificate.get("subproblem_certificates", {})
        parts = [subproblems.get("u", {}), subproblems.get("v", {})]
        kkt = max((float(part.get("relative_kkt", float("inf"))) for part in parts), default=float("inf"))
        condition = max((float(part.get("condition_number", float("inf"))) for part in parts), default=float("inf"))
    passed = kkt <= float(thresholds["relative_kkt"]) and condition <= float(thresholds["condition_hard_fail"])
    return ("PASS" if passed else "FAILED", kkt, condition)


def _evaluate_candidate(
    accessor: BaseAccessor,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    channel: str,
    profile: tuple[int, int],
    m_tau: int,
    kind: str,
    m_x: int,
    lambdas: tuple[float, float, float],
    solver: dict[str, Any],
) -> list[float]:
    losses = []
    for train_index, validation_index in folds:
        train_subset = train.iloc[train_index]
        validation_subset = train.iloc[validation_index]
        train_values, _ = profile_values(accessor, train_subset, channel, profile, m_tau)
        validation_values, _ = profile_values(accessor, validation_subset, channel, profile, m_tau)
        prediction, _, _ = _fit_candidate(
            train_values,
            train_subset["y_true"].to_numpy(dtype=np.float64),
            validation_values,
            kind,
            m_x,
            lambdas,
            solver,
        )
        losses.append(mse(validation_subset["y_true"].to_numpy(dtype=np.float64), prediction))
    return losses


def _run_channel(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    channel: str,
) -> dict[str, Any]:
    started = time.time()
    destination = output / "CHANNELS" / view.head.head_id / view.proxy_policy / channel
    destination.mkdir(parents=True, exist_ok=True)
    try:
        freeze = _freeze(project)
        config = freeze["c4"]
        train = load_samples(shared, view, "train")
        validation = load_samples(shared, view, "validation")
        accessor = BaseAccessor(shared, view.head.dataset, "train", [channel])
        fit_cap = int(config["fit_row_cap"])
        validation_cap = int(config["selection_validation_row_cap"])
        folds = [
            (_capped(train, train_index, fit_cap), _capped(train, validation_index, validation_cap))
            for train_index, validation_index in _folds(train, view)
        ]
        profiles = channel_profiles(view, channel, config)
        pilot = config["penalties"]["pilot"]
        pilot_lambdas = (float(pilot["lambda_0"]), float(pilot["lambda_tau"]), float(pilot["lambda_x"]))
        pilot_m_tau = 8
        profile_losses = {
            profile: _evaluate_candidate(
                accessor,
                train,
                folds,
                channel,
                profile,
                pilot_m_tau,
                "linear",
                1,
                pilot_lambdas,
                config["solver"],
            )
            for profile in profiles
        }
        zero_losses = [
            float(np.mean(np.square(train.iloc[validation_index]["y_true"].to_numpy(dtype=np.float64)), dtype=np.float64))
            for _, validation_index in folds
        ]
        profile_selection = select_one_se(_finite(profile_losses), lambda value: (value[1], -value[0]))
        selected_profile = tuple(profile_selection.candidate)
        zero_linear_losses: dict[Any, list[float]] = {"exact_zero": zero_losses, "linear": profile_losses[selected_profile]}
        zero_linear = select_one_se(zero_linear_losses, lambda value: (0,) if value == "exact_zero" else (1,))
        zero_mean = float(np.mean(zero_losses, dtype=np.float64))
        linear_mean = float(np.mean(profile_losses[selected_profile], dtype=np.float64))
        improvement = 0.0 if zero_mean == 0 else (zero_mean - linear_mean) / zero_mean
        if zero_linear.candidate == "exact_zero" or improvement < float(config["channel_candidate_gate"]["validation_improvement_vs_zero_min"]):
            selected_kind = "exact_zero"
            selected_m_tau = pilot_m_tau
            selected_m_x = 1
            selected_lambdas = pilot_lambdas
            resolution_selection = None
            ladder_selection = zero_linear.__dict__
            penalty_audit = {}
        else:
            resolution_losses = {}
            for m_tau in config["lag_basis"]["candidate_m_tau"]:
                for m_x in config["amplitude_basis"]["candidate_m_x"]:
                    try:
                        resolution_losses[(int(m_tau), int(m_x))] = _evaluate_candidate(
                            accessor,
                            train,
                            folds,
                            channel,
                            selected_profile,
                            int(m_tau),
                            "full",
                            int(m_x),
                            pilot_lambdas,
                            config["solver"],
                        )
                    except Exception:
                        resolution_losses[(int(m_tau), int(m_x))] = [float("inf")] * len(folds)
            resolution_selection = select_one_se(_finite(resolution_losses), lambda value: (value[0], value[1]))
            selected_m_tau, selected_m_x = resolution_selection.candidate
            ladder_losses: dict[str, list[float]] = {"exact_zero": zero_losses}
            for kind in ["linear", "rank_1", "rank_2", "rank_3", "rank_4", "full"]:
                try:
                    ladder_losses[kind] = _evaluate_candidate(
                        accessor,
                        train,
                        folds,
                        channel,
                        selected_profile,
                        selected_m_tau,
                        kind,
                        selected_m_x,
                        pilot_lambdas,
                        config["solver"],
                    )
                except Exception:
                    ladder_losses[kind] = [float("inf")] * len(folds)
            order = {name: index for index, name in enumerate(config["complexity_ladder"])}
            ladder_selection_object = select_one_se(_finite(ladder_losses), lambda value: (order[value],))
            selected_kind = str(ladder_selection_object.candidate)
            ladder_selection = ladder_selection_object.__dict__
            lambdas = list(pilot_lambdas)
            penalty_audit = {}
            for position, name in enumerate(("lambda_0", "lambda_tau", "lambda_x")):
                scan_losses = {}
                for value in config["penalties"][name]:
                    candidate_lambdas = list(lambdas)
                    candidate_lambdas[position] = float(value)
                    try:
                        scan_losses[float(value)] = _evaluate_candidate(
                            accessor,
                            train,
                            folds,
                            channel,
                            selected_profile,
                            selected_m_tau,
                            selected_kind,
                            selected_m_x,
                            tuple(candidate_lambdas),
                            config["solver"],
                        )
                    except Exception:
                        scan_losses[float(value)] = [float("inf")] * len(folds)
                selection = select_one_se(_finite(scan_losses), lambda value: (-value,))
                lambdas[position] = float(selection.candidate)
                penalty_audit[name] = {"selection": selection.__dict__, "fold_losses": scan_losses}
            selected_lambdas = tuple(lambdas)

        final_index = deterministic_subsample(train, fit_cap)
        train_subset = train.iloc[final_index]
        validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", [channel])
        train_values, intervals = profile_values(accessor, train_subset, channel, selected_profile, selected_m_tau)
        validation_values, _ = profile_values(validation_accessor, validation, channel, selected_profile, selected_m_tau)
        prediction, model_contract, parameter_count = _fit_candidate(
            train_values,
            train_subset["y_true"].to_numpy(dtype=np.float64),
            validation_values,
            selected_kind,
            selected_m_x,
            selected_lambdas,
            config["solver"],
        )
        numerical_status, relative_kkt, condition_number = _numerical_status(
            model_contract,
            config["numerical_thresholds"],
        )
        frame = _prediction_frame(validation, view, f"PRISM_CHANNEL_{channel}", prediction, parameter_count)
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        result = {
            "status": "PASS",
            "stage": "C4_CHANNEL_PROFILE_AND_LADDER",
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "channel": channel,
            "channel_class": channel_class(view.head.dataset, channel),
            "selected_profile": list(selected_profile),
            "selected_intervals": [list(value) for value in intervals],
            "selected_kind": selected_kind,
            "selected_m_tau": int(selected_m_tau),
            "selected_m_x": int(selected_m_x),
            "selected_lambdas": list(selected_lambdas),
            "linear_improvement_vs_zero": improvement,
            "candidate_for_joint_refit": selected_kind != "exact_zero" and numerical_status == "PASS",
            "numerical_status": numerical_status,
            "relative_kkt": relative_kkt,
            "condition_number": condition_number,
            "profile_selection": profile_selection.__dict__,
            "profile_fold_losses": {str(key): value for key, value in profile_losses.items()},
            "resolution_selection": None if resolution_selection is None else resolution_selection.__dict__,
            "ladder_selection": ladder_selection,
            "penalty_audit": penalty_audit,
            "model_contract": model_contract,
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "parameter_count": parameter_count,
            "dtype": "float64",
            "test_accessed": False,
            "elapsed_seconds": time.time() - started,
            **regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()),
        }
    except Exception as error:
        result = {
            "status": "FAILED_RETAINED",
            "stage": "C4_CHANNEL_PROFILE_AND_LADDER",
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "channel": channel,
            "test_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    write_json(destination / "RESULT.json", result)
    return result


def _run_channel_jobs(shared: Path, project: Path, output: Path, n_jobs: int) -> list[dict[str, Any]]:
    jobs = []
    for view in main_views(shared, "input_only"):
        jobs.extend((view, channel) for channel in input_columns(shared, view.head.task_id, view.proxy_policy))
    results = []
    pending = []
    for view, channel in jobs:
        path = output / "CHANNELS" / view.head.head_id / view.proxy_policy / channel / "RESULT.json"
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("status") in {"PASS", "FAILED_RETAINED"}:
                results.append(previous)
                continue
        pending.append((view, channel))
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(_run_channel, shared, project, output, view, channel): (view, channel)
            for view, channel in pending
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def run_c4(shared: Path, project: Path, output: Path, n_jobs: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    results = _run_channel_jobs(shared, project, output, n_jobs)
    flat = [{key: value for key, value in result.items() if key not in {"model_contract", "penalty_audit", "traceback", "profile_fold_losses"}} for result in results]
    pd.DataFrame(flat).sort_values(["task", "channel"]).to_csv(output / "PRISM_PROFILE_AUDIT.csv", index=False)
    manifest = {
        "status": "PASS" if all(result["status"] == "PASS" for result in results) else "PASS_WITH_RETAINED_FAILURES",
        "channel_jobs": len(results),
        "passed": sum(result["status"] == "PASS" for result in results),
        "failed_retained": sum(result["status"] != "PASS" for result in results),
        "exact_zero": sum(result.get("selected_kind") == "exact_zero" for result in results),
        "joint_candidates": sum(bool(result.get("candidate_for_joint_refit")) for result in results),
        "test_accessed": False,
        "freeze_sha256": sha256_file(project / "configs/cpu_model_freeze_v1.json"),
    }
    write_json(output / "C4_MANIFEST.json", manifest)
    return manifest
