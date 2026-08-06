from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .v2_numerics import solve_certified
from .v21_w import IDENTITY, build_w_design


J_K = "J_K"
J_KW = "J_KW"
J_KA = "J_KA"
J_KWA = "J_KWA"
JOINT_CANDIDATES = (J_K, J_KW, J_KA, J_KWA)


def registered_joint_candidates() -> tuple[str, ...]:
    return JOINT_CANDIDATES


def _required_blocks(candidate: str) -> tuple[str, ...]:
    if candidate == J_K:
        return ("K",)
    if candidate == J_KW:
        return ("K", "W")
    if candidate == J_KA:
        return ("K", "A")
    if candidate == J_KWA:
        return ("K", "W", "A")
    raise ValueError(f"unregistered Joint candidate: {candidate}")


def _as_matrix(value: np.ndarray | None, rows: int) -> np.ndarray:
    if value is None:
        return np.empty((rows, 0), dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or len(matrix) != rows:
        raise ValueError("Joint block has invalid shape")
    return matrix


def joint_w_basis(
    train_seed_latent: np.ndarray,
    evaluation_seed_latent: np.ndarray,
    w_contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    family = str(w_contract["family"])
    if family == IDENTITY:
        return (
            np.empty((len(train_seed_latent), 0), dtype=np.float64),
            np.empty((len(evaluation_seed_latent), 0), dtype=np.float64),
            {"family": IDENTITY, "columns": 0, "hard_feature_residualization": False},
        )
    design = build_w_design(
        train_seed_latent,
        evaluation_seed_latent,
        family,
        int(w_contract["knot_count"]),
    )
    return design.train, design.evaluation, design.metadata


def fit_joint_candidate(
    train_blocks: Mapping[str, np.ndarray | None],
    target: np.ndarray,
    evaluation_blocks: Mapping[str, np.ndarray | None],
    *,
    candidate: str,
    alpha: float,
    k_over_a_ratio: float = 1.0,
    w_over_a_ratio: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if candidate not in JOINT_CANDIDATES:
        raise ValueError("Joint candidate set cannot contain AR-only or K-zero routes")
    if alpha < 0 or k_over_a_ratio <= 0 or w_over_a_ratio <= 0:
        raise ValueError("invalid Joint penalty")
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    rows = len(y)
    required = _required_blocks(candidate)
    train: dict[str, np.ndarray] = {}
    evaluation: dict[str, np.ndarray] = {}
    for block in required:
        train[block] = _as_matrix(train_blocks.get(block), rows)
        evaluation[block] = _as_matrix(
            evaluation_blocks.get(block),
            len(next(value for value in evaluation_blocks.values() if value is not None)),
        )
        if train[block].shape[1] == 0:
            raise ValueError(f"Joint candidate {candidate} has an empty required {block} block")
        if evaluation[block].shape[1] != train[block].shape[1]:
            raise ValueError(f"Joint train/evaluation {block} columns differ")
    evaluation_rows = len(evaluation[required[0]])
    if any(len(value) != evaluation_rows for value in evaluation.values()):
        raise ValueError("Joint evaluation blocks have different row counts")
    standardized_train = []
    standardized_evaluation = []
    penalty_diagonal = []
    block_contracts: dict[str, Any] = {}
    slices: dict[str, list[int]] = {}
    start = 0
    for block in required:
        block_train = train[block]
        mean = block_train.mean(axis=0, dtype=np.float64)
        scale = block_train.std(axis=0, dtype=np.float64)
        scale[scale * scale < 1e-12] = 1.0
        x = (block_train - mean) / scale
        z = (evaluation[block] - mean) / scale
        standardized_train.append(x)
        standardized_evaluation.append(z)
        ratio = k_over_a_ratio if block == "K" else w_over_a_ratio if block == "W" else 1.0
        penalty_diagonal.extend([float(alpha) * float(ratio)] * x.shape[1])
        stop = start + x.shape[1]
        slices[block] = [start, stop]
        block_contracts[block] = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "columns": x.shape[1],
            "penalty_ratio_to_a": float(ratio),
        }
        start = stop
    x = np.concatenate(standardized_train, axis=1)
    z = np.concatenate(standardized_evaluation, axis=1)
    target_mean = float(np.mean(y, dtype=np.float64))
    coefficient, certificate = solve_certified(
        x,
        y - target_mean,
        np.diag(np.asarray(penalty_diagonal, dtype=np.float64)),
    )
    total = z @ coefficient + target_mean
    components: dict[str, np.ndarray] = {}
    for block, (left, right) in slices.items():
        components[block] = z[:, left:right] @ coefficient[left:right]
    input_prediction = components["K"].copy()
    if "W" in components:
        input_prediction += components["W"]
    components["INPUT"] = input_prediction
    components["TOTAL"] = total
    contract = {
        "family": candidate,
        "blocks": block_contracts,
        "block_slices": slices,
        "coefficient": coefficient.tolist(),
        "intercept": target_mean,
        "alpha": float(alpha),
        "k_over_a_ratio": float(k_over_a_ratio),
        "w_over_a_ratio": float(w_over_a_ratio),
        "hard_feature_residualization": False,
        "soft_overlap_penalty": False,
        "input_path_required": True,
        "numerical_certificate": certificate.to_json(),
        "parameter_count": len(coefficient) + 1,
        "block_coefficient_norms": {
            block: float(np.linalg.norm(coefficient[left:right]))
            for block, (left, right) in slices.items()
        },
        "block_prediction_variance": {
            block: float(np.var(values, dtype=np.float64))
            for block, values in components.items()
            if block != "TOTAL"
        },
    }
    return total, contract, components


def predict_joint_candidate(
    blocks: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    candidate = str(contract["family"])
    required = _required_blocks(candidate)
    standardized = []
    rows: int | None = None
    for block in required:
        matrix = np.asarray(blocks[block], dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if rows is None:
            rows = len(matrix)
        elif len(matrix) != rows:
            raise ValueError("Joint prediction blocks have different row counts")
        block_contract = contract["blocks"][block]
        standardized.append(
            (matrix - np.asarray(block_contract["mean"], dtype=np.float64))
            / np.asarray(block_contract["scale"], dtype=np.float64)
        )
    design = np.concatenate(standardized, axis=1)
    coefficient = np.asarray(contract["coefficient"], dtype=np.float64)
    total = design @ coefficient + float(contract["intercept"])
    components: dict[str, np.ndarray] = {}
    for block, (left, right) in contract["block_slices"].items():
        components[block] = design[:, int(left):int(right)] @ coefficient[int(left):int(right)]
    components["INPUT"] = components["K"].copy()
    if "W" in components:
        components["INPUT"] += components["W"]
    components["TOTAL"] = total
    return total, components


def input_path_gate(
    ar_only_fold_losses: list[float],
    joint_fold_losses: list[float],
    input_prediction: np.ndarray,
    *,
    minimum_relative_gain: float = 0.01,
    minimum_positive_fraction: float = 0.75,
    numerical_floor: float = 1e-12,
) -> dict[str, Any]:
    ar = np.asarray(ar_only_fold_losses, dtype=np.float64)
    joint = np.asarray(joint_fold_losses, dtype=np.float64)
    mask = np.isfinite(ar) & np.isfinite(joint)
    if int(mask.sum()) < 3:
        return {
            "status": "JOINT_INPUT_PATH_COLLAPSED",
            "pass": False,
            "reason": "INSUFFICIENT_PAIRED_FOLDS",
        }
    ar = ar[mask]
    joint = joint[mask]
    denominator = max(float(np.mean(np.abs(ar))), numerical_floor)
    relative_gain = (float(np.mean(ar)) - float(np.mean(joint))) / denominator
    positive_fraction = float(np.mean(joint < ar))
    variance = float(np.var(np.asarray(input_prediction, dtype=np.float64), dtype=np.float64))
    passed = (
        relative_gain >= minimum_relative_gain
        and positive_fraction >= minimum_positive_fraction
        and variance > numerical_floor
    )
    return {
        "status": "JOINT_INPUT_PATH_VALIDATED" if passed else "JOINT_INPUT_PATH_COLLAPSED",
        "pass": bool(passed),
        "relative_gain_over_ar_only_diagnostic": relative_gain,
        "positive_fold_fraction": positive_fraction,
        "input_prediction_variance": variance,
        "finite_fold_count": int(mask.sum()),
    }


def run_joint_view(shared: "Path", project: "Path", output: "Path", view: Any) -> dict[str, Any]:
    import json
    import time
    import traceback
    from .cpu_data import BaseAccessor, load_samples, realized_state_profiles, sha256_file
    from .cpu_selection import mse, regression_metrics
    from .stage0 import write_json
    from .v2_c import fit_physical_features
    from .v2_k import _cap
    from .v2_selection import one_se_select
    from .v2_w import fit_c_fold_prediction
    from .v21_a import fit_mature_residual_ar
    from .v21_config import load_v21_and_v2_config
    from .v21_k import load_active_channels
    from .v21_selection import guarded_local_one_se_select
    from .v21_w import IDENTITY

    started = time.time()
    destination = output / "DEVELOPMENT" / "JOINT" / view.head.head_id / view.availability_scenario / view.proxy_policy
    destination.mkdir(parents=True, exist_ok=True)
    try:
        v21, v2 = load_v21_and_v2_config(project)
        c_result = json.loads((output / "DEVELOPMENT" / "C" / view.head.head_id / view.proxy_policy / "RESULT.json").read_text(encoding="utf-8"))
        w_result = json.loads((output / "DEVELOPMENT" / "W" / view.head.head_id / view.proxy_policy / "RESULT.json").read_text(encoding="utf-8"))
        a_result = json.loads((output / "DEVELOPMENT" / "A" / view.head.head_id / view.availability_scenario / view.proxy_policy / "RESULT.json").read_text(encoding="utf-8"))
        if any(item.get("status") != "PASS" for item in (c_result, w_result, a_result)):
            raise RuntimeError("E2-E4 prerequisite is not PASS")
        active = load_active_channels(output, view)
        oof = pd.read_parquet(output / w_result["oof_path"])
        w_contract = w_result["w_contract"]
        a_profile = tuple(a_result["a_contract"].get("profile", realized_state_profiles(view.head)[0]))
        alpha_grid = [float(value) for value in v2["J_module"]["ridge_alpha_grid"]]
        ratios_k = [float(value) for value in v21["J"]["k_over_a_ratio"]]
        ratios_w = [float(value) for value in v21["J"]["w_over_a_ratio"]]
        available_routes = [J_K, J_KA]
        if w_contract["family"] != IDENTITY:
            available_routes.extend([J_KW, J_KWA])
        candidates = [(route, alpha, rk, rw) for route in available_routes for alpha in alpha_grid for rk in ratios_k for rw in ratios_w]
        losses = {candidate: [] for candidate in candidates}
        ar_losses = {alpha: [] for alpha in alpha_grid}
        fold_payloads = []
        target_accessor = BaseAccessor(
            shared,
            view.head.dataset,
            "validation",
            [view.head.target],
        )
        usable_folds = sorted(int(value) for value in oof["oof_fold"].unique())[1:]
        for fold in usable_folds:
            fit = oof[oof["oof_fold"] < fold].reset_index(drop=True)
            evaluation = oof[oof["oof_fold"] == fold].reset_index(drop=True)
            features = fit_physical_features(shared, view, fit, evaluation, active, v2, fit_split="train", evaluation_split="train")
            k_train, k_eval = features["joint_train"], features["joint_evaluation"]
            if w_contract["family"] == IDENTITY:
                w_train, w_eval = np.empty((len(fit), 0)), np.empty((len(evaluation), 0))
            else:
                w_train, w_eval, _ = joint_w_basis(fit["physical_oof"].to_numpy(dtype=np.float64), evaluation["physical_oof"].to_numpy(dtype=np.float64), w_contract)
            delta, history = a_profile
            a_train = target_accessor.target_state(
                fit,
                view.head.target,
                delta,
                history,
            )
            a_eval = target_accessor.target_state(
                evaluation,
                view.head.target,
                delta,
                history,
            )
            target = fit["y_true"].to_numpy(dtype=np.float64)
            evaluation_target = evaluation["y_true"].to_numpy(dtype=np.float64)
            for candidate in candidates:
                route, alpha, rk, rw = candidate
                prediction, _, _ = fit_joint_candidate({"K": k_train, "W": w_train, "A": a_train}, target, {"K": k_eval, "W": w_eval, "A": a_eval}, candidate=route, alpha=alpha, k_over_a_ratio=rk, w_over_a_ratio=rw)
                losses[candidate].append(mse(evaluation_target, prediction))
            for alpha in alpha_grid:
                prediction, _ = fit_mature_residual_ar(a_train, target, a_eval, alpha=alpha, mu=0.0)
                ar_losses[alpha].append(mse(evaluation_target, prediction))
            fold_payloads.append((fit, evaluation, k_train, k_eval, w_train, w_eval, a_train, a_eval))
        route_best = {}
        for route in available_routes:
            subset = {key: value for key, value in losses.items() if key[0] == route}
            route_best[route] = one_se_select(subset, lambda value: (-value[1], abs(np.log(value[2])), abs(np.log(value[3]))), minimum_usable_folds=int(v21["selection"]["minimum_usable_folds"])).selected
        neutral = route_best[J_K]
        reduced = {candidate: losses[candidate] for candidate in route_best.values()}
        selection = guarded_local_one_se_select(reduced, lambda value: (JOINT_CANDIDATES.index(value[0]), -value[1], abs(np.log(value[2])), abs(np.log(value[3]))), neutral=neutral, minimum_relative_improvement=float(v21["selection"]["minimum_relative_improvement"]["J"]), minimum_positive_fraction=float(v21["selection"]["minimum_positive_fold_fraction"]), minimum_usable_folds=int(v21["selection"]["minimum_usable_folds"]))
        selected = selection.final_selected_candidate
        selected_route, selected_alpha, selected_rk, selected_rw = selected
        ar_selection = one_se_select(ar_losses, lambda value: (-value,), minimum_usable_folds=int(v21["selection"]["minimum_usable_folds"]))
        input_oof = []
        correlations = []
        for fit, evaluation, k_train, k_eval, w_train, w_eval, a_train, a_eval in fold_payloads:
            prediction, _, components = fit_joint_candidate({"K": k_train, "W": w_train, "A": a_train}, fit["y_true"].to_numpy(dtype=np.float64), {"K": k_eval, "W": w_eval, "A": a_eval}, candidate=selected_route, alpha=selected_alpha, k_over_a_ratio=selected_rk, w_over_a_ratio=selected_rw)
            input_oof.append(components["INPUT"])
            arrays = [components.get(name, np.zeros(len(prediction))) for name in ("K", "W", "A")]
            correlations.append(np.corrcoef(np.column_stack(arrays), rowvar=False).tolist())
        input_values = np.concatenate(input_oof)
        gate = input_path_gate(ar_losses[ar_selection.selected], selection.final_selected_fold_losses, input_values, minimum_relative_gain=float(v21["J"]["input_path_gain_over_ar_diagnostic_min"]), minimum_positive_fraction=float(v21["J"]["input_path_positive_fold_fraction_min"]))
        train = _cap(load_samples(shared, view, "train"), int(v2["row_caps"]["joint_predictive_fit"]))
        validation = load_samples(shared, view, "validation")
        final_features = fit_physical_features(shared, view, train, validation, active, v2, fit_split="train", evaluation_split="validation")
        k_train, k_validation = final_features["joint_train"], final_features["joint_evaluation"]
        fit_seed, validation_seed, _ = fit_c_fold_prediction(shared, view, train, validation, active, v2, c_result, fit_split="train", evaluation_split="validation")
        if w_contract["family"] == IDENTITY:
            w_train, w_validation = np.empty((len(train), 0)), np.empty((len(validation), 0))
        else:
            w_train, w_validation, _ = joint_w_basis(fit_seed, validation_seed, w_contract)
        delta, history = a_profile
        a_train = target_accessor.target_state(
            train,
            view.head.target,
            delta,
            history,
        )
        a_validation = target_accessor.target_state(
            validation,
            view.head.target,
            delta,
            history,
        )
        prediction, contract, components = fit_joint_candidate({"K": k_train, "W": w_train, "A": a_train}, train["y_true"].to_numpy(dtype=np.float64), {"K": k_validation, "W": w_validation, "A": a_validation}, candidate=selected_route, alpha=selected_alpha, k_over_a_ratio=selected_rk, w_over_a_ratio=selected_rw)
        frame = validation[["base_origin_id", "view_sample_id", "entity_id", "origin", "latest_available_target_index", "y_true"]].copy()
        frame["y_pred"], frame["input_prediction"] = prediction, components["INPUT"]
        frame["model"], frame["dtype"] = "PRISM_V2_1_JOINT_KWA", "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        result = {"status": "PASS" if gate["pass"] else "JOINT_INPUT_PATH_COLLAPSED", "stage": "E5_JOINT", "dataset": view.head.dataset, "target_head": view.head.head_id, "availability_scenario": view.availability_scenario, "proxy_policy": view.proxy_policy, "registered_candidates": list(JOINT_CANDIDATES), "applicable_candidates": available_routes, "route_local_selected": {route: list(candidate) for route, candidate in route_best.items()}, "ar_profile": list(a_profile), "selected_candidate": selected_route, "selection": selection.to_json(), "candidate_fold_losses": {str(key): value for key, value in losses.items()}, "ar_only_diagnostic": {"selected_alpha": ar_selection.selected, "fold_losses": ar_losses[ar_selection.selected]}, "input_path_gate": gate, "block_correlations_by_fold": correlations, "joint_contract": contract, "final_selected_candidate": selected_route, "final_selected_fold_losses": list(selection.final_selected_fold_losses), "final_selected_prediction_path": str(prediction_path.relative_to(output)), "final_selected_contract": contract, "final_prediction_loss": mse(frame["y_true"].to_numpy(dtype=np.float64), prediction), "prediction_path": str(prediction_path.relative_to(output)), "prediction_sha256": sha256_file(prediction_path), "test_accessed": False, "elapsed_seconds": time.time() - started, **regression_metrics(frame["y_true"].to_numpy(dtype=np.float64), prediction)}
    except Exception as error:
        result = {"status": "SOLVER_FAILED_RETAINED", "stage": "E5_JOINT", "target_head": view.head.head_id, "availability_scenario": view.availability_scenario, "proxy_policy": view.proxy_policy, "test_accessed": False, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started}
    write_json(destination / "RESULT.json", result)
    return result


def run_e5_joint(shared: "Path", project: "Path", output: "Path") -> dict[str, Any]:
    from .stage0 import write_json
    from .v21_assembly import build_joint_card
    from .v21_views import sru_dynamic_views
    results = []
    for view in sru_dynamic_views(shared):
        result = run_joint_view(shared, project, output, view)
        results.append(result)
        if result.get("status") in {"PASS", "JOINT_INPUT_PATH_COLLAPSED"}:
            card = build_joint_card(result)
            card.update({"target_head": view.head.head_id, "availability_scenario": view.availability_scenario, "proxy_policy": view.proxy_policy, "validation_prediction_path": result.get("prediction_path")})
            write_json(output / "ASSEMBLY_CARDS" / view.head.head_id / view.availability_scenario / view.proxy_policy / "JOINT_ASSEMBLY_CARD.json", card)
    summary = {"status": "PASS" if all(item["status"] == "PASS" for item in results) else "COMPLETED_WITH_RETAINED_FAILURES", "stage": "E5_JOINT", "views": len(results), "pass": sum(item["status"] == "PASS" for item in results), "collapsed": sum(item["status"] == "JOINT_INPUT_PATH_COLLAPSED" for item in results), "test_accessed": False}
    write_json(output / "DEVELOPMENT" / "JOINT" / "SUMMARY.json", summary)
    return summary
