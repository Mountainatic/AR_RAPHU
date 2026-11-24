"""Portable PRISM K/C/W/A/Joint final fitting and inference-only replay."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor, ViewSpec, sha256_file
from .level_reconstruction import metric_bundle_delta_and_level, support_hash
from .portable_checkpoints import (
    assert_fitting_allowed,
    assert_inference_only,
    checkpoint_key,
    load_portable_checkpoint,
    stable_hash,
    write_portable_checkpoint,
)
from .stage0 import write_json
from .v2_basis import natural_cubic_columns
from .v2_c import _ridge_fit, _ridge_predict, fit_physical_features
from .v2_config import load_frozen_config
from .v2_k import profile_values
from .v2_urysohn import basis_from_metadata, predict_contract
from .v211_a import EXACT_ZERO, fit_mature_residual_ar, mature_residual_features
from .v211_joint import joint_w_basis
from .v211_joint_stability import fit_joint_candidate_stability, k_representation_blocks
from .v211_k import load_active_channels
from .v211_public_all_baselines import SupportRequirement
from .v211_public_all_closure import common_support_record
from .v211_public_all_config import PublicAllPaths
from .v211_public_all_materialization import _prediction_frame, _prediction_root
from .v211_support import support_id_hash
from .v211_w import (
    IDENTITY,
    MONOTONE,
    NATURAL_CUBIC,
    _ispline_fixed,
    fit_w_correction,
    predict_w_correction,
)


INPUT_MODELS = ("PRISM_V2_1_1_K_C", "PRISM_V2_1_1_K_C_W")
DYNAMIC_MODELS = (
    "PRISM_V2_1_1_K_C_DYNAMIC",
    "PRISM_V2_1_1_K_C_W_DYNAMIC",
    "PRISM_V2_1_1_K_C_A_ABLATION",
    "PRISM_V2_1_1_PHYSICS_FIRST",
    "PRISM_V2_1_1_JOINT_KWA",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _read_pass(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("status") != "PASS":
        raise RuntimeError(f"PRISM development prerequisite is not PASS: {path}")
    return value


def _checkpoint_dir(root: Path, view: ViewSpec) -> Path:
    return root / "prism" / checkpoint_key(
        view.head.dataset,
        view.head.head_id,
        view.information_set,
        view.availability_scenario,
        view.proxy_policy,
    )


def _common_development(paths: PublicAllPaths, view: ViewSpec) -> pd.DataFrame:
    from .v211_public_all_materialization import _development

    record = common_support_record(paths, view)
    requirements = [SupportRequirement(**item) for item in record.get("requirements", [])]
    return _development(paths.shared, view, requirements or [SupportRequirement()])


def _development_results(paths: PublicAllPaths, view: ViewSpec) -> dict[str, dict[str, Any]]:
    c = _read_pass(
        paths.output / "DEVELOPMENT" / "C" / view.head.head_id / view.proxy_policy / "RESULT.json"
    )
    w = _read_pass(
        paths.output / "DEVELOPMENT" / "W" / view.head.head_id / view.proxy_policy / "RESULT.json"
    )
    result = {"c": c, "w": w}
    if view.information_set == "dynamic":
        result["a"] = _read_json(
            paths.output
            / "DEVELOPMENT"
            / "A"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "RESULT.json"
        )
        joint_path = (
            paths.output
            / "DEVELOPMENT"
            / "JOINT"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "RESULT.json"
        )
        result["joint"] = _read_json(joint_path) if joint_path.is_file() else {}
    return result


def _fit_c_state(
    paths: PublicAllPaths,
    view: ViewSpec,
    fit: pd.DataFrame,
    c_result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    assert_fitting_allowed()
    config = load_frozen_config(paths.project)
    allowed = set(c_result.get("active_channels", []))
    active = [item for item in load_active_channels(paths.output, view) if item.get("channel") in allowed]
    features = fit_physical_features(
        paths.shared,
        view,
        fit,
        fit,
        active,
        config,
        fit_split="validation",
        evaluation_split="validation",
    )
    family = str(c_result["selected_family"])
    target = fit["y_true"].to_numpy(dtype=np.float64)
    if family == "K_EXACT_ZERO" or not active:
        contract = {
            "family": "K_EXACT_ZERO",
            "coefficient": [],
            "intercept": float(target.mean(dtype=np.float64)),
            "parameter_count": 1,
        }
        prediction = np.full(len(fit), float(contract["intercept"]), dtype=np.float64)
    elif family == "BEST_ACTIVE_K":
        channel = str(c_result["best_active_k_channel"])
        prediction = features["compressed_train"][:, features["channels"].index(channel)].copy()
        contract = {
            "family": family,
            "channel": channel,
            "coefficient": [1.0],
            "intercept": 0.0,
            "parameter_count": 1,
        }
    else:
        key = "joint" if family == "ADDITIVE_JOINT_BASIS" else "compressed"
        matrix = np.asarray(features[f"{key}_train"], dtype=np.float64)
        prediction, contract = _ridge_fit(
            matrix, target, matrix, float(c_result["selected_alpha"])
        )
        contract = {"family": family, **contract}
    physical = {
        "channels": list(features["channels"]),
        "channel_contracts": features.get("channel_contracts", []),
        "global_joint_columns": features.get("global_joint_columns", []),
    }
    matrices = {
        "compressed": np.asarray(features["compressed_train"], dtype=np.float64),
        "joint": np.asarray(features["joint_train"], dtype=np.float64),
    }
    return physical, contract, prediction, matrices


def _predict_physical_features(
    paths: PublicAllPaths,
    view: ViewSpec,
    samples: pd.DataFrame,
    split: str,
    physical: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    channels = list(physical.get("channels", []))
    if not channels:
        return {
            "compressed": np.empty((len(samples), 0), dtype=np.float64),
            "joint": np.empty((len(samples), 0), dtype=np.float64),
        }
    accessor = BaseAccessor(paths.shared, view.head.dataset, split, channels)
    compressed: list[np.ndarray] = []
    joint: list[np.ndarray] = []
    for item in physical["channel_contracts"]:
        values, _ = profile_values(
            accessor,
            samples,
            str(item["channel"]),
            tuple(int(value) for value in item["profile"]),
            int(item["m_tau"]),
        )
        contract = item["k_contract"]
        compressed.append(predict_contract(values, contract))
        raw = basis_from_metadata(contract["basis"]).transform(values).reshape(len(values), -1)
        joint.append(raw[:, np.asarray(item["joint_columns"], dtype=np.int64)])
    compressed_matrix = np.column_stack(compressed)
    joint_matrix = np.concatenate(joint, axis=1)
    global_columns = np.asarray(physical.get("global_joint_columns", []), dtype=np.int64)
    if len(global_columns):
        joint_matrix = joint_matrix[:, global_columns]
    return {"compressed": compressed_matrix, "joint": joint_matrix}


def _predict_c(
    matrices: Mapping[str, np.ndarray], c_contract: Mapping[str, Any], channels: list[str]
) -> np.ndarray:
    family = str(c_contract["family"])
    rows = len(next(iter(matrices.values())))
    if family == "K_EXACT_ZERO":
        return np.full(rows, float(c_contract["intercept"]), dtype=np.float64)
    if family == "BEST_ACTIVE_K":
        return matrices["compressed"][:, channels.index(str(c_contract["channel"]))]
    key = "joint" if family == "ADDITIVE_JOINT_BASIS" else "compressed"
    return _ridge_predict(matrices[key], dict(c_contract))


def _w_basis_from_contract(latent: np.ndarray, contract: Mapping[str, Any]) -> np.ndarray:
    metadata = contract["basis"]
    standardized = (np.asarray(latent, dtype=np.float64) - float(metadata["mean"])) / float(metadata["scale"])
    knots = np.asarray(metadata["knots"], dtype=np.float64)
    if str(contract["family"]) == MONOTONE:
        return _ispline_fixed(
            standardized, knots, float(metadata["train_min"]), float(metadata["train_max"])
        )
    if str(contract["family"]) == NATURAL_CUBIC:
        return natural_cubic_columns(standardized, knots)[:, 2:]
    if str(contract["family"]) == IDENTITY:
        return np.empty((len(latent), 0), dtype=np.float64)
    raise ValueError(f"unsupported frozen W basis: {contract['family']}")


def _predict_joint(blocks: Mapping[str, np.ndarray], contract: Mapping[str, Any]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    # JSON checkpoint serialization sorts mapping keys. Reconstruct the fitted
    # design from the frozen slice offsets, not from post-reload key order.
    ordered_blocks = sorted(
        contract["block_slices"],
        key=lambda block: int(contract["block_slices"][block][0]),
    )
    for block in ordered_blocks:
        matrix = np.asarray(blocks[block], dtype=np.float64)
        block_contract = contract["blocks"][block]
        pieces.append(
            (matrix - np.asarray(block_contract["mean"], dtype=np.float64))
            / np.asarray(block_contract["scale"], dtype=np.float64)
        )
    design = np.concatenate(pieces, axis=1)
    return design @ np.asarray(contract["coefficient"], dtype=np.float64) + float(contract["intercept"])


def fit_prism_checkpoint_for_view(
    paths: PublicAllPaths, view: ViewSpec, checkpoint_root: Path
) -> dict[str, Any]:
    assert_fitting_allowed()
    fit = _common_development(paths, view)
    selected = _development_results(paths, view)
    physical, c_contract, c_seed, fit_matrices = _fit_c_state(
        paths, view, fit, selected["c"]
    )
    w_selected = selected["w"]["w_contract"]
    if w_selected["family"] == IDENTITY:
        w_contract = dict(w_selected)
        correction = np.zeros(len(fit), dtype=np.float64)
    else:
        correction, w_contract = fit_w_correction(
            c_seed,
            fit["y_true"].to_numpy(dtype=np.float64) - c_seed,
            c_seed,
            family=str(w_selected["family"]),
            knot_count=int(w_selected["knot_count"]),
            smoothness=float(w_selected["smoothness"]),
            mu=float(w_selected["soft_overlap_mu"]),
            upstream_predictions=fit_matrices["compressed"],
            direction=int(w_selected["direction"]),
        )
    state: dict[str, Any] = {
        "codec": "PRISM_KCWA_JOINT_PIPELINE",
        "artifact_type": "FORMAL_FINAL_MODEL",
        "family": "PRISM",
        "dataset": view.head.dataset,
        "task": view.head.task_id,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "physical": physical,
        "c_contract": c_contract,
        "w_contract": w_contract,
        "fit_partition": "train_plus_validation_common_support",
        "fit_rows": int(len(fit)),
        "fit_support_hash": support_id_hash(fit),
        "missing_value_policy": "REJECT_NONFINITE_C1",
        "feature_order": list(physical["channels"]),
        "selection_hash": stable_hash(selected),
        "models": list(
            INPUT_MODELS
            if view.information_set == "input_only"
            else DYNAMIC_MODELS[:4]
        ),
        "reload_prediction_tolerance": 1e-10,
    }
    fit_physical = c_seed + correction
    replay_rows = min(32, len(fit))
    replay: dict[str, np.ndarray] = {
        "compressed": fit_matrices["compressed"][:replay_rows],
        "joint": fit_matrices["joint"][:replay_rows],
        "c_prediction": c_seed[:replay_rows],
        "w_prediction": fit_physical[:replay_rows],
    }
    if view.information_set == "dynamic":
        a_selected = selected["a"]
        a_hyper = a_selected.get("a_contract", {})
        if a_selected.get("status") != "PASS" or a_hyper.get("family") == EXACT_ZERO:
            state["a_contract"] = {"family": EXACT_ZERO, "parameter_count": 0}
            state["ka_contract"] = {"family": EXACT_ZERO, "parameter_count": 0}
        else:
            delta, history = (int(value) for value in a_hyper["profile"])
            config = load_frozen_config(paths.project)
            feature_kwargs = {
                "h_steps": view.head.h_steps,
                "w_steps": view.head.w_steps,
                "delta": delta,
                "history": history,
                "maximum_lags": int(config["A_module"]["state_profile"]["maximum_lags"]),
            }
            residual_source = fit[["entity_id", "origin"]].assign(
                residual=fit["y_true"].to_numpy(dtype=np.float64) - fit_physical
            )
            x_a, _, causal_audit = mature_residual_features(
                fit,
                residual_source,
                **{
                    **feature_kwargs,
                    "residual_mean": float(residual_source["residual"].mean()),
                },
            )
            a_fit_prediction, a_contract = fit_mature_residual_ar(
                x_a,
                residual_source["residual"].to_numpy(dtype=np.float64),
                x_a,
                alpha=float(a_hyper["alpha"]),
                mu=float(a_hyper["soft_overlap_mu"]),
                upstream_predictions=np.column_stack([fit_matrices["compressed"], correction]),
            )
            k_residual_source = fit[["entity_id", "origin"]].assign(
                residual=fit["y_true"].to_numpy(dtype=np.float64) - c_seed
            )
            x_ka, _, _ = mature_residual_features(
                fit,
                k_residual_source,
                **{
                    **feature_kwargs,
                    "residual_mean": float(k_residual_source["residual"].mean()),
                },
            )
            ka_fit_prediction, ka_contract = fit_mature_residual_ar(
                x_ka,
                k_residual_source["residual"].to_numpy(dtype=np.float64),
                x_ka,
                alpha=float(a_hyper["alpha"]),
                mu=float(a_hyper["soft_overlap_mu"]),
                upstream_predictions=fit_matrices["compressed"],
            )
            state.update(
                {
                    "a_profile": [delta, history],
                    "a_maximum_lags": int(feature_kwargs["maximum_lags"]),
                    "a_residual_mean": float(residual_source["residual"].mean()),
                    "ka_residual_mean": float(k_residual_source["residual"].mean()),
                    "a_contract": a_contract,
                    "ka_contract": ka_contract,
                    "a_causal_feature_audit": causal_audit,
                }
            )
            replay.update(
                {
                    "a_features": x_a[:replay_rows],
                    "a_prediction": a_fit_prediction[:replay_rows],
                    "ka_features": x_ka[:replay_rows],
                    "ka_prediction": ka_fit_prediction[:replay_rows],
                }
            )
        joint_selected = selected["joint"]
        if joint_selected.get("status") == "PASS" and joint_selected.get("ar_profile") is not None:
            contract = joint_selected["joint_contract"]
            raw_support = list(contract["raw_k_support"])
            if raw_support != list(physical["channels"]):
                raise RuntimeError("STOP_JOINT_RAW_K_SUPPORT_DIFFERS_FROM_FINAL_C_SUPPORT")
            representations, _ = k_representation_blocks(
                {
                    "channels": physical["channels"],
                    "compressed_train": fit_matrices["compressed"],
                    "compressed_evaluation": fit_matrices["compressed"],
                    "joint_train": fit_matrices["joint"],
                    "joint_evaluation": fit_matrices["joint"],
                },
                raw_support,
            )
            k_fit = representations[str(contract["k_representation"])][0]
            joint_w_selected = selected["w"].get("joint_w_basis_contract", {})
            w_fit, _, w_metadata = joint_w_basis(c_seed, c_seed, joint_w_selected)
            target_accessor = BaseAccessor(
                paths.shared, view.head.dataset, "validation", [view.head.target]
            )
            ar_profile = tuple(int(value) for value in joint_selected["ar_profile"])
            a_fit = target_accessor.target_state(fit, view.head.target, *ar_profile)
            joint_fit_prediction, joint_contract, _ = fit_joint_candidate_stability(
                {"K": k_fit, "W": w_fit, "A": a_fit},
                fit["y_true"].to_numpy(dtype=np.float64),
                {"K": k_fit, "W": w_fit, "A": a_fit},
                candidate=str(contract["family"]),
                k_representation=str(contract["k_representation"]),
                numerical_alpha=float(contract["numerical_alpha"]),
                predictive_eta=float(contract["predictive_eta"]),
                raw_k_support=raw_support,
            )
            state.update(
                {
                    "joint_status": "PASS",
                    "joint_contract": joint_contract,
                    "joint_ar_profile": list(ar_profile),
                    "joint_w_contract": {**dict(joint_w_selected), "basis": w_metadata},
                }
            )
            state["models"].append(DYNAMIC_MODELS[4])
            replay.update(
                {
                    "joint_k": k_fit[:replay_rows],
                    "joint_w": w_fit[:replay_rows],
                    "joint_a": a_fit[:replay_rows],
                    "joint_prediction": joint_fit_prediction[:replay_rows],
                }
            )
        else:
            state["joint_status"] = "NOT_RUN_PROTOCOL_INCOMPATIBLE"
            state["joint_reason"] = str(
                joint_selected.get("reason", "DEVELOPMENT_JOINT_NOT_PASS")
            )
    manifest = write_portable_checkpoint(
        _checkpoint_dir(checkpoint_root, view), state, replay
    )
    return {
        "status": "PASS",
        "family": "PRISM",
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "models": state["models"],
        "fit_rows": int(len(fit)),
        "fit_support_hash": support_id_hash(fit),
        "checkpoint_dir": str(_checkpoint_dir(checkpoint_root, view)),
        "checkpoint_hash": manifest["checkpoint_hash"],
    }


def _current_levels(paths: PublicAllPaths, view: ViewSpec, samples: pd.DataFrame, split: str) -> np.ndarray:
    if "current_level" in samples.columns:
        return samples["current_level"].to_numpy(dtype=np.float64)
    accessor = BaseAccessor(paths.shared, view.head.dataset, split, [view.head.target])
    return accessor.block_means(samples, view.head.target, [(0, int(view.head.w0_steps))]).reshape(-1)


def _write_model(
    paths: PublicAllPaths,
    view: ViewSpec,
    samples: pd.DataFrame,
    model: str,
    prediction: np.ndarray,
    parameter_count: int,
    checkpoint: Path,
    checkpoint_hash: str,
    fit_support_hash: str,
    started: float,
    split: str,
) -> dict[str, Any]:
    frame = _prediction_frame(samples, view, model, prediction, parameter_count)
    frame["split"] = split
    destination = _prediction_root(paths, split) / view.relative_root / f"{model}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False, compression="zstd")
    metrics = metric_bundle_delta_and_level(
        frame["y_true"].to_numpy(dtype=np.float64),
        frame["y_pred"].to_numpy(dtype=np.float64),
        _current_levels(paths, view, samples, split),
    )
    metrics.pop("future_level_true")
    metrics.pop("future_level_pred")
    return {
        "status": "PASS",
        "dataset": view.head.dataset,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "model": model,
        "split": split,
        "rows": int(len(frame)),
        "sample_id_order_hash": support_hash(frame["sample_id"].astype(str).tolist()),
        "scoring_support_hash": support_id_hash(samples),
        "fit_support_hash": fit_support_hash,
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_dir": str(checkpoint),
        "prediction_path": str(destination.relative_to(paths.run_root)),
        "prediction_sha256": sha256_file(destination),
        "test_accessed": split == "test",
        "ood_accessed": split == "ood",
        "fit_called_in_inference": False,
        "elapsed_seconds": time.time() - started,
        **metrics,
    }


def verify_prism_checkpoint_reload(checkpoint: Path) -> dict[str, Any]:
    assert_inference_only()
    state, arrays, manifest = load_portable_checkpoint(checkpoint)
    matrices = {"compressed": arrays["compressed"], "joint": arrays["joint"]}
    c_prediction = _predict_c(
        matrices, state["c_contract"], list(state["physical"]["channels"])
    )
    w_prediction = c_prediction + predict_w_correction(c_prediction, state["w_contract"])
    errors = {
        "c": float(np.max(np.abs(c_prediction - arrays["c_prediction"]), initial=0.0)),
        "w": float(np.max(np.abs(w_prediction - arrays["w_prediction"]), initial=0.0)),
    }
    if "a_features" in arrays:
        from .v211_a import predict_mature_residual_ar

        errors["a"] = float(
            np.max(
                np.abs(
                    predict_mature_residual_ar(arrays["a_features"], state["a_contract"])
                    - arrays["a_prediction"]
                ),
                initial=0.0,
            )
        )
        errors["ka"] = float(
            np.max(
                np.abs(
                    predict_mature_residual_ar(arrays["ka_features"], state["ka_contract"])
                    - arrays["ka_prediction"]
                ),
                initial=0.0,
            )
        )
    if "joint_k" in arrays:
        observed = _predict_joint(
            {"K": arrays["joint_k"], "W": arrays["joint_w"], "A": arrays["joint_a"]},
            state["joint_contract"],
        )
        errors["joint"] = float(
            np.max(np.abs(observed - arrays["joint_prediction"]), initial=0.0)
        )
    maximum = max(errors.values(), default=0.0)
    tolerance = float(state.get("reload_prediction_tolerance", 1e-10))
    if maximum > tolerance:
        raise RuntimeError(f"STOP_PRISM_CHECKPOINT_RELOAD_MISMATCH:{checkpoint}:{maximum}")
    return {
        "status": "PASS",
        "checkpoint_dir": str(checkpoint),
        "checkpoint_hash": manifest["checkpoint_hash"],
        "maximum_absolute_prediction_error": maximum,
        "component_errors": errors,
        "tolerance": tolerance,
    }


def predict_prism_checkpoint_for_view(
    paths: PublicAllPaths,
    view: ViewSpec,
    checkpoint_root: Path,
    *,
    split: str = "test",
) -> list[dict[str, Any]]:
    assert_inference_only()
    started = time.time()
    checkpoint = _checkpoint_dir(checkpoint_root, view)
    state, _, manifest = load_portable_checkpoint(checkpoint)
    from .v211_public_all_baseline_materialization import _common_test

    samples = _common_test(paths, view, split)
    fit = _common_development(paths, view)
    fit_matrices = _predict_physical_features(paths, view, fit, "validation", state["physical"])
    test_matrices = _predict_physical_features(paths, view, samples, split, state["physical"])
    fit_seed = _predict_c(fit_matrices, state["c_contract"], list(state["physical"]["channels"]))
    test_seed = _predict_c(test_matrices, state["c_contract"], list(state["physical"]["channels"]))
    fit_correction = predict_w_correction(fit_seed, state["w_contract"])
    test_correction = predict_w_correction(test_seed, state["w_contract"])
    fit_physical = fit_seed + fit_correction
    test_physical = test_seed + test_correction
    predictions: dict[str, tuple[np.ndarray, int]] = {
        INPUT_MODELS[0] if view.information_set == "input_only" else DYNAMIC_MODELS[0]: (
            test_seed,
            int(state["c_contract"].get("parameter_count", 0)),
        ),
        INPUT_MODELS[1] if view.information_set == "input_only" else DYNAMIC_MODELS[1]: (
            test_physical,
            int(state["c_contract"].get("parameter_count", 0))
            + int(state["w_contract"].get("parameter_count", 0)),
        ),
    }
    causal_audit: dict[str, Any] = {"past_test_truth_allowed": False}
    if view.information_set == "dynamic":
        if state["a_contract"]["family"] == EXACT_ZERO:
            a_prediction = np.zeros(len(samples), dtype=np.float64)
            ka_prediction = np.zeros(len(samples), dtype=np.float64)
        else:
            delta, history = (int(value) for value in state["a_profile"])
            kwargs = {
                "h_steps": view.head.h_steps,
                "w_steps": view.head.w_steps,
                "delta": delta,
                "history": history,
                "maximum_lags": int(state["a_maximum_lags"]),
            }
            fit_residuals = fit[["entity_id", "origin"]].assign(
                residual=fit["y_true"].to_numpy(dtype=np.float64) - fit_physical
            )
            test_residuals = samples[["entity_id", "origin"]].assign(
                residual=samples["y_true"].to_numpy(dtype=np.float64) - test_physical
            )
            source = pd.concat([fit_residuals, test_residuals], ignore_index=True)
            x_a, _, causal_audit = mature_residual_features(
                samples, source, **{**kwargs, "residual_mean": float(state["a_residual_mean"])}
            )
            from .v211_a import predict_mature_residual_ar

            a_prediction = predict_mature_residual_ar(x_a, state["a_contract"])
            fit_k_residuals = fit[["entity_id", "origin"]].assign(
                residual=fit["y_true"].to_numpy(dtype=np.float64) - fit_seed
            )
            test_k_residuals = samples[["entity_id", "origin"]].assign(
                residual=samples["y_true"].to_numpy(dtype=np.float64) - test_seed
            )
            k_source = pd.concat([fit_k_residuals, test_k_residuals], ignore_index=True)
            x_ka, _, _ = mature_residual_features(
                samples, k_source, **{**kwargs, "residual_mean": float(state["ka_residual_mean"])}
            )
            ka_prediction = predict_mature_residual_ar(x_ka, state["ka_contract"])
            causal_audit["past_test_truth_allowed"] = True
            causal_audit["current_or_future_test_truth_used"] = False
        predictions[DYNAMIC_MODELS[2]] = (
            test_seed + ka_prediction,
            int(state["c_contract"].get("parameter_count", 0))
            + int(state["ka_contract"].get("parameter_count", 0)),
        )
        predictions[DYNAMIC_MODELS[3]] = (
            test_physical + a_prediction,
            int(state["c_contract"].get("parameter_count", 0))
            + int(state["w_contract"].get("parameter_count", 0))
            + int(state["a_contract"].get("parameter_count", 0)),
        )
        if state.get("joint_status") == "PASS":
            representation = str(state["joint_contract"]["k_representation"])
            k_block = test_matrices["compressed"] if representation == "CHANNEL_COMPRESSED" else test_matrices["joint"]
            w_block = _w_basis_from_contract(test_seed, state["joint_w_contract"])
            target_accessor = BaseAccessor(
                paths.shared, view.head.dataset, split, [view.head.target]
            )
            a_block = target_accessor.target_state(
                samples, view.head.target, *tuple(state["joint_ar_profile"])
            )
            joint_prediction = _predict_joint(
                {"K": k_block, "W": w_block, "A": a_block}, state["joint_contract"]
            )
            predictions[DYNAMIC_MODELS[4]] = (
                joint_prediction, int(state["joint_contract"]["parameter_count"])
            )
    records = [
        _write_model(
            paths,
            view,
            samples,
            model,
            prediction,
            parameter_count,
            checkpoint,
            manifest["checkpoint_hash"],
            str(state["fit_support_hash"]),
            started,
            split,
        )
        for model, (prediction, parameter_count) in predictions.items()
    ]
    if view.information_set == "dynamic" and state.get("joint_status") != "PASS":
        records.append(
            {
                "status": "NOT_RUN_PROTOCOL_INCOMPATIBLE",
                "model": DYNAMIC_MODELS[4],
                "reason": state.get("joint_reason", "DEVELOPMENT_JOINT_NOT_PASS"),
                "test_accessed": False,
            }
        )
    write_json(
        _prediction_root(paths, split) / view.relative_root / "PRISM_INFERENCE_RESULT.json",
        {
            "status": "PASS",
            "models": records,
            "inference_only": True,
            "causal_state_audit": causal_audit,
        },
    )
    return records
