from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor, ViewSpec, sha256_file
from .cpu_selection import regression_metrics
from .stage0 import write_json
from .v2_config import load_frozen_config
from .v2_c import fit_physical_features
from .v211_a import EXACT_ZERO, fit_mature_residual_ar, mature_residual_features
from .v211_k import load_active_channels
from .v211_public_all_baselines import SupportRequirement, apply_common_requirements
from .v211_public_all_closure import common_support_record
from .v211_public_all_config import PublicAllPaths
from .v211_support import load_native_samples, support_id_hash
from .v211_w import IDENTITY, _fit_c_routed, fit_w_correction
from .v211_joint import fit_joint_candidate, joint_w_basis


def _read_pass(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError(f"prerequisite is not PASS: {path}")
    return value


def _requirements(values: Iterable[Mapping[str, Any]]) -> tuple[SupportRequirement, ...]:
    result = tuple(SupportRequirement(**dict(value)) for value in values)
    return result or (SupportRequirement(),)


def _filtered_split(
    shared: Path,
    view: ViewSpec,
    split: str,
    requirements: Iterable[SupportRequirement],
) -> pd.DataFrame:
    samples = load_native_samples(shared, view, split)
    return apply_common_requirements(samples, requirements).reset_index(drop=True)


def _development(
    shared: Path,
    view: ViewSpec,
    requirements: Iterable[SupportRequirement],
) -> pd.DataFrame:
    return (
        pd.concat(
            [
                _filtered_split(shared, view, "train", requirements),
                _filtered_split(shared, view, "validation", requirements),
            ],
            ignore_index=True,
        )
        .sort_values(["entity_id", "origin"])
        .reset_index(drop=True)
    )


def _common_evaluation(paths: PublicAllPaths, view: ViewSpec, split: str) -> pd.DataFrame:
    record = common_support_record(paths, view)
    requirements = _requirements(record.get("requirements", ()))
    return _filtered_split(paths.shared, view, split, requirements)


def _prediction_frame(
    samples: pd.DataFrame,
    view: ViewSpec,
    model: str,
    prediction: np.ndarray,
    parameter_count: int,
) -> pd.DataFrame:
    frame = samples[
        [
            "view_sample_id",
            "base_origin_id",
            "dataset",
            "entity_id",
            "task_id",
            "target_head",
            "split",
            "origin",
            "y_true",
        ]
    ].copy()
    frame = frame.rename(columns={"view_sample_id": "sample_id"})
    frame["y_pred"] = np.asarray(prediction, dtype=np.float64)
    frame["model"] = model
    frame["information_set"] = view.information_set
    frame["availability_scenario"] = view.availability_scenario
    frame["proxy_policy"] = view.proxy_policy
    frame["parameter_count"] = int(parameter_count)
    frame["dtype"] = "float64"
    return frame


def _prediction_root(paths: PublicAllPaths, split: str, *, baseline: bool = False) -> Path:
    if split not in {"test", "ood"}:
        raise ValueError(f"final materialization split must be test or ood: {split}")
    prefix = "baseline_" if baseline else ""
    return paths.final / f"{prefix}{split}_predictions"


def _prior_test_residuals(
    paths: PublicAllPaths,
    view: ViewSpec,
    split: str,
    model: str,
) -> pd.DataFrame:
    if split != "ood":
        return pd.DataFrame(columns=["entity_id", "origin", "residual"])
    source = _prediction_root(paths, "test") / view.relative_root / f"{model}.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"OOD state requires frozen test predictions: {source}")
    frame = pd.read_parquet(
        source,
        columns=["entity_id", "origin", "split", "y_true", "y_pred"],
    )
    if set(frame["split"].astype(str).unique()) != {"test"}:
        raise RuntimeError("prior OOD state source is not isolated to test")
    return frame[["entity_id", "origin"]].assign(
        residual=frame["y_true"].to_numpy(dtype=np.float64)
        - frame["y_pred"].to_numpy(dtype=np.float64)
    )


def _write_prediction(
    paths: PublicAllPaths,
    view: ViewSpec,
    model: str,
    samples: pd.DataFrame,
    prediction: np.ndarray,
    parameter_count: int,
    started: float,
    native_fit: pd.DataFrame,
    *,
    split: str,
) -> dict[str, Any]:
    if set(samples["split"].astype(str).unique()) != {split}:
        raise RuntimeError(f"prediction samples are not isolated to {split}")
    destination = _prediction_root(paths, split) / view.relative_root / f"{model}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = _prediction_frame(
        samples, view, model, prediction, parameter_count
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
        "prediction_path": str(destination.relative_to(paths.run_root)),
        "prediction_sha256": sha256_file(destination),
        "fit_and_prediction_seconds": time.time() - started,
        "test_accessed": split == "test",
        "ood_accessed": split == "ood",
        **regression_metrics(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        ),
    }


def materialize_input_prism_view(
    paths: PublicAllPaths, view: ViewSpec, *, split: str = "test"
) -> list[dict[str, Any]]:
    started = time.time()
    c_path = (
        paths.output
        / "DEVELOPMENT"
        / "C"
        / view.head.head_id
        / view.proxy_policy
        / "RESULT.json"
    )
    w_path = (
        paths.output
        / "DEVELOPMENT"
        / "W"
        / view.head.head_id
        / view.proxy_policy
        / "RESULT.json"
    )
    c = _read_pass(c_path)
    w = _read_pass(w_path)
    active = [
        item
        for item in load_active_channels(paths.output, view)
        if item.get("channel") in set(c.get("active_channels", ()))
    ]
    histories = [
        int(value)
        for value in c.get("active_selected_k_histories", {}).values()
    ]
    native_requirements = tuple(
        SupportRequirement(input_history_steps=value) for value in histories
    ) or (SupportRequirement(),)
    development = _development(paths.shared, view, native_requirements)
    evaluation = _common_evaluation(paths, view, split)
    config = load_frozen_config(paths.project)
    development_seed, evaluation_seed, development_upstream, _, _ = _fit_c_routed(
        paths.shared,
        view,
        development,
        evaluation,
        active,
        config,
        c,
        fit_split="validation",
        evaluation_split=split,
    )
    w_contract = w["w_contract"]
    if w_contract["family"] == IDENTITY:
        correction = np.zeros(len(evaluation), dtype=np.float64)
    else:
        correction, _ = fit_w_correction(
            development_seed,
            development["y_true"].to_numpy(dtype=np.float64) - development_seed,
            evaluation_seed,
            family=w_contract["family"],
            knot_count=int(w_contract["knot_count"]),
            smoothness=float(w_contract["smoothness"]),
            mu=float(w_contract["soft_overlap_mu"]),
            upstream_predictions=development_upstream,
            direction=int(w_contract["direction"]),
        )
    c_parameters = int(c.get("fusion_contract", {}).get("parameter_count", 0))
    w_parameters = int(w_contract.get("parameter_count", 0))
    audits = [
        _write_prediction(
            paths,
            view,
            "PRISM_V2_1_1_K_C",
            evaluation,
            evaluation_seed,
            c_parameters,
            started,
            development,
            split=split,
        ),
        _write_prediction(
            paths,
            view,
            "PRISM_V2_1_1_K_C_W",
            evaluation,
            evaluation_seed + correction,
            c_parameters + w_parameters,
            started,
            development,
            split=split,
        ),
    ]
    write_json(
        _prediction_root(paths, split) / view.relative_root / "PRISM_INPUT_RESULT.json",
        {
            "status": "PASS",
            "models": audits,
            "test_accessed": split == "test",
            "ood_accessed": split == "ood",
        },
    )
    return audits


def _fit_c_w(
    paths: PublicAllPaths,
    view: ViewSpec,
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    active: list[dict[str, Any]],
    config: dict[str, Any],
    c: dict[str, Any],
    w: dict[str, Any],
    *,
    evaluation_split: str,
) -> dict[str, Any]:
    fit_seed, evaluation_seed, fit_upstream, evaluation_upstream, _ = _fit_c_routed(
        paths.shared,
        view,
        fit,
        evaluation,
        active,
        config,
        c,
        fit_split="validation",
        evaluation_split=evaluation_split,
    )
    w_contract = w["w_contract"]
    if w_contract["family"] == IDENTITY:
        fit_correction = np.zeros(len(fit), dtype=np.float64)
        evaluation_correction = np.zeros(len(evaluation), dtype=np.float64)
    else:
        fit_correction, _ = fit_w_correction(
            fit_seed,
            fit["y_true"].to_numpy(dtype=np.float64) - fit_seed,
            fit_seed,
            family=w_contract["family"],
            knot_count=int(w_contract["knot_count"]),
            smoothness=float(w_contract["smoothness"]),
            mu=float(w_contract["soft_overlap_mu"]),
            upstream_predictions=fit_upstream,
            direction=int(w_contract["direction"]),
        )
        evaluation_correction, _ = fit_w_correction(
            fit_seed,
            fit["y_true"].to_numpy(dtype=np.float64) - fit_seed,
            evaluation_seed,
            family=w_contract["family"],
            knot_count=int(w_contract["knot_count"]),
            smoothness=float(w_contract["smoothness"]),
            mu=float(w_contract["soft_overlap_mu"]),
            upstream_predictions=fit_upstream,
            direction=int(w_contract["direction"]),
        )
    return {
        "fit_seed": fit_seed,
        "evaluation_seed": evaluation_seed,
        "fit_upstream": fit_upstream,
        "evaluation_upstream": evaluation_upstream,
        "fit_correction": fit_correction,
        "evaluation_correction": evaluation_correction,
        "fit_physical": fit_seed + fit_correction,
        "evaluation_physical": evaluation_seed + evaluation_correction,
    }


def _target_requirement(profile: Any) -> SupportRequirement | None:
    if profile is None:
        return None
    delta, history = (int(value) for value in profile)
    if delta <= 0 or history <= 0:
        return None
    return SupportRequirement(
        target_delta_steps=delta,
        target_history_steps=history,
    )


def materialize_dynamic_prism_view(
    paths: PublicAllPaths, view: ViewSpec, *, split: str = "test"
) -> list[dict[str, Any]]:
    started = time.time()
    c = _read_pass(
        paths.output
        / "DEVELOPMENT"
        / "C"
        / view.head.head_id
        / view.proxy_policy
        / "RESULT.json"
    )
    w = _read_pass(
        paths.output
        / "DEVELOPMENT"
        / "W"
        / view.head.head_id
        / view.proxy_policy
        / "RESULT.json"
    )
    a = json.loads(
        (
            paths.output
            / "DEVELOPMENT"
            / "A"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "RESULT.json"
        ).read_text(encoding="utf-8")
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
    joint = json.loads(joint_path.read_text(encoding="utf-8")) if joint_path.is_file() else {}
    active = [
        item
        for item in load_active_channels(paths.output, view)
        if item.get("channel") in set(c.get("active_channels", ()))
    ]
    histories = [
        int(value)
        for value in c.get("active_selected_k_histories", {}).values()
    ]
    assembly_requirements = tuple(
        SupportRequirement(input_history_steps=value) for value in histories
    ) or (SupportRequirement(),)
    assembly = _development(paths.shared, view, assembly_requirements)
    evaluation = _common_evaluation(paths, view, split)
    config = load_frozen_config(paths.project)
    c_w_test = _fit_c_w(
        paths,
        view,
        assembly,
        evaluation,
        active,
        config,
        c,
        w,
        evaluation_split=split,
    )
    audits: list[dict[str, Any]] = []
    c_parameters = int(c.get("fusion_contract", {}).get("parameter_count", 0))
    w_contract = w["w_contract"]
    w_parameters = int(w_contract.get("parameter_count", 0))
    audits.append(
        _write_prediction(
            paths,
            view,
            "PRISM_V2_1_1_K_C_DYNAMIC",
            evaluation,
            c_w_test["evaluation_seed"],
            c_parameters,
            started,
            assembly,
            split=split,
        )
    )
    audits.append(
        _write_prediction(
            paths,
            view,
            "PRISM_V2_1_1_K_C_W_DYNAMIC",
            evaluation,
            c_w_test["evaluation_physical"],
            c_parameters + w_parameters,
            started,
            assembly,
            split=split,
        )
    )

    a_contract = a.get("a_contract", {})
    a_requirement = (
        None
        if a_contract.get("family") == EXACT_ZERO
        else _target_requirement(a_contract.get("profile"))
    )
    a_development = (
        assembly
        if a_requirement is None
        else apply_common_requirements(assembly, [a_requirement]).reset_index(drop=True)
    )
    prior_test_residual_rows = 0
    if a_requirement is None:
        pf_prediction = c_w_test["evaluation_physical"]
        k_a_prediction = c_w_test["evaluation_seed"]
    else:
        c_w_a = _fit_c_w(
            paths,
            view,
            assembly,
            a_development,
            active,
            config,
            c,
            w,
            evaluation_split="validation",
        )
        prior_w_residuals = _prior_test_residuals(
            paths,
            view,
            split,
            "PRISM_V2_1_1_K_C_W_DYNAMIC",
        )
        prior_test_residual_rows = len(prior_w_residuals)
        residual_source = pd.concat(
            [
                assembly[["entity_id", "origin"]].assign(
                    residual=assembly["y_true"].to_numpy(dtype=np.float64)
                    - c_w_test["fit_physical"]
                ),
                prior_w_residuals,
                evaluation[["entity_id", "origin"]].assign(
                    residual=evaluation["y_true"].to_numpy(dtype=np.float64)
                    - c_w_test["evaluation_physical"]
                ),
            ],
            ignore_index=True,
        )
        delta, history = (int(value) for value in a_contract["profile"])
        residual_mean = float(
            np.mean(
                assembly["y_true"].to_numpy(dtype=np.float64)
                - c_w_test["fit_physical"]
            )
        )
        feature_kwargs = {
            "h_steps": view.head.h_steps,
            "w_steps": view.head.w_steps,
            "delta": delta,
            "history": history,
            "maximum_lags": int(config["A_module"]["state_profile"]["maximum_lags"]),
            "residual_mean": residual_mean,
        }
        a_x_fit, _, _ = mature_residual_features(
            a_development, residual_source, **feature_kwargs
        )
        a_x_eval, _, _ = mature_residual_features(
            evaluation, residual_source, **feature_kwargs
        )
        a_upstream_fit = np.column_stack(
            [c_w_a["evaluation_upstream"], c_w_a["evaluation_correction"]]
        )
        a_prediction, _ = fit_mature_residual_ar(
            a_x_fit,
            a_development["y_true"].to_numpy(dtype=np.float64)
            - c_w_a["evaluation_physical"],
            a_x_eval,
            alpha=float(a_contract["alpha"]),
            mu=float(a_contract["soft_overlap_mu"]),
            upstream_predictions=a_upstream_fit,
        )
        pf_prediction = c_w_test["evaluation_physical"] + a_prediction
        prior_k_residuals = _prior_test_residuals(
            paths,
            view,
            split,
            "PRISM_V2_1_1_K_C_DYNAMIC",
        )
        if len(prior_k_residuals) != prior_test_residual_rows:
            raise RuntimeError("OOD K and KW prior-test state rows disagree")
        k_residual_source = pd.concat(
            [
                assembly[["entity_id", "origin"]].assign(
                    residual=assembly["y_true"].to_numpy(dtype=np.float64)
                    - c_w_test["fit_seed"]
                ),
                prior_k_residuals,
                evaluation[["entity_id", "origin"]].assign(
                    residual=evaluation["y_true"].to_numpy(dtype=np.float64)
                    - c_w_test["evaluation_seed"]
                ),
            ],
            ignore_index=True,
        )
        k_residual_mean = float(
            np.mean(
                assembly["y_true"].to_numpy(dtype=np.float64)
                - c_w_test["fit_seed"]
            )
        )
        k_a_x_fit, _, _ = mature_residual_features(
            a_development,
            k_residual_source,
            **{**feature_kwargs, "residual_mean": k_residual_mean},
        )
        k_a_x_eval, _, _ = mature_residual_features(
            evaluation,
            k_residual_source,
            **{**feature_kwargs, "residual_mean": k_residual_mean},
        )
        k_a, _ = fit_mature_residual_ar(
            k_a_x_fit,
            a_development["y_true"].to_numpy(dtype=np.float64)
            - c_w_a["evaluation_seed"],
            k_a_x_eval,
            alpha=float(a_contract["alpha"]),
            mu=float(a_contract["soft_overlap_mu"]),
            upstream_predictions=c_w_a["evaluation_upstream"],
        )
        k_a_prediction = c_w_test["evaluation_seed"] + k_a
    a_parameters = int(a_contract.get("parameter_count", 0))
    audits.append(
        _write_prediction(
            paths,
            view,
            "PRISM_V2_1_1_K_C_A_ABLATION",
            evaluation,
            k_a_prediction,
            c_parameters + a_parameters,
            started,
            a_development,
            split=split,
        )
    )
    audits.append(
        _write_prediction(
            paths,
            view,
            "PRISM_V2_1_1_PHYSICS_FIRST",
            evaluation,
            pf_prediction,
            c_parameters + w_parameters + a_parameters,
            started,
            a_development,
            split=split,
        )
    )

    if joint.get("status") == "PASS" and joint.get("ar_profile") is not None:
        joint_requirement = _target_requirement(joint.get("ar_profile"))
        joint_development = (
            assembly
            if joint_requirement is None
            else apply_common_requirements(assembly, [joint_requirement]).reset_index(drop=True)
        )
        combined_evaluation = pd.concat(
            [joint_development, evaluation], ignore_index=True
        ).reset_index(drop=True)
        c_w_joint = _fit_c_w(
            paths,
            view,
            assembly,
            combined_evaluation,
            active,
            config,
            c,
            w,
            evaluation_split=split,
        )
        features = fit_physical_features(
            paths.shared,
            view,
            assembly,
            combined_evaluation,
            active,
            config,
            fit_split="validation",
            evaluation_split=split,
        )
        n_joint = len(joint_development)
        joint_k = features["joint_evaluation"][:n_joint]
        test_k = features["joint_evaluation"][n_joint:]
        _, combined_w, _ = joint_w_basis(
            c_w_test["fit_seed"],
            c_w_joint["evaluation_seed"],
            w_contract,
        )
        joint_w = combined_w[:n_joint]
        test_w = combined_w[n_joint:]
        target_accessor = BaseAccessor(
            paths.shared, view.head.dataset, split, [view.head.target]
        )
        delta, history = (int(value) for value in joint["ar_profile"])
        joint_a = target_accessor.target_state(
            joint_development, view.head.target, delta, history
        )
        test_a = target_accessor.target_state(evaluation, view.head.target, delta, history)
        contract = joint["joint_contract"]
        joint_prediction, refit_contract, _ = fit_joint_candidate(
            {"K": joint_k, "W": joint_w, "A": joint_a},
            joint_development["y_true"].to_numpy(dtype=np.float64),
            {"K": test_k, "W": test_w, "A": test_a},
            candidate=contract["family"],
            alpha=float(contract["alpha"]),
            k_over_a_ratio=float(contract["k_over_a_ratio"]),
            w_over_a_ratio=float(contract["w_over_a_ratio"]),
        )
        audits.append(
            _write_prediction(
                paths,
                view,
                "PRISM_V2_1_1_JOINT_KWA",
                evaluation,
                joint_prediction,
                int(refit_contract.get("parameter_count", 0)),
                started,
                joint_development,
                split=split,
            )
        )
    write_json(
        _prediction_root(paths, split) / view.relative_root / "PRISM_DYNAMIC_RESULT.json",
        {
            "status": "PASS",
            "models": audits,
            "test_accessed": split == "test",
            "ood_accessed": split == "ood",
            "prior_test_residual_rows_for_a": prior_test_residual_rows,
        },
    )
    return audits
