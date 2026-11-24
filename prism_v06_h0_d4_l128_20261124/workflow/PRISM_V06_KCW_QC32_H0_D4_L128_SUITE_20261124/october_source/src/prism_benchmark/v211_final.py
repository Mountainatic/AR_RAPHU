from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor, ViewSpec, load_samples, sha256_file
from .cpu_selection import regression_metrics
from .stage0 import write_json
from .v2_c import fit_physical_features
from .v21_baselines import materialize_test_baselines
from .v21_views import sru_dynamic_views, sru_input_views
from .v211_a import EXACT_ZERO, fit_mature_residual_ar, mature_residual_features
from .v211_config import V211Paths, require_v211_test_freeze
from .v211_joint import fit_joint_candidate, joint_w_basis
from .v211_k import load_active_channels
from .v211_w import IDENTITY, _fit_c_routed, fit_w_correction


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError(f"prerequisite is not PASS: {path}")
    return value


def _selected_results(
    paths: V211Paths, view: ViewSpec
) -> tuple[dict[str, Any], ...]:
    c = _read(
        paths.output
        / "DEVELOPMENT"
        / "C"
        / view.head.head_id
        / view.proxy_policy
        / "RESULT.json"
    )
    w = _read(
        paths.output
        / "DEVELOPMENT"
        / "W"
        / view.head.head_id
        / view.proxy_policy
        / "RESULT.json"
    )
    a = _read(
        paths.output
        / "DEVELOPMENT"
        / "A"
        / view.head.head_id
        / view.availability_scenario
        / view.proxy_policy
        / "RESULT.json"
    )
    joint = _read(
        paths.output
        / "DEVELOPMENT"
        / "JOINT"
        / view.head.head_id
        / view.availability_scenario
        / view.proxy_policy
        / "RESULT.json"
    )
    return c, w, a, joint


def _active_for_c(paths: V211Paths, view: ViewSpec, c: dict[str, Any]) -> list[dict[str, Any]]:
    frozen = set(c.get("active_channels", ()))
    return [
        item
        for item in load_active_channels(paths.output, view)
        if item.get("channel") in frozen
    ]


def _fit_final_input_view(paths: V211Paths, view: ViewSpec) -> list[dict[str, Any]]:
    from .v2_config import load_frozen_config

    started = time.perf_counter()
    v2 = load_frozen_config(paths.project)
    c = _read(
        paths.output
        / "DEVELOPMENT"
        / "C"
        / view.head.head_id
        / view.proxy_policy
        / "RESULT.json"
    )
    w = _read(
        paths.output
        / "DEVELOPMENT"
        / "W"
        / view.head.head_id
        / view.proxy_policy
        / "RESULT.json"
    )
    active = _active_for_c(paths, view, c)
    development = (
        pd.concat(
            [
                load_samples(paths.shared, view, "train"),
                load_samples(paths.shared, view, "validation"),
            ],
            ignore_index=True,
        )
        .sort_values(["entity_id", "origin"])
        .reset_index(drop=True)
    )
    test = load_samples(paths.shared, view, "test")
    dev_seed, test_seed, dev_upstream, _, _ = _fit_c_routed(
        paths.shared,
        view,
        development,
        test,
        active,
        v2,
        c,
        fit_split="validation",
        evaluation_split="test",
    )
    w_contract = w["w_contract"]
    if w_contract["family"] == IDENTITY:
        test_correction = np.zeros(len(test), dtype=np.float64)
        test_correction_mu0 = test_correction.copy()
    else:
        kwargs = dict(
            family=w_contract["family"],
            knot_count=int(w_contract["knot_count"]),
            smoothness=float(w_contract["smoothness"]),
            upstream_predictions=dev_upstream,
            direction=int(w_contract["direction"]),
        )
        test_correction, _ = fit_w_correction(
            dev_seed,
            development["y_true"].to_numpy(dtype=np.float64) - dev_seed,
            test_seed,
            mu=float(w_contract["soft_overlap_mu"]),
            **kwargs,
        )
        test_correction_mu0, _ = fit_w_correction(
            dev_seed,
            development["y_true"].to_numpy(dtype=np.float64) - dev_seed,
            test_seed,
            mu=0.0,
            **kwargs,
        )
    root = (
        paths.output
        / "FINAL"
        / "test_predictions"
        / view.head.head_id
        / view.information_set
        / view.availability_scenario
        / view.proxy_policy
    )
    root.mkdir(parents=True, exist_ok=True)
    audits = []
    c_parameters = int(c.get("fusion_contract", {}).get("parameter_count", 0))
    w_parameters = int(w_contract.get("parameter_count", 0))
    for model, prediction, parameter_count in (
        ("PRISM_V2_1_1_K_C", test_seed, c_parameters),
        (
            "PRISM_V2_1_1_K_C_W",
            test_seed + test_correction,
            c_parameters + w_parameters,
        ),
        (
            "PRISM_V2_1_1_K_C_W_MU0_ABLATION",
            test_seed + test_correction_mu0,
            c_parameters + w_parameters,
        ),
    ):
        frame = test[
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
        ].copy().rename(columns={"view_sample_id": "sample_id"})
        frame["y_pred"] = prediction
        frame["model"] = model
        frame["dtype"] = "float64"
        path = root / f"{model}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        audits.append(
            {
                "status": "PASS",
                "target_head": view.head.head_id,
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "model": model,
                "rows": len(frame),
                "parameter_count": parameter_count,
                "effective_df": (
                    w_contract.get("effective_df") if model.endswith("_W") else None
                ),
                "fit_and_prediction_seconds": time.perf_counter() - started,
                "prediction_path": str(path.relative_to(paths.output)),
                "prediction_sha256": sha256_file(path),
                "test_accessed": True,
                **regression_metrics(
                    frame["y_true"].to_numpy(dtype=np.float64), prediction
                ),
            }
        )
    return audits


def _fit_final_view(paths: V211Paths, view: ViewSpec) -> list[dict[str, Any]]:
    from .v2_config import load_frozen_config

    started = time.perf_counter()
    v2 = load_frozen_config(paths.project)
    c, w, a, joint = _selected_results(paths, view)
    active = _active_for_c(paths, view, c)
    development = (
        pd.concat(
            [
                load_samples(paths.shared, view, "train"),
                load_samples(paths.shared, view, "validation"),
            ],
            ignore_index=True,
        )
        .sort_values(["entity_id", "origin"])
        .reset_index(drop=True)
    )
    test = load_samples(paths.shared, view, "test")
    dev_seed, test_seed, dev_upstream, _, _ = _fit_c_routed(
        paths.shared,
        view,
        development,
        test,
        active,
        v2,
        c,
        fit_split="validation",
        evaluation_split="test",
    )
    w_contract = w["w_contract"]
    if w_contract["family"] == IDENTITY:
        dev_correction = np.zeros(len(development), dtype=np.float64)
        test_correction = np.zeros(len(test), dtype=np.float64)
    else:
        all_correction, _ = fit_w_correction(
            dev_seed,
            development["y_true"].to_numpy(dtype=np.float64) - dev_seed,
            np.concatenate([dev_seed, test_seed]),
            family=w_contract["family"],
            knot_count=int(w_contract["knot_count"]),
            smoothness=float(w_contract["smoothness"]),
            mu=float(w_contract["soft_overlap_mu"]),
            upstream_predictions=dev_upstream,
            direction=int(w_contract["direction"]),
        )
        dev_correction = all_correction[: len(development)]
        test_correction = all_correction[len(development) :]
    dev_physical = dev_seed + dev_correction
    test_physical = test_seed + test_correction
    dev_residual = development["y_true"].to_numpy(dtype=np.float64) - dev_physical
    test_residual = test["y_true"].to_numpy(dtype=np.float64) - test_physical
    residual_source = pd.concat(
        [
            development[["entity_id", "origin"]].assign(residual=dev_residual),
            test[["entity_id", "origin"]].assign(residual=test_residual),
        ],
        ignore_index=True,
    )
    a_contract = a["a_contract"]
    if a_contract["family"] == EXACT_ZERO:
        pf_residual = np.zeros(len(test), dtype=np.float64)
        pf_residual_mu0 = pf_residual.copy()
    else:
        delta, history = (int(value) for value in a_contract["profile"])
        residual_mean = float(np.mean(dev_residual, dtype=np.float64))
        kwargs = dict(
            h_steps=view.head.h_steps,
            w_steps=view.head.w_steps,
            delta=delta,
            history=history,
            maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]),
            residual_mean=residual_mean,
        )
        dev_a, _, _ = mature_residual_features(
            development, residual_source, **kwargs
        )
        test_a, _, _ = mature_residual_features(test, residual_source, **kwargs)
        pf_residual, _ = fit_mature_residual_ar(
            dev_a,
            dev_residual,
            test_a,
            alpha=float(a_contract["alpha"]),
            mu=float(a_contract["soft_overlap_mu"]),
            upstream_predictions=np.column_stack([dev_upstream, dev_correction]),
        )
        pf_residual_mu0, _ = fit_mature_residual_ar(
            dev_a,
            dev_residual,
            test_a,
            alpha=float(a_contract["alpha"]),
            mu=0.0,
            upstream_predictions=np.column_stack([dev_upstream, dev_correction]),
        )
    pf_prediction = test_physical + pf_residual
    pf_mu0_prediction = test_physical + pf_residual_mu0
    k_residual_dev = development["y_true"].to_numpy(dtype=np.float64) - dev_seed
    k_residual_test = test["y_true"].to_numpy(dtype=np.float64) - test_seed
    if a_contract["family"] == EXACT_ZERO:
        k_a_prediction = test_seed.copy()
    else:
        k_residual_source = pd.concat(
            [
                development[["entity_id", "origin"]].assign(
                    residual=k_residual_dev
                ),
                test[["entity_id", "origin"]].assign(residual=k_residual_test),
            ],
            ignore_index=True,
        )
        delta, history = (int(value) for value in a_contract["profile"])
        k_mean = float(np.mean(k_residual_dev, dtype=np.float64))
        kwargs = dict(
            h_steps=view.head.h_steps,
            w_steps=view.head.w_steps,
            delta=delta,
            history=history,
            maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]),
            residual_mean=k_mean,
        )
        k_a_dev, _, _ = mature_residual_features(
            development, k_residual_source, **kwargs
        )
        k_a_test, _, _ = mature_residual_features(test, k_residual_source, **kwargs)
        k_a_residual, _ = fit_mature_residual_ar(
            k_a_dev,
            k_residual_dev,
            k_a_test,
            alpha=float(a_contract["alpha"]),
            mu=float(a_contract["soft_overlap_mu"]),
            upstream_predictions=dev_upstream,
        )
        k_a_prediction = test_seed + k_a_residual
    physical_features = fit_physical_features(
        paths.shared,
        view,
        development,
        test,
        active,
        v2,
        fit_split="validation",
        evaluation_split="test",
    )
    k_dev = physical_features["joint_train"]
    k_test = physical_features["joint_evaluation"]
    if w_contract["family"] == IDENTITY:
        w_dev = np.empty((len(development), 0), dtype=np.float64)
        w_test = np.empty((len(test), 0), dtype=np.float64)
    else:
        w_dev, w_test, _ = joint_w_basis(dev_seed, test_seed, w_contract)
    joint_profile = tuple(int(value) for value in joint["ar_profile"])
    target_accessor = BaseAccessor(
        paths.shared, view.head.dataset, "test", [view.head.target]
    )
    joint_a_dev = target_accessor.target_state(
        development, view.head.target, *joint_profile
    )
    joint_a_test = target_accessor.target_state(test, view.head.target, *joint_profile)
    joint_contract = joint["joint_contract"]
    joint_prediction, refit_contract, components = fit_joint_candidate(
        {"K": k_dev, "W": w_dev, "A": joint_a_dev},
        development["y_true"].to_numpy(dtype=np.float64),
        {"K": k_test, "W": w_test, "A": joint_a_test},
        candidate=joint_contract["family"],
        alpha=float(joint_contract["alpha"]),
        k_over_a_ratio=float(joint_contract["k_over_a_ratio"]),
        w_over_a_ratio=float(joint_contract["w_over_a_ratio"]),
    )
    route_predictions: dict[
        str, tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]
    ] = {}
    for route in ("J_KA", "J_KWA"):
        selected = joint.get("route_local_selected", {}).get(route)
        if selected is None:
            continue
        _, alpha, k_ratio, w_ratio = selected
        route_predictions[route] = fit_joint_candidate(
            {"K": k_dev, "W": w_dev, "A": joint_a_dev},
            development["y_true"].to_numpy(dtype=np.float64),
            {"K": k_test, "W": w_test, "A": joint_a_test},
            candidate=route,
            alpha=float(alpha),
            k_over_a_ratio=float(k_ratio),
            w_over_a_ratio=float(w_ratio),
        )
    root = (
        paths.output
        / "FINAL"
        / "test_predictions"
        / view.head.head_id
        / view.information_set
        / view.availability_scenario
        / view.proxy_policy
    )
    root.mkdir(parents=True, exist_ok=True)
    audits = []
    c_parameters = int(c.get("fusion_contract", {}).get("parameter_count", 0))
    pf_parameters = (
        c_parameters
        + int(w_contract.get("parameter_count", 0))
        + int(a_contract.get("parameter_count", 0))
    )
    models: list[tuple[str, np.ndarray, dict[str, np.ndarray], int]] = [
        ("PRISM_V2_1_1_K_C_DYNAMIC", test_seed, {}, c_parameters),
        (
            "PRISM_V2_1_1_K_C_W_DYNAMIC",
            test_physical,
            {},
            c_parameters + int(w_contract.get("parameter_count", 0)),
        ),
        (
            "PRISM_V2_1_1_K_C_A_ABLATION",
            k_a_prediction,
            {},
            c_parameters + int(a_contract.get("parameter_count", 0)),
        ),
        ("PRISM_V2_1_1_PHYSICS_FIRST", pf_prediction, {}, pf_parameters),
        ("PRISM_V2_1_1_PF_A_MU0_ABLATION", pf_mu0_prediction, {}, pf_parameters),
        (
            "PRISM_V2_1_1_JOINT_KWA",
            joint_prediction,
            {"input_prediction": components["INPUT"]},
            int(refit_contract.get("parameter_count", 0)),
        ),
    ]
    for route, (prediction, contract, route_components) in route_predictions.items():
        models.append(
            (
                f"PRISM_V2_1_1_{route}",
                prediction,
                {"input_prediction": route_components["INPUT"]},
                int(contract.get("parameter_count", 0)),
            )
        )
    for model, prediction, extra, parameter_count in models:
        frame = test[
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
        ].copy().rename(columns={"view_sample_id": "sample_id"})
        frame["y_pred"] = prediction
        frame["model"] = model
        frame["dtype"] = "float64"
        for name, values in extra.items():
            frame[name] = values
        path = root / f"{model}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        audits.append(
            {
                "status": "PASS",
                "target_head": view.head.head_id,
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "model": model,
                "rows": len(frame),
                "parameter_count": parameter_count,
                "effective_df": (
                    w_contract.get("effective_df")
                    if model == "PRISM_V2_1_1_PHYSICS_FIRST"
                    else None
                ),
                "fit_and_prediction_seconds": time.perf_counter() - started,
                "prediction_path": str(path.relative_to(paths.output)),
                "prediction_sha256": sha256_file(path),
                "test_accessed": True,
                **regression_metrics(
                    frame["y_true"].to_numpy(dtype=np.float64), prediction
                ),
            }
        )
    return audits


def run_e7r_test(paths: V211Paths) -> dict[str, Any]:
    manifest = require_v211_test_freeze(paths)
    sentinel = paths.output / "FINAL" / "TEST_ACCESS_AUDIT.json"
    if sentinel.is_file():
        raise RuntimeError("v2.1.1 SRU candidate test has already been accessed")
    write_json(
        sentinel,
        {
            "status": "TEST_ACCESS_STARTED",
            "stage": "E7R_SRU_TEST",
            "freeze_sha256": sha256_file(paths.final_freeze_path),
            "frozen_code_commit": manifest["code_commit"],
            "models": [],
            "test_accessed": True,
        },
    )
    audits = []
    for view in sru_input_views(paths.shared):
        c_path = (
            paths.output
            / "DEVELOPMENT"
            / "C"
            / view.head.head_id
            / view.proxy_policy
            / "RESULT.json"
        )
        c = json.loads(c_path.read_text(encoding="utf-8"))
        if bool(c.get("input_path_preservation", {}).get("pass", False)):
            audits.extend(_fit_final_input_view(paths, view))
    for view in sru_dynamic_views(paths.shared):
        joint_path = (
            paths.output
            / "DEVELOPMENT"
            / "JOINT"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "RESULT.json"
        )
        joint = json.loads(joint_path.read_text(encoding="utf-8"))
        if bool(joint.get("input_path_preservation", {}).get("pass", False)):
            audits.extend(_fit_final_view(paths, view))
    audits.extend(materialize_test_baselines(paths))
    result = {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in audits)
        else "FAILED",
        "stage": "E7R_SRU_TEST",
        "freeze_sha256": sha256_file(paths.final_freeze_path),
        "frozen_code_commit": manifest["code_commit"],
        "models": audits,
        "test_accessed": True,
    }
    write_json(sentinel, result)
    return result


def run_e8r_report(paths: V211Paths) -> dict[str, Any]:
    from .v211_reporting import build_report_and_package

    access = paths.output / "FINAL" / "TEST_ACCESS_AUDIT.json"
    if not access.is_file():
        raise RuntimeError("E8R requires completed E7R test access audit")
    audit = json.loads(access.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("E8R requires a PASS E7R audit")
    result = build_report_and_package(paths, audit)
    write_json(paths.output / "REPORTS" / "E8R_SUMMARY.json", result)
    return result
