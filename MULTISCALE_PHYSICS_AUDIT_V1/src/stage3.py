"""Nested nonlinear K confirmation for Stage-2-passed profiles."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import stratified_two_direction_improvement
from .data_loader import load_workbook_data
from .linear_q import fit_block_ridge, fit_ridge, relative_improvement
from .pipeline import prepare_direction
from .runtime import atomic_json, load_config
from .timebase import Timebase
from .validation import rolling_origin_folds, select_block_alphas, select_ridge_alpha


def nonlinear_features(q: np.ndarray) -> np.ndarray:
    """Hermite-style amplitude terms; linear Q remains a separate exact block."""
    values = np.asarray(q, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    z = (values - mean) / scale
    return np.column_stack(((z**2 - 1.0) / np.sqrt(2.0), (z**3 - 3.0 * z) / np.sqrt(6.0)))


def _direction_nonlinear(
    features,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train, test = features.train, features.test
    train_nl = nonlinear_features(train.q)
    # Transform test using train moments, preserving train-only preprocessing.
    mean = train.q.mean(axis=0)
    scale = train.q.std(axis=0)
    scale[scale == 0] = 1.0
    z_test = (test.q - mean) / scale
    test_nl = np.column_stack(
        ((z_test**2 - 1.0) / np.sqrt(2.0), (z_test**3 - 3.0 * z_test) / np.sqrt(6.0))
    )
    purge = train.rows.horizon_samples + train.rows.window_samples
    folds = rolling_origin_folds(
        train.rows.origins,
        config["validation"]["inner_validation_fractions"],
        purge_samples=purge,
    )
    linear_alpha, _ = select_ridge_alpha(
        train.q, train.rows.target, folds, config["validation"]["ridge_grid"]
    )
    linear = fit_ridge(train.q, train.rows.target, alpha=linear_alpha)
    (alpha_linear, alpha_nonlinear), selection = select_block_alphas(
        train.q,
        train_nl,
        train.rows.target,
        folds,
        config["nonlinear"]["penalty_grid"],
    )
    nested = fit_block_ridge(
        train.q,
        train_nl,
        train.rows.target,
        alpha_ar=alpha_linear,
        alpha_q=alpha_nonlinear,
    )
    linear_prediction = linear.predict(test.q)
    nonlinear_prediction = nested.predict(test.q, test_nl)
    linear_loss = (test.rows.target - linear_prediction) ** 2
    nonlinear_loss = (test.rows.target - nonlinear_prediction) ** 2
    return {
        "train_sheet": features.train_sheet,
        "test_sheet": features.test_sheet,
        "linear_alpha": linear_alpha,
        "nested_linear_alpha": alpha_linear,
        "nested_nonlinear_alpha": (
            "NONLINEAR_EXACT_ZERO"
            if alpha_nonlinear is None
            else alpha_nonlinear
        ),
        "selection_losses": selection,
        "relative_improvement": relative_improvement(linear_loss, nonlinear_loss),
        "linear_MSE": float(np.mean(linear_loss)),
        "nonlinear_MSE": float(np.mean(nonlinear_loss)),
        "relative_kkt": nested.relative_kkt,
        "exact_linear_selected": alpha_nonlinear is None,
        "nonlinear_physical_coefficients": nested.physical_q_coefficients().tolist(),
    }, {"linear_loss": linear_loss, "nonlinear_loss": nonlinear_loss}


def run_stage3(
    *,
    root: Path,
    config_path: Path,
    data_path: Path,
    sample_period_sec: float,
) -> dict[str, Any]:
    config, config_sha = load_config(config_path)
    workbook = load_workbook_data(
        data_path,
        required_sheets=config["data"]["required_sheets"],
        required_columns=config["data"]["required_columns"],
    )
    starts = {
        sheet: max(values) + 1 if values else 0
        for sheet, values in config["data"]["frozen_breakpoints"].items()
    }
    stage2 = json.loads(
        (root / "results" / "stage2" / "summary.json").read_text(encoding="utf-8")
    )
    results = []
    for task_id in stage2["confirmed_tasks"]:
        stage1_result = json.loads(
            (
                root
                / "results"
                / "stage1"
                / "profiles"
                / task_id
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        directions = []
        loss_pairs = []
        block_lengths = []
        for train_sheet, test_sheet in config["validation"]["outer_directions"]:
            features = prepare_direction(
                workbook,
                starts=starts,
                profile=stage1_result["profile"],
                variant=stage1_result["variant"],
                train_sheet=train_sheet,
                test_sheet=test_sheet,
                config=config,
                timebase=Timebase(sample_period_sec),
            )
            direction, arrays = _direction_nonlinear(features, config)
            directions.append(direction)
            loss_pairs.append((arrays["linear_loss"], arrays["nonlinear_loss"]))
            branch = config["branches"][stage1_result["profile"]["channel"]]
            block_lengths.append(max(
                1,
                int(np.ceil(
                    max(
                        10.0,
                        float(stage1_result["profile"]["target_window_min"]),
                    )
                    * 60.0
                    / float(branch["cadence_sec"])
                )),
            ))
        bootstrap = stratified_two_direction_improvement(
            loss_pairs,
            replicates=int(config["validation"]["stage2_bootstrap_replicates"]),
            block_lengths=block_lengths,
            seed=int(config["random_seed"]) + sum(map(ord, task_id)) + 90_000,
        )
        pooled = 1.0 - sum(float(np.sum(pair[1])) for pair in loss_pairs) / max(
            sum(float(np.sum(pair[0])) for pair in loss_pairs),
            np.finfo(np.float64).eps,
        )
        nonlinear_pass = bool(
            all(direction["relative_improvement"] > 0.0 for direction in directions)
            and pooled >= float(config["gates"]["nonlinear_min_pooled_improvement"])
            and bootstrap["positive_probability"]
            >= float(config["gates"]["nonlinear_min_bootstrap_positive_probability"])
        )
        item = {
            "task_id": task_id,
            "profile": stage1_result["profile"],
            "variant": stage1_result["variant"],
            "status": "COMPLETED",
            "directions": directions,
            "pooled_nonlinear_improvement": float(pooled),
            "bootstrap_500": bootstrap,
            "nonlinear_K_gain": nonlinear_pass,
            "linear_submodel_exactly_nested": True,
        }
        results.append(item)
        atomic_json(
            root / "results" / "stage3" / "profiles" / task_id / "result.json",
            item,
        )
        print(
            f"STAGE3_TASK task={task_id} nonlinear_gain={nonlinear_pass}",
            flush=True,
        )
    final_decision = {
        "schema": config["schema"],
        "status": "COMPLETED",
        "config_sha256": config_sha,
        "data_sha256": workbook.sha256,
        "confirmatory_linear_structure_tasks": stage2["confirmed_tasks"],
        "AR_conditional_gain_tasks": stage2["conditional_gain_tasks"],
        "nonlinear_K_gain_tasks": [
            item["task_id"] for item in results if item["nonlinear_K_gain"]
        ],
        "failed_channels_exact_zero": [
            channel
            for channel in config["branches"]
            if not any(
                item["profile"]["channel"] == channel
                for item in stage2["results"]
                if item["gates"]["S2_stable_structure"]
            )
        ],
        "combined_model_status": (
            "NOT_APPLICABLE_NO_SHARED_TARGET_PROFILE"
        ),
        "combined_model_note": (
            "Profiles intentionally use different horizons, output windows, and "
            "cadences. A cross-channel combined model would require a separately "
            "predeclared shared target profile and was not selected after seeing data."
        ),
        "stage3_results": results,
    }
    atomic_json(root / "results" / "FINAL_DECISION.json", final_decision)
    atomic_json(root / "results" / "stage3" / "summary.json", final_decision)
    atomic_json(
        root / "results" / "checkpoints" / "latest.json",
        {"completed_stage": "STAGE3", **final_decision},
    )
    _write_final_tables(root, stage2, results)
    return final_decision


def _write_final_tables(
    root: Path, stage2: dict[str, Any], stage3_results: list[dict[str, Any]]
) -> None:
    stage3 = {item["task_id"]: item for item in stage3_results}
    rows = []
    for item in stage2["results"]:
        nonlinear = stage3.get(item["task_id"])
        rows.append({
            "task_id": item["task_id"],
            "channel": item["profile"]["channel"],
            "variant": item["variant"],
            "horizon_min": item["profile"]["horizon_min"],
            "target_window_min": item["profile"]["target_window_min"],
            "history_min": item["profile"]["history_min"],
            "structure_evidence": item["gates"]["S2_stable_structure"],
            "AR_conditional_gain": item["gates"]["C_AR_conditional_gain"],
            "nonlinear_K_gain": (
                nonlinear["nonlinear_K_gain"] if nonlinear else False
            ),
            "pooled_linear_Q_improvement": item["pooled_q_improvement"],
            "pooled_AR_conditional_improvement": item[
                "pooled_conditional_improvement"
            ],
            "pooled_nonlinear_improvement": (
                nonlinear["pooled_nonlinear_improvement"] if nonlinear else ""
            ),
        })
    fields = [
        "task_id", "channel", "variant", "horizon_min", "target_window_min",
        "history_min", "structure_evidence", "AR_conditional_gain",
        "nonlinear_K_gain", "pooled_linear_Q_improvement",
        "pooled_AR_conditional_improvement", "pooled_nonlinear_improvement",
    ]
    for name in ("CHANNEL_TIMESCALE_SUMMARY.csv", "FINAL_BASELINE_TABLE.csv"):
        with (root / "results" / name).open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
