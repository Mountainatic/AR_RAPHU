"""Stage-2 bidirectional structural confirmation and sensitivity audits."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import stratified_two_direction_improvement
from .data_loader import load_workbook_data
from .linear_q import fit_ridge, relative_improvement
from .pipeline import evaluate_profile, prepare_direction
from .runtime import atomic_json, load_config
from .timebase import Timebase
from .validation import rolling_origin_folds, select_ridge_alpha


def _kernel_structure(first: list[float], second: list[float]) -> dict[str, Any]:
    a, b = np.asarray(first), np.asarray(second)
    if len(a) != len(b) or len(a) == 0:
        return {"correlation": 0.0, "support_jaccard": 0.0, "peak_overlap": False}
    correlation = (
        float(np.corrcoef(a, b)[0, 1])
        if np.std(a) > 0 and np.std(b) > 0
        else 0.0
    )
    support_a = np.abs(a) >= 0.2 * max(float(np.max(np.abs(a))), 1.0e-15)
    support_b = np.abs(b) >= 0.2 * max(float(np.max(np.abs(b))), 1.0e-15)
    union = int(np.sum(support_a | support_b))
    jaccard = float(np.sum(support_a & support_b) / union) if union else 1.0
    return {
        "correlation": correlation,
        "support_jaccard": jaccard,
        "peak_overlap": bool(abs(int(np.argmax(np.abs(a))) - int(np.argmax(np.abs(b)))) <= 1),
    }


def _common_support(result: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    signals = [arrays["d1_current_signal"], arrays["d2_current_signal"]]
    bounds = [
        [float(np.quantile(signal, 0.01)), float(np.quantile(signal, 0.99))]
        for signal in signals
    ]
    lower, upper = max(value[0] for value in bounds), min(value[1] for value in bounds)
    direction_improvements = []
    counts = []
    for index in (1, 2):
        signal = arrays[f"d{index}_current_signal"]
        mask = (signal >= lower) & (signal <= upper)
        baseline = arrays[f"d{index}_baseline_loss"][mask]
        q_loss = arrays[f"d{index}_q_loss"][mask]
        direction_improvements.append(relative_improvement(baseline, q_loss))
        counts.append(int(np.sum(mask)))
    pooled = 1.0 - sum(
        float(np.sum(arrays[f"d{i}_q_loss"][
            (arrays[f"d{i}_current_signal"] >= lower)
            & (arrays[f"d{i}_current_signal"] <= upper)
        ]))
        for i in (1, 2)
    ) / max(
        sum(
            float(np.sum(arrays[f"d{i}_baseline_loss"][
                (arrays[f"d{i}_current_signal"] >= lower)
                & (arrays[f"d{i}_current_signal"] <= upper)
            ]))
            for i in (1, 2)
        ),
        np.finfo(np.float64).eps,
    )
    return {
        "bounds": [lower, upper],
        "counts": counts,
        "direction_improvements": direction_improvements,
        "pooled_improvement": float(pooled),
        "drop_from_full": float(result["pooled"]["q_improvement"] - pooled),
    }


def _placebo_prediction_shift(
    arrays: dict[str, np.ndarray], cadence_rows: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for multiplier in (-2, -1, 1, 2):
        shift = multiplier * cadence_rows
        improvements = []
        for index in (1, 2):
            prediction = arrays[f"d{index}_q_prediction"]
            target = arrays[f"d{index}_target"]
            baseline = arrays[f"d{index}_baseline_loss"]
            if shift > 0:
                prediction, target, baseline = (
                    prediction[:-shift],
                    target[shift:],
                    baseline[shift:],
                )
            else:
                prediction, target, baseline = (
                    prediction[-shift:],
                    target[:shift],
                    baseline[:shift],
                )
            improvements.append(
                relative_improvement(baseline, (target - prediction) ** 2)
            )
        output[str(multiplier)] = improvements
    return output


def _regularization_sensitivity(
    workbook,
    starts,
    profile,
    variant,
    config,
    timebase,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for direction_index, (train_sheet, test_sheet) in enumerate(
        config["validation"]["outer_directions"], start=1
    ):
        features = prepare_direction(
            workbook,
            starts=starts,
            profile=profile,
            variant=variant,
            train_sheet=train_sheet,
            test_sheet=test_sheet,
            config=config,
            timebase=timebase,
        )
        purge = features.train.rows.horizon_samples + features.train.rows.window_samples
        folds = rolling_origin_folds(
            features.train.rows.origins,
            config["validation"]["inner_validation_fractions"],
            purge_samples=purge,
        )
        selected, _ = select_ridge_alpha(
            features.train.q,
            features.train.rows.target,
            folds,
            config["validation"]["ridge_grid"],
        )
        direction = {}
        for multiplier in config["validation"]["regularization_sensitivity_multipliers"]:
            alpha = max(float(selected) * float(multiplier), 1.0e-12)
            fit = fit_ridge(features.train.q, features.train.rows.target, alpha=alpha)
            loss = (features.test.rows.target - fit.predict(features.test.q)) ** 2
            direction[str(multiplier)] = {
                "alpha": alpha,
                "MSE": float(np.mean(loss)),
            }
        output[f"d{direction_index}"] = direction
    return output


def confirm_candidate(
    root: Path,
    result: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    workbook,
    starts,
    config,
    timebase,
) -> dict[str, Any]:
    losses = [
        (arrays[f"d{i}_baseline_loss"], arrays[f"d{i}_q_loss"])
        for i in (1, 2)
    ]
    conditional_losses = [
        (arrays[f"d{i}_ar_scale_loss"], arrays[f"d{i}_frozen_loss"])
        for i in (1, 2)
    ]
    block_lengths = [
        int(result["directions"][i]["bootstrap_block_length"]) for i in (0, 1)
    ]
    seed = int(config["random_seed"]) + sum(map(ord, result["task_id"]))
    q_bootstrap = stratified_two_direction_improvement(
        losses,
        replicates=int(config["validation"]["stage2_bootstrap_replicates"]),
        block_lengths=block_lengths,
        seed=seed,
    )
    conditional_bootstrap = stratified_two_direction_improvement(
        conditional_losses,
        replicates=int(config["validation"]["stage2_bootstrap_replicates"]),
        block_lengths=block_lengths,
        seed=seed + 1,
    )
    structure = _kernel_structure(
        result["directions"][0]["linear_kernel"],
        result["directions"][1]["linear_kernel"],
    )
    common = _common_support(result, arrays)
    profile = result["profile"]
    branch = config["branches"][profile["channel"]]
    cadence_rows = max(
        1,
        int(round(
            max(10.0, float(profile["horizon_min"])) * 60.0
            / float(branch["cadence_sec"])
        )),
    )
    resolution = {"1.0": result["pooled"]["q_improvement"]}
    for multiplier in config["validation"]["resolution_sensitivity_multipliers"]:
        multiplier = float(multiplier)
        if multiplier == 1.0:
            continue
        sensitivity, _ = evaluate_profile(
            workbook,
            starts=starts,
            profile=profile,
            variant=result["variant"],
            config=config,
            timebase=timebase,
            resolution_multiplier=multiplier,
        )
        resolution[str(multiplier)] = sensitivity["pooled"]["q_improvement"]
    gates = config["gates"]
    direction_improvements = [
        direction["models"]["Q1_CHANNEL_LINEAR"]["relative_improvement"]
        for direction in result["directions"]
    ]
    s2 = bool(
        all(value > 0.0 for value in direction_improvements)
        and result["pooled"]["q_improvement"] >= gates["s2_min_pooled_improvement"]
        and q_bootstrap["positive_probability"]
        >= gates["s2_min_bootstrap_positive_probability"]
        and structure["correlation"] >= gates["s2_min_kernel_correlation"]
        and structure["support_jaccard"] >= gates["s2_min_support_jaccard"]
        and common["drop_from_full"] <= gates["s2_max_common_support_drop"]
    )
    conditional_directions = [
        direction["models"]["Q2_FROZEN_AR_PLUS_Q"]["relative_improvement"]
        for direction in result["directions"]
    ]
    conditional = bool(
        all(value >= 0.0 for value in conditional_directions)
        and result["pooled"]["conditional_improvement"]
        >= gates["conditional_min_pooled_improvement"]
        and conditional_bootstrap["positive_probability"]
        >= gates["conditional_min_bootstrap_positive_probability"]
    )
    return {
        "task_id": result["task_id"],
        "profile": profile,
        "variant": result["variant"],
        "status": "COMPLETED",
        "direction_improvements": direction_improvements,
        "pooled_q_improvement": result["pooled"]["q_improvement"],
        "bootstrap_500": q_bootstrap,
        "kernel_structure": structure,
        "common_support": common,
        "prediction_shift_placebo": _placebo_prediction_shift(arrays, cadence_rows),
        "regularization_sensitivity": _regularization_sensitivity(
            workbook, starts, profile, result["variant"], config, timebase
        ),
        "resolution_sensitivity": resolution,
        "conditional_direction_improvements": conditional_directions,
        "pooled_conditional_improvement": result["pooled"]["conditional_improvement"],
        "conditional_bootstrap_500": conditional_bootstrap,
        "gates": {
            "S2_stable_structure": s2,
            "C_AR_conditional_gain": conditional,
        },
    }


def run_stage2(
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
    summary_path = root / "results" / "stage1" / "summary.json"
    stage1 = json.loads(summary_path.read_text(encoding="utf-8"))
    confirmations = []
    for task_id in stage1["stage2_candidates"]:
        task_root = root / "results" / "stage1" / "profiles" / task_id
        result = json.loads((task_root / "result.json").read_text(encoding="utf-8"))
        with np.load(task_root / "arrays.npz") as stored:
            arrays = {name: stored[name] for name in stored.files}
        confirmation = confirm_candidate(
            root,
            result,
            arrays,
            workbook=workbook,
            starts=starts,
            config=config,
            timebase=Timebase(sample_period_sec),
        )
        confirmations.append(confirmation)
        atomic_json(
            root / "results" / "stage2" / "profiles" / task_id / "result.json",
            confirmation,
        )
        print(
            f"STAGE2_TASK task={task_id} "
            f"s2={confirmation['gates']['S2_stable_structure']} "
            f"conditional={confirmation['gates']['C_AR_conditional_gain']}",
            flush=True,
        )
    rows = []
    for item in confirmations:
        rows.append({
            "task_id": item["task_id"],
            "channel": item["profile"]["channel"],
            "variant": item["variant"],
            "pooled_q_improvement": item["pooled_q_improvement"],
            "bootstrap_positive_probability": item["bootstrap_500"][
                "positive_probability"
            ],
            "kernel_correlation": item["kernel_structure"]["correlation"],
            "support_jaccard": item["kernel_structure"]["support_jaccard"],
            "common_support_drop": item["common_support"]["drop_from_full"],
            "pooled_conditional_improvement": item[
                "pooled_conditional_improvement"
            ],
            "conditional_positive_probability": item[
                "conditional_bootstrap_500"
            ]["positive_probability"],
            "S2_stable_structure": item["gates"]["S2_stable_structure"],
            "C_AR_conditional_gain": item["gates"]["C_AR_conditional_gain"],
        })
    output = root / "results" / "STAGE2_CONFIRMATION.csv"
    if rows:
        with output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        output.write_text(
            "task_id,status\n,NO_STAGE1_CANDIDATES\n", encoding="utf-8"
        )
    summary = {
        "status": "COMPLETED",
        "config_sha256": config_sha,
        "data_sha256": workbook.sha256,
        "confirmed_tasks": [
            item["task_id"]
            for item in confirmations
            if item["gates"]["S2_stable_structure"]
        ],
        "conditional_gain_tasks": [
            item["task_id"]
            for item in confirmations
            if item["gates"]["C_AR_conditional_gain"]
        ],
        "results": confirmations,
    }
    atomic_json(root / "results" / "stage2" / "summary.json", summary)
    atomic_json(
        root / "results" / "checkpoints" / "latest.json",
        {"completed_stage": "STAGE2", **summary},
    )
    return summary
