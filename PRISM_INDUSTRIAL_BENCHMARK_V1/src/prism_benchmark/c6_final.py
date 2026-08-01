from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .c2_models import _prediction_frame
from .c3_models import _load_ar_contract, _ridge_block_predict
from .c5_models import _fit_frozen_channel_shapes, mature_residual_features
from .cpu_data import BaseAccessor, ViewSpec, deterministic_subsample, load_samples, main_views, sha256_file
from .cpu_selection import regression_metrics
from .stage0 import write_json


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _freeze(project: Path) -> dict[str, Any]:
    return _load_json(project / "configs/cpu_model_freeze_v1.json")


def _result_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("RESULT.json") if path.is_file())


def write_final_freeze_manifest(
    shared: Path,
    project: Path,
    c2_output: Path,
    c3_output: Path,
    c4_output: Path,
    c5_output: Path,
    output: Path,
) -> Path:
    prerequisite_manifests = [
        c2_output / "C2_CLASSICAL_MANIFEST.json",
        c3_output / "C3_MANIFEST.json",
        c4_output / "C4_MANIFEST.json",
        c5_output / "C5_MANIFEST.json",
    ]
    for path in prerequisite_manifests:
        if not path.is_file():
            raise FileNotFoundError(f"missing prerequisite manifest: {path}")
        status = _load_json(path).get("status")
        if status not in {"PASS", "PASS_WITH_RETAINED_FAILURES"}:
            raise RuntimeError(f"prerequisite is not complete: {path} status={status}")
    frozen_files = [
        project / "configs/cpu_model_freeze_v1.json",
        shared / "TASK_REGISTRY.json",
        shared / "PROTOCOL.json",
        shared / "dataset_views/VIEW_REGISTRY.json",
        *prerequisite_manifests,
        *_result_files(c2_output),
        *_result_files(c3_output),
        *_result_files(c4_output),
        *_result_files(c5_output),
    ]
    records = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in frozen_files]
    manifest = {
        "status": "FROZEN",
        "protocol": "PRISM_CPU_C6_FINAL_FREEZE_V1",
        "test_access_authorized": True,
        "test_accessed_at_manifest_creation": False,
        "finalist_policy": _freeze(project)["c6"]["finalist_policy"],
        "frozen_files": records,
    }
    path = output / "FINAL_FREEZE_MANIFEST.json"
    write_json(path, manifest)
    return path


def _development_samples(shared: Path, view: ViewSpec) -> pd.DataFrame:
    return pd.concat([load_samples(shared, view, "train"), load_samples(shared, view, "validation")], ignore_index=True)


def _c4_result(c4_output: Path, view: ViewSpec) -> dict[str, Any]:
    path = c4_output / "JOINT" / view.head.head_id / view.proxy_policy / "RESULT.json"
    result = _load_json(path)
    if result.get("status") != "PASS":
        raise RuntimeError(f"C4 finalist unavailable: {path}")
    return result


def _c5_result(c5_output: Path, view: ViewSpec) -> dict[str, Any]:
    path = c5_output / view.head.head_id / view.proxy_policy / "RESULT.json"
    result = _load_json(path)
    if result.get("status") != "PASS":
        raise RuntimeError(f"C5 finalist unavailable: {path}")
    return result


def _ar_features(shared: Path, view: ViewSpec, samples: pd.DataFrame, split: str, profile: tuple[int, int]) -> np.ndarray:
    return BaseAccessor(shared, view.head.dataset, split, [view.head.target]).target_state(samples, view.head.target, *profile)


def _fit_ar_final(shared: Path, project: Path, c3_output: Path, view: ViewSpec, evaluation: pd.DataFrame, split: str) -> tuple[np.ndarray, int]:
    contract = _load_ar_contract(c3_output, view)
    profile = tuple(contract["selected_profile"])
    alpha = float(contract["selected_alpha"])
    development = _development_samples(shared, view)
    index = deterministic_subsample(development, int(_freeze(project)["selection"]["fit_row_cap_default"]))
    fit = development.iloc[index]
    x_fit = _ar_features(shared, view, fit, "validation", profile)
    x_evaluation = _ar_features(shared, view, evaluation, split, profile)
    prediction, _ = _ridge_block_predict(x_fit, fit["y_true"].to_numpy(dtype=np.float64), x_evaluation, alpha)
    return prediction, x_fit.shape[1] + 1


def _fit_k_final(
    shared: Path,
    project: Path,
    view: ViewSpec,
    evaluation: pd.DataFrame,
    contracts: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    development = _development_samples(shared, view)
    index = deterministic_subsample(development, int(_freeze(project)["c4"]["fit_row_cap"]))
    fit = development.iloc[index]
    train_shapes, evaluation_shapes = _fit_frozen_channel_shapes(shared, project, view, fit, evaluation, contracts)
    if train_shapes.shape[1]:
        prediction, _ = _ridge_block_predict(train_shapes, fit["y_true"].to_numpy(dtype=np.float64), evaluation_shapes, 0.0001)
    else:
        prediction = np.zeros(len(evaluation), dtype=np.float64)
    return prediction, train_shapes, evaluation_shapes, fit


def _physics_first_final(
    shared: Path,
    project: Path,
    c4_output: Path,
    c5_output: Path,
    view: ViewSpec,
    evaluation: pd.DataFrame,
    split: str,
    k_prediction: np.ndarray,
) -> tuple[np.ndarray, int]:
    c5 = _c5_result(c5_output, view)
    contracts = _c4_result(c4_output, view)["selected_channel_contracts"]
    selected = c5["physics_first"]["selected_candidate"]
    if selected == "EXACT_ZERO":
        return k_prediction, len(contracts)
    profile = tuple(c5["physics_first"]["selected_profile"])
    alpha = float(c5["physics_first"]["selected_alpha"])
    oof = pd.read_parquet(c5_output / c5["oof_path"])
    validation_view = ViewSpec(view.head, "dynamic", view.availability_scenario, view.proxy_policy)
    validation = load_samples(shared, validation_view, "validation")
    validation_k = pd.read_parquet(c4_output / "JOINT" / view.head.head_id / view.proxy_policy / "PRISM_CHANNEL_SPECIFIC.validation.parquet")["y_pred"].to_numpy(dtype=np.float64)
    validation_residuals = validation[["base_origin_id", "view_sample_id", "entity_id", "origin"]].copy()
    validation_residuals["residual"] = validation["y_true"].to_numpy(dtype=np.float64) - validation_k
    development_residuals = pd.concat(
        [oof[["base_origin_id", "view_sample_id", "entity_id", "origin", "residual"]], validation_residuals],
        ignore_index=True,
    )
    residual_mean = float(development_residuals["residual"].mean())
    config = _freeze(project)["c5"]["physics_first"]
    x_fit = mature_residual_features(
        development_residuals,
        development_residuals,
        view.head.h_steps,
        view.head.w_steps,
        *profile,
        int(config["maximum_residual_lags"]),
        residual_mean,
    )
    evaluation_residuals = evaluation[["entity_id", "origin"]].copy()
    evaluation_residuals["residual"] = evaluation["y_true"].to_numpy(dtype=np.float64) - k_prediction
    residual_table = pd.concat([development_residuals[["entity_id", "origin", "residual"]], evaluation_residuals], ignore_index=True)
    x_evaluation = mature_residual_features(
        evaluation,
        residual_table,
        view.head.h_steps,
        view.head.w_steps,
        *profile,
        int(config["maximum_residual_lags"]),
        residual_mean,
    )
    index = deterministic_subsample(development_residuals, int(_freeze(project)["selection"]["fit_row_cap_default"]))
    residual_prediction, _ = _ridge_block_predict(
        x_fit[index],
        development_residuals.iloc[index]["residual"].to_numpy(dtype=np.float64),
        x_evaluation,
        alpha,
    )
    return k_prediction + residual_prediction, len(contracts) + x_fit.shape[1] + 1


def _k_joint_final(
    shared: Path,
    project: Path,
    c3_output: Path,
    c5_output: Path,
    view: ViewSpec,
    evaluation: pd.DataFrame,
    split: str,
    train_shapes: np.ndarray,
    evaluation_shapes: np.ndarray,
    fit: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    c5 = _c5_result(c5_output, view)
    selected = c5["k_joint_ar"]["selected_candidate"]
    if selected == "EXACT_BOTH_ZERO":
        return np.zeros(len(evaluation), dtype=np.float64), 0
    ar_contract = _load_ar_contract(c3_output, view)
    profile = tuple(ar_contract["selected_profile"])
    alpha = float(ar_contract["selected_alpha"])
    dynamic_development = _development_samples(shared, view).set_index("base_origin_id", drop=False)
    fit_dynamic = dynamic_development.loc[fit["base_origin_id"].astype(str)].reset_index(drop=True)
    fit_ar = _ar_features(shared, view, fit_dynamic, "validation", profile)
    evaluation_ar = _ar_features(shared, view, evaluation, split, profile)
    y_fit = fit["y_true"].to_numpy(dtype=np.float64)
    if selected == "EXACT_K_ZERO":
        prediction, _ = _ridge_block_predict(fit_ar, y_fit, evaluation_ar, alpha)
        return prediction, fit_ar.shape[1] + 1
    if selected == "EXACT_AR_ZERO":
        if not train_shapes.shape[1]:
            return np.zeros(len(evaluation), dtype=np.float64), 0
        prediction, _ = _ridge_block_predict(train_shapes, y_fit, evaluation_shapes, alpha)
        return prediction, train_shapes.shape[1] + 1
    train_joint = np.concatenate([fit_ar, train_shapes], axis=1)
    evaluation_joint = np.concatenate([evaluation_ar, evaluation_shapes], axis=1)
    penalties = np.concatenate([np.full(fit_ar.shape[1], alpha), np.full(train_shapes.shape[1], alpha * float(selected))])
    prediction, _ = _ridge_block_predict(train_joint, y_fit, evaluation_joint, penalties)
    return prediction, train_joint.shape[1] + 1


def _frame(
    samples: pd.DataFrame,
    view: ViewSpec,
    model: str,
    prediction: np.ndarray,
    parameter_count: int,
    split: str,
    core_history_steps: int,
) -> pd.DataFrame:
    frame = _prediction_frame(samples, view, model, prediction, parameter_count)
    frame["split"] = split
    frame["h_steps"] = view.head.h_steps
    frame["w_steps"] = view.head.w_steps
    frame["core_history_steps"] = core_history_steps
    return frame


def _evaluate_split(
    shared: Path,
    project: Path,
    c3_output: Path,
    c4_output: Path,
    c5_output: Path,
    output: Path,
    head_view: ViewSpec,
    split: str,
) -> list[pd.DataFrame]:
    input_view = ViewSpec(head_view.head, "input_only", head_view.availability_scenario, head_view.proxy_policy)
    dynamic_view = ViewSpec(head_view.head, "dynamic", head_view.availability_scenario, head_view.proxy_policy)
    input_samples = load_samples(shared, input_view, split)
    dynamic_samples = load_samples(shared, dynamic_view, split)
    c4 = _c4_result(c4_output, input_view)
    contracts = list(c4["selected_channel_contracts"])
    c5 = _c5_result(c5_output, dynamic_view)
    histories = [view.head.h_steps + view.head.w_steps for view in [head_view]]
    histories.extend(int(item["selected_profile"][1]) for item in contracts)
    if c5["physics_first"].get("selected_profile") is not None:
        histories.append(int(c5["physics_first"]["selected_profile"][1]))
    histories.append(int(c5["k_joint_ar"]["state_profile"][1]))
    core_history_steps = max(histories)
    if not np.array_equal(input_samples["base_origin_id"].to_numpy(), dynamic_samples["base_origin_id"].to_numpy()):
        raise AssertionError("input-only and dynamic finalists do not share base origins")
    k_prediction, train_shapes, evaluation_shapes, fit = _fit_k_final(shared, project, input_view, input_samples, contracts)
    frames = [
        _frame(input_samples, input_view, "PERSISTENCE", np.zeros(len(input_samples)), 0, split, core_history_steps),
        _frame(input_samples, input_view, "PRISM_CHANNEL_SPECIFIC", k_prediction, len(contracts) + 1, split, core_history_steps),
        _frame(dynamic_samples, dynamic_view, "PERSISTENCE", np.zeros(len(dynamic_samples)), 0, split, core_history_steps),
    ]
    ar_prediction, ar_parameters = _fit_ar_final(shared, project, c3_output, dynamic_view, dynamic_samples, split)
    frames.append(_frame(dynamic_samples, dynamic_view, "AR", ar_prediction, ar_parameters, split, core_history_steps))
    physics_prediction, physics_parameters = _physics_first_final(shared, project, c4_output, c5_output, dynamic_view, dynamic_samples, split, k_prediction)
    frames.append(_frame(dynamic_samples, dynamic_view, "PRISM_PHYSICS_FIRST", physics_prediction, physics_parameters, split, core_history_steps))
    joint_prediction, joint_parameters = _k_joint_final(
        shared, project, c3_output, c5_output, dynamic_view, dynamic_samples, split, train_shapes, evaluation_shapes, fit
    )
    frames.append(_frame(dynamic_samples, dynamic_view, "PRISM_K_JOINT_AR", joint_prediction, joint_parameters, split, core_history_steps))
    destination = output / "PREDICTIONS" / head_view.head.head_id / split
    destination.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        info = str(frame["information_set"].iloc[0])
        model = str(frame["model"].iloc[0])
        frame.to_parquet(destination / f"{info}__{model}.parquet", index=False, compression="zstd")
    return frames


def _paired_bootstrap(diff: np.ndarray, entities: np.ndarray, block: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = pd.unique(entities)
    entity_blocks: dict[str, list[np.ndarray]] = {}
    for entity in unique:
        values = diff[entities == entity]
        entity_blocks[str(entity)] = [values[start : start + block] for start in range(0, len(values), block) if len(values[start : start + block])]
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_entities = rng.choice(unique, size=len(unique), replace=True) if len(unique) > 1 else unique
        total = 0.0
        count = 0
        for entity in selected_entities:
            blocks = entity_blocks[str(entity)]
            selected = rng.integers(0, len(blocks), size=len(blocks))
            for index in selected:
                block_values = blocks[int(index)]
                total += float(np.sum(block_values, dtype=np.float64))
                count += len(block_values)
        draws[replicate] = total / max(count, 1)
    return draws


def _holm(rows: list[dict[str, Any]], alpha: float) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_value"])
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * rows[index]["p_value"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
        rows[index]["holm_reject"] = running <= alpha


def _statistics(frames: list[pd.DataFrame], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    by_key: dict[tuple[str, str, str], dict[str, pd.DataFrame]] = {}
    for frame in frames:
        row = {
            "dataset": frame["dataset"].iloc[0],
            "task": frame["task"].iloc[0],
            "target_head": frame["target_head"].iloc[0],
            "split": frame["split"].iloc[0],
            "information_set": frame["information_set"].iloc[0],
            "model": frame["model"].iloc[0],
            "rows": len(frame),
            "parameter_count": int(frame["parameter_count"].iloc[0]),
            **regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()),
        }
        metrics.append(row)
        key = (row["target_head"], row["split"], row["information_set"])
        by_key.setdefault(key, {})[row["model"]] = frame
    metric_frame = pd.DataFrame(metrics)
    bootstrap_rows: list[dict[str, Any]] = []
    for key, models in by_key.items():
        comparisons = []
        if "PRISM_CHANNEL_SPECIFIC" in models and "PERSISTENCE" in models:
            comparisons.append(("PRISM_CHANNEL_SPECIFIC", "PERSISTENCE"))
        for model in ("AR", "PRISM_PHYSICS_FIRST", "PRISM_K_JOINT_AR"):
            if model in models and "PERSISTENCE" in models:
                comparisons.append((model, "PERSISTENCE"))
        for model in ("PRISM_PHYSICS_FIRST", "PRISM_K_JOINT_AR"):
            if model in models and "AR" in models:
                comparisons.append((model, "AR"))
        example = next(frame for frame in models.values())
        base_block = max(1, int(example["h_steps"].iloc[0]) + int(example["w_steps"].iloc[0]))
        history_block = max(1, int(math.ceil(int(example["core_history_steps"].iloc[0]) / 4)))
        blocks = sorted({base_block, 2 * base_block, history_block})
        for model, reference in comparisons:
            left = models[model].sort_values("sample_id")
            right = models[reference].sort_values("sample_id")
            if not np.array_equal(left["sample_id"].to_numpy(), right["sample_id"].to_numpy()):
                raise AssertionError(f"unpaired finalists: {key} {model} {reference}")
            diff = np.square(right["y_pred"].to_numpy() - right["y_true"].to_numpy()) - np.square(left["y_pred"].to_numpy() - left["y_true"].to_numpy())
            for block in blocks:
                seed_text = f"{key}|{model}|{reference}|{block}|{config['bootstrap_seed']}"
                seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
                draws = _paired_bootstrap(diff, left["profile_id"].astype(str).to_numpy(), block, int(config["bootstrap_replicates"]), seed)
                probability_positive = float(np.mean(draws > 0.0))
                p_value = min(1.0, 2.0 * min(probability_positive, 1.0 - probability_positive))
                bootstrap_rows.append({
                    "target_head": key[0], "split": key[1], "information_set": key[2], "model": model, "reference": reference,
                    "block_length": block, "mean_mse_improvement": float(np.mean(diff)), "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)), "positive_probability": probability_positive, "p_value": p_value,
                })
    _holm(bootstrap_rows, float(config["holm_alpha"]))
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    rank_frame = metric_frame.copy()
    rank_frame["rank"] = rank_frame.groupby(["target_head", "split", "information_set"])["mse"].rank(method="average")
    ranks = rank_frame.groupby(["split", "information_set", "model"], as_index=False).agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"), tasks=("target_head", "nunique"))
    return metric_frame, bootstrap_frame, ranks


def run_c6(
    shared: Path,
    project: Path,
    c2_output: Path,
    c3_output: Path,
    c4_output: Path,
    c5_output: Path,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    freeze_manifest = write_final_freeze_manifest(shared, project, c2_output, c3_output, c4_output, c5_output, output)
    if not freeze_manifest.is_file() or _load_json(freeze_manifest).get("status") != "FROZEN":
        raise RuntimeError("test access denied: final freeze manifest absent")
    frames: list[pd.DataFrame] = []
    failures = []
    for view in main_views(shared, "dynamic"):
        for split in ("test", "ood"):
            sample_path = shared / "sample_ids" / view.relative_root / f"{split}.parquet"
            if not sample_path.is_file():
                continue
            try:
                frames.extend(_evaluate_split(shared, project, c3_output, c4_output, c5_output, output, view, split))
            except Exception as error:
                failures.append({"target_head": view.head.head_id, "split": split, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
    if not frames:
        raise RuntimeError("no C6 finalist predictions were produced")
    metrics, bootstrap, ranks = _statistics(frames, _freeze(project)["c6"])
    metrics.to_csv(output / "CPU_FINAL_METRICS.csv", index=False)
    bootstrap.to_csv(output / "BOOTSTRAP_PAIRED.csv", index=False)
    ranks.to_csv(output / "CROSS_TASK_RANKS.csv", index=False)
    write_json(output / "C6_FAILURES.json", failures)
    report_lines = [
        "# PRISM Industrial CPU Final Report",
        "",
        f"Status: `{'PASS' if not failures else 'PASS_WITH_RETAINED_FAILURES'}`",
        "",
        "The C6 test/OOD evaluation used only the result-independent finalists registered before development results were inspected.",
        "Input-only and dynamic leaderboards remain separate. All test predictions were generated after FINAL_FREEZE_MANIFEST.json was written.",
        "",
        "## Cross-task ranks",
        "",
        ranks.to_markdown(index=False),
        "",
        "## Test and OOD metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Statistical audit",
        "",
        f"Paired block-bootstrap comparisons: {len(bootstrap)}; replicates per comparison: {_freeze(project)['c6']['bootstrap_replicates']}; Holm correction applied.",
        "",
        "Retained failures are listed in `C6_FAILURES.json` and are not silently discarded.",
    ]
    (output / "CPU_FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    decision = {
        "status": "PASS" if not failures else "PASS_WITH_RETAINED_FAILURES",
        "stage": "C6_CPU_FINALISTS_AND_STATISTICS",
        "finalist_policy": _freeze(project)["c6"]["finalist_policy"],
        "test_accessed": True,
        "prediction_frames": len(frames),
        "metric_rows": len(metrics),
        "bootstrap_rows": len(bootstrap),
        "failures": len(failures),
        "final_freeze_sha256": sha256_file(freeze_manifest),
    }
    write_json(output / "CPU_FINAL_DECISION.json", decision)
    return decision
