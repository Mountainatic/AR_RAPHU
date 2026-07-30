"""Parallel, resumable Stage-1 linear multiscale scan."""

from __future__ import annotations

import csv
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .data_loader import load_workbook_data
from .pipeline import evaluate_profile, profile_task_id, profile_variants
from .runtime import atomic_json, load_config, task_is_complete
from .timebase import Timebase


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(payload["root"])
    result_path = root / "results" / "stage1" / "profiles" / payload["task_id"]
    result_json = result_path / "result.json"
    try:
        config, config_sha = load_config(payload["config_path"])
        workbook = load_workbook_data(
            payload["data_path"],
            required_sheets=config["data"]["required_sheets"],
            required_columns=config["data"]["required_columns"],
        )
        if task_is_complete(
            result_json,
            config_sha256=config_sha,
            data_sha256=workbook.sha256,
            sample_period_sec=payload["sample_period_sec"],
        ):
            return {"task_id": payload["task_id"], "status": "SKIPPED_COMPLETE"}
        starts = {
            sheet: max(values) + 1 if values else 0
            for sheet, values in config["data"]["frozen_breakpoints"].items()
        }
        result, arrays = evaluate_profile(
            workbook,
            starts=starts,
            profile=payload["profile"],
            variant=payload["variant"],
            config=config,
            timebase=Timebase(payload["sample_period_sec"]),
        )
        result.update(
            {
                "config_sha256": config_sha,
                "data_sha256": workbook.sha256,
                "sample_period_sec": payload["sample_period_sec"],
            }
        )
        _atomic_npz(result_path / "arrays.npz", arrays)
        atomic_json(result_json, result)
        return {"task_id": payload["task_id"], "status": "COMPLETED"}
    except Exception as exc:  # isolate profile failures by design
        failure = {
            "task_id": payload["task_id"],
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(result_path / "failure.json", failure)
        return failure


def task_payloads(
    root: Path,
    config_path: Path,
    data_path: Path,
    sample_period_sec: float,
) -> list[dict[str, Any]]:
    config, _ = load_config(config_path)
    tasks = []
    for profile in config["profiles"]:
        for variant in profile_variants(profile, config):
            tasks.append(
                {
                    "root": str(root),
                    "config_path": str(config_path),
                    "data_path": str(data_path),
                    "sample_period_sec": float(sample_period_sec),
                    "profile": profile,
                    "variant": variant,
                    "task_id": profile_task_id(profile["id"], variant),
                }
            )
    return tasks


def aggregate_stage1(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    profile_root = root / "results" / "stage1" / "profiles"
    results = []
    failures = []
    for task_dir in sorted(profile_root.glob("*")):
        result_file = task_dir / "result.json"
        failure_file = task_dir / "failure.json"
        if result_file.is_file():
            results.append(json.loads(result_file.read_text(encoding="utf-8")))
        elif failure_file.is_file():
            failures.append(json.loads(failure_file.read_text(encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "task_id": result["task_id"],
            "profile": result["profile"]["id"],
            "channel": result["profile"]["channel"],
            "variant": result["variant"],
            "confirmatory": result["confirmatory"],
            "horizon_min": result["profile"]["horizon_min"],
            "target_window_min": result["profile"]["target_window_min"],
            "history_min": result["profile"]["history_min"],
            "direction_1_improvement": result["directions"][0]["models"][
                "Q1_CHANNEL_LINEAR"
            ]["relative_improvement"],
            "direction_2_improvement": result["directions"][1]["models"][
                "Q1_CHANNEL_LINEAR"
            ]["relative_improvement"],
            "pooled_q_improvement": result["pooled"]["q_improvement"],
            "q_bootstrap_positive_probability": result["pooled"][
                "q_bootstrap"
            ]["positive_probability"],
            "pooled_conditional_improvement": result["pooled"][
                "conditional_improvement"
            ],
            "conditional_bootstrap_positive_probability": result["pooled"][
                "conditional_bootstrap"
            ]["positive_probability"],
            "S1_candidate": result["gates"]["S1_candidate"],
            "S1_status": result["gates"]["S1_status"],
        }
        for direction_index, direction in enumerate(result["directions"], start=1):
            row[f"d{direction_index}_q_rmse"] = direction["models"][
                "Q1_CHANNEL_LINEAR"
            ]["RMSE"]
            row[f"d{direction_index}_ar_scale_rmse"] = direction["models"][
                "B3_AR_SCALE"
            ]["RMSE"]
            row[f"d{direction_index}_frozen_arq_improvement"] = direction[
                "models"
            ]["Q2_FROZEN_AR_PLUS_Q"]["relative_improvement"]
        rows.append(row)
    rows.sort(key=lambda row: (row["channel"], row["profile"], row["variant"]))
    csv_path = root / "results" / "STAGE1_SCALE_SCAN.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    candidates = []
    maximum = int(config["gates"]["maximum_stage2_candidates_per_channel"])
    for channel in config["branches"]:
        eligible = [
            row
            for row in rows
            if row["channel"] == channel and row["S1_candidate"]
        ]
        eligible.sort(key=lambda row: -float(row["pooled_q_improvement"]))
        candidates.extend(row["task_id"] for row in eligible[:maximum])
    summary = {
        "status": "COMPLETED" if not failures else "COMPLETED_WITH_FAILURES",
        "completed_tasks": len(results),
        "failed_tasks": len(failures),
        "failures": failures,
        "stage2_candidates": candidates,
    }
    atomic_json(root / "results" / "stage1" / "summary.json", summary)
    atomic_json(
        root / "results" / "checkpoints" / "latest.json",
        {"completed_stage": "STAGE1", **summary},
    )
    _plot_heatmap(root, rows)
    return summary


def _plot_heatmap(root: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    labels = [row["task_id"] for row in rows]
    values = np.array(
        [
            [
                row["direction_1_improvement"],
                row["direction_2_improvement"],
                row["pooled_q_improvement"],
                row["pooled_conditional_improvement"],
            ]
            for row in rows
        ]
    )
    figure, axis = plt.subplots(figsize=(9, max(7, len(rows) * 0.27)))
    image = axis.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    axis.set_xticks(range(4), ["Rod1→Rod2", "Rod2→Rod1", "Pooled Q", "AR|Q"])
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    axis.set_title("Stage 1 relative MSE improvement")
    figure.colorbar(image, ax=axis, label="Improvement")
    figure.tight_layout()
    figure.savefig(root / "results" / "STAGE1_SCALE_HEATMAP.png", dpi=180)
    plt.close(figure)


def run_stage1(
    *,
    root: Path,
    config_path: Path,
    data_path: Path,
    sample_period_sec: float,
    n_jobs: int,
    task_ids: set[str] | None = None,
) -> dict[str, Any]:
    config, _ = load_config(config_path)
    tasks = task_payloads(root, config_path, data_path, sample_period_sec)
    if task_ids:
        tasks = [task for task in tasks if task["task_id"] in task_ids]
        missing = task_ids - {task["task_id"] for task in tasks}
        if missing:
            raise ValueError(f"UNKNOWN_TASK_IDS:{sorted(missing)}")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_map = {executor.submit(_run_one, task): task for task in tasks}
        for future in as_completed(future_map):
            status = future.result()
            print(
                f"STAGE1_TASK task={status['task_id']} status={status['status']}",
                flush=True,
            )
    return aggregate_stage1(root, config)
