"""Build the E3 machine-readable tables and concise interpretation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .runner import FULL_MODEL, TASKS


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stage_result(root: Path, arm: str, task_result: dict[str, Any], stage: str) -> dict[str, Any]:
    record = task_result[arm]
    head = str(record["target_head"])
    proxy = str(record["proxy_policy"])
    path = root / task_result["task"] / arm / "results" / "DEVELOPMENT" / stage / head
    if stage == "A":
        path = path / str(record["availability_scenario"]) / proxy
    else:
        path = path / proxy
    return _read(path / "RESULT.json")


def _scale_class(history: int, registered: list[int]) -> str:
    ordered = sorted(set(registered))
    if history == ordered[0]:
        return "fast"
    if history == ordered[-1]:
        return "slow"
    return "mid"


def build_report(run_root: Path, output: Path) -> dict[str, Any]:
    task_results = [_read(run_root / task / "TASK_RESULT.json") for task in TASKS]
    aggregate = []
    folds = []
    scales = []
    efficiency = []
    for result in task_results:
        task = result["task"]
        selection = _read(run_root / task / "UNIFORM_HISTORY_SELECTION.json")
        registered_histories = [int(value) for value in selection["common_histories"]]
        budget = _read(run_root / task / "budget_manifest.json")
        for arm in ("uniform", "multiscale"):
            record = result[arm]
            aggregate.append(
                {
                    "task": task,
                    "dataset": result["dataset"],
                    "arm": arm,
                    "model": FULL_MODEL,
                    "rmse": record["rmse"],
                    "mae": record["mae"],
                    "r2_delta": record["r2_delta"],
                    "r2_level_reconstructed": record["r2_level_reconstructed"],
                    "rows": record["rows"],
                    "support_hash": record["scoring_support_hash"],
                }
            )
            a_result = _stage_result(run_root, arm, result, "A")
            for fold, mse in enumerate(a_result["final_selected_fold_losses"]):
                folds.append(
                    {
                        "task": task,
                        "arm": arm,
                        "fold_kind": "development_inner_fold",
                        "fold": fold,
                        "mse": mse,
                        "rmse": math.sqrt(float(mse)),
                        "outer_test_used": False,
                    }
                )
            k_root = (
                run_root
                / task
                / arm
                / "results"
                / "DEVELOPMENT"
                / "K"
                / str(record["target_head"])
                / str(record["proxy_policy"])
            )
            for path in sorted(k_root.glob("*/RESULT.json")):
                channel = _read(path)
                history = int(channel["selected_profile_history_steps"])
                scales.append(
                    {
                        "task": task,
                        "arm": arm,
                        "channel": channel["channel"],
                        "active": channel["active"],
                        "history": history,
                        "scale_class": _scale_class(history, registered_histories),
                    }
                )
            fits = budget[f"{arm}_candidate_fit_attempts"]["total"]
            efficiency.append(
                {
                    "task": task,
                    "arm": arm,
                    "candidate_fit_attempts": fits,
                    "fair_budget": budget["fair_budget"],
                    "budget_status": budget["budget_status"],
                    "strict_equal_budget": budget["strict_equal_budget"],
                    "test_rmse": record["rmse"],
                    "rmse_per_candidate_fit": float(record["rmse"]) / max(1, fits),
                    "multiscale_gain_per_fair_budget_fit": (
                        result["delta_rmse_uniform_minus_multiscale"]
                        / max(1, budget["fair_budget"])
                    ),
                }
            )

    _write_csv(output / "aggregate.csv", aggregate)
    _write_csv(output / "per_fold.csv", folds)
    _write_csv(output / "selected_scales.csv", scales)
    _write_csv(output / "efficiency.csv", efficiency)
    gains = {
        result["task"]: float(result["relative_multiscale_gain"])
        for result in task_results
    }
    positive = sum(value > 0.0 for value in gains.values())
    budget_statuses = {
        task: _read(run_root / task / "budget_manifest.json")["budget_status"]
        for task in TASKS
    }
    interpretation = (
        "# E3 Budget-Controlled Multiscale Ablation\n\n"
        "Status: `COMPLETED_WITH_USER_AUTHORIZED_CHECKPOINT_A_OVERRIDE`.\n\n"
        "Checkpoint A had failed before E3. The user explicitly authorized E3; this "
        "experiment does not reverse or weaken `MODEL REVIEW REQUIRED`. Private CZ data "
        "remained excluded under the repository's standing 2026-07-25 instruction.\n\n"
        "The registered public tasks were TEP_G12, DEB_C4, and PMSM_PM5. Uniform and "
        "per-channel-scale arms used identical task heads, splits, preprocessing, stage "
        "logic, and test scoring support. Scale selection used development inner folds "
        "only; both checkpoints were sealed before isolated test inference.\n\n"
        + "Relative multiscale RMSE gains (positive favors multiscale):\n\n"
        + "\n".join(f"- {task}: {gain:.6%}" for task, gain in gains.items())
        + "\n\nBudget classifications:\n\n"
        + "\n".join(f"- {task}: {status}" for task, status in budget_statuses.items())
        + f"\n\nMultiscale improved RMSE on {positive}/{len(gains)} tasks. Small or negative "
        "gains must not be described as a major multiscale contribution.\n"
    )
    (output / "E3_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    manifest = {
        "status": "COMPLETED",
        "tasks": list(TASKS),
        "relative_multiscale_gain": gains,
        "budget_status": budget_statuses,
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.run_root.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
