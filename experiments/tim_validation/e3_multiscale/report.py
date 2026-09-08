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


PRIMARY_TASKS = ("CZ_H4", "TEP_G12", "DEB_C4")
OPTIONAL_TASKS = ("PMSM_PM5",)
CANONICAL_TASK = {"TEP_G12": "TEP_H0"}


def _canonical_task(task: str) -> str:
    """Expose the frozen public task identity while retaining legacy run paths."""

    return CANONICAL_TASK.get(task, task)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register_cz_single_scale_block(
    run_root: Path, frozen_registry: Path | None
) -> dict[str, Any] | None:
    """Record the non-identifiability of a scale ablation with one legal scale."""

    path = run_root / "CZ_H4" / "PROTOCOL_STATUS.json"
    if path.is_file():
        return _read(path)
    if frozen_registry is None:
        return None
    with frozen_registry.open(encoding="utf-8", newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row.get("task") == "CZ_H4"]
    if len(matches) != 1:
        raise RuntimeError(f"expected one CZ_H4 registry row, got {len(matches)}")
    row = matches[0]
    histories = json.loads(row["history_steps"])
    if len(histories) != 1:
        return None
    result = {
        "status": "PROTOCOL_BLOCKED",
        "task": "CZ_H4",
        "head": row["head"],
        "H": int(row["H_steps"]),
        "W": int(row["W_steps"]),
        "registered_histories": histories,
        "reason": "SINGLE_REGISTERED_HISTORY_NO_UNIFORM_VS_SCALE_AWARE_CONTRAST",
        "scale_arms_identical_by_construction": True,
        "test_accessed": False,
        "frozen_registry_sha256": _sha256(frozen_registry),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


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


def build_report(
    run_root: Path, output: Path, frozen_registry: Path | None = None
) -> dict[str, Any]:
    cz_status = _register_cz_single_scale_block(run_root, frozen_registry)
    completed_tasks = [
        task for task in TASKS if (run_root / task / "TASK_RESULT.json").is_file()
    ]
    task_results = [_read(run_root / task / "TASK_RESULT.json") for task in completed_tasks]
    if not task_results:
        raise RuntimeError("no completed E3 task results")
    aggregate = []
    folds = []
    scales = []
    efficiency = []
    for result in task_results:
        task = result["task"]
        canonical_task = _canonical_task(task)
        selection = _read(run_root / task / "UNIFORM_HISTORY_SELECTION.json")
        registered_histories = [int(value) for value in selection["common_histories"]]
        budget = _read(run_root / task / "budget_manifest.json")
        for arm in ("uniform", "multiscale"):
            record = result[arm]
            aggregate.append(
                {
                    "task": canonical_task,
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
                        "task": canonical_task,
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
                        "task": canonical_task,
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
                    "task": canonical_task,
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
        _canonical_task(result["task"]): float(result["relative_multiscale_gain"])
        for result in task_results
    }
    positive = sum(value > 0.0 for value in gains.values())
    budget_statuses = {
        _canonical_task(task): _read(run_root / task / "budget_manifest.json")["budget_status"]
        for task in completed_tasks
    }
    task_status = {
        _canonical_task(task): ("COMPLETED" if task in completed_tasks else "NOT_RUN")
        for task in TASKS
    }
    task_status["CZ_H4"] = (
        str(cz_status["status"]) if cz_status is not None else "NOT_RUN"
    )
    overall = (
        "COMPLETED"
        if all(task_status.get(_canonical_task(task)) == "COMPLETED" for task in PRIMARY_TASKS)
        else "PARTIAL"
    )
    interpretation = (
        "# E3 Budget-Controlled Multiscale Ablation\n\n"
        f"Status: `{overall}`.\n\n"
        "Uniform and "
        "per-channel-scale arms used identical task heads, splits, preprocessing, stage "
        "logic, and test scoring support. Scale selection used development inner folds "
        "only; both checkpoints were sealed before isolated test inference.\n\n"
        + "Relative multiscale RMSE gains (positive favors multiscale):\n\n"
        + "\n".join(f"- {task}: {gain:.6%}" for task, gain in gains.items())
        + "\n\nBudget classifications:\n\n"
        + "\n".join(f"- {task}: {status}" for task, status in budget_statuses.items())
        + f"\n\nMultiscale improved RMSE on {positive}/{len(gains)} completed tasks. Small or negative "
        "gains must not be described as a major multiscale contribution.\n\n"
        + (
            "CZ_H4 is `PROTOCOL_BLOCKED`: its frozen registry contains only history "
            "256, so uniform and scale-aware arms would be identical by construction. "
            "No CZ test data were accessed for E3.\n"
            if cz_status is not None
            else "CZ_H4 has no valid E3 result.\n"
        )
    )
    (output / "E3_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    manifest = {
        "status": overall,
        "tasks": completed_tasks,
        "task_status": task_status,
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
    parser.add_argument("--frozen-registry", type=Path)
    args = parser.parse_args()
    build_report(
        args.run_root.resolve(),
        args.output.resolve(),
        args.frozen_registry.resolve() if args.frozen_registry else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
