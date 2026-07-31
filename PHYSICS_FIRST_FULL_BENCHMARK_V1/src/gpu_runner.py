"""Task orchestration for reproducible GPU screening and confirmation."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .gpu_common import (
    atomic_json,
    atomic_npz,
    environment_snapshot,
    regression_metrics,
    sha256_array,
    sha256_file,
    write_csv,
)
from .gpu_data import (
    DirectionData,
    Standardizer,
    TargetScaler,
    chronological_folds,
    list_directions,
    load_direction,
    load_k_predictions,
    make_model_view,
    matured_residual_history,
    resolve_shared_root,
    simple_train_validation_split,
    validate_shared_dataset,
)
from .gpu_models import build_model, parameter_count
from .gpu_training import (
    TrainConfig,
    fit_fixed_epochs,
    fit_with_early_stopping,
    predict,
    safe_fit,
    write_training_trace,
)


@dataclass(frozen=True)
class TaskSpec:
    model_id: str
    architecture: str
    mode: str
    stage: str
    label: str
    parameters: dict[str, Any]
    batch_size: int
    max_parameters: int


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def task_specs(config: dict[str, Any], stage: str, selected: set[str] | None = None) -> list[TaskSpec]:
    result = []
    for raw in config["models"]:
        if raw["stage"] != stage:
            continue
        if selected and raw["id"] not in selected:
            continue
        result.append(
            TaskSpec(
                model_id=raw["id"],
                architecture=raw["architecture"],
                mode=raw["mode"],
                stage=raw["stage"],
                label=raw.get("label", "STANDARD_BASELINE"),
                parameters=dict(raw.get("parameters", {})),
                batch_size=int(raw.get("batch_size", config["training"]["batch_size"])),
                max_parameters=int(raw.get("max_parameters", config.get("max_parameters", 200000))),
            )
        )
    return result


def _precision_config(
    config: dict[str, Any],
    task: TaskSpec,
    *,
    stage: str,
    epoch_override: int | None,
    workers_override: int | None,
) -> TrainConfig:
    training = dict(config["training"])
    precision = config["precision"]["final" if stage == "finalists" else "screening"]
    return TrainConfig(
        epochs=int(epoch_override or training["epochs"]),
        patience=int(training["patience"]),
        min_delta=float(training["min_delta"]),
        batch_size=task.batch_size,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        warmup_epochs=int(training["warmup_epochs"]),
        num_workers=int(workers_override if workers_override is not None else training["num_workers"]),
        dtype=str(precision["dtype"]),
        tf32=bool(precision["tf32"]),
        deterministic=bool(precision.get("deterministic", False)),
    )


def _scaled_views(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Standardizer, TargetScaler]:
    x_scaler = Standardizer.fit(x_train[train_indices], axes=(0, 1))
    y_scaler = TargetScaler.fit(y_train[train_indices])
    return (
        x_scaler.transform(x_train),
        x_scaler.transform(x_test),
        y_scaler.transform(y_train),
        x_scaler,
        y_scaler,
    )


def _task_arrays(
    direction: DirectionData,
    task: TaskSpec,
    *,
    cpu_results_root: str | Path | None,
    residual_history_rows: int,
) -> dict[str, Any]:
    if task.mode in {"input", "dynamic"}:
        return {
            "x_train": make_model_view(direction.train, task.mode),
            "x_test": make_model_view(direction.test, task.mode),
            "y_train": direction.train.target_z,
            "y_test": direction.test.target_z,
            "base_test": np.zeros(len(direction.test.target_z), dtype=np.float64),
            "train_available": np.ones(len(direction.train.target_z), dtype=bool),
            "test_available": np.ones(len(direction.test.target_z), dtype=bool),
        }
    if task.mode == "residual":
        if cpu_results_root is None:
            raise RuntimeError("CPU_RESULTS_REQUIRED_FOR_RESIDUAL_MODE")
        k_train, k_test = load_k_predictions(cpu_results_root, direction)
        maturity_rows = int(direction.metadata["maturity_rows"])
        x_train, train_available = matured_residual_history(
            direction.train.target_z,
            k_train,
            maturity_rows=maturity_rows,
            history_rows=residual_history_rows,
        )
        x_test, test_available = matured_residual_history(
            direction.test.target_z,
            k_test,
            maturity_rows=maturity_rows,
            history_rows=residual_history_rows,
        )
        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": direction.train.target_z - k_train,
            "y_test": direction.test.target_z - k_test,
            "base_test": k_test,
            "train_available": train_available,
            "test_available": test_available,
        }
    raise ValueError(f"UNKNOWN_TASK_MODE:{task.mode}")


def _build_checked_model(task: TaskSpec, sequence_length: int, input_dim: int) -> torch.nn.Module:
    model = build_model(
        task.architecture,
        sequence_length=sequence_length,
        input_dim=input_dim,
        parameters=task.parameters,
    )
    count = parameter_count(model)
    if count > task.max_parameters and task.label != "LARGE_REPRODUCTION":
        raise RuntimeError(f"PARAMETER_BUDGET_EXCEEDED:{count}:{task.max_parameters}")
    return model


def _selection_epochs(
    task: TaskSpec,
    arrays: dict[str, Any],
    direction: DirectionData,
    *,
    device: torch.device,
    train_config: TrainConfig,
    seed: int,
    strict_folds: bool,
    task_root: Path,
) -> tuple[int, list[dict[str, Any]], int]:
    x_train = arrays["x_train"]
    y_train = arrays["y_train"]
    available = arrays["train_available"]
    selection_rows = np.flatnonzero(available)
    if len(selection_rows) < 128:
        raise RuntimeError(f"INSUFFICIENT_AVAILABLE_TRAIN_ROWS:{len(selection_rows)}")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if strict_folds:
        folds = list(
            chronological_folds(
                len(x_train),
                direction.metadata["_inner_folds"],
                int(direction.metadata["_purge_rows"]),
            )
        )
    if not folds:
        train_idx, valid_idx = simple_train_validation_split(
            len(x_train),
            validation_fraction=0.15,
            purge_rows=int(direction.metadata["_purge_rows"]),
        )
        folds = [(train_idx, valid_idx)]
    fold_records: list[dict[str, Any]] = []
    best_epochs: list[int] = []
    parameter_total = 0
    for fold_index, (raw_train, raw_valid) in enumerate(folds):
        train_idx = raw_train[available[raw_train]]
        valid_idx = raw_valid[available[raw_valid]]
        if len(train_idx) < 64 or len(valid_idx) < 16:
            continue
        x_scaled, _, y_scaled, _, _ = _scaled_views(x_train, x_train, y_train, train_idx)
        parameter_total = parameter_count(
            _build_checked_model(task, x_scaled.shape[1], x_scaled.shape[2])
        )
        checkpoint = task_root / "selection" / f"fold_{fold_index}.pt"

        def perform(config_for_attempt: TrainConfig):
            return fit_with_early_stopping(
                model=_build_checked_model(task, x_scaled.shape[1], x_scaled.shape[2]),
                x=x_scaled,
                y=y_scaled,
                train_indices=train_idx,
                validation_indices=valid_idx,
                device=device,
                config=config_for_attempt,
                seed=seed + fold_index * 1009,
                checkpoint_path=checkpoint,
            )

        result = safe_fit(perform, initial_config=train_config)
        write_training_trace(task_root / "selection" / f"fold_{fold_index}.json", result)
        best_epochs.append(result.best_epoch)
        fold_records.append(
            {
                "fold": fold_index,
                "train_rows": len(train_idx),
                "validation_rows": len(valid_idx),
                "best_epoch": result.best_epoch,
                "best_validation_loss_scaled": result.best_validation_loss,
                "train_seconds": result.train_seconds,
                "dtype_used": result.dtype_used,
                "fallback_reason": result.fallback_reason,
            }
        )
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not best_epochs:
        raise RuntimeError("NO_VALID_SELECTION_FOLD")
    chosen = int(max(1, round(statistics.median(best_epochs))))
    return chosen, fold_records, parameter_total


def run_task(
    *,
    project_root: Path,
    shared_root: Path,
    cpu_results_root: Path | None,
    results_root: Path,
    direction_name: str,
    task: TaskSpec,
    seed: int,
    device: torch.device,
    config: dict[str, Any],
    epoch_override: int | None,
    workers_override: int | None,
    strict_folds: bool,
    force: bool,
) -> dict[str, Any]:
    task_root = results_root / "tasks" / task.stage / direction_name / task.model_id / f"seed_{seed}"
    result_path = task_root / "result.json"
    if result_path.is_file() and not force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            return existing
    task_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    direction = load_direction(shared_root, direction_name, validate=False)
    protocol = json.loads((shared_root / "BENCHMARK_PROTOCOL.json").read_text(encoding="utf-8"))
    direction.metadata["_inner_folds"] = protocol["inner_folds"]
    direction.metadata["_purge_rows"] = int(
        round(float(protocol["purge_min"]) * 60.0 / float(protocol["cadence_sec"]))
    )
    arrays = _task_arrays(
        direction,
        task,
        cpu_results_root=cpu_results_root,
        residual_history_rows=int(config["residual_history_rows"]),
    )
    train_config = _precision_config(
        config,
        task,
        stage=task.stage,
        epoch_override=epoch_override,
        workers_override=workers_override,
    )
    chosen_epochs, fold_records, count = _selection_epochs(
        task,
        arrays,
        direction,
        device=device,
        train_config=train_config,
        seed=seed,
        strict_folds=strict_folds,
        task_root=task_root,
    )
    full_train_idx = np.flatnonzero(arrays["train_available"])
    x_train_scaled, x_test_scaled, y_train_scaled, x_scaler, y_scaler = _scaled_views(
        arrays["x_train"], arrays["x_test"], arrays["y_train"], full_train_idx
    )

    def perform_final(config_for_attempt: TrainConfig):
        return fit_fixed_epochs(
            model=_build_checked_model(task, x_train_scaled.shape[1], x_train_scaled.shape[2]),
            x=x_train_scaled,
            y=y_train_scaled,
            train_indices=full_train_idx,
            device=device,
            config=config_for_attempt,
            epochs=chosen_epochs,
            seed=seed + 7919,
            checkpoint_path=task_root / "best_model.pt",
        )

    final = safe_fit(perform_final, initial_config=train_config)
    write_training_trace(task_root / "final_training.json", final)
    prediction_scaled, inference_ms = predict(
        final.model,
        x_test_scaled,
        device=device,
        batch_size=max(256, train_config.batch_size),
        num_workers=min(2, train_config.num_workers),
    )
    correction = y_scaler.inverse(prediction_scaled)
    prediction = arrays["base_test"] + correction
    evaluation_mask = direction.test.evaluation_mask & arrays["test_available"]
    if int(evaluation_mask.sum()) < 16:
        raise RuntimeError(f"INSUFFICIENT_EVALUATION_ROWS:{int(evaluation_mask.sum())}")
    y_true = direction.test.target_z
    metrics = regression_metrics(y_true[evaluation_mask], prediction[evaluation_mask])
    persistence = regression_metrics(
        y_true[evaluation_mask], np.zeros(int(evaluation_mask.sum()), dtype=np.float64)
    )
    relative_persistence = 1.0 - metrics["MSE"] / max(persistence["MSE"], np.finfo(float).eps)
    atomic_npz(
        task_root / "predictions.npz",
        sample_id=direction.test.sample_id,
        y_true=y_true.astype(np.float64),
        y_pred=prediction.astype(np.float32),
        evaluation_mask=evaluation_mask,
        direction=np.asarray(direction_name),
        model=np.asarray(task.model_id),
        seed=np.asarray(seed, dtype=np.int64),
        training_dtype=np.asarray(final.dtype_used),
        parameter_count=np.asarray(count, dtype=np.int64),
    )
    record = {
        "status": "PASS",
        "stage": task.stage,
        "direction": direction_name,
        "model_id": task.model_id,
        "architecture": task.architecture,
        "mode": task.mode,
        "implementation_label": task.label,
        "seed": seed,
        "parameter_count": count,
        "chosen_epochs": chosen_epochs,
        "selection_folds": fold_records,
        "selection_validation_mse_scaled_mean": float(
            np.mean([row["best_validation_loss_scaled"] for row in fold_records])
        ),
        "selection_validation_mse_scaled_worst": float(
            np.max([row["best_validation_loss_scaled"] for row in fold_records])
        ),
        "training_dtype": final.dtype_used,
        "tf32": train_config.tf32,
        "dataloader_workers": train_config.num_workers,
        "fallback_reason": final.fallback_reason,
        "train_seconds_final": final.train_seconds,
        "train_seconds_total": float(time.perf_counter() - started),
        "infer_ms_per_1000": inference_ms,
        "peak_memory_bytes": final.peak_memory_bytes,
        "evaluation_rows": int(evaluation_mask.sum()),
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"],
        "relative_persistence": float(relative_persistence),
        "prediction_sha256": sha256_array(prediction.astype(np.float32)),
        "test_sample_id_sha256": sha256_array(direction.test.sample_id),
        "x_scaler_mean": x_scaler.mean.reshape(-1).tolist(),
        "x_scaler_scale": x_scaler.scale.reshape(-1).tolist(),
        "target_scaler": {"mean": y_scaler.mean, "scale": y_scaler.scale},
        "shared_protocol_sha256": sha256_file(shared_root / "BENCHMARK_PROTOCOL.json"),
    }
    atomic_json(result_path, record)
    del final
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def aggregate_results(results_root: str | Path) -> list[dict[str, Any]]:
    root = Path(results_root)
    records: list[dict[str, Any]] = []
    for path in sorted((root / "tasks").glob("*/*/*/seed_*/result.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append(record)
    write_csv(root / "GPU_ALL_RUNS.csv", records)

    summaries: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") != "PASS":
            continue
        key = (record["stage"], record["model_id"], record["mode"])
        groups.setdefault(key, []).append(record)

    for (stage, model_id, mode), group in groups.items():
        by_seed: dict[int, list[dict[str, Any]]] = {}
        for row in group:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        pooled_seed_mse: list[float] = []
        complete_seed_count = 0
        for seed_rows in by_seed.values():
            total_rows = sum(int(row["evaluation_rows"]) for row in seed_rows)
            if total_rows <= 0:
                continue
            pooled = sum(float(row["MSE"]) * int(row["evaluation_rows"]) for row in seed_rows) / total_rows
            pooled_seed_mse.append(float(pooled))
            if len({row["direction"] for row in seed_rows}) >= 2:
                complete_seed_count += 1
        if not pooled_seed_mse:
            continue
        direction_medians = {
            direction: float(np.median([row["MSE"] for row in group if row["direction"] == direction]))
            for direction in sorted({row["direction"] for row in group})
        }
        pooled_array = np.asarray(pooled_seed_mse, dtype=np.float64)
        summaries.append(
            {
                "stage": stage,
                "model_id": model_id,
                "mode": mode,
                "runs": len(group),
                "directions": len(direction_medians),
                "seeds": len(by_seed),
                "complete_two_direction_seeds": complete_seed_count,
                "pooled_MSE_seed_median": float(np.median(pooled_array)),
                "pooled_MSE_seed_IQR": float(np.quantile(pooled_array, 0.75) - np.quantile(pooled_array, 0.25)),
                "pooled_RMSE_seed_median": float(np.sqrt(np.median(pooled_array))),
                "direction_worst_MSE": float(max(direction_medians.values())),
                "direction_best_MSE": float(min(direction_medians.values())),
                "direction_MSE_json": json.dumps(direction_medians, sort_keys=True),
                "selection_validation_mse_scaled_median": float(
                    np.median(
                        [
                            row.get(
                                "selection_validation_mse_scaled_mean",
                                float(
                                    np.mean(
                                        [
                                            fold["best_validation_loss_scaled"]
                                            for fold in row.get("selection_folds", [])
                                        ]
                                    )
                                ),
                            )
                            for row in group
                            if row.get("selection_folds")
                        ]
                    )
                ),
                "R2_run_median": float(np.median([row["R2"] for row in group])),
                "relative_persistence_run_median": float(np.median([row["relative_persistence"] for row in group])),
                "parameter_count": int(np.median([row["parameter_count"] for row in group])),
                "train_seconds_median": float(np.median([row["train_seconds_total"] for row in group])),
                "infer_ms_per_1000_median": float(np.median([row["infer_ms_per_1000"] for row in group])),
                "peak_memory_bytes_max": int(max(row["peak_memory_bytes"] for row in group)),
                "implementation_label": group[0]["implementation_label"],
            }
        )
    summaries.sort(key=lambda row: (row["mode"], row["pooled_MSE_seed_median"]))
    write_csv(root / "GPU_MODEL_SUMMARY.csv", summaries)
    write_csv(root / "GPU_CORE_MODELS.csv", [row for row in summaries if row["stage"] == "core"])
    write_csv(root / "GPU_FRONTIER_MODELS.csv", [row for row in summaries if row["stage"] == "frontier"])
    write_csv(root / "GPU_FINALISTS.csv", [row for row in summaries if row["stage"] == "finalists"])

    eligible = [
        row for row in summaries
        if row["stage"] in {"core", "frontier"}
        and row["directions"] >= 2
        and row["implementation_label"]
        != "NONCAUSAL_CONTROL_EXCLUDED_FROM_ONLINE_BOARD"
    ]
    provisional = {
        "input": sorted(
            [row for row in eligible if row["mode"] == "input"],
            key=lambda row: (
                row["selection_validation_mse_scaled_median"],
                row["parameter_count"],
                row["model_id"],
            ),
        )[:6],
        "dynamic": sorted(
            [row for row in eligible if row["mode"] in {"dynamic", "residual"}],
            key=lambda row: (
                row["selection_validation_mse_scaled_median"],
                row["parameter_count"],
                row["model_id"],
            ),
        )[:6],
    }
    decision = {
        "schema": "PHYSICS_FIRST_GPU_DECISION_V1",
        "status": "PROVISIONAL_SCREENING" if eligible else "NO_COMPLETE_SCREENING_RESULTS",
        "formal_cpu_main_model_unchanged": "K-to-Residual-AR",
        "provisional_top_six_by_leaderboard": provisional,
        "note": (
            "GPU rankings are competitive baselines/ablations. Final paired block bootstrap "
            "is performed later on saved per-sample predictions in the CPU aggregation pass."
        ),
    }
    atomic_json(root / "GPU_FINAL_DECISION.json", decision)
    lines = [
        "# GPU Benchmark Report",
        "",
        f"- Completed PASS runs: **{sum(1 for row in records if row.get('status') == 'PASS')}**",
        f"- Failed/partial runs: **{sum(1 for row in records if row.get('status') != 'PASS')}**",
        f"- Aggregated model rows: **{len(summaries)}**",
        "- Formal main model remains the CPU `K → Residual AR` model.",
        "- `_adapted` methods are protocol-faithful compact adaptations, not exact paper reproductions.",
        "",
    ]
    for mode in ("input", "dynamic"):
        lines.extend([f"## {mode.capitalize()} leaderboard", ""])
        rows = provisional.get(mode, [])
        if not rows:
            lines.extend(["No complete two-direction screening result yet.", ""])
            continue
        lines.extend([
            "| Rank | Model | Validation MSE | Pooled test MSE | Worst-direction MSE | Params |",
            "|---:|---|---:|---:|---:|---:|",
        ])
        for rank, row in enumerate(rows, 1):
            lines.append(
                f"| {rank} | {row['model_id']} | "
                f"{row['selection_validation_mse_scaled_median']:.8g} | "
                f"{row['pooled_MSE_seed_median']:.8g} | "
                f"{row['direction_worst_MSE']:.8g} | {row['parameter_count']} |"
            )
        lines.append("")
    (root / "GPU_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def run_benchmark(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    shared_root = resolve_shared_root(args.shared)
    config = load_config(args.config)
    results_root = Path(args.results).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    validation = validate_shared_dataset(shared_root)
    atomic_json(results_root / "SHARED_VALIDATION.json", validation)
    if validation["status"] != "PASS":
        raise RuntimeError(f"SHARED_VALIDATION_FAILED:{validation['problems']}")
    atomic_json(results_root / "environment.json", environment_snapshot(repo_root))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    selected = set(args.models.split(",")) if args.models else None
    specs = task_specs(config, args.stage, selected)
    if not specs:
        raise RuntimeError(f"NO_MODELS_FOR_STAGE:{args.stage}:{selected}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    directions = args.directions.split(",") if args.directions else list_directions(shared_root)
    latest = {
        "status": "RUNNING",
        "stage": args.stage,
        "directions": directions,
        "models": [spec.model_id for spec in specs],
        "seeds": seeds,
        "completed": [],
        "failed": [],
    }
    atomic_json(results_root / "checkpoints" / "latest.json", latest)
    for direction_name in directions:
        for spec in specs:
            for seed in seeds:
                task_name = f"{direction_name}:{spec.model_id}:seed={seed}"
                try:
                    record = run_task(
                        project_root=project_root,
                        shared_root=shared_root,
                        cpu_results_root=Path(args.cpu_results).resolve() if args.cpu_results else None,
                        results_root=results_root,
                        direction_name=direction_name,
                        task=spec,
                        seed=seed,
                        device=device,
                        config=config,
                        epoch_override=args.epochs,
                        workers_override=args.workers,
                        strict_folds=args.strict_folds,
                        force=args.force,
                    )
                    latest["completed"].append(task_name)
                    print("TASK_PASS=" + json.dumps(record, ensure_ascii=False), flush=True)
                except Exception as exc:
                    failure = {
                        "status": "FAIL",
                        "task": task_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failure_root = (
                        results_root
                        / "tasks"
                        / spec.stage
                        / direction_name
                        / spec.model_id
                        / f"seed_{seed}"
                    )
                    atomic_json(failure_root / "result.json", failure)
                    latest["failed"].append(failure)
                    print("TASK_FAIL=" + json.dumps(failure, ensure_ascii=False), flush=True)
                finally:
                    atomic_json(results_root / "checkpoints" / "latest.json", latest)
    summaries = aggregate_results(results_root)
    latest["status"] = "PASS" if not latest["failed"] else "PARTIAL"
    latest["summary_rows"] = len(summaries)
    atomic_json(results_root / "checkpoints" / "latest.json", latest)
    print("GPU_STAGE_RESULT=" + json.dumps(latest, ensure_ascii=False), flush=True)
    return 0 if latest["status"] == "PASS" else 2


def build_parser(default_stage: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", required=True)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "gpu_models.yaml"),
    )
    parser.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results_gpu"))
    parser.add_argument("--cpu-results", default=None)
    parser.add_argument("--stage", default=default_stage or "core", choices=["smoke", "core", "frontier", "finalists"])
    parser.add_argument("--models", default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--directions", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--strict-folds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(default_stage: str | None = None) -> int:
    return run_benchmark(build_parser(default_stage).parse_args())
