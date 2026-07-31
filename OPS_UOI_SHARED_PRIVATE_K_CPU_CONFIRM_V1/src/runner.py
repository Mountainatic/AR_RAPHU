from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .io_data import (
    DIRECTIONS,
    atomic_json,
    find_named_root,
    inner_folds,
    load_direction,
    load_gpu_seed_ensemble,
    load_prediction,
    load_protocol,
    metrics,
    pooled_metrics,
    prediction_files,
    published_mse,
    resolve_prediction,
    safe_extract,
    sha256_file,
)
from .linear_k import (
    CHANNELS,
    absolute_gram_correlation,
    block_penalty,
    coefficient_matrix,
    evaluate_regularization_task,
    filtered_mode_features,
    flatten_support,
    fold_mse,
    full_feature_tensor,
    mode_coefficients,
    mother_basis,
    participation,
    principal_angle_degrees,
    ridge_fit,
    select_one_se,
    support_subsets,
    verify_nested,
    whitened_svd,
)
from .nonlinear import (
    evaluate_amplitude_basis,
    fit_amplitude_basis,
    nonlinear_mode_features,
)
from .packaging import build_bundle


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _coefficient_hash(coefficient: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(coefficient, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def _moving_block_bootstrap(
    difference: np.ndarray,
    *,
    block_rows: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(difference, dtype=np.float64)
    n_rows = len(values)
    block_rows = max(1, min(int(block_rows), n_rows))
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    starts_max = max(n_rows - block_rows + 1, 1)
    blocks_needed = int(math.ceil(n_rows / block_rows))
    for replicate in range(replicates):
        starts = rng.integers(0, starts_max, size=blocks_needed)
        sampled = np.concatenate(
            [values[start : start + block_rows] for start in starts]
        )[:n_rows]
        estimates[replicate] = float(np.mean(sampled))
    return {
        "mean_difference": float(np.mean(values)),
        "median": float(np.median(estimates)),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "positive_probability": float(np.mean(estimates > 0.0)),
    }


class Experiment:
    STAGES = tuple(f"E{index}" for index in range(9))

    def __init__(
        self,
        *,
        root: Path,
        shared_bundle: Path | None,
        cpu_bundle: Path | None,
        gpu_bundle: Path | None,
        config_path: Path,
        n_jobs: int,
        bootstrap_jobs: int,
        resume: bool,
    ) -> None:
        self.root = root.resolve()
        self.results = self.root / "results"
        self.work = self.root / "work"
        self.checkpoint_path = self.results / "checkpoints/latest.json"
        self.config_path = config_path.resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        if not self.config.get("protocol_frozen"):
            raise RuntimeError("PROTOCOL_NOT_FROZEN")
        self.n_jobs = max(1, int(n_jobs))
        self.bootstrap_jobs = max(1, int(bootstrap_jobs))
        self.resume = bool(resume)
        self.results.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        checkpoint = (
            json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if self.checkpoint_path.exists()
            else {}
        )
        stored_inputs = checkpoint.get("inputs", {})
        self.shared_bundle = (
            shared_bundle.resolve()
            if shared_bundle is not None
            else Path(stored_inputs["shared_bundle"])
        )
        self.cpu_bundle = (
            cpu_bundle.resolve()
            if cpu_bundle is not None
            else Path(stored_inputs["cpu_bundle"])
        )
        self.gpu_bundle = (
            gpu_bundle.resolve()
            if gpu_bundle is not None
            else Path(stored_inputs["gpu_bundle"])
        )
        self.completed = list(checkpoint.get("completed", [])) if resume else []
        self.state: dict[str, Any] = checkpoint.get("state", {}) if resume else {}
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        atomic_json(
            self.checkpoint_path,
            {
                "schema": self.config["schema"],
                "inputs": {
                    "shared_bundle": str(self.shared_bundle),
                    "cpu_bundle": str(self.cpu_bundle),
                    "gpu_bundle": str(self.gpu_bundle),
                },
                "config": str(self.config_path),
                "config_sha256": sha256_file(self.config_path),
                "completed": self.completed,
                "state": self.state,
                "updated_unix": time.time(),
            },
        )

    def _mark_complete(self, stage: str, payload: dict[str, Any]) -> None:
        self.state[stage] = payload
        if stage not in self.completed:
            self.completed.append(stage)
        self._save_checkpoint()

    def _paths(self) -> tuple[Path, Path, Path]:
        shared_extract = safe_extract(
            self.shared_bundle, self.work / "shared_bundle"
        )
        cpu_extract = safe_extract(self.cpu_bundle, self.work / "cpu_bundle")
        gpu_extract = safe_extract(self.gpu_bundle, self.work / "gpu_bundle")
        return (
            find_named_root(shared_extract, "SHARED_BENCHMARK_DATASET"),
            find_named_root(cpu_extract, "PHYSICS_FIRST_CPU_RESULTS"),
            find_named_root(gpu_extract, "PHYSICS_FIRST_GPU_RESULTS"),
        )

    def run(self, *, stop_after: str | None) -> int:
        for stage in self.STAGES:
            if stage in self.completed:
                print(f"{stage}=ALREADY_COMPLETED", flush=True)
            else:
                print(f"{stage}=START", flush=True)
                method = getattr(self, f"stage_{stage.lower()}")
                payload = method()
                self._mark_complete(stage, payload)
                print(f"{stage}=COMPLETED", flush=True)
            if stop_after == stage:
                print(f"STOP_AFTER={stage}", flush=True)
                return 0
        self._final_terminal()
        return 0

    def stage_e0(self) -> dict[str, Any]:
        for path in (self.shared_bundle, self.cpu_bundle, self.gpu_bundle):
            if not path.is_file():
                raise FileNotFoundError(path)
        hashes = {
            "shared": sha256_file(self.shared_bundle),
            "cpu": sha256_file(self.cpu_bundle),
            "gpu": sha256_file(self.gpu_bundle),
        }
        if hashes["shared"] != self.config["shared_dataset_sha256"]:
            raise RuntimeError(
                f"SHARED_HASH_MISMATCH:{hashes['shared']}:"
                f"{self.config['shared_dataset_sha256']}"
            )
        shared_root, cpu_root, gpu_root = self._paths()
        protocol = load_protocol(shared_root)
        frozen_fields = {
            "cadence_sec": self.config["cadence_sec"],
            "history_min": self.config["history_min"],
            "horizon_min": self.config["horizon_min"],
            "target_window_min": self.config["target_window_min"],
            "controls": self.config["controls"],
        }
        for key, expected in frozen_fields.items():
            if protocol[key] != expected:
                raise RuntimeError(
                    f"PROTOCOL_MISMATCH:{key}:{protocol[key]}:{expected}"
                )
        cpu_index = prediction_files(cpu_root)
        model_sources = {
            "Persistence": (cpu_root, cpu_index),
            "K-only": (cpu_root, cpu_index),
            "Dynamic-PLS": (cpu_root, cpu_index),
            "Joint-K+AR": (cpu_root, cpu_index),
        }
        model_report: dict[str, Any] = {}
        for model_name, (source_root, index) in model_sources.items():
            payloads = []
            direction_metrics: dict[str, Any] = {}
            for direction in DIRECTIONS:
                shared = load_direction(shared_root, direction).test
                path = resolve_prediction(index, direction, model_name)
                prediction = load_prediction(path)
                if not np.array_equal(
                    prediction["sample_id"], shared["sample_id"].astype("U")
                ):
                    raise RuntimeError(
                        f"SAMPLE_ID_MISMATCH:{model_name}:{direction}"
                    )
                if not np.array_equal(
                    prediction["evaluation_mask"], shared["evaluation_mask"]
                ):
                    raise RuntimeError(
                        f"EVALUATION_MASK_MISMATCH:{model_name}:{direction}"
                    )
                if not np.array_equal(
                    prediction["target_z"], shared["target_z"]
                ):
                    maximum = float(
                        np.max(np.abs(prediction["target_z"] - shared["target_z"]))
                    )
                    raise RuntimeError(
                        f"TARGET_MISMATCH:{model_name}:{direction}:{maximum}"
                    )
                mask = prediction["evaluation_mask"]
                direction_metrics[direction] = metrics(
                    prediction["target_z"][mask],
                    prediction["prediction"][mask],
                )
                payloads.append(prediction)
            pooled = pooled_metrics(payloads)
            published = published_mse(source_root, model_name)
            if published is None:
                raise RuntimeError(f"PUBLISHED_MSE_NOT_FOUND:{model_name}")
            difference = abs(float(pooled["MSE"]) - float(published))
            if difference > 1e-10:
                raise RuntimeError(
                    f"PUBLISHED_METRIC_MISMATCH:{model_name}:"
                    f"{pooled['MSE']}:{published}:{difference}"
                )
            model_report[model_name] = {
                "directions": direction_metrics,
                "pooled": pooled,
                "published_mse": published,
                "absolute_difference": difference,
            }
        for model_name in ("NLinear-U", "Temporal Autoencoder"):
            payloads = []
            direction_metrics: dict[str, Any] = {}
            direction_seed_mse: dict[str, list[float]] = {}
            direction_rows: dict[str, int] = {}
            for direction in DIRECTIONS:
                shared = load_direction(shared_root, direction).test
                prediction, seed_mse = load_gpu_seed_ensemble(
                    gpu_root, direction, model_name
                )
                if not np.array_equal(
                    prediction["sample_id"], shared["sample_id"].astype("U")
                ):
                    raise RuntimeError(
                        f"SAMPLE_ID_MISMATCH:{model_name}:{direction}"
                    )
                if not np.array_equal(
                    prediction["evaluation_mask"], shared["evaluation_mask"]
                ):
                    raise RuntimeError(
                        f"EVALUATION_MASK_MISMATCH:{model_name}:{direction}"
                    )
                if not np.array_equal(
                    prediction["target_z"], shared["target_z"]
                ):
                    raise RuntimeError(
                        f"TARGET_MISMATCH:{model_name}:{direction}"
                    )
                mask = prediction["evaluation_mask"]
                direction_metrics[direction] = metrics(
                    prediction["target_z"][mask],
                    prediction["prediction"][mask],
                )
                direction_seed_mse[direction] = seed_mse
                direction_rows[direction] = int(np.sum(mask))
                payloads.append(prediction)
            seed_count = min(
                len(direction_seed_mse[direction]) for direction in DIRECTIONS
            )
            pooled_seed_mse = [
                sum(
                    direction_seed_mse[direction][seed] * direction_rows[direction]
                    for direction in DIRECTIONS
                )
                / sum(direction_rows.values())
                for seed in range(seed_count)
            ]
            reproduced = float(np.median(pooled_seed_mse))
            published = published_mse(gpu_root, model_name)
            if published is None:
                raise RuntimeError(f"PUBLISHED_MSE_NOT_FOUND:{model_name}")
            difference = abs(reproduced - published)
            if difference > 1e-10:
                raise RuntimeError(
                    f"PUBLISHED_METRIC_MISMATCH:{model_name}:"
                    f"{reproduced}:{published}:{difference}"
                )
            model_report[model_name] = {
                "directions_seed_median_ensemble": direction_metrics,
                "ensemble_pooled": pooled_metrics(payloads),
                "published_seed_median_pooled_mse": published,
                "reproduced_seed_median_pooled_mse": reproduced,
                "absolute_difference": difference,
                "seed_count": seed_count,
            }
        rows = {
            direction: int(
                np.sum(load_direction(shared_root, direction).test["evaluation_mask"])
            )
            for direction in DIRECTIONS
        }
        rows["pooled"] = sum(rows.values())
        if rows != self.config["expected_evaluation_rows"]:
            raise RuntimeError(f"EVALUATION_ROW_MISMATCH:{rows}")
        payload = {
            "status": "PASS",
            "hashes": hashes,
            "protocol": protocol,
            "evaluation_rows": rows,
            "models": model_report,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "float64": str(np.dtype(np.float64)),
                "n_jobs": self.n_jobs,
                "bootstrap_jobs": self.bootstrap_jobs,
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
        }
        atomic_json(self.results / "preflight/precheck.json", payload)
        lines = [
            "# PRECHECK REPORT",
            "",
            "`STATUS=PASS`",
            "",
            f"- Shared bundle SHA256: `{hashes['shared']}`",
            f"- CPU baseline bundle SHA256: `{hashes['cpu']}`",
            f"- GPU baseline bundle SHA256: `{hashes['gpu']}`",
            f"- Common evaluation rows: `{rows['pooled']}`",
            "- All required prediction sample IDs, targets, masks and published "
            "pooled MSE values reproduce within `1e-10`.",
        ]
        (self.results / "PRECHECK_REPORT.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        self.state["roots"] = {
            "shared": str(shared_root),
            "cpu": str(cpu_root),
            "gpu": str(gpu_root),
        }
        return payload

    def _shared_root(self) -> Path:
        roots = self.state.get("roots")
        if roots:
            return Path(roots["shared"])
        return self._paths()[0]

    def _basis(self, direction: str, mother_name: str):
        data = load_direction(self._shared_root(), direction)
        return mother_basis(
            mother_name,
            self.config["mother_spaces"][mother_name],
            sequence_steps=int(data.train["sequence_u"].shape[1]),
            cadence_sec=float(self.config["cadence_sec"]),
        )

    def _run_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=min(self.n_jobs, len(tasks))) as pool:
            futures = [
                pool.submit(evaluate_regularization_task, task) for task in tasks
            ]
            for completed, future in enumerate(as_completed(futures), 1):
                output.append(future.result())
                print(f"TASK_PROGRESS={completed}/{len(futures)}", flush=True)
        return output

    def stage_e1(self) -> dict[str, Any]:
        shared_root = self._shared_root()
        nested_error = verify_nested(
            self._basis(DIRECTIONS[0], "V0"),
            self._basis(DIRECTIONS[0], "V1"),
        )
        if nested_error > 1e-10:
            raise RuntimeError(f"MOTHER_SPACES_NOT_NESTED:{nested_error}")
        tasks = [
            {
                "shared_root": str(shared_root),
                "direction": direction,
                "support": list(range(len(CHANNELS))),
                "mother_name": mother,
                "config": self.config,
            }
            for direction, mother in itertools.product(
                DIRECTIONS, self.config["mother_spaces"]
            )
        ]
        raw = self._run_tasks(tasks)
        flat = [row for task in raw for row in task["rows"]]
        selected: dict[str, dict[str, Any]] = {}
        summary_rows: list[dict[str, Any]] = []
        for mother in self.config["mother_spaces"]:
            selected[mother] = {}
            for direction in DIRECTIONS:
                rows = [
                    row
                    for row in flat
                    if row["mother_space"] == mother
                    and row["direction"] == direction
                ]
                chosen = min(
                    rows,
                    key=lambda row: (
                        float(row["oof_mse"]),
                        -float(row["lambda_2"]),
                        float(row["lambda_0"]),
                    ),
                )
                selected[mother][direction] = chosen
                summary_rows.append(
                    {
                        "mother_space": mother,
                        "direction": direction,
                        "basis_size": chosen["basis_size"],
                        "lambda_0": chosen["lambda_0"],
                        "lambda_2": chosen["lambda_2"],
                        "oof_mse": chosen["oof_mse"],
                        "oof_se": chosen["oof_se"],
                        "maximum_kkt": chosen["maximum_kkt"],
                        "maximum_condition": chosen["maximum_condition"],
                    }
                )
        mean_v0 = float(
            np.mean([selected["V0"][direction]["oof_mse"] for direction in DIRECTIONS])
        )
        mean_v1 = float(
            np.mean([selected["V1"][direction]["oof_mse"] for direction in DIRECTIONS])
        )
        improvement = (mean_v0 - mean_v1) / max(mean_v0, 1e-30)
        correlations: list[float] = []
        for direction in DIRECTIONS:
            basis = self._basis(direction, "V1")
            coefficients = [
                coefficient_matrix(
                    np.asarray(values, dtype=np.float64),
                    tuple(range(len(CHANNELS))),
                    basis.size,
                )
                for values in selected["V1"][direction]["fold_coefficients"]
            ]
            for first, second in itertools.combinations(coefficients, 2):
                left = (basis.gram_sqrt @ first).ravel()
                right = (basis.gram_sqrt @ second).ravel()
                denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                correlations.append(
                    abs(float(left @ right)) / denominator if denominator else 0.0
                )
        median_stability = float(np.median(correlations)) if correlations else 0.0
        choose_v1 = (
            improvement >= float(self.config["mother_v1_min_improvement"])
            and median_stability
            >= float(self.config["fold_stability_min_abs_correlation"])
        )
        chosen_mother = "V1" if choose_v1 else "V0"
        payload = {
            "status": "PASS",
            "nested_max_abs_error": nested_error,
            "mean_oof_mse": {"V0": mean_v0, "V1": mean_v1},
            "v1_relative_improvement": improvement,
            "v1_fold_median_abs_correlation": median_stability,
            "selected_mother_space": chosen_mother,
            "selection_reason": (
                "V1_IMPROVEMENT_AND_STABILITY_PASS"
                if choose_v1
                else "ONE_SE_OR_STABILITY_PREFERS_V0"
            ),
            "direction_hyperparameters": {
                mother: {
                    direction: {
                        key: selected[mother][direction][key]
                        for key in ("lambda_0", "lambda_2", "oof_mse", "oof_se")
                    }
                    for direction in DIRECTIONS
                }
                for mother in self.config["mother_spaces"]
            },
        }
        _write_csv(self.results / "MOTHER_SPACE_AUDIT.csv", summary_rows)
        atomic_json(self.results / "diagnostics/e1_mother_selection.json", payload)
        return payload

    def stage_e2(self) -> dict[str, Any]:
        mother = self.state["E1"]["selected_mother_space"]
        shared_root = self._shared_root()
        tasks = [
            {
                "shared_root": str(shared_root),
                "direction": direction,
                "support": list(support),
                "mother_name": mother,
                "config": self.config,
            }
            for direction, support in itertools.product(
                DIRECTIONS, support_subsets()
            )
        ]
        raw = self._run_tasks(tasks)
        flat = [row for task in raw for row in task["rows"]]
        selections: dict[str, Any] = {}
        compact_rows: list[dict[str, Any]] = []
        for row in flat:
            compact_rows.append(
                {
                    key: row[key]
                    for key in (
                        "direction",
                        "mother_space",
                        "support",
                        "support_mask",
                        "lambda_0",
                        "lambda_2",
                        "oof_mse",
                        "oof_se",
                        "maximum_kkt",
                        "maximum_condition",
                    )
                }
            )
        for direction in DIRECTIONS:
            rows = [row for row in flat if row["direction"] == direction]
            selected, trace = select_one_se(rows)
            data = load_direction(shared_root, direction)
            basis = self._basis(direction, mother)
            train_tensor = full_feature_tensor(
                data.train["sequence_u"], basis, self.config["cadence_sec"]
            )
            test_tensor = full_feature_tensor(
                data.test["sequence_u"], basis, self.config["cadence_sec"]
            )
            support = tuple(selected["support"])
            penalty = block_penalty(
                basis,
                len(support),
                selected["lambda_0"],
                selected["lambda_2"],
            )
            fit = ridge_fit(
                flatten_support(train_tensor, support),
                data.train["target_z"],
                penalty,
                predict_matrix=flatten_support(test_tensor, support),
            )
            coefficient = coefficient_matrix(fit.coefficient, support, basis.size)
            prediction_path = self.results / f"predictions/e2_full/{direction}.npz"
            _atomic_npz(
                prediction_path,
                sample_id=data.test["sample_id"],
                prediction=fit.prediction,
                target_z=data.test["target_z"],
                evaluation_mask=data.test["evaluation_mask"],
                coefficient_by_channel=coefficient,
            )
            selections[direction] = {
                "selected": {
                    key: selected[key]
                    for key in (
                        "support",
                        "support_mask",
                        "lambda_0",
                        "lambda_2",
                        "oof_mse",
                        "oof_se",
                    )
                },
                "trace": trace,
                "fit": {
                    "kkt_relative": fit.kkt_relative,
                    "condition_number": fit.condition_number,
                    "coefficient_hash": _coefficient_hash(coefficient),
                    "test_metrics": metrics(
                        data.test["target_z"][data.test["evaluation_mask"]],
                        fit.prediction[data.test["evaluation_mask"]],
                    ),
                },
            }
        _write_csv(self.results / "FULL_LINEAR_K.csv", compact_rows)
        payload = {
            "status": "PASS",
            "mother_space": mother,
            "directions": selections,
        }
        atomic_json(self.results / "diagnostics/e2_selection.json", payload)
        return payload

    def _fold_full_fit(
        self,
        direction: str,
        training: np.ndarray,
        *,
        support: tuple[int, ...],
        basis,
        tensor: np.ndarray,
        target: np.ndarray,
        lambda_0: float,
        lambda_2: float,
    ) -> np.ndarray:
        fit = ridge_fit(
            flatten_support(tensor[training], support),
            target[training],
            block_penalty(basis, len(support), lambda_0, lambda_2),
        )
        return coefficient_matrix(fit.coefficient, support, basis.size)

    def _rank_rows(self, direction: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        shared_root = self._shared_root()
        data = load_direction(shared_root, direction)
        protocol = load_protocol(shared_root)
        mother = self.state["E2"]["mother_space"]
        basis = self._basis(direction, mother)
        selected = self.state["E2"]["directions"][direction]["selected"]
        support = tuple(selected["support"])
        lambda_0 = float(selected["lambda_0"])
        lambda_2 = float(selected["lambda_2"])
        tensor = full_feature_tensor(
            data.train["sequence_u"], basis, self.config["cadence_sec"]
        )
        purge_raw = int(
            round(
                self.config["purge_min"]
                * 60.0
                / float(protocol["sample_period_sec"])
            )
        )
        folds = inner_folds(
            data.train["origin_raw_index"],
            protocol["inner_folds"],
            purge_raw_samples=purge_raw,
        )
        full_fit = ridge_fit(
            flatten_support(tensor, support),
            data.train["target_z"],
            block_penalty(basis, len(support), lambda_0, lambda_2),
        )
        full_coefficient = coefficient_matrix(
            full_fit.coefficient, support, basis.size
        )
        full_u, full_s, full_vt = whitened_svd(full_coefficient, basis)
        rows: list[dict[str, Any]] = []
        fold_modes: dict[int, list[np.ndarray]] = {rank: [] for rank in (1, 2)}
        fold_errors: dict[int, list[float]] = {rank: [] for rank in (0, 1, 2)}
        fold_participation: dict[int, list[dict[str, Any]]] = {
            rank: [] for rank in (1, 2)
        }
        for training, validation in folds:
            coefficient = self._fold_full_fit(
                direction,
                training,
                support=support,
                basis=basis,
                tensor=tensor,
                target=data.train["target_z"],
                lambda_0=lambda_0,
                lambda_2=lambda_2,
            )
            left, singular, right_t = whitened_svd(coefficient, basis)
            zero_error = data.train["target_z"][validation]
            fold_errors[0].append(float(np.mean(zero_error * zero_error)))
            for rank in (1, 2):
                effective = min(rank, len(singular), len(support))
                modes = mode_coefficients(left[:, :effective], basis)
                train_features = filtered_mode_features(
                    tensor[training], modes, support
                )
                validation_features = filtered_mode_features(
                    tensor[validation], modes, support
                )
                refit = ridge_fit(
                    train_features,
                    data.train["target_z"][training],
                    lambda_0 * np.eye(train_features.shape[1]),
                    predict_matrix=validation_features,
                )
                error = data.train["target_z"][validation] - refit.prediction
                fold_errors[rank].append(float(np.mean(error * error)))
                fold_modes[rank].append(left[:, :effective])
                mode_participation = [
                    participation(right_t[mode_index, list(support)])
                    for mode_index in range(effective)
                ]
                fold_participation[rank].append(
                    {"modes": mode_participation}
                )
        loo_by_rank: dict[int, float] = {}
        for rank in (1, 2):
            correlations: list[float] = []
            for removed in support:
                remaining = tuple(channel for channel in support if channel != removed)
                if not remaining:
                    correlations.append(0.0)
                    continue
                fit = ridge_fit(
                    flatten_support(tensor, remaining),
                    data.train["target_z"],
                    block_penalty(basis, len(remaining), lambda_0, lambda_2),
                )
                coefficient = coefficient_matrix(
                    fit.coefficient, remaining, basis.size
                )
                left, singular, _ = whitened_svd(coefficient, basis)
                effective = min(rank, len(singular), len(remaining))
                angles = principal_angle_degrees(
                    full_u[:, :effective], left[:, :effective]
                )
                correlations.append(
                    float(np.cos(np.radians(np.max(angles))))
                    if len(angles)
                    else 0.0
                )
            loo_by_rank[rank] = min(correlations) if correlations else 0.0
        for rank in (0, 1, 2):
            mean, standard_error = fold_mse(fold_errors[rank])
            if rank == 0:
                gate = True
                diagnostics = {}
            else:
                effective = min(rank, len(full_s), len(support))
                participations = [
                    participation(full_vt[index, list(support)])
                    for index in range(effective)
                ]
                angles = [
                    float(np.max(principal_angle_degrees(full_u[:, :effective], mode)))
                    for mode in fold_modes[rank]
                    if mode.shape[1] == effective
                ]
                median_angle = float(np.median(angles)) if angles else 180.0
                previous_mean = float(np.mean(fold_errors[rank - 1]))
                improvement = (previous_mean - mean) / max(previous_mean, 1e-30)
                gate = (
                    all(
                        item["participation_ratio"]
                        >= self.config["shared_min_participation"]
                        and item["channels_at_least_10pct"]
                        >= self.config["shared_min_channels"]
                        for item in participations
                    )
                    and loo_by_rank[rank] >= self.config["shared_min_loo_correlation"]
                    and median_angle
                    <= self.config["shared_max_median_principal_angle_deg"]
                    and improvement >= self.config["shared_min_oof_improvement"]
                )
                diagnostics = {
                    "participation": participations,
                    "minimum_loo_correlation": loo_by_rank[rank],
                    "median_principal_angle_deg": median_angle,
                    "incremental_oof_improvement": improvement,
                }
            rows.append(
                {
                    "direction": direction,
                    "rank": rank,
                    "oof_mse": mean,
                    "oof_se": standard_error,
                    "fold_mse": fold_errors[rank],
                    "gate_pass": gate,
                    "diagnostics": diagnostics,
                }
            )
        return rows, {
            "support": list(support),
            "lambda_0": lambda_0,
            "lambda_2": lambda_2,
            "full_singular_values": full_s.tolist(),
            "full_left_vectors": full_u.tolist(),
            "full_right_vectors": full_vt.tolist(),
            "full_coefficient": full_coefficient.tolist(),
        }

    def _fit_rank_prediction(
        self, direction: str, rank: int
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        shared_root = self._shared_root()
        data = load_direction(shared_root, direction)
        mother = self.state["E2"]["mother_space"]
        basis = self._basis(direction, mother)
        selected = self.state["E2"]["directions"][direction]["selected"]
        support = tuple(selected["support"])
        train_tensor = full_feature_tensor(
            data.train["sequence_u"], basis, self.config["cadence_sec"]
        )
        test_tensor = full_feature_tensor(
            data.test["sequence_u"], basis, self.config["cadence_sec"]
        )
        fit = ridge_fit(
            flatten_support(train_tensor, support),
            data.train["target_z"],
            block_penalty(
                basis,
                len(support),
                selected["lambda_0"],
                selected["lambda_2"],
            ),
        )
        coefficient = coefficient_matrix(fit.coefficient, support, basis.size)
        left, singular, right_t = whitened_svd(coefficient, basis)
        effective = min(rank, len(singular), len(support))
        modes = mode_coefficients(left[:, :effective], basis)
        train_features = filtered_mode_features(train_tensor, modes, support)
        test_features = filtered_mode_features(test_tensor, modes, support)
        refit = ridge_fit(
            train_features,
            data.train["target_z"],
            float(selected["lambda_0"]) * np.eye(train_features.shape[1]),
            predict_matrix=test_features,
        )
        payload = {
            "rank": rank,
            "effective_rank": effective,
            "support": list(support),
            "singular_values": singular.tolist(),
            "mode_coefficients": modes.tolist(),
            "loadings": refit.coefficient.tolist(),
            "kkt_relative": refit.kkt_relative,
            "condition_number": refit.condition_number,
            "test_metrics": metrics(
                data.test["target_z"][data.test["evaluation_mask"]],
                refit.prediction[data.test["evaluation_mask"]],
            ),
        }
        return payload, refit.prediction, modes

    def stage_e3(self) -> dict[str, Any]:
        rows_by_direction: dict[str, list[dict[str, Any]]] = {}
        full_models: dict[str, Any] = {}
        one_se_sets: dict[str, list[int]] = {}
        for direction in DIRECTIONS:
            rows, full = self._rank_rows(direction)
            rows_by_direction[direction] = rows
            full_models[direction] = full
            minimum = min(rows, key=lambda row: row["oof_mse"])
            threshold = minimum["oof_mse"] + minimum["oof_se"]
            eligible = [
                row["rank"]
                for row in rows
                if row["gate_pass"] and row["oof_mse"] <= threshold + 1e-15
            ]
            one_se_sets[direction] = eligible or [0]
        common = sorted(set(one_se_sets[DIRECTIONS[0]]) & set(one_se_sets[DIRECTIONS[1]]))
        global_rank = min(common) if common else min(
            min(one_se_sets[DIRECTIONS[0]]), min(one_se_sets[DIRECTIONS[1]])
        )
        direction_models: dict[str, Any] = {}
        direction_modes: dict[str, np.ndarray] = {}
        for direction in DIRECTIONS:
            model, prediction, modes = self._fit_rank_prediction(
                direction, global_rank
            )
            direction_models[direction] = model
            direction_modes[direction] = modes
            data = load_direction(self._shared_root(), direction)
            _atomic_npz(
                self.results / f"predictions/e3_shared/{direction}.npz",
                sample_id=data.test["sample_id"],
                prediction=prediction,
                target_z=data.test["target_z"],
                evaluation_mask=data.test["evaluation_mask"],
            )
        if global_rank > 0:
            basis = self._basis(DIRECTIONS[0], self.state["E2"]["mother_space"])
            angles = principal_angle_degrees(
                basis.gram_sqrt @ direction_modes[DIRECTIONS[0]],
                basis.gram_sqrt @ direction_modes[DIRECTIONS[1]],
            )
            cross_angle = float(np.max(angles)) if len(angles) else 180.0
        else:
            cross_angle = 0.0
        shared_certified = (
            bool(common)
            and global_rank > 0
            and cross_angle <= self.config["shared_max_median_principal_angle_deg"]
        )
        flat_rows = []
        for direction, rows in rows_by_direction.items():
            for row in rows:
                flat_rows.append(row)
        _write_csv(self.results / "SHARED_RANK_RESULTS.csv", flat_rows)
        payload = {
            "status": "PASS",
            "selected_rank": global_rank,
            "one_se_rank_sets": one_se_sets,
            "one_se_intersection": common,
            "cross_direction_max_principal_angle_deg": cross_angle,
            "shared_certification": (
                "SHARED_K_CERTIFIED"
                if shared_certified
                else "PREDICTIVE_SHARED_LOW_RANK_NOT_CERTIFIED"
            ),
            "directions": direction_models,
            "full_models": full_models,
        }
        atomic_json(self.results / "subspaces/shared_rank.json", payload)
        return payload

    def _private_candidate(
        self, direction: str, candidate: int | None
    ) -> dict[str, Any]:
        shared_root = self._shared_root()
        data = load_direction(shared_root, direction)
        protocol = load_protocol(shared_root)
        mother = self.state["E2"]["mother_space"]
        basis = self._basis(direction, mother)
        selected = self.state["E2"]["directions"][direction]["selected"]
        support = tuple(selected["support"])
        rank = int(self.state["E3"]["selected_rank"])
        lambda_0 = float(selected["lambda_0"])
        lambda_2 = float(selected["lambda_2"])
        tensor = full_feature_tensor(
            data.train["sequence_u"], basis, self.config["cadence_sec"]
        )
        purge_raw = int(
            round(
                self.config["purge_min"]
                * 60.0
                / float(protocol["sample_period_sec"])
            )
        )
        folds = inner_folds(
            data.train["origin_raw_index"],
            protocol["inner_folds"],
            purge_raw_samples=purge_raw,
        )
        baseline_errors: list[float] = []
        candidate_errors: list[float] = []
        private_modes: list[np.ndarray] = []
        energy_fractions: list[float] = []
        orthogonality: list[float] = []
        for training, validation in folds:
            full_coefficient = self._fold_full_fit(
                direction,
                training,
                support=support,
                basis=basis,
                tensor=tensor,
                target=data.train["target_z"],
                lambda_0=lambda_0,
                lambda_2=lambda_2,
            )
            left, singular, _ = whitened_svd(full_coefficient, basis)
            effective = min(rank, len(singular), len(support))
            shared_modes = mode_coefficients(left[:, :effective], basis)
            shared_train = filtered_mode_features(
                tensor[training], shared_modes, support
            )
            shared_validation = filtered_mode_features(
                tensor[validation], shared_modes, support
            )
            shared_fit = ridge_fit(
                shared_train,
                data.train["target_z"][training],
                lambda_0 * np.eye(shared_train.shape[1]),
                predict_matrix=shared_validation,
            )
            baseline_error = (
                data.train["target_z"][validation] - shared_fit.prediction
            )
            baseline_errors.append(float(np.mean(baseline_error * baseline_error)))
            if candidate is None or candidate not in support:
                candidate_errors.append(baseline_errors[-1])
                continue
            whitened = basis.gram_sqrt @ full_coefficient
            projector = (
                left[:, :effective] @ left[:, :effective].T
                if effective
                else np.zeros((basis.size, basis.size), dtype=np.float64)
            )
            residual = (np.eye(basis.size) - projector) @ whitened
            channel_vector = residual[:, candidate]
            norm = float(np.linalg.norm(channel_vector))
            total_energy = float(np.sum(residual * residual))
            fraction = norm * norm / total_energy if total_energy else 0.0
            energy_fractions.append(fraction)
            if norm <= 1e-14:
                candidate_errors.append(baseline_errors[-1])
                continue
            private_whitened = channel_vector[:, None] / norm
            private_mode = basis.gram_inv_sqrt @ private_whitened
            private_modes.append(private_mode[:, 0])
            orthogonality.append(
                float(
                    np.max(
                        np.abs(
                            shared_modes.T @ basis.gram @ private_mode
                        )
                    )
                )
                if effective
                else 0.0
            )
            private_train = tensor[training, candidate, :] @ private_mode
            private_validation = tensor[validation, candidate, :] @ private_mode
            train_features = np.column_stack((shared_train, private_train))
            validation_features = np.column_stack(
                (shared_validation, private_validation)
            )
            candidate_fit = ridge_fit(
                train_features,
                data.train["target_z"][training],
                lambda_0 * np.eye(train_features.shape[1]),
                predict_matrix=validation_features,
            )
            error = data.train["target_z"][validation] - candidate_fit.prediction
            candidate_errors.append(float(np.mean(error * error)))
        baseline_mean, baseline_se = fold_mse(baseline_errors)
        candidate_mean, candidate_se = fold_mse(candidate_errors)
        improvement = (
            (baseline_mean - candidate_mean) / max(baseline_mean, 1e-30)
            if candidate is not None
            else 0.0
        )
        correlations = [
            absolute_gram_correlation(first, second, basis.gram)
            for first, second in itertools.combinations(private_modes, 2)
        ]
        median_correlation = float(np.median(correlations)) if correlations else 0.0
        gate = candidate is None or (
            max(orthogonality or [0.0]) < self.config["orthogonality_tolerance"]
            and min(energy_fractions or [0.0])
            >= self.config["private_min_channel_energy_fraction"]
            and median_correlation >= self.config["private_min_fold_correlation"]
            and improvement >= self.config["private_min_oof_improvement"]
        )
        return {
            "direction": direction,
            "candidate_channel_index": candidate,
            "candidate_channel": None if candidate is None else CHANNELS[candidate],
            "baseline_fold_mse": baseline_errors,
            "candidate_fold_mse": candidate_errors,
            "baseline_oof_mse": baseline_mean,
            "candidate_oof_mse": candidate_mean,
            "candidate_oof_se": candidate_se,
            "conditional_oof_improvement": improvement,
            "minimum_channel_energy_fraction": min(energy_fractions or [0.0]),
            "fold_private_median_abs_correlation": median_correlation,
            "maximum_orthogonality_error": max(orthogonality or [0.0]),
            "inner_gate_pass": gate,
        }

    def _fit_shared_private_prediction(
        self, direction: str, private_channel: int | None
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
        data = load_direction(self._shared_root(), direction)
        mother = self.state["E2"]["mother_space"]
        basis = self._basis(direction, mother)
        selected = self.state["E2"]["directions"][direction]["selected"]
        support = tuple(selected["support"])
        rank = int(self.state["E3"]["selected_rank"])
        lambda_0 = float(selected["lambda_0"])
        train_tensor = full_feature_tensor(
            data.train["sequence_u"], basis, self.config["cadence_sec"]
        )
        test_tensor = full_feature_tensor(
            data.test["sequence_u"], basis, self.config["cadence_sec"]
        )
        full_fit = ridge_fit(
            flatten_support(train_tensor, support),
            data.train["target_z"],
            block_penalty(
                basis,
                len(support),
                lambda_0,
                selected["lambda_2"],
            ),
        )
        full_coefficient = coefficient_matrix(
            full_fit.coefficient, support, basis.size
        )
        left, singular, _ = whitened_svd(full_coefficient, basis)
        effective = min(rank, len(singular), len(support))
        shared_modes = mode_coefficients(left[:, :effective], basis)
        train_features = filtered_mode_features(train_tensor, shared_modes, support)
        test_features = filtered_mode_features(test_tensor, shared_modes, support)
        private_mode = None
        orthogonality = 0.0
        if private_channel is not None and private_channel in support:
            whitened = basis.gram_sqrt @ full_coefficient
            projector = (
                left[:, :effective] @ left[:, :effective].T
                if effective
                else np.zeros((basis.size, basis.size), dtype=np.float64)
            )
            vector = (np.eye(basis.size) - projector) @ whitened[:, private_channel]
            norm = float(np.linalg.norm(vector))
            if norm > 1e-14:
                private_mode = basis.gram_inv_sqrt @ (vector / norm)
                train_features = np.column_stack(
                    (
                        train_features,
                        train_tensor[:, private_channel, :] @ private_mode,
                    )
                )
                test_features = np.column_stack(
                    (
                        test_features,
                        test_tensor[:, private_channel, :] @ private_mode,
                    )
                )
                orthogonality = (
                    float(
                        np.max(
                            np.abs(
                                shared_modes.T @ basis.gram @ private_mode
                            )
                        )
                    )
                    if effective
                    else 0.0
                )
        refit = ridge_fit(
            train_features,
            data.train["target_z"],
            lambda_0 * np.eye(train_features.shape[1]),
            predict_matrix=test_features,
        )
        return (
            {
                "support": list(support),
                "rank": rank,
                "private_channel": (
                    None if private_channel is None else CHANNELS[private_channel]
                ),
                "shared_modes": shared_modes.tolist(),
                "private_mode": None if private_mode is None else private_mode.tolist(),
                "loadings": refit.coefficient.tolist(),
                "orthogonality_error": orthogonality,
                "kkt_relative": refit.kkt_relative,
                "condition_number": refit.condition_number,
            },
            refit.prediction,
            private_mode,
        )

    def stage_e4(self) -> dict[str, Any]:
        support_union = sorted(
            set(
                channel
                for direction in DIRECTIONS
                for channel in self.state["E2"]["directions"][direction][
                    "selected"
                ]["support"]
            )
        )
        candidates: list[int | None] = [None, *support_union]
        rows = [
            self._private_candidate(direction, candidate)
            for direction, candidate in itertools.product(DIRECTIONS, candidates)
        ]
        aggregate: list[dict[str, Any]] = []
        for candidate in candidates:
            relevant = [
                row for row in rows if row["candidate_channel_index"] == candidate
            ]
            fold_values = np.asarray(
                [row["candidate_fold_mse"] for row in relevant],
                dtype=np.float64,
            ).ravel()
            mean, standard_error = fold_mse(fold_values)
            aggregate.append(
                {
                    "candidate_channel_index": candidate,
                    "candidate_channel": (
                        None if candidate is None else CHANNELS[candidate]
                    ),
                    "combined_oof_mse": mean,
                    "combined_oof_se": standard_error,
                    "inner_gate_pass": all(row["inner_gate_pass"] for row in relevant),
                }
            )
        minimum = min(aggregate, key=lambda row: row["combined_oof_mse"])
        threshold = minimum["combined_oof_mse"] + minimum["combined_oof_se"]
        eligible = [
            row
            for row in aggregate
            if row["inner_gate_pass"]
            and row["combined_oof_mse"] <= threshold + 1e-15
        ]
        chosen = min(
            eligible,
            key=lambda row: (
                row["candidate_channel_index"] is not None,
                999
                if row["candidate_channel_index"] is None
                else row["candidate_channel_index"],
            ),
        )
        candidate = chosen["candidate_channel_index"]
        models: dict[str, Any] = {}
        predictions: dict[str, np.ndarray] = {}
        private_modes: dict[str, np.ndarray | None] = {}
        outer_gains: dict[str, float] = {}
        bootstrap: dict[str, Any] = {}
        for direction in DIRECTIONS:
            baseline_path = self.results / f"predictions/e3_shared/{direction}.npz"
            with np.load(baseline_path) as stored:
                baseline_prediction = stored["prediction"]
            model, prediction, private_mode = self._fit_shared_private_prediction(
                direction, candidate
            )
            data = load_direction(self._shared_root(), direction)
            mask = data.test["evaluation_mask"]
            baseline_error = (
                data.test["target_z"][mask] - baseline_prediction[mask]
            ) ** 2
            candidate_error = (
                data.test["target_z"][mask] - prediction[mask]
            ) ** 2
            gain = float(
                (np.mean(baseline_error) - np.mean(candidate_error))
                / max(np.mean(baseline_error), 1e-30)
            )
            outer_gains[direction] = gain
            block_rows = int(
                round(
                    self.config["bootstrap_primary_block_min"]
                    * 60.0
                    / self.config["cadence_sec"]
                )
            )
            bootstrap[direction] = _moving_block_bootstrap(
                baseline_error - candidate_error,
                block_rows=block_rows,
                replicates=int(self.config["bootstrap_replicates"]),
                seed=int(self.config["random_seed"])
                + DIRECTIONS.index(direction) * 1000,
            )
            models[direction] = model
            predictions[direction] = prediction
            private_modes[direction] = private_mode
        if candidate is not None and all(
            private_modes[direction] is not None for direction in DIRECTIONS
        ):
            basis = self._basis(DIRECTIONS[0], self.state["E2"]["mother_space"])
            cross_correlation = absolute_gram_correlation(
                private_modes[DIRECTIONS[0]],
                private_modes[DIRECTIONS[1]],
                basis.gram,
            )
        else:
            cross_correlation = 1.0 if candidate is None else 0.0
        outer_pass = candidate is None or (
            all(value >= 0.0 for value in outer_gains.values())
            and cross_correlation
            >= self.config["private_min_cross_direction_correlation"]
            and all(
                value["positive_probability"]
                >= self.config["private_min_bootstrap_positive_probability"]
                for value in bootstrap.values()
            )
        )
        selected_private = candidate if outer_pass else None
        if selected_private != candidate:
            for direction in DIRECTIONS:
                models[direction], predictions[direction], _ = (
                    self._fit_shared_private_prediction(direction, None)
                )
        for direction in DIRECTIONS:
            data = load_direction(self._shared_root(), direction)
            _atomic_npz(
                self.results / f"predictions/e4_private/{direction}.npz",
                sample_id=data.test["sample_id"],
                prediction=predictions[direction],
                target_z=data.test["target_z"],
                evaluation_mask=data.test["evaluation_mask"],
            )
        _write_csv(self.results / "PRIVATE_RANK_RESULTS.csv", rows)
        payload = {
            "status": "PASS",
            "inner_aggregate": aggregate,
            "inner_selected_candidate": (
                None if candidate is None else CHANNELS[candidate]
            ),
            "outer_direction_gains": outer_gains,
            "outer_bootstrap": bootstrap,
            "cross_direction_private_correlation": cross_correlation,
            "outer_gate_pass": outer_pass,
            "selected_private_channel_index": selected_private,
            "selected_private_channel": (
                None if selected_private is None else CHANNELS[selected_private]
            ),
            "private_certification": (
                "PRIVATE_EXACT_ZERO"
                if selected_private is None
                else "PRIVATE_RANK1_CERTIFIED"
            ),
            "directions": models,
        }
        atomic_json(self.results / "subspaces/private_rank.json", payload)
        return payload

    def stage_e5(self) -> dict[str, Any]:
        private_channel = self.state["E4"]["selected_private_channel_index"]
        directions: dict[str, Any] = {}
        for direction in DIRECTIONS:
            model, prediction, private_mode = self._fit_shared_private_prediction(
                direction, private_channel
            )
            if model["kkt_relative"] > self.config["kkt_tolerance"]:
                raise RuntimeError(
                    f"KKT_FAILED:{direction}:{model['kkt_relative']}"
                )
            if model["orthogonality_error"] > self.config["orthogonality_tolerance"]:
                raise RuntimeError(
                    f"ORTHOGONALITY_FAILED:{direction}:"
                    f"{model['orthogonality_error']}"
                )
            data = load_direction(self._shared_root(), direction)
            mask = data.test["evaluation_mask"]
            model["test_metrics"] = metrics(
                data.test["target_z"][mask], prediction[mask]
            )
            coefficient_payload = np.asarray(model["loadings"], dtype=np.float64)
            model["coefficient_hash"] = _coefficient_hash(coefficient_payload)
            _atomic_npz(
                self.results / f"predictions/e5_fixed/{direction}.npz",
                sample_id=data.test["sample_id"],
                prediction=prediction.astype(np.float64),
                target_z=data.test["target_z"],
                evaluation_mask=data.test["evaluation_mask"],
                coefficient=coefficient_payload,
            )
            directions[direction] = model
        payload = {
            "status": "PASS",
            "mother_space": self.state["E2"]["mother_space"],
            "shared_rank": self.state["E3"]["selected_rank"],
            "private_channel": self.state["E4"]["selected_private_channel"],
            "directions": directions,
            "fp64_prediction": True,
            "kkt_certification": "PASS",
            "orthogonality_certification": "PASS",
        }
        atomic_json(self.results / "diagnostics/e5_refit.json", payload)
        return payload

    def stage_e6(self) -> dict[str, Any]:
        shared_certified = (
            self.state["E3"]["shared_certification"] == "SHARED_K_CERTIFIED"
        )
        if not shared_certified:
            payload = {
                "status": "NOT_APPLICABLE",
                "reason": "LINEAR_SHARED_PRIVATE_CERTIFICATE_NOT_PASSED",
                "selected_nonlinear_channel": None,
                "nonlinear_exact_zero": True,
            }
            _write_csv(self.results / "NONLINEAR_RESULTS.csv", [payload])
            return payload
        shared_root = self._shared_root()
        active_sets = [
            set(self.state["E2"]["directions"][direction]["selected"]["support"])
            for direction in DIRECTIONS
        ]
        active_channels = sorted(set.intersection(*active_sets)) if active_sets else []
        candidates: list[int | None] = [None, *active_channels]
        rows: list[dict[str, Any]] = []
        cached: dict[tuple[str, int], dict[str, Any]] = {}
        for direction in DIRECTIONS:
            data = load_direction(shared_root, direction)
            protocol = load_protocol(shared_root)
            mother = self.state["E2"]["mother_space"]
            basis = self._basis(direction, mother)
            support = tuple(
                self.state["E2"]["directions"][direction]["selected"]["support"]
            )
            model = self.state["E5"]["directions"][direction]
            shared_modes = np.asarray(model["shared_modes"], dtype=np.float64)
            private_mode = (
                None
                if model["private_mode"] is None
                else np.asarray(model["private_mode"], dtype=np.float64)
            )
            train_tensor = full_feature_tensor(
                data.train["sequence_u"], basis, self.config["cadence_sec"]
            )
            base_features = filtered_mode_features(
                train_tensor, shared_modes, support
            )
            private_index = self.state["E4"]["selected_private_channel_index"]
            if private_mode is not None and private_index is not None:
                base_features = np.column_stack(
                    (
                        base_features,
                        train_tensor[:, private_index, :] @ private_mode,
                    )
                )
            purge_raw = int(
                round(
                    self.config["purge_min"]
                    * 60.0
                    / float(protocol["sample_period_sec"])
                )
            )
            folds = inner_folds(
                data.train["origin_raw_index"],
                protocol["inner_folds"],
                purge_raw_samples=purge_raw,
            )
            baseline_fold_mse: list[float] = []
            for training, validation in folds:
                base_fit = ridge_fit(
                    base_features[training],
                    data.train["target_z"][training],
                    float(
                        self.state["E2"]["directions"][direction]["selected"][
                            "lambda_0"
                        ]
                    )
                    * np.eye(base_features.shape[1]),
                    predict_matrix=base_features[validation],
                )
                error = data.train["target_z"][validation] - base_fit.prediction
                baseline_fold_mse.append(float(np.mean(error * error)))
            for candidate in candidates:
                if candidate is None:
                    mean, standard_error = fold_mse(baseline_fold_mse)
                    rows.append(
                        {
                            "direction": direction,
                            "candidate_channel_index": None,
                            "candidate_channel": None,
                            "lambda_nonlinear": None,
                            "fold_mse": baseline_fold_mse,
                            "oof_mse": mean,
                            "oof_se": standard_error,
                            "relative_improvement": 0.0,
                        }
                    )
                    continue
                amplitude = fit_amplitude_basis(
                    np.asarray(data.train["sequence_u"][:, :, candidate])
                )
                train_amplitude, train_diagnostics = evaluate_amplitude_basis(
                    amplitude, data.train["sequence_u"][:, :, candidate]
                )
                mode_list = [shared_modes]
                if private_mode is not None and private_index == candidate:
                    mode_list.append(private_mode[:, None])
                time_modes = np.column_stack(mode_list)
                nonlinear_features = nonlinear_mode_features(
                    train_amplitude,
                    basis.lag_design @ time_modes,
                    cadence_sec=self.config["cadence_sec"],
                )
                cached[(direction, candidate)] = {
                    "amplitude": amplitude,
                    "time_modes": time_modes,
                    "train_diagnostics": train_diagnostics,
                }
                for nonlinear_penalty in self.config["ridge_grid"]:
                    errors: list[float] = []
                    for training, validation in folds:
                        base_fit = ridge_fit(
                            base_features[training],
                            data.train["target_z"][training],
                            float(
                                self.state["E2"]["directions"][direction][
                                    "selected"
                                ]["lambda_0"]
                            )
                            * np.eye(base_features.shape[1]),
                            predict_matrix=base_features[validation],
                        )
                        training_residual = (
                            data.train["target_z"][training]
                            - base_features[training] @ base_fit.coefficient
                        )
                        nonlinear_fit = ridge_fit(
                            nonlinear_features[training],
                            training_residual,
                            float(nonlinear_penalty)
                            * np.eye(nonlinear_features.shape[1]),
                            predict_matrix=nonlinear_features[validation],
                        )
                        prediction = base_fit.prediction + nonlinear_fit.prediction
                        error = data.train["target_z"][validation] - prediction
                        errors.append(float(np.mean(error * error)))
                    mean, standard_error = fold_mse(errors)
                    base_mean = float(np.mean(baseline_fold_mse))
                    rows.append(
                        {
                            "direction": direction,
                            "candidate_channel_index": candidate,
                            "candidate_channel": CHANNELS[candidate],
                            "lambda_nonlinear": float(nonlinear_penalty),
                            "fold_mse": errors,
                            "oof_mse": mean,
                            "oof_se": standard_error,
                            "relative_improvement": (
                                base_mean - mean
                            )
                            / max(base_mean, 1e-30),
                        }
                    )
        aggregate: list[dict[str, Any]] = []
        for candidate in candidates:
            relevant = [
                row for row in rows if row["candidate_channel_index"] == candidate
            ]
            if candidate is None:
                values = np.asarray(
                    [row["fold_mse"] for row in relevant], dtype=np.float64
                ).ravel()
                mean, standard_error = fold_mse(values)
                aggregate.append(
                    {
                        "candidate_channel_index": None,
                        "candidate_channel": None,
                        "lambda_nonlinear": None,
                        "oof_mse": mean,
                        "oof_se": standard_error,
                    }
                )
            else:
                for nonlinear_penalty in self.config["ridge_grid"]:
                    matching = [
                        row
                        for row in relevant
                        if row["lambda_nonlinear"] == nonlinear_penalty
                    ]
                    values = np.asarray(
                        [row["fold_mse"] for row in matching], dtype=np.float64
                    ).ravel()
                    mean, standard_error = fold_mse(values)
                    aggregate.append(
                        {
                            "candidate_channel_index": candidate,
                            "candidate_channel": CHANNELS[candidate],
                            "lambda_nonlinear": float(nonlinear_penalty),
                            "oof_mse": mean,
                            "oof_se": standard_error,
                        }
                    )
        minimum = min(aggregate, key=lambda row: row["oof_mse"])
        threshold = minimum["oof_mse"] + minimum["oof_se"]
        one_se = [
            row for row in aggregate if row["oof_mse"] <= threshold + 1e-15
        ]
        selected = min(
            one_se,
            key=lambda row: (
                row["candidate_channel_index"] is not None,
                -float(row["lambda_nonlinear"] or 0.0),
                999
                if row["candidate_channel_index"] is None
                else row["candidate_channel_index"],
            ),
        )
        selected_channel = selected["candidate_channel_index"]
        selected_penalty = selected["lambda_nonlinear"]
        direction_results: dict[str, Any] = {}
        for direction in DIRECTIONS:
            data = load_direction(shared_root, direction)
            mother = self.state["E2"]["mother_space"]
            basis = self._basis(direction, mother)
            support = tuple(
                self.state["E2"]["directions"][direction]["selected"]["support"]
            )
            model = self.state["E5"]["directions"][direction]
            shared_modes = np.asarray(model["shared_modes"], dtype=np.float64)
            private_mode = (
                None
                if model["private_mode"] is None
                else np.asarray(model["private_mode"], dtype=np.float64)
            )
            private_index = self.state["E4"]["selected_private_channel_index"]
            train_tensor = full_feature_tensor(
                data.train["sequence_u"], basis, self.config["cadence_sec"]
            )
            test_tensor = full_feature_tensor(
                data.test["sequence_u"], basis, self.config["cadence_sec"]
            )
            train_base = filtered_mode_features(train_tensor, shared_modes, support)
            test_base = filtered_mode_features(test_tensor, shared_modes, support)
            if private_mode is not None and private_index is not None:
                train_base = np.column_stack(
                    (
                        train_base,
                        train_tensor[:, private_index, :] @ private_mode,
                    )
                )
                test_base = np.column_stack(
                    (
                        test_base,
                        test_tensor[:, private_index, :] @ private_mode,
                    )
                )
            base_fit = ridge_fit(
                train_base,
                data.train["target_z"],
                float(
                    self.state["E2"]["directions"][direction]["selected"][
                        "lambda_0"
                    ]
                )
                * np.eye(train_base.shape[1]),
                predict_matrix=test_base,
            )
            prediction = base_fit.prediction.copy()
            ood = {}
            if selected_channel is not None:
                amplitude = cached[(direction, selected_channel)]["amplitude"]
                time_modes = cached[(direction, selected_channel)]["time_modes"]
                train_amplitude, _ = evaluate_amplitude_basis(
                    amplitude, data.train["sequence_u"][:, :, selected_channel]
                )
                test_amplitude, ood = evaluate_amplitude_basis(
                    amplitude, data.test["sequence_u"][:, :, selected_channel]
                )
                train_nonlinear = nonlinear_mode_features(
                    train_amplitude,
                    basis.lag_design @ time_modes,
                    cadence_sec=self.config["cadence_sec"],
                )
                test_nonlinear = nonlinear_mode_features(
                    test_amplitude,
                    basis.lag_design @ time_modes,
                    cadence_sec=self.config["cadence_sec"],
                )
                nonlinear_fit = ridge_fit(
                    train_nonlinear,
                    data.train["target_z"] - train_base @ base_fit.coefficient,
                    float(selected_penalty) * np.eye(train_nonlinear.shape[1]),
                    predict_matrix=test_nonlinear,
                )
                prediction += nonlinear_fit.prediction
                ood["prediction_contribution_fraction"] = float(
                    np.mean(np.abs(nonlinear_fit.prediction))
                    / max(np.mean(np.abs(prediction)), 1e-30)
                )
            _atomic_npz(
                self.results / f"predictions/e6_nonlinear/{direction}.npz",
                sample_id=data.test["sample_id"],
                prediction=prediction,
                target_z=data.test["target_z"],
                evaluation_mask=data.test["evaluation_mask"],
            )
            direction_results[direction] = {
                "test_metrics": metrics(
                    data.test["target_z"][data.test["evaluation_mask"]],
                    prediction[data.test["evaluation_mask"]],
                ),
                "c1_extension": ood,
            }
        _write_csv(self.results / "NONLINEAR_RESULTS.csv", rows)
        payload = {
            "status": "PASS",
            "selected_nonlinear_channel_index": selected_channel,
            "selected_nonlinear_channel": (
                None if selected_channel is None else CHANNELS[selected_channel]
            ),
            "selected_nonlinear_penalty": selected_penalty,
            "nonlinear_exact_zero": selected_channel is None,
            "one_se_threshold": threshold,
            "directions": direction_results,
        }
        atomic_json(self.results / "c1_extension/nonlinear_selection.json", payload)
        return payload

    def stage_e7(self) -> dict[str, Any]:
        source_stage = "e6_nonlinear"
        if self.state["E6"]["status"] == "NOT_APPLICABLE":
            source_stage = "e5_fixed"
        direction_results: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        for direction in DIRECTIONS:
            data = load_direction(self._shared_root(), direction)
            with np.load(
                self.results / f"predictions/{source_stage}/{direction}.npz"
            ) as stored:
                test_k_prediction = np.asarray(
                    stored["prediction"], dtype=np.float64
                )
            model = self.state["E5"]["directions"][direction]
            mother = self.state["E2"]["mother_space"]
            basis = self._basis(direction, mother)
            support = tuple(
                self.state["E2"]["directions"][direction]["selected"]["support"]
            )
            shared_modes = np.asarray(model["shared_modes"], dtype=np.float64)
            private_mode = (
                None
                if model["private_mode"] is None
                else np.asarray(model["private_mode"], dtype=np.float64)
            )
            private_index = self.state["E4"]["selected_private_channel_index"]
            train_tensor = full_feature_tensor(
                data.train["sequence_u"], basis, self.config["cadence_sec"]
            )
            train_base = filtered_mode_features(train_tensor, shared_modes, support)
            if private_mode is not None and private_index is not None:
                train_base = np.column_stack(
                    (
                        train_base,
                        train_tensor[:, private_index, :] @ private_mode,
                    )
                )
            protocol = load_protocol(self._shared_root())
            purge_raw = int(
                round(
                    self.config["purge_min"]
                    * 60.0
                    / float(protocol["sample_period_sec"])
                )
            )
            folds = inner_folds(
                data.train["origin_raw_index"],
                protocol["inner_folds"],
                purge_raw_samples=purge_raw,
            )
            oof_prediction = np.full(len(data.train["target_z"]), np.nan)
            for training, validation in folds:
                fit = ridge_fit(
                    train_base[training],
                    data.train["target_z"][training],
                    float(
                        self.state["E2"]["directions"][direction]["selected"][
                            "lambda_0"
                        ]
                    )
                    * np.eye(train_base.shape[1]),
                    predict_matrix=train_base[validation],
                )
                oof_prediction[validation] = fit.prediction
            residual = data.train["target_z"] - oof_prediction
            maturity_rows = int(
                round(
                    (self.config["horizon_min"] + self.config["target_window_min"])
                    * 60.0
                    / self.config["cadence_sec"]
                )
            )

            def residual_design(values: np.ndarray, candidate: str) -> tuple[np.ndarray, np.ndarray]:
                maximum_history = int(
                    round(
                        max(self.config["residual_history_candidates_min"])
                        * 60.0
                        / self.config["cadence_sec"]
                    )
                )
                eligible = np.arange(
                    maturity_rows + maximum_history, len(values), dtype=np.int64
                )
                finite = np.isfinite(values[eligible])
                if candidate.startswith("AR:"):
                    minutes = float(candidate.split(":", 1)[1])
                    history = int(
                        round(
                            minutes
                            * 60.0
                            / self.config["cadence_sec"]
                        )
                    )
                    if history == 0:
                        return np.zeros((len(eligible), 0)), eligible[finite]
                    index = (
                        eligible[:, None]
                        - maturity_rows
                        - np.arange(history, dtype=np.int64)[None, :]
                    )
                    finite &= np.all(np.isfinite(values[index]), axis=1)
                    return values[index[finite]], eligible[finite]
                dimension = int(candidate.split(":", 1)[1])
                history = maximum_history
                index = (
                    eligible[:, None]
                    - maturity_rows
                    - np.arange(history, dtype=np.int64)[None, :]
                )
                finite &= np.all(np.isfinite(values[index]), axis=1)
                histories = values[index[finite]]
                time_constants = np.geomspace(2.0, 40.0, dimension)
                lag_minutes = (
                    np.arange(history, dtype=np.float64)
                    * self.config["cadence_sec"]
                    / 60.0
                )
                weights = np.exp(
                    -lag_minutes[:, None] / time_constants[None, :]
                )
                weights /= np.sum(weights, axis=0, keepdims=True)
                return histories @ weights, eligible[finite]

            candidates = ["A0"] + [
                f"AR:{minutes:g}"
                for minutes in self.config["residual_history_candidates_min"]
                if minutes > 0
            ] + [
                f"STATE:{dimension}"
                for dimension in self.config["residual_state_dimensions"]
            ]
            candidate_rows: list[dict[str, Any]] = []
            for candidate in candidates:
                if candidate == "A0":
                    eligible = np.flatnonzero(np.isfinite(residual))
                    values = residual[eligible]
                    mean = float(np.mean(values * values))
                    candidate_rows.append(
                        {
                            "direction": direction,
                            "candidate": candidate,
                            "lambda": None,
                            "oof_mse": mean,
                            "oof_se": 0.0,
                        }
                    )
                    continue
                design, eligible = residual_design(residual, candidate)
                if len(eligible) < 100:
                    continue
                origins = data.train["origin_raw_index"][eligible]
                candidate_folds = inner_folds(
                    origins,
                    protocol["inner_folds"],
                    purge_raw_samples=purge_raw,
                )
                for ridge in self.config["ridge_grid"]:
                    fold_errors = []
                    for training, validation in candidate_folds:
                        fit = ridge_fit(
                            design[training],
                            residual[eligible[training]],
                            float(ridge) * np.eye(design.shape[1]),
                            predict_matrix=design[validation],
                        )
                        error = residual[eligible[validation]] - fit.prediction
                        fold_errors.append(float(np.mean(error * error)))
                    mean, standard_error = fold_mse(fold_errors)
                    candidate_rows.append(
                        {
                            "direction": direction,
                            "candidate": candidate,
                            "lambda": float(ridge),
                            "oof_mse": mean,
                            "oof_se": standard_error,
                            "fold_mse": fold_errors,
                        }
                    )
            minimum = min(candidate_rows, key=lambda row: row["oof_mse"])
            threshold = minimum["oof_mse"] + minimum["oof_se"]
            one_se = [
                row
                for row in candidate_rows
                if row["oof_mse"] <= threshold + 1e-15
            ]
            selected = min(
                one_se,
                key=lambda row: (
                    row["candidate"] != "A0",
                    0
                    if row["candidate"] == "A0"
                    else (
                        float(row["candidate"].split(":", 1)[1])
                        if row["candidate"].startswith("AR:")
                        else 1000
                        + int(row["candidate"].split(":", 1)[1])
                    ),
                    -float(row["lambda"] or 0.0),
                ),
            )
            correction = np.zeros(len(test_k_prediction), dtype=np.float64)
            if selected["candidate"] != "A0":
                train_design, train_eligible = residual_design(
                    residual, selected["candidate"]
                )
                fit = ridge_fit(
                    train_design,
                    residual[train_eligible],
                    float(selected["lambda"]) * np.eye(train_design.shape[1]),
                )
                test_residual = data.test["target_z"] - test_k_prediction
                test_design, test_eligible = residual_design(
                    test_residual, selected["candidate"]
                )
                correction[test_eligible] = test_design @ fit.coefficient
            prediction = test_k_prediction + correction
            _atomic_npz(
                self.results / f"predictions/e7_residual/{direction}.npz",
                sample_id=data.test["sample_id"],
                prediction=prediction,
                target_z=data.test["target_z"],
                evaluation_mask=data.test["evaluation_mask"],
                k_prediction=test_k_prediction,
                residual_correction=correction,
            )
            rows.extend(candidate_rows)
            direction_results[direction] = {
                "selected_model": selected["candidate"],
                "selected_lambda": selected["lambda"],
                "one_se_threshold": threshold,
                "test_metrics": metrics(
                    data.test["target_z"][data.test["evaluation_mask"]],
                    prediction[data.test["evaluation_mask"]],
                ),
            }
        _write_csv(self.results / "RESIDUAL_AR_RESULTS.csv", rows)
        payload = {"status": "PASS", "directions": direction_results}
        atomic_json(self.results / "diagnostics/e7_residual.json", payload)
        return payload

    def stage_e8(self) -> dict[str, Any]:
        shared_root, cpu_root, gpu_root = self._paths()
        cpu_index = prediction_files(cpu_root)
        input_models = {
            "Persistence": (cpu_index, None),
            "K-only": (cpu_index, None),
            "Dynamic-PLS": (cpu_index, None),
            "NLinear-U": (gpu_root, "GPU_ENSEMBLE"),
            "Shared-Private-K": (None, "e6_nonlinear"),
        }
        if self.state["E6"]["status"] == "NOT_APPLICABLE":
            input_models["Shared-Private-K"] = (None, "e5_fixed")
        dynamic_models = {
            "Temporal Autoencoder": (gpu_root, "GPU_ENSEMBLE"),
            "Joint-K+AR": (cpu_index, None),
            "Shared-Private-K-to-Residual": (None, "e7_residual"),
        }

        def collect(model_name: str, source) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
            index, stage = source
            payloads = []
            directions = {}
            for direction in DIRECTIONS:
                if stage == "GPU_ENSEMBLE":
                    payload, _ = load_gpu_seed_ensemble(
                        Path(index), direction, model_name
                    )
                elif index is None:
                    payload = load_prediction(
                        self.results / f"predictions/{stage}/{direction}.npz"
                    )
                else:
                    payload = load_prediction(
                        resolve_prediction(index, direction, model_name)
                    )
                mask = payload["evaluation_mask"]
                directions[direction] = metrics(
                    payload["target_z"][mask], payload["prediction"][mask]
                )
                payloads.append(payload)
            return {
                "model": model_name,
                "directions": directions,
                "pooled": pooled_metrics(payloads),
            }, payloads

        input_rows = []
        dynamic_rows = []
        payload_by_model: dict[str, list[dict[str, np.ndarray]]] = {}
        for name, source in input_models.items():
            result, payloads = collect(name, source)
            payload_by_model[name] = payloads
            input_rows.append(
                {
                    "model": name,
                    "pooled_MSE": result["pooled"]["MSE"],
                    "pooled_RMSE": result["pooled"]["RMSE"],
                    "pooled_MAE": result["pooled"]["MAE"],
                    "pooled_R2": result["pooled"]["R2"],
                    **{
                        f"{direction}_RMSE": result["directions"][direction][
                            "RMSE"
                        ]
                        for direction in DIRECTIONS
                    },
                }
            )
        for name, source in dynamic_models.items():
            result, payloads = collect(name, source)
            payload_by_model[name] = payloads
            dynamic_rows.append(
                {
                    "model": name,
                    "pooled_MSE": result["pooled"]["MSE"],
                    "pooled_RMSE": result["pooled"]["RMSE"],
                    "pooled_MAE": result["pooled"]["MAE"],
                    "pooled_R2": result["pooled"]["R2"],
                    **{
                        f"{direction}_RMSE": result["directions"][direction][
                            "RMSE"
                        ]
                        for direction in DIRECTIONS
                    },
                }
            )
        input_rows.sort(key=lambda row: row["pooled_MSE"])
        dynamic_rows.sort(key=lambda row: row["pooled_MSE"])
        _write_csv(self.results / "FINAL_INPUT_LEADERBOARD.csv", input_rows)
        _write_csv(self.results / "FINAL_DYNAMIC_LEADERBOARD.csv", dynamic_rows)
        comparisons = [
            ("Shared-Private-K", model)
            for model in ("Persistence", "K-only", "Dynamic-PLS", "NLinear-U")
        ] + [
            ("Shared-Private-K-to-Residual", model)
            for model in ("Temporal Autoencoder", "Joint-K+AR")
        ]
        bootstrap_rows: list[dict[str, Any]] = []
        for candidate, baseline in comparisons:
            candidate_payloads = payload_by_model[candidate]
            baseline_payloads = payload_by_model[baseline]
            for direction_index, direction in enumerate(DIRECTIONS):
                candidate_payload = candidate_payloads[direction_index]
                baseline_payload = baseline_payloads[direction_index]
                mask = candidate_payload["evaluation_mask"]
                candidate_error = (
                    candidate_payload["target_z"][mask]
                    - candidate_payload["prediction"][mask]
                ) ** 2
                baseline_error = (
                    baseline_payload["target_z"][mask]
                    - baseline_payload["prediction"][mask]
                ) ** 2
                for block_minutes in self.config["bootstrap_block_lengths_min"]:
                    block_rows = int(
                        round(
                            float(block_minutes)
                            * 60.0
                            / self.config["cadence_sec"]
                        )
                    )
                    result = _moving_block_bootstrap(
                        baseline_error - candidate_error,
                        block_rows=block_rows,
                        replicates=self.config["bootstrap_replicates"],
                        seed=self.config["random_seed"]
                        + 10000 * direction_index
                        + 100 * int(block_minutes),
                    )
                    bootstrap_rows.append(
                        {
                            "candidate": candidate,
                            "baseline": baseline,
                            "direction": direction,
                            "block_min": block_minutes,
                            **result,
                            "relative_mse_improvement": float(
                                (
                                    np.mean(baseline_error)
                                    - np.mean(candidate_error)
                                )
                                / max(np.mean(baseline_error), 1e-30)
                            ),
                        }
                    )
            pooled_candidate_error = np.concatenate(
                [
                    (
                        payload["target_z"][payload["evaluation_mask"]]
                        - payload["prediction"][payload["evaluation_mask"]]
                    )
                    ** 2
                    for payload in candidate_payloads
                ]
            )
            pooled_baseline_error = np.concatenate(
                [
                    (
                        payload["target_z"][payload["evaluation_mask"]]
                        - payload["prediction"][payload["evaluation_mask"]]
                    )
                    ** 2
                    for payload in baseline_payloads
                ]
            )
            for block_minutes in self.config["bootstrap_block_lengths_min"]:
                result = _moving_block_bootstrap(
                    pooled_baseline_error - pooled_candidate_error,
                    block_rows=int(
                        round(
                            float(block_minutes)
                            * 60.0
                            / self.config["cadence_sec"]
                        )
                    ),
                    replicates=self.config["bootstrap_replicates"],
                    seed=self.config["random_seed"] + 50000 + int(block_minutes),
                )
                bootstrap_rows.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "direction": "pooled",
                        "block_min": block_minutes,
                        **result,
                        "relative_mse_improvement": float(
                            (
                                np.mean(pooled_baseline_error)
                                - np.mean(pooled_candidate_error)
                            )
                            / max(np.mean(pooled_baseline_error), 1e-30)
                        ),
                    }
                )
        _write_csv(self.results / "PAIRWISE_BOOTSTRAP.csv", bootstrap_rows)
        new_input = next(
            row for row in input_rows if row["model"] == "Shared-Private-K"
        )
        nlinear = next(row for row in input_rows if row["model"] == "NLinear-U")
        old_k = next(row for row in input_rows if row["model"] == "K-only")
        persistence = next(
            row for row in input_rows if row["model"] == "Persistence"
        )
        primary_old_k = next(
            row
            for row in bootstrap_rows
            if row["candidate"] == "Shared-Private-K"
            and row["baseline"] == "K-only"
            and row["direction"] == "pooled"
            and row["block_min"] == self.config["bootstrap_primary_block_min"]
        )
        both_positive = all(
            new_input[f"{direction}_RMSE"]
            < persistence[f"{direction}_RMSE"]
            for direction in DIRECTIONS
        )
        shared_certified = (
            self.state["E3"]["shared_certification"] == "SHARED_K_CERTIFIED"
        )
        private_ok = self.state["E4"]["private_certification"] in {
            "PRIVATE_EXACT_ZERO",
            "PRIVATE_RANK1_CERTIFIED",
        }
        numerical = (
            self.state["E5"]["kkt_certification"] == "PASS"
            and self.state["E5"]["orthogonality_certification"] == "PASS"
        )
        if (
            new_input["pooled_RMSE"] < nlinear["pooled_RMSE"]
            and both_positive
            and primary_old_k["positive_probability"] >= 0.95
            and shared_certified
            and private_ok
            and numerical
        ):
            level = "LEVEL_A_K_CERTIFIED"
            registration = "K_CERTIFIED"
        elif (
            abs(new_input["pooled_MSE"] - nlinear["pooled_MSE"])
            / nlinear["pooled_MSE"]
            <= 0.01
            and both_positive
            and shared_certified
            and numerical
        ):
            level = "LEVEL_B_INTERPRETABLE_PARETO"
            registration = "K_CERTIFIED"
        elif new_input["pooled_MSE"] < old_k["pooled_MSE"]:
            level = "LEVEL_C_PARTIAL_SUCCESS"
            registration = "PREDICTIVE_ONLY"
        else:
            level = "LEVEL_D_REJECTED"
            registration = "REJECTED"
        decision = {
            "mother_space": self.state["E2"]["mother_space"],
            "active_channels": {
                direction: [
                    CHANNELS[index]
                    for index in self.state["E2"]["directions"][direction][
                        "selected"
                    ]["support"]
                ]
                for direction in DIRECTIONS
            },
            "shared_rank": self.state["E3"]["selected_rank"],
            "private_channel": self.state["E4"]["selected_private_channel"],
            "private_rank": (
                0 if self.state["E4"]["selected_private_channel"] is None else 1
            ),
            "nonlinear_channel": self.state["E6"].get(
                "selected_nonlinear_channel"
            ),
            "nonlinear_exact_zero": self.state["E6"].get(
                "nonlinear_exact_zero", True
            ),
            "residual_model": {
                direction: self.state["E7"]["directions"][direction][
                    "selected_model"
                ]
                for direction in DIRECTIONS
            },
            "direction_metrics": {
                direction: {
                    "input_RMSE": new_input[f"{direction}_RMSE"],
                    "dynamic_RMSE": next(
                        row
                        for row in dynamic_rows
                        if row["model"] == "Shared-Private-K-to-Residual"
                    )[f"{direction}_RMSE"],
                }
                for direction in DIRECTIONS
            },
            "pooled_metrics": {
                "input": new_input,
                "dynamic": next(
                    row
                    for row in dynamic_rows
                    if row["model"] == "Shared-Private-K-to-Residual"
                ),
            },
            "shared_certification": self.state["E3"]["shared_certification"],
            "private_certification": self.state["E4"][
                "private_certification"
            ],
            "physical_registration": registration,
            "success_level": level,
            "comparison_vs_old_K": {
                "relative_MSE_improvement": (
                    old_k["pooled_MSE"] - new_input["pooled_MSE"]
                )
                / old_k["pooled_MSE"],
                "bootstrap_positive_probability_40min": primary_old_k[
                    "positive_probability"
                ],
            },
            "comparison_vs_NLinear": {
                "relative_MSE_improvement": (
                    nlinear["pooled_MSE"] - new_input["pooled_MSE"]
                )
                / nlinear["pooled_MSE"]
            },
            "comparison_vs_Joint_K_AR": {
                "relative_MSE_improvement": next(
                    row
                    for row in bootstrap_rows
                    if row["candidate"] == "Shared-Private-K-to-Residual"
                    and row["baseline"] == "Joint-K+AR"
                    and row["direction"] == "pooled"
                    and row["block_min"]
                    == self.config["bootstrap_primary_block_min"]
                )["relative_mse_improvement"]
            },
        }
        atomic_json(self.results / "FINAL_DECISION.json", decision)
        self._plots(input_rows, dynamic_rows, bootstrap_rows)
        self._report(decision, input_rows, dynamic_rows)
        payload = {
            "status": "PASS",
            "decision": decision,
            "input_leaderboard": input_rows,
            "dynamic_leaderboard": dynamic_rows,
        }
        return payload

    def _plots(
        self,
        input_rows: list[dict[str, Any]],
        dynamic_rows: list[dict[str, Any]],
        bootstrap_rows: list[dict[str, Any]],
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_root = self.results / "plots"
        plot_root.mkdir(parents=True, exist_ok=True)

        complexity_names = ["FULL-LIN", "S", "SP", "SP-NL", "SP-AR"]
        complexity_rmse = []
        for direction in DIRECTIONS:
            with np.load(
                self.results / f"predictions/e2_full/{direction}.npz"
            ) as stored:
                mask = stored["evaluation_mask"]
                complexity_rmse.append(
                    [
                        metrics(
                            stored["target_z"][mask], stored["prediction"][mask]
                        )["RMSE"]
                    ]
                )
            stages = ["e3_shared", "e4_private"]
            stages.append(
                "e5_fixed"
                if self.state["E6"]["status"] == "NOT_APPLICABLE"
                else "e6_nonlinear"
            )
            stages.append("e7_residual")
            for stage in stages:
                with np.load(
                    self.results / f"predictions/{stage}/{direction}.npz"
                ) as stored:
                    mask = stored["evaluation_mask"]
                    complexity_rmse[-1].append(
                        metrics(
                            stored["target_z"][mask],
                            stored["prediction"][mask],
                        )["RMSE"]
                    )
        figure, axis = plt.subplots(figsize=(9, 5))
        for direction, values in zip(DIRECTIONS, complexity_rmse):
            axis.plot(complexity_names, values, marker="o", label=direction)
        axis.set_ylabel("RMSE")
        axis.set_title("RMSE versus registered model complexity")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_root / "01_rmse_vs_model_complexity.png", dpi=180)
        plt.close(figure)

        selected_rows = input_rows + dynamic_rows[-1:]
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for axis, direction in zip(axes, DIRECTIONS):
            names = [row["model"] for row in selected_rows]
            values = [row[f"{direction}_RMSE"] for row in selected_rows]
            axis.barh(names, values, color="#4c78a8")
            axis.invert_yaxis()
            axis.set_title(direction)
            axis.set_xlabel("RMSE")
        figure.suptitle("Bidirectional outer-transfer RMSE")
        figure.tight_layout()
        figure.savefig(plot_root / "02_direction_rmse.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 5))
        for direction in DIRECTIONS:
            singular = np.asarray(
                self.state["E3"]["directions"][direction]["singular_values"]
            )
            axis.semilogy(
                np.arange(1, len(singular) + 1),
                np.maximum(singular, 1e-15),
                marker="o",
                label=direction,
            )
        axis.set_xlabel("Mode")
        axis.set_ylabel("Gram-whitened singular value")
        axis.set_title("Public singular-value spectrum")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_root / "03_shared_singular_spectrum.png", dpi=180)
        plt.close(figure)

        figure, axes = plt.subplots(
            len(DIRECTIONS), 1, figsize=(10, 7), sharex=True
        )
        for axis, direction in zip(axes, DIRECTIONS):
            basis = self._basis(direction, self.state["E2"]["mother_space"])
            modes = np.asarray(
                self.state["E5"]["directions"][direction]["shared_modes"]
            )
            curves = basis.lag_design @ modes
            for index in range(curves.shape[1]):
                axis.plot(
                    basis.lag_minutes,
                    curves[:, index],
                    label=f"q{index + 1}",
                )
            axis.set_title(direction)
            axis.set_ylabel("q(tau)")
            if curves.shape[1]:
                axis.legend()
            else:
                axis.text(
                    0.5,
                    0.5,
                    "SHARED_RANK_ZERO",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel("Lag (min)")
        figure.suptitle("Public time bases")
        figure.tight_layout()
        figure.savefig(plot_root / "04_shared_time_bases.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(9, 5))
        plotted = False
        for direction in DIRECTIONS:
            private = self.state["E5"]["directions"][direction]["private_mode"]
            if private is None:
                continue
            basis = self._basis(direction, self.state["E2"]["mother_space"])
            axis.plot(
                basis.lag_minutes,
                basis.lag_design @ np.asarray(private),
                label=direction,
            )
            plotted = True
        if not plotted:
            axis.text(
                0.5,
                0.5,
                "PRIVATE_EXACT_ZERO",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_title("Private candidate time basis")
        axis.set_xlabel("Lag (min)")
        axis.set_ylabel("p(tau)")
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend()
        figure.tight_layout()
        figure.savefig(plot_root / "05_private_time_basis.png", dpi=180)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for axis, direction in zip(axes, DIRECTIONS):
            rank = int(self.state["E3"]["selected_rank"])
            full_right = np.asarray(
                self.state["E3"]["full_models"][direction]["full_right_vectors"]
            )
            support = self.state["E3"]["full_models"][direction]["support"]
            if rank:
                width = 0.8 / rank
                x = np.arange(len(CHANNELS))
                for mode_index in range(rank):
                    load = np.zeros(len(CHANNELS))
                    load[support] = full_right[mode_index, support]
                    axis.bar(
                        x + mode_index * width,
                        load,
                        width=width,
                        label=f"mode {mode_index + 1}",
                    )
                axis.set_xticks(x + width * (rank - 1) / 2, CHANNELS, rotation=25)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "SHARED_RANK_ZERO",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_title(direction)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Whitened loading")
        if int(self.state["E3"]["selected_rank"]) > 0:
            axes[0].legend()
        figure.suptitle("Channel loadings")
        figure.tight_layout()
        figure.savefig(plot_root / "06_channel_loadings.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(9, 5))
        stages = ["Shared", "Private", "Nonlinear", "Residual"]
        for direction in DIRECTIONS:
            values = []
            previous = None
            for stage in ("e3_shared", "e4_private", "e6_nonlinear", "e7_residual"):
                if stage == "e6_nonlinear" and self.state["E6"]["status"] == "NOT_APPLICABLE":
                    stage = "e5_fixed"
                with np.load(
                    self.results / f"predictions/{stage}/{direction}.npz"
                ) as stored:
                    mask = stored["evaluation_mask"]
                    mse = metrics(
                        stored["target_z"][mask], stored["prediction"][mask]
                    )["MSE"]
                values.append(0.0 if previous is None else previous - mse)
                previous = mse
            axis.plot(stages, values, marker="o", label=direction)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_ylabel("Incremental MSE reduction")
        axis.set_title("Shared/private/nonlinear/residual gain decomposition")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_root / "07_gain_decomposition.png", dpi=180)
        plt.close(figure)

        shared_root, cpu_root, gpu_root = self._paths()
        cpu_index = prediction_files(cpu_root)
        gpu_index = prediction_files(gpu_root)
        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        for axis, direction in zip(axes, DIRECTIONS):
            new = load_prediction(
                self.results
                / (
                    f"predictions/e5_fixed/{direction}.npz"
                    if self.state["E6"]["status"] == "NOT_APPLICABLE"
                    else f"predictions/e6_nonlinear/{direction}.npz"
                )
            )
            mask = new["evaluation_mask"]
            for name, index in (("old K", cpu_index), ("NLinear", None)):
                requested = "K-only" if name == "old K" else "NLinear-U"
                if index is None:
                    old, _ = load_gpu_seed_ensemble(
                        gpu_root, direction, requested
                    )
                else:
                    old = load_prediction(
                        resolve_prediction(index, direction, requested)
                    )
                difference = (
                    old["target_z"][mask] - old["prediction"][mask]
                ) ** 2 - (
                    new["target_z"][mask] - new["prediction"][mask]
                ) ** 2
                axis.plot(difference, alpha=0.7, linewidth=0.8, label=name)
            axis.axhline(0, color="black", linewidth=1)
            axis.set_title(direction)
            axis.set_ylabel("Old SE - new SE")
            axis.legend()
        axes[-1].set_xlabel("Common evaluation row")
        figure.suptitle("Paired error-difference time series")
        figure.tight_layout()
        figure.savefig(plot_root / "08_error_difference_timeseries.png", dpi=180)
        plt.close(figure)

        primary = [
            row
            for row in bootstrap_rows
            if row["candidate"] == "Shared-Private-K"
            and row["direction"] == "pooled"
            and row["block_min"] == self.config["bootstrap_primary_block_min"]
        ]
        figure, axis = plt.subplots(figsize=(9, 5))
        labels = [row["baseline"] for row in primary]
        medians = [row["median"] for row in primary]
        lower = [median - row["lower_95"] for median, row in zip(medians, primary)]
        upper = [row["upper_95"] - median for median, row in zip(medians, primary)]
        axis.errorbar(
            np.arange(len(labels)),
            medians,
            yerr=np.vstack((lower, upper)),
            fmt="o",
            capsize=4,
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=20)
        axis.set_ylabel("Baseline MSE - new K MSE")
        axis.set_title("40-minute paired moving-block bootstrap")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_root / "09_bootstrap_intervals.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(9, 5))
        if self.state["E6"]["status"] == "PASS":
            labels = list(DIRECTIONS)
            ood = [
                self.state["E6"]["directions"][direction]["c1_extension"].get(
                    "ood_fraction", 0.0
                )
                for direction in labels
            ]
            contribution = [
                self.state["E6"]["directions"][direction]["c1_extension"].get(
                    "prediction_contribution_fraction", 0.0
                )
                for direction in labels
            ]
            x = np.arange(len(labels))
            axis.bar(x - 0.18, ood, 0.36, label="OOD fraction")
            axis.bar(
                x + 0.18,
                contribution,
                0.36,
                label="prediction contribution",
            )
            axis.set_xticks(x, labels)
            axis.legend()
        else:
            axis.text(
                0.5,
                0.5,
                "NONLINEAR_NOT_APPLICABLE\nC1 contribution = 0",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_title("OOD and finite-band C1 continuation")
        axis.set_ylabel("Fraction")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_root / "10_c1_ood_contribution.png", dpi=180)
        plt.close(figure)

    def _report(
        self,
        decision: dict[str, Any],
        input_rows: list[dict[str, Any]],
        dynamic_rows: list[dict[str, Any]],
    ) -> None:
        input_new = next(
            row for row in input_rows if row["model"] == "Shared-Private-K"
        )
        dynamic_new = next(
            row
            for row in dynamic_rows
            if row["model"] == "Shared-Private-K-to-Residual"
        )
        lines = [
            "# OPS-UOI Shared–Private K CPU Confirm V1 Final Report",
            "",
            f"- Registration: `{decision['physical_registration']}`",
            f"- Success level: `{decision['success_level']}`",
            f"- Mother space: `{decision['mother_space']}`",
            f"- Shared rank: `{decision['shared_rank']}`",
            f"- Private channel: `{decision['private_channel']}`",
            f"- Nonlinear channel: `{decision['nonlinear_channel']}`",
            f"- Residual models: `{json.dumps(decision['residual_model'], ensure_ascii=False)}`",
            "",
            "## Frozen L6 result",
            "",
            (
                f"The input-only shared/private K has pooled RMSE "
                f"`{input_new['pooled_RMSE']:.9f}` and pooled MSE "
                f"`{input_new['pooled_MSE']:.9f}`. The final K→Residual model "
                f"has pooled RMSE `{dynamic_new['pooled_RMSE']:.9f}`."
            ),
            "",
            "| Direction | Input K RMSE | K→Residual RMSE |",
            "|---|---:|---:|",
        ]
        for direction in DIRECTIONS:
            lines.append(
                f"| {direction} | {input_new[f'{direction}_RMSE']:.9f} | "
                f"{dynamic_new[f'{direction}_RMSE']:.9f} |"
            )
        lines.extend(
            [
                "",
                "## Certification",
                "",
                f"- Shared: `{decision['shared_certification']}`",
                f"- Private: `{decision['private_certification']}`",
                f"- Nonlinear exact zero: `{decision['nonlinear_exact_zero']}`",
                (
                    "- Relative MSE improvement versus old K-only: "
                    f"`{decision['comparison_vs_old_K']['relative_MSE_improvement']:.6%}`"
                ),
                (
                    "- 40-minute paired-bootstrap positive probability versus "
                    f"old K-only: "
                    f"`{decision['comparison_vs_old_K']['bootstrap_positive_probability_40min']:.6f}`"
                ),
                (
                    "- Relative MSE improvement versus NLinear-U: "
                    f"`{decision['comparison_vs_NLinear']['relative_MSE_improvement']:.6%}`"
                ),
                (
                    "- Relative MSE improvement of final dynamic model versus "
                    f"Joint-K+AR: "
                    f"`{decision['comparison_vs_Joint_K_AR']['relative_MSE_improvement']:.6%}`"
                ),
                "",
                "The physical registration follows the frozen bidirectional and "
                "subspace gates; pooled RMSE alone is not used to claim cross-rod "
                "physical stability.",
            ]
        )
        (self.results / "FINAL_REPORT.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _final_terminal(self) -> None:
        checkpoint = json.loads(
            self.checkpoint_path.read_text(encoding="utf-8")
        )
        bundle = build_bundle(
            self.root,
            shared_hash=sha256_file(self.shared_bundle),
            cpu_hash=sha256_file(self.cpu_bundle),
            gpu_hash=sha256_file(self.gpu_bundle),
            protocol_hash=checkpoint["config_sha256"],
        )
        print(f"FINAL_ZIP={bundle['zip']}")
        print(f"FINAL_SHA256={bundle['sha256']}")
        print(f"ZIP_SIZE={bundle['size']}")
        print(f"MANIFEST_FILE_COUNT={bundle['manifest_file_count']}")
        print(f"PROTOCOL_SHA256={checkpoint['config_sha256']}")
        print(f"SHARED_DATASET_SHA256={sha256_file(self.shared_bundle)}")
        print(f"CPU_BASELINE_BUNDLE_SHA256={sha256_file(self.cpu_bundle)}")
        print(f"GPU_BASELINE_BUNDLE_SHA256={sha256_file(self.gpu_bundle)}")
        print(f"VALIDATION_STATUS={self.state.get('E8', {}).get('status', 'UNKNOWN')}")
