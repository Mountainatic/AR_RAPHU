from __future__ import annotations

import csv
import json
import os
import pickle
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .basis import _hermite_nonlinear_features, _spline_matrix, build_projected_designs
from .io_data import (
    DIRECTIONS,
    atomic_json,
    atomic_npz,
    find_named_root,
    inner_folds,
    load_cpu_prediction,
    load_direction,
    load_gpu_ensemble,
    load_protocol,
    metrics,
    moving_block_bootstrap,
    pooled_metrics,
    published_mse,
    safe_extract,
    sha256_array,
    sha256_file,
    validate_alignment,
)
from .model import direction_metrics, fit_task
from .packaging import build_bundle
from .residual import apply_residual_model, select_residual_model


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nNOT_APPLICABLE\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
    os.replace(temporary, path)


def _save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)
    os.replace(temporary, path)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _corr(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if np.std(first) <= 1e-15 or np.std(second) <= 1e-15:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


class Experiment:
    def __init__(
        self,
        root: Path,
        *,
        shared_bundle: Path,
        cpu_bundle: Path,
        gpu_bundle: Path,
        protocol_path: Path,
        n_jobs: int,
        bootstrap_jobs: int,
    ) -> None:
        self.root = root.resolve()
        self.shared_bundle = shared_bundle.resolve()
        self.cpu_bundle = cpu_bundle.resolve()
        self.gpu_bundle = gpu_bundle.resolve()
        self.protocol_path = protocol_path.resolve()
        self.config = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.n_jobs = int(n_jobs)
        self.bootstrap_jobs = int(bootstrap_jobs)
        self.results = self.root / "results"
        self.work = self.root / "work"
        self.logs = self.root / "logs"
        self.checkpoint = self.results / "checkpoints/latest.json"
        for directory in (self.results, self.work, self.logs, self.results / "diagnostics", self.results / "plots", self.results / "predictions", self.results / "surfaces", self.results / "projections", self.results / "spectra", self.results / "ood"):
            directory.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.checkpoint.read_text(encoding="utf-8")) if self.checkpoint.exists() else {"schema": self.config["schema"], "completed": []}

    def _mark(self, stage: str, payload: dict[str, Any]) -> None:
        self.state[stage] = payload
        if stage not in self.state["completed"]:
            self.state["completed"].append(stage)
        atomic_json(self.checkpoint, self.state)

    def _roots(self) -> tuple[Path, Path, Path]:
        shared = find_named_root(safe_extract(self.shared_bundle, self.work / "shared_bundle"), "SHARED_BENCHMARK_DATASET")
        cpu = find_named_root(safe_extract(self.cpu_bundle, self.work / "cpu_bundle"), "PHYSICS_FIRST_CPU_RESULTS")
        gpu = find_named_root(safe_extract(self.gpu_bundle, self.work / "gpu_bundle"), "PHYSICS_FIRST_GPU_RESULTS")
        return shared, cpu, gpu

    def _artifact_path(self, direction: str, kind: str) -> Path:
        return self.work / "artifacts" / f"{direction}__{kind}.pkl"

    def _artifact(self, direction: str, kind: str = "main") -> dict[str, Any]:
        return _load_pickle(self._artifact_path(direction, kind))

    def run(self) -> int:
        stages = [self.stage_e0, self.stage_e1_e2, self.stage_e3, self.stage_e4, self.stage_e5, self.stage_e6_e7, self.stage_e8, self.stage_e9]
        for function in stages:
            stage = function.__name__.replace("stage_", "").upper()
            if stage in self.state.get("completed", []):
                print(f"{stage}=CHECKPOINT_REUSED", flush=True)
                continue
            print(f"{stage}=START", flush=True)
            payload = function()
            self._mark(stage, payload)
            print(f"{stage}=PASS", flush=True)
        bundle = build_bundle(
            self.root,
            protocol_sha256=sha256_file(self.protocol_path),
            shared_sha256=sha256_file(self.shared_bundle),
            cpu_sha256=sha256_file(self.cpu_bundle),
            gpu_sha256=sha256_file(self.gpu_bundle),
        )
        print(f"FINAL_ZIP={bundle['zip']}")
        print(f"FINAL_SHA256={bundle['sha256']}")
        print(f"ZIP_SIZE={bundle['size']}")
        print(f"MANIFEST_FILE_COUNT={bundle['manifest_file_count']}")
        print(f"PROTOCOL_SHA256={sha256_file(self.protocol_path)}")
        print(f"SHARED_DATASET_SHA256={sha256_file(self.shared_bundle)}")
        print(f"CPU_BASELINE_BUNDLE_SHA256={sha256_file(self.cpu_bundle)}")
        print(f"GPU_BASELINE_BUNDLE_SHA256={sha256_file(self.gpu_bundle)}")
        print("VALIDATION_STATUS=PASS")
        return 0

    def stage_e0(self) -> dict[str, Any]:
        hashes = {"shared": sha256_file(self.shared_bundle), "cpu": sha256_file(self.cpu_bundle), "gpu": sha256_file(self.gpu_bundle)}
        expected = {
            "shared": self.config["shared_dataset_sha256"],
            "cpu": self.config["cpu_baseline_bundle_sha256"],
            "gpu": self.config["gpu_baseline_bundle_sha256"],
        }
        if hashes != expected:
            raise RuntimeError(f"BUNDLE_HASH_MISMATCH:{hashes}:{expected}")
        shared_root, cpu_root, gpu_root = self._roots()
        protocol = load_protocol(shared_root)
        for key in ("cadence_sec", "history_min", "horizon_min", "target_window_min", "controls"):
            if protocol[key] != self.config[key]:
                raise RuntimeError(f"PROTOCOL_MISMATCH:{key}:{protocol[key]}:{self.config[key]}")
        report: dict[str, Any] = {}
        for model in ("Persistence", "K-only", "Dynamic-PLS", "Joint-K+AR"):
            payloads = []
            directions = {}
            for direction in DIRECTIONS:
                reference_data = load_direction(shared_root, direction).test
                reference = {key: reference_data[key] for key in ("sample_id", "target_z", "evaluation_mask")}
                candidate = load_cpu_prediction(cpu_root, direction, model)
                validate_alignment(reference, candidate, f"{model}:{direction}")
                mask = candidate["evaluation_mask"]
                directions[direction] = metrics(candidate["target_z"][mask], candidate["prediction"][mask])
                payloads.append(candidate)
            pooled = pooled_metrics(payloads)
            published = published_mse(cpu_root, model)
            difference = abs(float(pooled["MSE"]) - published)
            if difference > 1e-10:
                raise RuntimeError(f"PUBLISHED_METRIC_MISMATCH:{model}:{difference}")
            report[model] = {"directions": directions, "pooled": pooled, "published_mse": published, "difference": difference}
        for model in ("NLinear-U", "Temporal Autoencoder"):
            payloads = []
            seed_mse_by_direction: dict[str, list[float]] = {}
            row_count: dict[str, int] = {}
            directions = {}
            for direction in DIRECTIONS:
                reference_data = load_direction(shared_root, direction).test
                reference = {key: reference_data[key] for key in ("sample_id", "target_z", "evaluation_mask")}
                candidate, seed_mse = load_gpu_ensemble(gpu_root, direction, model)
                validate_alignment(reference, candidate, f"{model}:{direction}")
                mask = candidate["evaluation_mask"]
                directions[direction] = metrics(candidate["target_z"][mask], candidate["prediction"][mask])
                seed_mse_by_direction[direction] = seed_mse
                row_count[direction] = int(np.sum(mask))
                payloads.append(candidate)
            pooled_seed = [sum(seed_mse_by_direction[d][seed] * row_count[d] for d in DIRECTIONS) / sum(row_count.values()) for seed in range(min(map(len, seed_mse_by_direction.values())))]
            reproduced = float(np.median(pooled_seed))
            published = published_mse(gpu_root, model)
            if abs(reproduced - published) > 1e-10:
                raise RuntimeError(f"PUBLISHED_METRIC_MISMATCH:{model}:{reproduced}:{published}")
            report[model] = {"directions_seed_median_ensemble": directions, "ensemble_pooled": pooled_metrics(payloads), "published_seed_median_pooled_mse": published, "reproduced": reproduced, "difference": abs(reproduced - published)}
        rows = {direction: int(np.sum(load_direction(shared_root, direction).test["evaluation_mask"])) for direction in DIRECTIONS}
        rows["pooled"] = sum(rows.values())
        if rows != self.config["expected_evaluation_rows"]:
            raise RuntimeError(f"EVALUATION_ROWS_MISMATCH:{rows}")
        payload = {
            "status": "PASS", "hashes": hashes, "evaluation_rows": rows,
            "target_hash": {direction: sha256_array(load_direction(shared_root, direction).test["target_z"]) for direction in DIRECTIONS},
            "models": report,
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "n_jobs": self.n_jobs, "bootstrap_jobs": self.bootstrap_jobs, "threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
            "raw_excel_present": any(path.suffix.lower() in {".xlsx", ".xls"} for path in self.root.rglob("*")),
        }
        if payload["raw_excel_present"]:
            raise RuntimeError("RAW_EXCEL_ENTERED_EXPERIMENT")
        atomic_json(self.results / "preflight/precheck.json", payload)
        (self.results / "PRECHECK_REPORT.md").write_text(
            "# PRECHECK REPORT\n\n`STATUS=PASS`\n\n"
            f"- Shared SHA256: `{hashes['shared']}`\n- CPU baseline SHA256: `{hashes['cpu']}`\n- GPU baseline SHA256: `{hashes['gpu']}`\n"
            f"- Registered evaluation rows: `{rows['pooled']}`\n- Sample IDs, targets, masks and published metrics reproduce within `1e-10`.\n- Raw Excel present: `false`.\n",
            encoding="utf-8",
        )
        return payload

    def stage_e1_e2(self) -> dict[str, Any]:
        shared_root, _, _ = self._roots()
        main_payloads = [{"shared_root": str(shared_root), "direction": direction, "mode": "main", "config": self.config} for direction in DIRECTIONS if not self._artifact_path(direction, "main").exists()]
        if main_payloads:
            with ProcessPoolExecutor(max_workers=min(2, self.n_jobs)) as pool:
                futures = {pool.submit(fit_task, payload): payload for payload in main_payloads}
                for future in as_completed(futures):
                    artifact = future.result()
                    _save_pickle(self._artifact_path(artifact["direction"], "main"), artifact)
                    print(f"MAIN_FIT_DONE={artifact['direction']}", flush=True)
        protocol = load_protocol(shared_root)
        purge_raw = int(round(float(self.config["purge_min"]) * 60.0 / float(protocol["sample_period_sec"])))
        tasks: list[dict[str, Any]] = []
        for direction in DIRECTIONS:
            data = load_direction(shared_root, direction)
            if not self._artifact_path(direction, "refine").exists():
                tasks.append({"shared_root": str(shared_root), "direction": direction, "mode": "refine", "config": self.config})
            folds = inner_folds(data.train["origin_raw_index"], protocol["inner_folds"], purge_raw_samples=purge_raw)
            for fold_index, (training, validation) in enumerate(folds):
                if not self._artifact_path(direction, f"fold{fold_index}").exists():
                    tasks.append({"shared_root": str(shared_root), "direction": direction, "mode": "fold", "fold": fold_index, "training": training, "validation": validation, "config": self.config})
        if tasks:
            with ProcessPoolExecutor(max_workers=min(self.n_jobs, len(tasks))) as pool:
                futures = {pool.submit(fit_task, payload): payload for payload in tasks}
                for future in as_completed(futures):
                    result = future.result()
                    kind = "refine" if futures[future]["mode"] == "refine" else f"fold{result['fold']}"
                    _save_pickle(self._artifact_path(result["direction"], kind), result)
                    print(f"FIT_DONE={result['direction']}:{kind}", flush=True)
        design_rows = []
        gcv_rows = []
        metrics_rows = []
        for direction in DIRECTIONS:
            artifact = self._artifact(direction)
            data = load_direction(shared_root, direction)
            if artifact["partition_error"] > 1e-12:
                raise RuntimeError(f"PARTITION_OF_UNITY_FAILED:{direction}:{artifact['partition_error']}")
            if artifact["kkt_residual"] > self.config["kkt_tolerance"]:
                raise RuntimeError(f"KKT_FAILED:{direction}:{artifact['kkt_residual']}")
            if artifact["reconstruction_error"] > self.config["reconstruction_tolerance"]:
                raise RuntimeError(f"PROJECTION_RECONSTRUCTION_FAILED:{direction}:{artifact['reconstruction_error']}")
            if artifact["constant_constraint_residual"] > self.config["orthogonality_tolerance"] or artifact["linear_constraint_residual"] > self.config["orthogonality_tolerance"]:
                raise RuntimeError(f"ORTHOGONALITY_FAILED:{direction}")
            for row in artifact["gcv_curve"]:
                gcv_rows.append({"direction": direction, **row})
            design_rows.append({
                "direction": direction, "lag_basis": artifact["lag"].number_of_basis,
                "coefficient_count": len(artifact["coefficient"]), "partition_error": artifact["partition_error"],
                "penalty_min_generalized_eigenvalue": float(np.min(artifact["generalized_eigenvalues"])),
                "selected_lambda": artifact["selected_lambda"], "effective_df": artifact["effective_df"],
                "kkt_residual": artifact["kkt_residual"], "condition_number": artifact["condition_number"],
                "coefficient_sha256": artifact["coefficient_sha256"], "prediction_sha256": artifact["prediction_sha256"],
                "amplitude_support": artifact["amplitude_manifest"],
                "no_future_input": bool(np.all(data.test["origin_raw_index"] < data.test["future_left_raw_index"])),
                "no_break_crossing": True,
            })
            for model, key in (("R1-LIN-DERIVED", "prediction_rank1"), ("LIN-UOI", "prediction_linear"), ("FULL-UOI", "prediction_full")):
                metrics_rows.append({"direction": direction, "model": model, **direction_metrics(artifact, key)})
            for channel, surface in artifact["surfaces"].items():
                atomic_npz(self.results / f"surfaces/{direction}__{channel}.npz", **surface)
            for key, name in (("prediction_rank1", "R1-LIN-DERIVED"), ("prediction_linear", "LIN-UOI"), ("prediction_full", "FULL-UOI")):
                atomic_npz(self.results / f"predictions/{name}/{direction}.npz", sample_id=artifact["sample_id"], target_z=artifact["target_z"], evaluation_mask=artifact["evaluation_mask"], prediction=artifact[key])
        atomic_json(self.results / "DESIGN_AUDIT.json", {"status": "PASS", "directions": design_rows})
        _write_csv(self.results / "GCV_RESULTS.csv", gcv_rows)
        _write_csv(self.results / "FULL_URYSOHN_METRICS.csv", metrics_rows)
        return {"status": "PASS", "directions": design_rows}

    def stage_e3(self) -> dict[str, Any]:
        rows = []
        for direction in DIRECTIONS:
            artifact = self._artifact(direction)
            for channel in self.config["controls"]:
                linear = artifact["linear_energy"][channel]
                nonlinear = artifact["nonlinear_energy"][channel]
                rows.append({
                    "direction": direction, "channel": channel,
                    "linear_energy": linear, "nonlinear_energy": nonlinear,
                    "nonlinear_fraction": nonlinear / max(linear + nonlinear, 1e-30),
                    "constant_constraint_residual": artifact["constant_constraint_residual"],
                    "linear_constraint_residual": artifact["linear_constraint_residual"],
                    "reconstruction_error": artifact["reconstruction_error"],
                })
        _write_csv(self.results / "AMPLITUDE_PROJECTION_METRICS.csv", rows)
        return {"status": "PASS", "rows": rows}

    def stage_e4(self) -> dict[str, Any]:
        rows = []
        canonical: dict[str, dict[str, np.ndarray]] = {}
        for direction in DIRECTIONS:
            artifact = self._artifact(direction)
            time = artifact["rank1_time_shape"].copy()
            loading = artifact["rank1_channel_coordinates"].copy()
            pivot = int(np.argmax(np.abs(loading)))
            if loading[pivot] < 0:
                time *= -1
                loading *= -1
            canonical[direction] = {"time": time, "loading": loading}
            mask = artifact["evaluation_mask"]
            rank1_metric = metrics(artifact["target_z"][mask], artifact["prediction_rank1"][mask])
            linear_metric = metrics(artifact["target_z"][mask], artifact["prediction_linear"][mask])
            rows.append({
                "direction": direction, "rank1_energy_ratio": artifact["rank1_energy_ratio"],
                "singular_values": artifact["singular_values"].tolist(),
                "spectral_gap": float(artifact["singular_values"][0] / max(artifact["singular_values"][1], 1e-30)),
                "rank1_RMSE": rank1_metric["RMSE"], "general_linear_RMSE": linear_metric["RMSE"],
                "rank1_vs_linear_relative_MSE_gap": abs(float(rank1_metric["MSE"]) - float(linear_metric["MSE"])) / max(float(linear_metric["MSE"]), 1e-30),
                "channel_coordinates": loading.tolist(),
            })
            atomic_npz(self.results / f"spectra/{direction}__nlinear_rank1.npz", singular_values=artifact["singular_values"], time_shape=time, channel_coordinates=loading, beta=artifact["beta"], beta_rank1=artifact["beta_rank1"])
        time_correlation = abs(_corr(canonical[DIRECTIONS[0]]["time"], canonical[DIRECTIONS[1]]["time"]))
        sign_agreement = float(np.mean(np.sign(canonical[DIRECTIONS[0]]["loading"]) == np.sign(canonical[DIRECTIONS[1]]["loading"])))
        pooled_rank1 = self._pooled_new("prediction_rank1")
        pooled_linear = self._pooled_new("prediction_linear")
        pooled_gap = abs(float(pooled_rank1["MSE"]) - float(pooled_linear["MSE"])) / max(float(pooled_linear["MSE"]), 1e-30)
        persistence = self._baseline_payloads("Persistence", gpu=False)
        direction_beats = []
        for direction, baseline in zip(DIRECTIONS, persistence):
            artifact = self._artifact(direction)
            mask = artifact["evaluation_mask"]
            direction_beats.append(float(metrics(artifact["target_z"][mask], artifact["prediction_rank1"][mask])["MSE"]) < float(metrics(baseline["target_z"][mask], baseline["prediction"][mask])["MSE"]))
        supported = (
            all(self._artifact(direction)["rank1_energy_ratio"] >= 0.90 for direction in DIRECTIONS)
            and time_correlation >= 0.7 and sign_agreement >= 0.75 and pooled_gap <= 0.02 and all(direction_beats)
        )
        payload = {"status": "PASS", "structure_status": "NLINEAR_PROJECTION_SUPPORTED" if supported else "NLINEAR_PREDICTIVE_ONLY", "time_shape_abs_correlation": time_correlation, "channel_sign_agreement": sign_agreement, "pooled_rank1_vs_linear_relative_mse_gap": pooled_gap, "directions": rows}
        _write_csv(self.results / "NLINEAR_DERIVED_AUDIT.csv", [{**row, "time_shape_abs_correlation": time_correlation, "channel_sign_agreement": sign_agreement, "structure_status": payload["structure_status"]} for row in rows])
        atomic_json(self.results / "diagnostics/nlinear_projection.json", payload)
        return payload

    def _baseline_payloads(self, model: str, *, gpu: bool) -> list[dict[str, np.ndarray]]:
        _, cpu_root, gpu_root = self._roots()
        if gpu:
            return [load_gpu_ensemble(gpu_root, direction, model)[0] for direction in DIRECTIONS]
        return [load_cpu_prediction(cpu_root, direction, model) for direction in DIRECTIONS]

    def _pooled_new(self, key: str) -> dict[str, float | int]:
        targets = []
        predictions = []
        for direction in DIRECTIONS:
            artifact = self._artifact(direction)
            mask = artifact["evaluation_mask"]
            targets.append(artifact["target_z"][mask])
            predictions.append(artifact[key][mask])
        return metrics(np.concatenate(targets), np.concatenate(predictions))

    def _surface_correlations(self) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        first = self._artifact(DIRECTIONS[0])
        second = self._artifact(DIRECTIONS[1])
        for channel in self.config["controls"]:
            a = first["surfaces"][channel]
            b = second["surfaces"][channel]
            lower = max(float(a["amplitude_value"].min()), float(b["amplitude_value"].min()))
            upper = min(float(a["amplitude_value"].max()), float(b["amplitude_value"].max()))
            if upper <= lower:
                output[channel] = {"beta": 0.0, "nonlinear": 0.0, "full": 0.0}
                continue
            grid = np.linspace(lower, upper, 121)
            def interpolate(surface: dict[str, np.ndarray], key: str) -> np.ndarray:
                return np.vstack([np.interp(grid, surface["amplitude_value"], row) for row in surface[key]])
            output[channel] = {
                "beta": _corr(a["beta"], b["beta"]),
                "nonlinear": _corr(interpolate(a, "nonlinear"), interpolate(b, "nonlinear")),
                "full": _corr(interpolate(a, "full"), interpolate(b, "full")),
            }
        return output

    def stage_e5(self) -> dict[str, Any]:
        shared_root, _, _ = self._roots()
        bootstrap_rows: list[dict[str, Any]] = []
        direction_results: dict[str, Any] = {}
        observed_nonlinear_gain: dict[str, float] = {}
        placebo_destroyed: dict[str, bool] = {}
        ood_not_primary: dict[str, bool] = {}
        mesh_stable: dict[str, bool] = {}
        for direction_index, direction in enumerate(DIRECTIONS):
            artifact = self._artifact(direction)
            refine = self._artifact(direction, "refine")
            data = load_direction(shared_root, direction)
            mask = artifact["evaluation_mask"]
            linear_error = artifact["target_z"][mask] - artifact["prediction_linear"][mask]
            full_error = artifact["target_z"][mask] - artifact["prediction_full"][mask]
            observed_nonlinear_gain[direction] = float(np.mean(linear_error**2) - np.mean(full_error**2))
            for block in self.config["bootstrap_block_lengths_min"]:
                result = moving_block_bootstrap([linear_error], [full_error], block_rows=int(round(block * 60 / self.config["cadence_sec"])), replicates=self.config["bootstrap_replicates"], seed=self.config["bootstrap_seed"] + direction_index * 1000 + int(block))
                bootstrap_rows.append({"comparison": "LIN-UOI_vs_FULL-UOI", "direction": direction, "block_min": block, **result})
            shift = int(round(self.config["placebo_shift_min"] * 60 / self.config["cadence_sec"]))
            shifted_sequence = data.test["sequence_u"][:-shift]
            linear_design, nonlinear_design, _, _ = build_projected_designs(shifted_sequence, artifact["lag"], artifact["amplitude"], self.config["cadence_sec"])
            beta_vector = artifact["beta"].T.reshape(-1)
            placebo_linear = artifact["intercept"] + linear_design @ beta_vector
            placebo_full = placebo_linear + nonlinear_design @ artifact["nonlinear_coefficient"]
            placebo_mask = mask[shift:]
            placebo_target = artifact["target_z"][shift:][placebo_mask]
            placebo_gain = float(np.mean((placebo_target - placebo_linear[placebo_mask]) ** 2) - np.mean((placebo_target - placebo_full[placebo_mask]) ** 2))
            placebo_destroyed[direction] = placebo_gain <= max(0.0, 0.5 * observed_nonlinear_gain[direction])
            squared_delta = linear_error**2 - full_error**2
            evaluation_common = artifact["common_support_mask"][mask]
            positive_total = float(np.sum(np.maximum(squared_delta, 0.0)))
            ood_positive = float(np.sum(np.maximum(squared_delta[~evaluation_common], 0.0)))
            ood_fraction = ood_positive / max(positive_total, 1e-30)
            ood_not_primary[direction] = ood_fraction <= 0.5
            main_metrics = metrics(artifact["target_z"][mask], artifact["prediction_full"][mask])
            refine_metrics = metrics(artifact["target_z"][mask], refine["prediction_full"][mask])
            old = load_cpu_prediction(self._roots()[1], direction, "K-only")
            old_metrics = metrics(old["target_z"][mask], old["prediction"][mask])
            same_conclusion = (float(main_metrics["MSE"]) < float(old_metrics["MSE"])) == (float(refine_metrics["MSE"]) < float(old_metrics["MSE"]))
            mesh_delta = abs(float(main_metrics["RMSE"]) - float(refine_metrics["RMSE"])) / max(float(main_metrics["RMSE"]), 1e-30)
            mesh_stable[direction] = same_conclusion and mesh_delta <= 0.05
            direction_results[direction] = {
                "linear_mse": float(np.mean(linear_error**2)), "full_mse": float(np.mean(full_error**2)),
                "nonlinear_mse_gain": observed_nonlinear_gain[direction], "placebo_gain": placebo_gain,
                "placebo_destroyed": placebo_destroyed[direction], "ood_positive_gain_fraction": ood_fraction,
                "ood_not_primary": ood_not_primary[direction], "refinement_rmse": refine_metrics["RMSE"],
                "mesh_relative_rmse_delta": mesh_delta, "mesh_stable": mesh_stable[direction],
            }
        pooled_linear_errors = []
        pooled_full_errors = []
        for direction in DIRECTIONS:
            artifact = self._artifact(direction)
            mask = artifact["evaluation_mask"]
            pooled_linear_errors.append(artifact["target_z"][mask] - artifact["prediction_linear"][mask])
            pooled_full_errors.append(artifact["target_z"][mask] - artifact["prediction_full"][mask])
        for block in self.config["bootstrap_block_lengths_min"]:
            result = moving_block_bootstrap(pooled_linear_errors, pooled_full_errors, block_rows=int(round(block * 60 / self.config["cadence_sec"])), replicates=self.config["bootstrap_replicates"], seed=self.config["bootstrap_seed"] + 9000 + int(block))
            bootstrap_rows.append({"comparison": "LIN-UOI_vs_FULL-UOI", "direction": "pooled", "block_min": block, **result})
        primary = next(row for row in bootstrap_rows if row["direction"] == "pooled" and row["block_min"] == self.config["bootstrap_primary_block_min"])
        sensitivity_consistent = all((row["positive_probability"] >= 0.95) == (primary["positive_probability"] >= 0.95) for row in bootstrap_rows if row["direction"] == "pooled")
        surface = self._surface_correlations()
        minimum_nonlinear_correlation = min(abs(value["nonlinear"]) for value in surface.values())
        certified = (
            all(result["full_mse"] <= result["linear_mse"] + 1e-15 for result in direction_results.values())
            and primary["positive_probability"] >= 0.95 and sensitivity_consistent
            and minimum_nonlinear_correlation >= 0.6 and all(placebo_destroyed.values()) and all(ood_not_primary.values())
        )
        payload = {
            "status": "PASS", "nonlinear_increment_status": "NONLINEAR_INCREMENT_CERTIFIED" if certified else "NONLINEAR_INCREMENT_NOT_CERTIFIED",
            "directions": direction_results, "primary_bootstrap": primary, "block_sensitivity_consistent": sensitivity_consistent,
            "surface_correlations": surface, "minimum_common_support_nonlinear_correlation": minimum_nonlinear_correlation,
            "mesh_status": "BASIS_RESOLUTION_STABLE" if all(mesh_stable.values()) else "BASIS_RESOLUTION_INSUFFICIENT",
        }
        _write_csv(self.results / "NONLINEAR_INCREMENT_AUDIT.csv", bootstrap_rows)
        atomic_json(self.results / "diagnostics/nonlinear_increment.json", payload)
        return payload

    def _c1_audit(self, artifact: dict[str, Any]) -> dict[str, float | bool]:
        maximum_value_jump = 0.0
        maximum_derivative_jump = 0.0
        for spec in artifact["amplitude"]:
            for boundary, band in ((spec.lower, spec.left_band), (spec.upper, spec.right_band)):
                epsilon = max(band * 1e-6, 1e-8)
                xi = np.array([boundary - 2 * epsilon, boundary - epsilon, boundary, boundary + epsilon, boundary + 2 * epsilon])
                values = spec.mean + spec.scale * xi
                feature, _ = _hermite_nonlinear_features(spec, values)
                left_derivative = (feature[2] - feature[0]) / (2 * epsilon)
                right_derivative = (feature[4] - feature[2]) / (2 * epsilon)
                maximum_value_jump = max(maximum_value_jump, float(np.max(np.abs((feature[1] + feature[3]) / 2 - feature[2]))))
                maximum_derivative_jump = max(maximum_derivative_jump, float(np.max(np.abs(left_derivative - right_derivative))))
        return {"maximum_value_continuity_residual": maximum_value_jump, "maximum_derivative_continuity_residual": maximum_derivative_jump, "pass": maximum_value_jump < 1e-5 and maximum_derivative_jump < 1e-3}

    def stage_e6_e7(self) -> dict[str, Any]:
        surface_corr = self._surface_correlations()
        channel_rows: list[dict[str, Any]] = []
        c1_rows: list[dict[str, Any]] = []
        channel_direction: dict[str, dict[str, Any]] = {channel: {} for channel in self.config["controls"]}
        for direction_index, direction in enumerate(DIRECTIONS):
            artifact = self._artifact(direction)
            mask = artifact["evaluation_mask"]
            target = artifact["target_z"][mask]
            full = artifact["prediction_full"][mask]
            full_error = target - full
            sample_region = np.max(artifact["predict_region"], axis=(1, 2))
            c1 = self._c1_audit(artifact)
            for region_code, region_name in ((-1, "all_registered"), (0, "fit_support"), (1, "extension_band"), (2, "saturated")):
                region_mask = mask if region_code < 0 else mask & (sample_region == region_code)
                row = {"direction": direction, "region": region_name, "rows": int(np.sum(region_mask)), **c1}
                if np.any(region_mask):
                    row.update(metrics(artifact["target_z"][region_mask], artifact["prediction_full"][region_mask]))
                c1_rows.append(row)
            for channel in self.config["controls"]:
                loo = full - artifact["channel_prediction"][channel][mask]
                loo_error = target - loo
                delta = float(np.mean(loo_error**2) - np.mean(full_error**2))
                boot = moving_block_bootstrap([loo_error], [full_error], block_rows=int(round(40 * 60 / self.config["cadence_sec"])), replicates=self.config["bootstrap_replicates"], seed=self.config["bootstrap_seed"] + 12000 + direction_index * 100 + list(self.config["controls"]).index(channel))
                row = {
                    "direction": direction, "channel": channel,
                    "linear_energy": artifact["linear_energy"][channel], "nonlinear_energy": artifact["nonlinear_energy"][channel],
                    "loo_mse_delta": delta, "bootstrap_positive_probability": boot["positive_probability"],
                    "beta_cross_direction_correlation": surface_corr[channel]["beta"],
                    "nonlinear_cross_direction_correlation": surface_corr[channel]["nonlinear"],
                    "full_surface_cross_direction_correlation": surface_corr[channel]["full"],
                    "placebo_destroyed": self.state["E5"]["directions"][direction]["placebo_destroyed"],
                    "ood_positive_gain_fraction": self.state["E5"]["directions"][direction]["ood_positive_gain_fraction"],
                }
                channel_direction[channel][direction] = row
                channel_rows.append(row)
        statuses = {}
        for channel in self.config["controls"]:
            rows = [channel_direction[channel][direction] for direction in DIRECTIONS]
            if all(row["loo_mse_delta"] > 0 and row["bootstrap_positive_probability"] >= 0.95 and row["placebo_destroyed"] for row in rows) and abs(rows[0]["beta_cross_direction_correlation"]) >= 0.6 and abs(rows[0]["nonlinear_cross_direction_correlation"]) >= 0.6:
                status = "CHANNEL_SURFACE_CERTIFIED"
            elif sum(row["loo_mse_delta"] for row in rows) > 0:
                status = "PREDICTIVE_ONLY"
            else:
                status = "UNRESOLVED"
            statuses[channel] = status
            for row in channel_rows:
                if row["channel"] == channel:
                    row["channel_status"] = status
        if not all(row["pass"] for row in c1_rows):
            raise RuntimeError("C1_CONTINUITY_FAILED")
        _write_csv(self.results / "CHANNEL_SURFACE_AUDIT.csv", channel_rows)
        _write_csv(self.results / "C1_EXTENSION_AUDIT.csv", c1_rows)
        payload = {"status": "PASS", "channel_status": statuses, "c1_status": "PASS", "surface_correlations": surface_corr}
        atomic_json(self.results / "diagnostics/channel_c1.json", payload)
        return payload

    def stage_e8(self) -> dict[str, Any]:
        shared_root, _, _ = self._roots()
        protocol = load_protocol(shared_root)
        rows: list[dict[str, Any]] = []
        directions: dict[str, Any] = {}
        for direction in DIRECTIONS:
            data = load_direction(shared_root, direction)
            oof = np.full(len(data.train["target_z"]), np.nan, dtype=np.float64)
            fold_lambdas = []
            for fold in range(len(protocol["inner_folds"])):
                artifact = self._artifact(direction, f"fold{fold}")
                oof[artifact["validation"]] = artifact["prediction_full"]
                fold_lambdas.append(artifact["selected_lambda"])
            oof_residual = data.train["target_z"] - oof
            selected = select_residual_model(oof_residual, data.train["origin_raw_index"], data.train["future_right_raw_index"], config=self.config, protocol=protocol)
            for row in selected["rows"]:
                rows.append({"direction": direction, **row})
            main = self._artifact(direction)
            test_residual = data.test["target_z"] - main["prediction_full"]
            correction, eligible, maximum_source = apply_residual_model(selected, test_residual, data.test["origin_raw_index"], data.test["future_right_raw_index"], config=self.config, protocol=protocol)
            final = main["prediction_full"] + correction
            mask = main["evaluation_mask"]
            atomic_npz(self.results / f"predictions/FULL-UOI-PSAR/{direction}.npz", sample_id=main["sample_id"], target_z=main["target_z"], evaluation_mask=mask, prediction=final, full_k_prediction=main["prediction_full"], residual_correction=correction)
            directions[direction] = {
                "selected_model": selected["selected"]["candidate"], "selected_ridge": selected["selected"]["ridge"],
                "one_se_threshold": selected["selected"]["one_se_threshold"], "fold_k_lambda": fold_lambdas,
                "oof_rows": int(np.sum(np.isfinite(oof))), "test_residual_eligible_rows": int(len(eligible)),
                "minimum_causality_margin_raw": int(np.min(data.test["origin_raw_index"][eligible] - maximum_source)) if len(eligible) else None,
                "test_metrics": metrics(main["target_z"][mask], final[mask]),
            }
        _write_csv(self.results / "RESIDUAL_PSAR_RESULTS.csv", rows)
        payload = {"status": "PASS", "directions": directions}
        atomic_json(self.results / "diagnostics/residual_psar.json", payload)
        return payload

    def _model_payloads(self) -> dict[str, list[dict[str, np.ndarray]]]:
        models: dict[str, list[dict[str, np.ndarray]]] = {
            "Persistence": self._baseline_payloads("Persistence", gpu=False),
            "old K-only": self._baseline_payloads("K-only", gpu=False),
            "Dynamic-PLS": self._baseline_payloads("Dynamic-PLS", gpu=False),
            "NLinear-U": self._baseline_payloads("NLinear-U", gpu=True),
            "Temporal Autoencoder": self._baseline_payloads("Temporal Autoencoder", gpu=True),
            "Joint-K+AR": self._baseline_payloads("Joint-K+AR", gpu=False),
        }
        for model, key in (("R1-LIN-DERIVED", "prediction_rank1"), ("LIN-UOI", "prediction_linear"), ("FULL-UOI", "prediction_full")):
            models[model] = []
            for direction in DIRECTIONS:
                artifact = self._artifact(direction)
                models[model].append({"sample_id": artifact["sample_id"], "target_z": artifact["target_z"], "evaluation_mask": artifact["evaluation_mask"], "prediction": artifact[key]})
        models["FULL-UOI-PSAR"] = []
        for direction in DIRECTIONS:
            with np.load(self.results / f"predictions/FULL-UOI-PSAR/{direction}.npz", allow_pickle=False) as stored:
                models["FULL-UOI-PSAR"].append({key: stored[key] for key in ("sample_id", "target_z", "evaluation_mask", "prediction")})
        return models

    def _leaderboard_row(self, model: str, payloads: list[dict[str, np.ndarray]]) -> dict[str, Any]:
        row: dict[str, Any] = {"model": model}
        for direction, payload in zip(DIRECTIONS, payloads):
            mask = payload["evaluation_mask"]
            values = metrics(payload["target_z"][mask], payload["prediction"][mask])
            for key, value in values.items():
                row[f"{direction}_{key}"] = value
        pooled = pooled_metrics(payloads)
        for key, value in pooled.items():
            row[f"pooled_{key}"] = value
        return row

    def _pairwise_bootstrap(self, models: dict[str, list[dict[str, np.ndarray]]]) -> list[dict[str, Any]]:
        comparisons = [
            ("Persistence", "FULL-UOI"), ("old K-only", "FULL-UOI"), ("NLinear-U", "FULL-UOI"),
            ("LIN-UOI", "FULL-UOI"), ("R1-LIN-DERIVED", "LIN-UOI"),
            ("FULL-UOI", "FULL-UOI-PSAR"), ("Joint-K+AR", "FULL-UOI-PSAR"),
        ]
        rows = []
        for comparison_index, (baseline_name, candidate_name) in enumerate(comparisons):
            baseline_errors = []
            candidate_errors = []
            for direction_index, direction in enumerate(DIRECTIONS):
                baseline = models[baseline_name][direction_index]
                candidate = models[candidate_name][direction_index]
                validate_alignment(baseline, candidate, f"PAIR:{baseline_name}:{candidate_name}:{direction}")
                mask = baseline["evaluation_mask"]
                base_error = baseline["target_z"][mask] - baseline["prediction"][mask]
                cand_error = candidate["target_z"][mask] - candidate["prediction"][mask]
                baseline_errors.append(base_error)
                candidate_errors.append(cand_error)
                for block in self.config["bootstrap_block_lengths_min"]:
                    result = moving_block_bootstrap([base_error], [cand_error], block_rows=int(round(block * 60 / self.config["cadence_sec"])), replicates=self.config["bootstrap_replicates"], seed=self.config["bootstrap_seed"] + 20000 + comparison_index * 1000 + direction_index * 100 + int(block))
                    rows.append({"baseline": baseline_name, "candidate": candidate_name, "direction": direction, "block_min": block, **result})
            for block in self.config["bootstrap_block_lengths_min"]:
                result = moving_block_bootstrap(baseline_errors, candidate_errors, block_rows=int(round(block * 60 / self.config["cadence_sec"])), replicates=self.config["bootstrap_replicates"], seed=self.config["bootstrap_seed"] + 30000 + comparison_index * 1000 + int(block))
                rows.append({"baseline": baseline_name, "candidate": candidate_name, "direction": "pooled", "block_min": block, **result})
        return rows

    def stage_e9(self) -> dict[str, Any]:
        models = self._model_payloads()
        input_order = ["Persistence", "old K-only", "Dynamic-PLS", "NLinear-U", "R1-LIN-DERIVED", "LIN-UOI", "FULL-UOI"]
        dynamic_order = ["Temporal Autoencoder", "Joint-K+AR", "FULL-UOI-PSAR"]
        input_rows = [self._leaderboard_row(model, models[model]) for model in input_order]
        dynamic_rows = [self._leaderboard_row(model, models[model]) for model in dynamic_order]
        _write_csv(self.results / "FINAL_INPUT_LEADERBOARD.csv", input_rows)
        _write_csv(self.results / "FINAL_DYNAMIC_LEADERBOARD.csv", dynamic_rows)
        pairwise = self._pairwise_bootstrap(models)
        _write_csv(self.results / "PAIRWISE_BOOTSTRAP.csv", pairwise)
        primary = {(row["baseline"], row["candidate"]): row for row in pairwise if row["direction"] == "pooled" and row["block_min"] == self.config["bootstrap_primary_block_min"]}
        input_map = {row["model"]: row for row in input_rows}
        dynamic_map = {row["model"]: row for row in dynamic_rows}
        full = input_map["FULL-UOI"]
        nlinear = input_map["NLinear-U"]
        old = input_map["old K-only"]
        persistence = input_map["Persistence"]
        both_positive = all(full[f"{direction}_MSE"] < persistence[f"{direction}_MSE"] for direction in DIRECTIONS)
        mesh_ok = self.state["E5"]["mesh_status"] == "BASIS_RESOLUTION_STABLE"
        ood_ok = all(value["ood_not_primary"] for value in self.state["E5"]["directions"].values())
        numeric_ok = all(direction["kkt_residual"] <= self.config["kkt_tolerance"] for direction in self.state["E1_E2"]["directions"]) and self.state["E6_E7"]["c1_status"] == "PASS"
        if (
            full["pooled_RMSE"] < nlinear["pooled_RMSE"] and both_positive
            and primary[("old K-only", "FULL-UOI")]["positive_probability"] >= 0.95
            and primary[("NLinear-U", "FULL-UOI")]["positive_probability"] >= 0.90
            and mesh_ok and ood_ok and numeric_ok
        ):
            registration = "FULL_URYSOHN_CONFIRMED"
        elif abs(full["pooled_RMSE"] - nlinear["pooled_RMSE"]) / nlinear["pooled_RMSE"] <= 0.01 and both_positive and mesh_ok and ood_ok and numeric_ok:
            registration = "URYSOHN_PARETO_EQUIVALENT"
        elif full["pooled_RMSE"] < old["pooled_RMSE"]:
            registration = "URYSOHN_IMPROVES_OLD_K_ONLY"
        else:
            registration = "FULL_URYSOHN_REJECTED_ON_CURRENT_DATA"
        selected_residual = [self.state["E8"]["directions"][direction]["selected_model"] for direction in DIRECTIONS]
        residual_pair = primary[("FULL-UOI", "FULL-UOI-PSAR")]
        final = dynamic_map["FULL-UOI-PSAR"]
        residual_gain = (
            all(model != "A0" for model in selected_residual)
            and all(final[f"{direction}_MSE"] <= full[f"{direction}_MSE"] + 1e-15 for direction in DIRECTIONS)
            and residual_pair["positive_probability"] >= 0.90
        )
        residual_status = "MATURED_RESIDUAL_PREDICTIVE_GAIN" if residual_gain else "RESIDUAL_EXACT_ZERO"
        claims = ["registered closed-loop input-history prediction on frozen L6", "full Urysohn amplitude decomposition is numerically unique"]
        if self.state["E4"]["structure_status"] == "NLINEAR_PROJECTION_SUPPORTED":
            claims.append("linear amplitude projection is approximately shared Rank-1 on this protocol")
        if self.state["E5"]["nonlinear_increment_status"] == "NONLINEAR_INCREMENT_CERTIFIED":
            claims.append("pure nonlinear surface adds reproducible predictive information on this protocol")
        forbidden = ["open-loop plant kernel is proven", "two rods establish universal physical causality", "low RMSE alone certifies channel attribution", "OOD saturation is physically reliable"]
        decision = {
            "registration": registration,
            "nlinear_projection_status": self.state["E4"]["structure_status"],
            "nonlinear_increment_status": self.state["E5"]["nonlinear_increment_status"],
            "residual_status": residual_status,
            "gcv_lambda_by_direction": {direction: self._artifact(direction)["selected_lambda"] for direction in DIRECTIONS},
            "effective_df_by_direction": {direction: self._artifact(direction)["effective_df"] for direction in DIRECTIONS},
            "rank1_energy_ratio_by_direction": {direction: self._artifact(direction)["rank1_energy_ratio"] for direction in DIRECTIONS},
            "direction_metrics": {direction: {model: next(row for row in input_rows + dynamic_rows if row["model"] == model)[f"{direction}_RMSE"] for model in ("R1-LIN-DERIVED", "LIN-UOI", "FULL-UOI", "FULL-UOI-PSAR")} for direction in DIRECTIONS},
            "pooled_metrics": {model: {"MSE": next(row for row in input_rows + dynamic_rows if row["model"] == model)["pooled_MSE"], "RMSE": next(row for row in input_rows + dynamic_rows if row["model"] == model)["pooled_RMSE"]} for model in ("R1-LIN-DERIVED", "LIN-UOI", "FULL-UOI", "FULL-UOI-PSAR")},
            "comparison_vs_old_k": primary[("old K-only", "FULL-UOI")],
            "comparison_vs_nlinear": primary[("NLinear-U", "FULL-UOI")],
            "comparison_vs_joint_k_ar": primary[("Joint-K+AR", "FULL-UOI-PSAR")],
            "ood_summary": {direction: self.state["E5"]["directions"][direction]["ood_positive_gain_fraction"] for direction in DIRECTIONS},
            "numerical_certification": {"kkt": numeric_ok, "orthogonality": True, "c1": self.state["E6_E7"]["c1_status"], "mesh": self.state["E5"]["mesh_status"]},
            "channel_surface_status": self.state["E6_E7"]["channel_status"],
            "residual_models_by_direction": dict(zip(DIRECTIONS, selected_residual)),
            "scientific_claims_allowed": claims,
            "scientific_claims_forbidden": forbidden,
        }
        atomic_json(self.results / "FINAL_DECISION.json", decision)
        self._report(decision, input_rows, dynamic_rows)
        self._plots(input_rows, dynamic_rows, pairwise)
        return {"status": "PASS", "registration": registration, "decision": decision}

    def _report(self, decision: dict[str, Any], input_rows: list[dict[str, Any]], dynamic_rows: list[dict[str, Any]]) -> None:
        table = {row["model"]: row for row in input_rows + dynamic_rows}
        lines = [
            "# OD-FUOI NLinear Projection CPU Confirm V1 Final Report", "",
            f"- Registration: `{decision['registration']}`",
            f"- NLinear projection: `{decision['nlinear_projection_status']}`",
            f"- Nonlinear increment: `{decision['nonlinear_increment_status']}`",
            f"- Residual: `{decision['residual_status']}`", "",
            "## Frozen L6 results", "",
            "| Model | Sheet1→Sheet2 RMSE | Sheet2→Sheet1 RMSE | Pooled RMSE |",
            "|---|---:|---:|---:|",
        ]
        for model in ("Persistence", "old K-only", "Dynamic-PLS", "NLinear-U", "R1-LIN-DERIVED", "LIN-UOI", "FULL-UOI", "Temporal Autoencoder", "Joint-K+AR", "FULL-UOI-PSAR"):
            row = table[model]
            lines.append(f"| {model} | {row['sheet1_to_sheet2_RMSE']:.9f} | {row['sheet2_to_sheet1_RMSE']:.9f} | {row['pooled_RMSE']:.9f} |")
        lines.extend([
            "", "## Derived structure", "",
            f"- GCV lambda: `{json.dumps(decision['gcv_lambda_by_direction'])}`",
            f"- Effective df: `{json.dumps(decision['effective_df_by_direction'])}`",
            f"- Rank-1 energy ratios: `{json.dumps(decision['rank1_energy_ratio_by_direction'])}`",
            f"- Channel surface states: `{json.dumps(decision['channel_surface_status'])}`",
            f"- Residual models: `{json.dumps(decision['residual_models_by_direction'])}`", "",
            "## Paired 40-minute block bootstrap", "",
            f"- Full vs old K: median relative MSE improvement `{decision['comparison_vs_old_k']['median_relative_improvement']:.6%}`, positive probability `{decision['comparison_vs_old_k']['positive_probability']:.6f}`.",
            f"- Full vs NLinear: median relative MSE improvement `{decision['comparison_vs_nlinear']['median_relative_improvement']:.6%}`, positive probability `{decision['comparison_vs_nlinear']['positive_probability']:.6f}`.",
            f"- Full+PSAR vs Joint-K+AR: median relative MSE improvement `{decision['comparison_vs_joint_k_ar']['median_relative_improvement']:.6%}`, positive probability `{decision['comparison_vs_joint_k_ar']['positive_probability']:.6f}`.", "",
            "## Scientific boundary", "",
            "The fitted objects are registered closed-loop input-history response surfaces. Pooled RMSE, Rank-1 energy, or a visually smooth surface alone does not prove an open-loop plant mechanism or universal cross-rod causality.",
        ])
        (self.results / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _plots(self, input_rows: list[dict[str, Any]], dynamic_rows: list[dict[str, Any]], bootstrap: list[dict[str, Any]]) -> None:
        root = self.results / "plots"
        all_rows = input_rows + dynamic_rows
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.bar(np.arange(len(all_rows)), [row["pooled_RMSE"] for row in all_rows])
        axis.set_xticks(np.arange(len(all_rows)), [row["model"] for row in all_rows], rotation=35, ha="right")
        axis.set_ylabel("Pooled RMSE"); axis.grid(axis="y", alpha=.25); figure.tight_layout(); figure.savefig(root / "01_direction_and_pooled_rmse.png", dpi=180); plt.close(figure)

        figure, axes = plt.subplots(2, 1, figsize=(12, 7))
        for axis, direction in zip(axes, DIRECTIONS):
            artifact = self._artifact(direction); mask = artifact["evaluation_mask"]
            lin = (artifact["target_z"][mask] - artifact["prediction_linear"][mask])**2
            full = (artifact["target_z"][mask] - artifact["prediction_full"][mask])**2
            r1 = (artifact["target_z"][mask] - artifact["prediction_rank1"][mask])**2
            axis.plot(lin-full, label="LIN - FULL", lw=.8); axis.plot(r1-lin, label="R1 - LIN", lw=.8); axis.axhline(0,color="black",lw=1); axis.set_title(direction); axis.legend()
        figure.tight_layout(); figure.savefig(root / "02_full_linear_rank1_error_deltas.png", dpi=180); plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, direction in zip(axes, DIRECTIONS):
            curve = self._artifact(direction)["gcv_curve"]
            axis.semilogy([row["log10_lambda"] for row in curve], [row["gcv"] for row in curve], "o-"); axis.set_title(direction); axis.set_xlabel("log10 lambda"); axis.set_ylabel("GCV")
        figure.tight_layout(); figure.savefig(root / "03_gcv_curve.png", dpi=180); plt.close(figure)

        for plot_index, key in enumerate(("full", "beta", "nonlinear"), start=4):
            figure, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
            for row_index, direction in enumerate(DIRECTIONS):
                artifact = self._artifact(direction)
                for channel_index, channel in enumerate(self.config["controls"]):
                    surface = artifact["surfaces"][channel]
                    axis = axes[row_index, channel_index]
                    if key == "beta":
                        axis.plot(surface["lag_minutes"], surface["beta"]); axis.set_xlabel("lag min")
                    else:
                        image = axis.imshow(surface[key].T, aspect="auto", origin="lower", extent=[0,40,float(surface['amplitude_value'].min()),float(surface['amplitude_value'].max())], cmap="coolwarm")
                        figure.colorbar(image, ax=axis, shrink=.7)
                    axis.set_title(f"{direction}\n{channel}")
            figure.tight_layout(); figure.savefig(root / f"{plot_index:02d}_{key}_surfaces.png", dpi=160); plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, direction in zip(axes, DIRECTIONS):
            singular = self._artifact(direction)["singular_values"]
            axis.semilogy(np.arange(1, len(singular)+1), singular, "o-"); axis.set_title(direction); axis.set_xlabel("index")
        figure.tight_layout(); figure.savefig(root / "07_B_singular_spectrum.png", dpi=180); plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for direction in DIRECTIONS:
            artifact = self._artifact(direction); axes[0].plot(artifact["rank1_time_shape"], label=direction); axes[1].plot(self.config["controls"], artifact["rank1_channel_coordinates"], "o-", label=direction)
        axes[0].set_title("Rank-1 time shapes"); axes[1].set_title("Channel coordinates"); axes[1].tick_params(axis="x", rotation=30); axes[0].legend(); axes[1].legend(); figure.tight_layout(); figure.savefig(root / "08_rank1_shapes_and_coordinates.png", dpi=180); plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 5)); width=.35; x=np.arange(4)
        for index, direction in enumerate(DIRECTIONS):
            artifact=self._artifact(direction); ratios=[artifact["nonlinear_energy"][c]/max(artifact["linear_energy"][c]+artifact["nonlinear_energy"][c],1e-30) for c in self.config["controls"]]; axis.bar(x+(index-.5)*width,ratios,width,label=direction)
        axis.set_xticks(x,self.config["controls"],rotation=25); axis.set_ylabel("Nonlinear energy fraction"); axis.legend(); figure.tight_layout(); figure.savefig(root / "09_nonlinear_energy_ratio.png", dpi=180); plt.close(figure)

        corr=self._surface_correlations(); figure,axis=plt.subplots(figsize=(9,5)); axis.bar(self.config["controls"],[corr[c]["nonlinear"] for c in self.config["controls"]]); axis.axhline(.6,color="red",ls="--"); axis.set_ylabel("Cross-direction correlation"); axis.tick_params(axis="x",rotation=25); figure.tight_layout(); figure.savefig(root / "10_common_support_surface_comparison.png",dpi=180); plt.close(figure)

        c1_rows=list(csv.DictReader((self.results/"C1_EXTENSION_AUDIT.csv").open(encoding="utf-8"))); figure,axis=plt.subplots(figsize=(10,5)); labels=[f"{r['direction']}:{r['region']}" for r in c1_rows]; counts=[int(r['rows']) for r in c1_rows]; axis.bar(np.arange(len(labels)),counts); axis.set_xticks(np.arange(len(labels)),labels,rotation=35,ha="right"); axis.set_ylabel("Rows"); figure.tight_layout(); figure.savefig(root/"11_c1_extension_and_ood.png",dpi=180); plt.close(figure)

        primary=[row for row in bootstrap if row["direction"]=="pooled" and row["block_min"]==40]; figure,axis=plt.subplots(figsize=(12,5)); med=np.array([r["median_relative_improvement"] for r in primary]); low=med-np.array([r["lower_95"] for r in primary]); high=np.array([r["upper_95"] for r in primary])-med; axis.errorbar(np.arange(len(primary)),med,yerr=np.vstack((low,high)),fmt="o",capsize=4); axis.axhline(0,color="black"); labels=[f"{r['baseline']}→{r['candidate']}" for r in primary]; axis.set_xticks(np.arange(len(labels)),labels,rotation=30,ha="right"); axis.set_ylabel("Relative MSE improvement"); figure.tight_layout(); figure.savefig(root/"12_bootstrap_intervals.png",dpi=180); plt.close(figure)

        residual=list(csv.DictReader((self.results/"RESIDUAL_PSAR_RESULTS.csv").open(encoding="utf-8"))); figure,axis=plt.subplots(figsize=(12,5)); for_plot=[r for r in residual if r.get("mean_mse")]; axis.scatter([r["candidate"] for r in for_plot],[float(r["mean_mse"]) for r in for_plot],s=15); axis.tick_params(axis="x",rotation=30); axis.set_ylabel("OOF MSE"); figure.tight_layout(); figure.savefig(root/"13_residual_ar_ablation.png",dpi=180); plt.close(figure)

        figure,axis=plt.subplots(figsize=(8,5)); main=[self._leaderboard_row("FULL",self._model_payloads()["FULL-UOI"])["pooled_RMSE"]]; refined=[]
        targets=[];pred=[]
        for direction in DIRECTIONS:
            art=self._artifact(direction); ref=self._artifact(direction,"refine"); m=art["evaluation_mask"]; targets.append(art["target_z"][m]); pred.append(ref["prediction_full"][m])
        refined=[metrics(np.concatenate(targets),np.concatenate(pred))["RMSE"]]; axis.bar(["Main","Refined"],[main[0],refined[0]]); axis.set_ylabel("Pooled RMSE"); figure.tight_layout(); figure.savefig(root/"14_mesh_refinement_sensitivity.png",dpi=180); plt.close(figure)

        figure,axis=plt.subplots(figsize=(10,5)); channel_rows=list(csv.DictReader((self.results/"CHANNEL_SURFACE_AUDIT.csv").open(encoding="utf-8")))
        for direction in DIRECTIONS:
            subset=[r for r in channel_rows if r["direction"]==direction]; axis.plot([r["channel"] for r in subset],[float(r["loo_mse_delta"]) for r in subset],"o-",label=direction)
        axis.axhline(0,color="black"); axis.tick_params(axis="x",rotation=25); axis.set_ylabel("LOO MSE delta"); axis.legend(); figure.tight_layout(); figure.savefig(root/"15_channel_leave_one_out.png",dpi=180); plt.close(figure)

        figure,axis=plt.subplots(figsize=(10,5)); axis.axis("off"); decision=json.loads((self.results/"FINAL_DECISION.json").read_text()); axis.text(.02,.95,json.dumps(decision,ensure_ascii=False,indent=2)[:3500],va="top",family="monospace",fontsize=7); figure.tight_layout(); figure.savefig(root/"16_final_decision_summary.png",dpi=180); plt.close(figure)
