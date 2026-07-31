from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .centered import centered_increment, support_audit
from .edf import ContinuousProfile
from .io_data import (
    DIRECTIONS,
    atomic_json,
    atomic_npz,
    inner_folds,
    load_cpu_prediction,
    load_direction,
    load_protocol,
    metrics,
    moving_block_bootstrap,
    sha256_array,
    sha256_file,
)
from .packaging import build_bundle
from .runner import Experiment, _load_pickle, _save_pickle, _write_csv
from .v2_model import diagnostic_smoothing_curve, fit_prepared, predict_from_artifact, prepare_model


def _prepare_fold_worker(payload: dict[str, Any]) -> dict[str, Any]:
    data = load_direction(Path(payload["shared_root"]), payload["direction"])
    training = np.asarray(payload["training"], dtype=np.int64)
    validation = np.asarray(payload["validation"], dtype=np.int64)
    config = payload["config"]
    prepared = prepare_model(
        data.train["sequence_u"][training],
        data.train["target_z"][training],
        data.train["sequence_u"][validation],
        config=config,
        lag_count=int(config["lag_basis"]["number_of_basis"]),
        amplitude_quantiles=list(config["amplitude_basis"]["quantiles"]),
    )
    interval = prepared.edf_map.stable_interval(
        condition_epsilon_limit=float(config["continuous_edf"]["condition_epsilon_limit"]),
        lower_excess=float(config["continuous_edf"]["lower_excess"]),
    )
    artifact_path = Path(payload["artifact_path"])
    _save_pickle(artifact_path, {
        "prepared": prepared,
        "training": training,
        "validation": validation,
        "interval": interval,
        "direction": payload["direction"],
        "fold": int(payload["fold"]),
    })
    return {"direction": payload["direction"], "fold": int(payload["fold"]), "interval": interval}


def _fit_full_worker(payload: dict[str, Any]) -> dict[str, Any]:
    data = load_direction(Path(payload["shared_root"]), payload["direction"])
    config = payload["config"]
    refine = bool(payload["refine"])
    lag_count = int(config["mesh_refinement"]["lag_basis"] if refine else config["lag_basis"]["number_of_basis"])
    quantiles = list(config["mesh_refinement"]["amplitude_quantiles"] if refine else config["amplitude_basis"]["quantiles"])
    prepared = prepare_model(
        data.train["sequence_u"], data.train["target_z"], data.test["sequence_u"],
        config=config, lag_count=lag_count, amplitude_quantiles=quantiles,
    )
    artifact = fit_prepared(prepared, float(payload["selected_edf"]), config=config, include_surfaces=not refine)
    artifact["gcv_curve"] = diagnostic_smoothing_curve(prepared.edf_map, int(config["diagnostic_smoothing"]["points"]))
    artifact.update({
        "direction": payload["direction"],
        "sample_id": data.test["sample_id"].astype("U"),
        "target_z": data.test["target_z"].astype(np.float64),
        "evaluation_mask": data.test["evaluation_mask"].astype(bool),
        "train_sample_id": data.train["sample_id"].astype("U"),
        "train_target_z": data.train["target_z"].astype(np.float64),
        "selection_resolved": bool(payload["selection_resolved"]),
    })
    common = np.ones(len(data.test["target_z"]), dtype=bool)
    for channel in range(prepared.train_delta.shape[2]):
        lower = float(np.min(prepared.train_delta[:, :, channel]))
        upper = float(np.max(prepared.train_delta[:, :, channel]))
        common &= np.all((prepared.predict_delta[:, :, channel] >= lower) & (prepared.predict_delta[:, :, channel] <= upper), axis=1)
    artifact["common_support_mask"] = common
    _save_pickle(Path(payload["artifact_path"]), artifact)
    return {
        "direction": payload["direction"], "kind": "refine" if refine else "main",
        "selected_lambda": artifact["selected_lambda"], "effective_df": artifact["effective_df"],
        "kkt_residual": artifact["kkt_residual"], "condition_number": artifact["condition_number"],
    }


class V2Experiment(Experiment):
    def __init__(self, *args: Any, v1_bundle: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.v1_bundle = v1_bundle.resolve()
        if self.state.get("schema") != self.config["schema"]:
            raise RuntimeError(f"CHECKPOINT_SCHEMA_MISMATCH:{self.state.get('schema')}:{self.config['schema']}")
        for name in ("edf_profiles", "centered_increment", "projections"):
            (self.results / name).mkdir(parents=True, exist_ok=True)

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
            v1_sha256=sha256_file(self.v1_bundle),
        )
        print(f"FINAL_ZIP={bundle['zip']}")
        print(f"FINAL_SHA256={bundle['sha256']}")
        print(f"ZIP_SIZE={bundle['size']}")
        print(f"MANIFEST_FILE_COUNT={bundle['manifest_file_count']}")
        print(f"PROTOCOL_SHA256={sha256_file(self.protocol_path)}")
        print(f"SHARED_DATASET_SHA256={sha256_file(self.shared_bundle)}")
        print(f"V1_RESULTS_SHA256={sha256_file(self.v1_bundle)}")
        print(f"CPU_BASELINE_BUNDLE_SHA256={sha256_file(self.cpu_bundle)}")
        print(f"GPU_BASELINE_BUNDLE_SHA256={sha256_file(self.gpu_bundle)}")
        print("VALIDATION_STATUS=PASS")
        return 0

    def stage_e0(self) -> dict[str, Any]:
        if sha256_file(self.v1_bundle) != self.config["v1_results_sha256"]:
            raise RuntimeError("V1_RESULTS_HASH_MISMATCH")
        payload = super().stage_e0()
        nlinear = payload["models"]["NLinear-U"]["ensemble_pooled"]["RMSE"]
        if abs(float(nlinear) - float(self.config["frozen_baseline_rmse"]["NLinear-U"])) > 1e-6:
            raise RuntimeError(f"NLINEAR_BASELINE_MISMATCH:{nlinear}")
        payload["v1_results_sha256"] = sha256_file(self.v1_bundle)
        payload["v1_registration"] = "ABSOLUTE_AMPLITUDE_POINTWISE_GCV_REJECTED"
        atomic_json(self.results / "preflight/precheck.json", payload)
        (self.results / "PRECHECK_REPORT.md").write_text(
            "# CENTERED OD-FUOI V2 PRECHECK\n\n`STATUS=PASS`\n\n"
            f"- V1 registered as `ABSOLUTE_AMPLITUDE_POINTWISE_GCV_REJECTED`.\n"
            f"- Shared bundle: `{self.config['shared_dataset_sha256']}`.\n"
            f"- V1 results: `{self.config['v1_results_sha256']}`.\n"
            f"- Frozen NLinear-U pooled RMSE reproduced: `{nlinear:.9f}`.\n"
            "- Sample IDs, targets, masks, cadence, history, horizon and purge are unchanged.\n"
            "- Raw Excel is absent. GCV/REML/L-curve are diagnostics only.\n",
            encoding="utf-8",
        )
        return payload

    def _fold_prepared_path(self, direction: str, fold: int) -> Path:
        return self.work / "edf_maps" / f"{direction}__fold{fold}.pkl"

    def stage_e1_e2(self) -> dict[str, Any]:
        shared_root, _, _ = self._roots()
        protocol = load_protocol(shared_root)
        purge_raw = int(round(float(self.config["purge_min"]) * 60.0 / float(protocol["sample_period_sec"])))
        tasks: list[dict[str, Any]] = []
        fold_indices: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for direction in DIRECTIONS:
            data = load_direction(shared_root, direction)
            folds = inner_folds(data.train["origin_raw_index"], protocol["inner_folds"], purge_raw_samples=purge_raw)
            fold_indices[direction] = folds
            for fold, (training, validation) in enumerate(folds):
                path = self._fold_prepared_path(direction, fold)
                if not path.exists():
                    tasks.append({
                        "shared_root": str(shared_root), "direction": direction, "fold": fold,
                        "training": training, "validation": validation, "config": self.config,
                        "artifact_path": str(path),
                    })
        if tasks:
            with ProcessPoolExecutor(max_workers=min(self.n_jobs, len(tasks))) as pool:
                futures = [pool.submit(_prepare_fold_worker, payload) for payload in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    print(f"EDF_MAP_DONE={result['direction']}:fold{result['fold']}", flush=True)

        edf_map_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        common_payload: dict[str, Any] = {}
        minima_payload: dict[str, Any] = {}
        selection_payload: dict[str, Any] = {}
        direction_selection: dict[str, dict[str, Any]] = {}
        coordinate_payload: dict[str, Any] = {"coordinate": "delta_j,t(l)=x_j,t-l-x_j,t", "lag0_exact_zero": True, "directions": {}}
        for direction in DIRECTIONS:
            data = load_direction(shared_root, direction)
            train_delta = centered_increment(data.train["sequence_u"])
            test_delta = centered_increment(data.test["sequence_u"])
            centered_rows = support_audit(train_delta, test_delta)
            absolute_rows = support_audit(np.asarray(data.train["sequence_u"], dtype=np.float64), np.asarray(data.test["sequence_u"], dtype=np.float64))
            for index, row in enumerate(centered_rows):
                row["channel"] = self.config["controls"][index]
                row["absolute_extension_ratio"] = absolute_rows[index]["extension_ratio"]
                row["centered_extension_ratio"] = row["extension_ratio"]
            coordinate_payload["directions"][direction] = {
                "channels": centered_rows,
                "lag0_max_abs": float(np.max(np.abs(train_delta[:, 0, :]))),
                "no_future_input": bool(np.all(data.test["origin_raw_index"] < data.test["future_left_raw_index"])),
            }
            prepared_payloads = [_load_pickle(self._fold_prepared_path(direction, fold)) for fold in range(len(fold_indices[direction]))]
            maps = [item["prepared"].edf_map for item in prepared_payloads]
            intervals = [item["interval"] for item in prepared_payloads]
            lower = max(float(item["lower_df"]) for item in intervals)
            upper = min(float(item["upper_df"]) for item in intervals)
            if not lower < upper:
                raise RuntimeError("NO_COMMON_STABLE_EDF_INTERVAL")
            common_payload[direction] = {"lower_df": lower, "upper_df": upper, "fold_intervals": intervals}
            for fold, item in enumerate(prepared_payloads):
                values = item["prepared"].edf_map.eigenvalues
                for index, value in enumerate(values):
                    edf_map_rows.append({
                        "direction": direction, "fold": fold, "eigen_index": index,
                        "generalized_eigenvalue": float(value), "null_space_df": 1.0,
                        "stable_upper_df": intervals[fold]["upper_df"], "stable_lower_df": intervals[fold]["lower_df"],
                    })

            direction_profile_path = self.results / f"edf_profiles/{direction}.csv"
            def checkpoint(rows: list[dict[str, float]], *, current_direction: str = direction) -> None:
                _write_csv(direction_profile_path, [{"direction": current_direction, **row} for row in rows])

            profile = ContinuousProfile(
                maps,
                [data.train["target_z"][item["validation"]] for item in prepared_payloads],
                [len(item["validation"]) for item in prepared_payloads],
                lower=lower, upper=upper,
                max_evaluations=int(self.config["continuous_edf"]["max_actual_profile_evaluations"]),
                d_tolerance=float(self.config["continuous_edf"]["d_absolute_tolerance"]),
                interpolation_tolerance=float(self.config["continuous_edf"]["profile_interpolation_relative_tolerance"]),
                inversion_tolerance=float(self.config["continuous_edf"]["inversion_relative_tolerance"]),
                checkpoint=checkpoint,
            )
            resolution = profile.resolve()
            one_se = profile.one_se(resolution)
            for row in resolution["evaluations"]:
                profile_rows.append({"direction": direction, **row})
            minima_payload[direction] = {
                "resolved": resolution["resolved"], "upper_bound_hit": resolution["upper_bound_hit"],
                "d_min": resolution["d_min"], "minimum_mse": resolution["minimum_mse"],
                "minimum_se": resolution["minimum_se"], "minima": resolution["minima"],
                "evaluation_count": resolution["evaluation_count"],
                "max_quadratic_interpolation_error": resolution["max_quadratic_interpolation_error"],
            }
            selection_payload[direction] = one_se
            direction_selection[direction] = {**one_se, "resolved": bool(resolution["resolved"]), "common_lower": lower, "common_upper": upper}
            for fold, item in enumerate(prepared_payloads):
                prediction, lam, attained = item["prepared"].edf_map.predict_at_df(float(one_se["d_1se"]), float(self.config["continuous_edf"]["inversion_relative_tolerance"]))
                _save_pickle(self._artifact_path(direction, f"fold{fold}"), {
                    "direction": direction, "fold": fold, "validation": item["validation"],
                    "prediction_full": prediction, "selected_lambda": lam, "effective_df": attained,
                    "kkt_residual": 0.0,
                })
            print(f"CONTINUOUS_EDF_SELECTED={direction}:d_min={one_se['d_min']}:d_1se={one_se['d_1se']}", flush=True)

        atomic_json(self.results / "CENTERED_COORDINATE_AUDIT.json", coordinate_payload)
        _write_csv(self.results / "FOLD_EDF_MAPS.csv", edf_map_rows)
        atomic_json(self.results / "COMMON_EDF_INTERVAL.json", common_payload)
        _write_csv(self.results / "CONTINUOUS_EDF_PROFILE.csv", profile_rows)
        atomic_json(self.results / "CONTINUOUS_EDF_MINIMA.json", minima_payload)
        atomic_json(self.results / "CONTINUOUS_ONE_SE_SELECTION.json", selection_payload)

        fit_tasks: list[dict[str, Any]] = []
        for direction in DIRECTIONS:
            for refine, kind in ((False, "main"), (True, "refine")):
                path = self._artifact_path(direction, kind)
                if not path.exists():
                    fit_tasks.append({
                        "shared_root": str(shared_root), "direction": direction, "config": self.config,
                        "selected_edf": direction_selection[direction]["d_1se"], "selection_resolved": direction_selection[direction]["resolved"],
                        "refine": refine, "artifact_path": str(path),
                    })
        if fit_tasks:
            with ProcessPoolExecutor(max_workers=min(4, self.n_jobs, len(fit_tasks))) as pool:
                futures = [pool.submit(_fit_full_worker, payload) for payload in fit_tasks]
                for future in as_completed(futures):
                    result = future.result()
                    print(f"FULL_REFIT_DONE={result['direction']}:{result['kind']}", flush=True)

        diagnostic_rows: list[dict[str, Any]] = []
        metrics_rows: list[dict[str, Any]] = []
        design_rows: list[dict[str, Any]] = []
        for direction in DIRECTIONS:
            artifact = self._artifact(direction)
            data = load_direction(shared_root, direction)
            if artifact["kkt_residual"] > self.config["kkt_tolerance"]:
                raise RuntimeError(f"KKT_FAILED:{direction}:{artifact['kkt_residual']}")
            if artifact["reconstruction_error"] > self.config["reconstruction_tolerance"]:
                raise RuntimeError(f"PROJECTION_RECONSTRUCTION_FAILED:{direction}:{artifact['reconstruction_error']}")
            if artifact["constant_constraint_residual"] > self.config["orthogonality_tolerance"] or artifact["linear_constraint_residual"] > self.config["orthogonality_tolerance"]:
                raise RuntimeError(f"ORTHOGONALITY_FAILED:{direction}:{artifact['constant_constraint_residual']}:{artifact['linear_constraint_residual']}")
            for row in artifact["gcv_curve"]:
                diagnostic_rows.append({"direction": direction, **row})
            design_rows.append({
                "direction": direction, "selected_edf": artifact["effective_df"], "derived_lambda_full": artifact["selected_lambda"],
                "kkt_residual": artifact["kkt_residual"], "condition_number": artifact["condition_number"],
                "selection_resolved": artifact["selection_resolved"], "coefficient_count": len(artifact["coefficient"]),
                "coefficient_sha256": artifact["coefficient_sha256"], "prediction_sha256": artifact["prediction_sha256"],
                "no_future_input": bool(np.all(data.test["origin_raw_index"] < data.test["future_left_raw_index"])),
                "amplitude_support": artifact["amplitude_manifest"],
            })
            mask = artifact["evaluation_mask"]
            for model, key in (("CENTERED-R1-LIN-DERIVED", "prediction_rank1"), ("CENTERED-LIN-UOI", "prediction_linear"), ("CENTERED-FULL-UOI", "prediction_full")):
                metrics_rows.append({"direction": direction, "model": model, **metrics(artifact["target_z"][mask], artifact[key][mask])})
                atomic_npz(self.results / f"predictions/{model}/{direction}.npz", sample_id=artifact["sample_id"], target_z=artifact["target_z"], evaluation_mask=mask, prediction=artifact[key])
            for channel, surface in artifact["surfaces"].items():
                atomic_npz(self.results / f"surfaces/{direction}__{channel}.npz", **surface)
        _write_csv(self.results / "DIAGNOSTIC_GCV_REML_LCURVE.csv", diagnostic_rows)
        _write_csv(self.results / "CENTERED_FULL_URYSOHN_METRICS.csv", metrics_rows)
        atomic_json(self.results / "diagnostics/centered_full_refit.json", {"directions": design_rows})
        return {"status": "PASS", "directions": design_rows, "selection": direction_selection}

    @staticmethod
    def _alias(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def stage_e3(self) -> dict[str, Any]:
        payload = super().stage_e3()
        self._alias(self.results / "AMPLITUDE_PROJECTION_METRICS.csv", self.results / "CENTERED_PROJECTION_METRICS.csv")
        return payload

    def stage_e4(self) -> dict[str, Any]:
        payload = super().stage_e4()
        self._alias(self.results / "NLINEAR_DERIVED_AUDIT.csv", self.results / "CENTERED_NLINEAR_AUDIT.csv")
        return payload

    def stage_e5(self) -> dict[str, Any]:
        shared_root, _, _ = self._roots()
        bootstrap_rows: list[dict[str, Any]] = []
        direction_results: dict[str, Any] = {}
        surface = self._surface_correlations()
        for direction_index, direction in enumerate(DIRECTIONS):
            artifact = self._artifact(direction)
            refine = self._artifact(direction, "refine")
            data = load_direction(shared_root, direction)
            mask = artifact["evaluation_mask"]
            linear_error = artifact["target_z"][mask] - artifact["prediction_linear"][mask]
            full_error = artifact["target_z"][mask] - artifact["prediction_full"][mask]
            observed_gain = float(np.mean(linear_error**2) - np.mean(full_error**2))
            for block in self.config["bootstrap_block_lengths_min"]:
                result = moving_block_bootstrap([linear_error], [full_error], block_rows=int(round(block * 60 / self.config["cadence_sec"])), replicates=self.config["bootstrap_replicates"], seed=self.config["bootstrap_seed"] + direction_index * 1000 + int(block))
                bootstrap_rows.append({"comparison": "CENTERED-LIN-UOI_vs_CENTERED-FULL-UOI", "direction": direction, "block_min": block, **result})
            shift = int(round(self.config["placebo_shift_min"] * 60 / self.config["cadence_sec"]))
            placebo = predict_from_artifact(artifact, data.test["sequence_u"][:-shift], self.config["cadence_sec"])
            placebo_mask = mask[shift:]
            placebo_target = artifact["target_z"][shift:][placebo_mask]
            placebo_gain = float(np.mean((placebo_target - placebo["linear"][placebo_mask])**2) - np.mean((placebo_target - placebo["full"][placebo_mask])**2))
            placebo_destroyed = placebo_gain <= max(0.0, 0.5 * observed_gain)
            squared_delta = linear_error**2 - full_error**2
            common = artifact["common_support_mask"][mask]
            positive_total = float(np.sum(np.maximum(squared_delta, 0.0)))
            ood_fraction = float(np.sum(np.maximum(squared_delta[~common], 0.0))) / max(positive_total, 1e-30)
            main_metric = metrics(artifact["target_z"][mask], artifact["prediction_full"][mask])
            refine_metric = metrics(artifact["target_z"][mask], refine["prediction_full"][mask])
            old = load_cpu_prediction(self._roots()[1], direction, "K-only")
            old_metric = metrics(old["target_z"][mask], old["prediction"][mask])
            same = (main_metric["MSE"] < old_metric["MSE"]) == (refine_metric["MSE"] < old_metric["MSE"])
            mesh_delta = abs(main_metric["RMSE"] - refine_metric["RMSE"]) / max(main_metric["RMSE"], 1e-30)
            direction_results[direction] = {
                "linear_mse": float(np.mean(linear_error**2)), "full_mse": float(np.mean(full_error**2)),
                "nonlinear_mse_gain": observed_gain, "placebo_gain": placebo_gain, "placebo_destroyed": placebo_destroyed,
                "ood_positive_gain_fraction": ood_fraction, "ood_not_primary": ood_fraction <= 0.5,
                "refinement_rmse": refine_metric["RMSE"], "mesh_relative_rmse_delta": mesh_delta,
                "mesh_stable": bool(same and mesh_delta <= 0.05),
            }
        pooled_linear = []
        pooled_full = []
        for direction in DIRECTIONS:
            artifact = self._artifact(direction); mask = artifact["evaluation_mask"]
            pooled_linear.append(artifact["target_z"][mask] - artifact["prediction_linear"][mask])
            pooled_full.append(artifact["target_z"][mask] - artifact["prediction_full"][mask])
        for block in self.config["bootstrap_block_lengths_min"]:
            result = moving_block_bootstrap(pooled_linear, pooled_full, block_rows=int(round(block * 60 / self.config["cadence_sec"])), replicates=self.config["bootstrap_replicates"], seed=self.config["bootstrap_seed"] + 9000 + int(block))
            bootstrap_rows.append({"comparison": "CENTERED-LIN-UOI_vs_CENTERED-FULL-UOI", "direction": "pooled", "block_min": block, **result})
        primary = next(row for row in bootstrap_rows if row["direction"] == "pooled" and row["block_min"] == self.config["bootstrap_primary_block_min"])
        certified = (
            all(row["full_mse"] <= row["linear_mse"] + 1e-15 for row in direction_results.values())
            and primary["positive_probability"] >= 0.95
            and min(abs(value["nonlinear"]) for value in surface.values()) >= 0.6
            and all(row["placebo_destroyed"] and row["ood_not_primary"] for row in direction_results.values())
        )
        payload = {
            "status": "PASS", "nonlinear_increment_status": "NONLINEAR_INCREMENT_CERTIFIED" if certified else "NONLINEAR_INCREMENT_NOT_CERTIFIED",
            "directions": direction_results, "primary_bootstrap": primary, "surface_correlations": surface,
            "mesh_status": "BASIS_RESOLUTION_STABLE" if all(row["mesh_stable"] for row in direction_results.values()) else "BASIS_RESOLUTION_INSUFFICIENT",
        }
        _write_csv(self.results / "CENTERED_NONLINEAR_INCREMENT.csv", bootstrap_rows)
        _write_csv(self.results / "NONLINEAR_INCREMENT_AUDIT.csv", bootstrap_rows)
        atomic_json(self.results / "diagnostics/nonlinear_increment.json", payload)
        return payload

    def stage_e6_e7(self) -> dict[str, Any]:
        payload = super().stage_e6_e7()
        self._alias(self.results / "CHANNEL_SURFACE_AUDIT.csv", self.results / "CENTERED_CHANNEL_AUDIT.csv")
        self._alias(self.results / "C1_EXTENSION_AUDIT.csv", self.results / "CENTERED_C1_OOD_AUDIT.csv")
        return payload

    def stage_e8(self) -> dict[str, Any]:
        payload = super().stage_e8()
        self._alias(self.results / "RESIDUAL_PSAR_RESULTS.csv", self.results / "CENTERED_RESIDUAL_PSAR.csv")
        return payload

    def stage_e9(self) -> dict[str, Any]:
        payload = super().stage_e9()
        decision = json.loads((self.results / "FINAL_DECISION.json").read_text(encoding="utf-8"))
        input_rows = list(__import__("csv").DictReader((self.results / "FINAL_INPUT_LEADERBOARD.csv").open(encoding="utf-8")))
        table = {row["model"]: row for row in input_rows}
        full_rmse = float(table["FULL-UOI"]["pooled_RMSE"])
        nlinear_rmse = float(table["NLinear-U"]["pooled_RMSE"])
        old_rmse = float(table["old K-only"]["pooled_RMSE"])
        estimator_stable = all(bool(item["selection_resolved"]) for item in self.state["E1_E2"]["directions"])
        both_positive = all(float(table["FULL-UOI"][f"{direction}_RMSE"]) < float(table["Persistence"][f"{direction}_RMSE"]) for direction in DIRECTIONS)
        numerical = decision["numerical_certification"]
        certificate = numerical["kkt"] and numerical["c1"] == "PASS" and numerical["mesh"] == "BASIS_RESOLUTION_STABLE"
        comparison_nlinear = decision["comparison_vs_nlinear"]
        if not estimator_stable:
            registration = "ESTIMATOR_UNRESOLVED"
            estimator_status = "SMOOTHING_SELECTION_UNRESOLVED"
        else:
            estimator_status = "CENTERED_ESTIMATOR_STABLE"
            if full_rmse < nlinear_rmse and both_positive and comparison_nlinear["positive_probability"] >= 0.90 and certificate:
                registration = "CENTERED_FULL_URYSOHN_CONFIRMED"
            elif abs(full_rmse - nlinear_rmse) / nlinear_rmse <= 0.01 and both_positive and certificate:
                registration = "CENTERED_URYSOHN_PARETO"
            elif full_rmse < old_rmse:
                registration = "CENTERED_URYSOHN_IMPROVES_OLD_K_ONLY"
            else:
                registration = "CENTERED_FULL_URYSOHN_REJECTED"
        decision["estimator_status"] = estimator_status
        decision["registration"] = registration
        decision["model_status_detail"] = "MODEL_REJECTED_UNDER_STABLE_ESTIMATOR" if estimator_stable and registration == "CENTERED_FULL_URYSOHN_REJECTED" else registration
        decision["coordinate"] = "centered_increment"
        decision["selection_method"] = "continuous_effective_df_blocked_cv_one_se"
        decision["selected_edf_by_direction"] = {direction: self._artifact(direction)["effective_df"] for direction in DIRECTIONS}
        decision["derived_lambda_full_by_direction"] = {direction: self._artifact(direction)["selected_lambda"] for direction in DIRECTIONS}
        decision.pop("gcv_lambda_by_direction", None)
        atomic_json(self.results / "FINAL_DECISION.json", decision)
        self._write_v2_report(decision)
        self._write_v2_decision_plot(decision)
        return {"status": "PASS", "registration": registration, "decision": decision}

    def _write_v2_decision_plot(self, decision: dict[str, Any]) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.axis("off")
        axis.text(.02, .97, json.dumps(decision, ensure_ascii=False, indent=2)[:5000], va="top", family="monospace", fontsize=7)
        figure.tight_layout()
        figure.savefig(self.results / "plots/16_final_decision_summary.png", dpi=180)
        plt.close(figure)

    def _write_v2_report(self, decision: dict[str, Any]) -> None:
        import csv
        input_rows = {row["model"]: row for row in csv.DictReader((self.results / "FINAL_INPUT_LEADERBOARD.csv").open(encoding="utf-8"))}
        dynamic_rows = {row["model"]: row for row in csv.DictReader((self.results / "FINAL_DYNAMIC_LEADERBOARD.csv").open(encoding="utf-8"))}
        table = {**input_rows, **dynamic_rows}
        selection = json.loads((self.results / "CONTINUOUS_ONE_SE_SELECTION.json").read_text(encoding="utf-8"))
        coordinate = json.loads((self.results / "CENTERED_COORDINATE_AUDIT.json").read_text(encoding="utf-8"))
        lines = [
            "# Centered OD-FUOI Continuous-EDF CPU Confirm V2 Final Report", "",
            f"- Estimator status: `{decision['estimator_status']}`",
            f"- Model registration: `{decision['registration']}`",
            f"- NLinear projection: `{decision['nlinear_projection_status']}`",
            f"- Nonlinear increment: `{decision['nonlinear_increment_status']}`",
            f"- Residual: `{decision['residual_status']}`", "",
            "## Continuous EDF selection", "",
        ]
        for direction in DIRECTIONS:
            lines.append(f"- {direction}: d_min=`{selection[direction]['d_min']:.6f}`, d_1SE=`{selection[direction]['d_1se']:.6f}`, lambda_full=`{decision['derived_lambda_full_by_direction'][direction]:.9e}`.")
        lines.extend(["", "## Frozen L6 results", "", "| Model | Sheet1→Sheet2 RMSE | Sheet2→Sheet1 RMSE | Pooled RMSE |", "|---|---:|---:|---:|"])
        for model in ("Persistence", "old K-only", "NLinear-U", "R1-LIN-DERIVED", "LIN-UOI", "FULL-UOI", "Joint-K+AR", "FULL-UOI-PSAR"):
            row = table[model]
            lines.append(f"| {model} | {float(row['sheet1_to_sheet2_RMSE']):.9f} | {float(row['sheet2_to_sheet1_RMSE']):.9f} | {float(row['pooled_RMSE']):.9f} |")
        lines.extend(["", "## Centered-coordinate OOD", ""])
        for direction in DIRECTIONS:
            for row in coordinate["directions"][direction]["channels"]:
                lines.append(f"- {direction}/{row['channel']}: absolute extension `{row['absolute_extension_ratio']:.6%}` → centered extension `{row['centered_extension_ratio']:.6%}`.")
        lines.extend([
            "", "## Interpretation boundary", "",
            "The decimal continuous-EDF coordinate is an effective smoothing complexity, not a physically exact count of degrees of freedom. GCV, REML and L-curve values were retained only as diagnostics and never selected the model. The fitted surfaces remain registered closed-loop input-history response surfaces; two rods do not establish universal open-loop plant causality.",
        ])
        (self.results / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
