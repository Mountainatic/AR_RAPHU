from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .io_data import DIRECTIONS, atomic_json, atomic_npz, load_direction, load_protocol, metrics, sha256_file
from .runner import _load_pickle, _save_pickle, _write_csv
from .v2_1_packaging import build_v2_1_bundle
from .v2_1_selection import LocalPairedProfile
from .v2_runner import V2Experiment, _fit_full_worker


class V21Experiment(V2Experiment):
    def __init__(
        self,
        root: Path,
        *,
        v2_bundle: Path,
        v2_protocol_path: Path,
        **kwargs: Any,
    ) -> None:
        base_protocol = root / "configs/frozen_protocol.yaml"
        super().__init__(root, protocol_path=base_protocol, **kwargs)
        self.source_results = root / "results"
        self.v2_bundle = v2_bundle.resolve()
        self.v2_protocol_path = v2_protocol_path.resolve()
        self.config = json.loads(self.v2_protocol_path.read_text(encoding="utf-8"))
        self.protocol_path = self.v2_protocol_path
        self.results = root / "results_v2_1"
        self.checkpoint = self.results / "checkpoints/latest.json"
        self.state = json.loads(self.checkpoint.read_text(encoding="utf-8")) if self.checkpoint.exists() else {"schema": self.config["schema"], "completed": []}
        if self.state.get("schema") != self.config["schema"]:
            raise RuntimeError("V2_1_CHECKPOINT_SCHEMA_MISMATCH")
        for relative in ("", "checkpoints", "diagnostics", "plots", "predictions", "surfaces", "projections", "spectra", "ood", "edf_profiles"):
            (self.results / relative).mkdir(parents=True, exist_ok=True)

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
        bundle = build_v2_1_bundle(
            self.root,
            results_root=self.results,
            protocol_sha256=sha256_file(self.protocol_path),
            shared_sha256=sha256_file(self.shared_bundle),
            cpu_sha256=sha256_file(self.cpu_bundle),
            gpu_sha256=sha256_file(self.gpu_bundle),
            v1_sha256=sha256_file(self.v1_bundle),
            v2_sha256=sha256_file(self.v2_bundle),
            selection_status=self.state["E9"]["decision"]["selection_status"],
            estimator_status=self.state["E9"]["decision"]["estimator_status"],
        )
        print(f"FINAL_ZIP={bundle['zip']}")
        print(f"FINAL_SHA256={bundle['sha256']}")
        print(f"ZIP_SIZE={bundle['size']}")
        print(f"MANIFEST_FILE_COUNT={bundle['manifest_file_count']}")
        print(f"V2_RESULTS_SHA256={sha256_file(self.v2_bundle)}")
        print(f"PROTOCOL_SHA256={sha256_file(self.protocol_path)}")
        print(f"SELECTION_STATUS={self.state['E9']['decision']['selection_status']}")
        print(f"ESTIMATOR_STATUS={self.state['E9']['decision']['estimator_status']}")
        print("VALIDATION_STATUS=PASS")
        return 0

    def stage_e0(self) -> dict[str, Any]:
        if sha256_file(self.v2_bundle) != self.config["v2_results_sha256"]:
            raise RuntimeError("V2_RESULTS_HASH_MISMATCH")
        payload = super().stage_e0()
        source_profile = self.source_results / "CONTINUOUS_EDF_PROFILE.csv"
        source_minima = self.source_results / "CONTINUOUS_EDF_MINIMA.json"
        if not source_profile.exists() or not source_minima.exists():
            raise RuntimeError("V2_PROFILE_CACHE_MISSING")
        with source_profile.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        import_payload = {
            "status": "PASS", "v2_results_sha256": sha256_file(self.v2_bundle),
            "profile_rows": len(rows),
            "profile_rows_by_direction": {direction: sum(row["direction"] == direction for row in rows) for direction in DIRECTIONS},
            "source_commit": self.config["source_commit"],
            "old_surfaces_reuse_forbidden_if_selection_changes": True,
        }
        atomic_json(self.results / "V2_PROFILE_CACHE_IMPORT.json", import_payload)
        (self.results / "PRECHECK_REPORT.md").write_text(
            "# V2.1 PRECHECK\n\n`STATUS=PASS`\n\n"
            f"- V2 results SHA256: `{sha256_file(self.v2_bundle)}`.\n"
            f"- Imported V2 profile rows: `{len(rows)}`.\n"
            "- Model, centered coordinate, basis, penalty, folds, purge and baselines remain frozen.\n"
            "- Only basin resolution and paired-block one-SE are changed.\n",
            encoding="utf-8",
        )
        return {**payload, "v2_profile_import": import_payload}

    def _v2_profile_rows(self, direction: str) -> list[dict[str, float]]:
        with (self.source_results / "CONTINUOUS_EDF_PROFILE.csv").open(encoding="utf-8") as stream:
            return [row for row in csv.DictReader(stream) if row["direction"] == direction]

    def stage_e1_e2(self) -> dict[str, Any]:
        shared_root, _, _ = self._roots()
        protocol = load_protocol(shared_root)
        selection_output: dict[str, Any] = {}
        basin_rows: list[dict[str, Any]] = []
        far_rows: list[dict[str, Any]] = []
        interpolation_rows: list[dict[str, Any]] = []
        paired_rows: list[dict[str, Any]] = []
        sensitivity_rows: list[dict[str, Any]] = []
        import_details: dict[str, Any] = {}
        selected_by_direction: dict[str, float] = {}
        old_selection = json.loads((self.source_results / "CONTINUOUS_ONE_SE_SELECTION.json").read_text(encoding="utf-8"))
        common = json.loads((self.source_results / "COMMON_EDF_INTERVAL.json").read_text(encoding="utf-8"))
        for name in ("FOLD_EDF_MAPS.csv", "COMMON_EDF_INTERVAL.json", "CENTERED_COORDINATE_AUDIT.json"):
            source = self.source_results / name
            if source.exists():
                shutil.copy2(source, self.results / name)

        for direction in DIRECTIONS:
            data = load_direction(shared_root, direction)
            prepared_payloads = [_load_pickle(self._fold_prepared_path(direction, fold)) for fold in range(len(protocol["inner_folds"]))]
            initial = self._v2_profile_rows(direction)
            profile = LocalPairedProfile(
                [item["prepared"].edf_map for item in prepared_payloads],
                [data.train["target_z"][item["validation"]] for item in prepared_payloads],
                initial,
                lower=float(common[direction]["lower_df"]), upper=float(common[direction]["upper_df"]),
                max_evaluations=int(self.config["continuous_edf_v2_1"]["max_actual_profile_evaluations"]),
                d_tolerance=float(self.config["continuous_edf_v2_1"]["local_d_tolerance"]),
                inversion_tolerance=float(self.config["continuous_edf"]["inversion_relative_tolerance"]),
                bootstrap_replicates=int(self.config["continuous_edf_v2_1"]["paired_one_se"]["bootstrap_replicates"]),
                bootstrap_seed=int(self.config["continuous_edf_v2_1"]["paired_one_se"]["seed"]),
            )
            discovery = profile.discover_and_refine()
            primary_rows = int(round(self.config["continuous_edf_v2_1"]["paired_one_se"]["primary_block_min"] * 60 / self.config["cadence_sec"]))
            sensitivity_blocks = tuple(int(round(value * 60 / self.config["cadence_sec"])) for value in self.config["continuous_edf_v2_1"]["paired_one_se"]["sensitivity_block_min"])
            paired = profile.paired_one_se(float(discovery["d_min"]), primary_block_rows=primary_rows, sensitivity_block_rows=sensitivity_blocks)
            selection_resolved = bool(discovery["global_basin_discovery_pass"] and discovery["local_minimum_resolved"] and paired["paired_one_se_boundary_resolved"] and not discovery["upper_bound_hit"])
            selection = {
                "global_basin_discovery": discovery["global_basin_discovery"],
                "global_basin_discovery_pass": discovery["global_basin_discovery_pass"],
                "local_minimum_resolved": discovery["local_minimum_resolved"],
                "d_min": discovery["d_min"],
                "minimum_bracket": discovery["minimum_bracket"],
                "paired_one_se_boundary_resolved": paired["paired_one_se_boundary_resolved"],
                "d_paired_1se": paired["d_paired_1se"],
                "paired_delta": paired["paired_delta_at_selection"],
                "paired_se": paired["paired_se_at_selection"],
                "paired_g": paired["paired_g_at_selection"],
                "root_bracket": paired["root_bracket"],
                "upper_bound_hit": discovery["upper_bound_hit"],
                "far_field_interpolation_resolved": discovery["far_field_interpolation_resolved"],
                "far_field_required_for_selection": discovery["far_field_required_for_selection"],
                "selection_resolved": selection_resolved,
                "paired_one_se_hits_lower_bound": paired["paired_one_se_hits_lower_bound"],
                "profile_evaluations": discovery["profile_evaluations"],
                "new_profile_evaluations": discovery["new_profile_evaluations"],
                "v2_d_1se": old_selection[direction]["d_1se"],
                "selection_change": float(paired["d_paired_1se"] - old_selection[direction]["d_1se"]),
                "common_resamples_across_d": paired["common_resamples_across_d"],
                "fold_order_hashes": paired["fold_order_hashes"],
            }
            selection_output[direction] = selection
            selected_by_direction[direction] = float(paired["d_paired_1se"])
            import_details[direction] = {"imported_profile_rows": profile.imported_count, "new_profile_rows": profile.new_evaluations}
            for candidate in discovery["candidate_basins"]:
                basin_rows.append({"direction": direction, **candidate})
            for row in discovery["far_field"]:
                far_rows.append({"direction": direction, **row})
            interpolation_rows.append({
                "direction": direction,
                "far_field_interpolation_resolved": discovery["far_field_interpolation_resolved"],
                "far_field_required_for_selection": discovery["far_field_required_for_selection"],
                "legacy_max_quadratic_interpolation_error": json.loads((self.source_results / "CONTINUOUS_EDF_MINIMA.json").read_text())[direction]["max_quadratic_interpolation_error"],
                "used_as_selection_gate": False,
            })
            for row in paired["profile"]:
                paired_rows.append({"direction": direction, "block_min": self.config["continuous_edf_v2_1"]["paired_one_se"]["primary_block_min"], **row})
            for row in paired["sensitivity"]:
                sensitivity_rows.append({"direction": direction, "block_min": row["block_rows"] * self.config["cadence_sec"] / 60.0, **row})
            print(f"V2_1_SELECTION={direction}:d_min={discovery['d_min']}:d_p1se={paired['d_paired_1se']}:resolved={selection_resolved}", flush=True)

            for fold, item in enumerate(prepared_payloads):
                prediction, lam, attained = item["prepared"].edf_map.predict_at_df(selected_by_direction[direction], float(self.config["continuous_edf"]["inversion_relative_tolerance"]))
                _save_pickle(self._artifact_path(direction, f"fold{fold}"), {
                    "direction": direction, "fold": fold, "validation": item["validation"],
                    "prediction_full": prediction, "selected_lambda": lam, "effective_df": attained, "kkt_residual": 0.0,
                })

        atomic_json(self.results / "V2_PROFILE_CACHE_IMPORT.json", {"status": "PASS", "v2_results_sha256": sha256_file(self.v2_bundle), "directions": import_details})
        _write_csv(self.results / "V2_1_BASIN_DISCOVERY.csv", basin_rows)
        atomic_json(self.results / "V2_1_LOCAL_MINIMA.json", {direction: {key: selection_output[direction][key] for key in ("global_basin_discovery", "local_minimum_resolved", "d_min", "minimum_bracket", "upper_bound_hit")} for direction in DIRECTIONS})
        _write_csv(self.results / "V2_1_PAIRED_ONE_SE_PROFILE.csv", paired_rows)
        atomic_json(self.results / "V2_1_PAIRED_ONE_SE_SELECTION.json", selection_output)
        _write_csv(self.results / "V2_1_BLOCK_SENSITIVITY.csv", sensitivity_rows)
        _write_csv(self.results / "diagnostics/far_field_pruning.csv", far_rows)
        _write_csv(self.results / "diagnostics/global_interpolation_diagnostic.csv", interpolation_rows)

        fit_tasks: list[dict[str, Any]] = []
        for direction in DIRECTIONS:
            changed = abs(selection_output[direction]["selection_change"]) > self.config["continuous_edf_v2_1"]["local_d_tolerance"]
            if changed:
                for refine, kind in ((False, "main"), (True, "refine")):
                    fit_tasks.append({
                        "shared_root": str(shared_root), "direction": direction, "config": self.config,
                        "selected_edf": selected_by_direction[direction], "selection_resolved": selection_output[direction]["selection_resolved"],
                        "refine": refine, "artifact_path": str(self._artifact_path(direction, kind)),
                    })
        if fit_tasks:
            with ProcessPoolExecutor(max_workers=min(4, self.n_jobs, len(fit_tasks))) as pool:
                futures = [pool.submit(_fit_full_worker, payload) for payload in fit_tasks]
                for future in as_completed(futures):
                    result = future.result()
                    print(f"V2_1_REFIT_DONE={result['direction']}:{result['kind']}", flush=True)

        diagnostic_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        design_rows: list[dict[str, Any]] = []
        for direction in DIRECTIONS:
            artifact = self._artifact(direction)
            data = load_direction(shared_root, direction)
            if abs(artifact["effective_df"] - selected_by_direction[direction]) > 1e-6:
                raise RuntimeError("OLD_V2_SURFACE_REUSED_AFTER_SELECTION_CHANGE")
            if artifact["kkt_residual"] > self.config["kkt_tolerance"] or artifact["reconstruction_error"] > self.config["reconstruction_tolerance"]:
                raise RuntimeError("V2_1_NUMERICAL_CERTIFICATION_FAILED")
            for row in artifact["gcv_curve"]:
                diagnostic_rows.append({"direction": direction, **row})
            current = {
                "direction": direction, "selected_edf": artifact["effective_df"], "derived_lambda_full": artifact["selected_lambda"],
                "kkt_residual": artifact["kkt_residual"], "condition_number": artifact["condition_number"],
                "selection_resolved": selection_output[direction]["selection_resolved"],
                "global_basin_discovery_pass": selection_output[direction]["global_basin_discovery_pass"],
                "local_minimum_resolved": selection_output[direction]["local_minimum_resolved"],
                "paired_one_se_boundary_resolved": selection_output[direction]["paired_one_se_boundary_resolved"],
                "upper_bound_hit": selection_output[direction]["upper_bound_hit"],
                "coefficient_count": len(artifact["coefficient"]), "coefficient_sha256": artifact["coefficient_sha256"],
                "prediction_sha256": artifact["prediction_sha256"],
            }
            design_rows.append(current)
            mask = artifact["evaluation_mask"]
            for model, key in (("CENTERED-R1-LIN-DERIVED", "prediction_rank1"), ("CENTERED-LIN-UOI", "prediction_linear"), ("CENTERED-FULL-UOI", "prediction_full")):
                metric_rows.append({"direction": direction, "model": model, **metrics(artifact["target_z"][mask], artifact[key][mask])})
                atomic_npz(self.results / f"predictions/{model}/{direction}.npz", sample_id=artifact["sample_id"], target_z=artifact["target_z"], evaluation_mask=mask, prediction=artifact[key])
            for channel, surface in artifact["surfaces"].items():
                atomic_npz(self.results / f"surfaces/{direction}__{channel}.npz", **surface)
        _write_csv(self.results / "DIAGNOSTIC_GCV_REML_LCURVE.csv", diagnostic_rows)
        _write_csv(self.results / "CENTERED_FULL_URYSOHN_METRICS.csv", metric_rows)
        atomic_json(self.results / "V2_1_SELECTION_DECISION.json", selection_output)
        return {"status": "PASS", "directions": design_rows, "selection": selection_output}

    def stage_e9(self) -> dict[str, Any]:
        # The inherited V2 report writer reads this legacy filename.  Supply a
        # compatibility view of the already-frozen V2.1 selection; it is not a
        # second selection step and is replaced by the V2.1 report below.
        compatibility_selection = {
            direction: {
                "d_min": self.state["E1_E2"]["selection"][direction]["d_min"],
                "d_1se": self.state["E1_E2"]["selection"][direction]["d_paired_1se"],
            }
            for direction in DIRECTIONS
        }
        atomic_json(self.results / "CONTINUOUS_ONE_SE_SELECTION.json", compatibility_selection)
        payload = super().stage_e9()
        decision = json.loads((self.results / "FINAL_DECISION.json").read_text(encoding="utf-8"))
        directions = self.state["E1_E2"]["directions"]
        selection_resolved = all(
            row["global_basin_discovery_pass"] and row["local_minimum_resolved"]
            and row["paired_one_se_boundary_resolved"] and not row["upper_bound_hit"]
            for row in directions
        )
        if selection_resolved:
            selection_status = "SELECTION_RESOLVED_V2_1"
            estimator_status = "ESTIMATOR_STABLE_V2_1" if decision["numerical_certification"]["kkt"] and decision["numerical_certification"]["mesh"] == "BASIS_RESOLUTION_STABLE" else "ESTIMATOR_SELECTION_UNRESOLVED"
        else:
            local = all(row["local_minimum_resolved"] for row in directions)
            paired = all(row["paired_one_se_boundary_resolved"] for row in directions)
            selection_status = "LOCAL_MINIMUM_RESOLVED_PAIRED_ONE_SE_UNRESOLVED" if local and not paired else "BASIN_DISCOVERY_UNRESOLVED"
            estimator_status = "ESTIMATOR_SELECTION_UNRESOLVED"
        model_registration = decision["registration"] if estimator_status == "ESTIMATOR_STABLE_V2_1" else "ESTIMATOR_UNRESOLVED"
        decision.update({
            "selection_status": selection_status,
            "estimator_status": estimator_status,
            "registration": model_registration,
            "v2_1_selection": self.state["E1_E2"]["selection"],
            "selection_method": "log_excess_basin_local_refinement_paired_block_one_se",
            "far_field_interpolation_is_selection_gate": False,
        })
        atomic_json(self.results / "FINAL_DECISION.json", decision)
        atomic_json(self.results / "V2_1_SELECTION_DECISION.json", self.state["E1_E2"]["selection"])
        self._write_v2_1_report(decision)
        self._write_v2_decision_plot(decision)
        return {"status": "PASS", "registration": model_registration, "decision": decision}

    def _write_v2_1_report(self, decision: dict[str, Any]) -> None:
        input_rows = {row["model"]: row for row in csv.DictReader((self.results / "FINAL_INPUT_LEADERBOARD.csv").open(encoding="utf-8"))}
        dynamic_rows = {row["model"]: row for row in csv.DictReader((self.results / "FINAL_DYNAMIC_LEADERBOARD.csv").open(encoding="utf-8"))}
        table = {**input_rows, **dynamic_rows}
        lines = [
            "# Centered OD-FUOI Local Profile + Paired One-SE V2.1 Final Report", "",
            f"- Selection status: `{decision['selection_status']}`",
            f"- Estimator status: `{decision['estimator_status']}`",
            f"- Model registration: `{decision['registration']}`",
            f"- Nonlinear increment: `{decision['nonlinear_increment_status']}`",
            f"- Residual: `{decision['residual_status']}`", "",
            "## Local minimum and paired one-SE", "",
        ]
        for direction in DIRECTIONS:
            row = decision["v2_1_selection"][direction]
            lines.append(
                f"- {direction}: d_min=`{row['d_min']:.6f}`, d_P1SE=`{row['d_paired_1se']:.6f}`, "
                f"V2 d_1SE=`{row['v2_d_1se']:.6f}`, change=`{row['selection_change']:.6f}`, "
                f"paired delta=`{row['paired_delta']:.9g}`, paired SE=`{row['paired_se']:.9g}`."
            )
        lines.extend(["", "## Frozen L6 outer results", "", "| Model | Sheet1→Sheet2 RMSE | Sheet2→Sheet1 RMSE | Pooled RMSE |", "|---|---:|---:|---:|"])
        for model in ("Persistence", "old K-only", "NLinear-U", "R1-LIN-DERIVED", "LIN-UOI", "FULL-UOI", "Joint-K+AR", "FULL-UOI-PSAR"):
            row = table[model]
            lines.append(f"| {model} | {float(row['sheet1_to_sheet2_RMSE']):.9f} | {float(row['sheet2_to_sheet1_RMSE']):.9f} | {float(row['pooled_RMSE']):.9f} |")
        lines.extend([
            "", "## Interpretation", "",
            "V2.1 did not change the centered Full-Urysohn model. It replaced the whole-domain interpolation gate with deterministic log-excess basin discovery plus local minimum certification, and replaced absolute fold-MSE one-SE with paired 40-minute block one-SE on the same validation samples. Far-field interpolation remains diagnostic and is not a selection gate. Decimal EDF values are effective smoothing coordinates, not physical degrees of freedom.",
        ])
        text = "\n".join(lines) + "\n"
        (self.results / "FINAL_REPORT.md").write_text(text, encoding="utf-8")
        (self.results / "V2_1_FINAL_REPORT.md").write_text(text, encoding="utf-8")
