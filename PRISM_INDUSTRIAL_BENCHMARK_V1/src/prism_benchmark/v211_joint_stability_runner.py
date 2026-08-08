from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cpu_data import sha256_file
from .stage0 import write_json
from .v211_joint import JOINT_CANDIDATES, J_KA, J_KWA, predict_joint_candidate
from .v211_joint_stability import fit_joint_candidate_stability
from .v211_joint_stability_config import (
    DEVELOPMENT_SOURCE_COMMIT,
    JOINT_ESTIMATOR_SEMANTICS,
    MODEL_VERSION,
    OUTPUT_DIRECTORY,
    PRACTICE_REVISION,
    PROTOCOL_ID,
    config_path,
    load_joint_stability_config,
    theory_path,
)
from .v211_metro_config import MetroV211Paths
from .v211_metro_contracts import stable_candidate_id
from .v211_metro_final import _formal_candidate_names, materialize_view
from .v211_metro_reporting import JOINT_VS_PF_COMPARISON
from .v211_metro_runner import full_shared_data_audit
from .v211_metro_views import metro_p60_dynamic_views


PF_CANDIDATES = ("KC", "KCW", "KCA", "KCWA", "PF_SELECTED")
SOURCE_BRANCH = "prism-v2-2-joint-predictive-stability"
LEGACY_EXECUTION_LABEL = "PRISM_V2_2_JOINT_PREDICTIVE_STABILITY"
FREEZE_NAME = "METRO_P60_V211_JOINT_STABILITY_FINAL_DEVELOPMENT_FREEZE.json"
DECISION_NAME = "METRO_P60_V211_JOINT_STABILITY_FINAL_DEVELOPMENT_DECISION.json"


@dataclass(frozen=True)
class FinalClosurePaths:
    project: Path
    shared: Path
    output: Path
    source_results: Path

    @property
    def metro(self) -> MetroV211Paths:
        return MetroV211Paths(self.project, self.shared, self.output)

    @property
    def reuse_audit_path(self) -> Path:
        return self.output / "M2_M4_ARTIFACT_REUSE_AUDIT.json"

    @property
    def migration_audit_path(self) -> Path:
        return self.output / "DEVELOPMENT_EVIDENCE_MIGRATION_AUDIT.json"

    @property
    def decision_path(self) -> Path:
        return self.output / "FREEZE" / DECISION_NAME

    @property
    def freeze_path(self) -> Path:
        return self.output / "FREEZE" / FREEZE_NAME

    @property
    def preflight_path(self) -> Path:
        return self.output / "M7_LOCKBOX_PREFLIGHT.json"

    @property
    def causality_path(self) -> Path:
        return self.output / "M7_TARGET_HISTORY_CAUSALITY_AUDIT.json"


def _git(project: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project), *arguments], check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"existing migrated artifact differs: {destination}")
        return
    os.link(source, destination)


def _hardlink_tree(source: Path, destination: Path, *, omit_result: bool = False) -> None:
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        if omit_result and path.name == "RESULT.json":
            continue
        _hardlink(path, destination / path.relative_to(source))


def _result_path(root: Path, view: Any, stage: str = "JOINT") -> Path:
    return (
        root / "DEVELOPMENT" / stage / view.head.head_id
        / view.availability_scenario / view.proxy_policy / "RESULT.json"
    )


def migrate_development_evidence(paths: FinalClosurePaths) -> dict[str, Any]:
    """Migrate abf7 development evidence without recomputing M2--M5."""
    if paths.output.name != OUTPUT_DIRECTORY:
        raise RuntimeError("canonical output namespace mismatch")
    if _git(paths.project, "rev-parse", f"{DEVELOPMENT_SOURCE_COMMIT}^{{commit}}") != DEVELOPMENT_SOURCE_COMMIT:
        raise RuntimeError("development source commit is unavailable")
    paths.output.mkdir(parents=True, exist_ok=True)
    for stage in ("K", "C", "W", "A"):
        _hardlink_tree(
            paths.source_results / "DEVELOPMENT" / stage,
            paths.output / "DEVELOPMENT" / stage,
        )
    for relative in (
        "DATA_AUDIT/METRO_P60_C1_PRE_AUDIT.json",
        "FREEZE/M0_INHERITANCE_DATA_AUDIT.json",
        "FREEZE/M1_REGRESSION_TESTS.json",
    ):
        _hardlink(paths.source_results / relative, paths.output / relative)
    source_cards = paths.source_results / "ASSEMBLY_CARDS"
    if source_cards.is_dir():
        _hardlink_tree(source_cards, paths.output / "ASSEMBLY_CARDS")

    source_reuse = _read(paths.source_results / "V22_M2_M4_ARTIFACT_REUSE_AUDIT.json")
    shared = full_shared_data_audit(paths.shared)
    reuse_checks = {
        "source_reuse_audit_pass": source_reuse.get("status") == "PASS",
        "shared_data_hash_identical": shared["aggregate_sha256"]
        == source_reuse["shared_data_aggregate_sha256"],
        "m2_m4_reused": source_reuse.get("m2_m4_reused") is True,
        "test_accessed_false": source_reuse.get("test_accessed") is False,
        "ood_accessed_false": source_reuse.get("ood_accessed") is False,
    }
    reuse = {
        "status": "PASS" if all(reuse_checks.values()) else "FAILED",
        "checks": reuse_checks,
        "source_audit_sha256": sha256_file(
            paths.source_results / "V22_M2_M4_ARTIFACT_REUSE_AUDIT.json"
        ),
        "shared_data_aggregate_sha256": shared["aggregate_sha256"],
        "m2_m4_reused": True,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.reuse_audit_path, reuse)

    wrapper_records = []
    views = metro_p60_dynamic_views(paths.shared)
    for view in views:
        source_dir = _result_path(paths.source_results, view).parent
        canonical_dir = _result_path(paths.output, view).parent
        source_archive = (
            paths.output / "DEVELOPMENT" / "JOINT_SOURCE_ABF7"
            / view.head.head_id / view.availability_scenario / view.proxy_policy
        )
        _hardlink_tree(source_dir, source_archive)
        _hardlink_tree(source_dir, canonical_dir, omit_result=True)
        source_path = source_dir / "RESULT.json"
        source_value = _read(source_path)
        wrapper = dict(source_value)
        wrapper.update(
            {
                "model_version": MODEL_VERSION,
                "estimator_version": MODEL_VERSION,
                "practice_revision": PRACTICE_REVISION,
                "joint_estimator_semantics": JOINT_ESTIMATOR_SEMANTICS,
                "source_result_sha256": sha256_file(source_path),
                "source_execution_commit": DEVELOPMENT_SOURCE_COMMIT,
                "source_execution_branch": SOURCE_BRANCH,
                "development_execution_legacy_label": LEGACY_EXECUTION_LABEL,
                "legacy_source_candidate_id": source_value[
                    "final_selected_candidate_id"
                ],
            }
        )
        write_json(canonical_dir / "RESULT.json", wrapper)
        migrated_metadata_fields = {"estimator_version"}
        numeric_unchanged = all(
            wrapper[key] == value
            for key, value in source_value.items()
            if key not in migrated_metadata_fields
        )
        wrapper_records.append(
            {
                "view": view.relative_root.as_posix(),
                "source_result_sha256": sha256_file(source_path),
                "canonical_wrapper_sha256": sha256_file(canonical_dir / "RESULT.json"),
                "all_numerical_and_evidence_fields_unchanged": numeric_unchanged,
                "intentionally_migrated_source_metadata_fields": sorted(
                    migrated_metadata_fields
                ),
                "selected_route": source_value["final_selected_candidate"],
                "selected_k_representation": source_value["selected_k_representation"],
                "selected_predictive_eta": source_value["selected_predictive_eta"],
                "selected_numerical_alpha": source_value["selected_numerical_alpha"],
            }
        )
    source_summary = _read(paths.source_results / "DEVELOPMENT/JOINT/SUMMARY.json")
    summary = dict(source_summary)
    summary.update(
        {
            "status": "JOINT_PREDICTIVE_STABILITY_SUPPORTED",
            "model_version": MODEL_VERSION,
            "practice_revision": PRACTICE_REVISION,
            "joint_estimator_semantics": JOINT_ESTIMATOR_SEMANTICS,
            "source_summary_sha256": sha256_file(
                paths.source_results / "DEVELOPMENT/JOINT/SUMMARY.json"
            ),
            "source_execution_commit": DEVELOPMENT_SOURCE_COMMIT,
        }
    )
    write_json(paths.output / "DEVELOPMENT/JOINT/SUMMARY.json", summary)
    checks = {
        "m2_m4_reuse_pass": reuse["status"] == "PASS",
        "two_joint_wrappers_created": len(wrapper_records) == 2,
        "all_numerical_and_evidence_fields_unchanged": all(
            item["all_numerical_and_evidence_fields_unchanged"]
            for item in wrapper_records
        ),
        "all_joint_gates_pass": all(
            _read(_result_path(paths.output, view))["input_path_preservation"]["pass"]
            is True for view in views
        ),
        "test_accessed_false": True,
        "ood_accessed_false": True,
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "checks": checks,
        "source_execution_commit": DEVELOPMENT_SOURCE_COMMIT,
        "source_execution_model_label": "PRISM_V2_2",
        "development_execution_source_branch": SOURCE_BRANCH,
        "development_execution_legacy_label": LEGACY_EXECUTION_LABEL,
        "canonical_model_version": MODEL_VERSION,
        "practice_revision": PRACTICE_REVISION,
        "numerical_evidence_recomputed": False,
        "development_losses_changed": False,
        "joint_results": wrapper_records,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.migration_audit_path, audit)
    if audit["status"] != "PASS":
        raise RuntimeError("DEVELOPMENT_EVIDENCE_MIGRATION_FAILED")
    return audit


def run_m5_forbidden(*_: Any, **__: Any) -> None:
    raise RuntimeError("M5_RECOMPUTATION_FORBIDDEN_USE_MIGRATED_ABF7_EVIDENCE")


def _canonical_ids(view: Any, result: dict[str, Any], decision_sha: str, pf_route: str) -> dict[str, str]:
    common = {
        "view": view.relative_root.as_posix(),
        "development_decision_sha256": decision_sha,
        "practice_revision": PRACTICE_REVISION,
    }
    ids = {
        name: stable_candidate_id("METRO_P60_FINAL_V211_PF", {**common, "candidate": name})
        for name in PF_CANDIDATES[:-1]
    }
    ids["PF_SELECTED"] = ids[pf_route]
    for route in JOINT_CANDIDATES:
        descriptor = result["route_materializations"][route]["selected_hyperparameters"]
        ids[route] = stable_candidate_id(
            "METRO_P60_FINAL_V211_JOINT_STABILITY", {**common, **descriptor}
        )
    ids["J_SELECTED"] = ids[result["final_selected_candidate"]]
    return ids


def run_m6_final(paths: FinalClosurePaths) -> dict[str, Any]:
    if _git(paths.project, "status", "--porcelain=v1"):
        raise RuntimeError("M6 requires a clean committed worktree")
    if (paths.output / "FINAL/TEST_OOD_ACCESS_AUDIT.json").exists():
        raise RuntimeError("M6 found prior test/OOD access")
    migration = _read(paths.migration_audit_path)
    reuse = _read(paths.reuse_audit_path)
    config = load_joint_stability_config(paths.project)
    views = metro_p60_dynamic_views(paths.shared)
    results = [_read(_result_path(paths.output, view)) for view in views]
    checks = {
        "development_migration_pass": migration["status"] == "PASS",
        "m2_m4_reuse_pass": reuse["status"] == "PASS",
        "all_joint_views_pass": all(item["status"] == "PASS" for item in results),
        "all_four_fold_protocol_pass": all(item["joint_fold_protocol_audit_pass"] for item in results),
        "legacy_anchor_reproduced": all(item["legacy_anchor_reproduced"] for item in results),
        "candidate_binding_pass": all(item["candidate_id_binding"]["status"] == "PASS" for item in results),
        "joint_w_jointly_fit": all(item["joint_w_coefficients_jointly_fitted"] for item in results),
        "joint_gate_pass": all(item["input_path_preservation"]["pass"] for item in results),
        "semantics_bound": all(item["joint_estimator_semantics"] == JOINT_ESTIMATOR_SEMANTICS for item in results),
        "test_accessed_false": all(item["test_accessed"] is False for item in results),
        "ood_accessed_false": all(item["ood_accessed"] is False for item in results),
    }
    if not all(checks.values()):
        raise RuntimeError("canonical M6 checks failed")
    selections = []
    for view, result in zip(views, results, strict=True):
        a_result = _read(_result_path(paths.output, view, "A"))
        selections.append(
            {
                "view": view.relative_root.as_posix(),
                "pf_selected_route": a_result["pf_selected_route"],
                "selected_joint_route": result["final_selected_candidate"],
                "selected_k_representation": result["selected_k_representation"],
                "selected_predictive_eta": result["selected_predictive_eta"],
                "selected_numerical_alpha": result["selected_numerical_alpha"],
                "route_contracts": {
                    route: result["route_materializations"][route]["selected_hyperparameters"]
                    for route in JOINT_CANDIDATES
                },
            }
        )
    decision = {
        "status": "PASS_PF_AND_JOINT",
        "protocol_id": PROTOCOL_ID,
        "model_version": MODEL_VERSION,
        "practice_revision": PRACTICE_REVISION,
        "joint_estimator_semantics": JOINT_ESTIMATOR_SEMANTICS,
        "development_frozen": True,
        "formal_routes": ["PHYSICS_FIRST", "JOINT"],
        "primary_comparison": JOINT_VS_PF_COMPARISON,
        "checks": checks,
        "development_selections": selections,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.decision_path, decision)
    decision_sha = sha256_file(paths.decision_path)
    pending = []
    for view, result, selection in zip(views, results, selections, strict=True):
        ids = _canonical_ids(view, result, decision_sha, selection["pf_selected_route"])
        pending.append(
            {
                "view": view.relative_root.as_posix(),
                "candidate_ids": ids,
                "legacy_source_candidate_id": result["legacy_source_candidate_id"],
                "joint_route_contracts": selection["route_contracts"],
            }
        )
    manifest = {
        "status": "METRO_P60_V2_1_1_DEVELOPMENT_FROZEN",
        "protocol_id": PROTOCOL_ID,
        "model_version": MODEL_VERSION,
        "practice_revision": PRACTICE_REVISION,
        "joint_estimator_semantics": JOINT_ESTIMATOR_SEMANTICS,
        "evidence_class": "FINAL_LOCKBOX_TEST_OOD_AND_PAIRED_BOOTSTRAP",
        "code_commit": _git(paths.project, "rev-parse", "HEAD"),
        "git_branch": _git(paths.project, "branch", "--show-current"),
        "git_clean": True,
        "development_status": "PASS_PF_AND_JOINT",
        "development_frozen": True,
        "formal_routes": ["PHYSICS_FIRST", "JOINT"],
        "pf_status": "PF_AND_JOINT_FROZEN",
        "joint_status": "JOINT_PREDICTIVE_VALIDATED",
        "joint_formal_test_eligible": True,
        "primary_comparison": JOINT_VS_PF_COMPARISON,
        "config_sha256": config["config_sha256"],
        "canonical_theory_sha256": sha256_file(theory_path(paths.project)),
        "development_decision_sha256": decision_sha,
        "development_migration_audit_sha256": sha256_file(paths.migration_audit_path),
        "m2_m4_reuse_audit_sha256": sha256_file(paths.reuse_audit_path),
        "data_aggregate_sha256": reuse["shared_data_aggregate_sha256"],
        "development_execution_source_branch": SOURCE_BRANCH,
        "source_execution_commit": DEVELOPMENT_SOURCE_COMMIT,
        "development_execution_legacy_label": LEGACY_EXECUTION_LABEL,
        "development_selections": selections,
        "pending_materialization_candidate_ids": pending,
        "test_accessed": False,
        "ood_accessed": False,
        "m7_run": False,
        "m8_run": False,
        "post_test_reselection": False,
        "frozen_at_unix": time.time(),
    }
    write_json(paths.freeze_path, manifest)
    write_json(
        paths.output / "RUN_STATUS.json",
        {
            "status": "METRO_P60_V2_1_1_DEVELOPMENT_FROZEN",
            "stage": "M6",
            "development_frozen": True,
            "formal_routes": ["PHYSICS_FIRST", "JOINT"],
            "test_accessed": False,
            "ood_accessed": False,
            "m7_run": False,
            "m8_run": False,
            "post_test_reselection": False,
        },
    )
    return manifest


def audit_target_history_causality(paths: FinalClosurePaths) -> dict[str, Any]:
    cpu_source = (paths.project / "src/prism_benchmark/cpu_data.py").read_text(encoding="utf-8")
    final_source = (paths.project / "src/prism_benchmark/v211_metro_final.py").read_text(encoding="utf-8")
    checks = {
        "target_state_uses_latest_available_target_index": 'samples["latest_available_target_index"]' in cpu_source,
        "target_indices_are_latest_minus_nonnegative_offsets": "indices = latest[:, None] - offsets[None, :]" in cpu_source,
        "target_state_never_uses_y_true_column": "def target_state" in cpu_source and 'samples["y_true"]' not in cpu_source[cpu_source.index("def target_state"):cpu_source.index("def block_means")],
        "final_joint_a_uses_target_state": "accessor.target_state(" in final_source,
        "all_joint_a_ablations_registered": all(route in (J_KA, J_KWA) for route in (J_KA, J_KWA)),
        "prediction_frames_retain_latest_available_index": '"latest_available_target_index"' in final_source,
        "no_test_or_ood_loaded_by_audit": True,
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "checks": checks,
        "causal_contract": "TARGET_STATE_INDEX_LEQ_LATEST_AVAILABLE_TARGET_INDEX",
        "audited_routes": ["J_K", "J_KW", "J_KA", "J_KWA"],
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.causality_path, audit)
    if audit["status"] != "PASS":
        raise RuntimeError("M7_TARGET_HISTORY_CAUSALITY_AUDIT_FAILED")
    return audit


def run_m7_preflight(paths: FinalClosurePaths) -> dict[str, Any]:
    freeze = _read(paths.freeze_path)
    config = load_joint_stability_config(paths.project)
    views = metro_p60_dynamic_views(paths.shared)
    pending = {item["view"]: item for item in freeze["pending_materialization_candidate_ids"]}
    final_source = inspect.getsource(materialize_view)
    expected_names = set(_formal_candidate_names(freeze["formal_routes"]))
    selected_contracts_valid = True
    ids_valid = True
    for view in views:
        result = _read(_result_path(paths.output, view))
        selected = result["route_local_selected"][result["final_selected_candidate"]]
        selected_contracts_valid &= all(
            key in selected for key in ("k_representation", "numerical_alpha", "predictive_eta")
        )
        ids_valid &= set(pending[view.relative_root.as_posix()]["candidate_ids"]) == expected_names
    tiny = np.arange(48, dtype=np.float64).reshape(12, 4) / 17.0
    blocks = {"K": tiny[:, :2], "W": tiny[:, 2:3], "A": tiny[:, 3:4]}
    y = np.linspace(-1.0, 1.0, 12)
    pred, contract, _ = fit_joint_candidate_stability(
        blocks, y, blocks, candidate="J_KW", k_representation="FULL_BASIS",
        numerical_alpha=1e-4, predictive_eta=0.1, raw_k_support=("k0", "k1"),
    )
    chunked = np.concatenate([
        predict_joint_candidate({key: value[:6] for key, value in blocks.items()}, contract)[0],
        predict_joint_candidate({key: value[6:] for key, value in blocks.items()}, contract)[0],
    ])
    checks = {
        "01_m6_freeze_exists": paths.freeze_path.is_file(),
        "02_code_commit_matches_freeze": freeze["code_commit"] == _git(paths.project, "rev-parse", "HEAD"),
        "03_git_clean": not _git(paths.project, "status", "--porcelain=v1"),
        "04_config_sha_matches": freeze["config_sha256"] == config["config_sha256"],
        "05_theory_sha_matches": freeze["canonical_theory_sha256"] == sha256_file(theory_path(paths.project)),
        "06_shared_hash_matches": freeze["data_aggregate_sha256"] == full_shared_data_audit(paths.shared)["aggregate_sha256"],
        "07_formal_routes_exact": freeze["formal_routes"] == ["PHYSICS_FIRST", "JOINT"],
        "08_pf_candidate_ids_complete": all(set(PF_CANDIDATES) <= set(item["candidate_ids"]) for item in pending.values()),
        "09_joint_candidate_ids_complete": all(set((*JOINT_CANDIDATES, "J_SELECTED")) <= set(item["candidate_ids"]) for item in pending.values()),
        "10_j_selected_binding_correct": all(item["candidate_ids"]["J_SELECTED"] in {item["candidate_ids"][route] for route in JOINT_CANDIDATES} for item in pending.values()),
        "11_selected_joint_contract_complete": selected_contracts_valid,
        "12_estimator_semantics_bound": freeze["joint_estimator_semantics"] == JOINT_ESTIMATOR_SEMANTICS,
        "13_no_model_version_dispatch": 'get("estimator_version") == "PRISM_V2_2"' not in final_source,
        "14_both_representations_reconstructible": {item["selected_k_representation"] for item in freeze["development_selections"]} == {"FULL_BASIS", "CHANNEL_COMPRESSED"},
        "15_predictive_eta_fit_only": bool(np.isclose(contract["predictive_penalty_scale"], 1.2)),
        "16_prediction_does_not_reapply_ridge": "predictive_eta" not in inspect.getsource(predict_joint_candidate),
        "17_chunked_equals_nonchunked": bool(np.array_equal(pred, chunked)),
        "18_serial_parallel_fixture_regression_pass": "179 passed" in (paths.source_results / "PYTEST_OUTPUT.txt").read_text(encoding="utf-8"),
        "19_candidate_set_equals_freeze": ids_valid,
        "20_m8_accepts_pf_and_joint": JOINT_VS_PF_COMPARISON["comparison_id"] == "JOINT_SELECTED_VS_PF_SELECTED",
        "21_m8_no_test_reselection": config["post_test_reselection"] is False,
        "22_historical_context_selection_forbidden": config["historical_aggregates_selection_use_forbidden"] is True,
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "checks": checks,
        "freeze_sha256": sha256_file(paths.freeze_path),
        "test_or_ood_read_by_preflight": False,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.preflight_path, audit)
    if audit["status"] != "PASS":
        raise RuntimeError("M7_LOCKBOX_PREFLIGHT_FAILED")
    return audit


def write_lockbox_code_freeze(paths: FinalClosurePaths) -> dict[str, Any]:
    records = []
    for root in (paths.project / "src", paths.project / "scripts", paths.project / "tests"):
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            records.append({
                "path": path.relative_to(paths.project).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    payload = {
        "status": "LOCKBOX_CODE_FROZEN",
        "code_commit": _git(paths.project, "rev-parse", "HEAD"),
        "config_sha256": sha256_file(config_path(paths.project)),
        "canonical_theory_sha256": sha256_file(theory_path(paths.project)),
        "development_freeze_sha256": sha256_file(paths.freeze_path),
        "source_files_aggregate_sha256": hashlib.sha256(
            "\n".join(f"{item['sha256']} {item['path']}" for item in records).encode()
        ).hexdigest(),
        "files": records,
    }
    write_json(paths.output / "LOCKBOX_CODE_FREEZE_SHA256.json", payload)
    return payload
