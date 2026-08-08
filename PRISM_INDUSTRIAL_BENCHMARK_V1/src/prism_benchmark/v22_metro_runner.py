from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .cpu_data import sha256_file
from .stage0 import write_json
from .v2_runtime import run_parallel
from .v211_joint import JOINT_CANDIDATES
from .v211_metro_contracts import stable_candidate_id
from .v211_metro_runner import full_shared_data_audit
from .v211_metro_views import metro_p60_dynamic_views
from .v22_config import (
    DEVELOPMENT_ARTIFACT_COMMIT,
    MODEL_VERSION,
    OUTPUT_DIRECTORY,
    PROTOCOL_ID,
    config_path,
    load_v22_config,
    theory_path,
)
from .v22_joint import run_joint_v22_view


PF_CANDIDATES = ("KC", "KCW", "KCA", "KCWA", "PF_SELECTED")


@dataclass(frozen=True)
class V22Paths:
    project: Path
    shared: Path
    output: Path
    source_results: Path
    legacy_results: Path
    inherited_pf_freeze: Path

    @property
    def reuse_audit_path(self) -> Path:
        return self.output / "V22_M2_M4_ARTIFACT_REUSE_AUDIT.json"

    @property
    def decision_path(self) -> Path:
        return self.output / "FREEZE/METRO_P60_V22_DEVELOPMENT_DECISION.json"

    @property
    def freeze_path(self) -> Path:
        return self.output / "FREEZE/METRO_P60_V22_DEVELOPMENT_FREEZE.json"


def _git(project: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hardlink_tree(source: Path, destination: Path) -> None:
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != sha256_file(path):
                raise RuntimeError(f"existing linked prerequisite differs: {target}")
            continue
        os.link(path, target)


def prepare_v22_prerequisites(paths: V22Paths) -> dict[str, Any]:
    if paths.output.name != OUTPUT_DIRECTORY:
        raise RuntimeError("v2.2 output namespace mismatch")
    paths.output.mkdir(parents=True, exist_ok=True)
    for stage in ("K", "C", "W", "A"):
        _hardlink_tree(
            paths.source_results / "DEVELOPMENT" / stage,
            paths.output / "DEVELOPMENT" / stage,
        )
    source_cards = paths.source_results / "ASSEMBLY_CARDS"
    if source_cards.is_dir():
        for card in source_cards.rglob("PF_ASSEMBLY_CARD.json"):
            target = paths.output / "ASSEMBLY_CARDS" / card.relative_to(source_cards)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(card, target)
    for relative in (
        "DATA_AUDIT/METRO_P60_C1_PRE_AUDIT.json",
        "FREEZE/M0_INHERITANCE_DATA_AUDIT.json",
        "FREEZE/M1_REGRESSION_TESTS.json",
    ):
        source = paths.source_results / relative
        target = paths.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)
    return audit_m2_m4_reuse(paths)


def _aggregate_hash(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        digest.update(str(record["path"]).encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _git_blob_sha(project: Path, commit: str, relative: str) -> str:
    payload = subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "show",
            f"{commit}:PRISM_INDUSTRIAL_BENCHMARK_V1/{relative}",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def audit_m2_m4_reuse(paths: V22Paths) -> dict[str, Any]:
    result_records = []
    oof_records = []
    result_values = []
    for stage in ("K", "C", "W", "A"):
        for target in sorted((paths.output / "DEVELOPMENT" / stage).glob("**/RESULT.json")):
            relative = target.relative_to(paths.output)
            source = paths.source_results / relative
            value = _read(target)
            result_values.append(value)
            result_records.append(
                {
                    "path": relative.as_posix(),
                    "source_sha256": sha256_file(source),
                    "linked_sha256": sha256_file(target),
                    "identical": sha256_file(source) == sha256_file(target),
                }
            )
            for key in (
                "oof_prediction_path",
                "oof_path",
                "physical_oof_path",
            ):
                if value.get(key):
                    linked = paths.output / str(value[key])
                    original = paths.source_results / str(value[key])
                    oof_records.append(
                        {
                            "owner": relative.as_posix(),
                            "field": key,
                            "path": str(value[key]),
                            "source_sha256": sha256_file(original),
                            "linked_sha256": sha256_file(linked),
                            "identical": sha256_file(original) == sha256_file(linked),
                        }
                    )
    estimator_files = [
        "src/prism_benchmark/v211_k.py",
        "src/prism_benchmark/v211_c.py",
        "src/prism_benchmark/v211_w.py",
        "src/prism_benchmark/v211_a.py",
    ]
    estimator_records = []
    for relative in estimator_files:
        source_sha = _git_blob_sha(paths.project, DEVELOPMENT_ARTIFACT_COMMIT, relative)
        current_sha = sha256_file(paths.project / relative)
        estimator_records.append(
            {
                "path": relative,
                "generating_commit_sha256": source_sha,
                "current_sha256": current_sha,
                "unchanged": source_sha == current_sha,
            }
        )
    source_m0 = _read(paths.source_results / "FREEZE/M0_INHERITANCE_DATA_AUDIT.json")
    shared = full_shared_data_audit(paths.shared)
    test_accessed = any(value.get("test_accessed", False) is True for value in result_values)
    ood_accessed = any(value.get("ood_accessed", False) is True for value in result_values)
    checks = {
        "shared_data_hash_identical": shared["aggregate_sha256"]
        == source_m0["data_aggregate_sha256"],
        "all_result_hashes_identical": all(item["identical"] for item in result_records),
        "all_c_w_oof_hashes_identical": bool(oof_records)
        and all(item["identical"] for item in oof_records),
        "k_c_w_a_estimators_unchanged": all(
            item["unchanged"] for item in estimator_records
        ),
        "test_accessed_false": not test_accessed,
        "ood_accessed_false": not ood_accessed,
        "test_ood_access_audit_absent": not (
            paths.output / "FINAL/METRO_P60_V212_TEST_OOD_ACCESS_AUDIT.json"
        ).exists(),
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "checks": checks,
        "development_artifact_source_commit": DEVELOPMENT_ARTIFACT_COMMIT,
        "freeze_semantics_parent_commit": _git(paths.project, "rev-parse", "HEAD"),
        "shared_data_aggregate_sha256": shared["aggregate_sha256"],
        "result_hash_aggregate_sha256": _aggregate_hash(
            [
                {"path": item["path"], "sha256": item["linked_sha256"]}
                for item in result_records
            ]
        ),
        "result_files": result_records,
        "c_w_oof_files": oof_records,
        "estimator_files": estimator_records,
        "m2_m4_reused": all(checks.values()),
        "prerequisite_link_semantics": "READ_ONLY_HARDLINK_NO_RECOMPUTATION",
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.reuse_audit_path, audit)
    if audit["status"] != "PASS":
        raise RuntimeError("V22_M2_M4_ARTIFACT_REUSE_AUDIT_FAILED")
    return audit


def _result_path(output: Path, view: Any) -> Path:
    return (
        output
        / "DEVELOPMENT/JOINT"
        / view.head.head_id
        / view.availability_scenario
        / view.proxy_policy
        / "RESULT.json"
    )


def _candidate_record(
    result: dict[str, Any],
    *,
    route: str,
    representation: str,
    eta: float | None = None,
) -> dict[str, Any]:
    matches = [
        item
        for item in result["candidate_registry"]
        if item["route"] == route
        and item["k_representation"] == representation
        and (eta is None or float(item["predictive_eta"]) == float(eta))
    ]
    if eta is None:
        selected_eta = float(
            result["route_local_selected"][route]["predictive_eta"]
            if result["route_local_selected"][route]["k_representation"] == representation
            else result["representation_comparison"][route][
                "compressed_selected_eta"
                if representation == "CHANNEL_COMPRESSED"
                else "full_selected_eta"
            ]
        )
        matches = [
            item for item in matches if float(item["predictive_eta"]) == selected_eta
        ]
    if len(matches) != 1:
        raise RuntimeError((route, representation, eta, len(matches)))
    return matches[0]


def build_stability_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    views = []
    for result in results:
        route = str(result["final_selected_candidate"])
        compressed_zero = _candidate_record(
            result, route=route, representation="CHANNEL_COMPRESSED", eta=0.0
        )
        full_zero = _candidate_record(
            result, route=route, representation="FULL_BASIS", eta=0.0
        )
        comparison = result["representation_comparison"][route]
        compressed_best = _candidate_record(
            result, route=route, representation="CHANNEL_COMPRESSED"
        )
        full_best = _candidate_record(result, route=route, representation="FULL_BASIS")
        gate = result["input_path_preservation"]
        views.append(
            {
                "view": f"{result['target_head']}/dynamic/{result['availability_scenario']}/{result['proxy_policy']}",
                "legacy_v212_joint_fold_losses": result[
                    "legacy_v212_joint_anchor"
                ]["expected_fold_losses"],
                "compressed_eta_zero_fold_losses": compressed_zero["fold_losses"],
                "full_eta_zero_fold_losses": full_zero["fold_losses"],
                "selected_compressed_best_eta": comparison[
                    "compressed_selected_eta"
                ],
                "selected_compressed_fold_losses": compressed_best["fold_losses"],
                "selected_full_best_eta": comparison["full_selected_eta"],
                "selected_full_fold_losses": full_best["fold_losses"],
                "final_representation": result["selected_k_representation"],
                "final_eta": result["selected_predictive_eta"],
                "final_numerical_alpha": result["selected_numerical_alpha"],
                "final_route": route,
                "best_k_fold_losses": result["best_active_k_fold_losses"],
                "gate_allowed_mse": gate.get("maximum_candidate_mse"),
                "joint_oof_mse": gate.get("candidate_mse"),
                "worst_fold_mse": result["selected_stability"]["worst_fold_loss"],
                "validation_mse": result["validation_mse"],
                "gate_checks": gate.get("checks"),
                "gate_pass": gate.get("pass"),
                "failure_class": result["input_path_failure_class"],
                "development_diagnosis": result["development_diagnosis"],
            }
        )
    return {
        "status": "PASS",
        "model_version": MODEL_VERSION,
        "views": views,
        "test_accessed": False,
        "ood_accessed": False,
    }


def run_m5_v22(paths: V22Paths) -> dict[str, Any]:
    audit = _read(paths.reuse_audit_path)
    if audit.get("status") != "PASS":
        raise RuntimeError("M5 requires a passing M2-M4 reuse audit")
    config = load_v22_config(paths.project)
    views = metro_p60_dynamic_views(paths.shared)
    results = run_parallel(
        run_joint_v22_view,
        [
            (paths.shared, paths.project, paths.output, paths.legacy_results, view)
            for view in views
        ],
        int(config["outer_view_workers"]),
        per_worker_gib=float(os.environ.get("PRISM_V22_MEMORY_GIB_PER_VIEW", "24")),
        label="PRISM_V22_METRO_M5_JOINT_STABILITY",
    )
    hard_failures = [
        item
        for item in results
        if item.get("status", "").startswith("STOP_")
    ]
    if hard_failures:
        write_json(
            paths.output / "DEVELOPMENT/JOINT/SUMMARY.json",
            {
                "status": hard_failures[0]["status"],
                "views": results,
                "test_accessed": False,
                "ood_accessed": False,
            },
        )
        raise RuntimeError(hard_failures[0]["status"])
    all_supported = all(item.get("status") == "PASS" for item in results)
    any_improved = any(
        item.get("development_decision")
        == "JOINT_V22_STABILITY_IMPROVED_BUT_NOT_SUPPORTED"
        for item in results
    )
    decision = (
        "JOINT_V22_PREDICTIVE_STABILITY_SUPPORTED"
        if all_supported
        else "JOINT_V22_STABILITY_IMPROVED_BUT_NOT_SUPPORTED"
        if any_improved
        else "JOINT_V22_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT"
    )
    summary = {
        "status": decision,
        "stage": "M5_JOINT_V22_PREDICTIVE_STABILITY",
        "views": len(results),
        "joint_formal_test_eligible": all_supported,
        "legacy_anchor_reproduced": all(
            item.get("legacy_anchor_reproduced") is True for item in results
        ),
        "fold_protocol_all_pass": all(
            item.get("joint_fold_protocol_audit_pass") is True for item in results
        ),
        "formal_routes": ["PHYSICS_FIRST", "JOINT"]
        if all_supported
        else ["PHYSICS_FIRST"],
        "view_decisions": [
            {
                "proxy_policy": item["proxy_policy"],
                "decision": item["development_decision"],
                "diagnosis": item["development_diagnosis"],
                "selected_route": item["final_selected_candidate"],
                "selected_k_representation": item["selected_k_representation"],
                "selected_predictive_eta": item["selected_predictive_eta"],
                "selected_numerical_alpha": item["selected_numerical_alpha"],
                "fold_losses": item["final_selected_fold_losses"],
                "gate_pass": item["input_path_preservation"]["pass"],
            }
            for item in results
        ],
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.output / "DEVELOPMENT/JOINT/SUMMARY.json", summary)
    write_json(
        paths.output / "V22_JOINT_STABILITY_COMPARISON.json",
        build_stability_comparison(results),
    )
    return summary


def run_m6_v22(paths: V22Paths) -> dict[str, Any]:
    summary = _read(paths.output / "DEVELOPMENT/JOINT/SUMMARY.json")
    if summary.get("status") != "JOINT_V22_PREDICTIVE_STABILITY_SUPPORTED":
        raise RuntimeError("v2.2 M6 is forbidden unless all Joint views pass")
    if (paths.output / "FINAL").exists():
        raise RuntimeError("v2.2 M6 found an unexpected FINAL/test/OOD directory")
    if _git(paths.project, "status", "--porcelain=v1"):
        raise RuntimeError("v2.2 M6 requires a clean code tree")
    views = metro_p60_dynamic_views(paths.shared)
    results = [_read(_result_path(paths.output, view)) for view in views]
    checks = {
        "m2_m4_reuse_audit_pass": _read(paths.reuse_audit_path)["status"] == "PASS",
        "all_joint_views_pass": all(item["status"] == "PASS" for item in results),
        "all_four_fold_protocol_pass": all(
            item["joint_fold_protocol_audit_pass"] is True for item in results
        ),
        "legacy_anchor_reproduced": all(
            item["legacy_anchor_reproduced"] is True for item in results
        ),
        "candidate_binding_pass": all(
            item["candidate_id_binding"]["status"] == "PASS" for item in results
        ),
        "joint_w_jointly_fit": all(
            item["joint_w_coefficients_jointly_fitted"] is True for item in results
        ),
        "joint_gate_pass": all(
            item["input_path_preservation"]["pass"] is True for item in results
        ),
        "test_accessed_false": all(item["test_accessed"] is False for item in results),
        "ood_accessed_false": all(item["ood_accessed"] is False for item in results),
    }
    if not all(checks.values()):
        raise RuntimeError("v2.2 M6 freeze check failed")
    selections = [
        {
            "view": view.relative_root.as_posix(),
            "selected_route": result["final_selected_candidate"],
            "selected_k_representation": result["selected_k_representation"],
            "selected_predictive_eta": result["selected_predictive_eta"],
            "selected_numerical_alpha": result["selected_numerical_alpha"],
            "candidate_id": result["final_selected_candidate_id"],
            "route_contracts": {
                route: payload["selected_hyperparameters"]
                for route, payload in result["route_materializations"].items()
            },
        }
        for view, result in zip(views, results, strict=True)
    ]
    decision = {
        "status": "PASS_PF_AND_JOINT",
        "protocol_id": PROTOCOL_ID,
        "model_version": MODEL_VERSION,
        "development_frozen": True,
        "formal_routes": ["PHYSICS_FIRST", "JOINT"],
        "pf_status": "PF_AND_JOINT_FROZEN",
        "joint_status": "JOINT_PREDICTIVE_VALIDATED",
        "joint_formal_test_eligible": True,
        "checks": checks,
        "development_selections": selections,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.decision_path, decision)
    decision_sha = sha256_file(paths.decision_path)
    pending = []
    for view, result in zip(views, results, strict=True):
        common = {
            "view": view.relative_root.as_posix(),
            "development_decision_sha256": decision_sha,
        }
        identifiers = {
            name: stable_candidate_id("FINAL", {**common, "candidate": name})
            for name in PF_CANDIDATES[:-1]
        }
        a_result = _read(
            paths.output
            / "DEVELOPMENT/A"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "RESULT.json"
        )
        pf_route = str(a_result["pf_selected_route"])
        if pf_route not in PF_CANDIDATES[:-1]:
            raise RuntimeError(f"unregistered frozen PF route: {pf_route}")
        identifiers["PF_SELECTED"] = identifiers[pf_route]
        for route in JOINT_CANDIDATES:
            descriptor = result["route_materializations"][route][
                "selected_hyperparameters"
            ]
            identifiers[route] = stable_candidate_id(
                "FINAL_V22_JOINT", {**common, **descriptor}
            )
        identifiers["J_SELECTED"] = identifiers[result["final_selected_candidate"]]
        pending.append(
            {
                "view": view.relative_root.as_posix(),
                "candidate_ids": identifiers,
                "joint_route_contracts": {
                    route: result["route_materializations"][route][
                        "selected_hyperparameters"
                    ]
                    for route in JOINT_CANDIDATES
                },
            }
        )
    inherited = _read(paths.inherited_pf_freeze)
    config = load_v22_config(paths.project)
    manifest = {
        "status": "METRO_P60_V2_2_DEVELOPMENT_FROZEN",
        "protocol_id": PROTOCOL_ID,
        "model_version": MODEL_VERSION,
        "code_commit": _git(paths.project, "rev-parse", "HEAD"),
        "git_branch": _git(paths.project, "branch", "--show-current"),
        "git_clean": True,
        "development_status": "PASS_PF_AND_JOINT",
        "development_frozen": True,
        "formal_routes": ["PHYSICS_FIRST", "JOINT"],
        "pf_status": "PF_AND_JOINT_FROZEN",
        "joint_status": "JOINT_PREDICTIVE_VALIDATED",
        "joint_formal_test_eligible": True,
        "config_sha256": config["config_sha256"],
        "development_decision_sha256": decision_sha,
        "data_aggregate_sha256": _read(paths.reuse_audit_path)[
            "shared_data_aggregate_sha256"
        ],
        "parent_amended_theory_sha256": inherited[
            "amended_practice_theory_sha256"
        ],
        "v22_theory_sha256": sha256_file(theory_path(paths.project)),
        "m2_m4_reused": True,
        "legacy_anchor_reproduced": True,
        "pending_materialization_candidate_ids": pending,
        "test_accessed": False,
        "ood_accessed": False,
        "m7_run": False,
        "m8_run": False,
        "frozen_at_unix": time.time(),
    }
    write_json(paths.freeze_path, manifest)
    write_json(
        paths.output / "RUN_STATUS.json",
        {
            "status": "PASS_PF_AND_JOINT",
            "stage": "M6",
            "development_frozen": True,
            "formal_routes": ["PHYSICS_FIRST", "JOINT"],
            "test_accessed": False,
            "ood_accessed": False,
        },
    )
    return manifest
