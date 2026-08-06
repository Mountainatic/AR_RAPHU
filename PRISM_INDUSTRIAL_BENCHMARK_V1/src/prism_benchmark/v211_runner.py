from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cpu_data import inner_folds, load_samples, sha256_file
from .cpu_selection import mse
from .stage0 import write_json
from .v2_selection import practical_activation
from .v21_audit import write_post_audit, write_pre_audit
from .v21_baselines import (
    INVENTORY_NAME,
    REPLAY_MANIFEST_NAME,
    REPLAY_STATUS,
    freeze_baseline_inventory,
)
from .v21_config import ACTIVE_HEADS, V21Paths
from .v21_selection import assert_final_prediction_contract
from .v21_views import assert_only_sru, sru_dynamic_views, sru_input_views
from .v211_a import run_e4r_a
from .v211_c import run_e2r_c
from .v211_config import V211Paths, load_v211_config
from .v211_joint import run_e5r_joint
from .v211_k import run_e2r_k
from .v211_reporting import build_stop_report_and_package
from .v211_w import IDENTITY, run_e3r_w


E1_TESTS = (
    "tests/test_v211_repair.py",
    "tests/test_v21_selection.py",
    "tests/test_v21_maturity.py",
    "tests/test_v21_w.py",
    "tests/test_v21_a.py",
    "tests/test_v21_joint.py",
    "tests/test_v21_assembly.py",
    "tests/test_v21_data_immutability.py",
    "tests/test_v21_prediction_contract.py",
    "tests/test_v21_baselines.py",
    "tests/test_v21_reporting.py",
)

CHAIN_STAGES = (
    "e0r",
    "e1r",
    "e2rk",
    "e2rc",
    "e3r",
    "e4r",
    "e5r",
    "e55",
    "e6r",
    "e7r",
    "e8r",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(project: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "core.filemode=false", *args],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _link_or_copy(source: str, destination: str) -> str:
    if Path(destination).exists():
        return destination
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def run_e0r(paths: V211Paths) -> dict[str, Any]:
    config = load_v211_config(paths.project)
    source = paths.resolved_baseline_source
    if source == paths.output.resolve():
        raise RuntimeError("baseline source and v2.1.1 output cannot be the same")
    source_paths = V21Paths(
        project=paths.project, shared=paths.shared, output=source
    )
    source_inventory = freeze_baseline_inventory(source_paths)
    baseline_destination = paths.output / "BASELINES"
    baseline_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source / "BASELINES",
        baseline_destination,
        copy_function=_link_or_copy,
        dirs_exist_ok=True,
    )
    inventory = freeze_baseline_inventory(paths)
    pre = write_pre_audit(paths.shared, paths.output)
    post = write_post_audit(paths.shared, paths.output)
    validation = json.loads(
        (paths.shared / "C1_VALIDATION.json").read_text(encoding="utf-8")
    )
    input_views = sru_input_views(paths.shared)
    dynamic_views = sru_dynamic_views(paths.shared)
    assert_only_sru(input_views)
    assert_only_sru(dynamic_views)
    observed_heads = {
        view.head.head_id for view in [*input_views, *dynamic_views]
    }
    source_manifest = source / "BASELINES" / REPLAY_MANIFEST_NAME
    copied_manifest = paths.output / "BASELINES" / REPLAY_MANIFEST_NAME
    source_inventory_path = source / "BASELINES" / INVENTORY_NAME
    copied_inventory_path = paths.output / "BASELINES" / INVENTORY_NAME
    branch = _git(paths.project, "branch", "--show-current")
    dirty_status = _git(paths.project, "status", "--short")
    checks = {
        "C1_validation_pass": validation.get("status") == "PASS",
        "C1_hash_audit_pass": post["comparison_to_pre"]["status"] == "PASS",
        "active_heads_exact": observed_heads == ACTIVE_HEADS,
        "only_sru_active": config["active_datasets"] == ["sru"],
        "write_shared_data_false": config["write_shared_data"] is False,
        "baseline_manifest_hash_reused": _sha256(source_manifest)
        == _sha256(copied_manifest),
        "baseline_inventory_hash_reused": _sha256(source_inventory_path)
        == _sha256(copied_inventory_path),
        "baseline_inventory_entries_exact": source_inventory["entries"]
        == inventory["entries"],
        "baseline_replay_frozen": json.loads(
            copied_manifest.read_text(encoding="utf-8")
        ).get("status")
        == REPLAY_STATUS,
        "baseline_test_metrics_not_parsed": True,
        "baseline_retuning_forbidden": config["retune_baselines"] is False,
        "branch_is_v211_or_detached_release": branch
        in {"prism-v2-1-1-sru-implementation-correction", ""},
        "source_code_tree_clean": not dirty_status,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "STOP_INHERITANCE_MISMATCH",
        "stage": "E0R_HASH_AND_BASELINE_REUSE_AUDIT",
        "checks": checks,
        "git_head": _git(paths.project, "rev-parse", "HEAD"),
        "git_branch": branch,
        "git_dirty": bool(dirty_status),
        "baseline_source": str(source),
        "baseline_copy_method": "HARDLINK_WITH_COPY2_FALLBACK",
        "data_pre_audit_sha256": _sha256(
            paths.output / "DATA_AUDIT" / "V21_DATA_BASE_PRE_AUDIT.json"
        ),
        "data_post_audit_sha256": _sha256(
            paths.output / "DATA_AUDIT" / "V21_DATA_BASE_POST_AUDIT.json"
        ),
        "baseline_replay_manifest_sha256": _sha256(copied_manifest),
        "baseline_replay_test_accessed": True,
        "baseline_replay_authorized_exception": True,
        "baseline_test_metrics_exposed_to_selection": False,
        "v211_candidate_test_accessed": False,
        "test_accessed": False,
        "pre_audit_files": pre["total_files"],
    }
    write_json(paths.output / "FREEZE" / "E0R_HASH_AUDIT.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(result["status"])
    return result


def run_e1r(paths: V211Paths) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", *E1_TESTS, "-q"]
    completed = subprocess.run(
        command,
        cwd=paths.project,
        text=True,
        capture_output=True,
    )
    result = {
        "status": "PASS" if completed.returncode == 0 else "FAILED",
        "stage": "E1R_REGRESSION_TESTS",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "test_accessed": False,
    }
    write_json(paths.output / "FREEZE" / "E1R_REGRESSION_TESTS.json", result)
    if completed.returncode:
        raise RuntimeError("E1R v2.1.1 regression tests failed")
    return result


def _result_files(output: Path) -> list[Path]:
    return sorted(
        path
        for path in (output / "DEVELOPMENT").rglob("RESULT.json")
        if path.is_file()
    )


def _read_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_entry(
    inventory: dict[str, Any],
    *,
    target_head: str,
    information_set: str,
    availability_scenario: str,
    proxy_policy: str,
) -> dict[str, Any]:
    key = "|".join(
        (target_head, information_set, availability_scenario, proxy_policy)
    )
    model = str(inventory["best_by_validation"][key])
    for entry in inventory["entries"]:
        if all(
            str(entry[name]) == value
            for name, value in (
                ("target_head", target_head),
                ("information_set", information_set),
                ("availability_scenario", availability_scenario),
                ("proxy_policy", proxy_policy),
                ("model", model),
            )
        ):
            return entry
    raise RuntimeError(f"strongest development baseline is missing: {key}: {model}")


def _development_comparison(
    paths: V211Paths,
    view: Any,
    candidate_path: Path,
    candidate_model: str,
    baseline_entry: dict[str, Any],
) -> dict[str, Any]:
    candidate = pd.read_parquet(candidate_path)
    baseline_root = paths.output / "BASELINES" / "REPLAY" / "FINAL_FIT"
    baseline = pd.read_parquet(baseline_root / baseline_entry["validation_path"])
    candidate_id = "view_sample_id" if "view_sample_id" in candidate else "sample_id"
    candidate_index = candidate.set_index(candidate[candidate_id].astype(str))
    baseline_index = baseline.set_index(baseline["sample_id"].astype(str))
    validation = load_samples(paths.shared, view, "validation")
    ids = validation["view_sample_id"].astype(str)
    if not ids.isin(candidate_index.index).all() or not ids.isin(baseline_index.index).all():
        raise RuntimeError("development candidate/baseline sample alignment failed")
    candidate_aligned = candidate_index.loc[ids]
    baseline_aligned = baseline_index.loc[ids]
    target = validation["y_true"].to_numpy(dtype=np.float64)
    if not np.array_equal(
        target,
        candidate_aligned["y_true"].to_numpy(dtype=np.float64),
        equal_nan=True,
    ) or not np.array_equal(
        target,
        baseline_aligned["y_true"].to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise RuntimeError("development candidate/baseline target alignment failed")
    candidate_prediction = candidate_aligned["y_pred"].to_numpy(dtype=np.float64)
    baseline_prediction = baseline_aligned["y_pred"].to_numpy(dtype=np.float64)
    candidate_mse = mse(target, candidate_prediction)
    baseline_mse = mse(target, baseline_prediction)
    relative_improvement = (baseline_mse - candidate_mse) / max(
        abs(baseline_mse), np.finfo(np.float64).eps
    )
    blocks = []
    for block, (_, evaluation_index) in enumerate(inner_folds(validation, 4)):
        block_candidate = mse(target[evaluation_index], candidate_prediction[evaluation_index])
        block_baseline = mse(target[evaluation_index], baseline_prediction[evaluation_index])
        blocks.append(
            {
                "block": block,
                "candidate_mse": block_candidate,
                "baseline_mse": block_baseline,
                "improved": bool(block_candidate < block_baseline),
            }
        )
    positive = sum(item["improved"] for item in blocks)
    passed = relative_improvement >= 0.01 and positive >= 3
    return {
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "candidate_model": candidate_model,
        "candidate_prediction_path": str(candidate_path.relative_to(paths.output)),
        "baseline_model": baseline_entry["model"],
        "candidate_mse": candidate_mse,
        "baseline_mse": baseline_mse,
        "relative_improvement": relative_improvement,
        "positive_blocks": positive,
        "total_blocks": len(blocks),
        "blocks": blocks,
        "pass": bool(passed),
        "test_accessed": False,
    }


def run_e55(paths: V211Paths) -> dict[str, Any]:
    config = load_v211_config(paths.project)
    inventory = freeze_baseline_inventory(paths)
    c_results = [
        _read_result(path)
        for path in sorted((paths.output / "DEVELOPMENT" / "C").glob("*/*/RESULT.json"))
    ]
    w_results = [
        _read_result(path)
        for path in sorted((paths.output / "DEVELOPMENT" / "W").glob("*/*/RESULT.json"))
    ]
    a_results = [
        _read_result(path)
        for path in sorted((paths.output / "DEVELOPMENT" / "A").glob("*/*/*/RESULT.json"))
    ]
    joint_results = [
        _read_result(path)
        for path in sorted((paths.output / "DEVELOPMENT" / "JOINT").glob("*/*/*/RESULT.json"))
    ]
    c_by_head = {
        (item["target_head"], item["proxy_policy"]): item for item in c_results
    }
    joint_by_head = {
        (item["target_head"], item["proxy_policy"]): item for item in joint_results
    }
    pf_joint_matches = []
    for key, c_result in c_by_head.items():
        joint = joint_by_head.get(key)
        pf = bool(c_result.get("input_path_preservation", {}).get("pass", False))
        joint_pass = bool(
            joint
            and joint.get("input_path_preservation", {}).get("pass", False)
        )
        pf_joint_matches.append(pf == joint_pass)
    w_activation_audits = []
    for result in w_results:
        selected = str(result.get("final_selected_candidate"))
        if selected == IDENTITY:
            w_activation_audits.append(
                {"target_head": result.get("target_head"), "active": False, "pass": True}
            )
            continue
        losses = result.get("candidate_fold_losses", {})
        audit = practical_activation(
            list(losses[IDENTITY]),
            list(losses[selected]),
            minimum_relative_improvement=0.01,
            minimum_positive_fraction=0.75,
        )
        w_activation_audits.append(
            {
                "target_head": result.get("target_head"),
                "active": True,
                "usable_fold_count": result.get("usable_fold_count"),
                "activation": audit,
                "pass": bool(
                    result.get("usable_fold_count", 0) >= 3 and audit["pass"]
                ),
            }
        )
    views = {
        (
            view.head.head_id,
            view.information_set,
            view.availability_scenario,
            view.proxy_policy,
        ): view
        for view in [*sru_input_views(paths.shared), *sru_dynamic_views(paths.shared)]
    }
    formal_candidates = []
    for result in w_results:
        c_result = c_by_head[(result["target_head"], result["proxy_policy"])]
        if result.get("status") == "PASS" and c_result.get(
            "input_path_preservation", {}
        ).get("pass", False):
            formal_candidates.append(
                (result, "input_only", "PRISM_V2_1_1_K_C_W")
            )
    for result in a_results:
        c_result = c_by_head[(result["target_head"], result["proxy_policy"])]
        if result.get("status") == "PASS" and c_result.get(
            "input_path_preservation", {}
        ).get("pass", False):
            formal_candidates.append(
                (result, "dynamic", "PRISM_V2_1_1_PHYSICS_FIRST")
            )
    for result in joint_results:
        if result.get("status") == "PASS":
            formal_candidates.append(
                (result, "dynamic", "PRISM_V2_1_1_JOINT_KWA")
            )
    comparisons = []
    for result, information_set, model in formal_candidates:
        key = (
            result["target_head"],
            information_set,
            result.get("availability_scenario", "record_time"),
            result["proxy_policy"],
        )
        view = views[key]
        entry = _baseline_entry(
            inventory,
            target_head=key[0],
            information_set=key[1],
            availability_scenario=key[2],
            proxy_policy=key[3],
        )
        comparisons.append(
            _development_comparison(
                paths,
                view,
                paths.output / result["prediction_path"],
                model,
                entry,
            )
        )
    all_results = _result_files(paths.output)
    no_candidate_test_access = all(
        _read_result(path).get("test_accessed") is False for path in all_results
    )
    checks = {
        "minimum_supported_heads": sum(
            bool(item.get("input_path_preservation", {}).get("pass", False))
            for item in c_results
        )
        >= int(config["development_continue_gate"]["minimum_supported_heads"]),
        "pf_joint_input_status_match": bool(pf_joint_matches)
        and all(pf_joint_matches),
        "no_c_input_path_collapse_bug": all(
            item.get("selection_status") != "C_INPUT_PATH_COLLAPSE_BUG"
            for item in c_results
        ),
        "active_w_rules_pass": all(item["pass"] for item in w_activation_audits),
        "no_retained_solver_failure": all(
            not str(_read_result(path).get("status", "")).startswith("SOLVER_FAILED")
            for path in all_results
        ),
        "supported_increment_over_strongest_baseline": any(
            item["pass"] for item in comparisons
        ),
        "candidate_test_not_accessed": no_candidate_test_access,
    }
    passed = all(checks.values())
    decision = {
        "status": "PASS" if passed else "V2_1_1_DEVELOPMENT_STOP",
        "stage": "E5_5_DEVELOPMENT_CONTINUE_GATE",
        "decision": (
            "CONTINUE_TO_E6R" if passed else "NO_SUPPORTED_INCREMENT_AFTER_IMPLEMENTATION_REPAIR"
        ),
        "test_status": "TEST_NOT_ACCESSED",
        "checks": checks,
        "pf_joint_matches": pf_joint_matches,
        "w_activation_audits": w_activation_audits,
        "candidate_comparisons": comparisons,
        "test_accessed": False,
    }
    write_json(paths.development_decision_path, decision)
    if not passed:
        stop = build_stop_report_and_package(paths, decision)
        write_json(paths.output / "REPORTS" / "E55_STOP_PACKAGE_SUMMARY.json", stop)
        return {**decision, "stop_artifacts": stop}
    return decision


def freeze_e6r(paths: V211Paths) -> dict[str, Any]:
    load_v211_config(paths.project)
    decision = json.loads(paths.development_decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "PASS":
        raise RuntimeError("E6R requires a PASS development continue gate")
    required = {
        "E2R_K": False,
        "E2R_C": False,
        "E3R_W": False,
        "E4R_A": False,
        "E5R_JOINT": False,
    }
    selections = []
    validation_files = []
    for path in _result_files(paths.output):
        result = _read_result(path)
        stage = result.get("stage")
        if stage in required:
            required[stage] = True
        if result.get("status") not in {"PASS", "JOINT_INPUT_PATH_COLLAPSED"}:
            raise RuntimeError(f"cannot freeze retained failure: {path}")
        if result.get("test_accessed") is not False:
            raise RuntimeError(f"candidate test was accessed before E6R: {path}")
        if result.get("status") == "JOINT_INPUT_PATH_COLLAPSED":
            selections.append(
                {
                    "stage": stage,
                    "target_head": result.get("target_head"),
                    "proxy_policy": result.get("proxy_policy"),
                    "availability_scenario": result.get("availability_scenario"),
                    "selected": None,
                    "status": "JOINT_INPUT_PATH_COLLAPSED",
                    "contract_sha256": _sha256(path),
                }
            )
            continue
        prediction = paths.output / result["final_selected_prediction_path"]
        frame = pd.read_parquet(prediction, columns=["y_true", "y_pred"])
        recomputed = mse(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        )
        assert_final_prediction_contract(result, recomputed_loss=recomputed)
        validation_files.append(
            {
                "path": prediction.relative_to(paths.output).as_posix(),
                "sha256": _sha256(prediction),
            }
        )
        selections.append(
            {
                "stage": stage,
                "target_head": result.get("target_head"),
                "proxy_policy": result.get("proxy_policy"),
                "availability_scenario": result.get("availability_scenario"),
                "selected": result.get("final_selected_candidate"),
                "contract_sha256": _sha256(path),
            }
        )
    if not all(required.values()):
        raise RuntimeError(f"E6R prerequisites missing: {required}")
    baseline_inventory = freeze_baseline_inventory(paths)
    post = write_post_audit(paths.shared, paths.output)
    if post["comparison_to_pre"]["status"] != "PASS":
        raise RuntimeError("STOP_DATA_BASE_MUTATED")
    frozen_config = paths.output / "FREEZE" / "V211_SRU_CONFIG_FROZEN.json"
    frozen_config.parent.mkdir(parents=True, exist_ok=True)
    frozen_config.write_bytes(paths.config_path.read_bytes())
    runtime_profile = paths.output / "RUN_LOG" / "V211_RUNTIME_PROFILE.json"
    if not runtime_profile.is_file():
        raise FileNotFoundError(runtime_profile)
    baseline_replay = paths.output / "BASELINES" / REPLAY_MANIFEST_NAME
    dirty_status = _git(paths.project, "status", "--short")
    if dirty_status:
        raise RuntimeError("E6R refuses to freeze a dirty source tree")
    manifest = {
        "status": "V2_1_1_ASSEMBLY_FROZEN",
        "stage": "E6R_FINAL_FREEZE",
        "plan_sha256": _sha256(
            paths.plan / "PRISM_V2_1_1_SRU_REPAIR_PLAN.md"
        ),
        "config_sha256": _sha256(paths.config_path),
        "frozen_config_sha256": _sha256(frozen_config),
        "code_commit": _git(paths.project, "rev-parse", "HEAD"),
        "git_branch": _git(paths.project, "branch", "--show-current"),
        "dirty_status": dirty_status,
        "development_continue_gate": "PASS",
        "development_decision_sha256": _sha256(paths.development_decision_path),
        "data_pre_audit_sha256": _sha256(
            paths.output / "DATA_AUDIT" / "V21_DATA_BASE_PRE_AUDIT.json"
        ),
        "data_post_audit_sha256": _sha256(
            paths.output / "DATA_AUDIT" / "V21_DATA_BASE_POST_AUDIT.json"
        ),
        "baseline_inventory_sha256": _sha256(
            paths.output / "BASELINES" / INVENTORY_NAME
        ),
        "baseline_replay_manifest_sha256": _sha256(baseline_replay),
        "baseline_replay_amendment_sha256": _sha256(
            paths.baseline_amendment_path
        ),
        "runtime_profile_sha256": _sha256(runtime_profile),
        "baseline_inclusion": baseline_inventory["entries"],
        "best_frozen_baselines": baseline_inventory["best_by_validation"],
        "selections": selections,
        "validation_files": validation_files,
        "baseline_replay_test_accessed": True,
        "baseline_test_metrics_exposed_to_selection": False,
        "v211_candidate_test_accessed": False,
        "test_accessed": False,
    }
    write_json(paths.final_freeze_path, manifest)
    return manifest


def run_stage(stage: str, paths: V211Paths) -> dict[str, Any]:
    if paths.output.name != "results_prism_v2_1_1_sru":
        raise RuntimeError(
            "v2.1.1 output must use results_prism_v2_1_1_sru namespace"
        )
    if stage == "e0r":
        return run_e0r(paths)
    if stage == "e1r":
        return run_e1r(paths)
    if stage == "e2rk":
        return run_e2r_k(paths.shared, paths.project, paths.output)
    if stage == "e2rc":
        return run_e2r_c(paths.shared, paths.project, paths.output)
    if stage == "e3r":
        return run_e3r_w(paths.shared, paths.project, paths.output)
    if stage == "e4r":
        return run_e4r_a(paths.shared, paths.project, paths.output)
    if stage == "e5r":
        return run_e5r_joint(paths.shared, paths.project, paths.output)
    if stage == "e55":
        return run_e55(paths)
    if stage == "e6r":
        return freeze_e6r(paths)
    if stage in {"e7r", "e8r"}:
        from .v211_final import run_e7r_test, run_e8r_report

        return run_e7r_test(paths) if stage == "e7r" else run_e8r_report(paths)
    raise ValueError(f"unknown v2.1.1 stage: {stage}")
