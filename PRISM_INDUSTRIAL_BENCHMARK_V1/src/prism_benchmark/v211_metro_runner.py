from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cpu_data import input_columns, sha256_file
from .stage0 import write_json
from .v2_runtime import run_parallel
from .v211_a import run_a_view
from .v211_assembly import (
    build_joint_card,
    build_physics_first_card,
    pf_and_joint_input_status_match,
)
from .v211_c import run_c_view
from .v211_joint import JOINT_CANDIDATES, run_joint_view
from .v211_k import load_active_channels, run_k_channel
from .v211_metro_config import (
    ACTIVE_HEAD,
    CONFIG_SHA256,
    DEVELOPMENT_DECISION_NAME,
    DEVELOPMENT_FREEZE_NAME,
    EVIDENCE_CLASS,
    OUTPUT_DIRECTORY,
    PROTOCOL_ID,
    RECOMMENDED_BRANCH,
    SOURCE_COMMIT,
    MetroV211Paths,
    effective_worker_count,
    effective_k_outer_workers,
    git_value,
    load_metro_config,
    runtime_parallelism_audit,
)
from .v211_metro_contracts import (
    assert_candidate_id_binding,
    bind_result_candidate_ids,
    stable_candidate_id,
)
from .v211_metro_views import metro_p60_dynamic_views, metro_p60_input_views
from .v211_w import W_FAMILIES, run_w_view


PROTOCOL = "metro_p60"
STAGES = ("m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8")
M1_TESTS = (
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
    "tests/test_v211_metro_audit.py",
    "tests/test_v212_joint_oof_protocol.py",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(project: Path, *arguments: str) -> str:
    return git_value(project, *arguments)


def _aggregate_file_hash(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: str(value["path"])):
        digest.update(str(item["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def full_shared_data_audit(shared: Path) -> dict[str, Any]:
    records = []
    for path in sorted(value for value in shared.rglob("*") if value.is_file()):
        records.append(
            {
                "path": path.relative_to(shared).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "status": "PASS",
        "shared_root": str(shared),
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(int(item["bytes"]) for item in records),
        "aggregate_sha256": _aggregate_file_hash(records),
        "semantic_test_or_ood_access": False,
    }


def compare_shared_data_audits(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    left = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in before["files"]
    }
    right = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in after["files"]
    }
    added = sorted(right.keys() - left.keys())
    removed = sorted(left.keys() - right.keys())
    changed = sorted(key for key in left.keys() & right.keys() if left[key] != right[key])
    passed = not added and not removed and not changed
    return {
        "status": "PASS" if passed else "STOP_DATA_BASE_MUTATED",
        "identical": passed,
        "added": added,
        "removed": removed,
        "changed": changed,
        "before_aggregate_sha256": before["aggregate_sha256"],
        "after_aggregate_sha256": after["aggregate_sha256"],
    }


def _optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def runtime_environment_audit(config: MappingLike) -> dict[str, Any]:
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.__config__.show()
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "pandas", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "package_versions": versions,
        "numpy_blas_configuration": stream.getvalue(),
        "cgroup_cpu_max": _optional_text(Path("/sys/fs/cgroup/cpu.max")),
        "cgroup_cpuset_effective": _optional_text(
            Path("/sys/fs/cgroup/cpuset.cpus.effective")
        ),
        "cgroup_memory_max": _optional_text(Path("/sys/fs/cgroup/memory.max")),
        "cgroup_memory_swap_max": _optional_text(
            Path("/sys/fs/cgroup/memory.swap.max")
        ),
        "frozen_resource_contract": config["resource"],
        "runtime_parallelism": runtime_parallelism_audit(config),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def sample_id_audit(shared: Path, views: Iterable[Any]) -> list[dict[str, Any]]:
    records = []
    for view in views:
        for split in ("train", "validation", "test", "ood"):
            path = shared / "sample_ids" / view.relative_root / f"{split}.parquet"
            frame = pd.read_parquet(
                path, columns=["view_sample_id", "base_origin_id", "split"]
            )
            digest = hashlib.sha256()
            for sample_id, base_origin_id in zip(
                frame["view_sample_id"].astype(str),
                frame["base_origin_id"].astype(str),
                strict=True,
            ):
                digest.update(sample_id.encode("utf-8"))
                digest.update(b"\0")
                digest.update(base_origin_id.encode("utf-8"))
                digest.update(b"\n")
            records.append(
                {
                    "view": view.relative_root.as_posix(),
                    "split": split,
                    "rows": len(frame),
                    "sample_and_base_origin_id_sha256": digest.hexdigest(),
                    "split_column_exact": bool(
                        (frame["split"].astype(str) == split).all()
                    ),
                    "semantic_target_columns_read": False,
                }
            )
    return records


def _run_status(paths: MetroV211Paths, **updates: Any) -> dict[str, Any]:
    current = (
        _read(paths.output / "RUN_STATUS.json")
        if (paths.output / "RUN_STATUS.json").is_file()
        else {}
    )
    current.update(
        {
            "protocol_id": PROTOCOL_ID,
            "evidence_class": EVIDENCE_CLASS,
            "active_head": ACTIVE_HEAD,
            **updates,
        }
    )
    write_json(paths.output / "RUN_STATUS.json", current)
    return current


def stop(paths: MetroV211Paths, label: str, details: MappingLike | None = None) -> None:
    text = label + "\n"
    if details:
        text += json.dumps(dict(details), indent=2, sort_keys=True) + "\n"
    stop_path = paths.output / "STOP_STATE.txt"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text(text, encoding="utf-8")
    lockbox_accessed = paths.test_access_audit_path.exists()
    _run_status(
        paths,
        status="FAILED",
        stop_label=label,
        stage="STOPPED",
        test_accessed=lockbox_accessed,
        ood_accessed=lockbox_accessed,
    )
    raise RuntimeError(label)


MappingLike = dict[str, Any]


def _result_path(output: Path, stage: str, view: Any, channel: str | None = None) -> Path:
    if stage == "K":
        if channel is None:
            raise ValueError("K result path requires channel")
        return output / "DEVELOPMENT" / "K" / view.head.head_id / view.proxy_policy / channel / "RESULT.json"
    if stage in {"C", "W"}:
        return output / "DEVELOPMENT" / stage / view.head.head_id / view.proxy_policy / "RESULT.json"
    return (
        output
        / "DEVELOPMENT"
        / stage
        / view.head.head_id
        / view.availability_scenario
        / view.proxy_policy
        / "RESULT.json"
    )


def _load_stage_results(paths: list[Path]) -> list[dict[str, Any]]:
    return [_read(path) for path in paths]


def _bind(paths: MetroV211Paths, result_paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [bind_result_candidate_ids(paths.output, path) for path in result_paths]


def run_m0(paths: MetroV211Paths) -> dict[str, Any]:
    config = load_metro_config(paths.project)
    branch = _git(paths.project, "branch", "--show-current")
    head = _git(paths.project, "rev-parse", "HEAD")
    dirty = _git(paths.project, "status", "--porcelain=v1")
    ancestor = subprocess.run(
        ["git", "-C", str(paths.project), "merge-base", "--is-ancestor", SOURCE_COMMIT, head]
    ).returncode == 0
    if paths.output.name != OUTPUT_DIRECTORY:
        stop(paths, "STOP_OUTPUT_NAMESPACE_MISMATCH")
    if paths.test_access_audit_path.exists():
        stop(paths, "STOP_TEST_OR_OOD_EARLY_ACCESS")
    input_views = metro_p60_input_views(paths.shared)
    dynamic_views = metro_p60_dynamic_views(paths.shared)
    observed_heads = {view.head.head_id for view in [*input_views, *dynamic_views]}
    sample_counts = pd.read_csv(paths.shared / "C1_SAMPLE_COUNTS.csv")
    sample_counts = sample_counts[sample_counts["target_head"] == ACTIVE_HEAD].copy()
    pre = full_shared_data_audit(paths.shared)
    id_audit = sample_id_audit(paths.shared, [*input_views, *dynamic_views])
    environment = runtime_environment_audit(config)
    data_path = paths.output / "DATA_AUDIT" / "METRO_P60_C1_PRE_AUDIT.json"
    write_json(data_path, pre)
    diff_path = paths.output / "CODE_AUDIT" / "DIFF_FROM_SRU_V211_BASE.patch"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(
        _git(paths.project, "diff", "--binary", f"{SOURCE_COMMIT}..{head}") + "\n",
        encoding="utf-8",
    )
    checks = {
        "branch_exact": branch == RECOMMENDED_BRANCH,
        "source_commit_is_ancestor": ancestor,
        "source_tree_clean": not dirty,
        "active_head_exact": observed_heads == {ACTIVE_HEAD},
        "active_dataset_exact": config["active_datasets"] == ["metropt"],
        "input_views_present": len(input_views) > 0,
        "dynamic_views_present": len(dynamic_views) > 0,
        "sample_count_registry_present": len(sample_counts) == 16,
        "sample_id_splits_exact": all(
            item["split_column_exact"] for item in id_audit
        ),
        "write_shared_data_false": config["write_shared_data"] is False,
        "rebuild_or_resplit_false": config["rebuild_or_resplit_c1"] is False,
        "test_accessed_false": True,
        "ood_accessed_false": True,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "STOP_INHERITANCE_OR_DATA_AUDIT",
        "stage": "M0_INHERITANCE_DATA_AUDIT",
        "checks": checks,
        "git_head": head,
        "git_branch": branch,
        "git_status_porcelain": dirty,
        "source_commit": SOURCE_COMMIT,
        "config_sha256": sha256_file(paths.config_path),
        "theory_sha256": sha256_file(paths.theory_path),
        "historical_reference_sha256": sha256_file(paths.historical_reference_path),
        "data_pre_audit_path": str(data_path.relative_to(paths.output)),
        "data_pre_audit_sha256": sha256_file(data_path),
        "data_aggregate_sha256": pre["aggregate_sha256"],
        "sample_count_rows": sample_counts.to_dict(orient="records"),
        "sample_id_audit": id_audit,
        "runtime_environment": environment,
        "input_views": [view.relative_root.as_posix() for view in input_views],
        "dynamic_views": [view.relative_root.as_posix() for view in dynamic_views],
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.output / "FREEZE" / "M0_INHERITANCE_DATA_AUDIT.json", result)
    _run_status(
        paths,
        status=result["status"],
        stage="M0",
        branch=branch,
        commit=head,
        git_clean=not dirty,
        development_frozen=False,
        test_accessed=False,
        ood_accessed=False,
        data_hash=pre["aggregate_sha256"],
        config_hash=CONFIG_SHA256,
    )
    if result["status"] != "PASS":
        stop(paths, result["status"], result["checks"])
    return result


def run_m1(paths: MetroV211Paths) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", *M1_TESTS, "-q"]
    environment = os.environ.copy()
    source_root = str(paths.project / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    completed = subprocess.run(
        command,
        cwd=paths.project,
        env=environment,
        text=True,
        capture_output=True,
    )
    result = {
        "status": "PASS" if completed.returncode == 0 else "STOP_V211_REGRESSION_TEST_FAILED",
        "stage": "M1_REGRESSION_TESTS",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.output / "FREEZE" / "M1_REGRESSION_TESTS.json", result)
    _run_status(paths, status=result["status"], stage="M1")
    if completed.returncode:
        stop(paths, "STOP_V211_REGRESSION_TEST_FAILED")
    return result


def run_m2(paths: MetroV211Paths) -> dict[str, Any]:
    config = load_metro_config(paths.project)
    views = metro_p60_input_views(paths.shared)
    workers = effective_worker_count(config)
    memory = float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "20"))
    k_memory = float(os.environ.get("PRISM_V211_K_MEMORY_GIB_PER_WORKER", "20"))
    k_jobs = []
    k_paths = []
    for view in views:
        for channel in input_columns(paths.shared, view.head.task_id, view.proxy_policy):
            k_jobs.append((paths.shared, paths.project, paths.output, view, channel, PROTOCOL))
            k_paths.append(_result_path(paths.output, "K", view, channel))
    k_workers = effective_k_outer_workers(config, len(k_jobs))
    run_parallel(
        run_k_channel,
        k_jobs,
        k_workers,
        per_worker_gib=k_memory,
        label="PRISM_V211_METRO_M2_K",
    )
    k_results = _bind(paths, k_paths)
    c_paths = [_result_path(paths.output, "C", view) for view in views]
    run_parallel(
        run_c_view,
        [(paths.shared, paths.project, paths.output, view, PROTOCOL) for view in views],
        workers,
        per_worker_gib=memory,
        label="PRISM_V211_METRO_M2_C",
    )
    c_results = _bind(paths, c_paths)
    noncollapsed = all(
        item.get("status") == "PASS"
        and bool(item.get("input_path_preservation", {}).get("pass", False))
        for item in c_results
    )
    active_by_view = {
        view.proxy_policy: len(load_active_channels(paths.output, view)) for view in views
    }
    result = {
        "status": "PASS" if noncollapsed and all(active_by_view.values()) else "STOP_KC_INPUT_PATH_COLLAPSED",
        "stage": "M2_DEVELOPMENT_K_C",
        "k_jobs": len(k_results),
        "requested_k_workers": workers,
        "k_outer_task_workers": k_workers,
        "k_inner_candidate_workers": int(
            os.environ.get("PRISM_V211_K_INNER_WORKERS", "1")
        ),
        "k_memory_gib_per_worker": k_memory,
        "k_pass": sum(item.get("status") == "PASS" for item in k_results),
        "active_by_view": active_by_view,
        "c_views": len(c_results),
        "c_input_paths_noncollapsed": noncollapsed,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.output / "DEVELOPMENT" / "M2_KC_SUMMARY.json", result)
    _run_status(paths, status=result["status"], stage="M2")
    if result["status"] != "PASS":
        stop(paths, "STOP_KC_INPUT_PATH_COLLAPSED", result)
    return result


def run_m3(paths: MetroV211Paths) -> dict[str, Any]:
    config = load_metro_config(paths.project)
    views = metro_p60_input_views(paths.shared)
    result_paths = [_result_path(paths.output, "W", view) for view in views]
    run_parallel(
        run_w_view,
        [(paths.shared, paths.project, paths.output, view, PROTOCOL) for view in views],
        effective_worker_count(config),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "20")),
        label="PRISM_V211_METRO_M3_W",
    )
    results = _bind(paths, result_paths)
    expected = set(W_FAMILIES)
    candidates_ok = all(set(item.get("candidate_families_compared", ())) == expected for item in results)
    identity_ok = all(item.get("identity_equivalence", {}).get("pass") is True for item in results)
    joint_basis_ok = all(isinstance(item.get("joint_w_basis_contract"), dict) for item in results)
    status = "PASS"
    if not all(item.get("status") == "PASS" for item in results) or not candidates_ok or not joint_basis_ok:
        status = "STOP_W_CANDIDATES_NOT_ACTUALLY_COMPARED"
    elif not identity_ok:
        status = "STOP_IDENTITY_W_NOT_EQUIVALENT"
    summary = {
        "status": status,
        "stage": "M3_DEVELOPMENT_W",
        "views": len(results),
        "candidate_families_all_compared": candidates_ok,
        "identity_equivalence_pass": identity_ok,
        "joint_w_basis_frozen": joint_basis_ok,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.output / "DEVELOPMENT" / "W" / "SUMMARY.json", summary)
    _run_status(paths, status=status, stage="M3")
    if status != "PASS":
        stop(paths, status, summary)
    return summary


def run_m4(paths: MetroV211Paths) -> dict[str, Any]:
    config = load_metro_config(paths.project)
    views = metro_p60_dynamic_views(paths.shared)
    result_paths = [_result_path(paths.output, "A", view) for view in views]
    run_parallel(
        run_a_view,
        [(paths.shared, paths.project, paths.output, view, PROTOCOL) for view in views],
        effective_worker_count(config),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "20")),
        label="PRISM_V211_METRO_M4_A",
    )
    results = _bind(paths, result_paths)
    for view, a_result in zip(views, results):
        c_result = _read(_result_path(paths.output, "C", view))
        w_result = _read(_result_path(paths.output, "W", view))
        k_result = {
            "status": "PASS",
            "final_selected_candidate": c_result.get("active_channels", []),
            "input_path_preservation": c_result.get("input_path_preservation", {}),
        }
        card = build_physics_first_card(k_result, c_result, w_result, a_result)
        card.update(
            {
                "target_head": view.head.head_id,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "validation_prediction_path": a_result.get("prediction_path"),
            }
        )
        write_json(
            paths.output / "ASSEMBLY_CARDS" / view.head.head_id / view.availability_scenario / view.proxy_policy / "PF_ASSEMBLY_CARD.json",
            card,
        )
    passed = all(item.get("status") == "PASS" for item in results)
    summary = {
        "status": "PASS" if passed else "FAILED",
        "stage": "M4_DEVELOPMENT_A",
        "views": len(results),
        "pass": sum(item.get("status") == "PASS" for item in results),
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.output / "DEVELOPMENT" / "A" / "SUMMARY.json", summary)
    _run_status(paths, status=summary["status"], stage="M4")
    if not passed:
        stop(paths, "FAILED", summary)
    return summary


def run_m5(paths: MetroV211Paths) -> dict[str, Any]:
    config = load_metro_config(paths.project)
    views = metro_p60_dynamic_views(paths.shared)
    result_paths = [_result_path(paths.output, "JOINT", view) for view in views]
    run_parallel(
        run_joint_view,
        [(paths.shared, paths.project, paths.output, view, PROTOCOL) for view in views],
        effective_worker_count(config),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "20")),
        label="PRISM_V211_METRO_M5_JOINT",
    )
    results = _bind(paths, result_paths)
    gate_match = True
    routes_complete = True
    jointly_fit = True
    fold_protocol_complete = True
    original_support_only = True
    all_four_fold_losses = True
    protocol_mismatch = False
    for view, joint_result in zip(views, results):
        c_result = _read(_result_path(paths.output, "C", view))
        gate_match &= pf_and_joint_input_status_match(c_result, joint_result)
        protocol_mismatch |= (
            joint_result.get("status") == "STOP_JOINT_FOLD_PROTOCOL_MISMATCH"
        )
        fold_protocol_complete &= bool(
            joint_result.get("joint_fold_protocol_audit_pass")
        )
        original_support_only &= bool(
            joint_result.get("joint_fit_source")
            == "ORIGINAL_REGISTERED_INNER_TRAIN_SUPPORT"
            and joint_result.get("joint_evaluation_source")
            == "ORIGINAL_REGISTERED_INNER_VALIDATION_SUPPORT"
            and joint_result.get("nested_oof_training_used") is False
            and joint_result.get("w_physical_oof_used_as_training_pool") is False
        )
        all_four_fold_losses &= bool(
            joint_result.get("registered_inner_fold_count") == 4
            and joint_result.get("candidate_fold_loss_count") == 4
            and len(joint_result.get("joint_fold_protocol_audit", ())) == 4
        )
        routes_complete &= set(joint_result.get("route_materializations", {})) == set(JOINT_CANDIDATES)
        jointly_fit &= bool(joint_result.get("joint_w_coefficients_jointly_fitted"))
        if protocol_mismatch:
            continue
        card = build_joint_card(joint_result)
        card.update(
            {
                "target_head": view.head.head_id,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "validation_prediction_path": joint_result.get("prediction_path"),
            }
        )
        write_json(
            paths.output / "ASSEMBLY_CARDS" / view.head.head_id / view.availability_scenario / view.proxy_policy / "JOINT_ASSEMBLY_CARD.json",
            card,
        )
    passed = all(item.get("status") == "PASS" for item in results)
    corrected_model_gate_failed = bool(
        all(
            item.get("status")
            in {"PASS", "JOINT_OOF_PROTOCOL_CORRECTED_BUT_MODEL_GATE_FAILED"}
            for item in results
        )
        and any(
            item.get("status")
            == "JOINT_OOF_PROTOCOL_CORRECTED_BUT_MODEL_GATE_FAILED"
            for item in results
        )
    )
    status = "PASS"
    if protocol_mismatch or not fold_protocol_complete or not original_support_only or not all_four_fold_losses:
        status = "STOP_JOINT_FOLD_PROTOCOL_MISMATCH"
    elif corrected_model_gate_failed:
        status = "JOINT_OOF_PROTOCOL_CORRECTED_BUT_MODEL_GATE_FAILED"
    elif not gate_match:
        status = "STOP_PF_JOINT_INPUT_GATE_INCONSISTENT"
    elif not passed or not routes_complete or not jointly_fit:
        status = "STOP_JOINT_W_NOT_JOINTLY_FIT"
    summary = {
        "status": status,
        "stage": "M5_DEVELOPMENT_JOINT",
        "views": len(results),
        "pf_joint_gate_match": gate_match,
        "joint_fold_protocol_audit_pass": fold_protocol_complete,
        "original_registered_inner_support_only": original_support_only,
        "all_four_registered_fold_losses_present": all_four_fold_losses,
        "nested_oof_training_used": False,
        "w_physical_oof_used_as_training_pool": False,
        "all_four_routes_materialized": routes_complete,
        "w_coefficients_jointly_fitted": jointly_fit,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.output / "DEVELOPMENT" / "JOINT" / "SUMMARY.json", summary)
    _run_status(paths, status=status, stage="M5")
    if status != "PASS":
        stop(paths, status, summary)
    return summary


def _all_bound_results(paths: MetroV211Paths) -> list[dict[str, Any]]:
    files = sorted((paths.output / "DEVELOPMENT").glob("**/RESULT.json"))
    return [_read(path) for path in files if _read(path).get("status") == "PASS"]


def run_m6(paths: MetroV211Paths) -> dict[str, Any]:
    if paths.test_access_audit_path.exists():
        stop(paths, "STOP_TEST_OR_OOD_EARLY_ACCESS")
    config = load_metro_config(paths.project)
    pre_path = paths.output / "DATA_AUDIT" / "METRO_P60_C1_PRE_AUDIT.json"
    pre = _read(pre_path)
    post = full_shared_data_audit(paths.shared)
    comparison = compare_shared_data_audits(pre, post)
    post["comparison_to_pre"] = comparison
    post_path = paths.output / "DATA_AUDIT" / "METRO_P60_C1_POST_AUDIT.json"
    write_json(post_path, post)
    input_views = metro_p60_input_views(paths.shared)
    dynamic_views = metro_p60_dynamic_views(paths.shared)
    c_results = [_read(_result_path(paths.output, "C", view)) for view in input_views]
    w_results = [_read(_result_path(paths.output, "W", view)) for view in input_views]
    a_results = [_read(_result_path(paths.output, "A", view)) for view in dynamic_views]
    joint_results = [_read(_result_path(paths.output, "JOINT", view)) for view in dynamic_views]
    bound_results = _all_bound_results(paths)
    candidate_ids_ok = True
    try:
        for item in bound_results:
            assert_candidate_id_binding(item)
    except RuntimeError:
        candidate_ids_ok = False
    gates_match = all(
        pf_and_joint_input_status_match(c_result, joint_result)
        for c_result, joint_result in zip(c_results, joint_results)
    )
    checks = {
        "data_hash_unchanged": comparison["status"] == "PASS",
        "k_c_input_path_noncollapsed": all(
            bool(item.get("input_path_preservation", {}).get("pass")) for item in c_results
        ),
        "pf_joint_input_status_match": gates_match,
        "joint_fold_protocol_all_pass": all(
            bool(item.get("joint_fold_protocol_audit_pass"))
            for item in joint_results
        ),
        "joint_uses_original_registered_inner_support": all(
            item.get("joint_fit_source")
            == "ORIGINAL_REGISTERED_INNER_TRAIN_SUPPORT"
            and item.get("joint_evaluation_source")
            == "ORIGINAL_REGISTERED_INNER_VALIDATION_SUPPORT"
            and item.get("nested_oof_training_used") is False
            and item.get("w_physical_oof_used_as_training_pool") is False
            for item in joint_results
        ),
        "all_registered_joint_folds_present": all(
            item.get("registered_inner_fold_count") == 4
            and item.get("candidate_fold_loss_count") == 4
            and len(item.get("joint_fold_protocol_audit", ())) == 4
            for item in joint_results
        ),
        "w_candidates_actually_compared": all(
            set(item.get("candidate_families_compared", ())) == set(W_FAMILIES)
            for item in w_results
        ),
        "identity_equivalence_pass": all(
            item.get("identity_equivalence", {}).get("pass") is True for item in w_results
        ),
        "candidate_id_consistency_pass": candidate_ids_ok,
        "joint_w_jointly_fit": all(
            bool(item.get("joint_w_coefficients_jointly_fitted"))
            and set(item.get("route_materializations", {})) == set(JOINT_CANDIDATES)
            for item in joint_results
        ),
        "all_a_pass": all(item.get("status") == "PASS" for item in a_results),
        "test_accessed_false": True,
        "ood_accessed_false": True,
        "code_tree_clean": not _git(paths.project, "status", "--porcelain=v1"),
    }
    stop_mapping = (
        ("data_hash_unchanged", "STOP_DATA_BASE_MUTATED"),
        ("k_c_input_path_noncollapsed", "STOP_KC_INPUT_PATH_COLLAPSED"),
        ("pf_joint_input_status_match", "STOP_PF_JOINT_INPUT_GATE_INCONSISTENT"),
        ("joint_fold_protocol_all_pass", "STOP_JOINT_FOLD_PROTOCOL_MISMATCH"),
        ("joint_uses_original_registered_inner_support", "STOP_JOINT_FOLD_PROTOCOL_MISMATCH"),
        ("all_registered_joint_folds_present", "STOP_JOINT_FOLD_PROTOCOL_MISMATCH"),
        ("w_candidates_actually_compared", "STOP_W_CANDIDATES_NOT_ACTUALLY_COMPARED"),
        ("identity_equivalence_pass", "STOP_IDENTITY_W_NOT_EQUIVALENT"),
        ("candidate_id_consistency_pass", "STOP_CANDIDATE_ID_MISMATCH"),
        ("joint_w_jointly_fit", "STOP_JOINT_W_NOT_JOINTLY_FIT"),
    )
    failed = next((label for key, label in stop_mapping if not checks[key]), None)
    if failed is None and not all(checks.values()):
        failed = "FAILED"
    selections = []
    for view, c_result, w_result, a_result, joint_result in zip(
        dynamic_views, c_results, w_results, a_results, joint_results
    ):
        joint_selected = joint_result.get("final_selected_candidate")
        joint_selected_hyperparameters = joint_result.get(
            "route_local_selected", {}
        ).get(joint_selected)
        selections.append(
            {
                "view": view.relative_root.as_posix(),
                "active_channels": c_result.get("active_channels", []),
                "c_candidate_id": c_result.get("final_selected_candidate_id"),
                "c_family": c_result.get("selected_family"),
                "c_ridge_alpha": c_result.get("selected_alpha"),
                "pf_w_family": w_result.get("w_contract", {}).get("family"),
                "w_candidate_id": w_result.get("final_selected_candidate_id"),
                "pf_ablation_w_candidate": w_result.get(
                    "pf_ablation_w_candidate"
                ),
                "pf_ablation_selection_eligible": False,
                "a_family": a_result.get("a_contract", {}).get("family"),
                "a_candidate_id": a_result.get("final_selected_candidate_id"),
                "pf_selected_route": a_result.get("pf_selected_route"),
                "joint_selected": joint_selected,
                "joint_candidate_id": joint_result.get("final_selected_candidate_id"),
                "joint_route_hyperparameters": joint_result.get("route_local_selected", {}),
                "joint_selected_hyperparameters": joint_selected_hyperparameters,
                "joint_alpha": (
                    joint_selected_hyperparameters[1]
                    if joint_selected_hyperparameters is not None
                    else None
                ),
                "joint_k_over_a_ratio": (
                    joint_selected_hyperparameters[2]
                    if joint_selected_hyperparameters is not None
                    else None
                ),
                "joint_w_over_a_ratio": (
                    joint_selected_hyperparameters[3]
                    if joint_selected_hyperparameters is not None
                    else None
                ),
                "joint_fold_protocol_audit": joint_result.get(
                    "joint_fold_protocol_audit", []
                ),
                "joint_fit_source": joint_result.get("joint_fit_source"),
                "joint_evaluation_source": joint_result.get(
                    "joint_evaluation_source"
                ),
                "candidate_ids": [
                    item.get("candidate_id")
                    for item in joint_result.get("candidate_registry", [])
                ],
            }
        )
    decision = {
        "status": "PASS" if failed is None else failed,
        "protocol_id": PROTOCOL_ID,
        "evidence_class": EVIDENCE_CLASS,
        "stage": "M6_DEVELOPMENT_FREEZE_DECISION",
        "checks": checks,
        "development_selections": selections,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.development_decision_path, decision)
    if failed is not None:
        stop(paths, failed, decision)
    frozen_config = paths.output / "FREEZE" / "METRO_P60_V212_CONFIG_FROZEN.json"
    frozen_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.config_path, frozen_config)
    decision_sha256 = sha256_file(paths.development_decision_path)
    pending_candidate_ids = []
    for view, selection in zip(dynamic_views, selections, strict=True):
        common = {
            "view": view.relative_root.as_posix(),
            "development_decision_sha256": decision_sha256,
        }
        identifiers = {
            name: stable_candidate_id(
                "FINAL", {**common, "candidate": name}
            )
            for name in ("KC", "KCW", "KCA", "KCWA", *JOINT_CANDIDATES)
        }
        identifiers["PF_SELECTED"] = identifiers[
            str(selection["pf_selected_route"])
        ]
        identifiers["J_SELECTED"] = identifiers[
            str(selection["joint_selected"])
        ]
        pending_candidate_ids.append(
            {
                "view": view.relative_root.as_posix(),
                "candidate_ids": identifiers,
            }
        )
    manifest = {
        "status": "METRO_P60_V2_1_2_DEVELOPMENT_FROZEN",
        "protocol_id": PROTOCOL_ID,
        "evidence_class": EVIDENCE_CLASS,
        "code_commit": _git(paths.project, "rev-parse", "HEAD"),
        "git_branch": _git(paths.project, "branch", "--show-current"),
        "git_clean": True,
        "config_sha256": sha256_file(paths.config_path),
        "frozen_config_sha256": sha256_file(frozen_config),
        "development_decision_sha256": decision_sha256,
        "data_pre_audit_sha256": sha256_file(pre_path),
        "data_post_audit_sha256": sha256_file(post_path),
        "data_aggregate_sha256": post["aggregate_sha256"],
        "test_accessed": False,
        "ood_accessed": False,
        "materialize_after_freeze": config["post_freeze_materialized_candidates"],
        "runtime_parallelism": runtime_parallelism_audit(config),
        "pending_materialization_candidate_ids": pending_candidate_ids,
        "m0_audit_sha256": sha256_file(
            paths.output / "FREEZE" / "M0_INHERITANCE_DATA_AUDIT.json"
        ),
        "frozen_at_unix": time.time(),
    }
    write_json(paths.development_freeze_path, manifest)
    _run_status(
        paths,
        status="PASS",
        stage="M6",
        development_frozen=True,
        test_accessed=False,
        ood_accessed=False,
        code_hash=manifest["code_commit"],
        data_hash=manifest["data_aggregate_sha256"],
    )
    return manifest


def run_stage(stage: str, paths: MetroV211Paths) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(stage)
    functions = {
        "m0": run_m0,
        "m1": run_m1,
        "m2": run_m2,
        "m3": run_m3,
        "m4": run_m4,
        "m5": run_m5,
        "m6": run_m6,
    }
    if stage in functions:
        return functions[stage](paths)
    if stage == "m7":
        from .v211_metro_final import run_m7

        return run_m7(paths)
    from .v211_metro_reporting import run_m8

    return run_m8(paths)
