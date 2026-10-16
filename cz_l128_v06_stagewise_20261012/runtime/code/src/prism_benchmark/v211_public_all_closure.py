from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .cpu_data import SAMPLE_RUNTIME_COLUMNS, ViewSpec, load_samples, sha256_file
from .stage0 import write_json
from .v211_public_all_audit import (
    audit_k_stage,
    shared_development_metadata_sha256,
    write_k_audit,
)
from .v211_public_all_baselines import SupportRequirement, apply_common_requirements
from .v211_public_all_config import (
    EVIDENCE_CLASS,
    EXECUTION_BRANCH,
    PRIMARY_TASKS,
    PROTOCOL_ID,
    SOURCE_BRANCH,
    SOURCE_COMMIT,
    PublicAllPaths,
    load_public_all_descriptor,
)
from .v211_joint_stability_config import theory_path
from .v211_public_all_views import public_all_dynamic_views, public_all_input_views
from .v211_support import SUPPORT_COLUMNS, SUPPORT_CONTRACT, support_id_hash


METADATA_COLUMNS = [
    column for column in SAMPLE_RUNTIME_COLUMNS if column != "y_true"
]
FINAL_SUCCESS_STATUSES = {
    "PASS",
    "NOT_RUN_IMPLEMENTATION_ABSENT",
    "NOT_RUN_PROTOCOL_INCOMPATIBLE",
}
LEGAL_JOINT_DEVELOPMENT_STATUSES = {
    "PASS",
    "JOINT_STABILITY_STABILITY_IMPROVED_BUT_NOT_SUPPORTED",
    "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT",
}


@dataclass(frozen=True)
class FilteredView:
    """A view whose sample parquet files live under a frozen support root."""

    original: ViewSpec
    support_root: Path

    @property
    def head(self):
        return self.original.head

    @property
    def information_set(self) -> str:
        return self.original.information_set

    @property
    def availability_scenario(self) -> str:
        return self.original.availability_scenario

    @property
    def proxy_policy(self) -> str:
        return self.original.proxy_policy

    @property
    def relative_root(self) -> Path:
        return self.support_root / self.original.relative_root


def _git(project: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=project, text=True
    ).strip()


def _result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _sha_if_file(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _prism_result(output: Path, stage: str, view: ViewSpec) -> dict[str, Any] | None:
    root = output / "DEVELOPMENT" / stage
    if stage in {"C", "W"}:
        path = root / view.head.head_id / view.proxy_policy / "RESULT.json"
    else:
        path = (
            root
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "RESULT.json"
        )
    return _result(path)


def _baseline_result(output: Path, family: str, model: str, view: ViewSpec) -> dict[str, Any] | None:
    return _result(
        output
        / "BASELINE_DEVELOPMENT"
        / family
        / "PREDICTIONS"
        / model
        / view.relative_root
        / "RESULT.json"
    )


def _append_input_requirement(
    requirements: list[SupportRequirement], value: Any
) -> None:
    if value is None:
        return
    history = int(value)
    if history > 0:
        requirements.append(SupportRequirement(input_history_steps=history))


def _append_target_requirement(
    requirements: list[SupportRequirement], profile: Any
) -> None:
    if profile is None:
        return
    delta, history = (int(item) for item in profile)
    if delta > 0 and history > 0:
        requirements.append(
            SupportRequirement(
                target_delta_steps=delta,
                target_history_steps=history,
            )
        )


def _append_arx_requirement(
    requirements: list[SupportRequirement], profile: Any
) -> None:
    if profile is None:
        return
    delta, history = (int(item) for item in profile)
    if delta > 0 and history > 0:
        requirements.append(
            SupportRequirement(
                input_history_steps=history,
                target_delta_steps=delta,
                target_history_steps=history,
            )
        )


def _selected_k_histories(output: Path, view: ViewSpec) -> list[int]:
    root = output / "DEVELOPMENT" / "K" / view.head.head_id / view.proxy_policy
    histories = []
    for path in sorted(root.glob("*/RESULT.json")):
        result = _result(path)
        if not result or result.get("status") != "PASS" or not result.get("active"):
            continue
        value = result.get("selected_profile_history_steps")
        if value is not None:
            histories.append(int(value))
    return histories


def view_support_requirements(
    paths: PublicAllPaths, view: ViewSpec
) -> tuple[SupportRequirement, ...]:
    requirements: list[SupportRequirement] = []
    for history in _selected_k_histories(paths.output, view):
        _append_input_requirement(requirements, history)
    c_result = _prism_result(paths.output, "C", view)
    if c_result:
        for history in c_result.get("active_selected_k_histories", {}).values():
            _append_input_requirement(requirements, history)
    input_models = (
        ("C2", "DPLS"),
        ("C3", "PARALLEL_HAMMERSTEIN"),
        ("C3", "HAMMERSTEIN_WIENER"),
    )
    for family, model in input_models:
        result = _baseline_result(paths.output, family, model, view)
        if not result or result.get("status") != "PASS":
            continue
        selection = result.get("selection", {})
        if model == "DPLS":
            _append_input_requirement(requirements, selection.get("selected_history"))
        else:
            profile = selection.get("selected_profile")
            if isinstance(profile, Sequence) and len(profile) >= 2:
                _append_input_requirement(requirements, profile[1])
    if view.information_set == "dynamic":
        a_result = _prism_result(paths.output, "A", view)
        if a_result and a_result.get("status") == "PASS":
            contract = a_result.get("a_contract", {})
            if contract.get("family") != "EXACT_ZERO":
                _append_target_requirement(requirements, contract.get("profile"))
        joint_result = _prism_result(paths.output, "JOINT", view)
        if joint_result and joint_result.get("status") == "PASS":
            _append_target_requirement(requirements, joint_result.get("ar_profile"))
        ar = _baseline_result(paths.output, "C3", "AR", view)
        if ar and ar.get("status") == "PASS":
            _append_target_requirement(
                requirements, ar.get("selection", {}).get("selected_profile")
            )
        for model in ("ARX", "LINEAR_NARX"):
            result = _baseline_result(paths.output, "C3", model, view)
            if result and result.get("status") == "PASS":
                _append_arx_requirement(
                    requirements,
                    result.get("selection", {}).get("selected_profile"),
                )
    unique = tuple(sorted(set(requirements)))
    return unique or (SupportRequirement(),)


def _support_frame(shared: Path, view: ViewSpec, split: str) -> pd.DataFrame:
    path = shared / "sample_ids" / view.relative_root / f"{split}.parquet"
    columns = [*METADATA_COLUMNS, *SUPPORT_COLUMNS]
    return pd.read_parquet(path, columns=list(dict.fromkeys(columns)))


def _support_detail(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "support_hash": support_id_hash(frame),
        "support_contract": SUPPORT_CONTRACT,
    }


def build_common_support(paths: PublicAllPaths) -> dict[str, Any]:
    views = [*public_all_input_views(paths.shared), *public_all_dynamic_views(paths.shared)]
    records: list[dict[str, Any]] = []
    for view in views:
        requirements = view_support_requirements(paths, view)
        split_details: dict[str, Any] = {}
        for split in ("train", "validation", "test", "ood"):
            path = paths.shared / "sample_ids" / view.relative_root / f"{split}.parquet"
            if not path.is_file():
                continue
            frame = _support_frame(paths.shared, view, split)
            common = apply_common_requirements(frame, requirements)
            split_details[split] = _support_detail(common)
            split_details[split]["source_rows"] = int(len(frame))
        records.append(
            {
                "target_head": view.head.head_id,
                "dataset": view.head.dataset,
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "requirements": [item.to_json() for item in requirements],
                "splits": split_details,
            }
        )
    result = {
        "status": "PASS",
        "stage": "F2_TASK_LEADERBOARD_COMMON_SUPPORT",
        "support_contract": SUPPORT_CONTRACT,
        "views": records,
        "test_y_read": False,
        "ood_y_read": False,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.leaderboard_support_path, result)
    return result


def _stage_summaries(paths: PublicAllPaths) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for stage in ("K", "C", "W", "A", "JOINT"):
        path = paths.output / "DEVELOPMENT" / stage / "SUMMARY.json"
        if path.is_file():
            values[stage] = _result(path)
    baseline = paths.output / "BASELINE_DEVELOPMENT" / "SUMMARY.json"
    if baseline.is_file():
        values["BASELINES"] = _result(baseline)
    return values


def _formal_routes(
    view: ViewSpec,
    a_result: Mapping[str, Any] | None,
    joint_result: Mapping[str, Any] | None,
) -> list[str]:
    if view.information_set == "input_only":
        return ["INPUT_ONLY"]
    if not a_result or a_result.get("status") != "PASS":
        raise RuntimeError(f"dynamic PF development is not PASS: {view.relative_root}")
    if not joint_result:
        raise RuntimeError(f"Joint development result is missing: {view.relative_root}")
    joint_status = str(joint_result.get("status"))
    if joint_status not in LEGAL_JOINT_DEVELOPMENT_STATUSES:
        raise RuntimeError(
            f"unexpected Joint development status for {view.relative_root}: "
            f"{joint_status}"
        )
    routes = ["PHYSICS_FIRST"]
    if joint_status == "PASS":
        routes.append("JOINT")
    return routes


def _require_baseline_results(
    view: ViewSpec, results: Mapping[str, Mapping[str, Any] | None]
) -> None:
    common = {"MEAN", "PERSISTENCE", "SEASONAL_PERSISTENCE"}
    if view.information_set == "input_only":
        required = common | {
            "RIDGE",
            "PLS",
            "RBF_SVR",
            "XGBOOST",
            "DPLS",
            "PARALLEL_HAMMERSTEIN",
            "HAMMERSTEIN_WIENER",
        }
    else:
        required = common | {"AR", "ARX", "LINEAR_NARX", "N4SID"}
    invalid = {
        model: None if results.get(model) is None else results[model].get("status")
        for model in sorted(required)
        if results.get(model) is None
        or results[model].get("status") not in FINAL_SUCCESS_STATUSES
    }
    if invalid:
        raise RuntimeError(
            f"baseline development is incomplete for {view.relative_root}: {invalid}"
        )


def write_development_freeze(paths: PublicAllPaths) -> dict[str, Any]:
    descriptor = load_public_all_descriptor(paths.project)
    k_audit = write_k_audit(paths)
    if k_audit["status"] != "PASS":
        raise RuntimeError("K Native Support audit did not fully pass")
    summaries = _stage_summaries(paths)
    required = {"K", "C", "W", "A", "JOINT", "BASELINES"}
    if not required.issubset(summaries):
        raise RuntimeError(f"development summaries missing: {required - summaries.keys()}")
    for stage in ("K", "C", "W", "A"):
        summary = summaries[stage]
        if summary.get("status") != "PASS" or summary.get("pass") != summary.get("jobs"):
            raise RuntimeError(f"development stage did not fully pass: {stage}")
    execution_branch = _git(paths.project, "branch", "--show-current")
    if execution_branch != EXECUTION_BRANCH:
        raise RuntimeError(
            f"execution branch mismatch: {execution_branch} != {EXECUTION_BRANCH}"
        )
    common = build_common_support(paths)
    views = [*public_all_input_views(paths.shared), *public_all_dynamic_views(paths.shared)]
    view_records: list[dict[str, Any]] = []
    for view in views:
        c_result = _prism_result(paths.output, "C", view)
        w_result = _prism_result(paths.output, "W", view)
        a_result = (
            _prism_result(paths.output, "A", view)
            if view.information_set == "dynamic"
            else None
        )
        joint_result = (
            _prism_result(paths.output, "JOINT", view)
            if view.information_set == "dynamic"
            else None
        )
        baseline_results = {
            model: _baseline_result(paths.output, family, model, view)
            for family, model in (
                ("C2", "MEAN"),
                ("C2", "PERSISTENCE"),
                ("C2", "SEASONAL_PERSISTENCE"),
                ("C2", "RIDGE"),
                ("C2", "PLS"),
                ("C2", "RBF_SVR"),
                ("C2", "XGBOOST"),
                ("C2", "DPLS"),
                ("C3", "PARALLEL_HAMMERSTEIN"),
                ("C3", "HAMMERSTEIN_WIENER"),
                ("C3", "AR"),
                ("C3", "ARX"),
                ("C3", "LINEAR_NARX"),
                ("C3", "N4SID"),
            )
        }
        if not c_result or c_result.get("status") != "PASS":
            raise RuntimeError(f"C development is not PASS: {view.relative_root}")
        if not w_result or w_result.get("status") != "PASS":
            raise RuntimeError(f"W development is not PASS: {view.relative_root}")
        formal_routes = _formal_routes(view, a_result, joint_result)
        _require_baseline_results(view, baseline_results)
        view_records.append(
            {
                "dataset": view.head.dataset,
                "task_id": view.head.task_id,
                "target_head": view.head.head_id,
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "c_status": None if c_result is None else c_result.get("status"),
                "c_family": None if c_result is None else c_result.get("selected_family"),
                "c_active_channels": []
                if c_result is None
                else c_result.get("active_channels", []),
                "w_status": None if w_result is None else w_result.get("status"),
                "w_family": None
                if w_result is None
                else w_result.get("w_contract", {}).get("family"),
                "a_status": None if a_result is None else a_result.get("status"),
                "a_family": None
                if a_result is None
                else a_result.get("a_contract", {}).get("family"),
                "joint_status": None
                if joint_result is None
                else joint_result.get("status"),
                "joint_route": None
                if joint_result is None
                else joint_result.get("selected_candidate"),
                "joint_representation": None
                if joint_result is None
                else joint_result.get("selected_k_representation"),
                "joint_predictive_eta": None
                if joint_result is None
                else joint_result.get("selected_predictive_eta"),
                "joint_numerical_alpha": None
                if joint_result is None
                else joint_result.get("selected_numerical_alpha"),
                "pf_status": None if a_result is None else a_result.get("status"),
                "pf_route": None
                if a_result is None
                else a_result.get("pf_selected_route"),
                "formal_routes": formal_routes,
                "baseline_statuses": {
                    model: None if value is None else value.get("status")
                    for model, value in baseline_results.items()
                },
            }
        )
    descriptor_paths = {
        key: paths.project / str(value["path"])
        for key, value in (
            ("algorithm_config", descriptor["algorithm_config"]),
            ("joint_config", descriptor["joint_config"]),
            ("sru_patch", descriptor["sru_patch"]),
            ("cpu_model_freeze", descriptor["cpu_model_freeze"]),
        )
    }
    raw_audit_path = paths.run_root / "PUBLIC_ALL_RAW_DATA_REAUDIT.json"
    c1_audit_path = paths.run_root / "C1_NATIVE_SUPPORT_AUDIT.json"
    sample_registry_path = paths.shared / "SAMPLE_ID_REGISTRY.json"
    dataset_hashes_path = paths.shared / "DATASET_HASHES.json"
    manifest = {
        "status": "FROZEN",
        "stage": "F3_GLOBAL_DEVELOPMENT_FREEZE",
        "protocol_id": PROTOCOL_ID,
        "evidence_class": EVIDENCE_CLASS,
        "source_branch": SOURCE_BRANCH,
        "source_commit": SOURCE_COMMIT,
        "generating_commit": _git(paths.project, "rev-parse", "HEAD"),
        "execution_branch": execution_branch,
        "canonical_theory_sha256": sha256_file(theory_path(paths.project)),
        "support_contract": SUPPORT_CONTRACT,
        "primary_tasks": sorted(PRIMARY_TASKS),
        "active_datasets": list(descriptor["active_datasets"]),
        "config_sha256": descriptor["config_sha256"],
        "algorithm_config_sha256": descriptor["algorithm_config"]["sha256"],
        "joint_config_sha256": descriptor["joint_config"]["sha256"],
        "sru_patch_sha256": descriptor["sru_patch"]["sha256"],
        "cpu_model_freeze_sha256": descriptor["cpu_model_freeze"]["sha256"],
        "frozen_config_paths": {
            key: {
                "path": str(path.relative_to(paths.project)),
                "sha256": sha256_file(path),
            }
            for key, path in descriptor_paths.items()
        },
        "raw_hash_audit_path": str(raw_audit_path),
        "raw_hash_audit_sha256": _sha_if_file(raw_audit_path),
        "c1_native_support_audit_path": str(c1_audit_path),
        "c1_native_support_audit_sha256": _sha_if_file(c1_audit_path),
        "shared_dataset_hashes_sha256": _sha_if_file(dataset_hashes_path),
        "sample_id_registry_sha256": _sha_if_file(sample_registry_path),
        "shared_development_metadata_sha256": shared_development_metadata_sha256(
            paths.shared
        ),
        "k_audit_sha256": sha256_file(paths.freeze / "K_NATIVE_SUPPORT_AUDIT.json")
        if (paths.freeze / "K_NATIVE_SUPPORT_AUDIT.json").is_file()
        else None,
        "common_support_sha256": sha256_file(paths.leaderboard_support_path),
        "development_summaries": summaries,
        "views": view_records,
        "primary_head_count": len(PRIMARY_TASKS),
        "active_dataset_count": len(descriptor["active_datasets"]),
        "test_accessed": False,
        "ood_accessed": False,
        "baseline_test_metrics_exposed_to_selection": False,
    }
    paths.freeze.mkdir(parents=True, exist_ok=True)
    write_json(paths.development_freeze_path, manifest)
    return manifest


def materialize_filtered_view(
    paths: PublicAllPaths,
    view: ViewSpec,
    record: Mapping[str, Any],
    *,
    splits: Iterable[str] = ("train", "validation", "test"),
) -> FilteredView:
    support_root = paths.run_root / "final_support" / "sample_ids"
    filtered = FilteredView(view, support_root)
    destination = paths.shared / filtered.relative_root
    for split in splits:
        source = paths.shared / "sample_ids" / view.relative_root / f"{split}.parquet"
        if not source.is_file():
            continue
        frame = pd.read_parquet(
            source,
            columns=list(dict.fromkeys([*SAMPLE_RUNTIME_COLUMNS, *SUPPORT_COLUMNS])),
        )
        requirements = tuple(
            SupportRequirement(**item) for item in record.get("requirements", ())
        )
        common = apply_common_requirements(frame, requirements)
        target = destination / f"{split}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        common.to_parquet(target, index=False, compression="zstd")
    return filtered


def common_support_record(
    paths: PublicAllPaths, view: ViewSpec
) -> dict[str, Any]:
    payload = json.loads(paths.leaderboard_support_path.read_text(encoding="utf-8"))
    for item in payload["views"]:
        if (
            item["target_head"] == view.head.head_id
            and item["information_set"] == view.information_set
            and item["availability_scenario"] == view.availability_scenario
            and item["proxy_policy"] == view.proxy_policy
        ):
            return item
    raise KeyError(view.relative_root)


def _has_registered_split(paths: PublicAllPaths, view: ViewSpec, split: str) -> bool:
    return (
        paths.shared / "sample_ids" / view.relative_root / f"{split}.parquet"
    ).is_file()


def run_public_all_test(paths: PublicAllPaths) -> dict[str, Any]:
    """Materialize frozen test and registered OOD predictions in one access stage."""
    freeze = json.loads(paths.development_freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("test access requires a global development freeze")
    if paths.test_access_audit_path.is_file():
        raise RuntimeError("public-all test/OOD access audit already exists")
    paths.final.mkdir(parents=True, exist_ok=True)
    started = time.time()
    descriptor = load_public_all_descriptor(paths.project)
    input_views = list(public_all_input_views(paths.shared))
    dynamic_views = list(public_all_dynamic_views(paths.shared))
    all_views = [*input_views, *dynamic_views]
    ood_input_views = [
        view for view in input_views if _has_registered_split(paths, view, "ood")
    ]
    ood_dynamic_views = [
        view for view in dynamic_views if _has_registered_split(paths, view, "ood")
    ]
    ood_views = [*ood_input_views, *ood_dynamic_views]
    from .v211_public_all_materialization import (
        materialize_dynamic_prism_view,
        materialize_input_prism_view,
        preflight_public_all_materialization,
    )
    from .v211_public_all_baseline_materialization import materialize_baseline_view

    materialization_preflight = preflight_public_all_materialization(
        paths, dynamic_views
    )
    write_json(
        paths.test_access_audit_path,
        {
            "status": "TEST_OOD_ACCESS_STARTED",
            "stage": "T1_PUBLIC_ALL_TEST_OOD_ACCESS",
            "first_access_timestamp": started,
            "generating_commit": freeze.get("generating_commit"),
            "freeze_sha256": sha256_file(paths.development_freeze_path),
            "shared_sha256": freeze.get("shared_development_metadata_sha256"),
            "config_sha": descriptor.get("config_sha256"),
            "theory_sha": freeze.get("canonical_theory_sha256"),
            "registered_ood_views": len(ood_views),
            "test_accessed": True,
            "ood_accessed": False,
            "test_y_read": False,
            "ood_y_read": False,
            "materialization_contract_preflight": materialization_preflight,
        },
    )

    def write_baseline_summary(
        split: str, baseline_audits: list[dict[str, Any]]
    ) -> None:
        split_upper = split.upper()
        write_json(
            paths.final
            / f"baseline_{split}_predictions"
            / f"BASELINE_{split_upper}_RESULT.json",
            {
                "status": (
                    "PASS"
                    if baseline_audits
                    and all(
                        item.get("status") in FINAL_SUCCESS_STATUSES
                        for item in baseline_audits
                    )
                    else "FAILED"
                ),
                "stage": f"T1_PUBLIC_ALL_BASELINE_{split_upper}_ACCESS",
                "models": baseline_audits,
                "test_accessed": split == "test",
                "ood_accessed": split == "ood",
            },
        )

    audits: list[dict[str, Any]] = []
    test_y_read = False
    ood_accessed = False
    ood_y_read = False
    try:
        test_y_read = True
        for view in input_views:
            audits.extend(materialize_input_prism_view(paths, view, split="test"))
        for view in dynamic_views:
            audits.extend(materialize_dynamic_prism_view(paths, view, split="test"))
        baseline_test_audits: list[dict[str, Any]] = []
        for view in all_views:
            baseline_test_audits.extend(
                materialize_baseline_view(paths, view, split="test")
            )
        write_baseline_summary("test", baseline_test_audits)
        audits.extend(baseline_test_audits)

        if ood_views:
            ood_accessed = True
            ood_y_read = True
            for view in ood_input_views:
                audits.extend(materialize_input_prism_view(paths, view, split="ood"))
            for view in ood_dynamic_views:
                audits.extend(materialize_dynamic_prism_view(paths, view, split="ood"))
            baseline_ood_audits: list[dict[str, Any]] = []
            for view in ood_views:
                baseline_ood_audits.extend(
                    materialize_baseline_view(paths, view, split="ood")
                )
            write_baseline_summary("ood", baseline_ood_audits)
            audits.extend(baseline_ood_audits)
    except Exception as error:
        write_json(
            paths.final / "LOCKBOX_ACCESSED_RUNTIME_FAILURE.json",
            {
                "status": "LOCKBOX_ACCESSED_RUNTIME_FAILURE",
                "stage": "T1_PUBLIC_ALL_TEST_OOD_ACCESS",
                "error_type": type(error).__name__,
                "error": str(error),
                "test_accessed": True,
                "ood_accessed": ood_accessed,
                "test_y_read": test_y_read,
                "ood_y_read": ood_y_read,
                "materialization_contract_preflight": materialization_preflight,
            },
        )
        raise
    status_counts: dict[str, int] = {}
    for item in audits:
        status = str(item.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    result = {
        "status": (
            "PASS"
            if audits
            and all(
                item.get("status") in FINAL_SUCCESS_STATUSES for item in audits
            )
            else "FAILED"
        ),
        "stage": "T1_PUBLIC_ALL_TEST_OOD_ACCESS",
        "models": audits,
        "status_counts": status_counts,
        "first_access_timestamp": started,
        "freeze_sha256": sha256_file(paths.development_freeze_path),
        "registered_ood_views": len(ood_views),
        "test_accessed": True,
        "ood_accessed": ood_accessed,
        "test_y_read": test_y_read,
        "ood_y_read": ood_y_read,
        "materialization_contract_preflight": materialization_preflight,
    }
    write_json(paths.test_access_audit_path, result)
    return result
