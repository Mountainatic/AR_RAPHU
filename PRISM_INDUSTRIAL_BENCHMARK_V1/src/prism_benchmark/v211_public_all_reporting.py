from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cpu_data import HeadSpec, ViewSpec, load_heads, sha256_file
from .cpu_selection import regression_metrics
from .v21_reporting import _align_pair, _moving_block_means
from .v2_views import evaluation_level
from .v211_public_all_closure import common_support_record
from .v211_public_all_config import (
    EXECUTION_BRANCH,
    PublicAllPaths,
    SOURCE_BRANCH,
    SOURCE_COMMIT,
    load_public_all_descriptor,
)
from .v211_public_all_views import public_all_dynamic_views, public_all_input_views
from .v211_joint_stability_config import theory_path
from .v211_support import SUPPORT_CONTRACT, load_native_samples, support_id_hash


BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 20260815
PACKAGE_NAME = "PRISM_V2_1_1_NATIVE_SUPPORT_PUBLIC_ALL_RESULTS_bundle"
REPAIR_MANIFEST_NAME = "POST_FREEZE_MATERIALIZATION_REPAIR.json"
REUSED_ARTIFACT_MANIFEST_NAME = "REUSED_DEVELOPMENT_ARTIFACT_MANIFEST.json"
REPAIR_EVIDENCE_CLASS = (
    "POST_LOCKBOX_MATERIALIZATION_REPAIR_WITH_FROZEN_DEVELOPMENT_REUSE"
)
INPUT_PRISM_MODELS = {"PRISM_V2_1_1_K_C_W"}
DYNAMIC_PRISM_MODELS = {
    "PRISM_V2_1_1_PHYSICS_FIRST",
    "PRISM_V2_1_1_JOINT_KWA",
}
INPUT_BASELINE_MODELS = {
    "MEAN",
    "PERSISTENCE",
    "SEASONAL_PERSISTENCE",
    "RIDGE",
    "PLS",
    "RBF_SVR",
    "XGBOOST",
    "DPLS",
    "PARALLEL_HAMMERSTEIN",
    "HAMMERSTEIN_WIENER",
}
DYNAMIC_BASELINE_MODELS = {
    "MEAN",
    "PERSISTENCE",
    "SEASONAL_PERSISTENCE",
    "AR",
    "ARX",
    "LINEAR_NARX",
    "N4SID",
}
FINAL_SUCCESS_STATUSES = {
    "PASS",
    "NOT_RUN_IMPLEMENTATION_ABSENT",
    "NOT_RUN_PROTOCOL_INCOMPATIBLE",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _repair_manifest(paths: PublicAllPaths) -> dict[str, Any]:
    path = paths.freeze / REPAIR_MANIFEST_NAME
    if not path.is_file():
        return {}
    value = _read_json(path)
    if value.get("status") != "ACCEPTED_AUDITED_REUSE":
        raise RuntimeError("materialization repair manifest is not accepted")
    if value.get("post_test_reselection") is not False:
        raise RuntimeError("materialization repair cannot include reselection")
    return value


def _raw_audit_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    summary = value.get("summary", {})
    dataset_status = value.get("dataset_status", {})
    files = value.get("files", ())
    datasets_total = int(summary.get("datasets_total", len(dataset_status)))
    datasets_pass = int(
        summary.get(
            "datasets_pass",
            sum(status == "PASS" for status in dataset_status.values()),
        )
    )
    files_total = int(summary.get("files_total", len(files)))
    files_pass = int(
        summary.get("files_pass", sum(item.get("match") is True for item in files))
    )
    return {
        "datasets": datasets_total,
        "datasets_pass": datasets_pass,
        "files": files_total,
        "files_pass": files_pass,
        "pass": (
            value.get("status") == "PASS"
            and datasets_pass == datasets_total
            and files_pass == files_total
        ),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _head_map(shared: Path) -> dict[str, HeadSpec]:
    return {head.head_id: head for head in load_heads(shared, primary_only=False)}


def _view_key(view: ViewSpec) -> tuple[str, str, str, str]:
    return (
        view.head.head_id,
        view.information_set,
        view.availability_scenario,
        view.proxy_policy,
    )


def _view_role(shared: Path, view: ViewSpec) -> str:
    return (
        "primary"
        if evaluation_level(view, shared) == "LEVEL_B_PRIMARY_EXPLORATORY"
        else "secondary"
    )


def _all_view_map(shared: Path) -> dict[tuple[str, str, str, str], ViewSpec]:
    views = [*public_all_input_views(shared), *public_all_dynamic_views(shared)]
    return {_view_key(view): view for view in views}


def _prediction_path(paths: PublicAllPaths, audit: Mapping[str, Any]) -> Path:
    value = Path(str(audit["prediction_path"]))
    return value if value.is_absolute() else paths.run_root / value


def _read_prediction(paths: PublicAllPaths, audit: Mapping[str, Any]) -> pd.DataFrame:
    path = _prediction_path(paths, audit)
    frame = pd.read_parquet(path)
    required = {"sample_id", "entity_id", "origin", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"prediction artifact lacks columns {sorted(missing)}")
    return frame


def _audit_key(audit: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(audit.get("target_head")),
        str(audit.get("information_set")),
        str(audit.get("availability_scenario")),
        str(audit.get("proxy_policy")),
        str(audit.get("model")),
        str(audit.get("split")),
    )


def _base_row(
    audit: Mapping[str, Any],
    heads: Mapping[str, HeadSpec],
    view_map: Mapping[tuple[str, str, str, str], ViewSpec],
) -> dict[str, Any]:
    head_id = str(audit.get("target_head"))
    key = (
        head_id,
        str(audit.get("information_set")),
        str(audit.get("availability_scenario")),
        str(audit.get("proxy_policy")),
    )
    view = view_map.get(key)
    head = heads.get(head_id)
    return {
        "status": str(audit.get("status", "MISSING")),
        "dataset": str(audit.get("dataset", head.dataset if head else "")),
        "task_id": head.task_id if head else head_id.split("__", 1)[0],
        "target_head": head_id,
        "target": head.target if head else None,
        "information_set": str(audit.get("information_set", "")),
        "availability_scenario": str(audit.get("availability_scenario", "")),
        "proxy_policy": str(audit.get("proxy_policy", "")),
        "view_role": "unknown" if view is None else None,
        "model": str(audit.get("model", "")),
        "model_source": (
            "PRISM" if str(audit.get("model", "")).startswith("PRISM_") else "CPU_BASELINE"
        ),
        "split": str(audit.get("split", "")),
        "prediction_path": audit.get("prediction_path"),
        "prediction_sha256": audit.get("prediction_sha256"),
        "parameter_count": audit.get("parameter_count"),
        "fit_and_prediction_seconds": audit.get("fit_and_prediction_seconds"),
        "test_accessed": audit.get("test_accessed"),
        "ood_accessed": audit.get("ood_accessed"),
    }


def _metric_row(
    paths: PublicAllPaths,
    audit: Mapping[str, Any],
    heads: Mapping[str, HeadSpec],
    view_map: Mapping[tuple[str, str, str, str], ViewSpec],
) -> dict[str, Any]:
    row = _base_row(audit, heads, view_map)
    view = view_map.get(
        (
            str(audit.get("target_head")),
            str(audit.get("information_set")),
            str(audit.get("availability_scenario")),
            str(audit.get("proxy_policy")),
        )
    )
    row["view_role"] = "unknown" if view is None else _view_role(paths.shared, view)
    row.update(
        {
            "rows": None,
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
            "nrmse": None,
            "persistence_skill": None,
            "dynamic_ar_skill": None,
            "rank": None,
        }
    )
    if str(audit.get("status")) == "PASS" and audit.get("prediction_path"):
        frame = _read_prediction(paths, audit)
        metrics = regression_metrics(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        )
        row.update({"rows": int(len(frame)), **metrics})
    return row


def _formal_models(information_set: str) -> set[str]:
    return INPUT_PRISM_MODELS if information_set == "input_only" else DYNAMIC_PRISM_MODELS


def _leaderboard_models(information_set: str) -> set[str]:
    return (
        INPUT_BASELINE_MODELS | INPUT_PRISM_MODELS
        if information_set == "input_only"
        else DYNAMIC_BASELINE_MODELS | DYNAMIC_PRISM_MODELS
    )


def _add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["rank"] = np.nan
    group_columns = [
        "task_id",
        "target_head",
        "information_set",
        "availability_scenario",
        "proxy_policy",
        "split",
    ]
    for _, index in result.groupby(group_columns, dropna=False).groups.items():
        subset = result.loc[index]
        eligible = subset[subset["status"] == "PASS"].copy()
        if eligible.empty:
            continue
        result.loc[eligible.index, "rank"] = eligible["mse"].rank(method="min")
    return result


def _load_access_audits(paths: PublicAllPaths) -> dict[str, Any]:
    path = paths.test_access_audit_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return _read_json(path)


def _materialization_rows(
    paths: PublicAllPaths,
    access: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str, str, str], Mapping[str, Any]]]:
    heads = _head_map(paths.shared)
    views = _all_view_map(paths.shared)
    audits = [item for item in access.get("models", []) if isinstance(item, Mapping)]
    rows = [_metric_row(paths, item, heads, views) for item in audits]
    index = {_audit_key(item): item for item in audits}
    return pd.DataFrame(rows), index


def _add_skills(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    keys = [
        "target_head",
        "information_set",
        "availability_scenario",
        "proxy_policy",
        "split",
    ]
    for _, index in result.groupby(keys, dropna=False).groups.items():
        subset = result.loc[index]
        persistence = subset[
            (subset["model"] == "PERSISTENCE") & (subset["status"] == "PASS")
        ]
        ar = subset[(subset["model"] == "AR") & (subset["status"] == "PASS")]
        persistence_mse = None if persistence.empty else float(persistence.iloc[0]["mse"])
        ar_mse = None if ar.empty else float(ar.iloc[0]["mse"])
        for row_index in index:
            mse_value = result.at[row_index, "mse"]
            if pd.notna(mse_value) and persistence_mse not in {None, 0.0}:
                result.at[row_index, "persistence_skill"] = 1.0 - float(mse_value) / persistence_mse
            if (
                result.at[row_index, "information_set"] == "dynamic"
                and pd.notna(mse_value)
                and ar_mse not in {None, 0.0}
            ):
                result.at[row_index, "dynamic_ar_skill"] = 1.0 - float(mse_value) / ar_mse
    return result


def _inject_joint_not_applicable(
    paths: PublicAllPaths,
    metrics: pd.DataFrame,
    audit_index: Mapping[tuple[str, str, str, str, str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    freeze = _read_json(paths.development_freeze_path)
    existing = set(audit_index)
    rows: list[dict[str, Any]] = []
    for view in public_all_dynamic_views(paths.shared):
        key = (*_view_key(view), "PRISM_V2_1_1_JOINT_KWA", "test")
        if key in existing:
            continue
        record = next(
            item
            for item in freeze.get("views", [])
            if item.get("target_head") == view.head.head_id
            and item.get("information_set") == view.information_set
            and item.get("availability_scenario") == view.availability_scenario
            and item.get("proxy_policy") == view.proxy_policy
        )
        status = (
            "NOT_APPLICABLE_JOINT_NOT_FROZEN"
            if "JOINT" not in record.get("formal_routes", [])
            else "MISSING_TEST_MATERIALIZATION"
        )
        rows.append(
            {
                "status": status,
                "dataset": view.head.dataset,
                "task_id": view.head.task_id,
                "target_head": view.head.head_id,
                "target": view.head.target,
                "information_set": "dynamic",
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "view_role": _view_role(paths.shared, view),
                "model": "PRISM_V2_1_1_JOINT_KWA",
                "model_source": "PRISM",
                "split": "test",
                "prediction_path": None,
                "prediction_sha256": None,
                "parameter_count": None,
                "fit_and_prediction_seconds": None,
                "test_accessed": True,
                "ood_accessed": False,
                "rows": None,
                "mse": None,
                "rmse": None,
                "mae": None,
                "r2": None,
                "nrmse": None,
                "persistence_skill": None,
                "dynamic_ar_skill": None,
                "rank": None,
            }
        )
    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)


def _write_leaderboards(paths: PublicAllPaths, metrics: pd.DataFrame) -> None:
    for information_set, name in (
        ("input_only", "PUBLIC_ALL_INPUT_ONLY_LEADERBOARD.csv"),
        ("dynamic", "PUBLIC_ALL_DYNAMIC_LEADERBOARD.csv"),
    ):
        selected = metrics[
            (metrics["information_set"] == information_set)
            & metrics["model"].isin(_leaderboard_models(information_set))
            & (metrics["split"] == "test")
        ].copy()
        selected = _add_ranks(selected)
        selected.sort_values(
            ["view_role", "task_id", "target_head", "mse", "model"],
            na_position="last",
        ).to_csv(paths.final / name, index=False)


def _stable_seed(identifier: str) -> int:
    suffix = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16)
    return (BOOTSTRAP_SEED + suffix) % (2**32)


def _block_length(view: ViewSpec, paths: PublicAllPaths) -> int:
    length = max(1, view.head.h_steps + view.head.w_steps)
    if view.information_set == "dynamic":
        path = (
            paths.output
            / "DEVELOPMENT"
            / "A"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "RESULT.json"
        )
        if path.is_file():
            result = _read_json(path)
            profile = result.get("a_contract", {}).get("profile", [0, 0])
            length = max(length, int(profile[1]))
    return length


def _holm(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(
            (str(row["task_id"]), str(row["information_set"]), str(row["comparison_family"])),
            [],
        ).append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: float(rows[index]["raw_p_value"]))
        m = len(ordered)
        previous = 0.0
        for rank, index in enumerate(ordered):
            value = min(1.0, float(rows[index]["raw_p_value"]) * (m - rank))
            value = max(previous, value)
            rows[index]["holm_p_value"] = value
            previous = value


def _bootstrap_row(
    paths: PublicAllPaths,
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    view: ViewSpec,
    candidate: str,
    comparator: str,
    family: str,
) -> dict[str, Any]:
    paired = _align_pair(left, right)
    identifier = "|".join(
        [view.head.task_id, view.information_set, view.availability_scenario, view.proxy_policy, candidate, comparator]
    )
    samples = _moving_block_means(
        paired,
        _block_length(view, paths),
        BOOTSTRAP_REPLICATES,
        _stable_seed(identifier),
    )
    observed = float(np.mean(paired["loss_difference"], dtype=np.float64))
    comparator_mse = float(np.mean(np.square(right["y_true"] - right["y_pred"])))
    better = int(np.sum(samples < 0.0))
    opposite = int(np.sum(samples > 0.0))
    raw_p = min(1.0, 2.0 * min(better + 1, opposite + 1) / (BOOTSTRAP_REPLICATES + 1))
    return {
        "status": "PASS",
        "dataset": view.head.dataset,
        "task_id": view.head.task_id,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "split": "test",
        "candidate": candidate,
        "comparator": comparator,
        "comparison_family": family,
        "paired_rows": int(len(paired)),
        "block_length": _block_length(view, paths),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": _stable_seed(identifier),
        "mse_difference": observed,
        "relative_improvement": None if comparator_mse == 0.0 else -observed / comparator_mse,
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "probability_candidate_better": (better + 1) / (BOOTSTRAP_REPLICATES + 1),
        "raw_p_value": raw_p,
        "holm_p_value": None,
    }


def _bootstrap_rows(
    paths: PublicAllPaths,
    metrics: pd.DataFrame,
    audit_index: Mapping[tuple[str, str, str, str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    views = _all_view_map(paths.shared)
    rows: list[dict[str, Any]] = []
    comparisons = {
        "input_only": [
            ("PRISM_V2_1_1_K_C_W", "PERSISTENCE", "INPUT_PRISM_VS_PERSISTENCE"),
            ("PRISM_V2_1_1_K_C_W", "PRISM_V2_1_1_K_C", "KCW_VS_KC"),
        ],
        "dynamic": [
            ("PRISM_V2_1_1_PHYSICS_FIRST", "AR", "PF_VS_AR"),
            ("PRISM_V2_1_1_PHYSICS_FIRST", "PERSISTENCE", "PF_VS_PERSISTENCE"),
            ("PRISM_V2_1_1_PHYSICS_FIRST", "PRISM_V2_1_1_JOINT_KWA", "PF_VS_JOINT"),
            ("PRISM_V2_1_1_K_C_W_DYNAMIC", "PRISM_V2_1_1_K_C_DYNAMIC", "KCW_VS_KC"),
            ("PRISM_V2_1_1_K_C_A_ABLATION", "PRISM_V2_1_1_K_C_DYNAMIC", "KCA_VS_KC"),
            ("PRISM_V2_1_1_PHYSICS_FIRST", "PRISM_V2_1_1_K_C_A_ABLATION", "KCWA_VS_KCA"),
            ("PRISM_V2_1_1_PHYSICS_FIRST", "PRISM_V2_1_1_K_C_W_DYNAMIC", "KCWA_VS_KCW"),
        ],
    }
    frame_cache: dict[tuple[str, str, str, str, str, str], pd.DataFrame] = {}
    for key, audit in audit_index.items():
        if audit.get("status") == "PASS" and audit.get("prediction_path"):
            frame_cache[key] = _read_prediction(paths, audit)
    for view in views.values():
        for candidate, comparator, family in comparisons[view.information_set]:
            left_key = (*_view_key(view), candidate, "test")
            right_key = (*_view_key(view), comparator, "test")
            if left_key not in frame_cache or right_key not in frame_cache:
                continue
            try:
                row = _bootstrap_row(
                    paths,
                    frame_cache[left_key],
                    frame_cache[right_key],
                    view=view,
                    candidate=candidate,
                    comparator=comparator,
                    family=family,
                )
            except RuntimeError as error:
                row = {
                    "status": "FAILED_RETAINED",
                    "task_id": view.head.task_id,
                    "target_head": view.head.head_id,
                    "information_set": view.information_set,
                    "candidate": candidate,
                    "comparator": comparator,
                    "comparison_family": family,
                    "error": str(error),
                }
            rows.append(row)
    _holm([row for row in rows if row.get("status") == "PASS"])
    return rows


def _write_native_support_audit(paths: PublicAllPaths) -> pd.DataFrame:
    reclaim_path = paths.run_root / "NATIVE_SUPPORT_RECLAIM_AUDIT.csv"
    reclaim = pd.read_csv(reclaim_path) if reclaim_path.is_file() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    views = _all_view_map(paths.shared)
    primary_inputs = [
        view for view in public_all_input_views(paths.shared) if _view_role(paths.shared, view) == "primary"
    ]
    for view in primary_inputs:
        k_root = paths.output / "DEVELOPMENT" / "K" / view.head.head_id / view.proxy_policy
        c_path = paths.output / "DEVELOPMENT" / "C" / view.head.head_id / view.proxy_policy / "RESULT.json"
        c = _read_json(c_path)
        assembly_test_rows = common_support_record(paths, view).get("splits", {}).get("test", {}).get("rows")
        assembly_train_rows = c.get("assembly_train_rows")
        assembly_validation_rows = c.get("assembly_validation_rows")
        for path in sorted(k_root.glob("*/RESULT.json")):
            k = _read_json(path)
            channel = str(k["channel"])
            train = load_native_samples(paths.shared, view, "train")
            legacy = None
            if not reclaim.empty:
                subset = reclaim[
                    (reclaim["head"].astype(str) == view.head.head_id)
                    & (reclaim["information_set"].astype(str) == "input_only")
                    & (reclaim["availability_scenario"].astype(str) == view.availability_scenario)
                    & (reclaim["proxy_policy"].astype(str) == view.proxy_policy)
                    & (reclaim["split"].astype(str) == "train")
                ]
                if not subset.empty:
                    legacy = int(subset.iloc[0]["legacy_anchor_rows"])
            native_rows = int(k.get("selected_native_train_rows", 0))
            rows.append(
                {
                    "dataset": view.head.dataset,
                    "task_id": view.head.task_id,
                    "head": view.head.head_id,
                    "channel": channel,
                    "selected_history": k.get("selected_profile_history_steps"),
                    "anchor_train_rows": int(len(train)),
                    "selected_native_train_rows": native_rows,
                    "legacy_common_train_rows": legacy,
                    "reclaimed_train_rows": None if legacy is None else native_rows - legacy,
                    "local_scoring_rows_by_fold": json.dumps(k.get("local_scoring_rows_by_fold", [])),
                    "local_scoring_rows": min(k.get("local_scoring_rows_by_fold", [0]), default=0),
                    "assembly_train_rows": assembly_train_rows,
                    "assembly_validation_rows": assembly_validation_rows,
                    "assembly_test_rows": assembly_test_rows,
                    "native_support_hash": k.get("selected_native_support_hash", {}).get("train"),
                    "local_score_hash": json.dumps(k.get("local_scoring_support_hash_by_fold", [])),
                    "assembly_support_hash": c.get("assembly_train_support_hash"),
                    "support_contract": SUPPORT_CONTRACT,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(paths.final / "PUBLIC_ALL_NATIVE_SUPPORT_AUDIT.csv", index=False)
    frame.to_csv(paths.final / "NATIVE_SUPPORT_FINAL_AUDIT.csv", index=False)
    total = int(frame["reclaimed_train_rows"].clip(lower=0).sum()) if not frame.empty else 0
    by_channel = (
        frame.groupby("channel", dropna=False)["reclaimed_train_rows"]
        .sum()
        .sort_values(ascending=False)
        if not frame.empty
        else pd.Series(dtype=float)
    )
    summary = [
        "# Native Support Summary",
        "",
        f"Channel-level reclaimed training rows (non-negative sum): **{total}**.",
        "This is a support-efficiency statistic, not a causal or predictive improvement claim.",
        "",
        "## Highest recovery channels",
        "",
    ]
    for channel, value in by_channel.head(10).items():
        summary.append(f"- `{channel}`: {int(max(0, value))} rows")
    summary.extend(
        [
            "",
            "## Selection and route changes",
            "",
            "Historical per-channel selection metadata was not available in the frozen run namespace; selection changes are reported as `NOT_AVAILABLE` rather than inferred from aggregate historical metrics.",
            "PF and Joint route changes are read from the global freeze records. Test-direction statements are descriptive correlations only.",
        ]
    )
    (paths.final / "NATIVE_SUPPORT_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return frame


def _head_summary(paths: PublicAllPaths, metrics: pd.DataFrame, native: pd.DataFrame) -> list[dict[str, Any]]:
    heads = _head_map(paths.shared)
    freeze = _read_json(paths.development_freeze_path)
    records = freeze.get("views", [])
    primary_dynamic = {
        item["target_head"]: item
        for item in records
        if item.get("information_set") == "dynamic" and item.get("availability_scenario") == "record_time"
        and item.get("proxy_policy") in {"primary", "proxy_excluded"}
        and item.get("formal_routes")
        and item.get("view_role", "primary") != "secondary"
    }
    result: list[dict[str, Any]] = []
    for head_id, record in sorted(primary_dynamic.items()):
        head = heads.get(head_id)
        subset = metrics[
            (metrics["target_head"] == head_id)
            & (metrics["information_set"] == "dynamic")
            & (metrics["availability_scenario"] == record["availability_scenario"])
            & (metrics["proxy_policy"] == record["proxy_policy"])
        ]
        def one(model: str, split: str = "test") -> dict[str, Any] | None:
            value = subset[(subset["model"] == model) & (subset["split"] == split)]
            return None if value.empty else value.iloc[0].to_dict()
        pf = one("PRISM_V2_1_1_PHYSICS_FIRST")
        joint = one("PRISM_V2_1_1_JOINT_KWA")
        ood_pf = one("PRISM_V2_1_1_PHYSICS_FIRST", "ood")
        c_path = paths.output / "DEVELOPMENT" / "C" / head_id / record["proxy_policy"] / "RESULT.json"
        a_path = paths.output / "DEVELOPMENT" / "A" / head_id / record["availability_scenario"] / record["proxy_policy"] / "RESULT.json"
        w_path = paths.output / "DEVELOPMENT" / "W" / head_id / record["proxy_policy"] / "RESULT.json"
        c = _read_json(c_path)
        w = _read_json(w_path)
        a = _read_json(a_path)
        j_path = paths.output / "DEVELOPMENT" / "JOINT" / head_id / record["availability_scenario"] / record["proxy_policy"] / "RESULT.json"
        j = _read_json(j_path) if j_path.is_file() else {}
        recovery = native[native["head"] == head_id]
        result.append(
            {
                "dataset": head.dataset if head else record.get("dataset"),
                "head": head_id,
                "task_id": record.get("task_id"),
                "target": head.target if head else None,
                "view": {
                    "availability_scenario": record.get("availability_scenario"),
                    "proxy_policy": record.get("proxy_policy"),
                },
                "k_active_channels": c.get("active_channels", []),
                "selected_profiles": c.get("active_selected_k_histories", {}),
                "native_sample_recovery": recovery.to_dict(orient="records"),
                "c_family": c.get("selected_family"),
                "w_family": w.get("w_contract", {}).get("family"),
                "a_family": a.get("a_contract", {}).get("family"),
                "pf_status": record.get("pf_status"),
                "pf_route": record.get("pf_route"),
                "pf_validation_mse": a.get("final_prediction_loss"),
                "pf_test": pf,
                "joint_development_status": record.get("joint_status"),
                "joint_route": record.get("joint_route"),
                "joint_representation": record.get("joint_representation"),
                "predictive_eta": record.get("joint_predictive_eta"),
                "numerical_alpha": record.get("joint_numerical_alpha"),
                "joint_validation_mse": j.get("final_prediction_loss"),
                "joint_test": joint,
                "joint_vs_pf_test_relative_gain": (
                    None
                    if not pf or not joint or not pf.get("mse")
                    else 1.0 - float(joint["mse"]) / float(pf["mse"])
                ),
                "ood": ood_pf,
                "support_hashes": {
                    "c_assembly_train": c.get("assembly_train_support_hash"),
                    "c_assembly_validation": c.get("assembly_validation_support_hash"),
                    "a_validation": a.get("a_raw_input_support_hash"),
                    "joint_validation": j.get("joint_raw_input_validation_support_hash"),
                },
            }
        )
    _write_json(paths.final / "PUBLIC_ALL_PRISM_HEAD_SUMMARY.json", {"heads": result})
    pd.DataFrame(
        [
            {
                "task_id": item["task_id"],
                "head": item["head"],
                "pf_status": item["pf_status"],
                "pf_mse": None if item["pf_test"] is None else item["pf_test"].get("mse"),
                "pf_rmse": None if item["pf_test"] is None else item["pf_test"].get("rmse"),
                "pf_r2": None if item["pf_test"] is None else item["pf_test"].get("r2"),
                "joint_status": item["joint_development_status"],
                "joint_mse": None if item["joint_test"] is None else item["joint_test"].get("mse"),
                "joint_rmse": None if item["joint_test"] is None else item["joint_test"].get("rmse"),
                "joint_r2": None if item["joint_test"] is None else item["joint_test"].get("r2"),
                "joint_vs_pf_relative_gain": item["joint_vs_pf_test_relative_gain"],
            }
            for item in result
        ]
    ).to_csv(paths.final / "PUBLIC_ALL_HEAD_SUMMARY.csv", index=False)
    return result


def _cross_task_ranking(paths: PublicAllPaths, leaderboard: pd.DataFrame) -> pd.DataFrame:
    primary = leaderboard[
        (leaderboard["view_role"] == "primary")
        & (leaderboard["status"] == "PASS")
        & (leaderboard["split"] == "test")
    ].copy()
    rows: list[dict[str, Any]] = []
    for information_set, group in primary.groupby("information_set"):
        for model, values in group.groupby("model"):
            wins = 0
            ties = 0
            losses = 0
            for _, value in values.iterrows():
                task_values = group[group["task_id"] == value["task_id"]]
                best = float(task_values["mse"].min())
                best_count = int(np.isclose(task_values["mse"].to_numpy(dtype=float), best).sum())
                if np.isclose(float(value["mse"]), best):
                    if best_count == 1:
                        wins += 1
                    else:
                        ties += 1
                else:
                    losses += 1
            rows.append(
                {
                    "information_set": information_set,
                    "model": model,
                    "tasks_covered": int(values["task_id"].nunique()),
                    "mean_rank": float(values["rank"].mean()),
                    "median_rank": float(values["rank"].median()),
                    "win": wins,
                    "tie": ties,
                    "loss": losses,
                    "descriptive_only": True,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["information_set", "mean_rank", "model"])
    frame.to_csv(paths.final / "PUBLIC_ALL_CROSS_TASK_RANKING.csv", index=False)
    return frame


def _runtime_resources(paths: PublicAllPaths, access: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for stage_path in sorted((paths.output / "DEVELOPMENT").rglob("RESULT.json")):
        result = _read_json(stage_path)
        rows.append(
            {
                "stage": result.get("stage"),
                "target_head": result.get("target_head"),
                "model": result.get("model"),
                "status": result.get("status"),
                "training_wall_time_seconds": result.get("elapsed_seconds"),
                "prediction_wall_time_seconds": None,
                "peak_ram_bytes": None,
                "resource_note": "peak RAM and fit/predict split were not persisted by the frozen runner",
                "path": str(stage_path.relative_to(paths.run_root)),
            }
        )
    for result in access.get("models", []):
        rows.append(
            {
                "stage": "T1_PUBLIC_ALL_TEST_OOD_ACCESS",
                "target_head": result.get("target_head"),
                "model": result.get("model"),
                "status": result.get("status"),
                "training_wall_time_seconds": None,
                "prediction_wall_time_seconds": result.get("fit_and_prediction_seconds"),
                "peak_ram_bytes": None,
                "resource_note": "materialization audit records combined fit and prediction time",
                "path": result.get("prediction_path"),
            }
        )
    pd.DataFrame(rows).to_csv(paths.final / "PUBLIC_ALL_RUNTIME_RESOURCES.csv", index=False)


def _group_metrics(paths: PublicAllPaths, access: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for audit in access.get("models", []):
        if audit.get("status") != "PASS" or not audit.get("prediction_path"):
            continue
        frame = _read_prediction(paths, audit)
        for entity, group in frame.groupby("entity_id", dropna=False):
            metric = regression_metrics(
                group["y_true"].to_numpy(dtype=np.float64),
                group["y_pred"].to_numpy(dtype=np.float64),
            )
            rows.append(
                {
                    "dataset": audit.get("dataset"),
                    "task_id": str(audit.get("target_head", "")).split("__", 1)[0],
                    "target_head": audit.get("target_head"),
                    "information_set": audit.get("information_set"),
                    "availability_scenario": audit.get("availability_scenario"),
                    "proxy_policy": audit.get("proxy_policy"),
                    "split": audit.get("split"),
                    "model": audit.get("model"),
                    "group_label": str(entity),
                    "rows": len(group),
                    **metric,
                }
            )
    pd.DataFrame(rows).to_csv(paths.final / "PUBLIC_ALL_GROUP_METRICS.csv", index=False)


def _final_report(
    paths: PublicAllPaths,
    metrics: pd.DataFrame,
    head_summary: list[dict[str, Any]],
    ranking: pd.DataFrame,
    native: pd.DataFrame,
    evidence: Mapping[str, Any],
) -> None:
    if evidence.get("materialization_repair_after_lockbox_failure"):
        evidence_summary = (
            "The first lockbox access ended in a final-materialization runtime "
            "failure. This result reuses the unchanged frozen development artifacts "
            "after a code-equivalence and SHA256 audit, applies only the accepted "
            "materialization repair, and records two lockbox access attempts. No "
            "test result was used for reselection."
        )
    else:
        evidence_summary = (
            "This is a prospective Native Support protocol rerun with prior "
            "historical context. Historical aggregates were not used for selection."
        )
    lines = [
        "# PRISM v2.1.1 Native Support Public-All Final Report",
        "",
        f"Evidence class: `{evidence['evidence_class']}`.",
        "",
        evidence_summary,
        "",
        "## Scope",
        "",
        "Five public datasets and seven primary heads were evaluated with the frozen primary views. GPU baselines and multihorizon scale sweeps were out of scope.",
        "",
        "## Primary heads",
        "",
        "| Task | PF status | PF test MSE | Joint status | Joint test MSE | Joint vs PF |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in head_summary:
        pf = item.get("pf_test") or {}
        joint = item.get("joint_test") or {}
        lines.append(
            f"| {item['task_id']} | {item['pf_status']} | {pf.get('mse')} | {item['joint_development_status']} | {joint.get('mse')} | {item['joint_vs_pf_test_relative_gain']} |"
        )
    lines.extend(
        [
            "",
            "## Leaderboards",
            "",
            "Input-only and dynamic leaderboards are separate. Test ranking uses the frozen task-level common support; extra native coverage is reported separately and is not used for ranking.",
            "",
            "Top rows are descriptive test outcomes after the global freeze; they do not alter frozen selections.",
            "",
            "## Native Support",
            "",
            f"The final audit contains {len(native)} channel rows. Reclaimed rows are support-efficiency measurements and are not interpreted as direct prediction improvements.",
            "",
            "## Statistics",
            "",
            "Paired moving-block bootstrap uses 500 replicates with fixed seeds and finite-sample p-value correction. Holm correction is applied within registered comparison families.",
            "",
            "## OOD",
            "",
            "OOD is reported only for registered TEP/Metro and other available OOD views. OOD residual-state construction reuses frozen test residuals where required by the registered causal protocol.",
            "",
            "## Interpretation",
            "",
            "The results describe predictive contribution, structured response evidence, conditional novelty, and module activation or degradation. They do not prove causality or mechanism.",
            "",
            "The most direct residual risk is that the frozen runner did not persist peak-RAM and separate fit/prediction timing counters; resource output marks those fields as not recorded.",
        ]
    )
    report = paths.final / "PRISM_V211_NATIVE_SUPPORT_PUBLIC_ALL_FINAL_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _full_repro_manifest(paths: PublicAllPaths) -> dict[str, Any]:
    records = []
    for path in sorted(paths.run_root.rglob("*")):
        if not path.is_file() or paths.return_root in path.parents:
            continue
        rel = path.relative_to(paths.run_root).as_posix()
        if rel.startswith("final/FULL_REPRO_MANIFEST.json"):
            continue
        if rel.startswith("final/") and path.suffix not in {".json", ".csv", ".md", ".txt"}:
            continue
        role = "shared_data" if rel.startswith("shared/") else "prediction" if "prediction" in rel else "artifact"
        stage = rel.split("/", 1)[0]
        records.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
                "generated_by_stage": stage,
            }
        )
    reused_path = paths.freeze / REUSED_ARTIFACT_MANIFEST_NAME
    if reused_path.is_file():
        reused = _read_json(reused_path)
        if reused.get("status") != "PASS":
            raise RuntimeError("reused development artifact manifest is not PASS")
        records.extend(dict(item) for item in reused.get("files", ()))
    value = {"status": "PASS", "files": records}
    _write_json(paths.final / "FULL_REPRO_MANIFEST.json", value)
    return value


def _copy_small_artifacts(paths: PublicAllPaths, stage: Path, reporting_commit: str | None) -> None:
    def copy(source: Path, destination: str) -> None:
        if not source.is_file():
            return
        target = stage / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    descriptor = load_public_all_descriptor(paths.project)
    repair = _repair_manifest(paths)
    copy(paths.project / "configs/prism_v211_native_public_all.json", "protocol/prism_v211_native_public_all.json")
    copy(paths.project / "configs/cpu_model_freeze_v1.json", "protocol/cpu_model_freeze_v1.json")
    theory = theory_path(paths.project)
    copy(theory, "protocol/canonical_theory.md")
    for source in (
        paths.run_root / "PUBLIC_ALL_RAW_DATA_REAUDIT.json",
        paths.run_root / "C1_NATIVE_SUPPORT_AUDIT.json",
        paths.run_root / "NATIVE_SUPPORT_RECLAIM_AUDIT.csv",
        paths.run_root / "PRE_RUN_ENVIRONMENT.json",
        paths.run_root / "logs/PRE_RUN_PYTEST.txt",
        paths.run_root / "logs/PRE_RUN_PYTEST_RERUNN2.txt",
        paths.freeze / "K_NATIVE_SUPPORT_AUDIT.json",
        paths.freeze / "TASK_LEADERBOARD_COMMON_SUPPORT.json",
        paths.development_freeze_path,
        paths.freeze / REPAIR_MANIFEST_NAME,
        paths.freeze / REUSED_ARTIFACT_MANIFEST_NAME,
        paths.test_access_audit_path,
        paths.final / "FULL_REPRO_MANIFEST.json",
    ):
        copy(source, f"audit/{source.name}")
    for pattern in (
        "*LOCKBOX_ACCESSED_RUNTIME_FAILURE.json",
        "*PUBLIC_ALL_TEST_OOD_ACCESS_AUDIT.json",
    ):
        for source in sorted(paths.freeze.glob(pattern)):
            copy(source, f"audit/{source.name}")
    for source in sorted(paths.final.glob("*")):
        if source.is_file() and source.suffix.lower() in {".json", ".csv", ".md", ".txt"}:
            copy(source, f"results/{source.name}")
    if repair:
        attempts = max(int(repair.get("lockbox_access_attempts", 1)), 1)
        failures = attempts - 1
        access_note = (
            f"- {failures} test/OOD materialization attempt(s) failed after lockbox "
            f"access; the repaired result records {attempts} attempts and no "
            "reselection.\n"
        )
    else:
        access_note = (
            "- Test/OOD access occurred once after global development freeze.\n"
        )
    changelog = stage / "CHANGELOG.md"
    changelog.write_text(
        "# PRISM v2.1.1 Native Support Public-All\n\n"
        "- Fresh C1: `NATIVE_K_COMMON_ASSEMBLY_R1`.\n"
        "- Five datasets, seven primary heads, primary views.\n"
        "- GPU baselines and scale sweeps are out of scope.\n"
        + access_note,
        encoding="utf-8",
    )
    (stage / "GENERATING_COMMIT.txt").write_text(
        _read_json(paths.development_freeze_path).get("generating_commit", "") + "\n",
        encoding="utf-8",
    )
    (stage / "REPORTING_COMMIT.txt").write_text(
        (reporting_commit or _git(paths.project, "rev-parse", "HEAD")) + "\n",
        encoding="utf-8",
    )
    if repair:
        (stage / "MATERIALIZATION_REPAIR_COMMIT.txt").write_text(
            str(repair.get("materialization_repair_commit", "")) + "\n",
            encoding="utf-8",
        )
    (stage / "GIT_STATUS.txt").write_text(
        _git(paths.project, "status", "--short") + "\n", encoding="utf-8"
    )


def _git(project: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=project, text=True).strip()


def _package(paths: PublicAllPaths, reporting_commit: str | None) -> dict[str, Any]:
    paths.return_root.mkdir(parents=True, exist_ok=True)
    zip_path = paths.return_root / f"{PACKAGE_NAME}.zip"
    sidecar = paths.return_root / f"{PACKAGE_NAME}.zip.sha256"
    if zip_path.exists():
        zip_path.unlink()
    if sidecar.exists():
        sidecar.unlink()
    with tempfile.TemporaryDirectory(prefix="public_all_package_", dir=paths.return_root) as temp:
        stage = Path(temp) / PACKAGE_NAME
        stage.mkdir(parents=True)
        _copy_small_artifacts(paths, stage, reporting_commit)
        records = []
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name not in {"MANIFEST.txt", "SHA256SUMS.txt"}:
                records.append(
                    {
                        "path": path.relative_to(stage).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        (stage / "MANIFEST.txt").write_text(
            "\n".join(f"{row['path']}\t{row['bytes']}\t{row['sha256']}" for row in records) + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "path": "MANIFEST.txt",
                "bytes": (stage / "MANIFEST.txt").stat().st_size,
                "sha256": sha256_file(stage / "MANIFEST.txt"),
            }
        )
        (stage / "SHA256SUMS.txt").write_text(
            "\n".join(f"{row['sha256']}  {row['path']}" for row in records) + "\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage.parent))
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("unzip -t failed")
        sums = [line.split(maxsplit=1) for line in archive.read(f"{PACKAGE_NAME}/SHA256SUMS.txt").decode().splitlines() if line.strip()]
        for digest, name in sums:
            archive_name = f"{PACKAGE_NAME}/{name}"
            if archive_name not in archive.namelist():
                raise RuntimeError(f"internal hash target missing: {name}")
            if hashlib.sha256(archive.read(archive_name)).hexdigest() != digest:
                raise RuntimeError(f"internal hash mismatch: {name}")
    digest = sha256_file(zip_path)
    sidecar.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return {
        "status": "PASS",
        "path": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "sha256": digest,
        "unzip_test": "PASS",
        "internal_hashes": "PASS",
    }


def _evidence(
    paths: PublicAllPaths,
    access: Mapping[str, Any],
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    head_summary: list[dict[str, Any]],
    reporting_commit: str | None,
) -> dict[str, Any]:
    freeze = _read_json(paths.development_freeze_path)
    repair = _repair_manifest(paths)
    raw = _read_json(paths.run_root / "PUBLIC_ALL_RAW_DATA_REAUDIT.json")
    primary = [item for item in head_summary if item.get("pf_status") == "PASS"]
    joint_count = sum(item.get("joint_development_status") == "PASS" for item in head_summary)
    pf_ranks = ranking[
        (ranking["information_set"] == "dynamic")
        & (ranking["model"] == "PRISM_V2_1_1_PHYSICS_FIRST")
    ]
    result = {
        "status": "PASS" if access.get("status") == "PASS" else "FAILED",
        "evidence_class": (
            REPAIR_EVIDENCE_CLASS if repair else freeze.get("evidence_class")
        ),
        "source_branch": SOURCE_BRANCH,
        "source_commit": SOURCE_COMMIT,
        "execution_branch": EXECUTION_BRANCH,
        "generating_commit": freeze.get("generating_commit"),
        "materialization_repair_commit": repair.get(
            "materialization_repair_commit"
        ),
        "reporting_commit": reporting_commit or _git(paths.project, "rev-parse", "HEAD"),
        "canonical_theory_sha256": freeze.get("canonical_theory_sha256"),
        "support_contract": SUPPORT_CONTRACT,
        "raw_data_hash_audit": _raw_audit_summary(raw),
        "shared_data_sha": freeze.get("shared_development_metadata_sha256"),
        "development_freeze_sha": sha256_file(paths.development_freeze_path),
        "test_access_audit_sha": sha256_file(paths.test_access_audit_path),
        "datasets": 5,
        "primary_heads": 7,
        "heads_passed": len(primary),
        "heads_failed": 7 - len(primary),
        "heads_unsupported": 0,
        "PF_formal_count": len(primary),
        "Joint_formal_count": int(joint_count),
        "input_only_best_models": {},
        "dynamic_best_models": {},
        "PRISM_mean_rank": None if pf_ranks.empty else float(pf_ranks.iloc[0]["mean_rank"]),
        "PRISM_median_rank": None if pf_ranks.empty else float(pf_ranks.iloc[0]["median_rank"]),
        "OOD_conclusions": metrics[metrics["split"] == "ood"][
            ["task_id", "target_head", "model", "mse", "status"]
        ].to_dict(orient="records"),
        "test_accessed": access.get("test_accessed") is True,
        "ood_accessed": access.get("ood_accessed") is True,
        "post_test_reselection": False,
        "materialization_repair_after_lockbox_failure": bool(repair),
        "development_artifacts_reused": bool(repair),
        "lockbox_access_attempts": int(repair.get("lockbox_access_attempts", 1)),
        "one_shot_test_access": not bool(repair),
        "original_lockbox_failure_sha256": repair.get(
            "original_lockbox_failure_sha256"
        ),
        "lockbox_failure_history": repair.get("lockbox_failure_history", []),
        "development_code_equivalence_audit": repair.get(
            "development_code_equivalence_audit"
        ),
    }
    for information_set, key in (("input_only", "input_only_best_models"), ("dynamic", "dynamic_best_models")):
        subset = metrics[
            (metrics["information_set"] == information_set)
            & (metrics["split"] == "test")
            & (metrics["status"] == "PASS")
            & metrics["model"].isin(_leaderboard_models(information_set))
            & (metrics["view_role"] == "primary")
        ]
        if not subset.empty:
            result[key] = {
                task: str(group.sort_values("mse").iloc[0]["model"])
                for task, group in subset.groupby("task_id")
            }
    _write_json(paths.final / "PUBLIC_ALL_FINAL_EVIDENCE_SUMMARY.json", result)
    return result


def report_all(paths: PublicAllPaths, reporting_commit: str | None = None) -> dict[str, Any]:
    access = _load_access_audits(paths)
    metrics, audit_index = _materialization_rows(paths, access)
    metrics = _inject_joint_not_applicable(paths, metrics, audit_index)
    metrics = _add_skills(metrics)
    paths.final.mkdir(parents=True, exist_ok=True)
    metrics = metrics.sort_values(
        ["information_set", "task_id", "target_head", "split", "model"],
        na_position="last",
    ).reset_index(drop=True)
    metrics.to_csv(paths.final / "PUBLIC_ALL_METRICS.csv", index=False)
    _write_leaderboards(paths, metrics)
    prism = metrics[metrics["model"].isin(INPUT_PRISM_MODELS | DYNAMIC_PRISM_MODELS)]
    prism.to_csv(paths.final / "PUBLIC_ALL_PRISM_SELECTED_RESULTS.csv", index=False)
    bootstrap = pd.DataFrame(_bootstrap_rows(paths, metrics, audit_index))
    bootstrap.to_csv(paths.final / "PUBLIC_ALL_BOOTSTRAP.csv", index=False)
    native = _write_native_support_audit(paths)
    head_summary = _head_summary(paths, metrics, native)
    leaderboard = pd.concat(
        [pd.read_csv(paths.final / "PUBLIC_ALL_INPUT_ONLY_LEADERBOARD.csv"), pd.read_csv(paths.final / "PUBLIC_ALL_DYNAMIC_LEADERBOARD.csv")],
        ignore_index=True,
    )
    ranking = _cross_task_ranking(paths, leaderboard)
    _runtime_resources(paths, access)
    _group_metrics(paths, access)
    evidence = _evidence(paths, access, metrics, ranking, head_summary, reporting_commit)
    _full_repro_manifest(paths)
    _final_report(paths, metrics, head_summary, ranking, native, evidence)
    package = _package(paths, reporting_commit)
    evidence["package"] = package
    _write_json(paths.final / "PUBLIC_ALL_FINAL_EVIDENCE_SUMMARY.json", evidence)
    return {"status": evidence["status"], "evidence": evidence, "package": package}
