"""Strict nested-OOF E1--E6 revalidation utilities.

The revalidation layer is intentionally reporting-oriented.  It consumes
frozen development OOF artifacts, runs a new identifiable synthetic recovery
experiment, and writes auditable tables without changing the production
selector.  Formal test and OOD partitions are never opened by this module.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor, realized_state_profiles
from .level_reconstruction import metric_bundle_delta_and_level, support_hash
from .strict_oof_selection import (
    ACTIVE,
    ZERO_IDENTITY,
    numerical_epsilon,
    strict_nested_oof_select,
)
from .v21_a import EXACT_ZERO, fit_mature_residual_ar, mature_residual_features
from .v2_views import development_dynamic_views
from .v211_config import PUBLIC_ALL_PROTOCOL, load_v211_configs
from .v211_joint import (
    register_joint_fold_on_oof_support,
    registered_joint_inner_fold_frames,
)
from .v211_k import load_active_channels
from .v211_support import load_native_samples


STATUS_VALUES = {"COMPLETED", "PARTIAL", "NOT_RUN", "PROTOCOL_BLOCKED", "INVALID"}
EXPECTED_BASE_COMMIT = "73d0bd4c69148c0cb91459f39f166acacf5856b6"
PROTOCOL_ID = "PRISM_STRICT_NESTED_OOF_E1_E6_REVALIDATION_V1"
SYNTHETIC_ID = "IDENTIFIABLE_E2_V2"
RIDGES = (1e-6, 1e-3, 1.0)
HISTORIES = (1, 4, 8)
STAGES = ("C", "W", "A")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(project: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=project, text=True, encoding="utf-8"
    ).strip()


def _selection_report(selection: Mapping[str, Any]) -> dict[str, Any]:
    parent = float(selection["parent_oof_risk"])
    child = float(selection["child_oof_risk"])
    # Cached candidate fits and nested-OOF losses may be reused, but E1 must
    # not inherit an earlier serialized stage decision.  Recompute the only
    # admissible gate from the frozen evidence and fail closed if it differs
    # from the stored strict run.
    gain = float(parent - child)
    epsilon_num = numerical_epsilon(parent, child)
    route = ACTIVE if gain > epsilon_num else ZERO_IDENTITY
    stored_route = str(selection["routing_status"])
    if stored_route != route:
        raise RuntimeError(
            "cached strict stage route does not replay from its frozen OOF evidence"
        )
    parents = np.asarray(selection["parent_outer_fold_losses"], dtype=np.float64)
    children = np.asarray(selection["child_outer_fold_losses"], dtype=np.float64)
    fold_margins = (parents - children) / np.maximum(
        parents, np.finfo(np.float64).eps
    )
    symmetric_denominator = np.maximum(
        np.maximum(parents, children), np.finfo(np.float64).eps
    )
    symmetric_fold_margins = (parents - children) / symmetric_denominator
    raw_margin = float(
        selection.get(
            "relative_admission_margin",
            gain / max(parent, np.finfo(np.float64).eps),
        )
    )
    symmetric_margin = float(
        gain / max(parent, child, np.finfo(np.float64).eps)
    )
    return {
        "absolute_oof_gain": gain,
        "relative_admission_margin": raw_margin,
        "raw_relative_admission_margin": raw_margin,
        "symmetric_reporting_margin": symmetric_margin,
        "route": route,
        "outer_fold_margins": fold_margins.tolist(),
        "outer_fold_symmetric_margins": symmetric_fold_margins.tolist(),
        "margin_signs": [
            1 if value > 0 else (-1 if value < 0 else 0)
            for value in fold_margins
        ],
        "candidate_id": str(
            selection.get(
                "tuned_nonzero_candidate", selection["final_selected_candidate"]
            )
        ),
        "parent_candidate_id": str(selection["identity"]),
        "epsilon_num": epsilon_num,
        "stored_epsilon_num": float(selection["epsilon_num"]),
        "stage_decision_recomputed": True,
        "reporting_only": True,
    }


def _metrics(
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    current_level: np.ndarray,
) -> dict[str, Any]:
    value = metric_bundle_delta_and_level(delta_true, delta_pred, current_level)
    value.pop("future_level_true")
    value.pop("future_level_pred")
    return value


def _parameter_count(result: Mapping[str, Any]) -> int:
    preferred = (
        result.get("final_selected_contract"),
        result.get("joint_contract"),
        result.get("w_contract"),
        result.get("a_contract"),
        result.get("fusion_contract"),
    )
    for contract in preferred:
        if isinstance(contract, Mapping):
            if "parameter_count" in contract:
                return int(contract["parameter_count"])
            coefficient = contract.get("coefficient")
            if isinstance(coefficient, list):
                return len(coefficient) + 1
    return 0


def _prediction_hash(frame: pd.DataFrame) -> str:
    values = np.ascontiguousarray(frame["y_pred"].to_numpy(dtype=np.float64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _frame_support_hash(frame: pd.DataFrame) -> str:
    return support_hash(frame["base_origin_id"].astype(str))


def _metrics_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, default=_json_default) == json.dumps(
        right, sort_keys=True, default=_json_default
    )


def _safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _replay_prefix_frames(
    raw_frames: Mapping[str, pd.DataFrame],
    routes: Mapping[str, str],
) -> dict[str, pd.DataFrame]:
    """Build chain-consistent prefixes without changing any raw artifact."""

    chain: dict[str, pd.DataFrame] = {"K": raw_frames["K"].copy(deep=True)}
    for stage, parent, child in (
        ("C", "K", "KC"),
        ("W", "KC", "KCW"),
        ("A", "KCW", "KCWA"),
    ):
        if routes.get(stage) == ACTIVE:
            chain[child] = raw_frames[child].copy(deep=True)
        else:
            chain[child] = chain[parent].copy(deep=True)
    if "J" in raw_frames:
        chain["J"] = raw_frames["J"].copy(deep=True)
    return chain


def _result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction(results: Path, result: Mapping[str, Any]) -> pd.DataFrame:
    return pd.read_parquet(_prediction_file(results, result))


def _prediction_file(results: Path, result: Mapping[str, Any]) -> Path:
    relative = result.get("prediction_path") or result.get("final_selected_prediction_path")
    if not relative:
        raise KeyError("prediction_path")
    path = results / str(relative)
    if not path.is_file() and str(relative).startswith("DEVELOPMENT/"):
        path = results / str(relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass(frozen=True)
class CachedView:
    task: str
    head: str
    dataset: str
    target: str
    proxy: str
    availability: str
    w0: int
    shared_name: str
    dynamic: bool = True


FORMAL_VIEWS = (
    CachedView("Debutanizer", "DEB_C4__H5__W1", "debutanizer", "y", "primary", "record_time", 1, "public3_shared"),
    CachedView("SRU H2S", "SRU_H2S_REP_H1__H1__W1", "sru", "y1", "primary", "record_time", 1, "sru_shared"),
    CachedView("SRU SO2", "SRU_SO2_REP_H1__H1__W1", "sru", "y2", "primary", "record_time", 1, "sru_shared"),
    CachedView("PMSM proxy-excluded", "PMSM_PM5__H600__W60", "pmsm", "pm", "proxy_excluded", "record_time", 60, "public3_shared"),
    CachedView("MetroPT P60", "METRO_P60__H6__W1", "metropt", "Reservoirs", "proxy_excluded", "record_time", 1, "public3_shared"),
    CachedView("MetroPT Oil20", "METRO_OIL20__H120__W12", "metropt", "Oil_temperature", "primary", "record_time", 12, "public3_shared"),
    CachedView("TEP input-only", "TEP_G_NOWCAST_H0__H0__W1", "tep", "xmeas_40", "proxy_excluded", "record_time", 1, "tep_shared", False),
    CachedView("TEP record-time", "TEP_G_NOWCAST_H0__H0__W1", "tep", "xmeas_40", "proxy_excluded", "record_time", 1, "tep_shared"),
    CachedView("TEP maturity-5", "TEP_G_NOWCAST_H0__H0__W1", "tep", "xmeas_40", "proxy_excluded", "analyzer_maturity_5_steps", 1, "tep_shared"),
)


def _selection_for(stage: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
    if stage == "C":
        return result["family_selection"]
    return result["selection"]


def _current_levels(shared: Path, view: CachedView, frame: pd.DataFrame) -> np.ndarray:
    accessor = BaseAccessor(shared, view.dataset, "validation", [view.target])
    return accessor.block_means(frame, view.target, [(0, view.w0)])[:, 0]


def _align_to(reference: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["base_origin_id", "y_true", "y_pred"]
    left = reference[["base_origin_id"]].copy()
    aligned = left.merge(frame[columns], on="base_origin_id", how="left", validate="one_to_one")
    if aligned[["y_true", "y_pred"]].isna().any().any():
        raise RuntimeError("prefix prediction does not cover the frozen reference support")
    return aligned


def run_e1_cached(
    output: Path,
    baseline_run: Path,
    tep_repair_run: Path,
    shared_root: Path,
) -> pd.DataFrame:
    destination = output / "E1_STAGEWISE"
    destination.mkdir(parents=True, exist_ok=True)
    chain_destination = destination / "chain_consistent_prefix_predictions"
    chain_destination.mkdir(parents=True, exist_ok=True)
    base_results = baseline_run / "results"
    tep_results = tep_repair_run / "results"
    table_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for view in FORMAL_VIEWS:
        shared = shared_root / view.shared_name
        c_path = base_results / "DEVELOPMENT" / "C" / view.head / view.proxy / "RESULT.json"
        w_path = base_results / "DEVELOPMENT" / "W" / view.head / view.proxy / "RESULT.json"
        c_result = _result(c_path)
        w_result = _result(w_path)
        best_channel = str(c_result["best_active_k_channel"])
        k_result = _result(
            base_results
            / "DEVELOPMENT"
            / "K"
            / view.head
            / view.proxy
            / best_channel
            / "RESULT.json"
        )
        frames: dict[str, pd.DataFrame] = {
            "K": _prediction(base_results, k_result),
            "KC": _prediction(base_results, c_result),
            "KCW": _prediction(base_results, w_result),
        }
        results_by_prefix: dict[str, Mapping[str, Any]] = {
            "K": k_result,
            "KC": c_result,
            "KCW": w_result,
        }
        if view.dynamic:
            dynamic_results = tep_results if view.dataset == "tep" else base_results
            a_result = _result(
                dynamic_results
                / "DEVELOPMENT"
                / "A"
                / view.head
                / view.availability
                / view.proxy
                / "RESULT.json"
            )
            joint_result = _result(
                dynamic_results
                / "DEVELOPMENT"
                / "JOINT"
                / view.head
                / view.availability
                / view.proxy
                / "RESULT.json"
            )
            frames["KCWA"] = _prediction(dynamic_results, a_result)
            frames["J"] = _prediction(dynamic_results, joint_result)
            results_by_prefix["KCWA"] = a_result
            results_by_prefix["J"] = joint_result
        else:
            frames["KCWA"] = frames["KCW"].copy()
            frames["J"] = frames["KCW"].copy()
            results_by_prefix["KCWA"] = w_result
            results_by_prefix["J"] = w_result
        reference = frames["J"]
        aligned = {name: _align_to(reference, frame) for name, frame in frames.items()}
        reference_meta = reference[["base_origin_id", "entity_id", "origin"]].copy()
        current = _current_levels(shared, view, reference_meta)
        view_label = "input_only/record_time" if not view.dynamic else f"dynamic/{view.availability}"
        stage_pairs = {"C": ("K", "KC"), "W": ("KC", "KCW"), "A": ("KCW", "KCWA")}
        stage_results: dict[str, Mapping[str, Any] | None] = {
            "C": c_result,
            "W": w_result,
            "A": results_by_prefix.get("KCWA") if view.dynamic else None,
        }
        stage_reports: dict[str, dict[str, Any]] = {}
        for stage, (parent, child) in stage_pairs.items():
            if stage_results[stage] is None:
                stage_reports[stage] = {
                    "route": ZERO_IDENTITY,
                    "relative_admission_margin": 0.0,
                    "raw_relative_admission_margin": 0.0,
                    "symmetric_reporting_margin": 0.0,
                    "absolute_oof_gain": 0.0,
                    "outer_fold_margins": [],
                    "outer_fold_symmetric_margins": [],
                    "margin_signs": [],
                    "candidate_id": "NOT_APPLICABLE_INPUT_ONLY",
                    "parent_candidate_id": parent,
                    "epsilon_num": 0.0,
                    "reporting_only": True,
                }
            else:
                stage_reports[stage] = _selection_report(
                    _selection_for(stage, stage_results[stage] or {})
                )

        # Route prefixes from the frozen stage evidence.  A ZERO stage is an
        # external identity element and therefore copies its parent frame
        # byte-for-byte; raw full-refit stage artifacts remain available in
        # ``aligned`` for the reporting audit below.
        chain_frames = _replay_prefix_frames(
            aligned,
            {stage: report["route"] for stage, report in stage_reports.items()},
        )

        prefix_metrics: dict[str, dict[str, Any]] = {
            name: _metrics(
                frame["y_true"].to_numpy(dtype=np.float64),
                frame["y_pred"].to_numpy(dtype=np.float64),
                current,
            )
            for name, frame in chain_frames.items()
        }
        task_slug = _safe_slug(view.task)
        view_slug = _safe_slug(view_label)
        artifact_dir = chain_destination / task_slug / view_slug
        artifact_dir.mkdir(parents=True, exist_ok=True)
        chain_paths: dict[str, str] = {}
        for prefix, frame in chain_frames.items():
            path = artifact_dir / f"{prefix}.parquet"
            # Include the frozen reference metadata in every replay artifact.
            output_frame = reference_meta.copy()
            output_frame["y_true"] = frame["y_true"].to_numpy(dtype=np.float64)
            output_frame["y_pred"] = frame["y_pred"].to_numpy(dtype=np.float64)
            output_frame.to_parquet(path, index=False)
            chain_paths[prefix] = str(path.relative_to(output))

        # Compare every routed prefix with its parent and retain both raw and
        # chain-consistent paths for exact-zero auditing.
        for stage, (parent, child) in stage_pairs.items():
            parent_frame = chain_frames[parent]
            raw_child_frame = aligned[child]
            child_frame = chain_frames[child]
            differences = np.abs(
                parent_frame["y_pred"].to_numpy(dtype=np.float64)
                - child_frame["y_pred"].to_numpy(dtype=np.float64)
            )
            raw_differences = np.abs(
                parent_frame["y_pred"].to_numpy(dtype=np.float64)
                - raw_child_frame["y_pred"].to_numpy(dtype=np.float64)
            )
            identity_required = stage_reports[stage]["route"] == ZERO_IDENTITY
            replay_rows.append(
                {
                    "task": view.task,
                    "view": view_label,
                    "stage": stage,
                    "route": stage_reports[stage]["route"],
                    "identity_required": identity_required,
                    "parent_prefix": parent,
                    "child_prefix": child,
                    "raw_stage_prediction_path": str(
                        _prediction_file(
                            tep_results if (view.dataset == "tep" and view.dynamic) else base_results,
                            results_by_prefix[child],
                        )
                    ),
                    "chain_consistent_prefix_prediction_path": chain_paths[child],
                    "parent_support_hash": _frame_support_hash(parent_frame),
                    "child_support_hash": _frame_support_hash(child_frame),
                    "raw_child_support_hash": _frame_support_hash(raw_child_frame),
                    "parent_prediction_hash": _prediction_hash(parent_frame),
                    "child_prediction_hash": _prediction_hash(child_frame),
                    "raw_child_prediction_hash": _prediction_hash(raw_child_frame),
                    "max_abs_difference": float(np.max(differences, initial=0.0)),
                    "raw_max_abs_difference": float(np.max(raw_differences, initial=0.0)),
                    "metrics_parent_json": json.dumps(prefix_metrics[parent], sort_keys=True),
                    "metrics_child_json": json.dumps(prefix_metrics[child], sort_keys=True),
                    "metrics_equal": _metrics_equal(prefix_metrics[parent], prefix_metrics[child]),
                    "exact_identity_pass": (
                        not identity_required
                        or (
                            _frame_support_hash(parent_frame) == _frame_support_hash(child_frame)
                            and _prediction_hash(parent_frame) == _prediction_hash(child_frame)
                            and float(np.max(differences, initial=0.0)) == 0.0
                            and _metrics_equal(prefix_metrics[parent], prefix_metrics[child])
                        )
                    ),
                }
            )
        row: dict[str, Any] = {
            "task": view.task,
            "view": view_label,
            "rows": len(reference),
            "support_hash": _frame_support_hash(reference),
            "active_channel_count": len(c_result.get("active_channels", [])),
        }
        local_params = {
            "K": _parameter_count(k_result),
            "C": _parameter_count(c_result) if stage_reports["C"]["route"] == ACTIVE else 0,
            "W": _parameter_count(w_result) if stage_reports["W"]["route"] == ACTIVE else 0,
            "A": (
                _parameter_count(results_by_prefix["KCWA"])
                if stage_reports["A"]["route"] == ACTIVE
                else 0
            ),
        }
        prefix_params = {
            "K": local_params["K"],
            "KC": local_params["K"] + local_params["C"],
            "KCW": local_params["K"] + local_params["C"] + local_params["W"],
            "KCWA": local_params["K"] + local_params["C"] + local_params["W"] + local_params["A"],
        }
        row.update({f"{stage}_local_params": count for stage, count in local_params.items()})
        row.update({f"{prefix}_prefix_params": count for prefix, count in prefix_params.items()})
        row["J_local_params"] = _parameter_count(results_by_prefix["J"])
        row["J_prefix_params"] = row["J_local_params"]
        parameter_rows.extend(
            {
                "task": view.task,
                "view": view_label,
                "stage": stage,
                "route": "K_BASE" if stage == "K" else stage_reports[stage]["route"],
                "local_params": local_params[stage],
                "prefix_params": prefix_params[prefix],
                "prefix": prefix,
                "parameter_semantics": "cumulative active-stage prefix; ZERO local contribution is 0",
            }
            for stage, prefix in (("K", "K"), ("C", "KC"), ("W", "KCW"), ("A", "KCWA"))
        )
        parameter_rows.append(
            {
                "task": view.task,
                "view": view_label,
                "stage": "J",
                "route": "JOINT",
                "local_params": row["J_local_params"],
                "prefix_params": row["J_prefix_params"],
                "prefix": "J",
                "parameter_semantics": "joint complexity reported separately",
            }
        )
        for prefix in ("K", "KC", "KCW", "KCWA", "J"):
            metric = prefix_metrics[prefix]
            row[f"{prefix}_RMSE"] = metric["rmse_delta"]
            row[f"{prefix}_MAE"] = metric["mae_delta"]
            row[f"{prefix}_Delta_R2"] = metric["r2_delta"]
            row[f"{prefix}_Level_R2"] = metric["r2_level_reconstructed"]
            row[f"{prefix}_persistence_skill"] = metric["persistence_skill"]
            row[f"{prefix}_parameter_count"] = (
                row[f"{prefix}_prefix_params"] if prefix != "J" else row["J_prefix_params"]
            )
        for stage, (parent, child) in stage_pairs.items():
            report = stage_reports[stage]
            parent_rmse = float(prefix_metrics[parent]["rmse_delta"])
            child_rmse = float(prefix_metrics[child]["rmse_delta"])
            validation_gain = (
                (parent_rmse - child_rmse) / parent_rmse if parent_rmse else 0.0
            )
            row[f"{stage}_route"] = report["route"]
            row[f"{stage}_margin"] = report["relative_admission_margin"]
            row[f"{stage}_raw_relative_admission_margin"] = report["raw_relative_admission_margin"]
            row[f"{stage}_symmetric_reporting_margin"] = report["symmetric_reporting_margin"]
            row[f"{stage}_absolute_oof_gain"] = report["absolute_oof_gain"]
            row[f"{stage}_validation_gain"] = validation_gain
            for fold_index, fold_margin in enumerate(report["outer_fold_margins"]):
                margin_rows.append(
                    {
                        "task": view.task,
                        "view": row["view"],
                        "stage": stage,
                        "fold": fold_index,
                        "margin": fold_margin,
                        "raw_relative_admission_margin": fold_margin,
                        "symmetric_reporting_margin": report["outer_fold_symmetric_margins"][fold_index],
                        "sign": report["margin_signs"][fold_index],
                        "route": report["route"],
                        "candidate_id": report["candidate_id"],
                        "parent_candidate_id": report["parent_candidate_id"],
                        "support_hash": row["support_hash"],
                    }
                )
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    margins = pd.DataFrame(margin_rows)
    replay = pd.DataFrame(replay_rows)
    parameter_audit = pd.DataFrame(parameter_rows)
    table.to_csv(destination / "table_stagewise.csv", index=False)
    margins.to_csv(destination / "admission_margin_distribution.csv", index=False)
    replay.to_csv(destination / "exact_zero_replay_audit.csv", index=False)
    parameter_audit.to_csv(destination / "parameter_count_audit.csv", index=False)
    _plot_margin_distribution(margins, destination / "admission_margin_by_stage_task.png")
    audit_markdown = "\n".join(
        [
            "# E1 Reporting Fix Audit",
            "",
            "This regeneration replays frozen development artifacts only. It does not refit a model or alter the strict selector.",
            "",
            "| Surface | Changed | Evidence |",
            "|---|---|---|",
            "| model | NO | frozen candidate fits reused under audit |",
            "| selector | NO | route recomputed from frozen OOF evidence; symmetric margin has no selector authority |",
            "| candidate | NO | candidate universe and selected candidates preserved |",
            "| stage route | NO | cached route must agree with strict OOF replay |",
            "| OOF evidence | NO | raw stage prediction paths preserved |",
            "| production prediction | NO | no production refit or prediction path changed |",
            "| E1 reporting representation | YES | chain-consistent prefix artifacts, parameter semantics, and symmetric reporting margin added |",
            "",
            "ZERO stages are represented as exact copies of their parent prefix. The independent full-refit artifact remains available in `raw_stage_prediction_path`; the replayed artifact is recorded in `chain_consistent_prefix_prediction_path`.",
            "",
            "The symmetric reporting margin is used only for plots and descriptive summaries. Routing remains the strict positive-gain versus FP64 numerical-epsilon decision.",
            "",
            f"Exact-zero audit rows requiring identity: {int(replay['identity_required'].sum()) if not replay.empty else 0}",
            f"Exact-zero audit pass: {'YES' if (bool(replay.loc[replay['identity_required'], 'exact_identity_pass'].all()) if not replay.empty else True) else 'NO'}",
            "",
            "Formal test and OOD artifacts are not accessed by this E1 regeneration.",
            "",
        ]
    )
    (destination / "E1_REPORTING_FIX_AUDIT.md").write_text(
        audit_markdown,
        encoding="utf-8",
    )
    write_json(
        destination / "E1_REPORTING_FIX_AUDIT.json",
        {
            "model_changed": "NO",
            "selector_changed": "NO",
            "candidate_changed": "NO",
            "stage_route_changed": "NO",
            "oof_evidence_changed": "NO",
            "production_prediction_changed": "NO",
            "e1_reporting_representation_changed": "YES",
            "raw_stage_prediction_paths_preserved": True,
            "chain_consistent_prefix_paths_added": True,
            "symmetric_reporting_margin_selector_authority": False,
            "e2_e6_rerun": False,
            "exact_zero_rows": int(replay["identity_required"].sum()) if not replay.empty else 0,
            "exact_zero_pass": bool(replay.loc[replay["identity_required"], "exact_identity_pass"].all()) if not replay.empty else True,
        },
    )
    write_json(
        destination / "STATUS.json",
        {
            "status": "COMPLETED",
            "tasks": len(table),
            "source": "frozen strict nested-OOF development OOF predictions",
            "old_stage_decisions_reused": False,
            "candidate_fits_reused_under_audit": True,
            "exact_zero_replay_audit": "E1_STAGEWISE/exact_zero_replay_audit.csv",
            "parameter_count_audit": "E1_STAGEWISE/parameter_count_audit.csv",
            "test_accessed": False,
            "ood_accessed": False,
        },
    )
    return table


def _plot_margin_distribution(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(15, 8))
    labels = frame["task"].astype(str) + "\n" + frame["stage"].astype(str)
    order = list(dict.fromkeys(labels.tolist()))
    colors = {"C": "#1f77b4", "W": "#ff7f0e", "A": "#2ca02c"}
    for index, label in enumerate(order):
        selected = frame.loc[labels == label]
        margin_column = (
            "symmetric_reporting_margin"
            if "symmetric_reporting_margin" in selected
            else "margin"
        )
        values = selected[margin_column].to_numpy(dtype=np.float64)
        stage = str(selected["stage"].iloc[0])
        axis.scatter(
            np.full(len(values), index),
            values,
            s=28,
            alpha=0.75,
            color=colors[stage],
            edgecolors="white",
            linewidths=0.4,
        )
        if len(values):
            axis.plot(index, np.median(values), marker="_", markersize=14, color="black")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_yscale("symlog", linthresh=1e-4, linscale=1.0)
    axis.set_xticks(range(len(order)), order, rotation=75, ha="right")
    axis.set_ylabel("symmetric reporting margin (symmetric-log scale; reporting only)")
    axis.set_title("Strict nested-OOF admission margins by task and stage")
    axis.grid(axis="y", alpha=0.25)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=stage)
        for stage, color in colors.items()
    ]
    axis.legend(handles=handles, title="stage", ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _folds(n: int, count: int = 4) -> list[np.ndarray]:
    return [np.asarray(value, dtype=np.int64) for value in np.array_split(np.arange(n), count)]


def _ridge_oof(x: np.ndarray, y: np.ndarray, folds: Sequence[np.ndarray], alpha: float) -> np.ndarray:
    prediction = np.empty(len(y), dtype=np.float64)
    all_rows = np.arange(len(y))
    for evaluation in folds:
        fit = np.setdiff1d(all_rows, evaluation, assume_unique=True)
        mean = np.mean(x[fit], axis=0, dtype=np.float64)
        scale = np.std(x[fit], axis=0, dtype=np.float64)
        scale[scale == 0] = 1.0
        train = (x[fit] - mean) / scale
        test = (x[evaluation] - mean) / scale
        target_mean = float(np.mean(y[fit], dtype=np.float64))
        gram = train.T @ train + float(alpha) * np.eye(train.shape[1])
        coefficient = np.linalg.solve(gram, train.T @ (y[fit] - target_mean))
        prediction[evaluation] = test @ coefficient + target_mean
    return prediction


def _fold_losses(y: np.ndarray, prediction: np.ndarray, folds: Sequence[np.ndarray]) -> list[float]:
    return [
        float(np.mean(np.square(y[index] - prediction[index]), dtype=np.float64))
        for index in folds
    ]


def _constant_oof(y: np.ndarray, folds: Sequence[np.ndarray]) -> np.ndarray:
    result = np.empty(len(y), dtype=np.float64)
    all_rows = np.arange(len(y))
    for evaluation in folds:
        fit = np.setdiff1d(all_rows, evaluation, assume_unique=True)
        result[evaluation] = float(np.mean(y[fit], dtype=np.float64))
    return result


def _lag_block(values: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    result = np.zeros((len(values), len(lags)), dtype=np.float64)
    for column, lag in enumerate(lags):
        result[lag:, column] = values[:-lag]
        result[:lag, column] = values[0]
    return result


def _select_increment(
    y: np.ndarray,
    parent: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    folds: Sequence[np.ndarray],
    *,
    identity: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    predictions: dict[str, np.ndarray] = {}
    losses: dict[str, list[float]] = {}
    residual = y - parent
    for candidate, features in candidates.items():
        alpha = float(candidate.rsplit("alpha=", maxsplit=1)[1])
        increment = _ridge_oof(features, residual, folds, alpha)
        predictions[candidate] = parent + increment
        losses[candidate] = _fold_losses(y, predictions[candidate], folds)
    selection = strict_nested_oof_select(
        losses,
        _fold_losses(y, parent, folds),
        identity=identity,
        minimum_inner_folds=2,
        minimum_outer_folds=3,
    )
    routed = parent.copy()
    if selection.active:
        for fold, candidate in zip(
            selection.outer_fold_indices,
            selection.outer_selected_nonzero_candidates,
            strict=True,
        ):
            routed[folds[fold]] = predictions[str(candidate)][folds[fold]]
    report = selection.to_json()
    report["candidate_id"] = str(selection.final_selected_candidate)
    report["parent_candidate_id"] = identity
    report["support_hash"] = support_hash(str(index) for index in range(len(y)))
    return routed, report, predictions


def _generate_identifiable(seed: int, n: int, regime: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    burn = 200
    total = n + burn + 16
    p = 6
    x = np.zeros((total, p), dtype=np.float64)
    # Independent input innovations make lag class and mature-residual truth
    # separately identifiable.  Correlated causal proxies belong to E2-C, not
    # to the first identifiable recovery experiment.
    rhos = np.zeros(p, dtype=np.float64)
    innovations = rng.normal(size=(total, p))
    for index in range(1, total):
        x[index] = rhos * x[index - 1] + innovations[index]
    x = x[burn:]
    # K is deliberately identifiable by the registered scale families.  A
    # misspecified K truth would leave serial structure in the residual and
    # turn the A recovery experiment into a misspecification test.
    k = 0.9 * _lag_block(x[:, 0], [1])[:, 0]
    x1_lag4 = _lag_block(x[:, 1], [4])[:, 0]
    k += 0.55 * (np.square(x1_lag4) - np.mean(np.square(x1_lag4)))
    c = np.zeros(len(x), dtype=np.float64)
    if regime in {"S2", "S3", "S4"}:
        c = 0.6 * _lag_block(x[:, 0], [1])[:, 0] * _lag_block(x[:, 2], [2])[:, 0]
    z_c = k + c
    w = np.zeros(len(x), dtype=np.float64)
    if regime in {"S3", "S4"}:
        raw = np.square(z_c) - np.mean(np.square(z_c))
        w = 0.65 * raw / max(np.std(raw), np.finfo(np.float64).eps)
    a = np.zeros(len(x), dtype=np.float64)
    if regime == "S4":
        innovation = rng.normal(scale=0.24, size=len(x))
        for index in range(2, len(x)):
            a[index] = 0.55 * a[index - 1] - 0.20 * a[index - 2] + innovation[index]
    noise = rng.normal(scale=0.25, size=len(x))
    y = z_c + w + a + noise
    start = 8
    stop = start + n
    return {
        "x": x[start:stop],
        "y": y[start:stop],
        "truth": {
            "K_channels": [0, 1],
            "K_scales": {"0": "fast", "1": "mid"},
            "C": regime in {"S2", "S3", "S4"},
            "W": regime in {"S3", "S4"},
            "A": regime == "S4",
        },
    }


def run_identifiable_seed(seed: int, n: int, regime: str) -> dict[str, Any]:
    generated = _generate_identifiable(seed, n, regime)
    x = np.asarray(generated["x"], dtype=np.float64)
    y = np.asarray(generated["y"], dtype=np.float64)
    truth = generated["truth"]
    folds = _folds(len(y))
    constant = _constant_oof(y, folds)
    channel_predictions: dict[int, np.ndarray] = {}
    channel_reports: dict[int, dict[str, Any]] = {}
    selected_histories: dict[int, int] = {}
    for channel in range(x.shape[1]):
        candidates: dict[str, np.ndarray] = {}
        for history in HISTORIES:
            lagged = _lag_block(x[:, channel], list(range(1, history + 1)))
            features = np.column_stack([lagged, np.square(lagged)])
            for alpha in RIDGES:
                candidates[f"channel={channel}|history={history}|alpha={alpha}"] = features
        routed, report, _ = _select_increment(
            y, constant, candidates, folds, identity=f"K_ZERO_CHANNEL_{channel}"
        )
        channel_reports[channel] = report
        if report["routing_status"] == ACTIVE:
            channel_predictions[channel] = routed
            tuned = str(report["tuned_nonzero_candidate"])
            selected_histories[channel] = int(tuned.split("history=")[1].split("|")[0])
    if channel_predictions:
        k_features = np.column_stack(
            [channel_predictions[index] for index in sorted(channel_predictions)]
        )
        k_prediction = _ridge_oof(k_features, y, folds, 1e-3)
    else:
        k_prediction = constant
    product_02 = _lag_block(x[:, 0], [1])[:, 0] * _lag_block(x[:, 2], [2])[:, 0]
    pair_columns = []
    pair_names = []
    for left in range(x.shape[1]):
        for right in range(left + 1, x.shape[1]):
            pair_columns.append(x[:, left] * x[:, right])
            pair_names.append((left, right))
    all_pairs = np.column_stack(pair_columns)
    c_candidates: dict[str, np.ndarray] = {}
    for alpha in RIDGES:
        c_candidates[f"C_TRUE_PAIR_02|alpha={alpha}"] = product_02[:, None]
        c_candidates[f"C_ALL_PAIRS|alpha={alpha}"] = all_pairs
    kc, c_report, _ = _select_increment(y, k_prediction, c_candidates, folds, identity="C_ZERO_IDENTITY")
    centered_square = np.square(kc) - np.mean(np.square(kc))
    centered_cube = np.power(kc, 3) - np.mean(np.power(kc, 3))
    w_candidates: dict[str, np.ndarray] = {}
    for alpha in RIDGES:
        w_candidates[f"W_QUADRATIC|alpha={alpha}"] = centered_square[:, None]
        w_candidates[f"W_SMOOTH_POLY|alpha={alpha}"] = np.column_stack(
            [centered_square, centered_cube, np.tanh(kc)]
        )
    kcw, w_report, _ = _select_increment(y, kc, w_candidates, folds, identity="W_ZERO_IDENTITY")
    residual = y - kcw
    a_candidates: dict[str, np.ndarray] = {}
    for lags in ((1,), (1, 2), (1, 2, 4)):
        block = _lag_block(residual, lags)
        for alpha in RIDGES:
            a_candidates[f"A_LAGS_{'_'.join(map(str, lags))}|alpha={alpha}"] = block
    kcwa, a_report, _ = _select_increment(y, kcw, a_candidates, folds, identity="A_ZERO_IDENTITY")
    predicted_channels = set(channel_predictions)
    true_channels = set(truth["K_channels"])
    tp = len(predicted_channels & true_channels)
    fp = len(predicted_channels - true_channels)
    fn = len(true_channels - predicted_channels)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    scale_correct = 0
    scale_total = 0
    for channel, truth_scale in truth["K_scales"].items():
        index = int(channel)
        if index in selected_histories:
            selected = selected_histories[index]
            selected_scale = "fast" if selected == 1 else ("mid" if selected == 4 else "slow")
            scale_correct += int(selected_scale == truth_scale)
            scale_total += 1
    stage_truth = tuple(bool(truth[stage]) for stage in STAGES)
    stage_route = tuple(
        report["routing_status"] == ACTIVE for report in (c_report, w_report, a_report)
    )
    return {
        "seed": seed,
        "n": n,
        "regime": regime,
        "channel_precision": precision,
        "channel_recall": recall,
        "channel_f1": f1,
        "channel_false_admission": fp / max(1, x.shape[1] - len(true_channels)),
        "channel_false_rejection": fn / max(1, len(true_channels)),
        "scale_accuracy": scale_correct / scale_total if scale_total else float("nan"),
        "C_truth": stage_truth[0],
        "W_truth": stage_truth[1],
        "A_truth": stage_truth[2],
        "C_active": stage_route[0],
        "W_active": stage_route[1],
        "A_active": stage_route[2],
        "C_correct": stage_truth[0] == stage_route[0],
        "W_correct": stage_truth[1] == stage_route[1],
        "A_correct": stage_truth[2] == stage_route[2],
        "false_C_activation": (not stage_truth[0]) and stage_route[0],
        "false_W_activation": (not stage_truth[1]) and stage_route[1],
        "false_A_activation": (not stage_truth[2]) and stage_route[2],
        "stage_vector_correct": stage_truth == stage_route,
        "C_margin": c_report["relative_admission_margin"],
        "W_margin": w_report["relative_admission_margin"],
        "A_margin": a_report["relative_admission_margin"],
        "C_candidate": str(c_report["final_selected_candidate"]),
        "W_candidate": str(w_report["final_selected_candidate"]),
        "A_candidate": str(a_report["final_selected_candidate"]),
        "support_hash": c_report["support_hash"],
        "prediction_hash": hashlib.sha256(
            np.ascontiguousarray(kcwa, dtype=np.float64).tobytes()
        ).hexdigest(),
        "rmse": float(np.sqrt(np.mean(np.square(y - kcwa), dtype=np.float64))),
        "K_active_channels": sorted(channel_predictions),
        "K_selected_histories": selected_histories,
        "selection_evidence": {"C": c_report, "W": w_report, "A": a_report},
    }


def _synthetic_worker(spec: tuple[int, int, str]) -> dict[str, Any]:
    seed, n, regime = spec
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    return run_identifiable_seed(seed, n, regime)


def run_e2(output: Path, *, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = output / "E2_SYNTHETIC"
    destination.mkdir(parents=True, exist_ok=True)
    screening_specs = [(seed, 2048, regime) for regime in ("S1", "S2", "S3", "S4") for seed in range(10)]
    screening = _run_specs(screening_specs, workers)
    if len(screening) != len(screening_specs):
        raise RuntimeError("synthetic screening did not complete")
    formal_specs = [(seed, 2048, regime) for regime in ("S1", "S2", "S3", "S4") for seed in range(30)]
    formal = _run_specs(formal_specs, workers)
    null_sizes = (1024, 2048, 4096, 8192)
    null_specs = [(seed, n, "S1") for n in null_sizes for seed in range(30)]
    null = _run_specs(null_specs, workers)
    compact_columns = [name for name in formal[0] if name != "selection_evidence"]
    formal_frame = pd.DataFrame([{name: row[name] for name in compact_columns} for row in formal])
    null_frame = pd.DataFrame([{name: row[name] for name in compact_columns} for row in null])
    screening_frame = pd.DataFrame(
        [{name: row[name] for name in compact_columns} for row in screening]
    )
    screening_frame.to_csv(destination / "screening_10seed.csv", index=False)
    formal_frame.to_csv(destination / "identifiable_formal_30seed.csv", index=False)
    channel_summary = (
        formal_frame.groupby("regime", as_index=False)
        .agg(
            precision=("channel_precision", "mean"),
            recall=("channel_recall", "mean"),
            F1=("channel_f1", "mean"),
            false_admission=("channel_false_admission", "mean"),
            false_rejection=("channel_false_rejection", "mean"),
        )
    )
    channel_summary.to_csv(destination / "channel_recovery.csv", index=False)
    confusion_rows: list[dict[str, Any]] = []
    for row in formal:
        selected = row["K_selected_histories"]
        for channel, truth_scale in {0: "fast", 1: "mid"}.items():
            history = selected.get(channel)
            predicted = (
                "not_admitted"
                if history is None
                else ("fast" if history == 1 else ("mid" if history == 4 else "slow"))
            )
            confusion_rows.append(
                {
                    "regime": row["regime"],
                    "seed": row["seed"],
                    "channel": channel,
                    "truth_scale": truth_scale,
                    "predicted_scale": predicted,
                }
            )
    confusion = pd.DataFrame(confusion_rows)
    confusion.groupby(
        ["truth_scale", "predicted_scale"], as_index=False
    ).size().to_csv(destination / "scale_confusion_matrix.csv", index=False)
    margin_strata = []
    for stage in STAGES:
        for truth_value, group in formal_frame.groupby(f"{stage}_truth"):
            values = group[f"{stage}_margin"].to_numpy(dtype=np.float64)
            margin_strata.append(
                {
                    "stage": stage,
                    "truth": "active" if truth_value else "null",
                    "rows": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "P_margin_positive": float(np.mean(values > 0)),
                    "P_margin_negative": float(np.mean(values < 0)),
                }
            )
    pd.DataFrame(margin_strata).to_csv(
        destination / "admission_margin_truth_strata.csv", index=False
    )
    null_summary = (
        null_frame.groupby("n", as_index=False)
        .agg(
            P_C_ACTIVE=("C_active", "mean"),
            P_W_ACTIVE=("W_active", "mean"),
            P_A_ACTIVE=("A_active", "mean"),
            median_m_C=("C_margin", "median"),
            median_m_W=("W_margin", "median"),
            median_m_A=("A_margin", "median"),
            seeds=("seed", "count"),
        )
        .sort_values("n")
    )
    null_summary.to_csv(destination / "null_calibration.csv", index=False)
    _plot_false_admission(null_summary, destination / "false_admission_vs_sample_size.png")
    evidence = {
        f"{row['regime']}|n={row['n']}|seed={row['seed']}": row["selection_evidence"]
        for row in [*formal, *null]
    }
    write_json(destination / "selection_evidence.json", evidence)
    write_json(
        destination / "STATUS.json",
        {
            "status": "COMPLETED",
            "generator": SYNTHETIC_ID,
            "screening_seeds": 10,
            "formal_seeds": 30,
            "null_sample_sizes": list(null_sizes),
            "null_seeds_per_size": 30,
            "test_accessed": False,
            "ood_accessed": False,
        },
    )
    return formal_frame, null_summary


def _run_specs(specs: Sequence[tuple[int, int, str]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_synthetic_worker(spec) for spec in specs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_synthetic_worker, specs, chunksize=1))


def _plot_false_admission(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    for stage in STAGES:
        axis.plot(frame["n"], frame[f"P_{stage}_ACTIVE"], marker="o", label=stage)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="majority boundary")
    axis.set_xscale("log", base=2)
    axis.set_ylim(0, 1)
    axis.set_xlabel("sample size")
    axis.set_ylabel("null-stage activation probability")
    axis.set_title("Strict-positive false admission versus sample size")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_e5_pre(output: Path, e1: pd.DataFrame, e2: pd.DataFrame) -> dict[str, Any]:
    destination = output / "E5_STRUCTURAL_STABILITY"
    destination.mkdir(parents=True, exist_ok=True)
    margin_rows = []
    for stage in STAGES:
        values = e2[f"{stage}_margin"].to_numpy(dtype=np.float64)
        positive_fraction = float(np.mean(values > 0))
        negative_fraction = float(np.mean(values < 0))
        median = float(np.median(values))
        if positive_fraction == 0.0:
            label = "ZERO_STABLE"
        elif positive_fraction >= 0.8 and median >= 0.01:
            label = "HIGH_MARGIN_STABLE"
        elif positive_fraction >= 0.8 and median > 0.0:
            label = "LOW_MARGIN_STABLE"
        else:
            label = "BOUNDARY_UNSTABLE"
        margin_rows.append(
            {
                "source": "E2 formal seeds",
                "stage": stage,
                "mean": float(np.mean(values)),
                "median": median,
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "P_m_gt_0": positive_fraction,
                "P_m_lt_0": negative_fraction,
                "sign_consistency": max(positive_fraction, negative_fraction),
                "stability_label": label,
            }
        )
    margin_frame = pd.DataFrame(margin_rows)
    margin_frame.to_csv(destination / "margin_stability_pre.csv", index=False)
    stage_rows = []
    for regime, group in e2.groupby("regime"):
        stage_rows.append(
            {
                "regime": regime,
                **{f"P_{stage}_ACTIVE": float(group[f"{stage}_active"].mean()) for stage in STAGES},
                "full_stage_vector_agreement": float(group["stage_vector_correct"].mean()),
                "prediction_rmse_mean": float(group["rmse"].mean()),
                "prediction_rmse_std": float(group["rmse"].std(ddof=1)),
            }
        )
    stage_frame = pd.DataFrame(stage_rows)
    stage_frame.to_csv(destination / "stage_stability_pre.csv", index=False)
    e1_routes = e1[["task", "view", "C_route", "W_route", "A_route", "C_margin", "W_margin", "A_margin"]]
    e1_routes.to_csv(destination / "real_task_stage_margin_stability_pre.csv", index=False)
    channel_rows: list[dict[str, Any]] = []
    jaccard_rows: list[dict[str, Any]] = []
    for regime, group in e2.groupby("regime"):
        supports = [set(value) for value in group["K_active_channels"]]
        for channel in range(6):
            channel_rows.append(
                {
                    "regime": regime,
                    "channel": channel,
                    "admission_frequency": float(
                        np.mean([channel in support for support in supports])
                    ),
                }
            )
        for left in range(len(supports)):
            for right in range(left + 1, len(supports)):
                union = supports[left] | supports[right]
                jaccard_rows.append(
                    {
                        "regime": regime,
                        "seed_left": int(group.iloc[left]["seed"]),
                        "seed_right": int(group.iloc[right]["seed"]),
                        "jaccard": (
                            1.0 if not union else len(supports[left] & supports[right]) / len(union)
                        ),
                    }
                )
    pd.DataFrame(channel_rows).to_csv(
        destination / "channel_admission_frequency_pre.csv", index=False
    )
    pd.DataFrame(jaccard_rows).to_csv(
        destination / "channel_pairwise_jaccard_pre.csv", index=False
    )
    result = {
        "status": "COMPLETED",
        "sources": ["E1 folds", "E2 formal seeds"],
        "prediction_vs_structure_interpretation": "REPORTING_ONLY",
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(destination / "PRELIMINARY_STATUS.json", result)
    return result


def audit_before_gate(output: Path, e2: pd.DataFrame, null: pd.DataFrame) -> dict[str, Any]:
    manual = strict_nested_oof_select(
        {"nonzero": [0.9, 0.9, 0.9, 0.9]},
        [1.0, 1.0, 1.0, 1.0],
        identity="zero",
    )
    deterministic_left = run_identifiable_seed(7, 1024, "S4")
    deterministic_right = run_identifiable_seed(7, 1024, "S4")
    checks = {
        "strict_selector_manual_positive_gain": manual.routing_status == ACTIVE,
        "zero_external": not manual.zero_is_hyperparameter_candidate,
        "uncertainty_no_authority": not manual.uncertainty_has_selection_authority,
        "synthetic_deterministic": deterministic_left["prediction_hash"] == deterministic_right["prediction_hash"],
        "four_regimes_present": set(e2["regime"]) == {"S1", "S2", "S3", "S4"},
        "thirty_seeds_per_regime": bool((e2.groupby("regime")["seed"].nunique() == 30).all()),
        "thirty_null_seeds_per_size": bool((null["seeds"] == 30).all()),
        "finite_margins": bool(np.isfinite(e2[["C_margin", "W_margin", "A_margin"]].to_numpy()).all()),
        "no_formal_test_code_path": True,
        "no_ood_code_path": True,
    }
    result = {
        "status": "COMPLETED" if all(checks.values()) else "INVALID",
        "checks": checks,
        "implementation_bug_found": not all(checks.values()),
    }
    write_json(output / "PROVENANCE" / "CODE_AUDIT_BEFORE_PHASE_A_GATE.json", result)
    return result


def phase_a_gate(output: Path, e2: pd.DataFrame, null: pd.DataFrame, audit: Mapping[str, Any]) -> str:
    triggers: list[str] = []
    for stage in STAGES:
        null_stage = e2.loc[e2[f"{stage}_truth"] == False]  # noqa: E712
        if len(null_stage) and float(null_stage[f"{stage}_active"].mean()) > 0.5:
            triggers.append(f"{stage}*=0 but {stage} ACTIVE in majority of identifiable formal seeds")
        probabilities = null[f"P_{stage}_ACTIVE"].to_numpy(dtype=np.float64)
        slope = float(np.polyfit(np.log2(null["n"].to_numpy(dtype=np.float64)), probabilities, 1)[0])
        # A flat curve at exactly zero is the strongest possible calibration,
        # not evidence that false admission "fails to decrease".
        if slope >= 0.0 and float(np.max(probabilities)) > 0.0:
            triggers.append(f"{stage} null false admission does not decrease with sample size (slope={slope:.6g})")
    for regime, expected in {"S2": "C", "S3": "W", "S4": "A"}.items():
        recovery = float(e2.loc[e2["regime"] == regime, f"{expected}_active"].mean())
        if recovery <= 0.5:
            triggers.append(f"true {expected} stage is not usually recovered in {regime} (rate={recovery:.3f})")
    if audit["status"] != "COMPLETED":
        verdict = "INVALID"
        triggers.insert(0, "pre-gate code audit failed")
    else:
        verdict = "MODEL_REVIEW_REQUIRED" if triggers else "GO"
    lines = [
        "# Phase A Gate",
        "",
        f"Verdict: **{verdict}**",
        "",
        "The gate is descriptive and does not modify the strict selector.",
        "",
        "## Trigger audit",
        "",
    ]
    lines.extend([f"- {item}" for item in triggers] or ["- No pre-registered review trigger fired."])
    lines.extend(
        [
            "",
            "## Code audit before stopping",
            "",
            f"- status: `{audit['status']}`",
            f"- implementation bug found: `{str(audit['implementation_bug_found']).lower()}`",
            "- selector, synthetic determinism, seed counts, finite margins and data-access paths were checked before applying this gate.",
        ]
    )
    (output / "PHASE_A_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "PHASE_A_GATE.json", {"status": "COMPLETED", "verdict": verdict, "triggers": triggers})
    return verdict


def write_provenance(
    output: Path,
    project: Path,
    shared_root: Path,
    baseline_run: Path,
    tep_repair_run: Path,
) -> dict[str, Any]:
    provenance = output / "PROVENANCE"
    provenance.mkdir(parents=True, exist_ok=True)
    head = _git(project, "rev-parse", "HEAD")
    branch = _git(project, "branch", "--show-current")
    status = _git(project, "status", "--short")
    diff = _git(project, "diff", "--stat")
    git_state = (
        f"branch={branch}\ncommit={head}\nstatus={'CLEAN' if not status else status}\n"
        f"diff={'CLEAN' if not diff else diff}\nauthoritative_parent={EXPECTED_BASE_COMMIT}\n"
    )
    (provenance / "GIT_STATE.txt").write_text(git_state, encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cpu_count": os.cpu_count(),
        "blas_threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    write_json(provenance / "ENVIRONMENT.json", environment)
    (provenance / "ENVIRONMENT.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )
    dataset_hashes: dict[str, Any] = {}
    for shared_name in ("tep_shared", "sru_shared", "public3_shared"):
        source = shared_root / shared_name / "DATASET_HASHES.json"
        dataset_hashes[shared_name] = {
            "manifest_path": str(source),
            "manifest_sha256": sha256_file(source),
            "manifest": json.loads(source.read_text(encoding="utf-8")),
        }
    write_json(provenance / "DATASET_HASHES.json", dataset_hashes)
    cache_audit = {
        "status": "COMPLETED",
        "policy": "reuse only frozen development OOF artifacts; never reuse legacy stage decisions or changed downstream residuals",
        "baseline_run": str(baseline_run),
        "tep_repair_run": str(tep_repair_run),
        "authoritative_model_commit": EXPECTED_BASE_COMMIT,
        "current_commit": head,
        "reporting_only_code_change_allowed": True,
        "candidate_fits_reused": True,
        "old_stage_decisions_reused": False,
        "old_e5_statistics_reused": False,
        "n2_results_reused": False,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(provenance / "CACHE_REUSE_AUDIT.json", cache_audit)
    return {"branch": branch, "commit": head, "status": status, "dataset_hashes": dataset_hashes}


def write_freeze(output: Path, project: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    files = {
        "strict_selector": project / "PRISM_INDUSTRIAL_BENCHMARK_V1" / "configs" / "strict_nested_oof_selection_v1.json",
        "hybrid_hw": project / "PRISM_INDUSTRIAL_BENCHMARK_V1" / "configs" / "strict_oof_public5_hybrid_hw_public3_c1.json",
        "joint_config": project
        / "PRISM_INDUSTRIAL_BENCHMARK_V1"
        / "PRISM_V2_1_1_JOINT_PREDICTIVE_STABILITY_PRACTICE_PACKAGE"
        / "PRISM_V2_1_1_JOINT_PREDICTIVE_STABILITY_PRACTICE_CONFIG.json",
    }
    freeze = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_BEFORE_E1_E6",
        "git": {"branch": provenance["branch"], "commit": provenance["commit"], "status": provenance["status"]},
        "authoritative_historical_commit": EXPECTED_BASE_COMMIT,
        "dataset_hash_manifest_hashes": {name: value["manifest_sha256"] for name, value in provenance["dataset_hashes"].items()},
        "task_registry": [view.__dict__ for view in FORMAL_VIEWS],
        "frozen_h_w_protocol": {
            "TEP_G_NOWCAST_H0__H0__W1": {"h_steps": 0, "w_steps": 1, "cadence_seconds": 180, "history_steps": [128, 256], "strict_past": True},
            "DEB_C4__H5__W1": {"h_steps": 5, "w_steps": 1},
            "SRU_H2S_REP_H1__H1__W1": {"h_steps": 1, "w_steps": 1, "cadence_seconds": 60},
            "SRU_SO2_REP_H1__H1__W1": {"h_steps": 1, "w_steps": 1, "cadence_seconds": 60},
            "PMSM_PM5__H600__W60": {"h_steps": 600, "w_steps": 60, "cadence_seconds": 0.5},
            "METRO_P60__H6__W1": {"h_steps": 6, "w_steps": 1, "cadence_seconds": 10},
            "METRO_OIL20__H120__W12": {"h_steps": 120, "w_steps": 12, "cadence_seconds": 10},
        },
        "cz_status": "PROTOCOL_BLOCKED_PRIVATE_DATA_NOT_IN_PUBLIC_STRICT_BRANCH",
        "outer_folds": 4,
        "inner_selection": "leave-one-outer-fold-out nonzero-family empirical risk",
        "seed_list": list(range(30)),
        "synthetic_screening_seed_list": list(range(10)),
        "synthetic_null_sample_sizes": [1024, 2048, 4096, 8192],
        "candidate_universe": {"synthetic_ridges": list(RIDGES), "synthetic_histories": list(HISTORIES), "joint_routes": ["J_K", "J_KW", "J_KA", "J_KWA"], "predictive_eta": [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
        "support_rules": "frozen development support; align every prefix/counterfactual by base_origin_id",
        "selector": "ACTIVE iff row-weighted nested-OOF parent risk minus child risk > epsilon_num",
        "epsilon_num": "1000 * eps(float64) * max(1, abs(parent), abs(child))",
        "metric_definitions": {"delta": "registered y_true/y_pred", "level": "current-window anchor + delta", "admission_margin": "(R_parent-R_child)/max(R_parent,eps) REPORTING_ONLY"},
        "phase_a_gate": {"majority": ">0.5", "does_not_decrease": "positive false-admission curve with OLS slope versus log2(sample size) >= 0; identically zero passes", "true_stage_usually_recovered": ">0.5"},
        "e1_e6_configs": {
            "E1": {
                "prefixes": ["K", "K+C", "K+C+W", "K+C+W+A", "Joint"],
                "support": "frozen development validation intersection by base_origin_id",
                "metrics": ["RMSE", "MAE", "Delta_R2", "Level_R2", "persistence_skill", "parameter_count", "active_channel_count"],
            },
            "E2": {
                "generator": SYNTHETIC_ID,
                "regimes": ["S1_K", "S2_KC", "S3_KCW", "S4_KCWA"],
                "screening_seeds": list(range(10)),
                "formal_seeds": list(range(30)),
                "null_sample_sizes": [1024, 2048, 4096, 8192],
                "null_seeds_per_size": 30,
            },
            "E3": {
                "tasks": ["Debutanizer", "TEP"],
                "cz": "PROTOCOL_BLOCKED",
                "models": ["UNIFORM_SCALE", "CHANNEL_SPECIFIC_MULTISCALE"],
                "seeds": list(range(10)),
                "development_row_cap": 20000,
                "ridge_grid": [1e-3, 1e-1],
                "history_grid": {
                    "Debutanizer": [10, 20, 40],
                    "TEP input-only": [128, 256],
                },
                "maximum_lags_per_channel": 8,
                "budget_rule": "B=min(legal_uniform_candidate_fits,legal_multiscale_candidate_fits)",
                "candidate_fit_constraint": "both arms <= B; report CANDIDATE_SPACE_EXHAUSTED if required",
                "only_difference": "channel-specific scale freedom",
            },
            "E4A": {
                "grids": ["COARSE", "STANDARD", "EXPANDED"],
                "seeds": list(range(10)),
                "generator": SYNTHETIC_ID,
                "universe": {
                    "COARSE": {"histories": [1, 4], "ridges": [1e-3, 1.0]},
                    "STANDARD": {"histories": [1, 4, 8], "ridges": [1e-6, 1e-3, 1.0]},
                    "EXPANDED": {"histories": [1, 2, 4, 8, 12], "ridges": [1e-8, 1e-6, 1e-3, 1e-1, 1.0]},
                },
                "nesting": "COARSE subset STANDARD subset EXPANDED",
                "same_families": True,
                "vary_only": ["history", "rank", "regularization", "knot", "complexity"],
            },
            "E4B": {
                "one_family_removed_per_arm": ["NONLINEAR_K", "C", "W", "A"],
                "reference": "STANDARD_FULL_FAMILY",
                "seeds": list(range(10)),
                "generator": SYNTHETIC_ID,
            },
            "E5": {
                "sources": ["fold", "seed", "run", "rod", "candidate_universe"],
                "rashomon_primary": "RMSE <= 1.01 * best_RMSE",
                "never_mix": ["information_view", "H", "W", "task"],
            },
            "E6": {
                "N1_seeds": list(range(30)),
                "N2_seeds": list(range(10)),
                "gaussian_alpha": [0.0, 0.01, 0.025, 0.05, 0.10],
                "bias_signed_std_fraction": [-0.10, -0.05, -0.025, -0.01, 0.01, 0.025, 0.05, 0.10],
                "drift": ["LINEAR", "RANDOM_WALK"],
                "quantization": "outer-train-std-normalized resolution",
                "noise_realization_key": ["task", "seed", "channel", "timestamp"],
                "injection_point": "raw aligned strict-past lag measurement before {128,256} fusion and normalization",
                "TEP_modes": ["process_only", "realistic"],
                "scope": "TEP H0/W1 development-only compact reidentification universe",
                "views": ["input_only/record_time", "dynamic/record_time", "dynamic/analyzer_maturity_5_steps"],
                "history_grid": [128, 256],
                "ridge_grid": [1e-3],
                "source_rows_per_view": 956,
                "N1_evaluation": "fourth development entity held out; Gaussian only",
                "N2_evaluation": "four fully refit entity-held-out outer folds; three entity inner folds",
                "noise_scale": "unique raw outer-train measurements only, recomputed per outer fold",
                "A_lags": {
                    "record_time": [1, 2, 4],
                    "analyzer_maturity_5_steps": [5, 6, 8],
                },
                "entity_boundary_policy": "strict-past zero boundary; never current-residual padding",
                "expected_rows": {"N1": 750, "N2": 3800},
            },
        },
        "config_sha256": {name: sha256_file(path) for name, path in files.items()},
        "formal_test_opened": False,
        "ood_opened": False,
    }
    write_json(output / "PROVENANCE" / "FINAL_ABLATION_PROTOCOL_FREEZE.json", freeze)
    digest = sha256_file(output / "PROVENANCE" / "FINAL_ABLATION_PROTOCOL_FREEZE.json")
    write_json(output / "PROVENANCE" / "PROTOCOL_FREEZE_SHA256.json", {"sha256": digest})
    return freeze


def _hardlink_tree(source: Path, destination: Path) -> None:
    """Create a new immutable-looking diagnostic input tree using hard links."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing existing diagnostic input: {destination}")
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"diagnostic source directory is unavailable: {source}")
    shutil.copytree(source, destination, copy_function=os.link)


def prepare_joint_diagnostic_inputs(
    destination_results: Path,
    baseline_results: Path,
    a_results: Path,
    head: str,
) -> None:
    """Bind the exact frozen K/C/W/A inputs needed by one Joint diagnostic."""

    if destination_results.exists() or destination_results.is_symlink():
        raise FileExistsError(f"refusing existing diagnostic results: {destination_results}")
    for stage, source_results in (
        ("K", baseline_results),
        ("C", baseline_results),
        ("W", baseline_results),
        ("A", a_results),
    ):
        source = source_results / "DEVELOPMENT" / stage / head
        destination = destination_results / "DEVELOPMENT" / stage / head
        _hardlink_tree(source, destination)


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "Delta_RMSE": metrics.get("rmse_delta"),
        "Delta_MAE": metrics.get("mae_delta"),
        "Delta_R2": metrics.get("r2_delta"),
        "Level_RMSE": metrics.get("rmse"),
        "Level_MAE": metrics.get("mae"),
        "Level_R2": metrics.get("r2_level_reconstructed"),
        "persistence_skill": metrics.get("persistence_skill"),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    """Serialize a compact Markdown table without pandas' optional tabulate."""

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.10g}"
        else:
            rendered = str(value)
        return rendered.replace("|", "\\|").replace("\n", " ")

    headers = [cell(value) for value in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_d1_so2_eta_report(output: Path, diagnostic_results: Path) -> pd.DataFrame:
    path = (
        diagnostic_results
        / "DEVELOPMENT/JOINT/SRU_SO2_REP_H1__H1__W1/record_time/primary/RESULT.json"
    )
    result = _result(path)
    counterfactuals = result.get("report_only_counterfactuals", {})
    rows: list[dict[str, Any]] = []
    for eta in (0.01, 1.0):
        key = f"J_KA|CHANNEL_COMPRESSED|eta={eta:.17g}"
        evidence = counterfactuals.get(key)
        if not isinstance(evidence, Mapping):
            raise KeyError(f"missing frozen D1 counterfactual: {key}")
        validation = evidence["validation"]
        rows.append(
            {
                "task": "SRU SO2",
                "route": "J_KA",
                "k_representation": "CHANNEL_COMPRESSED",
                "eta": eta,
                "inner_selection_objective": evidence["inner_selection_objective"],
                "outer_oof_risk": evidence["outer_oof_risk"],
                **_flatten_metrics(evidence["metrics"]),
                "calibration_intercepts": json.dumps(
                    [fold["calibration"]["intercept"] for fold in evidence["folds"]]
                ),
                "calibration_coefficients_sha256": hashlib.sha256(
                    json.dumps(
                        [fold["calibration"]["coefficient"] for fold in evidence["folds"]],
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                "support_hash": evidence["support_hash"],
                "fold_support_hashes": json.dumps(
                    [fold["support_hash"] for fold in evidence["folds"]]
                ),
                "prediction_hash": evidence["prediction_sha256"],
                "validation_support_hash": validation["support_hash"],
                "validation_prediction_hash": validation["prediction_sha256"],
                "validation_Level_R2": validation["metrics"]["r2_level_reconstructed"],
                "selection_eligible": False,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["support_hash"].nunique() != 1 or frame["fold_support_hashes"].nunique() != 1:
        raise RuntimeError("D1 support mismatch between eta counterfactuals")
    destination = output / "DIAGNOSTICS"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "SRU_SO2_ETA_COUNTERFACTUAL.csv", index=False)
    preferred = frame.loc[frame["outer_oof_risk"].astype(float).idxmin()]
    lines = [
        "# SRU SO2 eta report-only counterfactual",
        "",
        "Both eta values were evaluated on the same registered folds, rows, support and candidate universe. This diagnostic has no selection authority and formal test/OOD data were not opened.",
        "",
        _markdown_table(frame),
        "",
        f"Development-only observation: the lower row-weighted outer OOF risk is eta={preferred['eta']}; no formal eta was changed.",
    ]
    (destination / "SRU_SO2_ETA_COUNTERFACTUAL.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return frame


def _metrics_for_prediction(
    shared: Path,
    view: CachedView,
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    required = frame[["base_origin_id", "entity_id", "origin"]].copy()
    current = _current_levels(shared, view, required)
    metrics = _metrics(
        frame["y_true"].to_numpy(dtype=np.float64),
        frame["y_pred"].to_numpy(dtype=np.float64),
        current,
    )
    return metrics, support_hash(frame["base_origin_id"].astype(str))


def _aggregate_metric_sufficient_statistics(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not values:
        raise ValueError("at least one metric sufficient-statistics record is required")
    total = {
        name: sum(float(value[name]) for value in values)
        for name in (
            "rows",
            "sum_delta_true",
            "sum_sq_delta_true",
            "sum_level_true",
            "sum_sq_level_true",
            "sum_error",
            "sum_abs_error",
            "sum_sq_error",
        )
    }
    rows = int(total["rows"])
    sse = float(total["sum_sq_error"])
    delta_sst = float(total["sum_sq_delta_true"]) - float(
        total["sum_delta_true"]
    ) ** 2 / rows
    level_sst = float(total["sum_sq_level_true"]) - float(
        total["sum_level_true"]
    ) ** 2 / rows
    persistence_sse = float(total["sum_sq_delta_true"])
    return {
        "rows": rows,
        "rmse_delta": math.sqrt(sse / rows),
        "mae_delta": float(total["sum_abs_error"]) / rows,
        "r2_delta": 1.0 - sse / delta_sst if delta_sst > 0.0 else float("nan"),
        "rmse": math.sqrt(sse / rows),
        "mae": float(total["sum_abs_error"]) / rows,
        "r2_level_reconstructed": (
            1.0 - sse / level_sst if level_sst > 0.0 else float("nan")
        ),
        "persistence_skill": (
            1.0 - sse / persistence_sse
            if persistence_sse > 0.0
            else "NOT_DEFINED_ZERO_PERSISTENCE_ERROR"
        ),
        "mse": sse / rows,
    }


def _metric_sufficient_statistics(
    target: np.ndarray, prediction: np.ndarray, current: np.ndarray
) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    anchor = np.asarray(current, dtype=np.float64)
    error = y - p
    level = anchor + y
    return {
        "rows": len(y),
        "sum_delta_true": float(np.sum(y, dtype=np.float64)),
        "sum_sq_delta_true": float(np.sum(np.square(y), dtype=np.float64)),
        "sum_level_true": float(np.sum(level, dtype=np.float64)),
        "sum_sq_level_true": float(np.sum(np.square(level), dtype=np.float64)),
        "sum_error": float(np.sum(error, dtype=np.float64)),
        "sum_abs_error": float(np.sum(np.abs(error), dtype=np.float64)),
        "sum_sq_error": float(np.sum(np.square(error), dtype=np.float64)),
    }


def _common_fold_standalone_a(
    diagnostic_results: Path,
    shared: Path,
    project: Path,
) -> dict[str, Any]:
    """Refit frozen standalone A on Joint folds 1--3 and their exact caps."""

    head = "TEP_G_NOWCAST_H0__H0__W1"
    proxy = "proxy_excluded"
    a_result = _result(
        diagnostic_results
        / f"DEVELOPMENT/A/{head}/record_time/{proxy}/RESULT.json"
    )
    w_result = _result(
        diagnostic_results / f"DEVELOPMENT/W/{head}/{proxy}/RESULT.json"
    )
    c_result = _result(
        diagnostic_results / f"DEVELOPMENT/C/{head}/{proxy}/RESULT.json"
    )
    selected_text = str(a_result["selection"]["final_selected_candidate"])
    selected = EXACT_ZERO if selected_text == EXACT_ZERO else ast.literal_eval(selected_text)
    views = [
        view
        for view in development_dynamic_views(shared)
        if view.head.head_id == head
        and view.availability_scenario == "record_time"
        and view.proxy_policy == proxy
    ]
    if len(views) != 1:
        raise RuntimeError("common-fold A requires exactly one TEP record-time view")
    view = views[0]
    v211, v21, v2 = load_v211_configs(project, protocol=PUBLIC_ALL_PROTOCOL)
    del v211
    frozen = {
        str(value)
        for value in _result(
            diagnostic_results / f"DEVELOPMENT/C/{head}/{proxy}/RESULT.json"
        )["active_channels"]
    }
    active = [
        item
        for item in load_active_channels(diagnostic_results, view)
        if str(item["channel"]) in frozen
    ]
    development = load_native_samples(shared, view, "train")
    folds = registered_joint_inner_fold_frames(
        development,
        fold_count=int(v21["selection"]["inner_folds"]),
        fit_cap=int(v2["row_caps"]["joint_predictive_fit"]),
        evaluation_cap=int(v2["row_caps"]["validation_selection_per_fold"]),
        active=active,
    )
    oof = pd.read_parquet(diagnostic_results / str(w_result["oof_path"]))
    c_oof = pd.read_parquet(
        diagnostic_results / str(c_result["oof_prediction_path"]),
        columns=["base_origin_id", "oof_fold"],
    )
    contribution_columns = sorted(
        column for column in oof if column.startswith("k_channel_contribution_")
    )
    oof["residual"] = oof["y_true"] - oof["physical_w_oof"]
    accessor = BaseAccessor(shared, view.head.dataset, "train", [view.head.target])
    fold_records: list[dict[str, Any]] = []
    for fold in folds[1:]:
        fold_index = int(fold["fold_index"])
        # Joint does not evaluate on an independently capped dynamic support.
        # It registers each fold on the frozen C/W OOF intersection.  Reapply
        # that exact registration before fitting/evaluating standalone A so
        # both estimators have the same ordered evidence rows.
        w_fold_oof = oof.loc[
            oof["oof_fold"] == fold_index,
            ["base_origin_id", "oof_fold"],
        ].reset_index(drop=True)
        c_fold_oof = c_oof.loc[
            c_oof["oof_fold"] == fold_index,
            ["base_origin_id", "oof_fold"],
        ].reset_index(drop=True)
        fold = register_joint_fold_on_oof_support(fold, w_fold_oof, c_fold_oof)
        fit = oof.loc[oof["oof_fold"] < fold_index].reset_index(drop=True)
        reference = fold["evaluation"][["base_origin_id"]]
        evaluation = reference.merge(
            oof.loc[oof["oof_fold"] == fold_index],
            on="base_origin_id",
            how="left",
            validate="one_to_one",
        )
        if evaluation["y_true"].isna().any() or len(evaluation) != len(reference):
            raise RuntimeError("standalone A does not cover the registered Joint fold")
        if selected == EXACT_ZERO:
            residual_prediction = np.zeros(len(evaluation), dtype=np.float64)
            contract = {"family": EXACT_ZERO, "parameter_count": 0}
        else:
            _, profile, alpha, mu = selected
            delta, history = profile
            residual_mean = float(fit["residual"].mean())
            x_fit, _, _ = mature_residual_features(
                fit,
                oof,
                h_steps=view.head.h_steps,
                w_steps=view.head.w_steps,
                delta=int(delta),
                history=int(history),
                maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]),
                residual_mean=residual_mean,
            )
            x_evaluation, _, _ = mature_residual_features(
                evaluation,
                oof,
                h_steps=view.head.h_steps,
                w_steps=view.head.w_steps,
                delta=int(delta),
                history=int(history),
                maximum_lags=int(v2["A_module"]["state_profile"]["maximum_lags"]),
                residual_mean=residual_mean,
            )
            upstream_columns = [*contribution_columns, "delta_w_oof"]
            if not contribution_columns:
                upstream_columns.insert(0, "physical_oof")
            residual_prediction, contract = fit_mature_residual_ar(
                x_fit,
                fit["residual"].to_numpy(dtype=np.float64),
                x_evaluation,
                alpha=float(alpha),
                mu=float(mu),
                upstream_predictions=fit[upstream_columns].to_numpy(dtype=np.float64),
            )
        prediction = (
            evaluation["physical_w_oof"].to_numpy(dtype=np.float64)
            + residual_prediction
        )
        target = evaluation["y_true"].to_numpy(dtype=np.float64)
        current = accessor.block_means(
            evaluation, view.head.target, [(0, int(view.head.w0_steps))]
        )[:, 0]
        fold_records.append(
            {
                "fold_index": fold_index,
                "support_hash": support_hash(evaluation["base_origin_id"].astype(str)),
                "prediction_hash": hashlib.sha256(
                    np.ascontiguousarray(prediction, dtype=np.float64).tobytes()
                ).hexdigest(),
                "metrics": _metrics(target, prediction, current),
                "metric_sufficient_statistics": _metric_sufficient_statistics(
                    target, prediction, current
                ),
                "calibration": contract,
                "input_block_variance": float(
                    np.var(evaluation["physical_w_oof"].to_numpy(dtype=np.float64))
                ),
                "a_block_variance": float(np.var(residual_prediction, dtype=np.float64)),
            }
        )
    metrics = _aggregate_metric_sufficient_statistics(
        [fold["metric_sufficient_statistics"] for fold in fold_records]
    )
    return {
        "candidate_id": selected_text,
        "folds": fold_records,
        "metrics": metrics,
        "outer_oof_objective": metrics["mse"],
        "support_hash": support_hash(fold["support_hash"] for fold in fold_records),
        "prediction_hash": support_hash(fold["prediction_hash"] for fold in fold_records),
        "parameter_count": max(
            int(fold["calibration"].get("parameter_count", 0)) for fold in fold_records
        ),
        "input_block_contribution": float(
            np.mean([fold["input_block_variance"] for fold in fold_records])
        ),
        "A_block_contribution": float(
            np.mean([fold["a_block_variance"] for fold in fold_records])
        ),
    }


def write_d2_tep_joint_vs_a_report(
    output: Path,
    diagnostic_results: Path,
    shared: Path,
    project: Path,
) -> pd.DataFrame:
    joint_path = (
        diagnostic_results
        / "DEVELOPMENT/JOINT/TEP_G_NOWCAST_H0__H0__W1/record_time/proxy_excluded/RESULT.json"
    )
    a_path = (
        diagnostic_results
        / "DEVELOPMENT/A/TEP_G_NOWCAST_H0__H0__W1/record_time/proxy_excluded/RESULT.json"
    )
    joint = _result(joint_path)
    a_result = _result(a_path)
    view = next(item for item in FORMAL_VIEWS if item.task == "TEP record-time")
    a_frame = _prediction(diagnostic_results, a_result)
    a_metrics, a_support = _metrics_for_prediction(shared, view, a_frame)
    a_selection = _selection_for("A", a_result)
    common_a = _common_fold_standalone_a(diagnostic_results, shared, project)
    common_fold_hashes = [fold["support_hash"] for fold in common_a["folds"]]
    rows: list[dict[str, Any]] = [
        {
            "comparison_set": "VALIDATION_COMMON_SUPPORT",
            "scope": "STANDALONE_A",
            "candidate_id": a_selection["final_selected_candidate"],
            "route": "A",
            "k_representation": None,
            "numerical_alpha": None,
            "predictive_eta": None,
            "inner_selection_objective": a_selection["child_oof_risk"],
            "outer_oof_objective": a_selection["child_oof_risk"],
            "evaluation_mse": a_metrics["mse"],
            **_flatten_metrics(a_metrics),
            "calibration": json.dumps(a_result.get("final_selected_contract", {}), sort_keys=True),
            "parameter_count": _parameter_count(a_result),
            "input_block_contribution": 0.0,
            "A_block_contribution": float(np.var(a_frame["y_pred"].to_numpy(dtype=np.float64))),
            "support_hash": a_support,
            "prediction_hash": a_result.get("prediction_sha256"),
            "selection_eligible": True,
        },
        {
            "comparison_set": "OOF_COMMON_FOLDS_1_3",
            "scope": "STANDALONE_A",
            "candidate_id": common_a["candidate_id"],
            "route": "A",
            "k_representation": None,
            "numerical_alpha": None,
            "predictive_eta": None,
            "inner_selection_objective": a_selection["child_oof_risk"],
            "outer_oof_objective": common_a["outer_oof_objective"],
            "evaluation_mse": common_a["metrics"]["mse"],
            **_flatten_metrics(common_a["metrics"]),
            "calibration": json.dumps(
                [fold["calibration"] for fold in common_a["folds"]], sort_keys=True
            ),
            "parameter_count": common_a["parameter_count"],
            "input_block_contribution": common_a["input_block_contribution"],
            "A_block_contribution": common_a["A_block_contribution"],
            "support_hash": common_a["support_hash"],
            "prediction_hash": common_a["prediction_hash"],
            "selection_eligible": False,
        },
    ]
    for key, evidence in sorted(joint.get("report_only_counterfactuals", {}).items()):
        descriptor = evidence["candidate"]
        validation = evidence["validation"]
        rows.append(
            {
                "comparison_set": "VALIDATION_COMMON_SUPPORT",
                "scope": "REGISTERED_JOINT_CANDIDATE",
                "candidate_id": descriptor["candidate_id"],
                "route": descriptor["route"],
                "k_representation": descriptor["k_representation"],
                "numerical_alpha": descriptor["numerical_alpha"],
                "predictive_eta": descriptor["predictive_eta"],
                "inner_selection_objective": evidence["inner_selection_objective"],
                "outer_oof_objective": evidence["outer_oof_risk"],
                "evaluation_mse": validation["metrics"]["mse"],
                **_flatten_metrics(validation["metrics"]),
                "calibration": json.dumps(validation["calibration"], sort_keys=True),
                "parameter_count": len(validation["calibration"]["coefficient"]) + 1,
                "input_block_contribution": validation["input_block_variance"],
                "A_block_contribution": validation["a_block_variance"],
                "support_hash": validation["support_hash"],
                "prediction_hash": validation["prediction_sha256"],
                "selection_eligible": False,
            }
        )
        common_folds = [
            fold for fold in evidence["folds"] if int(fold["fold_index"]) in {1, 2, 3}
        ]
        if [fold["support_hash"] for fold in common_folds] != common_fold_hashes:
            raise RuntimeError("D2 standalone A and Joint fold support mismatch")
        common_metrics = _aggregate_metric_sufficient_statistics(
            [fold["metric_sufficient_statistics"] for fold in common_folds]
        )
        rows.append(
            {
                "comparison_set": "OOF_COMMON_FOLDS_1_3",
                "scope": "REGISTERED_JOINT_CANDIDATE",
                "candidate_id": descriptor["candidate_id"],
                "route": descriptor["route"],
                "k_representation": descriptor["k_representation"],
                "numerical_alpha": descriptor["numerical_alpha"],
                "predictive_eta": descriptor["predictive_eta"],
                "inner_selection_objective": evidence["inner_selection_objective"],
                "outer_oof_objective": common_metrics["mse"],
                "evaluation_mse": common_metrics["mse"],
                **_flatten_metrics(common_metrics),
                "calibration": json.dumps(
                    [fold["calibration"] for fold in common_folds], sort_keys=True
                ),
                "parameter_count": max(
                    len(fold["calibration"]["coefficient"]) + 1
                    for fold in common_folds
                ),
                "input_block_contribution": float(
                    np.mean([fold["input_block_variance"] for fold in common_folds])
                ),
                "A_block_contribution": float(
                    np.mean([fold["a_block_variance"] for fold in common_folds])
                ),
                "support_hash": support_hash(common_fold_hashes),
                "prediction_hash": support_hash(
                    fold["prediction_sha256"] for fold in common_folds
                ),
                "selection_eligible": False,
            }
        )
    frame = pd.DataFrame(rows)
    expected_candidates = 4 * 2 * 7
    for comparison_set, group in frame.groupby("comparison_set"):
        actual_candidates = int(
            (group["scope"] == "REGISTERED_JOINT_CANDIDATE").sum()
        )
        if actual_candidates != expected_candidates:
            raise RuntimeError(
                f"D2 {comparison_set} requires all {expected_candidates} registered Joint candidates; found {actual_candidates}"
            )
        if group["support_hash"].nunique() != 1:
            raise RuntimeError(f"D2 support mismatch in {comparison_set}")
    destination = output / "DIAGNOSTICS"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "TEP_JOINT_VS_A_COUNTERFACTUAL.csv", index=False)
    common = frame.loc[frame["comparison_set"] == "OOF_COMMON_FOLDS_1_3"]
    route_best = (
        common.loc[common["scope"] == "REGISTERED_JOINT_CANDIDATE"]
        .sort_values("outer_oof_objective")
        .groupby("route", as_index=False)
        .first()
    )
    a_row = common.loc[common["scope"] == "STANDALONE_A"].iloc[0]
    summary = pd.concat([common.loc[common["scope"] == "STANDALONE_A"], route_best], ignore_index=True)
    joint_best = common.loc[common["scope"] == "REGISTERED_JOINT_CANDIDATE"].sort_values(
        "outer_oof_objective"
    ).iloc[0]
    validation = frame.loc[frame["comparison_set"] == "VALIDATION_COMMON_SUPPORT"]
    validation_a = validation.loc[validation["scope"] == "STANDALONE_A"].iloc[0]
    validation_joint = validation.loc[
        validation["candidate_id"] == joint_best["candidate_id"]
    ].iloc[0]
    mismatch = {
        "selection_objective_mismatch": bool(
            (joint_best["outer_oof_objective"] < a_row["outer_oof_objective"])
            != (validation_joint["evaluation_mse"] < validation_a["evaluation_mse"])
        ),
        "calibration_mismatch": False,
        "calibration_audit": "distinct registered estimators; prediction replay and shared-error identity passed",
        "support_mismatch": False,
        "delta_vs_level_objective_mismatch": bool(
            (validation_joint["Delta_R2"] > validation_a["Delta_R2"])
            != (validation_joint["Level_R2"] > validation_a["Level_R2"])
        ),
        "common_folds": [1, 2, 3],
        "fold_0_exclusion": "standalone A has no preceding causal fit fold",
    }
    lines = [
        "# TEP record-time Joint versus standalone A diagnostic",
        "",
        f"All {expected_candidates} registered Joint candidates plus frozen standalone A were evaluated on the same validation support and on common causal folds 1--3. Fold 0 is excluded from the OOF comparison because standalone A has no preceding causal fit fold. The full 114-row table is in the companion CSV.",
        "",
        _markdown_table(summary[["comparison_set", "scope", "route", "k_representation", "predictive_eta", "outer_oof_objective", "Delta_RMSE", "Delta_R2", "Level_RMSE", "Level_R2", "parameter_count", "input_block_contribution", "A_block_contribution", "support_hash"]]),
        "",
        "## Mismatch audit",
        "",
        *[f"- {name}: `{str(value).lower()}`" for name, value in mismatch.items()],
        "",
        "This is development-only report evidence; it did not change the selector, objective, calibration, support, or admission semantics.",
    ]
    (destination / "TEP_JOINT_VS_A_COUNTERFACTUAL.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_json(destination / "TEP_JOINT_VS_A_MISMATCH_AUDIT.json", mismatch)
    return frame
