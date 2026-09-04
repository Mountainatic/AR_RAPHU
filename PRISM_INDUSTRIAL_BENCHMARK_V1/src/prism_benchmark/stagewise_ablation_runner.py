"""Retrospective hybrid-h/w stagewise PRISM execution helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .cpu_data import ViewSpec, input_columns, sha256_file
from .portable_checkpoints import INFERENCE_ONLY_ENV, activate_inference_fit_guard
from .representative_prism_checkpoints import (
    fit_prism_checkpoint_for_view,
    predict_prism_checkpoint_for_view,
    verify_prism_checkpoint_reload,
)
from .stage0 import write_json
from .v2_runtime import run_parallel
from .v2_views import development_dynamic_views, development_input_views
from .v211_a import run_a_view
from .v211_c import run_c_view
from .v211_config import PUBLIC_ALL_PROTOCOL
from .v211_joint_stability import run_joint_stability_view
from .v211_k import run_k_channel
from .v211_public_all_baselines import apply_common_requirements
from .v211_public_all_closure import (
    METADATA_COLUMNS,
    _support_frame,
    view_support_requirements,
)
from .v211_public_all_config import PublicAllPaths
from .v211_support import SUPPORT_COLUMNS, SUPPORT_CONTRACT, support_id_hash
from .v211_w import run_w_view


CONFIG_RELATIVE_PATH = Path("configs/stagewise_ablation_hybrid_hw_20260904.json")
PROTOCOL_ID = "PRISM_V211_STAGEWISE_ABLATION_HYBRID_HW_20260904_R3"
ACCEPTABLE_JOINT_STATUSES = {
    "PASS",
    "NOT_RUN_PROTOCOL_INCOMPATIBLE",
    "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT",
    "SOLVER_FAILED_RETAINED",
}
BOOTSTRAP_BLOCK_REGISTRY = "STAGEWISE_BOOTSTRAP_BLOCK_LENGTHS.json"
BOOTSTRAP_ACF_MAXIMUM_LAG = 256


def load_stagewise_protocol(project: Path) -> dict[str, Any]:
    path = project / CONFIG_RELATIVE_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("stagewise ablation protocol id mismatch")
    if value.get("status") != "FROZEN_AFTER_IMPLEMENTATION_PREFLIGHT_BEFORE_EXECUTION":
        raise RuntimeError("stagewise ablation protocol is not frozen")
    return value


def _task_record(protocol: dict[str, Any], head_id: str) -> dict[str, Any]:
    matches = [item for item in protocol["task_matrix"] if item["head_id"] == head_id]
    if len(matches) != 1:
        raise RuntimeError(f"head is not uniquely registered: {head_id}")
    return matches[0]


def selected_views(shared: Path, protocol: dict[str, Any], head_id: str) -> tuple[list[ViewSpec], list[ViewSpec]]:
    registered = _task_record(protocol, head_id)
    input_views = [
        view for view in development_input_views(shared) if view.head.head_id == head_id
    ]
    dynamic_views = [
        view for view in development_dynamic_views(shared) if view.head.head_id == head_id
    ]
    views = [*input_views, *dynamic_views]
    if not views:
        raise RuntimeError(f"registered head is absent from shared C1: {head_id}")
    for view in views:
        if int(view.head.h_steps) != int(registered["h_steps"]):
            raise RuntimeError(f"h drift for {head_id}")
        if int(view.head.w_steps) != int(registered["w_steps"]):
            raise RuntimeError(f"w drift for {head_id}")
    return input_views, dynamic_views


def _result_path(output: Path, stage: str, view: ViewSpec, channel: str | None = None) -> Path:
    root = output / "DEVELOPMENT" / stage / view.head.head_id
    if stage == "K":
        if channel is None:
            raise ValueError("K result path requires a channel")
        return root / view.proxy_policy / channel / "RESULT.json"
    if stage in {"C", "W"}:
        return root / view.proxy_policy / "RESULT.json"
    return root / view.availability_scenario / view.proxy_policy / "RESULT.json"


def _completed(path: Path, stage: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    status = str(value.get("status"))
    allowed = {"PASS"} if stage != "JOINT" else ACCEPTABLE_JOINT_STATUSES
    if status not in allowed or value.get("test_accessed") is not False:
        return None
    return value


def _run_jobs(
    output: Path,
    stage: str,
    function: Callable[..., Any],
    jobs: list[tuple[Path, tuple[Any, ...]]],
    workers: int,
    per_worker_gib: float,
) -> dict[str, Any]:
    reused: list[dict[str, Any]] = []
    pending: list[tuple[Any, ...]] = []
    for path, arguments in jobs:
        previous = _completed(path, stage)
        if previous is None:
            pending.append(arguments)
        else:
            reused.append(previous)
    produced = (
        [
            dict(value)
            for value in run_parallel(
                function,
                pending,
                max(1, min(int(workers), len(pending))),
                per_worker_gib=float(per_worker_gib),
                label=f"STAGEWISE_HYBRID_{stage}",
            )
        ]
        if pending
        else []
    )
    records = [*reused, *produced]
    retained = [item for item in records if str(item.get("status")) != "PASS"]
    result = {
        "status": "PASS" if not retained else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": stage,
        "jobs": len(records),
        "new_jobs": len(produced),
        "reused_jobs": len(reused),
        "pass": sum(str(item.get("status")) == "PASS" for item in records),
        "retained_statuses": [str(item.get("status")) for item in retained],
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / stage / "SUMMARY.json", result)
    return result


def _build_common_support_metadata_only(
    paths: PublicAllPaths, views: Iterable[ViewSpec]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for view in views:
        requirements = view_support_requirements(paths, view)
        splits: dict[str, Any] = {}
        for split in ("train", "validation", "test", "ood"):
            source = paths.shared / "sample_ids" / view.relative_root / f"{split}.parquet"
            if not source.is_file():
                continue
            frame = _support_frame(paths.shared, view, split)
            common = apply_common_requirements(frame, requirements)
            splits[split] = {
                "rows": int(len(common)),
                "source_rows": int(len(frame)),
                "support_hash": support_id_hash(common),
                "support_contract": SUPPORT_CONTRACT,
            }
        records.append(
            {
                "target_head": view.head.head_id,
                "dataset": view.head.dataset,
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "requirements": [item.to_json() for item in requirements],
                "splits": splits,
            }
        )
    result = {
        "status": "PASS",
        "stage": "STAGEWISE_COMMON_SUPPORT_METADATA_ONLY_FREEZE",
        "support_contract": SUPPORT_CONTRACT,
        "views": records,
        "test_columns_read": list(dict.fromkeys([*METADATA_COLUMNS, *SUPPORT_COLUMNS])),
        "test_y_read": False,
        "ood_y_read": False,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.leaderboard_support_path, result)
    return result


def run_development(
    paths: PublicAllPaths,
    head_id: str,
    *,
    workers: int,
    per_worker_gib: float,
) -> dict[str, Any]:
    if os.environ.get(INFERENCE_ONLY_ENV) == "1":
        raise RuntimeError("development is forbidden in an inference-only process")
    protocol = load_stagewise_protocol(paths.project)
    input_views, dynamic_views = selected_views(paths.shared, protocol, head_id)
    paths.output.mkdir(parents=True, exist_ok=True)
    k_jobs = [
        (
            _result_path(paths.output, "K", view, channel),
            (
                paths.shared,
                paths.project,
                paths.output,
                view,
                channel,
                PUBLIC_ALL_PROTOCOL,
            ),
        )
        for view in input_views
        for channel in input_columns(paths.shared, view.head.task_id, view.proxy_policy)
    ]
    stages: list[tuple[str, Callable[..., Any], list[tuple[Path, tuple[Any, ...]]]]] = [
        ("K", run_k_channel, k_jobs),
        (
            "C",
            run_c_view,
            [
                (
                    _result_path(paths.output, "C", view),
                    (paths.shared, paths.project, paths.output, view, PUBLIC_ALL_PROTOCOL),
                )
                for view in input_views
            ],
        ),
        (
            "W",
            run_w_view,
            [
                (
                    _result_path(paths.output, "W", view),
                    (paths.shared, paths.project, paths.output, view, PUBLIC_ALL_PROTOCOL),
                )
                for view in input_views
            ],
        ),
        (
            "A",
            run_a_view,
            [
                (
                    _result_path(paths.output, "A", view),
                    (paths.shared, paths.project, paths.output, view, PUBLIC_ALL_PROTOCOL),
                )
                for view in dynamic_views
            ],
        ),
        (
            "JOINT",
            run_joint_stability_view,
            [
                (
                    _result_path(paths.output, "JOINT", view),
                    (
                        paths.shared,
                        paths.project,
                        paths.output,
                        None,
                        view,
                        PUBLIC_ALL_PROTOCOL,
                    ),
                )
                for view in dynamic_views
            ],
        ),
    ]
    summaries: dict[str, Any] = {}
    for stage, function, jobs in stages:
        summaries[stage] = _run_jobs(
            paths.output, stage, function, jobs, workers, per_worker_gib
        )
    support = _build_common_support_metadata_only(
        paths, [*input_views, *dynamic_views]
    )
    result = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "head_id": head_id,
        "input_views": len(input_views),
        "dynamic_views": len(dynamic_views),
        "stage_summaries": summaries,
        "common_support_views": len(support["views"]),
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.logs / "STAGEWISE_DEVELOPMENT_COMPLETE.json", result)
    return result


def matching_views(
    paths: PublicAllPaths,
    head_id: str,
    *,
    information_set: str,
    availability_scenario: str,
    proxy_policy: str,
) -> list[ViewSpec]:
    protocol = load_stagewise_protocol(paths.project)
    input_views, dynamic_views = selected_views(paths.shared, protocol, head_id)
    source = input_views if information_set == "input_only" else dynamic_views
    result = [
        view
        for view in source
        if view.availability_scenario == availability_scenario
        and view.proxy_policy == proxy_policy
    ]
    if len(result) != 1:
        raise RuntimeError(
            "stagewise view is not unique: "
            f"{head_id}/{information_set}/{availability_scenario}/{proxy_policy}"
        )
    return result


def development_residual_block_length(
    frame: pd.DataFrame, *, maximum_lag: int = BOOTSTRAP_ACF_MAXIMUM_LAG
) -> dict[str, Any]:
    """Choose the first immaterial residual-ACF lag without crossing entities."""

    required = {"entity_id", "origin", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"development residual frame is missing: {sorted(missing)}")
    ordered = frame.sort_values(["entity_id", "origin"], kind="stable")
    residuals = [
        part["y_true"].to_numpy(dtype=np.float64)
        - part["y_pred"].to_numpy(dtype=np.float64)
        for _, part in ordered.groupby("entity_id", sort=False)
    ]
    rows = sum(len(values) for values in residuals)
    if rows < 2:
        return {
            "block_length": 1,
            "rows": rows,
            "entities": len(residuals),
            "maximum_lag_evaluated": 0,
            "acf_threshold": float("inf"),
            "selection_rule": "SINGLETON_FALLBACK",
        }
    centered = [values - np.mean(values, dtype=np.float64) for values in residuals]
    denominator = float(
        sum(np.sum(np.square(values), dtype=np.float64) for values in centered)
    )
    threshold = float(1.96 / np.sqrt(rows))
    longest = max(len(values) for values in centered)
    limit = max(1, min(int(maximum_lag), longest - 1))
    acf: list[float] = []
    selected = limit
    rule = "MAXIMUM_LAG_FALLBACK"
    for lag in range(1, limit + 1):
        numerator = float(
            sum(
                np.sum(values[:-lag] * values[lag:], dtype=np.float64)
                for values in centered
                if len(values) > lag
            )
        )
        value = 0.0 if denominator == 0.0 else numerator / denominator
        acf.append(value)
        if abs(value) <= threshold:
            selected = lag
            rule = "FIRST_ABSOLUTE_ACF_WITHIN_95_PERCENT_WHITE_NOISE_BAND"
            break
    return {
        "block_length": int(max(1, selected)),
        "rows": int(rows),
        "entities": int(len(residuals)),
        "maximum_lag_evaluated": int(limit),
        "acf_threshold": threshold,
        "acf_until_selection": acf,
        "selection_rule": rule,
        "entity_boundaries_crossed": False,
    }


def freeze_bootstrap_block_lengths(
    paths: PublicAllPaths, views: Iterable[ViewSpec]
) -> dict[str, Any]:
    """Freeze test-bootstrap block lengths from development residuals only."""

    if os.environ.get(INFERENCE_ONLY_ENV) == "1":
        raise RuntimeError("bootstrap block selection is forbidden after inference starts")
    records: list[dict[str, Any]] = []
    for view in views:
        source = (
            paths.output
            / "DEVELOPMENT"
            / "A"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "validation.parquet"
        )
        if not source.is_file():
            raise RuntimeError(f"development PF prediction is missing: {source}")
        frame = pd.read_parquet(
            source, columns=["entity_id", "origin", "y_true", "y_pred"]
        )
        records.append(
            {
                "target_head": view.head.head_id,
                "dataset": view.head.dataset,
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "development_prediction_path": str(source.relative_to(paths.run_root)),
                "development_prediction_sha256": sha256_file(source),
                **development_residual_block_length(frame),
            }
        )
    result = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "method": "PAIRED_MOVING_BLOCK_BOOTSTRAP",
        "block_length_source": "DEVELOPMENT_PF_RESIDUAL_ACF_ONLY",
        "maximum_lag": BOOTSTRAP_ACF_MAXIMUM_LAG,
        "records": records,
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(paths.freeze / BOOTSTRAP_BLOCK_REGISTRY, result)
    return result


def fit_checkpoints(
    paths: PublicAllPaths,
    checkpoint_root: Path,
    views: Iterable[ViewSpec],
) -> dict[str, Any]:
    records = [fit_prism_checkpoint_for_view(paths, view, checkpoint_root) for view in views]
    result = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "records": records,
        "protocol_sha256": sha256_file(paths.project / CONFIG_RELATIVE_PATH),
        "test_accessed": False,
        "ood_accessed": False,
    }
    write_json(checkpoint_root / "STAGEWISE_CHECKPOINT_MANIFEST.json", result)
    return result


def infer_checkpoints(
    paths: PublicAllPaths,
    checkpoint_root: Path,
    destination_run_root: Path,
    views: Iterable[ViewSpec],
) -> dict[str, Any]:
    if os.environ.get(INFERENCE_ONLY_ENV) != "1":
        raise RuntimeError("inference requires PRISM_FORMAL_INFERENCE_ONLY=1")
    block_registry = paths.freeze / BOOTSTRAP_BLOCK_REGISTRY
    if not block_registry.is_file():
        raise RuntimeError("development-only bootstrap block-length freeze is missing")
    activate_inference_fit_guard()
    records: list[dict[str, Any]] = []
    reload_audits: list[dict[str, Any]] = []
    for view in views:
        checkpoint = checkpoint_root / "prism" / (
            "__".join(
                str(part).replace("/", "_").replace("\\", "_")
                for part in (
                    view.head.dataset,
                    view.head.head_id,
                    view.information_set,
                    view.availability_scenario,
                    view.proxy_policy,
                )
            )
        )
        reload_audits.append(verify_prism_checkpoint_reload(checkpoint))
        records.extend(
            predict_prism_checkpoint_for_view(
                paths,
                view,
                checkpoint_root,
                split="test",
                destination_run_root=destination_run_root,
            )
        )
    result = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "records": records,
        "reload_audits": reload_audits,
        "bootstrap_block_registry": str(block_registry),
        "bootstrap_block_registry_sha256": sha256_file(block_registry),
        "fit_called_in_inference": False,
        "test_accessed": True,
        "ood_accessed": False,
    }
    write_json(destination_run_root / "STAGEWISE_INFERENCE_COMPLETE.json", result)
    return result
