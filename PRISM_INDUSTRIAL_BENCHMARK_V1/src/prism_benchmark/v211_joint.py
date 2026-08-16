from __future__ import annotations

import json
import hashlib
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    inner_folds,
    realized_state_profiles,
    sha256_file,
)
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_c import fit_physical_features
from .v2_k import _cap
from .v2_numerics import solve_certified
from .v2_runtime import ordered_fork_map, run_parallel
from .v2_selection import one_se_select
from .v21_selection import guarded_local_one_se_select
from .v211_a import fit_mature_residual_ar
from .v211_c import _gate_config, _smallest_stable_alpha
from .v211_config import load_v211_configs
from .v211_k import load_active_channels
from .v211_selection import (
    attach_nonselecting_validation_confirmation,
    input_path_preservation_gate,
    numerical_contract_passes,
)
from .v211_support import (
    apply_assembly_support,
    base_origin_support_hash,
    fold_evaluation_causal_floor,
    load_native_samples,
    support_id_hash,
)
from .v211_w import IDENTITY, _fit_c_routed, build_w_design


J_K = "J_K"
J_KW = "J_KW"
J_KA = "J_KA"
J_KWA = "J_KWA"
JOINT_CANDIDATES = (J_K, J_KW, J_KA, J_KWA)
J_INNER_WORKERS_ENV = "PRISM_V211_J_INNER_WORKERS"
_J_CANDIDATE_CONTEXT: tuple[
    Mapping[str, np.ndarray],
    np.ndarray,
    Mapping[str, np.ndarray],
    np.ndarray,
    list[tuple[str, float, float, float]],
] | None = None


class JointFoldProtocolMismatch(RuntimeError):
    pass


def _ordered_id_hash(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _joint_fold_id_hash(frame: pd.DataFrame) -> dict[str, str]:
    """Hash both registered row identifiers in their deterministic row order."""
    missing = [
        column
        for column in ("base_origin_id", "view_sample_id")
        if column not in frame.columns
    ]
    if missing:
        raise JointFoldProtocolMismatch(
            f"Joint fold provenance frame is missing identifiers: {missing}"
        )
    digest = hashlib.sha256()
    for base_origin_id, view_sample_id in zip(
        frame["base_origin_id"].astype(str),
        frame["view_sample_id"].astype(str),
        strict=True,
    ):
        digest.update(base_origin_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(view_sample_id.encode("utf-8"))
        digest.update(b"\n")
    return {
        "base_origin_id_sha256": _ordered_id_hash(frame["base_origin_id"]),
        "view_sample_id_sha256": _ordered_id_hash(frame["view_sample_id"]),
        "base_origin_and_view_sample_id_sha256": digest.hexdigest(),
    }


def _same_ordered_ids(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: tuple[str, ...],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        left[column].astype(str).reset_index(drop=True).equals(
            right[column].astype(str).reset_index(drop=True)
        )
        for column in columns
    )


def registered_joint_inner_fold_frames(
    development_train: pd.DataFrame,
    *,
    fold_count: int,
    fit_cap: int,
    evaluation_cap: int,
    active: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Materialize registered original ``T_i -> V_i`` Joint fold supports."""
    result = []
    for fold_index, (fit_index, evaluation_index) in enumerate(
        inner_folds(development_train, int(fold_count))
    ):
        fit_raw = development_train.iloc[fit_index].reset_index(drop=True)
        evaluation_raw = development_train.iloc[evaluation_index].reset_index(
            drop=True
        )
        fit_supported = (
            apply_assembly_support(fit_raw, active) if active is not None else fit_raw
        )
        evaluation_supported = (
            apply_assembly_support(
                evaluation_raw,
                active,
                fold_evaluation_causal_floor(fit_raw, evaluation_raw),
            )
            if active is not None
            else evaluation_raw
        )
        result.append(
            {
                "fold_index": fold_index,
                "fit_raw": fit_raw,
                "evaluation_raw": evaluation_raw,
                "fit_supported": fit_supported.reset_index(drop=True),
                "evaluation_supported": evaluation_supported.reset_index(drop=True),
                "fit": _cap(fit_supported, int(fit_cap)).reset_index(drop=True),
                "evaluation": _cap(
                    evaluation_supported, int(evaluation_cap)
                ).reset_index(drop=True),
            }
        )
    return result


def _input_only_view(view: Any) -> ViewSpec:
    """Return the frozen input-only record-time namespace for a view."""
    return ViewSpec(
        head=view.head,
        information_set="input_only",
        availability_scenario="record_time",
        proxy_policy=view.proxy_policy,
    )


def _align_frame_by_base_origin_id(
    source: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Project a frozen support frame into the reference row namespace."""
    for frame_name, frame in (("source", source), ("reference", reference)):
        if "base_origin_id" not in frame.columns:
            raise JointFoldProtocolMismatch(
                f"{label} {frame_name} is missing base_origin_id"
            )
    source_ids = source["base_origin_id"].astype(str).reset_index(drop=True)
    reference_ids = reference["base_origin_id"].astype(str).reset_index(drop=True)
    if source_ids.duplicated().any() or reference_ids.duplicated().any():
        raise JointFoldProtocolMismatch(
            f"{label} requires unique base_origin_id rows"
        )
    positions = pd.Index(source_ids).get_indexer(reference_ids)
    if np.any(positions < 0):
        missing = reference_ids[positions < 0].tolist()[:3]
        raise JointFoldProtocolMismatch(
            f"{label} is missing registered base_origin_id rows: {missing}"
        )
    return source.iloc[positions].reset_index(drop=True)


def align_registered_joint_fold(
    joint_fold: Mapping[str, Any],
    registered_input_fold: Mapping[str, Any],
) -> dict[str, Any]:
    """Align input-only fold provenance to an availability-specific Joint fold."""
    aligned = dict(registered_input_fold)
    for name in (
        "fit_raw",
        "evaluation_raw",
        "fit_supported",
        "evaluation_supported",
        "fit",
        "evaluation",
    ):
        if name in joint_fold and name in registered_input_fold:
            aligned[name] = _align_frame_by_base_origin_id(
                registered_input_fold[name],
                joint_fold[name],
                label=f"registered input {name}",
            )
    return aligned


def align_joint_oof_rows(
    oof_rows: pd.DataFrame,
    joint_evaluation: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Align frozen C/W OOF rows to the Joint evaluation support."""
    return _align_frame_by_base_origin_id(
        oof_rows,
        joint_evaluation,
        label=label,
    )


def intersect_by_base_origin_id(
    samples: pd.DataFrame,
    registered_support: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Apply availability-specific support as an intersection with frozen C/W."""
    if "base_origin_id" not in samples or "base_origin_id" not in registered_support:
        raise JointFoldProtocolMismatch(
            f"{label} support intersection is missing base_origin_id"
        )
    registered_ids = set(registered_support["base_origin_id"].astype(str))
    result = samples.loc[
        samples["base_origin_id"].astype(str).isin(registered_ids)
    ].copy()
    if result.empty:
        raise JointFoldProtocolMismatch(
            f"{label} has no rows on the frozen C/W support intersection"
        )
    return result.reset_index(drop=True)


def _origin_bounds(frame: pd.DataFrame) -> tuple[int | None, int | None]:
    if frame.empty:
        return None, None
    values = frame["origin"].to_numpy(dtype=np.int64)
    return int(values.min()), int(values.max())


def audit_joint_fold_protocol(
    fold: Mapping[str, Any],
    registered_input_fold: Mapping[str, Any],
    w_evaluation: pd.DataFrame,
    c_evaluation: pd.DataFrame,
) -> dict[str, Any]:
    """Prove Joint uses raw registered support and shares C/W evaluation rows."""
    fit = fold["fit"]
    evaluation = fold["evaluation"]
    input_fit = registered_input_fold["fit"]
    input_evaluation = registered_input_fold["evaluation"]
    fit_hash = _joint_fold_id_hash(fit)
    evaluation_hash = _joint_fold_id_hash(evaluation)
    input_fit_hash = _joint_fold_id_hash(input_fit)
    input_evaluation_hash = _joint_fold_id_hash(input_evaluation)
    w_hash = _joint_fold_id_hash(w_evaluation)
    c_hash = _joint_fold_id_hash(c_evaluation)
    checks = {
        "fold_index_equal": int(fold["fold_index"])
        == int(registered_input_fold["fold_index"]),
        "joint_and_registered_input_fit_base_origin_ids_equal": _same_ordered_ids(
            fit, input_fit, ("base_origin_id",)
        ),
        "joint_and_registered_input_evaluation_base_origin_ids_equal": _same_ordered_ids(
            evaluation, input_evaluation, ("base_origin_id",)
        ),
        "registered_input_and_w_evaluation_ids_equal": _same_ordered_ids(
            input_evaluation,
            w_evaluation,
            ("base_origin_id", "view_sample_id"),
        ),
        "registered_input_and_c_evaluation_ids_equal": _same_ordered_ids(
            input_evaluation,
            c_evaluation,
            ("base_origin_id", "view_sample_id"),
        ),
        "c_and_w_evaluation_ids_equal": _same_ordered_ids(
            c_evaluation,
            w_evaluation,
            ("base_origin_id", "view_sample_id"),
        ),
    }
    minimum_fit_origin, maximum_fit_origin = _origin_bounds(fit)
    minimum_eval_origin, maximum_eval_origin = _origin_bounds(evaluation)
    return {
        "fold_index": int(fold["fold_index"]),
        "fit_rows_before_cap": len(fold["fit_raw"]),
        "fit_rows_after_assembly_support_before_cap": len(
            fold.get("fit_supported", fold["fit_raw"])
        ),
        "fit_rows_after_cap": len(fit),
        "evaluation_rows_before_cap": len(fold["evaluation_raw"]),
        "evaluation_rows_after_assembly_support_before_cap": len(
            fold.get("evaluation_supported", fold["evaluation_raw"])
        ),
        "evaluation_rows_after_cap": len(evaluation),
        "fit_base_origin_id_sha256": fit_hash["base_origin_id_sha256"],
        "fit_view_sample_id_sha256": fit_hash["view_sample_id_sha256"],
        "fit_joint_id_sha256": fit_hash[
            "base_origin_and_view_sample_id_sha256"
        ],
        "evaluation_base_origin_id_sha256": evaluation_hash[
            "base_origin_id_sha256"
        ],
        "evaluation_view_sample_id_sha256": evaluation_hash[
            "view_sample_id_sha256"
        ],
        "evaluation_joint_id_sha256": evaluation_hash[
            "base_origin_and_view_sample_id_sha256"
        ],
        "registered_input_fit_base_origin_id_sha256": input_fit_hash[
            "base_origin_id_sha256"
        ],
        "registered_input_fit_view_sample_id_sha256": input_fit_hash[
            "view_sample_id_sha256"
        ],
        "registered_input_evaluation_base_origin_id_sha256": input_evaluation_hash[
            "base_origin_id_sha256"
        ],
        "registered_input_evaluation_view_sample_id_sha256": input_evaluation_hash[
            "view_sample_id_sha256"
        ],
        "w_evaluation_base_origin_id_sha256": w_hash[
            "base_origin_id_sha256"
        ],
        "w_evaluation_view_sample_id_sha256": w_hash[
            "view_sample_id_sha256"
        ],
        "c_evaluation_base_origin_id_sha256": c_hash[
            "base_origin_id_sha256"
        ],
        "c_evaluation_view_sample_id_sha256": c_hash[
            "view_sample_id_sha256"
        ],
        "minimum_fit_origin": minimum_fit_origin,
        "maximum_fit_origin": maximum_fit_origin,
        "minimum_eval_origin": minimum_eval_origin,
        "maximum_eval_origin": maximum_eval_origin,
        "joint_native_view_sample_id_namespace": "dynamic",
        "c_w_comparison_view_sample_id_namespace": "input_only",
        "checks": checks,
        "pass": all(checks.values()),
    }


def _j_inner_workers() -> int:
    raw = os.environ.get(J_INNER_WORKERS_ENV, "1")
    try:
        workers = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{J_INNER_WORKERS_ENV} must be an integer") from error
    cpu_count = os.cpu_count() or 1
    if workers < 1 or workers > cpu_count:
        raise RuntimeError(
            f"{J_INNER_WORKERS_ENV} must be within [1, {cpu_count}]"
        )
    return workers


def registered_joint_candidates() -> tuple[str, ...]:
    return JOINT_CANDIDATES


def _required_blocks(candidate: str) -> tuple[str, ...]:
    if candidate == J_K:
        return ("K",)
    if candidate == J_KW:
        return ("K", "W")
    if candidate == J_KA:
        return ("K", "A")
    if candidate == J_KWA:
        return ("K", "W", "A")
    raise ValueError(f"unregistered Joint candidate: {candidate}")


def _as_matrix(value: np.ndarray | None, rows: int) -> np.ndarray:
    if value is None:
        return np.empty((rows, 0), dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or len(matrix) != rows:
        raise ValueError("Joint block has invalid shape")
    return matrix


def joint_w_basis(
    train_seed_latent: np.ndarray,
    evaluation_seed_latent: np.ndarray,
    w_contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    family = str(w_contract["family"])
    if family == IDENTITY:
        return (
            np.empty((len(train_seed_latent), 0), dtype=np.float64),
            np.empty((len(evaluation_seed_latent), 0), dtype=np.float64),
            {
                "family": IDENTITY,
                "columns": 0,
                "hard_feature_residualization": False,
            },
        )
    design = build_w_design(
        train_seed_latent,
        evaluation_seed_latent,
        family,
        int(w_contract["knot_count"]),
    )
    return design.train, design.evaluation, design.metadata


def fit_joint_candidate(
    train_blocks: Mapping[str, np.ndarray | None],
    target: np.ndarray,
    evaluation_blocks: Mapping[str, np.ndarray | None],
    *,
    candidate: str,
    alpha: float,
    k_over_a_ratio: float = 1.0,
    w_over_a_ratio: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if candidate not in JOINT_CANDIDATES:
        raise ValueError("Joint candidate set cannot contain AR-only or K-zero routes")
    if alpha < 0 or k_over_a_ratio <= 0 or w_over_a_ratio <= 0:
        raise ValueError("invalid Joint penalty")
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    rows = len(y)
    required = _required_blocks(candidate)
    evaluation_rows = next(
        (
            len(np.asarray(value))
            for value in evaluation_blocks.values()
            if value is not None
        ),
        None,
    )
    if evaluation_rows is None:
        raise ValueError("Joint evaluation has no blocks")
    train: dict[str, np.ndarray] = {}
    evaluation: dict[str, np.ndarray] = {}
    for block in required:
        train[block] = _as_matrix(train_blocks.get(block), rows)
        evaluation[block] = _as_matrix(
            evaluation_blocks.get(block), int(evaluation_rows)
        )
        if train[block].shape[1] == 0:
            raise ValueError(
                f"Joint candidate {candidate} has an empty required {block} block"
            )
        if evaluation[block].shape[1] != train[block].shape[1]:
            raise ValueError(f"Joint train/evaluation {block} columns differ")
    standardized_train = []
    standardized_evaluation = []
    penalty_diagonal = []
    block_contracts: dict[str, Any] = {}
    slices: dict[str, list[int]] = {}
    start = 0
    for block in required:
        block_train = train[block]
        mean = block_train.mean(axis=0, dtype=np.float64)
        scale = block_train.std(axis=0, dtype=np.float64)
        scale[scale * scale < 1e-12] = 1.0
        x = (block_train - mean) / scale
        z = (evaluation[block] - mean) / scale
        standardized_train.append(x)
        standardized_evaluation.append(z)
        ratio = (
            k_over_a_ratio
            if block == "K"
            else w_over_a_ratio
            if block == "W"
            else 1.0
        )
        penalty_diagonal.extend([float(alpha) * float(ratio)] * x.shape[1])
        stop = start + x.shape[1]
        slices[block] = [start, stop]
        block_contracts[block] = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "columns": x.shape[1],
            "penalty_ratio_to_a": float(ratio),
        }
        start = stop
    x = np.concatenate(standardized_train, axis=1)
    z = np.concatenate(standardized_evaluation, axis=1)
    target_mean = float(np.mean(y, dtype=np.float64))
    coefficient, certificate = solve_certified(
        x,
        y - target_mean,
        np.diag(np.asarray(penalty_diagonal, dtype=np.float64)),
    )
    total = z @ coefficient + target_mean
    components: dict[str, np.ndarray] = {}
    for block, (left, right) in slices.items():
        components[block] = z[:, left:right] @ coefficient[left:right]
    input_prediction = components["K"].copy()
    if "W" in components:
        input_prediction += components["W"]
    components["INPUT"] = input_prediction
    components["TOTAL"] = total
    input_indices = []
    for block in ("K", "W"):
        if block in slices:
            left, right = slices[block]
            input_indices.extend(range(left, right))
    contract = {
        "family": candidate,
        "blocks": block_contracts,
        "block_slices": slices,
        "coefficient": coefficient.tolist(),
        "input_coefficient": coefficient[input_indices].tolist(),
        "intercept": target_mean,
        "alpha": float(alpha),
        "k_over_a_ratio": float(k_over_a_ratio),
        "w_over_a_ratio": float(w_over_a_ratio),
        "hard_feature_residualization": False,
        "soft_overlap_penalty": False,
        "input_path_required": True,
        "numerical_certificate": certificate.to_json(),
        "parameter_count": len(coefficient) + 1,
        "block_coefficient_norms": {
            block: float(np.linalg.norm(coefficient[left:right]))
            for block, (left, right) in slices.items()
        },
        "block_prediction_variance": {
            block: float(np.var(values, dtype=np.float64))
            for block, values in components.items()
            if block != "TOTAL"
        },
    }
    return total, contract, components


def predict_joint_candidate(
    blocks: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    candidate = str(contract["family"])
    required = _required_blocks(candidate)
    standardized = []
    rows: int | None = None
    for block in required:
        matrix = np.asarray(blocks[block], dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if rows is None:
            rows = len(matrix)
        elif len(matrix) != rows:
            raise ValueError("Joint prediction blocks have different row counts")
        block_contract = contract["blocks"][block]
        standardized.append(
            (matrix - np.asarray(block_contract["mean"], dtype=np.float64))
            / np.asarray(block_contract["scale"], dtype=np.float64)
        )
    design = np.concatenate(standardized, axis=1)
    coefficient = np.asarray(contract["coefficient"], dtype=np.float64)
    total = design @ coefficient + float(contract["intercept"])
    components: dict[str, np.ndarray] = {}
    for block, (left, right) in contract["block_slices"].items():
        components[block] = (
            design[:, int(left) : int(right)] @ coefficient[int(left) : int(right)]
        )
    components["INPUT"] = components["K"].copy()
    if "W" in components:
        components["INPUT"] += components["W"]
    components["TOTAL"] = total
    return total, components


def _collapsed_joint_result(
    view: Any,
    *,
    reason: str,
    gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preservation = dict(gate or {})
    preservation.update(
        {
            "status": "INPUT_PATH_COLLAPSED",
            "pass": False,
            "reason": reason,
        }
    )
    return {
        "status": "JOINT_INPUT_PATH_COLLAPSED",
        "stage": "E5R_JOINT",
        "dataset": view.head.dataset,
        "target_head": view.head.head_id,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "registered_candidates": list(JOINT_CANDIDATES),
        "applicable_candidates": [],
        "selected_candidate": None,
        "final_selected_candidate": None,
        "input_path_preservation": preservation,
        "input_path_gate": preservation,
        "joint_contract": {
            "status": "NOT_FIT_INPUT_PATH_UNAVAILABLE",
            "input_path_required": True,
            "ar_only_fallback_allowed": False,
        },
        "test_accessed": False,
        "ood_accessed": False,
    }


def _evaluate_joint_candidate(
    train_blocks: Mapping[str, np.ndarray],
    target: np.ndarray,
    evaluation_blocks: Mapping[str, np.ndarray],
    evaluation_target: np.ndarray,
    candidate: tuple[str, float, float, float],
) -> tuple[dict[str, Any], float]:
    route, alpha, ratio_k, ratio_w = candidate
    prediction, contract, _ = fit_joint_candidate(
        train_blocks,
        target,
        evaluation_blocks,
        candidate=route,
        alpha=alpha,
        k_over_a_ratio=ratio_k,
        w_over_a_ratio=ratio_w,
    )
    return contract, (
        mse(evaluation_target, prediction)
        if numerical_contract_passes(contract)
        else float("nan")
    )


def _evaluate_joint_indexed(candidate_index: int) -> tuple[dict[str, Any], float]:
    if _J_CANDIDATE_CONTEXT is None:
        raise RuntimeError("Joint candidate context was not initialized before fork")
    train_blocks, target, evaluation_blocks, evaluation_target, candidates = (
        _J_CANDIDATE_CONTEXT
    )
    return _evaluate_joint_candidate(
        train_blocks,
        target,
        evaluation_blocks,
        evaluation_target,
        candidates[candidate_index],
    )


def evaluate_joint_candidates_ordered(
    train_blocks: Mapping[str, np.ndarray],
    target: np.ndarray,
    evaluation_blocks: Mapping[str, np.ndarray],
    evaluation_target: np.ndarray,
    candidates: list[tuple[str, float, float, float]],
    *,
    workers: int,
) -> list[tuple[dict[str, Any], float]]:
    """Evaluate candidates in registered order, identically in serial or fork mode."""
    global _J_CANDIDATE_CONTEXT
    _J_CANDIDATE_CONTEXT = (
        train_blocks,
        target,
        evaluation_blocks,
        evaluation_target,
        candidates,
    )
    try:
        return ordered_fork_map(
            _evaluate_joint_indexed,
            [(index,) for index in range(len(candidates))],
            workers,
            label="PRISM_V212_METRO_M5_JOINT_INNER",
        )
    finally:
        _J_CANDIDATE_CONTEXT = None


def run_joint_view(
    shared: Path,
    project: Path,
    output: Path,
    view: Any,
    protocol: str = "sru",
) -> dict[str, Any]:
    started = time.time()
    destination = (
        output
        / "DEVELOPMENT"
        / "JOINT"
        / view.head.head_id
        / view.availability_scenario
        / view.proxy_policy
    )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        v211, v21, v2 = load_v211_configs(project, protocol=protocol)
        inner_workers = _j_inner_workers()
        c_result = json.loads(
            (
                output
                / "DEVELOPMENT"
                / "C"
                / view.head.head_id
                / view.proxy_policy
                / "RESULT.json"
            ).read_text(encoding="utf-8")
        )
        w_result = json.loads(
            (
                output
                / "DEVELOPMENT"
                / "W"
                / view.head.head_id
                / view.proxy_policy
                / "RESULT.json"
            ).read_text(encoding="utf-8")
        )
        a_result = json.loads(
            (
                output
                / "DEVELOPMENT"
                / "A"
                / view.head.head_id
                / view.availability_scenario
                / view.proxy_policy
                / "RESULT.json"
            ).read_text(encoding="utf-8")
        )
        if any(item.get("status") != "PASS" for item in (c_result, w_result, a_result)):
            raise RuntimeError("E2R-E4R prerequisite is not PASS")
        c_gate = c_result.get("input_path_preservation", {})
        if not bool(c_gate.get("pass", False)):
            result = _collapsed_joint_result(
                view,
                reason="C_INPUT_PATH_NOT_PRESERVED",
                gate=c_gate,
            )
            write_json(destination / "RESULT.json", result)
            return result
        active = load_active_channels(output, view)
        frozen_channels = set(c_result.get("active_channels", ()))
        active = [item for item in active if item.get("channel") in frozen_channels]
        if not active:
            result = _collapsed_joint_result(view, reason="EMPTY_ACTIVE_K_SUPPORT")
            write_json(destination / "RESULT.json", result)
            return result
        w_oof = pd.read_parquet(
            output / w_result["oof_path"],
            columns=["base_origin_id", "view_sample_id", "oof_fold"],
        )
        c_oof = pd.read_parquet(
            output / c_result["oof_prediction_path"],
            columns=["base_origin_id", "view_sample_id", "oof_fold"],
        )
        pf_w_contract = w_result["w_contract"]
        w_contract = w_result.get("joint_w_basis_contract")
        if not isinstance(w_contract, Mapping):
            raise RuntimeError(
                "Joint W basis construction was not frozen from the registered W pool"
            )
        a_profile = tuple(
            a_result["a_contract"].get(
                "profile", realized_state_profiles(view.head)[0]
            )
        )
        alpha_grid = sorted(
            float(value) for value in v2["J_module"]["ridge_alpha_grid"]
        )
        ratios_k = [float(value) for value in v21["J"]["k_over_a_ratio"]]
        ratios_w = [float(value) for value in v21["J"]["w_over_a_ratio"]]
        available_routes = list(JOINT_CANDIDATES)
        all_candidates = [
            (route, alpha, ratio_k, ratio_w)
            for route in available_routes
            for ratio_k in ratios_k
            for ratio_w in ratios_w
            for alpha in alpha_grid
        ]
        losses = {candidate: [] for candidate in all_candidates}
        contracts_by_candidate: dict[
            tuple[str, float, float, float], list[dict[str, Any]]
        ] = {candidate: [] for candidate in all_candidates}
        ar_losses = {alpha: [] for alpha in alpha_grid}
        fold_payloads = []
        joint_fold_protocol_audit = []
        development_train = load_native_samples(shared, view, "train")
        registered_input_train = load_native_samples(
            shared, _input_only_view(view), "train"
        )
        fold_count = int(v21["selection"]["inner_folds"])
        fit_cap = int(v2["row_caps"]["joint_predictive_fit"])
        evaluation_cap = int(
            v2["row_caps"]["validation_selection_per_fold"]
        )
        joint_folds = registered_joint_inner_fold_frames(
            development_train,
            fold_count=fold_count,
            fit_cap=fit_cap,
            evaluation_cap=evaluation_cap,
            active=active,
        )
        registered_input_folds = registered_joint_inner_fold_frames(
            registered_input_train,
            fold_count=fold_count,
            fit_cap=fit_cap,
            evaluation_cap=evaluation_cap,
            active=active,
        )
        if len(joint_folds) != fold_count or len(registered_input_folds) != fold_count:
            raise JointFoldProtocolMismatch(
                "registered Joint fold count does not match frozen inner_folds"
            )
        inner_target_accessor = BaseAccessor(
            shared,
            view.head.dataset,
            "train",
            [view.head.target],
        )
        for fold_record, registered_input_fold in zip(
            joint_folds, registered_input_folds, strict=True
        ):
            fold = int(fold_record["fold_index"])
            fit = fold_record["fit"]
            evaluation = fold_record["evaluation"]
            registered_input_fold = align_registered_joint_fold(
                fold_record, registered_input_fold
            )
            w_evaluation = align_joint_oof_rows(
                w_oof[w_oof["oof_fold"] == fold].reset_index(drop=True),
                evaluation,
                label=f"W OOF fold {fold}",
            )
            c_evaluation = align_joint_oof_rows(
                c_oof[c_oof["oof_fold"] == fold].reset_index(drop=True),
                evaluation,
                label=f"C OOF fold {fold}",
            )
            protocol_audit = audit_joint_fold_protocol(
                fold_record,
                registered_input_fold,
                w_evaluation,
                c_evaluation,
            )
            joint_fold_protocol_audit.append(protocol_audit)
            if not protocol_audit["pass"]:
                raise JointFoldProtocolMismatch(
                    f"Joint fold {fold} IDs differ from registered C/W evaluation"
                )
            features = fit_physical_features(
                shared,
                view,
                fit,
                evaluation,
                active,
                v2,
                fit_split="train",
                evaluation_split="train",
            )
            k_train = features["joint_train"]
            k_eval = features["joint_evaluation"]
            if k_train.shape[1] == 0:
                result = _collapsed_joint_result(
                    view, reason="EMPTY_K_BLOCK_IN_TRAIN_FOLD"
                )
                write_json(destination / "RESULT.json", result)
                return result
            best_channel = str(c_result["best_active_k_channel"])
            best_index = features["channels"].index(best_channel)
            best_k_eval = features["compressed_evaluation"][:, best_index]
            fit_seed, evaluation_seed, _, _, _ = _fit_c_routed(
                shared,
                view,
                fit,
                evaluation,
                active,
                v2,
                c_result,
                fit_split="train",
                evaluation_split="train",
            )
            if w_contract["family"] == IDENTITY:
                w_train = np.empty((len(fit), 0), dtype=np.float64)
                w_eval = np.empty((len(evaluation), 0), dtype=np.float64)
            else:
                w_train, w_eval, _ = joint_w_basis(
                    fit_seed, evaluation_seed, w_contract
                )
            delta, history = a_profile
            a_train = inner_target_accessor.target_state(
                fit, view.head.target, delta, history
            )
            a_eval = inner_target_accessor.target_state(
                evaluation, view.head.target, delta, history
            )
            target = fit["y_true"].to_numpy(dtype=np.float64)
            evaluation_target = evaluation["y_true"].to_numpy(dtype=np.float64)
            train_blocks = {"K": k_train, "W": w_train, "A": a_train}
            evaluation_blocks = {"K": k_eval, "W": w_eval, "A": a_eval}
            candidate_results = evaluate_joint_candidates_ordered(
                train_blocks,
                target,
                evaluation_blocks,
                evaluation_target,
                all_candidates,
                workers=inner_workers,
            )
            for candidate, (contract, candidate_loss) in zip(
                all_candidates, candidate_results, strict=True
            ):
                contracts_by_candidate[candidate].append(contract)
                losses[candidate].append(candidate_loss)
            for alpha in alpha_grid:
                prediction, _ = fit_mature_residual_ar(
                    a_train,
                    target,
                    a_eval,
                    alpha=alpha,
                    mu=0.0,
                )
                ar_losses[alpha].append(mse(evaluation_target, prediction))
            fold_payloads.append(
                {
                    "fit": fit,
                    "evaluation": evaluation,
                    "k_train": k_train,
                    "k_eval": k_eval,
                    "w_train": w_train,
                    "w_eval": w_eval,
                    "a_train": a_train,
                    "a_eval": a_eval,
                    "evaluation_target": evaluation_target,
                    "best_k_eval": best_k_eval,
                }
            )

        minimum_folds = int(v21["selection"]["minimum_usable_folds"])
        ratio_candidates: dict[str, list[tuple[str, float, float, float]]] = {
            route: [] for route in available_routes
        }
        ridge_audits: dict[str, Any] = {}
        for route in available_routes:
            for ratio_k in ratios_k:
                for ratio_w in ratios_w:
                    contracts_by_alpha = {
                        alpha: contracts_by_candidate[
                            (route, alpha, ratio_k, ratio_w)
                        ]
                        for alpha in alpha_grid
                    }
                    try:
                        selected_alpha, ridge_audit = _smallest_stable_alpha(
                            alpha_grid, contracts_by_alpha, minimum_folds
                        )
                    except RuntimeError:
                        continue
                    candidate = (route, selected_alpha, ratio_k, ratio_w)
                    ratio_candidates[route].append(candidate)
                    ridge_audits[str((route, ratio_k, ratio_w))] = ridge_audit
        route_best: dict[str, tuple[str, float, float, float]] = {}
        for route in available_routes:
            subset = {
                candidate: losses[candidate]
                for candidate in ratio_candidates[route]
            }
            if not subset:
                continue
            route_best[route] = one_se_select(
                subset,
                lambda value: (
                    abs(np.log(value[2])),
                    abs(np.log(value[3])),
                ),
                minimum_usable_folds=minimum_folds,
            ).selected
        if J_K not in route_best:
            raise RuntimeError("Joint J_K has no numerically stable candidate")
        neutral = route_best[J_K]
        reduced = {candidate: losses[candidate] for candidate in route_best.values()}
        selection = guarded_local_one_se_select(
            reduced,
            lambda value: (
                JOINT_CANDIDATES.index(value[0]),
                abs(np.log(value[2])),
                abs(np.log(value[3])),
            ),
            neutral=neutral,
            minimum_relative_improvement=float(
                v21["selection"]["minimum_relative_improvement"]["J"]
            ),
            minimum_positive_fraction=float(
                v21["selection"]["minimum_positive_fold_fraction"]
            ),
            minimum_usable_folds=minimum_folds,
        )
        selected = selection.final_selected_candidate
        selected_route, selected_alpha, selected_rk, selected_rw = selected
        ar_selection = one_se_select(
            ar_losses,
            lambda value: (-value,),
            minimum_usable_folds=minimum_folds,
        )
        total_oof = []
        input_oof = []
        best_k_oof = []
        target_oof = []
        input_coefficients = []
        numeric_passes = []
        correlations = []
        for payload in fold_payloads:
            prediction, contract, components = fit_joint_candidate(
                {
                    "K": payload["k_train"],
                    "W": payload["w_train"],
                    "A": payload["a_train"],
                },
                payload["fit"]["y_true"].to_numpy(dtype=np.float64),
                {
                    "K": payload["k_eval"],
                    "W": payload["w_eval"],
                    "A": payload["a_eval"],
                },
                candidate=selected_route,
                alpha=selected_alpha,
                k_over_a_ratio=selected_rk,
                w_over_a_ratio=selected_rw,
            )
            total_oof.append(prediction)
            input_oof.append(components["INPUT"])
            best_k_oof.append(payload["best_k_eval"])
            target_oof.append(payload["evaluation_target"])
            input_coefficients.append(
                np.asarray(contract["input_coefficient"], dtype=np.float64)
            )
            numeric_passes.append(numerical_contract_passes(contract))
            arrays = [
                components.get(name, np.zeros(len(prediction), dtype=np.float64))
                for name in ("K", "W", "A")
            ]
            correlations.append(np.corrcoef(np.column_stack(arrays), rowvar=False).tolist())
        gate_parameters = _gate_config(v211)
        oof_gate = input_path_preservation_gate(
            np.concatenate(target_oof),
            np.concatenate(total_oof),
            np.concatenate(best_k_oof),
            input_prediction=np.concatenate(input_oof),
            nonintercept_coefficients=np.concatenate(input_coefficients),
            numerical_certificate_passed=all(numeric_passes),
            **gate_parameters,
        )
        registered_assembly_train = apply_assembly_support(
            registered_input_train, active
        )
        if support_id_hash(registered_assembly_train) != c_result.get(
            "assembly_train_support_hash"
        ):
            raise JointFoldProtocolMismatch(
                "frozen C assembly train support could not be replayed"
            )
        assembly_development_train = intersect_by_base_origin_id(
            apply_assembly_support(development_train, active),
            registered_assembly_train,
            label="Joint development train",
        )
        train = _cap(
            assembly_development_train,
            int(v2["row_caps"]["joint_predictive_fit"]),
        )
        registered_assembly_validation = apply_assembly_support(
            load_native_samples(shared, _input_only_view(view), "validation"),
            active,
        )
        if support_id_hash(registered_assembly_validation) != c_result.get(
            "assembly_validation_support_hash"
        ):
            raise JointFoldProtocolMismatch(
                "frozen C assembly validation support could not be replayed"
            )
        validation = intersect_by_base_origin_id(
            apply_assembly_support(
                load_native_samples(shared, view, "validation"), active
            ),
            registered_assembly_validation,
            label="Joint validation",
        )
        final_features = fit_physical_features(
            shared,
            view,
            train,
            validation,
            active,
            v2,
            fit_split="train",
            evaluation_split="validation",
        )
        k_train = final_features["joint_train"]
        k_validation = final_features["joint_evaluation"]
        best_index = final_features["channels"].index(
            str(c_result["best_active_k_channel"])
        )
        best_k_validation = final_features["compressed_evaluation"][:, best_index]
        fit_seed, validation_seed, _, _, _ = _fit_c_routed(
            shared,
            view,
            train,
            validation,
            active,
            v2,
            c_result,
            fit_split="train",
            evaluation_split="validation",
        )
        if w_contract["family"] == IDENTITY:
            w_train = np.empty((len(train), 0), dtype=np.float64)
            w_validation = np.empty((len(validation), 0), dtype=np.float64)
        else:
            w_train, w_validation, _ = joint_w_basis(
                fit_seed, validation_seed, w_contract
            )
        delta, history = a_profile
        final_target_accessor = BaseAccessor(
            shared,
            view.head.dataset,
            "validation",
            [view.head.target],
        )
        a_train = final_target_accessor.target_state(
            train, view.head.target, delta, history
        )
        a_validation = final_target_accessor.target_state(
            validation, view.head.target, delta, history
        )
        prediction, contract, components = fit_joint_candidate(
            {"K": k_train, "W": w_train, "A": a_train},
            train["y_true"].to_numpy(dtype=np.float64),
            {"K": k_validation, "W": w_validation, "A": a_validation},
            candidate=selected_route,
            alpha=selected_alpha,
            k_over_a_ratio=selected_rk,
            w_over_a_ratio=selected_rw,
        )
        validation_gate = input_path_preservation_gate(
            validation["y_true"].to_numpy(dtype=np.float64),
            prediction,
            best_k_validation,
            input_prediction=components["INPUT"],
            nonintercept_coefficients=np.asarray(
                contract["input_coefficient"], dtype=np.float64
            ),
            numerical_certificate_passed=numerical_contract_passes(contract),
            **gate_parameters,
        )
        final_numerical_pass = numerical_contract_passes(contract)
        formal_pass = bool(oof_gate["pass"] and final_numerical_pass)
        formal_gate = {
            **oof_gate,
            "status": "INPUT_PATH_PRESERVED"
            if formal_pass
            else "INPUT_PATH_COLLAPSED",
            "pass": formal_pass,
            "input_path_failure_class": (
                oof_gate.get("input_path_failure_class")
                if not oof_gate["pass"]
                else (
                    "INPUT_PATH_PRESERVED"
                    if final_numerical_pass
                    else "INPUT_PATH_NUMERICAL_FAILURE"
                )
            ),
            "final_refit_numerical_certificate_passed": final_numerical_pass,
        }
        gate = attach_nonselecting_validation_confirmation(
            formal_gate, validation_gate
        )
        route_materializations: dict[str, Any] = {}
        for route, route_candidate in sorted(route_best.items()):
            _, route_alpha, route_rk, route_rw = route_candidate
            route_prediction, route_contract, route_components = fit_joint_candidate(
                {"K": k_train, "W": w_train, "A": a_train},
                train["y_true"].to_numpy(dtype=np.float64),
                {"K": k_validation, "W": w_validation, "A": a_validation},
                candidate=route,
                alpha=float(route_alpha),
                k_over_a_ratio=float(route_rk),
                w_over_a_ratio=float(route_rw),
            )
            if route in {J_KW, J_KWA} and int(
                route_contract.get("blocks", {}).get("W", {}).get("columns", 0)
            ) <= 0:
                raise RuntimeError("Joint W route did not jointly fit a W basis block")
            route_frame = validation[
                [
                    "base_origin_id",
                    "view_sample_id",
                    "entity_id",
                    "origin",
                    "latest_available_target_index",
                    "y_true",
                ]
            ].copy()
            route_frame["y_pred"] = route_prediction
            route_frame["input_prediction"] = route_components["INPUT"]
            route_frame["model"] = f"PRISM_V2_1_2_{route}"
            route_frame["dtype"] = "float64"
            route_path = destination / f"validation_{route}.parquet"
            route_frame.to_parquet(route_path, index=False, compression="zstd")
            route_materializations[route] = {
                "selected_hyperparameters": list(route_candidate),
                "prediction_path": str(route_path.relative_to(output)),
                "prediction_sha256": sha256_file(route_path),
                "prediction_loss": mse(
                    route_frame["y_true"].to_numpy(dtype=np.float64),
                    route_prediction,
                ),
                "contract": route_contract,
                "block_dimensions": {
                    block: int(value["columns"])
                    for block, value in route_contract.get("blocks", {}).items()
                },
                "w_coefficients_jointly_fitted": route in {J_KW, J_KWA},
            }
        frame = validation[
            [
                "base_origin_id",
                "view_sample_id",
                "entity_id",
                "origin",
                "latest_available_target_index",
                "y_true",
            ]
        ].copy()
        frame["y_pred"] = prediction
        frame["input_prediction"] = components["INPUT"]
        frame["best_active_k_prediction"] = best_k_validation
        frame["model"] = f"PRISM_V2_1_2_{selected_route}"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        final_loss = mse(frame["y_true"].to_numpy(dtype=np.float64), prediction)
        result = {
            "status": (
                "PASS"
                if gate["pass"]
                else "JOINT_OOF_PROTOCOL_CORRECTED_BUT_MODEL_GATE_FAILED"
            ),
            "stage": "E5R_JOINT",
            "inner_candidate_workers": inner_workers,
            "inner_parallelism_scope": "ORDERED_INDEPENDENT_CANDIDATES_ONLY",
            "dataset": view.head.dataset,
            "target_head": view.head.head_id,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "registered_candidates": list(JOINT_CANDIDATES),
            "applicable_candidates": available_routes,
            "pf_w_contract": pf_w_contract,
            "joint_w_basis_contract": w_contract,
            "joint_w_coefficients_jointly_fitted": True,
            "joint_w_prefit_scalar_forbidden": True,
            "route_local_selected": {
                route: list(candidate) for route, candidate in route_best.items()
            },
            "route_materializations": route_materializations,
            "ridge_semantics": "NUMERICAL_STABILITY_ONLY",
            "minimal_stabilizing_ridge_audits": ridge_audits,
            "ar_profile": list(a_profile),
            "selected_candidate": selected_route,
            "selection": selection.to_json(),
            "candidate_fold_losses": {
                str(key): value for key, value in losses.items()
            },
            "ar_only_diagnostic": {
                "selection_eligible": False,
                "selected_alpha": ar_selection.selected,
                "fold_losses": ar_losses[ar_selection.selected],
            },
            "input_path_preservation": gate,
            "input_path_gate": gate,
            "input_path_failure_class": gate.get(
                "input_path_failure_class", "INPUT_PATH_PRESERVED"
            ),
            "joint_fold_protocol_audit": joint_fold_protocol_audit,
            "joint_fold_protocol_audit_pass": all(
                item["pass"] for item in joint_fold_protocol_audit
            ),
            "assembly_support_contract": c_result.get(
                "assembly_support_contract"
            ),
            "joint_raw_input_train_support_hash": support_id_hash(
                assembly_development_train
            ),
            "joint_raw_input_validation_support_hash": support_id_hash(
                validation
            ),
            "joint_raw_input_validation_base_origin_support_hash": base_origin_support_hash(
                validation
            ),
            "c_assembly_validation_support_hash": c_result.get(
                "assembly_validation_support_hash"
            ),
            "joint_raw_input_support_matches_c_assembly": base_origin_support_hash(
                validation
            )
            == c_result.get("assembly_validation_base_origin_support_hash"),
            "joint_raw_input_support_is_subset_of_c_assembly": set(
                validation["base_origin_id"].astype(str)
            ).issubset(
                set(registered_assembly_validation["base_origin_id"].astype(str))
            ),
            "joint_raw_input_support_contract": (
                "DYNAMIC_AVAILABILITY_INTERSECTION_C_ASSEMBLY"
            ),
            "joint_k_representations_share_rows": True,
            "joint_fit_source": "ORIGINAL_REGISTERED_ANCHOR_INNER_TRAIN_SUPPORT_AFTER_ASSEMBLY_MASK",
            "joint_evaluation_source": (
                "ORIGINAL_REGISTERED_ANCHOR_INNER_VALIDATION_SUPPORT_AFTER_ASSEMBLY_MASK"
            ),
            "nested_oof_training_used": False,
            "w_physical_oof_used_as_training_pool": False,
            "w_physical_oof_usage": "CROSS_STAGE_FOLD_ID_AND_PROVENANCE_AUDIT_ONLY",
            "registered_inner_fold_count": fold_count,
            "candidate_fold_loss_count": len(losses[selected]),
            "block_correlations_by_fold": correlations,
            "joint_contract": contract,
            "final_selected_candidate": selected_route,
            "final_selected_fold_losses": list(losses[selected]),
            "final_selected_prediction_path": str(prediction_path.relative_to(output)),
            "final_selected_contract": contract,
            "final_prediction_loss": final_loss,
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "row_cap_audit": {
                "cap_name": "joint_predictive_fit",
                "cap": int(v2["row_caps"]["joint_predictive_fit"]),
                "fit_rows": len(train),
                "validation_rows": len(validation),
                "fit_source": "train_only",
                "assembly_support_mask_before_cap": True,
            },
            "test_accessed": False,
            "ood_accessed": False,
            "elapsed_seconds": time.time() - started,
            **regression_metrics(
                frame["y_true"].to_numpy(dtype=np.float64), prediction
            ),
        }
    except JointFoldProtocolMismatch as error:
        result = {
            "status": "STOP_JOINT_FOLD_PROTOCOL_MISMATCH",
            "stage": "E5R_JOINT",
            "target_head": view.head.head_id,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "joint_fit_source": "ORIGINAL_REGISTERED_ANCHOR_INNER_TRAIN_SUPPORT_AFTER_ASSEMBLY_MASK",
            "joint_evaluation_source": (
                "ORIGINAL_REGISTERED_ANCHOR_INNER_VALIDATION_SUPPORT_AFTER_ASSEMBLY_MASK"
            ),
            "nested_oof_training_used": False,
            "w_physical_oof_used_as_training_pool": False,
            "joint_fold_protocol_audit": locals().get(
                "joint_fold_protocol_audit", []
            ),
            "test_accessed": False,
            "ood_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    except Exception as error:
        result = {
            "status": "SOLVER_FAILED_RETAINED",
            "stage": "E5R_JOINT",
            "target_head": view.head.head_id,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "test_accessed": False,
            "ood_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
    write_json(destination / "RESULT.json", result)
    return result


def run_e5r_joint(shared: Path, project: Path, output: Path) -> dict[str, Any]:
    from .v211_assembly import build_joint_card
    from .v21_views import sru_dynamic_views

    views = sru_dynamic_views(shared)
    results = run_parallel(
        run_joint_view,
        [(shared, project, output, view) for view in views],
        int(os.environ.get("PRISM_V211_WORKERS", "8")),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "4")),
        label="PRISM_V211_E5R_JOINT",
    )
    by_key = {
        (
            result.get("target_head"),
            result.get("availability_scenario"),
            result.get("proxy_policy"),
        ): result
        for result in results
    }
    for view in views:
        result = by_key[
            (view.head.head_id, view.availability_scenario, view.proxy_policy)
        ]
        if result.get("status") not in {
            "PASS",
            "JOINT_INPUT_PATH_COLLAPSED",
            "JOINT_OOF_PROTOCOL_CORRECTED_BUT_MODEL_GATE_FAILED",
        }:
            continue
        card = build_joint_card(result)
        card.update(
            {
                "target_head": view.head.head_id,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "validation_prediction_path": result.get("prediction_path"),
            }
        )
        write_json(
            output
            / "ASSEMBLY_CARDS"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "JOINT_ASSEMBLY_CARD.json",
            card,
        )
    summary = {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in results)
        else "COMPLETED_WITH_RETAINED_FAILURES",
        "stage": "E5R_JOINT",
        "views": len(results),
        "pass": sum(item["status"] == "PASS" for item in results),
        "collapsed": sum(
            item["status"]
            in {
                "JOINT_INPUT_PATH_COLLAPSED",
                "JOINT_OOF_PROTOCOL_CORRECTED_BUT_MODEL_GATE_FAILED",
            }
            for item in results
        ),
        "test_accessed": False,
    }
    write_json(output / "DEVELOPMENT" / "JOINT" / "SUMMARY.json", summary)
    return summary
