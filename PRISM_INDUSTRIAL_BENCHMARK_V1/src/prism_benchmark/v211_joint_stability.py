from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cpu_data import BaseAccessor, realized_state_profiles, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v2_c import fit_physical_features
from .v2_k import _cap
from .v2_numerics import solve_certified_gram
from .v2_runtime import ordered_fork_map
from .v2_selection import one_se_select, practical_activation
from .v21_selection import guarded_local_one_se_select
from .v211_c import _gate_config
from .v211_config import load_v211_configs
from .v211_joint import (
    J_K,
    J_KA,
    J_KW,
    J_KWA,
    JOINT_CANDIDATES,
    JointFoldProtocolMismatch,
    _input_only_view,
    audit_joint_fold_protocol,
    fit_joint_candidate,
    joint_w_basis,
    registered_joint_inner_fold_frames,
)
from .v211_k import load_active_channels
from .v211_metro_contracts import stable_candidate_id
from .v211_selection import (
    attach_nonselecting_validation_confirmation,
    input_path_preservation_gate,
    numerical_contract_passes,
)
from .v211_support import (
    apply_assembly_support,
    base_origin_support_hash,
    load_native_samples,
    support_id_hash,
)
from .v211_w import IDENTITY, _fit_c_routed
from .v211_joint_stability_config import (
    CHANNEL_COMPRESSED,
    ETA_PRED_GRID,
    FULL_BASIS,
    JOINT_ESTIMATOR_SEMANTICS,
    K_REPRESENTATIONS,
    MODEL_VERSION,
    PRACTICE_REVISION,
    load_joint_stability_config,
)


@dataclass(frozen=True)
class StabilityCandidate:
    route: str
    k_representation: str
    numerical_alpha: float
    predictive_eta: float

    def descriptor(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "k_representation": self.k_representation,
            "numerical_alpha": self.numerical_alpha,
            "predictive_eta": self.predictive_eta,
        }

    def key(self) -> str:
        return (
            f"{self.route}|{self.k_representation}|"
            f"alpha={self.numerical_alpha:.17g}|eta={self.predictive_eta:.17g}"
        )


@dataclass
class PreparedRepresentation:
    k_representation: str
    fit_rows: int
    evaluation_rows: int
    target_mean: float
    gram: np.ndarray
    rhs: np.ndarray
    evaluation_design: np.ndarray
    block_slices: dict[str, tuple[int, int]]
    block_contracts: dict[str, dict[str, Any]]
    raw_k_support: tuple[str, ...]


_STABILITY_EVALUATION_CONTEXT: tuple[
    list[dict[str, PreparedRepresentation]],
    list[np.ndarray],
    list[StabilityCandidate],
] | None = None


def registered_joint_stability_candidates() -> tuple[str, ...]:
    return JOINT_CANDIDATES


def _required_blocks(route: str) -> tuple[str, ...]:
    if route == J_K:
        return ("K",)
    if route == J_KW:
        return ("K", "W")
    if route == J_KA:
        return ("K", "A")
    if route == J_KWA:
        return ("K", "W", "A")
    raise ValueError(f"unregistered Joint route: {route}")


def _matrix(value: np.ndarray, rows: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or len(matrix) != rows:
        raise ValueError("v2.1.1 Joint stability practice Joint block has invalid shape")
    return matrix


def k_representation_blocks(
    features: Mapping[str, Any],
    active_channels: Sequence[str],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    channels = tuple(str(value) for value in features["channels"])
    expected = tuple(str(value) for value in active_channels)
    if channels != expected:
        raise RuntimeError("v2.1.1 Joint stability practice K representation raw support differs from frozen active K")
    compressed_train = np.asarray(features["compressed_train"], dtype=np.float64)
    compressed_evaluation = np.asarray(
        features["compressed_evaluation"], dtype=np.float64
    )
    full_train = np.asarray(features["joint_train"], dtype=np.float64)
    full_evaluation = np.asarray(features["joint_evaluation"], dtype=np.float64)
    if compressed_train.shape[1] != len(expected):
        raise RuntimeError("CHANNEL_COMPRESSED columns differ from active K count")
    if full_train.shape[1] < compressed_train.shape[1]:
        raise RuntimeError("FULL_BASIS unexpectedly has fewer columns than compressed K")
    return (
        {
            CHANNEL_COMPRESSED: (compressed_train, compressed_evaluation),
            FULL_BASIS: (full_train, full_evaluation),
        },
        {
            "raw_k_support": list(expected),
            "channel_compressed_columns": int(compressed_train.shape[1]),
            "full_basis_columns": int(full_train.shape[1]),
            "same_raw_support": True,
            "zeroed_raw_channels_resurrected": False,
            "physical_feature_construction_calls": 1,
        },
    )


def prepare_joint_representation(
    train_blocks: Mapping[str, np.ndarray],
    target: np.ndarray,
    evaluation_blocks: Mapping[str, np.ndarray],
    *,
    k_representation: str,
    raw_k_support: Sequence[str],
) -> PreparedRepresentation:
    if k_representation not in K_REPRESENTATIONS:
        raise ValueError(f"unregistered K representation: {k_representation}")
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    rows = len(y)
    evaluation_rows = len(np.asarray(evaluation_blocks["K"]))
    standardized_train: list[np.ndarray] = []
    standardized_evaluation: list[np.ndarray] = []
    block_slices: dict[str, tuple[int, int]] = {}
    block_contracts: dict[str, dict[str, Any]] = {}
    start = 0
    for block in ("K", "W", "A"):
        train = _matrix(train_blocks[block], rows)
        evaluation = _matrix(evaluation_blocks[block], evaluation_rows)
        if train.shape[1] != evaluation.shape[1]:
            raise ValueError(f"v2.1.1 Joint stability practice Joint {block} train/evaluation columns differ")
        if train.shape[1] == 0:
            block_slices[block] = (start, start)
            block_contracts[block] = {
                "mean": [],
                "scale": [],
                "columns": 0,
                "predictive_penalty_ratio": 1.0,
            }
            continue
        mean = train.mean(axis=0, dtype=np.float64)
        scale = train.std(axis=0, dtype=np.float64)
        scale[scale * scale < 1e-12] = 1.0
        x = (train - mean) / scale
        z = (evaluation - mean) / scale
        standardized_train.append(x)
        standardized_evaluation.append(z)
        stop = start + x.shape[1]
        block_slices[block] = (start, stop)
        block_contracts[block] = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "columns": int(x.shape[1]),
            "predictive_penalty_ratio": 1.0,
        }
        start = stop
    x_all = np.concatenate(standardized_train, axis=1)
    z_all = np.concatenate(standardized_evaluation, axis=1)
    target_mean = float(np.mean(y, dtype=np.float64))
    centered = y - target_mean
    gram = x_all.T @ x_all
    rhs = x_all.T @ centered
    return PreparedRepresentation(
        k_representation=k_representation,
        fit_rows=rows,
        evaluation_rows=evaluation_rows,
        target_mean=target_mean,
        gram=gram,
        rhs=rhs,
        evaluation_design=z_all,
        block_slices=block_slices,
        block_contracts=block_contracts,
        raw_k_support=tuple(str(value) for value in raw_k_support),
    )


def predictive_penalty_scale(fit_rows: int, predictive_eta: float) -> float:
    if fit_rows < 1 or predictive_eta < 0:
        raise ValueError("invalid predictive penalty")
    return float(fit_rows) * float(predictive_eta)


def _route_indices(
    prepared: PreparedRepresentation, route: str
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    indices: list[int] = []
    slices: dict[str, tuple[int, int]] = {}
    start = 0
    for block in _required_blocks(route):
        left, right = prepared.block_slices[block]
        if right <= left:
            raise ValueError(f"v2.1.1 Joint stability practice Joint route {route} has empty {block} block")
        block_indices = list(range(left, right))
        indices.extend(block_indices)
        stop = start + len(block_indices)
        slices[block] = (start, stop)
        start = stop
    return np.asarray(indices, dtype=np.int64), slices


def _effective_df(gram: np.ndarray, penalty: np.ndarray) -> float:
    system = (gram + penalty + (gram + penalty).T) * 0.5
    try:
        inverse = np.linalg.pinv(system, rcond=1e-12)
        return float(np.trace(gram @ inverse))
    except np.linalg.LinAlgError:
        return float("nan")


def solve_prepared_stability(
    prepared: PreparedRepresentation,
    *,
    route: str,
    numerical_alpha: float,
    predictive_eta: float,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if route not in JOINT_CANDIDATES:
        raise ValueError("v2.1.1 Joint stability practice Joint cannot select AR-only or K-zero")
    if numerical_alpha < 0 or predictive_eta < 0:
        raise ValueError("negative ridge is invalid")
    indices, route_slices = _route_indices(prepared, route)
    gram = prepared.gram[np.ix_(indices, indices)]
    rhs = prepared.rhs[indices]
    z = prepared.evaluation_design[:, indices]
    predictive_scale = predictive_penalty_scale(
        prepared.fit_rows, predictive_eta
    )
    diagonal_value = float(numerical_alpha) + predictive_scale
    penalty = np.eye(len(indices), dtype=np.float64) * diagonal_value
    coefficient, certificate = solve_certified_gram(gram, rhs, penalty)
    total = z @ coefficient + prepared.target_mean
    components: dict[str, np.ndarray] = {}
    for block, (left, right) in route_slices.items():
        components[block] = z[:, left:right] @ coefficient[left:right]
    components["INPUT"] = components["K"].copy()
    if "W" in components:
        components["INPUT"] += components["W"]
    components["TOTAL"] = total
    input_indices: list[int] = []
    for block in ("K", "W"):
        if block in route_slices:
            left, right = route_slices[block]
            input_indices.extend(range(left, right))
    contract = {
        "family": route,
        "estimator_version": MODEL_VERSION,
        "model_version": MODEL_VERSION,
        "practice_revision": PRACTICE_REVISION,
        "joint_estimator_semantics": JOINT_ESTIMATOR_SEMANTICS,
        "k_representation": prepared.k_representation,
        "raw_k_support": list(prepared.raw_k_support),
        "numerical_alpha": float(numerical_alpha),
        "predictive_eta": float(predictive_eta),
        "predictive_penalty_scale": predictive_scale,
        "predictive_block_ratio": 1.0,
        "fit_rows": prepared.fit_rows,
        "blocks": {
            block: prepared.block_contracts[block]
            for block in _required_blocks(route)
        },
        "block_slices": {
            block: [left, right]
            for block, (left, right) in route_slices.items()
        },
        "coefficient": coefficient.tolist(),
        "input_coefficient": coefficient[input_indices].tolist(),
        "intercept": prepared.target_mean,
        "numerical_certificate": certificate.to_json(),
        "parameter_count": len(coefficient) + 1,
        "effective_df": _effective_df(gram, penalty),
        "block_coefficient_norms": {
            block: float(np.linalg.norm(coefficient[left:right]))
            for block, (left, right) in route_slices.items()
        },
        "block_prediction_variance": {
            block: float(np.var(values, dtype=np.float64))
            for block, values in components.items()
            if block != "TOTAL"
        },
        "total_prediction_variance": float(np.var(total, dtype=np.float64)),
        "hard_feature_residualization": False,
        "soft_overlap_penalty": False,
        "input_path_required": True,
    }
    return total, contract, components


def solve_prepared_legacy_anchor(
    prepared: PreparedRepresentation,
    *,
    route: str,
    alpha: float,
    k_over_a_ratio: float,
    w_over_a_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    indices, route_slices = _route_indices(prepared, route)
    gram = prepared.gram[np.ix_(indices, indices)]
    rhs = prepared.rhs[indices]
    z = prepared.evaluation_design[:, indices]
    diagonal: list[float] = []
    for block in _required_blocks(route):
        left, right = route_slices[block]
        ratio = (
            k_over_a_ratio
            if block == "K"
            else w_over_a_ratio
            if block == "W"
            else 1.0
        )
        diagonal.extend([float(alpha) * float(ratio)] * (right - left))
    coefficient, certificate = solve_certified_gram(
        gram, rhs, np.diag(np.asarray(diagonal, dtype=np.float64))
    )
    return z @ coefficient + prepared.target_mean, certificate.to_json()


def fit_joint_candidate_stability(
    train_blocks: Mapping[str, np.ndarray],
    target: np.ndarray,
    evaluation_blocks: Mapping[str, np.ndarray],
    *,
    candidate: str,
    k_representation: str,
    numerical_alpha: float,
    predictive_eta: float,
    raw_k_support: Sequence[str] = (),
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    prepared = prepare_joint_representation(
        train_blocks,
        target,
        evaluation_blocks,
        k_representation=k_representation,
        raw_k_support=raw_k_support,
    )
    return solve_prepared_stability(
        prepared,
        route=candidate,
        numerical_alpha=numerical_alpha,
        predictive_eta=predictive_eta,
    )


# Historical execution compatibility.  This is deliberately an alias to the
# canonical implementation so the abf7 callable and the renamed callable can
# be regression-compared without maintaining a second estimator body.
fit_joint_candidate_v22 = fit_joint_candidate_stability


def select_smallest_numerical_alpha(
    alpha_grid: Sequence[float],
    contracts_by_alpha: Mapping[float, Sequence[Mapping[str, Any]]],
    *,
    required_folds: int,
) -> tuple[float, dict[str, Any]]:
    audit: dict[str, Any] = {}
    for alpha in sorted(set(float(value) for value in alpha_grid)):
        contracts = list(contracts_by_alpha[alpha])
        passes = [numerical_contract_passes(contract) for contract in contracts]
        audit[str(alpha)] = {
            "passing_folds": int(sum(passes)),
            "total_folds": len(passes),
            "certificates": [
                contract.get("numerical_certificate", {}) for contract in contracts
            ],
            "selection_loss_used": False,
        }
        if sum(passes) == required_folds and len(passes) == required_folds:
            return alpha, {
                "selection": "SMALLEST_ALL_REGISTERED_FOLDS_CERTIFIED_ALPHA",
                "selected_numerical_alpha": alpha,
                "selection_loss_used": False,
                "candidates": audit,
            }
    raise RuntimeError("no numerical alpha certified all registered folds")


def select_predictive_eta(
    eta_fold_losses: Mapping[float, list[float]],
    *,
    minimum_usable_folds: int = 4,
) -> tuple[float, dict[str, Any]]:
    selection = one_se_select(
        eta_fold_losses,
        lambda eta: (-float(eta),),
        minimum_usable_folds=minimum_usable_folds,
    )
    return float(selection.selected), selection.to_json()


def select_k_representation(
    compressed_losses: list[float],
    full_losses: list[float],
    *,
    minimum_relative_improvement: float,
    minimum_positive_fraction: float,
    minimum_usable_folds: int = 4,
) -> tuple[str, dict[str, Any]]:
    selection = guarded_local_one_se_select(
        {
            CHANNEL_COMPRESSED: compressed_losses,
            FULL_BASIS: full_losses,
        },
        lambda value: (0 if value == CHANNEL_COMPRESSED else 1,),
        neutral=CHANNEL_COMPRESSED,
        minimum_relative_improvement=minimum_relative_improvement,
        minimum_positive_fraction=minimum_positive_fraction,
        minimum_usable_folds=minimum_usable_folds,
    )
    return str(selection.final_selected_candidate), selection.to_json()


def stability_candidate_id(view_key: str, candidate: StabilityCandidate) -> str:
    return stable_candidate_id(
        "JOINT_STABILITY",
        {"view": view_key, **candidate.descriptor()},
    )


def stability_guarded_selection_json(selection: Any) -> dict[str, Any]:
    def descriptor(candidate: StabilityCandidate) -> dict[str, Any]:
        return {"candidate_key": candidate.key(), **candidate.descriptor()}

    return {
        "best_candidate": descriptor(selection.best_candidate),
        "best_mean": float(selection.best_mean),
        "best_standard_error": float(selection.best_standard_error),
        "acceptable_threshold": float(selection.acceptable_threshold),
        "acceptable_candidates": [
            descriptor(candidate) for candidate in selection.acceptable_candidates
        ],
        "passing_active_candidates": [
            descriptor(candidate)
            for candidate in selection.passing_active_candidates
        ],
        "final_selected_candidate": descriptor(
            selection.final_selected_candidate
        ),
        "final_selected_fold_losses": [
            float(value) for value in selection.final_selected_fold_losses
        ],
        "activation_audit": {
            candidate.key(): audit
            for candidate, audit in selection.activation_audit.items()
        },
        "usable_fold_count": {
            candidate.key(): int(value)
            for candidate, value in selection.usable_fold_count.items()
        },
        "means": {
            candidate.key(): float(value)
            for candidate, value in selection.means.items()
        },
        "standard_errors": {
            candidate.key(): float(value)
            for candidate, value in selection.standard_errors.items()
        },
    }


def _evaluate_candidate(candidate_index: int) -> dict[str, Any]:
    if _STABILITY_EVALUATION_CONTEXT is None:
        raise RuntimeError("v2.1.1 Joint stability practice candidate context is not initialized")
    prepared_folds, evaluation_targets, candidates = _STABILITY_EVALUATION_CONTEXT
    candidate = candidates[candidate_index]
    fold_losses: list[float] = []
    fold_diagnostics: list[dict[str, Any]] = []
    for fold_index, (prepared_by_representation, target) in enumerate(
        zip(prepared_folds, evaluation_targets, strict=True)
    ):
        prediction, contract, components = solve_prepared_stability(
            prepared_by_representation[candidate.k_representation],
            route=candidate.route,
            numerical_alpha=candidate.numerical_alpha,
            predictive_eta=candidate.predictive_eta,
        )
        loss = (
            mse(target, prediction)
            if numerical_contract_passes(contract)
            else float("nan")
        )
        fold_losses.append(loss)
        fold_diagnostics.append(
            {
                "fold_index": fold_index,
                "loss": loss,
                "effective_df": contract["effective_df"],
                "coefficient_l2": float(
                    np.linalg.norm(
                        np.asarray(contract["coefficient"], dtype=np.float64)
                    )
                ),
                "input_prediction_variance": float(
                    np.var(components["INPUT"], dtype=np.float64)
                ),
                "total_prediction_variance": float(
                    np.var(prediction, dtype=np.float64)
                ),
                "numerical_certificate": contract["numerical_certificate"],
            }
        )
    finite = np.asarray(fold_losses, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "candidate": candidate.descriptor(),
        "candidate_key": candidate.key(),
        "fold_losses": fold_losses,
        "mean_loss": float(np.mean(finite, dtype=np.float64)),
        "standard_error": float(
            np.std(finite, ddof=1) / math.sqrt(len(finite))
        ),
        "coefficient_l2_norm": float(
            np.mean(
                [item["coefficient_l2"] for item in fold_diagnostics],
                dtype=np.float64,
            )
        ),
        "effective_df": float(
            np.mean(
                [item["effective_df"] for item in fold_diagnostics],
                dtype=np.float64,
            )
        ),
        "input_prediction_variance": float(
            np.mean(
                [item["input_prediction_variance"] for item in fold_diagnostics],
                dtype=np.float64,
            )
        ),
        "total_prediction_variance": float(
            np.mean(
                [item["total_prediction_variance"] for item in fold_diagnostics],
                dtype=np.float64,
            )
        ),
        "fold_diagnostics": fold_diagnostics,
    }


def evaluate_stability_candidates_ordered(
    prepared_folds: list[dict[str, PreparedRepresentation]],
    evaluation_targets: list[np.ndarray],
    candidates: list[StabilityCandidate],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    global _STABILITY_EVALUATION_CONTEXT
    _STABILITY_EVALUATION_CONTEXT = (prepared_folds, evaluation_targets, candidates)
    try:
        return ordered_fork_map(
            _evaluate_candidate,
            [(index,) for index in range(len(candidates))],
            workers=workers,
            label="PRISM_V211_JOINT_STABILITY_JOINT_PREDICTIVE_PATH",
        )
    finally:
        _STABILITY_EVALUATION_CONTEXT = None


def _gate_for_candidate(
    candidate: StabilityCandidate,
    prepared_folds: list[dict[str, PreparedRepresentation]],
    fold_records: list[dict[str, Any]],
    gate_parameters: Mapping[str, float],
) -> dict[str, Any]:
    targets: list[np.ndarray] = []
    totals: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    best_k: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    numerical: list[bool] = []
    for prepared_by_representation, fold in zip(
        prepared_folds, fold_records, strict=True
    ):
        prediction, contract, components = solve_prepared_stability(
            prepared_by_representation[candidate.k_representation],
            route=candidate.route,
            numerical_alpha=candidate.numerical_alpha,
            predictive_eta=candidate.predictive_eta,
        )
        targets.append(fold["evaluation_target"])
        totals.append(prediction)
        inputs.append(components["INPUT"])
        best_k.append(fold["best_k_eval"])
        coefficients.append(
            np.asarray(contract["input_coefficient"], dtype=np.float64)
        )
        numerical.append(numerical_contract_passes(contract))
    return input_path_preservation_gate(
        np.concatenate(targets),
        np.concatenate(totals),
        np.concatenate(best_k),
        input_prediction=np.concatenate(inputs),
        nonintercept_coefficients=np.concatenate(coefficients),
        numerical_certificate_passed=all(numerical),
        **dict(gate_parameters),
    )


def _diagnosis(
    selected_route: str,
    route_best_by_representation: Mapping[str, Mapping[str, StabilityCandidate]],
    gates: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> str:
    candidates = route_best_by_representation[selected_route]
    route_gates = gates[selected_route]
    compressed = candidates[CHANNEL_COMPRESSED]
    full = candidates[FULL_BASIS]
    compressed_pass = bool(route_gates[CHANNEL_COMPRESSED]["selected"]["pass"])
    full_pass = bool(route_gates[FULL_BASIS]["selected"]["pass"])
    compressed_zero_pass = bool(route_gates[CHANNEL_COMPRESSED]["eta_zero"]["pass"])
    full_zero_pass = bool(route_gates[FULL_BASIS]["eta_zero"]["pass"])
    if (
        compressed_pass
        and compressed.predictive_eta > 0
        and not compressed_zero_pass
        and not full_pass
    ):
        return "JOINT_STABILITY_REQUIRES_COMPRESSION_AND_SHRINKAGE"
    if compressed_pass and not full_pass:
        return "JOINT_INSTABILITY_HIGH_DIMENSIONAL_K_BASIS_SUPPORTED"
    if full_pass and full.predictive_eta > 0 and not full_zero_pass:
        return "JOINT_INSTABILITY_PREDICTIVE_SHRINKAGE_DEFICIT_SUPPORTED"
    if not any(
        bool(value[representation]["selected"]["pass"])
        for value in gates.values()
        for representation in K_REPRESENTATIONS
    ):
        return "JOINT_STABILITY_NOT_RESCUED_BY_REGISTERED_STABILITY_CONTROLS"
    return "JOINT_STABILITY_STABILITY_SUPPORTED_WITHOUT_UNIQUE_FACTOR_ATTRIBUTION"


def _stability_summary(losses: Sequence[float]) -> dict[str, float]:
    values = np.asarray(losses, dtype=np.float64)
    median = float(np.median(values))
    worst = float(np.max(values))
    return {
        "mean_fold_loss": float(np.mean(values, dtype=np.float64)),
        "median_fold_loss": median,
        "worst_fold_loss": worst,
        "max_to_median_fold_loss_ratio": worst
        / max(abs(median), np.finfo(np.float64).tiny),
    }


def run_joint_stability_view(
    shared: Path,
    project: Path,
    output: Path,
    legacy_results_root: Path,
    view: Any,
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
        config = load_joint_stability_config(project)
        v211, v21, v2 = load_v211_configs(project, protocol="metro_p60")
        c_path = (
            output
            / "DEVELOPMENT/C"
            / view.head.head_id
            / view.proxy_policy
            / "RESULT.json"
        )
        w_path = (
            output
            / "DEVELOPMENT/W"
            / view.head.head_id
            / view.proxy_policy
            / "RESULT.json"
        )
        a_path = (
            output
            / "DEVELOPMENT/A"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "RESULT.json"
        )
        c_result = json.loads(c_path.read_text(encoding="utf-8"))
        w_result = json.loads(w_path.read_text(encoding="utf-8"))
        a_result = json.loads(a_path.read_text(encoding="utf-8"))
        if any(item.get("status") != "PASS" for item in (c_result, w_result, a_result)):
            raise RuntimeError("v2.1.1 Joint stability practice M2-M4 prerequisite is not PASS")
        if not bool(c_result.get("input_path_preservation", {}).get("pass")):
            raise RuntimeError("v2.1.1 Joint stability practice requires the frozen C input path")
        frozen_channel_set = {
            str(value) for value in c_result["active_channels"]
        }
        active = [
            item
            for item in load_active_channels(output, view)
            if str(item["channel"]) in frozen_channel_set
        ]
        frozen_channels = tuple(str(item["channel"]) for item in active)
        if set(frozen_channels) != frozen_channel_set:
            raise RuntimeError("v2.1.1 Joint stability practice active K support differs from frozen C support")
        w_oof = pd.read_parquet(
            output / w_result["oof_path"],
            columns=["base_origin_id", "view_sample_id", "oof_fold"],
        )
        c_oof = pd.read_parquet(
            output / c_result["oof_prediction_path"],
            columns=["base_origin_id", "view_sample_id", "oof_fold"],
        )
        w_contract = w_result["joint_w_basis_contract"]
        a_profile = tuple(
            a_result["a_contract"].get(
                "profile", realized_state_profiles(view.head)[0]
            )
        )
        development_train = load_native_samples(shared, view, "train")
        registered_input_train = load_native_samples(
            shared, _input_only_view(view), "train"
        )
        fold_count = int(v21["selection"]["inner_folds"])
        fit_cap = int(v2["row_caps"]["joint_predictive_fit"])
        evaluation_cap = int(v2["row_caps"]["validation_selection_per_fold"])
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
        inner_target_accessor = BaseAccessor(
            shared, view.head.dataset, "train", [view.head.target]
        )
        legacy_path = (
            legacy_results_root
            / "DEVELOPMENT/JOINT"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "RESULT.json"
        )
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        legacy_route = str(legacy["final_selected_candidate"])
        legacy_selected = legacy["route_local_selected"][legacy_route]
        _, legacy_alpha, legacy_ratio_k, legacy_ratio_w = legacy_selected
        observed_legacy_losses: list[float] = []
        prepared_folds: list[dict[str, PreparedRepresentation]] = []
        fold_records: list[dict[str, Any]] = []
        protocol_audits: list[dict[str, Any]] = []
        representation_audits: list[dict[str, Any]] = []
        for fold_record, registered_input_fold in zip(
            joint_folds, registered_input_folds, strict=True
        ):
            fold = int(fold_record["fold_index"])
            fit = fold_record["fit"]
            evaluation = fold_record["evaluation"]
            protocol_audit = audit_joint_fold_protocol(
                fold_record,
                registered_input_fold,
                w_oof[w_oof["oof_fold"] == fold].reset_index(drop=True),
                c_oof[c_oof["oof_fold"] == fold].reset_index(drop=True),
            )
            protocol_audits.append(protocol_audit)
            if not protocol_audit["pass"]:
                raise JointFoldProtocolMismatch(
                    f"v2.1.1 Joint stability practice Joint fold {fold} provenance differs from C/W"
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
            k_blocks, representation_audit = k_representation_blocks(
                features, frozen_channels
            )
            representation_audits.append({"fold_index": fold, **representation_audit})
            best_index = features["channels"].index(
                str(c_result["best_active_k_channel"])
            )
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
            delta, history = (int(value) for value in a_profile)
            a_train = inner_target_accessor.target_state(
                fit, view.head.target, delta, history
            )
            a_eval = inner_target_accessor.target_state(
                evaluation, view.head.target, delta, history
            )
            target = fit["y_true"].to_numpy(dtype=np.float64)
            evaluation_target = evaluation["y_true"].to_numpy(dtype=np.float64)
            legacy_prediction, _, _ = fit_joint_candidate(
                {
                    "K": k_blocks[FULL_BASIS][0],
                    "W": w_train,
                    "A": a_train,
                },
                target,
                {
                    "K": k_blocks[FULL_BASIS][1],
                    "W": w_eval,
                    "A": a_eval,
                },
                candidate=legacy_route,
                alpha=float(legacy_alpha),
                k_over_a_ratio=float(legacy_ratio_k),
                w_over_a_ratio=float(legacy_ratio_w),
            )
            observed_legacy_losses.append(mse(evaluation_target, legacy_prediction))
            prepared_by_representation = {}
            for representation in K_REPRESENTATIONS:
                k_train, k_evaluation = k_blocks[representation]
                prepared_by_representation[representation] = prepare_joint_representation(
                    {"K": k_train, "W": w_train, "A": a_train},
                    target,
                    {"K": k_evaluation, "W": w_eval, "A": a_eval},
                    k_representation=representation,
                    raw_k_support=frozen_channels,
                )
            prepared_folds.append(prepared_by_representation)
            fold_records.append(
                {
                    "fold_index": fold,
                    "evaluation_target": evaluation_target,
                    "best_k_eval": features["compressed_evaluation"][:, best_index],
                }
            )
        if len(prepared_folds) != 4:
            raise JointFoldProtocolMismatch("v2.1.1 Joint stability practice Joint requires all four folds")

        expected_legacy_losses = [
            float(value) for value in legacy["final_selected_fold_losses"]
        ]
        legacy_reproduced = bool(
            np.allclose(
                observed_legacy_losses,
                expected_legacy_losses,
                rtol=1e-10,
                atol=1e-12,
            )
        )
        if not legacy_reproduced:
            raise RuntimeError("LEGACY_V212_JOINT_ANCHOR_NOT_REPRODUCED")
        legacy_anchor = {
            "name": "LEGACY_V212_JOINT_ANCHOR",
            "selection_eligible": False,
            "route": legacy_route,
            "k_representation": FULL_BASIS,
            "alpha": float(legacy_alpha),
            "k_over_a_ratio": float(legacy_ratio_k),
            "w_over_a_ratio": float(legacy_ratio_w),
            "expected_fold_losses": expected_legacy_losses,
            "observed_fold_losses": observed_legacy_losses,
            "maximum_absolute_difference": float(
                np.max(
                    np.abs(
                        np.asarray(observed_legacy_losses)
                        - np.asarray(expected_legacy_losses)
                    )
                )
            ),
            "reproduced": True,
        }

        numerical_alpha_grid = sorted(
            {
                0.0,
                *(
                    float(value)
                    for value in v2["J_module"]["ridge_alpha_grid"]
                ),
            }
        )
        numerical_selected: dict[tuple[str, str], float] = {}
        numerical_audits: dict[str, Any] = {}
        for route in JOINT_CANDIDATES:
            for representation in K_REPRESENTATIONS:
                contracts_by_alpha: dict[float, list[dict[str, Any]]] = {}
                for alpha in numerical_alpha_grid:
                    contracts = []
                    for prepared_by_representation in prepared_folds:
                        _, contract, _ = solve_prepared_stability(
                            prepared_by_representation[representation],
                            route=route,
                            numerical_alpha=alpha,
                            predictive_eta=0.0,
                        )
                        contracts.append(contract)
                    contracts_by_alpha[alpha] = contracts
                selected_alpha, audit = select_smallest_numerical_alpha(
                    numerical_alpha_grid,
                    contracts_by_alpha,
                    required_folds=4,
                )
                numerical_selected[(route, representation)] = selected_alpha
                numerical_audits[f"{route}|{representation}"] = audit

        candidates = [
            StabilityCandidate(
                route=route,
                k_representation=representation,
                numerical_alpha=numerical_selected[(route, representation)],
                predictive_eta=float(eta),
            )
            for route in JOINT_CANDIDATES
            for representation in K_REPRESENTATIONS
            for eta in ETA_PRED_GRID
        ]
        workers = int(
            os.environ.get(
                "PRISM_V211_JOINT_STABILITY_J_INNER_WORKERS", config["inner_candidate_workers"]
            )
        )
        evaluations = evaluate_stability_candidates_ordered(
            prepared_folds,
            [fold["evaluation_target"] for fold in fold_records],
            candidates,
            workers=workers,
        )
        evaluation_by_candidate = {
            candidate: evaluation
            for candidate, evaluation in zip(candidates, evaluations, strict=True)
        }
        eta_selected: dict[tuple[str, str], StabilityCandidate] = {}
        eta_selections: dict[str, Any] = {}
        regularization_path: dict[str, list[dict[str, Any]]] = {}
        for route in JOINT_CANDIDATES:
            for representation in K_REPRESENTATIONS:
                subset = [
                    candidate
                    for candidate in candidates
                    if candidate.route == route
                    and candidate.k_representation == representation
                ]
                eta_losses = {
                    candidate.predictive_eta: evaluation_by_candidate[candidate][
                        "fold_losses"
                    ]
                    for candidate in subset
                }
                selected_eta, selection = select_predictive_eta(
                    eta_losses, minimum_usable_folds=4
                )
                selected_candidate = next(
                    candidate
                    for candidate in subset
                    if candidate.predictive_eta == selected_eta
                )
                eta_selected[(route, representation)] = selected_candidate
                key = f"{route}|{representation}"
                eta_selections[key] = selection
                regularization_path[key] = [
                    {
                        "eta": candidate.predictive_eta,
                        **{
                            name: value
                            for name, value in evaluation_by_candidate[candidate].items()
                            if name not in {"candidate", "candidate_key"}
                        },
                    }
                    for candidate in subset
                ]

        minimum_relative = float(
            v21["selection"]["minimum_relative_improvement"]["J"]
        )
        minimum_positive = float(
            v21["selection"]["minimum_positive_fold_fraction"]
        )
        route_best_by_representation: dict[str, dict[str, StabilityCandidate]] = {}
        representation_comparison: dict[str, Any] = {}
        route_best: dict[str, StabilityCandidate] = {}
        for route in JOINT_CANDIDATES:
            compressed = eta_selected[(route, CHANNEL_COMPRESSED)]
            full = eta_selected[(route, FULL_BASIS)]
            compressed_losses = evaluation_by_candidate[compressed]["fold_losses"]
            full_losses = evaluation_by_candidate[full]["fold_losses"]
            selected_representation, selection = select_k_representation(
                compressed_losses,
                full_losses,
                minimum_relative_improvement=minimum_relative,
                minimum_positive_fraction=minimum_positive,
                minimum_usable_folds=4,
            )
            route_best_by_representation[route] = {
                CHANNEL_COMPRESSED: compressed,
                FULL_BASIS: full,
            }
            route_best[route] = route_best_by_representation[route][
                selected_representation
            ]
            activation = practical_activation(
                compressed_losses,
                full_losses,
                minimum_relative_improvement=minimum_relative,
                minimum_positive_fraction=minimum_positive,
            )
            representation_comparison[route] = {
                "compressed_selected_eta": compressed.predictive_eta,
                "compressed_fold_losses": compressed_losses,
                "full_selected_eta": full.predictive_eta,
                "full_fold_losses": full_losses,
                "selected_representation": selected_representation,
                "full_vs_compressed_activation": activation,
                "selection": selection,
            }

        neutral = route_best[J_K]
        route_selection = guarded_local_one_se_select(
            {
                candidate: evaluation_by_candidate[candidate]["fold_losses"]
                for candidate in route_best.values()
            },
            lambda candidate: (
                JOINT_CANDIDATES.index(candidate.route),
                0 if candidate.k_representation == CHANNEL_COMPRESSED else 1,
                -candidate.predictive_eta,
            ),
            neutral=neutral,
            minimum_relative_improvement=minimum_relative,
            minimum_positive_fraction=minimum_positive,
            minimum_usable_folds=4,
        )
        selected = route_selection.final_selected_candidate
        gate_parameters = _gate_config(v211)
        gates: dict[str, dict[str, dict[str, Any]]] = {}
        for route in JOINT_CANDIDATES:
            gates[route] = {}
            for representation in K_REPRESENTATIONS:
                candidate = route_best_by_representation[route][representation]
                eta_zero = StabilityCandidate(
                    route=route,
                    k_representation=representation,
                    numerical_alpha=candidate.numerical_alpha,
                    predictive_eta=0.0,
                )
                gates[route][representation] = {
                    "selected": _gate_for_candidate(
                        candidate, prepared_folds, fold_records, gate_parameters
                    ),
                    "eta_zero": _gate_for_candidate(
                        eta_zero, prepared_folds, fold_records, gate_parameters
                    ),
                }
        oof_gate = gates[selected.route][selected.k_representation]["selected"]

        assembly_development_train = apply_assembly_support(
            development_train, active
        )
        train = _cap(
            assembly_development_train,
            int(v2["row_caps"]["joint_predictive_fit"]),
        )
        validation = apply_assembly_support(
            load_native_samples(shared, view, "validation"), active
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
        final_k_blocks, final_representation_audit = k_representation_blocks(
            final_features, frozen_channels
        )
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
        delta, history = (int(value) for value in a_profile)
        final_accessor = BaseAccessor(
            shared, view.head.dataset, "validation", [view.head.target]
        )
        a_train = final_accessor.target_state(train, view.head.target, delta, history)
        a_validation = final_accessor.target_state(
            validation, view.head.target, delta, history
        )
        final_prepared: dict[str, PreparedRepresentation] = {}
        for representation in K_REPRESENTATIONS:
            k_train, k_validation = final_k_blocks[representation]
            final_prepared[representation] = prepare_joint_representation(
                {"K": k_train, "W": w_train, "A": a_train},
                train["y_true"].to_numpy(dtype=np.float64),
                {"K": k_validation, "W": w_validation, "A": a_validation},
                k_representation=representation,
                raw_k_support=frozen_channels,
            )
        prediction, contract, components = solve_prepared_stability(
            final_prepared[selected.k_representation],
            route=selected.route,
            numerical_alpha=selected.numerical_alpha,
            predictive_eta=selected.predictive_eta,
        )
        _, bare_contract, _ = solve_prepared_stability(
            final_prepared[selected.k_representation],
            route=selected.route,
            numerical_alpha=selected.numerical_alpha,
            predictive_eta=0.0,
        )
        contract["bare_numerical_alpha_certificate"] = bare_contract[
            "numerical_certificate"
        ]
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
        formal_pass = bool(
            oof_gate["pass"]
            and numerical_contract_passes(contract)
            and numerical_contract_passes(bare_contract)
        )
        formal_gate = {
            **oof_gate,
            "pass": formal_pass,
            "status": "INPUT_PATH_PRESERVED" if formal_pass else "INPUT_PATH_COLLAPSED",
            "final_refit_numerical_certificate_passed": numerical_contract_passes(
                contract
            ),
            "bare_numerical_alpha_certificate_passed": numerical_contract_passes(
                bare_contract
            ),
        }
        gate = attach_nonselecting_validation_confirmation(
            formal_gate, validation_gate
        )

        route_materializations: dict[str, Any] = {}
        for route in JOINT_CANDIDATES:
            route_candidate = route_best[route]
            route_prediction, route_contract, route_components = solve_prepared_stability(
                final_prepared[route_candidate.k_representation],
                route=route,
                numerical_alpha=route_candidate.numerical_alpha,
                predictive_eta=route_candidate.predictive_eta,
            )
            if route in {J_KW, J_KWA} and int(
                route_contract["blocks"].get("W", {}).get("columns", 0)
            ) <= 0:
                raise RuntimeError("v2.1.1 Joint stability practice Joint W route lacks jointly fitted W columns")
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
            route_frame["model"] = f"PRISM_V2_1_1_{route}"
            route_frame["dtype"] = "float64"
            route_path = destination / f"validation_{route}.parquet"
            route_frame.to_parquet(route_path, index=False, compression="zstd")
            route_id = stability_candidate_id(
                view.relative_root.as_posix(), route_candidate
            )
            route_materializations[route] = {
                "candidate_id": route_id,
                "selected_hyperparameters": route_candidate.descriptor(),
                "prediction_path": str(route_path.relative_to(output)),
                "prediction_sha256": sha256_file(route_path),
                "prediction_loss": mse(
                    route_frame["y_true"].to_numpy(dtype=np.float64),
                    route_prediction,
                ),
                "contract": route_contract,
                "block_dimensions": {
                    block: int(value["columns"])
                    for block, value in route_contract["blocks"].items()
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
        frame["model"] = f"PRISM_V2_1_1_{selected.route}"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")
        final_loss = mse(frame["y_true"].to_numpy(dtype=np.float64), prediction)
        selected_losses = evaluation_by_candidate[selected]["fold_losses"]
        legacy_improvement = practical_activation(
            expected_legacy_losses,
            selected_losses,
            minimum_relative_improvement=minimum_relative,
            minimum_positive_fraction=minimum_positive,
        )
        decision_label = (
            "JOINT_STABILITY_PREDICTIVE_STABILITY_SUPPORTED"
            if gate["pass"]
            else "JOINT_STABILITY_STABILITY_IMPROVED_BUT_NOT_SUPPORTED"
            if legacy_improvement["pass"]
            else "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT"
        )
        diagnosis = _diagnosis(
            selected.route, route_best_by_representation, gates
        )
        view_key = view.relative_root.as_posix()
        selected_id = stability_candidate_id(view_key, selected)
        candidate_registry = [
            {
                "candidate_id": stability_candidate_id(view_key, candidate),
                **candidate.descriptor(),
                "fold_losses": evaluation_by_candidate[candidate]["fold_losses"],
            }
            for candidate in candidates
        ]
        result = {
            "status": "PASS" if gate["pass"] else decision_label,
            "development_decision": decision_label,
            "development_diagnosis": diagnosis,
            "diagnosis_scope": "DEVELOPMENT_MODEL_SELECTION_NOT_CAUSAL_PROOF",
            "stage": "M5_JOINT_STABILITY_PREDICTIVE_STABILITY",
            "estimator_version": MODEL_VERSION,
            "model_version": MODEL_VERSION,
            "practice_revision": PRACTICE_REVISION,
            "joint_estimator_semantics": JOINT_ESTIMATOR_SEMANTICS,
            "dataset": view.head.dataset,
            "target_head": view.head.head_id,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "registered_candidates": list(JOINT_CANDIDATES),
            "registered_k_representations": list(K_REPRESENTATIONS),
            "predictive_eta_grid": list(ETA_PRED_GRID),
            "numerical_alpha_grid": numerical_alpha_grid,
            "predictive_block_ratio": 1.0,
            "ridge_semantics": "NUMERICAL_ALPHA_PLUS_EXPLICIT_PREDICTIVE_ETA",
            "selected_candidate": selected.route,
            "selected_k_representation": selected.k_representation,
            "selected_predictive_eta": selected.predictive_eta,
            "selected_numerical_alpha": selected.numerical_alpha,
            "selection": stability_guarded_selection_json(route_selection),
            "route_local_selected": {
                route: {
                    **candidate.descriptor(),
                    "candidate_id": stability_candidate_id(view_key, candidate),
                }
                for route, candidate in route_best.items()
            },
            "route_materializations": route_materializations,
            "representation_comparison": representation_comparison,
            "predictive_regularization_path": regularization_path,
            "predictive_eta_selections": eta_selections,
            "minimal_stabilizing_numerical_alpha_audits": numerical_audits,
            "ar_profile": list(a_profile),
            "legacy_v212_joint_anchor": legacy_anchor,
            "legacy_anchor_reproduced": legacy_reproduced,
            "legacy_vs_stability_selected_activation": legacy_improvement,
            "candidate_fold_losses": {
                candidate.key(): evaluation_by_candidate[candidate]["fold_losses"]
                for candidate in candidates
            },
            "candidate_registry": candidate_registry,
            "candidate_id_binding": {
                "status": "PASS",
                "candidate_identity_fields": [
                    "route",
                    "k_representation",
                    "numerical_alpha",
                    "predictive_eta",
                ],
                "prediction_loss_recomputed": final_loss,
                "prediction_path": str(prediction_path.relative_to(output)),
                "prediction_sha256": sha256_file(prediction_path),
            },
            "final_selected_candidate": selected.route,
            "final_selected_candidate_id": selected_id,
            "selected_loss_candidate_id": selected_id,
            "selected_prediction_candidate_id": selected_id,
            "selected_contract_candidate_id": selected_id,
            "final_selected_fold_losses": selected_losses,
            "final_selected_prediction_path": str(prediction_path.relative_to(output)),
            "final_selected_contract": contract,
            "final_prediction_loss": final_loss,
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "joint_contract": contract,
            "input_path_preservation": gate,
            "input_path_gate": gate,
            "input_path_failure_class": gate.get(
                "input_path_failure_class", "INPUT_PATH_PRESERVED"
            ),
            "joint_w_basis_contract": w_contract,
            "joint_w_coefficients_jointly_fitted": True,
            "joint_w_prefit_scalar_forbidden": True,
            "joint_fold_protocol_audit": protocol_audits,
            "joint_fold_protocol_audit_pass": all(
                item["pass"] for item in protocol_audits
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
            "joint_k_representations_share_rows": True,
            "joint_fit_source": "ORIGINAL_REGISTERED_ANCHOR_INNER_TRAIN_SUPPORT_AFTER_ASSEMBLY_MASK",
            "joint_evaluation_source": "ORIGINAL_REGISTERED_ANCHOR_INNER_VALIDATION_SUPPORT_AFTER_ASSEMBLY_MASK",
            "nested_oof_training_used": False,
            "w_physical_oof_used_as_training_pool": False,
            "registered_inner_fold_count": 4,
            "candidate_fold_loss_count": 4,
            "k_representation_audit_by_fold": representation_audits,
            "final_k_representation_audit": final_representation_audit,
            "best_active_k_fold_losses": [
                mse(fold["evaluation_target"], fold["best_k_eval"])
                for fold in fold_records
            ],
            "selected_stability": _stability_summary(selected_losses),
            "legacy_stability": _stability_summary(expected_legacy_losses),
            "validation_mse": final_loss,
            "test_accessed": False,
            "ood_accessed": False,
            "inner_candidate_workers": workers,
            "elapsed_seconds": time.time() - started,
            **regression_metrics(
                frame["y_true"].to_numpy(dtype=np.float64), prediction
            ),
        }
    except JointFoldProtocolMismatch as error:
        result = {
            "status": "STOP_JOINT_FOLD_PROTOCOL_MISMATCH",
            "stage": "M5_JOINT_STABILITY_PREDICTIVE_STABILITY",
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
    except Exception as error:
        result = {
            "status": "STOP_JOINT_STABILITY_IMPLEMENTATION_OR_NUMERICAL_FAILURE",
            "stage": "M5_JOINT_STABILITY_PREDICTIVE_STABILITY",
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
