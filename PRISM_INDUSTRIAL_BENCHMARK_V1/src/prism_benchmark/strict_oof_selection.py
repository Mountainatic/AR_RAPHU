"""Strict stage admission with zero outside the hyperparameter search space.

This module implements the post-v2.1.1 routing rule.  A stage first tunes only
over its non-zero family.  Its identity increment is then compared with a
cross-fitted non-zero child on outer OOF folds.  Statistical uncertainty is
reporting-only and cannot alter the routing decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


ACTIVE = "ACTIVE"
ZERO_IDENTITY = "ZERO_IDENTITY"
SELECTION_RULE = "STRICT_NESTED_OOF_EMPIRICAL_RISK"


def numerical_epsilon(
    parent_risk: float,
    child_risk: float,
    *,
    multiplier: float = 1000.0,
) -> float:
    """Return a FP64 comparison tolerance with no statistical semantics."""
    if not np.isfinite(parent_risk) or not np.isfinite(child_risk):
        raise ValueError("stage risks must be finite")
    if multiplier <= 0.0 or not np.isfinite(multiplier):
        raise ValueError("epsilon multiplier must be finite and positive")
    return float(
        multiplier
        * np.finfo(np.float64).eps
        * max(1.0, abs(float(parent_risk)), abs(float(child_risk)))
    )


def _loss_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} fold losses must be one-dimensional")
    return result


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("inf")
    return float(np.average(values[mask], weights=weights[mask]))


def tune_best_nonzero(
    candidate_fold_losses: Mapping[Hashable, Sequence[float]],
    *,
    fold_weights: Sequence[float] | None = None,
    minimum_usable_folds: int = 2,
) -> Hashable:
    """Tune by empirical risk inside H+ only.

    Mapping insertion order is the deterministic tie-break for numerically
    identical risks.  It is a registry-order tie-break, never a complexity
    preference.
    """
    if not candidate_fold_losses:
        raise ValueError("the non-zero candidate family is empty")
    vectors = {
        candidate: _loss_vector(values, name=str(candidate))
        for candidate, values in candidate_fold_losses.items()
    }
    lengths = {len(values) for values in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("non-zero candidate fold losses have different lengths")
    count = lengths.pop()
    weights = (
        np.ones(count, dtype=np.float64)
        if fold_weights is None
        else _loss_vector(fold_weights, name="weight")
    )
    if len(weights) != count or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("fold weights must be finite, positive, and aligned")
    eligible = [
        candidate
        for candidate, values in vectors.items()
        if int(np.count_nonzero(np.isfinite(values))) >= minimum_usable_folds
    ]
    if not eligible:
        raise ValueError("no non-zero candidate has enough usable folds")
    return min(
        eligible,
        key=lambda candidate: _weighted_mean(vectors[candidate], weights),
    )


@dataclass(frozen=True)
class StrictOOFSelection:
    identity: Any
    routing_status: str
    tuned_nonzero_candidate: Any
    final_selected_candidate: Any
    outer_selected_nonzero_candidates: tuple[Any, ...]
    outer_fold_indices: tuple[int, ...]
    parent_outer_fold_losses: tuple[float, ...]
    child_outer_fold_losses: tuple[float, ...]
    outer_fold_weights: tuple[float, ...]
    parent_oof_risk: float
    child_oof_risk: float
    incremental_gain: float
    epsilon_num: float
    final_selected_fold_losses: tuple[float, ...]
    selection_rule: str = SELECTION_RULE
    zero_is_hyperparameter_candidate: bool = False
    uncertainty_has_selection_authority: bool = False
    evidence: Mapping[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self.routing_status == ACTIVE

    def with_reporting_evidence(self, evidence: Mapping[str, Any]) -> "StrictOOFSelection":
        """Attach evidence without recomputing or mutating the route."""
        forbidden = {"routing_status", "final_selected_candidate", "active"}
        overlap = forbidden.intersection(evidence)
        if overlap:
            raise ValueError(
                "reporting evidence cannot contain routing fields: "
                f"{sorted(overlap)}"
            )
        return replace(self, evidence=dict(evidence))

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "tuned_nonzero_candidate",
            "final_selected_candidate",
        ):
            result[name] = str(result[name])
        result["outer_selected_nonzero_candidates"] = [
            str(value) for value in self.outer_selected_nonzero_candidates
        ]
        result["active"] = self.active
        result["evidence_role"] = "REPORTING_ONLY"
        return result


def strict_nested_oof_select(
    candidate_fold_losses: Mapping[Hashable, Sequence[float]],
    parent_fold_losses: Sequence[float],
    *,
    identity: Hashable,
    fold_weights: Sequence[float] | None = None,
    minimum_inner_folds: int = 2,
    minimum_outer_folds: int = 3,
    epsilon_multiplier: float = 1000.0,
) -> StrictOOFSelection:
    """Tune H+ in leave-one-outer-fold-out fashion, then compare with identity.

    The parent/identity losses are deliberately a separate argument.  They are
    never inserted into ``candidate_fold_losses`` and cannot participate in
    hyperparameter tuning.  For each outer fold, the non-zero winner is chosen
    using every other fold; only its loss on the held-out fold is scored.
    """
    if identity in candidate_fold_losses:
        raise ValueError("zero/identity must not be a hyperparameter candidate")
    parent = _loss_vector(parent_fold_losses, name="parent")
    if len(parent) < minimum_outer_folds:
        raise ValueError("too few outer folds for strict nested OOF admission")
    vectors = {
        candidate: _loss_vector(values, name=str(candidate))
        for candidate, values in candidate_fold_losses.items()
    }
    if not vectors:
        raise ValueError("the non-zero candidate family is empty")
    if any(len(values) != len(parent) for values in vectors.values()):
        raise ValueError("candidate and parent fold losses must be paired")
    weights = (
        np.ones(len(parent), dtype=np.float64)
        if fold_weights is None
        else _loss_vector(fold_weights, name="weight")
    )
    if (
        len(weights) != len(parent)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("fold weights must be finite, positive, and aligned")

    selected_candidates: list[Hashable] = []
    outer_indices: list[int] = []
    parent_losses: list[float] = []
    child_losses: list[float] = []
    outer_weights: list[float] = []
    all_indices = np.arange(len(parent))
    for outer in range(len(parent)):
        inner = all_indices != outer
        inner_candidates = {
            candidate: values[inner]
            for candidate, values in vectors.items()
            if np.isfinite(values[outer])
        }
        if not np.isfinite(parent[outer]) or not inner_candidates:
            continue
        try:
            winner = tune_best_nonzero(
                inner_candidates,
                fold_weights=weights[inner],
                minimum_usable_folds=minimum_inner_folds,
            )
        except ValueError:
            continue
        selected_candidates.append(winner)
        outer_indices.append(outer)
        parent_losses.append(float(parent[outer]))
        child_losses.append(float(vectors[winner][outer]))
        outer_weights.append(float(weights[outer]))
    if len(outer_indices) < minimum_outer_folds:
        raise ValueError("non-zero family lacks complete enough nested OOF evidence")

    parent_array = np.asarray(parent_losses, dtype=np.float64)
    child_array = np.asarray(child_losses, dtype=np.float64)
    weight_array = np.asarray(outer_weights, dtype=np.float64)
    parent_risk = _weighted_mean(parent_array, weight_array)
    child_risk = _weighted_mean(child_array, weight_array)
    gain = float(parent_risk - child_risk)
    epsilon = numerical_epsilon(
        parent_risk, child_risk, multiplier=epsilon_multiplier
    )
    active = gain > epsilon
    tuned = tune_best_nonzero(
        vectors,
        fold_weights=weights,
        minimum_usable_folds=minimum_outer_folds,
    )
    final = tuned if active else identity
    final_losses = child_array if active else parent_array
    return StrictOOFSelection(
        identity=identity,
        routing_status=ACTIVE if active else ZERO_IDENTITY,
        tuned_nonzero_candidate=tuned,
        final_selected_candidate=final,
        outer_selected_nonzero_candidates=tuple(selected_candidates),
        outer_fold_indices=tuple(outer_indices),
        parent_outer_fold_losses=tuple(parent_losses),
        child_outer_fold_losses=tuple(child_losses),
        outer_fold_weights=tuple(outer_weights),
        parent_oof_risk=parent_risk,
        child_oof_risk=child_risk,
        incremental_gain=gain,
        epsilon_num=epsilon,
        final_selected_fold_losses=tuple(float(value) for value in final_losses),
    )


def apply_stage_increment(
    parent_prediction: Sequence[float],
    increment: Sequence[float],
    *,
    active: bool,
) -> np.ndarray:
    """Apply an increment; ZERO is an identity, never a downstream stop flag."""
    parent = np.asarray(parent_prediction, dtype=np.float64)
    delta = np.asarray(increment, dtype=np.float64)
    if parent.shape != delta.shape:
        raise ValueError("parent prediction and increment shapes differ")
    return parent + delta if active else parent.copy()
