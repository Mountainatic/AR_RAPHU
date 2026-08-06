from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Hashable, Mapping

import numpy as np

from .v2_selection import practical_activation


@dataclass(frozen=True)
class GuardedLocalSelection:
    best_candidate: Any
    best_mean: float
    best_standard_error: float
    acceptable_threshold: float
    acceptable_candidates: tuple[Any, ...]
    passing_active_candidates: tuple[Any, ...]
    final_selected_candidate: Any
    final_selected_fold_losses: tuple[float, ...]
    activation_audit: dict[str, dict[str, Any]]
    usable_fold_count: dict[Any, int]
    means: dict[Any, float]
    standard_errors: dict[Any, float]

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        for field in (
            "best_candidate",
            "final_selected_candidate",
        ):
            value[field] = str(value[field])
        for field in ("acceptable_candidates", "passing_active_candidates"):
            value[field] = [str(candidate) for candidate in value[field]]
        for field in ("usable_fold_count", "means", "standard_errors"):
            value[field] = {str(candidate): item for candidate, item in value[field].items()}
        value["activation_audit"] = {
            str(candidate): audit for candidate, audit in value["activation_audit"].items()
        }
        value["final_selected_fold_losses"] = list(value["final_selected_fold_losses"])
        return value


def guarded_local_one_se_select(
    fold_losses: Mapping[Hashable, list[float]],
    complexity_key: Callable[[Hashable], tuple[Any, ...]],
    *,
    neutral: Hashable,
    minimum_relative_improvement: float = 0.01,
    minimum_positive_fraction: float = 0.75,
    minimum_usable_folds: int = 3,
    denominator_floor_scale: float = 1e-12,
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> GuardedLocalSelection:
    """Module-local one-SE followed by the registered practical/stability guard.

    Unlike the v2.0 selector, an acceptable neutral candidate does not
    automatically suppress a non-neutral candidate that passes both guards.
    """
    if neutral not in fold_losses:
        raise KeyError(f"neutral candidate is absent: {neutral!r}")
    usable: dict[Hashable, np.ndarray] = {}
    for candidate, values in fold_losses.items():
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if len(finite) >= minimum_usable_folds:
            usable[candidate] = finite
    if neutral not in usable:
        raise ValueError("neutral candidate lacks the minimum usable folds")
    if not usable:
        raise ValueError("no candidate has the minimum usable folds")
    means = {candidate: float(np.mean(values, dtype=np.float64)) for candidate, values in usable.items()}
    errors = {
        candidate: float(np.std(values, ddof=1) / math.sqrt(len(values)))
        for candidate, values in usable.items()
    }
    best = min(usable, key=lambda candidate: (means[candidate], complexity_key(candidate)))
    threshold = means[best] + errors[best]
    acceptable = tuple(
        candidate
        for candidate in usable
        if means[candidate] < threshold
        or np.isclose(means[candidate], threshold, rtol=rtol, atol=atol)
    )
    audits: dict[Hashable, dict[str, Any]] = {}
    passing = []
    for candidate in acceptable:
        if candidate == neutral:
            continue
        audit = practical_activation(
            list(fold_losses[neutral]),
            list(fold_losses[candidate]),
            minimum_relative_improvement=minimum_relative_improvement,
            minimum_positive_fraction=minimum_positive_fraction,
            denominator_floor_scale=denominator_floor_scale,
        )
        audits[candidate] = audit
        if audit["pass"]:
            passing.append(candidate)
    selected = min(passing, key=complexity_key) if passing else neutral
    return GuardedLocalSelection(
        best_candidate=best,
        best_mean=means[best],
        best_standard_error=errors[best],
        acceptable_threshold=threshold,
        acceptable_candidates=acceptable,
        passing_active_candidates=tuple(passing),
        final_selected_candidate=selected,
        final_selected_fold_losses=tuple(float(value) for value in fold_losses[selected]),
        activation_audit=audits,
        usable_fold_count={candidate: len(values) for candidate, values in usable.items()},
        means=means,
        standard_errors=errors,
    )


def assert_final_prediction_contract(
    result: Mapping[str, Any],
    *,
    recomputed_loss: float | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> None:
    required = {
        "final_selected_candidate",
        "final_selected_fold_losses",
        "final_selected_prediction_path",
        "final_selected_contract",
    }
    missing = required - set(result)
    if missing:
        raise RuntimeError(f"incomplete final selection contract: {sorted(missing)}")
    if not str(result["final_selected_prediction_path"]):
        raise RuntimeError("final selected prediction path is empty")
    if not isinstance(result["final_selected_contract"], Mapping):
        raise RuntimeError("final selected contract is not a mapping")
    losses = np.asarray(result["final_selected_fold_losses"], dtype=np.float64)
    if losses.ndim != 1 or len(losses) < 3 or not np.isfinite(losses).all():
        raise RuntimeError("final selected fold losses are invalid")
    if recomputed_loss is not None:
        stored = result.get("final_prediction_loss")
        if stored is None or not np.isclose(float(stored), recomputed_loss, rtol=rtol, atol=atol):
            raise RuntimeError("final prediction and stored loss disagree")
