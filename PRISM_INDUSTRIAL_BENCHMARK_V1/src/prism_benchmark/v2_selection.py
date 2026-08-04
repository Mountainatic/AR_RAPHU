from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Hashable, Mapping

import numpy as np


@dataclass(frozen=True)
class OneSESelection:
    selected: Any
    best: Any
    best_mean: float
    best_se: float
    threshold: float
    acceptable: tuple[Any, ...]
    usable_fold_count: dict[Any, int]
    means: dict[Any, float]
    standard_errors: dict[Any, float]

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("usable_fold_count", "means", "standard_errors"):
            result[key] = {str(candidate): value for candidate, value in result[key].items()}
        result["selected"] = str(self.selected)
        result["best"] = str(self.best)
        result["acceptable"] = [str(value) for value in self.acceptable]
        return result


def one_se_select(
    fold_losses: Mapping[Hashable, list[float]],
    complexity_key: Callable[[Hashable], tuple[Any, ...]],
    *,
    neutral: Hashable | None = None,
    minimum_usable_folds: int = 3,
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> OneSESelection:
    """Frozen V2 one-SE rule with finite-fold filtering and neutral preference."""
    usable: dict[Hashable, np.ndarray] = {}
    for candidate, values in fold_losses.items():
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if len(array) >= minimum_usable_folds:
            usable[candidate] = array
    if not usable:
        raise ValueError("no candidate has the minimum number of finite folds")
    means = {candidate: float(np.mean(values, dtype=np.float64)) for candidate, values in usable.items()}
    standard_errors = {
        candidate: float(np.std(values, ddof=1) / math.sqrt(len(values)))
        for candidate, values in usable.items()
    }
    best = min(usable, key=lambda candidate: (means[candidate], complexity_key(candidate)))
    threshold = means[best] + standard_errors[best]
    acceptable = tuple(
        candidate
        for candidate in usable
        if means[candidate] < threshold
        or np.isclose(means[candidate], threshold, rtol=rtol, atol=atol)
    )
    if neutral is not None and neutral in acceptable:
        selected = neutral
    else:
        selected = min(acceptable, key=complexity_key)
    return OneSESelection(
        selected=selected,
        best=best,
        best_mean=means[best],
        best_se=standard_errors[best],
        threshold=threshold,
        acceptable=acceptable,
        usable_fold_count={candidate: len(values) for candidate, values in usable.items()},
        means=means,
        standard_errors=standard_errors,
    )


def practical_activation(
    neutral_losses: list[float],
    candidate_losses: list[float],
    *,
    minimum_relative_improvement: float = 0.01,
    minimum_positive_fraction: float = 0.75,
    denominator_floor_scale: float = 1e-12,
) -> dict[str, Any]:
    neutral = np.asarray(neutral_losses, dtype=np.float64)
    candidate = np.asarray(candidate_losses, dtype=np.float64)
    mask = np.isfinite(neutral) & np.isfinite(candidate)
    if int(mask.sum()) < 3:
        return {"pass": False, "reason": "INSUFFICIENT_FINITE_FOLDS", "finite_folds": int(mask.sum())}
    neutral = neutral[mask]
    candidate = candidate[mask]
    floor = denominator_floor_scale * max(1.0, float(np.mean(np.abs(neutral), dtype=np.float64)))
    relative = (neutral - candidate) / np.maximum(np.abs(neutral), floor)
    mean_improvement = (float(np.mean(neutral)) - float(np.mean(candidate))) / max(float(np.mean(neutral)), floor)
    positive_fraction = float(np.mean(relative > 0.0))
    return {
        "pass": bool(mean_improvement >= minimum_relative_improvement and positive_fraction >= minimum_positive_fraction),
        "finite_folds": int(mask.sum()),
        "mean_relative_improvement": mean_improvement,
        "positive_fold_fraction": positive_fraction,
        "fold_relative_improvements": relative.tolist(),
    }

