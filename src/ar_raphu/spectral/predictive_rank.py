"""Distribution-specific predictive effective-rank utilities."""

from __future__ import annotations

import numpy as np


def predictive_effective_rank(
    normalized_excess_curve: np.ndarray,
    budget: float,
) -> int:
    curve = np.asarray(normalized_excess_curve, dtype=np.float64)
    eligible = np.flatnonzero(curve <= float(budget))
    return int(eligible[0] + 1) if eligible.size else int(len(curve) + 1)


def predictive_rank_profile(
    normalized_excess_curve: np.ndarray,
    budgets: tuple[float, ...] = (0.10, 0.05, 0.02),
) -> dict[float, int]:
    return {
        float(budget): predictive_effective_rank(
            normalized_excess_curve, float(budget)
        )
        for budget in budgets
    }
