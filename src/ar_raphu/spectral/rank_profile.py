"""Effective-rank profiles under pre-registered spectral error budgets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RankProfile:
    singular_values: np.ndarray
    normalized_energy: np.ndarray
    cumulative_energy: np.ndarray
    tail_curve: np.ndarray
    tail_beyond_rank_max: float
    effective_ranks: dict[float, int]


def effective_rank(tail_curve: np.ndarray, budget: float) -> int:
    """Return the first rank whose tail is <= the budget."""

    tail = np.asarray(tail_curve, dtype=np.float64)
    eligible = np.flatnonzero(tail <= float(budget))
    return int(eligible[0] + 1) if eligible.size else int(len(tail) + 1)


def build_rank_profile(
    singular_values: np.ndarray,
    *,
    rank_max: int,
    budgets: tuple[float, ...] = (0.10, 0.05, 0.02),
) -> RankProfile:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or rank_max <= 0:
        raise ValueError("Need a non-empty spectrum and positive rank_max.")
    energy = values**2
    total = max(float(energy.sum()), np.finfo(np.float64).eps)
    normalized = energy / total
    ranks = np.arange(1, rank_max + 1)
    cumulative = np.array(
        [normalized[: min(rank, len(values))].sum() for rank in ranks]
    )
    tail = np.array(
        [
            np.sqrt(normalized[min(rank, len(values)) :].sum())
            for rank in ranks
        ]
    )
    padded_values = np.pad(values[:rank_max], (0, max(0, rank_max - len(values))))
    padded_energy = np.pad(
        normalized[:rank_max], (0, max(0, rank_max - len(normalized)))
    )
    return RankProfile(
        singular_values=padded_values,
        normalized_energy=padded_energy,
        cumulative_energy=cumulative,
        tail_curve=tail,
        tail_beyond_rank_max=float(tail[-1]),
        effective_ranks={
            float(budget): effective_rank(tail, float(budget))
            for budget in budgets
        },
    )


def classify_truth_profile(profile: RankProfile) -> str:
    primary = profile.effective_ranks[0.05]
    fine = profile.effective_ranks[0.02]
    if primary == 1 and fine == 1:
        return "near_rank1"
    if primary == 1 and fine >= 2:
        return "weak_rank2"
    if primary == 2:
        return "strong_rank2"
    return "higher_rank"
