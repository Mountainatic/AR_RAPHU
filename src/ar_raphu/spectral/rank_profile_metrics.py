"""Distances between truth and estimated effective-rank profiles."""

from __future__ import annotations

import numpy as np

from .rank_profile import RankProfile


def tail_curve_max_abs_error(
    truth: RankProfile, estimate: RankProfile
) -> float:
    return float(np.max(np.abs(truth.tail_curve - estimate.tail_curve)))


def normalized_spectrum_l1_distance(
    truth: RankProfile, estimate: RankProfile
) -> float:
    truth_vector = np.append(
        truth.normalized_energy,
        truth.tail_beyond_rank_max**2,
    )
    estimate_vector = np.append(
        estimate.normalized_energy,
        estimate.tail_beyond_rank_max**2,
    )
    return float(np.sum(np.abs(truth_vector - estimate_vector)))
