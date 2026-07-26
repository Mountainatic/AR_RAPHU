import numpy as np

from ar_raphu.spectral.rank_profile import (
    build_rank_profile,
    classify_truth_profile,
)


def test_tail_is_monotone_and_budget_rank_uses_less_equal():
    profile = build_rank_profile(
        np.array([1.0, 0.1, 0.0]), rank_max=3, budgets=(0.10, 0.05, 0.02)
    )
    assert np.all(np.diff(profile.tail_curve) <= 0)
    exact_budget = float(profile.tail_curve[0])
    exact = build_rank_profile(
        np.array([1.0, 0.1, 0.0]),
        rank_max=3,
        budgets=(exact_budget,),
    )
    assert exact.effective_ranks[exact_budget] == 1


def test_effective_rank_is_monotone_as_budget_tightens():
    profile = build_rank_profile(
        np.array([1.0, 0.2, 0.06, 0.01]),
        rank_max=4,
    )
    ranks = [profile.effective_ranks[value] for value in (0.10, 0.05, 0.02)]
    assert ranks == sorted(ranks)


def test_rank_classes_are_profile_driven_and_high_rank_is_not_rank2():
    rank1 = build_rank_profile(np.array([1.0, 0.0, 0.0]), rank_max=3)
    rank2 = build_rank_profile(np.array([1.0, 0.2, 0.0]), rank_max=3)
    high = build_rank_profile(np.array([1.0, 0.4, 0.2, 0.1]), rank_max=4)
    assert rank1.effective_ranks[0.05] == 1
    assert rank2.effective_ranks[0.05] == 2
    assert classify_truth_profile(high) == "higher_rank"
    assert high.tail_beyond_rank_max == 0.0


def test_energy_beyond_rank_max_is_preserved():
    profile = build_rank_profile(
        np.array([1.0, 0.3, 0.2, 0.1]), rank_max=2
    )
    assert profile.tail_beyond_rank_max > 0
