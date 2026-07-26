import numpy as np

from ar_raphu.spectral.predictive_rank import predictive_rank_profile


def test_predictive_rank_uses_first_budget_satisfying_rank():
    curve = np.array([0.15, 0.05, 0.01, 0.0])
    profile = predictive_rank_profile(curve)
    assert profile[0.10] == 2
    assert profile[0.05] == 2
    assert profile[0.02] == 3
