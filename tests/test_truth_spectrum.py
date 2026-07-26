import numpy as np

from ar_raphu.spectral.truth_spectrum import classify_truth_spectrum


def test_truth_rank_class_is_computed_from_spectrum():
    assert classify_truth_spectrum(np.array([2.0, 0.0, 0.0])).rank_class == "rank1"
    assert (
        classify_truth_spectrum(np.array([2.0, 0.05, 0.0])).rank_class
        == "weak_rank2"
    )
    assert (
        classify_truth_spectrum(np.array([2.0, 0.5, 0.0])).rank_class
        == "strong_rank2"
    )
