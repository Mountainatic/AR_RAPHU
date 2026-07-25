from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.synthetic import (  # noqa: E402
    SCENARIOS,
    generate_synthetic_sequence,
)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_preregistered_scenarios_are_finite_and_reproducible(
    scenario: str,
) -> None:
    first = generate_synthetic_sequence(
        scenario, seed=3, n_samples=256, external_variables=10
    )
    second = generate_synthetic_sequence(
        scenario, seed=3, n_samples=256, external_variables=10
    )

    assert first.x.shape == (379, 10)
    assert first.y_observed.shape == (379,)
    assert first.usable_target_count == 256
    assert np.isfinite(first.x).all()
    assert np.isfinite(first.y_observed).all()
    assert np.array_equal(first.x, second.x)
    assert np.array_equal(first.y_observed, second.y_observed)
    assert first.truth["current_target_in_input"] is False
    assert first.truth["future_X_in_primary_input"] is False


def test_s0_has_no_external_support_and_s1_has_normalized_rank1_truth() -> None:
    s0 = generate_synthetic_sequence("AR-S0", seed=0, n_samples=128)
    s1 = generate_synthetic_sequence("AR-S1", seed=0, n_samples=128)

    assert s0.truth["active_support"] == []
    assert np.count_nonzero(s0.truth["q_primary"]) == 0
    assert s0.truth["generator_version"] == 2
    assert s0.truth["latent_process_innovation_standard_deviation"] == 0.2
    assert np.std(
        s0.y_measurement_clean[s0.target_start : s0.target_stop]
    ) > 0.0
    assert np.std(s0.y_observed[s0.target_start : s0.target_stop]) > 0.0
    assert s1.truth["active_support"] == [0, 1, 2]
    assert np.allclose(s1.truth["q_primary"][:3].sum(axis=1), 1.0)
    assert set(s1.truth["rank_by_variable"].values()) == {1}


def test_s2_is_bimodal_rank1_and_s3_has_two_independent_components() -> None:
    s2 = generate_synthetic_sequence("AR-S2", seed=1, n_samples=128)
    s3 = generate_synthetic_sequence("AR-S3", seed=1, n_samples=128)

    mode_pairs = ((4, 18), (8, 28), (12, 44))
    for variable, (first_mode, second_mode) in enumerate(mode_pairs):
        q = s2.truth["q_primary"][variable]
        assert q[max(0, first_mode - 3) : first_mode + 4].sum() > 0.4
        assert q[second_mode - 3 : second_mode + 4].sum() > 0.4
        assert s2.truth["rank_by_variable"][str(variable)] == 1
        assert s3.truth["q_primary"][variable].sum() == pytest.approx(1.0)
        assert s3.truth["q_secondary"][variable].sum() == pytest.approx(1.0)
        assert s3.truth["rank_by_variable"][str(variable)] == 2


def test_s4_delay_increases_with_amplitude() -> None:
    sequence = generate_synthetic_sequence("AR-S4", seed=4, n_samples=512)
    means = sequence.truth["dynamic_delay_means"][
        sequence.target_start :, 0
    ]
    x_for_means = sequence.x[sequence.target_start - 1 : -1, 0]

    order = np.argsort(x_for_means)
    assert np.all(np.diff(means[order]) >= -1.0e-12)


def test_s5_filter_and_s6_delay_are_exact_before_noise() -> None:
    s5 = generate_synthetic_sequence("AR-S5", seed=5, n_samples=128)
    s6 = generate_synthetic_sequence("AR-S6", seed=5, n_samples=128)
    weights = s5.truth["measurement_filter_weights"]
    index = s5.target_start

    expected_filtered = np.dot(
        weights, s5.y_latent[index - np.arange(len(weights))]
    )
    assert s5.y_measurement_clean[index] == pytest.approx(expected_filtered)
    delay = s6.truth["measurement_delay_samples"]
    assert np.array_equal(
        s6.y_measurement_clean[delay:], s6.y_latent[:-delay]
    )


def test_s7_contains_feedback_and_correlated_distractor() -> None:
    sequence = generate_synthetic_sequence("AR-S7", seed=7, n_samples=1024)
    start = sequence.target_start
    correlation = np.corrcoef(
        sequence.x[start:, 0], sequence.x[start:, 3]
    )[0, 1]
    lagged_feedback = np.corrcoef(
        sequence.x[start:, 0], sequence.y_latent[start - 1 : -1]
    )[0, 1]

    assert correlation > 0.98
    assert abs(lagged_feedback) > 0.05


def test_splits_are_contiguous_and_cover_only_registered_targets() -> None:
    sequence = generate_synthetic_sequence("AR-S1", seed=2, n_samples=1000)
    train = sequence.split_target_intervals["train"]
    validation = sequence.split_target_intervals["validation"]
    test = sequence.split_target_intervals["test"]

    assert train == (sequence.target_start, sequence.target_start + 600)
    assert train[1] == validation[0]
    assert validation[1] == test[0]
    assert test[1] == sequence.target_stop
