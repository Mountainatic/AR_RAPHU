import numpy as np
import pytest

from ar_raphu.spectral.synthetic_components import (
    e2a_component_target,
    e2b_total_external_target,
    replay_synthetic_components,
)
from ar_raphu.synthetic import generate_synthetic_sequence


@pytest.mark.parametrize("scenario", ["AR-S0", "AR-S1", "AR-S3", "AR-S4"])
def test_synthetic_component_replay_closes_latent_identity(scenario):
    sequence = generate_synthetic_sequence(scenario, seed=3, n_samples=400)
    components = replay_synthetic_components(sequence)
    start, stop = sequence.split_target_intervals["train"]
    reconstructed = (
        components.ar_contribution
        + components.x_total_contribution
        + components.process_innovation
    )
    assert np.max(np.abs(sequence.y_latent[start:stop] - reconstructed[start:stop])) <= 1e-10
    noise = sequence.y_observed - sequence.y_measurement_clean
    assert np.max(np.abs(noise - components.measurement_noise)) <= 1e-10


def test_rank2_replay_retains_variable_components():
    sequence = generate_synthetic_sequence("AR-S3", seed=1, n_samples=300)
    components = replay_synthetic_components(sequence)
    start, stop = sequence.split_target_intervals["train"]
    assert np.all(
        np.var(components.x_contribution_by_variable[start:stop, :3], axis=0) > 0
    )
    assert np.allclose(
        components.x_contribution_by_variable[:, 3:], 0.0, atol=0.0
    )


def test_e2a_target_is_one_external_component_and_contains_no_ar():
    sequence = generate_synthetic_sequence("AR-S1", seed=2, n_samples=300)
    components = replay_synthetic_components(sequence)
    indices = np.arange(*sequence.split_target_intervals["validation"])
    target = e2a_component_target(components, 1, indices)
    assert np.array_equal(
        target, components.x_contribution_by_variable[indices, 1]
    )
    assert not np.allclose(target, target + components.ar_contribution[indices])


def test_e2b_target_is_active_external_total_and_contains_no_ar():
    sequence = generate_synthetic_sequence("AR-S3", seed=2, n_samples=300)
    components = replay_synthetic_components(sequence)
    indices = np.arange(*sequence.split_target_intervals["validation"])
    target = e2b_total_external_target(
        components, sequence.truth["active_support"], indices
    )
    assert np.allclose(target, components.x_total_contribution[indices])
    assert not np.allclose(target, target + components.ar_contribution[indices])
