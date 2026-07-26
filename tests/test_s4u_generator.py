import numpy as np
import pytest

from ar_raphu.spectral.capacity_diagnostics import (
    ModelClassMismatchError,
    direct_apply_truth_kernel,
)
from ar_raphu.spectral.synthetic_components import (
    replay_synthetic_components,
    true_kernel_surface,
)
from ar_raphu.synthetic import generate_synthetic_sequence


def test_s4u_component_replay_closes_latent_identity():
    sequence = generate_synthetic_sequence("AR-S4U", seed=3, n_samples=400)
    components = replay_synthetic_components(sequence)
    start, stop = sequence.split_target_intervals["train"]
    reconstructed = (
        components.ar_contribution
        + components.x_total_contribution
        + components.process_innovation
    )
    assert (
        np.max(
            np.abs(
                sequence.y_latent[start:stop] - reconstructed[start:stop]
            )
        )
        <= 1e-12
    )


def test_s4u_direct_truth_operator_matches_replayed_component():
    sequence = generate_synthetic_sequence("AR-S4U", seed=1, n_samples=300)
    components = replay_synthetic_components(sequence)
    indices = np.arange(*sequence.split_target_intervals["validation"])
    direct = direct_apply_truth_kernel(sequence, 0, indices)
    replayed = components.x_contribution_by_variable[indices, 0]
    assert np.max(np.abs(direct - replayed)) <= 1e-12


def test_s4c_is_rejected_as_a_2d_truth_surface():
    sequence = generate_synthetic_sequence("AR-S4", seed=0, n_samples=200)
    with pytest.raises(ModelClassMismatchError):
        true_kernel_surface(sequence, 0, np.linspace(-1.0, 1.0, 21))
