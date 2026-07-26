import numpy as np

from ar_raphu.spectral.amplitude_domain import AmplitudeDomain
from ar_raphu.spectral.excitation import (
    chronological_split_indices,
    permuted_marginal_excitation,
    space_filling_core_excitation,
)


def test_permuted_and_sobol_excitation_are_reproducible():
    values = np.linspace(-2.0, 2.0, 257)
    domain = AmplitudeDomain.fit(values)
    first_perm = permuted_marginal_excitation(values, length=20000, seed=20003)
    second_perm = permuted_marginal_excitation(values, length=20000, seed=20003)
    first_space = space_filling_core_excitation(
        domain, length=20000, seed=10003
    )
    second_space = space_filling_core_excitation(
        domain, length=20000, seed=10003
    )
    assert np.array_equal(first_perm, second_perm)
    assert np.array_equal(first_space, second_space)
    assert domain.in_domain_mask(first_space).all()


def test_excitation_splits_are_chronological_after_burn_in():
    split = chronological_split_indices(
        20000, burn_in=64, fractions=(0.70, 0.15, 0.15)
    )
    assert split["train"][0] == 64
    assert split["train"][-1] < split["validation"][0]
    assert split["validation"][-1] < split["test"][0]
    assert sum(len(values) for values in split.values()) == 20000 - 64
