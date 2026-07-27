from __future__ import annotations

import numpy as np

from ar_raphu.spectral.penalty_interval import (
    automatic_penalty_interval,
    expand_penalty_interval,
    normalize_penalty_relative_to_gram,
    numerical_jitter,
    penalty_boundary_status,
)


def test_penalty_normalization_is_invariant_to_raw_component_scale() -> None:
    gram = np.diag([2.0, 1.0, 0.5])
    penalty = np.array(
        [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]
    )
    first = normalize_penalty_relative_to_gram(penalty, gram)
    second = normalize_penalty_relative_to_gram(1e8 * penalty, gram)
    np.testing.assert_allclose(first.normalized, second.normalized)
    assert np.isclose(
        np.median(first.positive_generalized_eigenvalues), 1.0
    )


def test_automatic_interval_covers_near_zero_and_unpenalized_modes() -> None:
    gram = np.eye(3)
    penalty = np.diag([0.1, 1.0, 10.0])
    normalized = normalize_penalty_relative_to_gram(penalty, gram)
    interval = automatic_penalty_interval(normalized.normalized, gram)
    eigenvalues = normalized.positive_generalized_eigenvalues
    strongest_shrinkage = 1.0 / (
        1.0 + interval.lower * np.max(eigenvalues)
    )
    weakest_shrinkage = 1.0 / (
        1.0 + interval.upper * np.min(eigenvalues)
    )
    assert np.isclose(strongest_shrinkage, 0.999)
    assert np.isclose(weakest_shrinkage, 0.001)
    assert len(interval.grid(7)) == 7


def test_boundary_policy_expands_twice_then_blocks_confirmation() -> None:
    interval = automatic_penalty_interval(np.eye(3), np.eye(3))
    assert (
        penalty_boundary_status(
            selected_index=0, grid_size=7, expansion_count=0
        )
        == "PENALTY_INTERVAL_EXPANSION_REQUIRED"
    )
    interval = expand_penalty_interval(interval, boundary="lower")
    interval = expand_penalty_interval(interval, boundary="lower")
    assert interval.expansion_count == 2
    assert (
        penalty_boundary_status(
            selected_index=0,
            grid_size=7,
            expansion_count=interval.expansion_count,
        )
        == "PENALTY_INTERVAL_NOT_CERTIFIED"
    )
    assert (
        penalty_boundary_status(
            selected_index=3,
            grid_size=7,
            expansion_count=interval.expansion_count,
        )
        == "PENALTY_INTERVAL_CERTIFIED"
    )


def test_numerical_jitter_is_machine_scale_not_scientific_ridge() -> None:
    system = np.diag([1.0, 10.0])
    jitter = numerical_jitter(system)
    assert jitter == 64.0 * np.finfo(np.float64).eps * 10.0
    assert jitter < 1e-12
