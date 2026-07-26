"""Reproducible marginal-permuted and space-filling excitation sequences."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .amplitude_domain import AmplitudeDomain


def permuted_marginal_excitation(
    train_values: np.ndarray,
    *,
    length: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(train_values, dtype=np.float64).reshape(-1)
    if not values.size or length <= 0:
        raise ValueError("Need non-empty train values and positive length.")
    permutation = np.random.default_rng(seed).permutation(values)
    repeats = int(np.ceil(length / len(permutation)))
    return np.tile(permutation, repeats)[:length].copy()


def space_filling_core_excitation(
    domain: AmplitudeDomain,
    *,
    length: int,
    seed: int,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive.")
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    unit = sampler.random(length).reshape(-1)
    return qmc.scale(
        unit[:, None],
        [domain.core_lower],
        [domain.core_upper],
    ).reshape(-1)


def space_filling_history_excitation(
    domain: AmplitudeDomain,
    *,
    sample_count: int,
    lag_count: int,
    seed: int,
) -> np.ndarray:
    """Fill the complete lag-history cube for structural operator audits."""

    if sample_count <= 0 or lag_count <= 0:
        raise ValueError("sample_count and lag_count must be positive.")
    sampler = qmc.Sobol(d=lag_count, scramble=True, seed=seed)
    unit = sampler.random(sample_count)
    return qmc.scale(
        unit,
        np.full(lag_count, domain.core_lower),
        np.full(lag_count, domain.core_upper),
    )


def chronological_split_indices(
    length: int,
    *,
    burn_in: int,
    fractions: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    if burn_in >= length or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Invalid burn-in or split fractions.")
    indices = np.arange(burn_in, length, dtype=np.int64)
    train_stop = int(np.floor(fractions[0] * len(indices)))
    validation_stop = train_stop + int(np.floor(fractions[1] * len(indices)))
    return {
        "train": indices[:train_stop],
        "validation": indices[train_stop:validation_stop],
        "test": indices[validation_stop:],
    }
