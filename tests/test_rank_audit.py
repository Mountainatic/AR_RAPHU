import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.rank_audit import (
    discrete_lag_gram,
    empirical_amplitude_gram,
    gram_whitened_rank_audit,
    orthogonalize_lag_basis,
    sym_psd_sqrt,
)


def test_anchor_orthogonalization_is_exact_under_nonuniform_weights() -> None:
    rng = np.random.default_rng(2)
    q = rng.uniform(size=32)
    q /= q.sum()
    basis = rng.normal(size=(32, 7))
    weights = np.linspace(0.5, 2.0, 32)
    residual = orthogonalize_lag_basis(basis, q, weights=weights)
    np.testing.assert_allclose(q @ (weights[:, None] * residual), 0.0, atol=1e-12)


def test_psd_sqrt_reconstructs_singular_gram() -> None:
    design = np.arange(24, dtype=float).reshape(8, 3)
    gram = design.T @ design
    root = sym_psd_sqrt(gram)
    np.testing.assert_allclose(root @ root, gram, atol=1e-9, rtol=1e-9)


def test_whitened_audit_distinguishes_rank_one_and_rank_two() -> None:
    rng = np.random.default_rng(4)
    lag_basis = rng.normal(size=(64, 8))
    amplitude_basis = rng.normal(size=(1000, 9))
    lag_gram = discrete_lag_gram(lag_basis)
    amplitude_gram = empirical_amplitude_gram(amplitude_basis)
    rank1 = np.outer(rng.normal(size=8), rng.normal(size=9))
    audit1 = gram_whitened_rank_audit(rank1, lag_gram, amplitude_gram)
    assert audit1.first_singular_energy > 1.0 - 1e-12
    rank2 = rank1 + np.outer(rng.normal(size=8), rng.normal(size=9))
    audit2 = gram_whitened_rank_audit(rank2, lag_gram, amplitude_gram)
    assert len(audit2.singular_values) == 8
    assert audit2.first_singular_energy < 1.0 - 1e-6
    assert audit2.nonseparability > 0.0

