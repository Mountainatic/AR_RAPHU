"""Basis-invariant Gram-whitened kernel spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


def _symmetric_root(matrix: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    values, vectors = scipy.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    floor = max(values.max(), 1.0) * 1.0e-12
    values = np.maximum(values, floor)
    powers = values ** (-0.5 if inverse else 0.5)
    return (vectors * powers) @ vectors.T


@dataclass(frozen=True, slots=True)
class GramSpectrum:
    singular_values: np.ndarray
    left_vectors: np.ndarray
    right_vectors: np.ndarray
    whitened_coefficients: np.ndarray
    lag_root: np.ndarray
    amplitude_root: np.ndarray

    def truncate(self, rank: int) -> np.ndarray:
        if not 1 <= rank <= len(self.singular_values):
            raise ValueError("Invalid truncation rank.")
        white = (
            self.left_vectors[:, :rank]
            * self.singular_values[:rank]
        ) @ self.right_vectors[:rank]
        return (
            _symmetric_root(self.lag_root @ self.lag_root, inverse=True)
            @ white
            @ _symmetric_root(
                self.amplitude_root @ self.amplitude_root, inverse=True
            )
        )

    def tail_energy_ratio(self, rank: int) -> float:
        energy = self.singular_values**2
        return float(energy[rank:].sum() / max(energy.sum(), np.finfo(float).eps))


def gram_whitened_svd(
    coefficients: np.ndarray,
    lag_gram: np.ndarray,
    amplitude_gram: np.ndarray,
) -> GramSpectrum:
    theta = np.asarray(coefficients, dtype=np.float64)
    lag_root = _symmetric_root(lag_gram)
    amplitude_root = _symmetric_root(amplitude_gram)
    whitened = lag_root @ theta @ amplitude_root
    u, singular_values, vh = scipy.linalg.svd(
        whitened, full_matrices=False, check_finite=True
    )
    return GramSpectrum(
        singular_values=singular_values,
        left_vectors=u,
        right_vectors=vh,
        whitened_coefficients=whitened,
        lag_root=lag_root,
        amplitude_root=amplitude_root,
    )
