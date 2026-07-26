"""Oracle weighted truth-spectrum diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True, slots=True)
class TruthSpectrum:
    singular_values: np.ndarray
    tail_energy: float
    sigma2_sigma1: float
    spectral_gap: float
    rank_class: str


def classify_truth_spectrum(
    singular_values: np.ndarray,
    *,
    tail_energy_min: float = 0.15,
    sigma2_sigma1_min: float = 0.10,
) -> TruthSpectrum:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("singular_values must be a non-empty vector.")
    padded = np.pad(values, (0, max(0, 3 - len(values))))
    energy = values**2
    tail = float(
        np.sqrt(
            energy[1:].sum()
            / max(float(energy.sum()), np.finfo(np.float64).eps)
        )
    )
    ratio = float(
        padded[1] / max(float(padded[0]), np.finfo(np.float64).eps)
    )
    tolerance = max(float(padded[0]), 1.0) * 1.0e-10
    if padded[1] <= tolerance:
        rank_class = "rank1"
    elif tail >= tail_energy_min and ratio >= sigma2_sigma1_min:
        rank_class = "strong_rank2"
    else:
        rank_class = "weak_rank2"
    gap = float(min(padded[0] - padded[1], padded[1] - padded[2]))
    return TruthSpectrum(
        singular_values=values,
        tail_energy=tail,
        sigma2_sigma1=ratio,
        spectral_gap=max(gap, 0.0),
        rank_class=rank_class,
    )


def truth_spectrum(
    whitened_truth: np.ndarray,
    *,
    tail_energy_min: float = 0.15,
    sigma2_sigma1_min: float = 0.10,
) -> TruthSpectrum:
    return classify_truth_spectrum(
        scipy.linalg.svdvals(np.asarray(whitened_truth, dtype=np.float64)),
        tail_energy_min=tail_energy_min,
        sigma2_sigma1_min=sigma2_sigma1_min,
    )
