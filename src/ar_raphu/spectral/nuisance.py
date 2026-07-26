"""Ridge nuisance fitting for y and the external design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True, slots=True)
class NuisanceFit:
    y_coefficients: np.ndarray
    phi_coefficients: np.ndarray
    ridge_y: float
    ridge_phi: float

    def predict(self, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(psi, dtype=np.float64)
        return values @ self.y_coefficients, values @ self.phi_coefficients


def _ridge(psi: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    system = psi.T @ psi / len(psi) + ridge * np.eye(psi.shape[1])
    rhs = psi.T @ target / len(psi)
    return scipy.linalg.solve(system, rhs, assume_a="pos", check_finite=True)


def select_ridge(
    psi: np.ndarray,
    target: np.ndarray,
    candidates: tuple[float, ...],
    *,
    validation_fraction: float,
) -> float:
    split = max(1, int(np.floor((1.0 - validation_fraction) * len(psi))))
    if split >= len(psi):
        raise ValueError("Not enough nuisance samples for tail validation.")
    scores = []
    for ridge in candidates:
        coefficients = _ridge(psi[:split], target[:split], ridge)
        error = target[split:] - psi[split:] @ coefficients
        scores.append(float(np.mean(error**2)))
    return float(candidates[int(np.argmin(scores))])


def fit_nuisance(
    psi: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    *,
    ridge_candidates: tuple[float, ...],
    validation_fraction: float,
) -> NuisanceFit:
    ridge_y = select_ridge(
        psi, y, ridge_candidates, validation_fraction=validation_fraction
    )
    ridge_phi = select_ridge(
        psi, phi, ridge_candidates, validation_fraction=validation_fraction
    )
    return NuisanceFit(
        y_coefficients=_ridge(psi, y, ridge_y),
        phi_coefficients=_ridge(psi, phi, ridge_phi),
        ridge_y=ridge_y,
        ridge_phi=ridge_phi,
    )
