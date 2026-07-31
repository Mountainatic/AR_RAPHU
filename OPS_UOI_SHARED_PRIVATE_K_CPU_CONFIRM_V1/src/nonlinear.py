from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class AmplitudeBasis:
    knots: np.ndarray
    degree: int
    lower: float
    upper: float
    lower_band: float
    upper_band: float
    projection: np.ndarray
    orthogonal_transform: np.ndarray


def _raw_spline(
    values: np.ndarray,
    knots: np.ndarray,
    degree: int,
    *,
    lower: float,
    upper: float,
    lower_band: float,
    upper_band: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, lower, upper)
    in_support = BSpline.design_matrix(
        clipped.ravel(), knots, degree, extrapolate=False
    ).toarray()
    basis_count = in_support.shape[1]
    output = in_support.copy()
    identity = np.eye(basis_count)
    splines = [BSpline(knots, identity[:, index], degree) for index in range(basis_count)]
    below = values.ravel() < lower
    above = values.ravel() > upper
    for mask, boundary, band, sign in (
        (below, lower, lower_band, -1.0),
        (above, upper, upper_band, 1.0),
    ):
        if not np.any(mask):
            continue
        distance = sign * (values.ravel()[mask] - boundary)
        width = max(float(band), 1e-12)
        normalized = np.clip(distance / width, 0.0, 1.0)
        h00 = 2 * normalized**3 - 3 * normalized**2 + 1
        h10 = normalized**3 - 2 * normalized**2 + normalized
        for index, spline in enumerate(splines):
            boundary_value = float(spline(boundary))
            boundary_derivative = float(spline.derivative(1)(boundary)) * sign
            output[mask, index] = (
                h00 * boundary_value + h10 * width * boundary_derivative
            )
        outside = distance >= width
        output[np.flatnonzero(mask)[outside], :] = 0.0
    return output.reshape(values.shape + (basis_count,))


def fit_amplitude_basis(values: np.ndarray) -> AmplitudeBasis:
    flattened = np.asarray(values, dtype=np.float64).ravel()
    lower = float(np.min(flattened))
    upper = float(np.max(flattened))
    quantiles = np.quantile(flattened, [0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
    internal = np.unique(
        quantiles[(quantiles > lower + 1e-12) & (quantiles < upper - 1e-12)]
    )
    degree = 3
    knots = np.concatenate(
        (
            np.repeat(lower, degree + 1),
            internal,
            np.repeat(upper, degree + 1),
        )
    )
    lower_gap = float(internal[0] - lower) if len(internal) else upper - lower
    upper_gap = float(upper - internal[-1]) if len(internal) else upper - lower
    raw = _raw_spline(
        flattened,
        knots,
        degree,
        lower=lower,
        upper=upper,
        lower_band=max(lower_gap, 1e-9),
        upper_band=max(upper_gap, 1e-9),
    )
    linear = np.column_stack((np.ones_like(flattened), flattened))
    projection, *_ = np.linalg.lstsq(linear, raw, rcond=None)
    residual = raw - linear @ projection
    _, singular, right_t = np.linalg.svd(residual, full_matrices=False)
    keep = singular > max(float(singular[0]) * 1e-10, 1e-12)
    transform = right_t[keep].T
    return AmplitudeBasis(
        knots=knots,
        degree=degree,
        lower=lower,
        upper=upper,
        lower_band=max(lower_gap, 1e-9),
        upper_band=max(upper_gap, 1e-9),
        projection=projection,
        orthogonal_transform=transform,
    )


def evaluate_amplitude_basis(
    specification: AmplitudeBasis, values: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(values, dtype=np.float64)
    raw = _raw_spline(
        values,
        specification.knots,
        specification.degree,
        lower=specification.lower,
        upper=specification.upper,
        lower_band=specification.lower_band,
        upper_band=specification.upper_band,
    )
    linear = np.stack((np.ones_like(values), values), axis=-1)
    residual = raw - linear @ specification.projection
    transformed = residual @ specification.orthogonal_transform
    below = values < specification.lower
    above = values > specification.upper
    normalized_distance = np.zeros_like(values)
    normalized_distance[below] = (
        specification.lower - values[below]
    ) / specification.lower_band
    normalized_distance[above] = (
        values[above] - specification.upper
    ) / specification.upper_band
    diagnostics = {
        "lower": specification.lower,
        "upper": specification.upper,
        "lower_band": specification.lower_band,
        "upper_band": specification.upper_band,
        "ood_fraction": float(np.mean(below | above)),
        "beyond_finite_band_fraction": float(np.mean(normalized_distance > 1.0)),
        "maximum_normalized_distance": float(np.max(normalized_distance)),
    }
    return transformed, diagnostics


def nonlinear_mode_features(
    amplitude_features: np.ndarray,
    lag_weights: np.ndarray,
    *,
    cadence_sec: float,
) -> np.ndarray:
    # amplitude_features: samples x lag x amplitude-basis
    # lag_weights: lag x modes
    return (
        np.einsum(
            "nlb,lr->nrb",
            amplitude_features,
            lag_weights,
            optimize=True,
        ).reshape(len(amplitude_features), -1)
        * (float(cadence_sec) / 60.0)
    )
