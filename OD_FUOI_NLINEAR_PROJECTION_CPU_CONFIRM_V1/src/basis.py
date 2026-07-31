from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import block_diag


def _open_uniform_knots(number_of_basis: int, degree: int) -> np.ndarray:
    internal_count = int(number_of_basis) - int(degree) - 1
    internal = np.linspace(0.0, 1.0, internal_count + 2, dtype=np.float64)[1:-1]
    return np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))


def _spline_matrix(knots: np.ndarray, degree: int, x: np.ndarray, derivative: int = 0) -> np.ndarray:
    size = len(knots) - degree - 1
    spline = BSpline(knots, np.eye(size, dtype=np.float64), degree, extrapolate=False)
    return np.asarray(spline(np.asarray(x, dtype=np.float64), nu=derivative), dtype=np.float64)


@dataclass(frozen=True)
class LagBasis:
    number_of_basis: int
    degree: int
    knots: np.ndarray
    lag_minutes: np.ndarray
    sample_design: np.ndarray
    gram: np.ndarray
    derivative_gram: np.ndarray
    second_derivative_gram: np.ndarray
    partition_error: float


def build_lag_basis(
    *, number_of_basis: int, degree: int, sequence_steps: int, cadence_sec: float
) -> LagBasis:
    knots = _open_uniform_knots(number_of_basis, degree)
    lag_minutes = np.arange(sequence_steps, dtype=np.float64) * float(cadence_sec) / 60.0
    s = np.sqrt(np.clip(lag_minutes / 40.0, 0.0, 1.0))
    design = _spline_matrix(knots, degree, s)
    grid = np.linspace(0.0, 1.0, 4001, dtype=np.float64)
    b0 = _spline_matrix(knots, degree, grid, 0)
    b1 = _spline_matrix(knots, degree, grid, 1)
    b2 = _spline_matrix(knots, degree, grid, 2)
    gram = np.trapezoid(b0[:, :, None] * b0[:, None, :], grid, axis=0)
    derivative_gram = np.trapezoid(b1[:, :, None] * b1[:, None, :], grid, axis=0)
    second_gram = np.trapezoid(b2[:, :, None] * b2[:, None, :], grid, axis=0)
    return LagBasis(
        number_of_basis=number_of_basis,
        degree=degree,
        knots=knots,
        lag_minutes=lag_minutes,
        sample_design=design,
        gram=gram,
        derivative_gram=derivative_gram,
        second_derivative_gram=second_gram,
        partition_error=float(np.max(np.abs(np.sum(design, axis=1) - 1.0))),
    )


@dataclass(frozen=True)
class AmplitudeBasis:
    channel: str
    degree: int
    mean: float
    scale: float
    lower: float
    upper: float
    knots: np.ndarray
    distinct_quantiles: np.ndarray
    raw_mean: np.ndarray
    transform: np.ndarray
    linear_coordinate: np.ndarray
    gram_eigenvalues: np.ndarray
    left_band: float
    right_band: float
    degenerate: bool
    projection_error: float

    @property
    def reduced_size(self) -> int:
        return int(self.transform.shape[1])


def fit_amplitude_basis(
    values: np.ndarray,
    *,
    channel: str,
    quantiles: list[float],
    degree: int,
    band_spans: float,
) -> AmplitudeBasis:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(np.mean(flattened))
    scale = float(np.std(flattened))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError(f"AMPLITUDE_SUPPORT_DEGENERATE:{channel}:ZERO_SCALE")
    standardized = (flattened - mean) / scale
    points = np.unique(np.quantile(standardized, np.asarray(quantiles) / 100.0))
    degenerate = len(points) < len(quantiles)
    if len(points) < 4:
        raise RuntimeError(f"AMPLITUDE_SUPPORT_DEGENERATE:{channel}:UNIQUE={len(points)}")
    effective_degree = min(int(degree), len(points) - 1)
    internal = points[1:-1]
    knots = np.concatenate(
        (np.repeat(points[0], effective_degree + 1), internal, np.repeat(points[-1], effective_degree + 1))
    )
    raw = _spline_matrix(knots, effective_degree, standardized)
    raw_mean = np.mean(raw, axis=0)
    centered = raw - raw_mean
    gram = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = max(float(np.max(eigenvalues)) * 1e-12, 1e-14)
    keep = eigenvalues > tolerance
    transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    reduced = centered @ transform
    linear = reduced.T @ standardized / len(standardized)
    reconstructed = reduced @ linear
    projection_error = float(np.linalg.norm(reconstructed - standardized) / max(np.linalg.norm(standardized), 1e-30))
    if projection_error > 1e-10 or abs(float(linear @ linear) - 1.0) > 1e-9:
        raise RuntimeError(f"LINEAR_COORDINATE_NOT_IN_SPLINE_SPAN:{channel}:{projection_error}")
    return AmplitudeBasis(
        channel=channel,
        degree=effective_degree,
        mean=mean,
        scale=scale,
        lower=float(points[0]),
        upper=float(points[-1]),
        knots=knots,
        distinct_quantiles=points,
        raw_mean=raw_mean,
        transform=transform,
        linear_coordinate=linear,
        gram_eigenvalues=eigenvalues,
        left_band=float((points[1] - points[0]) * band_spans),
        right_band=float((points[-1] - points[-2]) * band_spans),
        degenerate=degenerate,
        projection_error=projection_error,
    )


def evaluate_reduced(spec: AmplitudeBasis, values: np.ndarray, derivative: int = 0) -> np.ndarray:
    xi = (np.asarray(values, dtype=np.float64) - spec.mean) / spec.scale
    raw = _spline_matrix(spec.knots, spec.degree, xi, derivative)
    if derivative == 0:
        raw = raw - spec.raw_mean
    return raw @ spec.transform


def _hermite_nonlinear_features(spec: AmplitudeBasis, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape
    xi = (values.reshape(-1) - spec.mean) / spec.scale
    output = np.empty((len(xi), spec.reduced_size), dtype=np.float64)
    regions = np.zeros(len(xi), dtype=np.int8)
    q = spec.linear_coordinate

    inside = (xi >= spec.lower) & (xi <= spec.upper)
    if np.any(inside):
        reduced = evaluate_reduced(spec, values.reshape(-1)[inside])
        output[inside] = reduced - xi[inside, None] * q[None, :]

    for side in ("left", "right"):
        boundary = spec.lower if side == "left" else spec.upper
        band = spec.left_band if side == "left" else spec.right_band
        boundary_value = evaluate_reduced(spec, np.array([spec.mean + spec.scale * boundary]))[0] - boundary * q
        boundary_derivative = (
            _spline_matrix(spec.knots, spec.degree, np.array([boundary]), 1) @ spec.transform
        )[0] - q
        if side == "left":
            extension = (xi < boundary) & (xi >= boundary - band)
            saturated = xi < boundary - band
            constant = boundary_value - band * boundary_derivative
            s = (xi[extension] - (boundary - band)) / band
            h00 = 2 * s**3 - 3 * s**2 + 1
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            output[extension] = h00[:, None] * constant + h01[:, None] * boundary_value + h11[:, None] * band * boundary_derivative
        else:
            extension = (xi > boundary) & (xi <= boundary + band)
            saturated = xi > boundary + band
            constant = boundary_value + band * boundary_derivative
            s = (xi[extension] - boundary) / band
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            output[extension] = h00[:, None] * boundary_value + h10[:, None] * band * boundary_derivative + h01[:, None] * constant
        output[saturated] = constant
        regions[extension] = np.maximum(regions[extension], 1)
        regions[saturated] = 2
    return output.reshape(shape + (spec.reduced_size,)), regions.reshape(shape)


def build_fit_design(
    sequence_u: np.ndarray,
    lag: LagBasis,
    amplitude: list[AmplitudeBasis],
    cadence_sec: float,
) -> tuple[np.ndarray, list[slice]]:
    sequence = np.asarray(sequence_u, dtype=np.float64)
    blocks: list[np.ndarray] = []
    slices: list[slice] = []
    start = 0
    dt = float(cadence_sec) / 60.0
    for channel, spec in enumerate(amplitude):
        xi = (sequence[:, :, channel] - spec.mean) / spec.scale
        if np.any(xi < spec.lower - 1e-12) or np.any(xi > spec.upper + 1e-12):
            raise RuntimeError(f"TRAIN_VALUE_OUTSIDE_SUPPORT:{spec.channel}")
        reduced = evaluate_reduced(spec, sequence[:, :, channel].reshape(-1)).reshape(sequence.shape[0], sequence.shape[1], -1)
        tensor = np.einsum("nlm,lt->ntm", reduced, lag.sample_design, optimize=True) * dt
        block = tensor.reshape(len(sequence), -1)
        blocks.append(block)
        slices.append(slice(start, start + block.shape[1]))
        start += block.shape[1]
    return np.concatenate(blocks, axis=1), slices


def build_projected_designs(
    sequence_u: np.ndarray,
    lag: LagBasis,
    amplitude: list[AmplitudeBasis],
    cadence_sec: float,
) -> tuple[np.ndarray, np.ndarray, list[slice], np.ndarray]:
    sequence = np.asarray(sequence_u, dtype=np.float64)
    dt = float(cadence_sec) / 60.0
    linear_blocks: list[np.ndarray] = []
    nonlinear_blocks: list[np.ndarray] = []
    nonlinear_slices: list[slice] = []
    region = np.zeros(sequence.shape[:2] + (sequence.shape[2],), dtype=np.int8)
    start = 0
    for channel, spec in enumerate(amplitude):
        xi = (sequence[:, :, channel] - spec.mean) / spec.scale
        linear_blocks.append(np.einsum("nl,lt->nt", xi, lag.sample_design, optimize=True) * dt)
        nonlinear, current_region = _hermite_nonlinear_features(spec, sequence[:, :, channel])
        region[:, :, channel] = current_region
        tensor = np.einsum("nlm,lt->ntm", nonlinear, lag.sample_design, optimize=True) * dt
        block = tensor.reshape(len(sequence), -1)
        nonlinear_blocks.append(block)
        nonlinear_slices.append(slice(start, start + block.shape[1]))
        start += block.shape[1]
    return (
        np.concatenate(linear_blocks, axis=1),
        np.concatenate(nonlinear_blocks, axis=1),
        nonlinear_slices,
        region,
    )


def sobolev_penalty(lag: LagBasis, amplitude: list[AmplitudeBasis]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for spec in amplitude:
        training_grid = np.linspace(spec.lower, spec.upper, 4001, dtype=np.float64)
        a0 = (_spline_matrix(spec.knots, spec.degree, training_grid, 0) - spec.raw_mean) @ spec.transform
        a1 = _spline_matrix(spec.knots, spec.degree, training_grid, 1) @ spec.transform
        a2 = _spline_matrix(spec.knots, spec.degree, training_grid, 2) @ spec.transform
        g0 = np.trapezoid(a0[:, :, None] * a0[:, None, :], training_grid, axis=0) / (spec.upper - spec.lower)
        g1 = np.trapezoid(a1[:, :, None] * a1[:, None, :], training_grid, axis=0) / (spec.upper - spec.lower)
        g2 = np.trapezoid(a2[:, :, None] * a2[:, None, :], training_grid, axis=0) / (spec.upper - spec.lower)
        block = (
            np.kron(lag.gram, g0)
            + np.kron(lag.second_derivative_gram, g0)
            + 2.0 * np.kron(lag.derivative_gram, g1)
            + np.kron(lag.gram, g2)
        )
        block = (block + block.T) / 2.0
        minimum = float(np.min(np.linalg.eigvalsh(block)))
        if minimum <= 0:
            block += (abs(minimum) + 1e-12) * np.eye(len(block))
        blocks.append(block)
    return block_diag(*blocks)


def split_projection_coefficients(
    coefficient: np.ndarray,
    lag: LagBasis,
    amplitude: list[AmplitudeBasis],
    slices: list[slice],
) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    beta = np.empty((lag.number_of_basis, len(amplitude)), dtype=np.float64)
    nonlinear_blocks: list[np.ndarray] = []
    nonlinear_slices: list[slice] = []
    start = 0
    for channel, (spec, current_slice) in enumerate(zip(amplitude, slices)):
        matrix = coefficient[current_slice].reshape(lag.number_of_basis, spec.reduced_size)
        beta[:, channel] = matrix @ spec.linear_coordinate
        nonlinear = matrix - np.outer(beta[:, channel], spec.linear_coordinate)
        nonlinear_blocks.append(nonlinear.reshape(-1))
        nonlinear_slices.append(slice(start, start + nonlinear.size))
        start += nonlinear.size
    return beta, np.concatenate(nonlinear_blocks), nonlinear_slices


def amplitude_manifest(spec: AmplitudeBasis) -> dict[str, Any]:
    return {
        "channel": spec.channel,
        "degree": spec.degree,
        "mean": spec.mean,
        "scale": spec.scale,
        "lower_xi": spec.lower,
        "upper_xi": spec.upper,
        "distinct_quantiles_xi": spec.distinct_quantiles.tolist(),
        "reduced_size": spec.reduced_size,
        "gram_eigenvalues": spec.gram_eigenvalues.tolist(),
        "left_band_xi": spec.left_band,
        "right_band_xi": spec.right_band,
        "degenerate": spec.degenerate,
        "linear_projection_error": spec.projection_error,
    }
