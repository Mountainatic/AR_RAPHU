"""Scale-aware spectral penalty normalization and interval certification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PenaltyNormalization:
    normalized: np.ndarray
    scale: float
    method: str
    positive_generalized_eigenvalues: np.ndarray


@dataclass(frozen=True, slots=True)
class LogPenaltyInterval:
    lower: float
    upper: float
    expansion_count: int = 0

    def grid(self, points: int) -> np.ndarray:
        if points < 3:
            raise ValueError("Penalty interval requires at least three points.")
        return np.geomspace(self.lower, self.upper, points)


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Expected a square matrix.")
    return 0.5 * (array + array.T)


def positive_generalized_eigenvalues(
    penalty: np.ndarray,
    gram: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return finite positive eigenvalues of P v = mu H v on range(H)."""

    p = _symmetric(penalty)
    h = _symmetric(gram)
    if p.shape != h.shape:
        raise ValueError("Penalty and Gram matrices must have equal shape.")
    h_values, h_vectors = np.linalg.eigh(h)
    h_scale = max(float(np.max(np.abs(h_values))), 1.0)
    keep = h_values > relative_tolerance * h_scale
    if not np.any(keep):
        return np.empty(0, dtype=np.float64)
    whitening = h_vectors[:, keep] / np.sqrt(h_values[keep])[None, :]
    whitened_penalty = _symmetric(whitening.T @ p @ whitening)
    values = np.linalg.eigvalsh(whitened_penalty)
    value_scale = max(float(np.max(np.abs(values))), 1.0)
    positive = values[values > relative_tolerance * value_scale]
    return np.asarray(positive, dtype=np.float64)


def normalize_penalty_relative_to_gram(
    penalty: np.ndarray,
    gram: np.ndarray,
) -> PenaltyNormalization:
    """Normalize P so its positive generalized-eigenvalue median is one."""

    p = _symmetric(penalty)
    h = _symmetric(gram)
    values = positive_generalized_eigenvalues(p, h)
    if len(values):
        scale = float(np.median(values))
        method = "POSITIVE_GENERALIZED_EIGENVALUE_MEDIAN"
    else:
        p_values = np.linalg.eigvalsh(p)
        h_values = np.linalg.eigvalsh(h)
        p_positive = p_values[p_values > np.finfo(np.float64).eps]
        h_positive = h_values[h_values > np.finfo(np.float64).eps]
        if not len(p_positive) or not len(h_positive):
            raise ValueError("Penalty normalization has no positive support.")
        scale = float(np.mean(p_positive) / np.mean(h_positive))
        method = "POSITIVE_TRACE_PER_RANK_RATIO"
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Penalty normalization scale is not positive finite.")
    normalized = p / scale
    normalized_values = positive_generalized_eigenvalues(normalized, h)
    return PenaltyNormalization(
        normalized=normalized,
        scale=scale,
        method=method,
        positive_generalized_eigenvalues=normalized_values,
    )


def automatic_penalty_interval(
    normalized_penalty: np.ndarray,
    gram: np.ndarray,
    *,
    near_unpenalized_shrinkage: float = 0.999,
    near_zero_shrinkage: float = 0.001,
) -> LogPenaltyInterval:
    """Cover from an almost-unpenalized strongest mode to a near-zero weakest mode."""

    if not 0.5 < near_unpenalized_shrinkage < 1.0:
        raise ValueError("near_unpenalized_shrinkage must be in (0.5, 1).")
    if not 0.0 < near_zero_shrinkage < 0.5:
        raise ValueError("near_zero_shrinkage must be in (0, 0.5).")
    values = positive_generalized_eigenvalues(normalized_penalty, gram)
    if not len(values):
        raise ValueError("Cannot construct an interval without positive modes.")
    lower = (
        1.0 / near_unpenalized_shrinkage - 1.0
    ) / float(np.max(values))
    upper = (1.0 / near_zero_shrinkage - 1.0) / float(np.min(values))
    lower = max(lower, np.finfo(np.float64).tiny)
    if not np.isfinite(upper) or upper <= lower:
        raise ValueError("Automatic penalty interval is not ordered.")
    return LogPenaltyInterval(lower=lower, upper=upper)


def expand_penalty_interval(
    interval: LogPenaltyInterval,
    *,
    boundary: str,
) -> LogPenaltyInterval:
    """Expand one side by the current log-width; never move the other side."""

    log_lower = float(np.log(interval.lower))
    log_upper = float(np.log(interval.upper))
    width = log_upper - log_lower
    if boundary == "lower":
        log_lower -= width
    elif boundary == "upper":
        log_upper += width
    else:
        raise ValueError("boundary must be 'lower' or 'upper'.")
    return LogPenaltyInterval(
        lower=float(np.exp(log_lower)),
        upper=float(np.exp(log_upper)),
        expansion_count=interval.expansion_count + 1,
    )


def penalty_boundary_status(
    *,
    selected_index: int,
    grid_size: int,
    expansion_count: int,
    maximum_expansions: int = 2,
) -> str:
    if grid_size < 3 or not 0 <= selected_index < grid_size:
        raise ValueError("Invalid selected penalty-grid index.")
    if 0 < selected_index < grid_size - 1:
        return "PENALTY_INTERVAL_CERTIFIED"
    if expansion_count < maximum_expansions:
        return "PENALTY_INTERVAL_EXPANSION_REQUIRED"
    return "PENALTY_INTERVAL_NOT_CERTIFIED"


def numerical_jitter(
    gram_plus_scientific_penalty: np.ndarray,
    *,
    multiplier: float = 64.0,
) -> float:
    """Machine-scale jitter recorded separately from scientific regularization."""

    system = _symmetric(gram_plus_scientific_penalty)
    spectral_norm = float(np.linalg.norm(system, ord=2))
    return float(
        multiplier
        * np.finfo(np.float64).eps
        * max(1.0, spectral_norm)
    )
