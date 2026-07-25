"""Pre-registered time-series bootstrap and multiple-testing utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def residual_acf_block_length(
    residuals: np.ndarray,
    *,
    minimum: int = 2,
    search_cap: int | None = None,
) -> int:
    """First non-significant residual ACF lag under 1.96/sqrt(T)."""

    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("At least four finite residuals are required.")
    centered = values - values.mean()
    denominator = float(centered @ centered)
    if denominator == 0.0:
        return minimum
    cap = (
        min(512, len(values) // 10)
        if search_cap is None
        else min(int(search_cap), len(values) - 1)
    )
    cap = max(minimum, cap)
    threshold = 1.96 / np.sqrt(len(values))
    for lag in range(1, cap + 1):
        autocorrelation = float(centered[:-lag] @ centered[lag:] / denominator)
        if abs(autocorrelation) <= threshold:
            return max(minimum, lag)
    return max(minimum, cap)


def moving_block_indices(
    length: int,
    block_length: int,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if length <= 0 or not 1 <= block_length <= length or replicates <= 0:
        raise ValueError("Invalid moving-block bootstrap dimensions.")
    blocks_per_replicate = int(np.ceil(length / block_length))
    starts = rng.integers(
        0,
        length - block_length + 1,
        size=(replicates, blocks_per_replicate),
    )
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets).reshape(replicates, -1)
    return indices[:, :length]


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    observed_loss_difference: float
    confidence_interval_95: tuple[float, float]
    one_sided_p_value: float
    block_length: int
    replicates: int
    bootstrap_differences: np.ndarray


def paired_moving_block_rmse_difference(
    observed: np.ndarray,
    prediction_reference: np.ndarray,
    prediction_candidate: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
    block_length: int | None = None,
) -> PairedBootstrapResult:
    """Bootstrap RMSE(reference)-RMSE(candidate); positive favors candidate."""

    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    reference = np.asarray(prediction_reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(prediction_candidate, dtype=np.float64).reshape(-1)
    if not (len(y) == len(reference) == len(candidate)):
        raise ValueError("Observed and both prediction vectors must align.")
    if not (
        np.isfinite(y).all()
        and np.isfinite(reference).all()
        and np.isfinite(candidate).all()
    ):
        raise ValueError("Bootstrap inputs must be finite.")
    residual_reference = reference - y
    residual_candidate = candidate - y
    if block_length is None:
        block_length = max(
            residual_acf_block_length(residual_reference),
            residual_acf_block_length(residual_candidate),
        )
    indices = moving_block_indices(
        len(y), block_length, replicates=replicates, rng=rng
    )
    ref_rmse = np.sqrt(np.mean(np.square(residual_reference[indices]), axis=1))
    candidate_rmse = np.sqrt(
        np.mean(np.square(residual_candidate[indices]), axis=1)
    )
    differences = ref_rmse - candidate_rmse
    observed_difference = float(
        np.sqrt(np.mean(np.square(residual_reference)))
        - np.sqrt(np.mean(np.square(residual_candidate)))
    )
    p_value = float((1 + np.count_nonzero(differences <= 0.0)) / (replicates + 1))
    interval = tuple(np.quantile(differences, [0.025, 0.975]).tolist())
    return PairedBootstrapResult(
        observed_loss_difference=observed_difference,
        confidence_interval_95=(float(interval[0]), float(interval[1])),
        one_sided_p_value=p_value,
        block_length=int(block_length),
        replicates=int(replicates),
        bootstrap_differences=differences,
    )


def benjamini_hochberg(
    p_values: np.ndarray, *, q: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Return reject flags and monotone BH-adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a nonempty vector in [0,1].")
    if not 0 < q < 1:
        raise ValueError("q must be in (0,1).")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    count = len(values)
    thresholds = q * np.arange(1, count + 1) / count
    eligible = np.where(ranked <= thresholds)[0]
    reject = np.zeros(count, dtype=bool)
    if len(eligible):
        reject[order[: eligible[-1] + 1]] = True
    adjusted_ranked = np.minimum.accumulate(
        (ranked * count / np.arange(1, count + 1))[::-1]
    )[::-1]
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return reject, adjusted

