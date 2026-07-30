"""Moving-block bootstrap and residual correlation-time diagnostics."""

from __future__ import annotations

import numpy as np


def residual_correlation_time(
    residual: np.ndarray,
    *,
    maximum_lag: int | None = None,
) -> int:
    values = np.asarray(residual, dtype=np.float64)
    centered = values - values.mean()
    denominator = float(centered @ centered)
    if denominator <= np.finfo(np.float64).eps:
        return 1
    limit = min(
        len(values) // 4,
        maximum_lag if maximum_lag is not None else 500,
    )
    for lag in range(1, max(2, limit + 1)):
        correlation = float(
            centered[:-lag] @ centered[lag:] / denominator
        )
        if correlation <= np.exp(-1.0):
            return lag
    return max(1, limit)


def moving_block_improvement(
    baseline_loss: np.ndarray,
    model_loss: np.ndarray,
    *,
    replicates: int,
    block_length: int,
    seed: int,
) -> dict[str, object]:
    baseline = np.asarray(baseline_loss, dtype=np.float64)
    model = np.asarray(model_loss, dtype=np.float64)
    if len(baseline) != len(model) or len(baseline) == 0:
        raise ValueError("BOOTSTRAP_LOSS_LENGTH_MISMATCH")
    rng = np.random.default_rng(seed)
    length = max(1, min(int(block_length), len(baseline)))
    blocks = int(np.ceil(len(baseline) / length))
    starts = rng.integers(0, len(baseline), size=(replicates, blocks))
    offsets = np.arange(length, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(baseline)
    indices = indices.reshape(replicates, -1)[:, : len(baseline)]
    values = 1.0 - np.mean(model[indices], axis=1) / np.maximum(
        np.mean(baseline[indices], axis=1),
        np.finfo(np.float64).eps,
    )
    return {
        "replicates": int(replicates),
        "block_length": length,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "upper_95": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
    }


def stratified_two_direction_improvement(
    losses: list[tuple[np.ndarray, np.ndarray]],
    *,
    replicates: int,
    block_lengths: list[int],
    seed: int,
) -> dict[str, object]:
    if len(losses) != 2 or len(block_lengths) != 2:
        raise ValueError("TWO_DIRECTION_BOOTSTRAP_REQUIRES_TWO_STRATA")
    rng = np.random.default_rng(seed)
    baseline_sums = np.zeros(replicates, dtype=np.float64)
    model_sums = np.zeros(replicates, dtype=np.float64)
    total_counts = np.zeros(replicates, dtype=np.float64)
    for (baseline, model), requested_length in zip(
        losses, block_lengths, strict=True
    ):
        b = np.asarray(baseline, dtype=np.float64)
        m = np.asarray(model, dtype=np.float64)
        length = max(1, min(int(requested_length), len(b)))
        blocks = int(np.ceil(len(b) / length))
        starts = rng.integers(0, len(b), size=(replicates, blocks))
        offsets = np.arange(length, dtype=np.int64)
        indices = (starts[:, :, None] + offsets[None, None, :]) % len(b)
        indices = indices.reshape(replicates, -1)[:, : len(b)]
        baseline_sums += np.sum(b[indices], axis=1)
        model_sums += np.sum(m[indices], axis=1)
        total_counts += len(b)
    values = 1.0 - model_sums / np.maximum(
        baseline_sums, np.finfo(np.float64).eps
    )
    return {
        "replicates": int(replicates),
        "block_lengths": [int(value) for value in block_lengths],
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "upper_95": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
    }
