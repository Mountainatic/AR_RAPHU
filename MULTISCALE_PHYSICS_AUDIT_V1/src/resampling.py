"""Causal anti-alias averages and train-frozen signal transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def causal_ema(values: np.ndarray, *, tau_samples: float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if tau_samples <= 0:
        raise ValueError("EMA_TAU_MUST_BE_POSITIVE")
    alpha = float(np.exp(-1.0 / tau_samples))
    output = np.empty_like(data)
    output[0] = data[0]
    for index in range(1, len(data)):
        output[index] = alpha * output[index - 1] + (1.0 - alpha) * data[index]
    return output


def causal_block_average(
    values: np.ndarray,
    origins: np.ndarray,
    *,
    start_samples: int,
    stop_samples: int,
) -> np.ndarray:
    """Average lags [start_samples, stop_samples), including origin at lag 0."""

    data = np.asarray(values, dtype=np.float64)
    points = np.asarray(origins, dtype=np.int64)
    if start_samples < 0 or stop_samples <= start_samples:
        raise ValueError("INVALID_CAUSAL_BLOCK")
    left = points - stop_samples + 1
    right = points - start_samples + 1
    if int(left.min()) < 0 or int(right.max()) > len(data):
        raise ValueError("CAUSAL_BLOCK_OUT_OF_RANGE")
    cumulative = np.concatenate(([0.0], np.cumsum(data, dtype=np.float64)))
    return (cumulative[right] - cumulative[left]) / (right - left)


@dataclass(frozen=True, slots=True)
class PCA1Transform:
    mean: np.ndarray
    scale: np.ndarray
    vector: np.ndarray
    explained_fraction: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "PCA1Transform":
        matrix = np.asarray(values, dtype=np.float64)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale == 0.0] = 1.0
        standardized = (matrix - mean) / scale
        gram = standardized.T @ standardized / len(standardized)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        vector = eigenvectors[:, -1]
        if float(vector.sum()) < 0.0:
            vector = -vector
        explained = float(
            eigenvalues[-1] / max(eigenvalues.sum(), np.finfo(np.float64).eps)
        )
        return cls(mean, scale, vector, explained)

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        return ((matrix - self.mean) / self.scale) @ self.vector


def transform_channel(
    channel: str,
    variant: str,
    columns: dict[str, np.ndarray],
    *,
    sample_period_sec: float,
    pca: PCA1Transform | None = None,
    fit_pca_slice: slice | None = None,
    branch_config: dict[str, object],
) -> tuple[np.ndarray, PCA1Transform | None]:
    if channel == "joint_lift":
        matrix = np.column_stack(
            (columns["晶升速度"], columns["埚升速度"])
        )
        if pca is None:
            if fit_pca_slice is None:
                raise ValueError("TRAIN_SLICE_REQUIRED_FOR_PCA")
            pca = PCA1Transform.fit(matrix[fit_pca_slice])
        return pca.transform(matrix), pca
    if channel == "crystal_rotation":
        raw = np.asarray(columns["晶转速度"], dtype=np.float64)
        if variant == "raw":
            return raw.copy(), None
        tau = float(branch_config["innovation_ema_min"]) * 60.0 / sample_period_sec
        return raw - causal_ema(raw, tau_samples=tau), None
    if channel == "crucible_rotation":
        raw = np.asarray(columns["埚转速度"], dtype=np.float64)
        if variant == "raw":
            return raw.copy(), None
        slow_tau = (
            float(branch_config["midband_slow_ema_min"])
            * 60.0
            / sample_period_sec
        )
        slow = causal_ema(raw, tau_samples=slow_tau)
        if variant == "lowpass":
            return slow, None
        fast_tau = (
            float(branch_config["midband_fast_ema_min"])
            * 60.0
            / sample_period_sec
        )
        return causal_ema(raw, tau_samples=fast_tau) - slow, None
    if channel == "heater_power":
        return np.asarray(columns["主加热功率"], dtype=np.float64).copy(), None
    raise ValueError(f"UNKNOWN_CHANNEL:{channel}:{variant}")
