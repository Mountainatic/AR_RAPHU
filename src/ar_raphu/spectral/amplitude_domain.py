"""Explicit train-fitted amplitude domains with no silent clipping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class AmplitudeOutOfDomainError(ValueError):
    """Raised when a strict structural transform encounters an OOD value."""


@dataclass(frozen=True, slots=True)
class AmplitudeDomain:
    fit_lower: float
    fit_upper: float
    core_lower: float
    core_upper: float
    padding_fraction: float

    @classmethod
    def fit(
        cls,
        train_values: np.ndarray,
        *,
        padding_fraction: float = 0.10,
        core_quantiles: tuple[float, float] = (0.01, 0.99),
    ) -> "AmplitudeDomain":
        values = np.asarray(train_values, dtype=np.float64).reshape(-1)
        if not values.size or not np.isfinite(values).all():
            raise ValueError("Training amplitudes must be finite and non-empty.")
        minimum = float(values.min())
        maximum = float(values.max())
        span = maximum - minimum
        if span <= 0.0:
            span = 1.0
        core_lower, core_upper = np.quantile(values, core_quantiles)
        return cls(
            fit_lower=minimum - padding_fraction * span,
            fit_upper=maximum + padding_fraction * span,
            core_lower=float(core_lower),
            core_upper=float(core_upper),
            padding_fraction=float(padding_fraction),
        )

    def in_domain_mask(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return (
            np.isfinite(array)
            & (array >= self.fit_lower)
            & (array <= self.fit_upper)
        )

    def ood_rate(self, values: np.ndarray) -> float:
        mask = self.in_domain_mask(values)
        return float(1.0 - np.mean(mask))
