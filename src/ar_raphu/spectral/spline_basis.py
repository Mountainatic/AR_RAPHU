"""Train-only centered cubic B-spline bases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


def clamped_knots(lower: float, upper: float, n_basis: int, degree: int) -> np.ndarray:
    if n_basis <= degree:
        raise ValueError("n_basis must exceed degree.")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("Basis bounds must be finite and increasing.")
    interior_count = n_basis - degree - 1
    interior = (
        np.linspace(lower, upper, interior_count + 2, dtype=np.float64)[1:-1]
        if interior_count
        else np.empty(0, dtype=np.float64)
    )
    return np.concatenate(
        [
            np.repeat(lower, degree + 1),
            interior,
            np.repeat(upper, degree + 1),
        ]
    )


def evaluate_basis(
    values: np.ndarray, knots: np.ndarray, degree: int
) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), knots[degree], knots[-degree - 1])
    return BSpline.design_matrix(
        clipped, knots, degree, extrapolate=False
    ).toarray()


@dataclass(frozen=True, slots=True)
class CenteredSplineBasis:
    knots: np.ndarray
    degree: int
    train_mean: np.ndarray
    lower: float
    upper: float

    @classmethod
    def fit(
        cls,
        train_values: np.ndarray,
        *,
        n_basis: int,
        degree: int = 3,
        quantiles: tuple[float, float] = (0.01, 0.99),
    ) -> "CenteredSplineBasis":
        values = np.asarray(train_values, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Training basis values contain non-finite entries.")
        lower, upper = np.quantile(values, quantiles)
        if lower == upper:
            upper = lower + 1.0
        knots = clamped_knots(float(lower), float(upper), n_basis, degree)
        raw = evaluate_basis(values, knots, degree)
        return cls(
            knots=knots,
            degree=degree,
            train_mean=raw.mean(axis=0),
            lower=float(lower),
            upper=float(upper),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        return evaluate_basis(values, self.knots, self.degree) - self.train_mean
