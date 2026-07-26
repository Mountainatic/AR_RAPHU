"""Train-only centered cubic B-spline bases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from .amplitude_domain import (
    AmplitudeDomain,
    AmplitudeOutOfDomainError,
)


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
    array = np.asarray(values, dtype=np.float64)
    lower = float(knots[degree])
    upper = float(knots[-degree - 1])
    if not np.isfinite(array).all() or np.any(array < lower) or np.any(array > upper):
        raise AmplitudeOutOfDomainError(
            f"Basis values must lie in [{lower}, {upper}]."
        )
    return BSpline.design_matrix(
        array, knots, degree, extrapolate=False
    ).toarray()


def legacy_evaluate_basis(
    values: np.ndarray, knots: np.ndarray, degree: int
) -> np.ndarray:
    """Frozen clipping behavior, allowed only for old-protocol audits."""

    clipped = np.clip(
        np.asarray(values, dtype=np.float64),
        knots[degree],
        knots[-degree - 1],
    )
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
    domain: AmplitudeDomain

    @classmethod
    def fit(
        cls,
        train_values: np.ndarray,
        *,
        n_basis: int,
        degree: int = 3,
        domain: AmplitudeDomain | None = None,
        quantiles: tuple[float, float] = (0.01, 0.99),
    ) -> "CenteredSplineBasis":
        values = np.asarray(train_values, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("Training basis values contain non-finite entries.")
        if domain is None:
            legacy_lower, legacy_upper = np.quantile(values, quantiles)
            if legacy_lower == legacy_upper:
                legacy_upper = legacy_lower + 1.0
            fitted_domain = AmplitudeDomain(
                fit_lower=float(legacy_lower),
                fit_upper=float(legacy_upper),
                core_lower=float(legacy_lower),
                core_upper=float(legacy_upper),
                padding_fraction=0.0,
            )
        else:
            fitted_domain = domain
        lower = fitted_domain.fit_lower
        upper = fitted_domain.fit_upper
        knots = clamped_knots(float(lower), float(upper), n_basis, degree)
        raw = (
            evaluate_basis(values, knots, degree)
            if domain is not None
            else legacy_evaluate_basis(values, knots, degree)
        )
        return cls(
            knots=knots,
            degree=degree,
            train_mean=raw.mean(axis=0),
            lower=float(lower),
            upper=float(upper),
            domain=fitted_domain,
        )

    def transform(self, values: np.ndarray, *, strict: bool = True) -> np.ndarray:
        basis, mask = self.transform_with_mask(values)
        if strict and not bool(np.all(mask)):
            count = int(np.size(mask) - np.count_nonzero(mask))
            raise AmplitudeOutOfDomainError(
                f"{count} amplitude values are outside the fitted domain."
            )
        return basis

    def transform_with_mask(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(values, dtype=np.float64)
        flat = array.reshape(-1)
        mask = self.domain.in_domain_mask(flat)
        result = np.full(
            (len(flat), len(self.train_mean)),
            np.nan,
            dtype=np.float64,
        )
        if np.any(mask):
            result[mask] = (
                evaluate_basis(flat[mask], self.knots, self.degree)
                - self.train_mean
            )
        return result.reshape(array.shape + (len(self.train_mean),)), mask.reshape(
            array.shape
        )

    def legacy_transform_for_audit(self, values: np.ndarray) -> np.ndarray:
        return (
            legacy_evaluate_basis(values, self.knots, self.degree)
            - self.train_mean
        )
