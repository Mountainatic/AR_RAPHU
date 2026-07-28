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

    def boundary_knot_spacings(self) -> tuple[float, float]:
        """Return train-frozen adjacent knot spacings at both boundaries."""

        unique_knots = np.unique(self.knots)
        if len(unique_knots) < 2:
            raise ValueError("Spline basis has no positive knot spacing.")
        left = float(unique_knots[1] - unique_knots[0])
        right = float(unique_knots[-1] - unique_knots[-2])
        if left <= 0.0 or right <= 0.0:
            raise ValueError("Boundary knot spacings must be positive.")
        return left, right

    def bounded_c1_transform(
        self,
        values: np.ndarray,
        *,
        scale_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the v4.1 bounded-C1 tanh continuation.

        Each continuation scale is ``scale_factor`` times its adjacent
        train-frozen boundary knot spacing. Interior values are exactly the
        strict spline transform. The returned mask is true inside train
        support and false where continuation was used.
        """

        if not np.isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError("Continuation scale_factor must be positive.")
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("Continuation inputs must be finite.")
        flat = array.reshape(-1)
        inside = self.domain.in_domain_mask(flat)
        result = np.empty((len(flat), len(self.train_mean)), dtype=np.float64)
        if np.any(inside):
            result[inside] = (
                evaluate_basis(flat[inside], self.knots, self.degree)
                - self.train_mean
            )

        vector_spline = BSpline(
            self.knots,
            np.eye(len(self.train_mean), dtype=np.float64),
            self.degree,
            extrapolate=False,
        )
        left_spacing, right_spacing = self.boundary_knot_spacings()
        rho_left = scale_factor * left_spacing
        rho_right = scale_factor * right_spacing
        left_value = (
            np.asarray(vector_spline(self.lower), dtype=np.float64)
            - self.train_mean
        )
        right_value = (
            np.asarray(vector_spline(self.upper), dtype=np.float64)
            - self.train_mean
        )
        left_derivative = np.asarray(
            vector_spline(self.lower, nu=1), dtype=np.float64
        )
        right_derivative = np.asarray(
            vector_spline(self.upper, nu=1), dtype=np.float64
        )

        left = flat < self.lower
        if np.any(left):
            distance = ((flat[left] - self.lower) / rho_left)[:, None]
            result[left] = (
                left_value[None, :]
                + rho_left * left_derivative[None, :] * np.tanh(distance)
            )
        right = flat > self.upper
        if np.any(right):
            distance = ((flat[right] - self.upper) / rho_right)[:, None]
            result[right] = (
                right_value[None, :]
                + rho_right * right_derivative[None, :] * np.tanh(distance)
            )
        return result.reshape(array.shape + (len(self.train_mean),)), inside.reshape(
            array.shape
        )

    def bounded_c1_derivative(
        self,
        values: np.ndarray,
        *,
        scale_factor: float,
    ) -> np.ndarray:
        """Derivative of :meth:`bounded_c1_transform` with respect to input."""

        if not np.isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError("Continuation scale_factor must be positive.")
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("Continuation inputs must be finite.")
        flat = array.reshape(-1)
        result = np.empty((len(flat), len(self.train_mean)), dtype=np.float64)
        vector_spline = BSpline(
            self.knots,
            np.eye(len(self.train_mean), dtype=np.float64),
            self.degree,
            extrapolate=False,
        )
        left_spacing, right_spacing = self.boundary_knot_spacings()
        rho_left = scale_factor * left_spacing
        rho_right = scale_factor * right_spacing
        left_derivative = np.asarray(
            vector_spline(self.lower, nu=1), dtype=np.float64
        )
        right_derivative = np.asarray(
            vector_spline(self.upper, nu=1), dtype=np.float64
        )
        inside = self.domain.in_domain_mask(flat)
        if np.any(inside):
            result[inside] = vector_spline(flat[inside], nu=1)
        left = flat < self.lower
        if np.any(left):
            normalized = (flat[left] - self.lower) / rho_left
            multiplier = 1.0 - np.tanh(normalized) ** 2
            result[left] = multiplier[:, None] * left_derivative[None, :]
        right = flat > self.upper
        if np.any(right):
            normalized = (flat[right] - self.upper) / rho_right
            multiplier = 1.0 - np.tanh(normalized) ** 2
            result[right] = multiplier[:, None] * right_derivative[None, :]
        return result.reshape(array.shape + (len(self.train_mean),))

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
