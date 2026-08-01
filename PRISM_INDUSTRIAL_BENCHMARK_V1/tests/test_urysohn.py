from __future__ import annotations

import numpy as np

from prism_benchmark.urysohn import NaturalCubicBasis, fit_full, fit_rank_als, tensor_design


def test_natural_cubic_basis_has_exact_centered_linear_first_column() -> None:
    values = np.linspace(-2.0, 3.0, 101)
    basis = NaturalCubicBasis.fit(values, requested_dimension=6)
    transformed = basis.transform(values)
    expected = (values - values.mean()) / values.std()
    np.testing.assert_allclose(transformed[:, 0], expected, atol=1e-12)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
    assert transformed.shape[1] <= 6


def test_linear_kernel_is_exact_subspace_of_full_design() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(200, 4))
    basis = NaturalCubicBasis.fit(values, requested_dimension=5)
    phi = tensor_design(values, basis)
    linear_design = phi[:, :, 0]
    full_design = phi.reshape(len(phi), -1)
    np.testing.assert_allclose(full_design[:, :: basis.dimension], linear_design)


def test_rank_als_recovers_rank_one_surface() -> None:
    rng = np.random.default_rng(5)
    phi = rng.normal(size=(400, 4, 3))
    theta = np.outer(np.array([1.0, -0.5, 0.2, 0.8]), np.array([0.7, -1.2, 0.3]))
    target = np.einsum("tbx,bx->t", phi, theta) + 0.4
    prediction, fitted_theta, certificate = fit_rank_als(
        phi,
        target,
        phi,
        rank=1,
        lambda_0=1e-10,
        lambda_tau=0.0,
        lambda_x=0.0,
        max_iterations=100,
        tolerance=1e-10,
        seed=1,
    )
    np.testing.assert_allclose(prediction, target, atol=1e-5)
    assert np.linalg.matrix_rank(fitted_theta, tol=1e-8) == 1
    assert certificate["realized_rank"] == 1


def test_full_solver_reports_finite_certificate() -> None:
    rng = np.random.default_rng(8)
    phi = rng.normal(size=(100, 3, 2))
    target = rng.normal(size=100)
    prediction, theta, certificate = fit_full(phi, target, phi, 1e-3, 1e-3, 1e-3)
    assert prediction.shape == (100,)
    assert theta.shape == (3, 2)
    assert certificate["relative_kkt"] <= 1e-8
    assert np.isfinite(certificate["condition_number"])
