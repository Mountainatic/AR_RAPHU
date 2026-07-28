from __future__ import annotations

import numpy as np
import pytest

from ar_raphu.cz_real.linear import TrainScaler, target_indices, window_designs
from ar_raphu.cz_real.audit import fit_exact_zero
from ar_raphu.cz_real.protocol import (
    FurnaceBLockedError,
    build_development_folds,
    confirmation_interval,
    load_furnace_b,
    purge_gap,
)
from ar_raphu.spectral.amplitude_domain import AmplitudeDomain
from ar_raphu.spectral.spline_basis import CenteredSplineBasis


def test_frozen_split_indices_and_purge() -> None:
    folds = build_development_folds(L_x=32, L_y=32)
    assert [fold.nominal_train_stop for fold in folds] == [
        8041,
        10051,
        12061,
        14072,
    ]
    assert [fold.validation_stop for fold in folds] == [
        10051,
        12061,
        14072,
        16082,
    ]
    assert purge_gap(L_x=32, L_y=32, h_max=60) == 91
    assert all(
        fold.effective_train_stop == fold.nominal_train_stop - 91
        for fold in folds
    )
    assert confirmation_interval() == (16082, 20103)


def test_furnace_b_is_rejected_before_r7_without_touching_path() -> None:
    with pytest.raises(FurnaceBLockedError):
        load_furnace_b(
            "/path/does/not/need/to/exist.xlsx",
            protocol_frozen=False,
            stage="R1",
        )
    with pytest.raises(FurnaceBLockedError):
        load_furnace_b(
            "/path/does/not/need/to/exist.xlsx",
            protocol_frozen=True,
            stage="R6",
        )


def test_direct_windows_stop_at_prediction_origin() -> None:
    samples = 200
    x = np.column_stack(
        [np.arange(samples, dtype=np.float64) + offset for offset in range(5)]
    )
    y = np.arange(samples, dtype=np.float64)
    scaler = TrainScaler.fit(x, y, 100)
    targets = target_indices(start=120, stop=140, horizon=15, max_history=32)
    x_design, y_design = window_designs(
        x,
        y,
        targets=targets,
        horizon=15,
        L_x=32,
        L_y=16,
        scaler=scaler,
    )
    assert x_design.shape == (20, 32 * 5)
    assert y_design.shape == (20, 16)
    first_origin = int(targets[0] - 15)
    recovered_current_y = y_design[0, 0] * scaler.y_scale + scaler.y_mean
    assert recovered_current_y == y[first_origin]
    assert first_origin < targets[0]


def test_v41_bounded_c1_continuation_is_interior_exact_and_bounded() -> None:
    from scipy.interpolate import BSpline

    train = np.linspace(-2.0, 3.0, 401)
    domain = AmplitudeDomain.fit(train, padding_fraction=0.10)
    basis = CenteredSplineBasis.fit(
        train,
        n_basis=16,
        degree=3,
        domain=domain,
    )
    interior = np.linspace(domain.fit_lower, domain.fit_upper, 101)
    strict = basis.transform(interior)
    continued, inside = basis.bounded_c1_transform(interior, scale_factor=1.0)
    np.testing.assert_allclose(continued, strict, rtol=0.0, atol=0.0)
    assert inside.all()

    extreme_basis, extreme_inside = basis.bounded_c1_transform(
        np.array([-1.0e12, 1.0e12]),
        scale_factor=1.0,
    )
    assert np.isfinite(extreme_basis).all()
    assert not extreme_inside.any()

    epsilon = 1.0e-10
    left_derivatives = basis.bounded_c1_derivative(
        np.array([domain.fit_lower - epsilon, domain.fit_lower + epsilon]),
        scale_factor=1.0,
    )
    expected_derivative = BSpline(
        basis.knots,
        np.eye(len(basis.train_mean), dtype=np.float64),
        basis.degree,
        extrapolate=False,
    )(domain.fit_lower, nu=1)
    np.testing.assert_allclose(
        left_derivatives,
        np.vstack((expected_derivative, expected_derivative)),
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_exact_zero_rescue_certifies_collinear_design() -> None:
    rng = np.random.default_rng(42)
    primitive = rng.normal(size=(300, 8))
    matrix = np.column_stack((primitive, primitive[:, :3], primitive[:, 0]))
    target = primitive @ np.arange(1.0, 9.0) + 0.01 * rng.normal(size=300)
    fit = fit_exact_zero(matrix, target)
    assert fit.relative_kkt_residual <= 1.0e-8
    assert fit.solver_stage == "DIAGONAL_EQUILIBRATION_CHOLESKY_REFINEMENT"
