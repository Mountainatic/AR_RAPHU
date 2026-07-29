from __future__ import annotations

import numpy as np
import pytest
import torch

from ar_raphu.orss.augmented import AugmentedRegularizedOperator
from ar_raphu.orss.dr_basis import (
    generalized_dr_basis,
    tensor_from_dr,
    tensor_to_dr,
)
from ar_raphu.orss.krylov import lsqr
from ar_raphu.orss.mixed_precision import iterative_refinement
from ar_raphu.orss.operator import (
    BranchCache,
    UrysohnLinearOperator,
    build_urysohn_operator,
)
from ar_raphu.orss.penalties import PenaltyWeights, SeparablePenalty
from ar_raphu.orss.reduced_basis import ParametricReducedBasis
from ar_raphu.orss.sweep import (
    candidate_grid,
    diagonal_spectral_normalization,
    reduced_sweep,
)
from ar_raphu.spectral.design import (
    build_ar_nuisance_design,
    build_spectral_design,
)


def _random_operator(dtype: torch.dtype = torch.float64) -> UrysohnLinearOperator:
    generator = torch.Generator().manual_seed(41)
    branches = [
        BranchCache(
            amplitude=torch.randn(40, 5, 4, generator=generator, dtype=dtype),
            lag_basis=torch.randn(5, 3, generator=generator, dtype=dtype),
            out_of_domain_fraction=0.0,
        )
        for _ in range(3)
    ]
    return UrysohnLinearOperator(branches, chunk_time=11)


def test_operator_forward_and_adjoint_match_dense() -> None:
    operator = _random_operator()
    theta = torch.linspace(-1.0, 1.0, operator.dimension, dtype=torch.float64)
    residual = torch.linspace(0.5, -0.7, operator.observations, dtype=torch.float64)
    dense = operator.dense_design()
    torch.testing.assert_close(
        operator.forward(theta), dense @ theta, rtol=1.0e-12, atol=1.0e-12
    )
    torch.testing.assert_close(
        operator.adjoint(residual),
        dense.T @ residual,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    left = torch.dot(operator.forward(theta), residual)
    right = torch.dot(theta, operator.adjoint(residual))
    assert abs(float(left - right)) / (abs(float(left)) + abs(float(right))) <= 1e-12


def test_operator_builder_matches_frozen_dense_design() -> None:
    rng = np.random.default_rng(7)
    samples = 240
    x = rng.normal(size=(samples, 2)).cumsum(axis=0) * 0.02
    y = rng.normal(size=samples).cumsum() * 0.01
    targets = np.arange(80, 150, dtype=np.int64)
    parameters = dict(
        target_indices=targets,
        train_target_stop=170,
        horizon=5,
        lag_basis_count=4,
        amplitude_basis_count=5,
        continuation_scale_factor=1.0,
    )
    external = build_spectral_design(x, L_x=8, **parameters)
    ar = build_ar_nuisance_design(y, L_y=6, **parameters)
    dense = np.column_stack((external.matrix, ar))
    dense -= dense.mean(axis=0)
    operator, _ = build_urysohn_operator(
        x,
        y,
        target_indices=targets,
        train_target_stop=170,
        horizon=5,
        L_x=8,
        L_y=6,
        lag_basis_count=4,
        amplitude_basis_count=5,
        continuation_scale_coefficient=1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
        chunk_time=23,
    )
    np.testing.assert_allclose(
        operator.dense_design().numpy(), dense, rtol=1e-11, atol=1e-11
    )


def test_augmented_lsqr_matches_dense_solution() -> None:
    operator = _random_operator()
    penalty = SeparablePenalty(
        channels=operator.channels,
        m_tau=operator.m_tau,
        m_x=operator.m_x,
        device=operator.device,
        dtype=operator.dtype,
    )
    weights = PenaltyWeights(0.03, 0.05, 0.01)
    augmented = AugmentedRegularizedOperator(operator, penalty, weights)
    target = torch.linspace(-0.3, 0.4, operator.observations, dtype=torch.float64)
    target -= target.mean()
    rhs = augmented.augmented_rhs(target)
    result = lsqr(
        augmented,
        rhs,
        relative_tolerance=1.0e-10,
        maximum_iterations=800,
    )
    dense = operator.dense_design()
    gram = dense.T @ dense / len(dense)
    right = dense.T @ target / len(dense)
    identity = torch.eye(operator.dimension, dtype=torch.float64)
    lag = torch.stack(
        [
            penalty.lag_adjoint(penalty.lag_forward(identity[:, index]))
            for index in range(operator.dimension)
        ],
        dim=1,
    )
    amplitude = torch.stack(
        [
            penalty.amplitude_adjoint(
                penalty.amplitude_forward(identity[:, index])
            )
            for index in range(operator.dimension)
        ],
        dim=1,
    )
    reference = torch.linalg.solve(
        gram + weights.lag * lag + weights.amplitude * amplitude + weights.ridge * identity,
        right,
    )
    torch.testing.assert_close(
        result.coefficients, reference, rtol=1.0e-7, atol=1.0e-8
    )
    assert result.relative_kkt_residual <= 1.0e-8


def test_dr_roundtrip_and_diagonalization() -> None:
    dtype = torch.float64
    mass = torch.tensor(
        [[2.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 1.2]],
        dtype=dtype,
    )
    difference = torch.tensor([[1.0, -2.0, 1.0]], dtype=dtype)
    penalty = difference.T @ difference
    dr = generalized_dr_basis(penalty, mass)
    torch.testing.assert_close(
        dr.vectors.T @ dr.mass @ dr.vectors,
        torch.eye(3, dtype=dtype),
        rtol=1e-10,
        atol=1e-10,
    )
    transformed_penalty = dr.vectors.T @ penalty @ dr.vectors
    torch.testing.assert_close(
        transformed_penalty,
        torch.diag(dr.eigenvalues),
        rtol=1e-9,
        atol=1e-9,
    )
    coefficients = torch.arange(9, dtype=dtype).reshape(3, 3) / 7
    spectral = tensor_to_dr(coefficients, dr, dr)
    torch.testing.assert_close(
        tensor_from_dr(spectral, dr, dr),
        coefficients,
        rtol=1e-9,
        atol=1e-9,
    )


def test_reduced_basis_recovers_anchor_solution() -> None:
    operator = _random_operator()
    penalty = SeparablePenalty(
        channels=operator.channels,
        m_tau=operator.m_tau,
        m_x=operator.m_x,
        device=operator.device,
        dtype=operator.dtype,
    )
    target = torch.randn(operator.observations, dtype=torch.float64)
    target -= target.mean()
    rhs = operator.rhs(target)
    weights = PenaltyWeights(0.1, 0.2, 0.05)
    dense = operator.dense_design()
    gram = dense.T @ dense / len(dense)
    identity = torch.eye(operator.dimension, dtype=torch.float64)
    system = torch.stack(
        [
            operator.normal(identity[:, index])
            + penalty.normal(identity[:, index], weights)
            for index in range(operator.dimension)
        ],
        dim=1,
    )
    full = torch.linalg.solve(system, rhs)
    reduced = ParametricReducedBasis(operator, penalty, rhs)
    assert reduced.add_anchor(weights, full)
    candidate = reduced.solve(weights)
    torch.testing.assert_close(
        candidate.coefficients, full, rtol=1e-9, atol=1e-9
    )
    assert candidate.relative_residual <= 1.0e-9


def test_matrix_free_penalty_sweep_has_complete_grid() -> None:
    torch.manual_seed(17)
    observations, lag_count, m_tau, m_x = 48, 4, 3, 3
    branch = BranchCache(
        amplitude=torch.randn(
            observations, lag_count, m_x, dtype=torch.float64
        ),
        lag_basis=torch.randn(lag_count, m_tau, dtype=torch.float64),
        out_of_domain_fraction=0.0,
    )
    train = UrysohnLinearOperator([branch], chunk_time=13)
    validation = UrysohnLinearOperator(
        [branch], feature_mean=train.feature_mean, chunk_time=11
    )
    penalty = SeparablePenalty(
        channels=1,
        m_tau=m_tau,
        m_x=m_x,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    normalization = diagonal_spectral_normalization(train, penalty)
    candidates, grids = candidate_grid(
        normalization.lower,
        normalization.upper,
        positive_points=2,
    )
    target = torch.randn(observations, dtype=torch.float64)
    target -= target.mean()
    result = reduced_sweep(
        train,
        validation,
        target,
        target,
        target_mean=0.0,
        candidates=candidates,
        grids=grids,
        normalization=normalization,
        residual_tolerance=1.0e-5,
        maximum_dimension=9,
        krylov_tolerance=1.0e-10,
        maximum_iterations=500,
    )
    assert len(candidates) == 27
    assert len(result["candidate_rows"]) == 27
    assert result["maximum_relative_residual"] <= 1.0e-5


def test_iterative_refinement_reaches_fp64_kkt() -> None:
    matrix = torch.tensor(
        [[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64
    )
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    initial = torch.linalg.solve(
        matrix.float(), rhs.float()
    ).double()

    def correction(residual: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(matrix, residual)

    result = iterative_refinement(
        initial,
        apply_fp64=lambda vector: matrix @ vector,
        rhs_fp64=rhs,
        correction_solver=correction,
        maximum_steps=5,
    )
    assert result.converged
    assert result.relative_kkt_residual <= 1e-8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_real_cuda_dispatch() -> None:
    operator = _random_operator()
    branches = [
        BranchCache(
            amplitude=branch.amplitude.cuda(),
            lag_basis=branch.lag_basis.cuda(),
            out_of_domain_fraction=0.0,
        )
        for branch in operator.branches
    ]
    cuda_operator = UrysohnLinearOperator(branches)
    result = cuda_operator.forward(
        torch.ones(
            cuda_operator.dimension,
            device="cuda",
            dtype=torch.float64,
        )
    )
    assert result.is_cuda
