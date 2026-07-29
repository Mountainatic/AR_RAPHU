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
from ar_raphu.orss.krylov import lsqr, pcg_normal_batch
from ar_raphu.orss.mixed_precision import iterative_refinement
from ar_raphu.orss.operator import (
    BranchCache,
    UrysohnLinearOperator,
    build_operator_timeline_cache,
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


def test_operator_batched_forward_adjoint_and_normal_match_scalar() -> None:
    operator = _random_operator()
    generator = torch.Generator().manual_seed(512)
    coefficients = torch.randn(
        7, operator.dimension, generator=generator, dtype=torch.float64
    )
    residuals = torch.randn(
        operator.observations, 7, generator=generator, dtype=torch.float64
    )
    torch.testing.assert_close(
        operator.forward_batch(coefficients),
        torch.stack(
            [operator.forward(row) for row in coefficients], dim=1
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        operator.adjoint_batch(residuals),
        torch.stack(
            [operator.adjoint(residuals[:, index]) for index in range(7)]
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        operator.normal_batch(coefficients),
        torch.stack([operator.normal(row) for row in coefficients]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


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


def test_timeline_cache_reuse_is_operator_equivalent() -> None:
    rng = np.random.default_rng(709)
    x = rng.normal(size=(260, 3)).cumsum(axis=0) * 0.01
    y = rng.normal(size=260).cumsum() * 0.02
    common = dict(
        train_target_stop=180,
        L_x=10,
        L_y=7,
        lag_basis_count=5,
        amplitude_basis_count=6,
        continuation_scale_coefficient=1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    cache = build_operator_timeline_cache(x, y, **common)
    targets = np.arange(90, 170, dtype=np.int64)
    cached, _ = build_urysohn_operator(
        x,
        y,
        target_indices=targets,
        horizon=5,
        chunk_time=19,
        timeline_cache=cache,
        **common,
    )
    direct, _ = build_urysohn_operator(
        x,
        y,
        target_indices=targets,
        horizon=5,
        chunk_time=19,
        **common,
    )
    torch.testing.assert_close(
        cached.dense_design(),
        direct.dense_design(),
        rtol=1.0e-12,
        atol=1.0e-12,
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


def test_reduced_basis_batched_solve_matches_scalar_systems() -> None:
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
    dense = operator.dense_design()
    gram = dense.T @ dense / len(dense)
    identity = torch.eye(operator.dimension, dtype=torch.float64)
    anchors = (
        PenaltyWeights(0.03, 0.02, 0.01),
        PenaltyWeights(0.2, 0.1, 0.04),
    )
    reduced = ParametricReducedBasis(operator, penalty, rhs)
    for weights in anchors:
        system = torch.stack(
            [
                operator.normal(identity[:, index])
                + penalty.normal(identity[:, index], weights)
                for index in range(operator.dimension)
            ],
            dim=1,
        )
        assert reduced.add_anchor(weights, torch.linalg.solve(system, rhs))
    candidates = [
        PenaltyWeights(0.01, 0.02, 0.03),
        PenaltyWeights(0.04, 0.05, 0.06),
        PenaltyWeights(0.2, 0.1, 0.02),
    ]
    batch = reduced.solve_many(candidates)
    h0, lag, amplitude, right = reduced.project_operators()
    for candidate, weights in zip(batch, candidates, strict=True):
        system = (
            h0
            + weights.lag * lag
            + weights.amplitude * amplitude
            + weights.ridge
            * torch.eye(reduced.dimension, dtype=torch.float64)
        )
        coordinates = torch.linalg.solve(system, right)
        torch.testing.assert_close(
            candidate.reduced_coordinates,
            coordinates,
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        torch.testing.assert_close(
            candidate.coefficients,
            reduced.basis @ coordinates,
            rtol=1.0e-11,
            atol=1.0e-11,
        )


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


def test_batched_full_fallback_has_complete_certified_grid() -> None:
    torch.manual_seed(171)
    observations, lag_count, m_tau, m_x = 52, 4, 3, 3
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
        residual_tolerance=1.0e-7,
        maximum_dimension=8,
        krylov_tolerance=1.0e-11,
        maximum_iterations=750,
    )
    assert result["full_orss_fallback"]
    assert len(result["candidate_rows"]) == len(candidates) == 27
    assert result["maximum_relative_residual"] <= 1.0e-7
    assert result["profiler"]["batched_full_solve_count"] > 0


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


def test_batched_pcg_matches_independent_dense_systems() -> None:
    generator = torch.Generator().manual_seed(1221)
    batch, dimension = 9, 12
    factors = torch.randn(
        batch, dimension, dimension, generator=generator, dtype=torch.float64
    )
    systems = (
        factors.transpose(1, 2) @ factors
        + 0.5 * torch.eye(dimension, dtype=torch.float64)[None, ...]
    )
    rhs = torch.randn(
        batch, dimension, generator=generator, dtype=torch.float64
    )
    result = pcg_normal_batch(
        lambda rows: torch.einsum("kij,kj->ki", systems, rows),
        rhs,
        preconditioner_diagonal=torch.diagonal(
            systems, dim1=1, dim2=2
        ),
        relative_tolerance=1.0e-11,
        maximum_iterations=200,
    )
    reference = torch.linalg.solve(systems, rhs[..., None]).squeeze(-1)
    assert all(result.converged)
    torch.testing.assert_close(
        result.coefficients,
        reference,
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_penalty_batched_diagonal_matches_scalar() -> None:
    penalty = SeparablePenalty(
        channels=3,
        m_tau=5,
        m_x=6,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    weights = [
        PenaltyWeights(0.01, 0.02, 0.03),
        PenaltyWeights(0.04, 0.05, 0.06),
        PenaltyWeights(0.2, 0.1, 0.02),
    ]
    torch.testing.assert_close(
        penalty.diagonal_batch(weights),
        torch.stack([penalty.diagonal(row) for row in weights]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_batched_operator_and_penalty_match_scalar() -> None:
    operator = _random_operator(torch.float32)
    branches = [
        BranchCache(
            amplitude=branch.amplitude.cuda(),
            lag_basis=branch.lag_basis.cuda(),
            out_of_domain_fraction=0.0,
        )
        for branch in operator.branches
    ]
    cuda_operator = UrysohnLinearOperator(branches, chunk_time=17)
    penalty = SeparablePenalty(
        channels=cuda_operator.channels,
        m_tau=cuda_operator.m_tau,
        m_x=cuda_operator.m_x,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cuda").manual_seed(911)
    coefficients = torch.randn(
        19,
        cuda_operator.dimension,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    weights = [
        PenaltyWeights(
            1.0e-4 * (index + 1),
            2.0e-4 * (index + 1),
            3.0e-4 * (index + 1),
        )
        for index in range(len(coefficients))
    ]
    torch.testing.assert_close(
        cuda_operator.normal_batch(coefficients),
        torch.stack(
            [cuda_operator.normal(row) for row in coefficients]
        ),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        penalty.normal_batch(coefficients, weights),
        torch.stack(
            [
                penalty.normal(row, weight)
                for row, weight in zip(
                    coefficients, weights, strict=True
                )
            ]
        ),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
