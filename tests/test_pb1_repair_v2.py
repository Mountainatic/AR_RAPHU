import numpy as np

from ar_raphu.datasets.base import DynamicDataset
from ar_raphu.datasets.pb1_protocol import apply_pb1_repair_v2_partition
from ar_raphu.spectral.pb1_development import (
    PB1PenaltyCandidate,
    _select_one_se,
)
from ar_raphu.spectral.pb1_repair import (
    iterative_refine_pb1,
    positive_lower_expansion_required,
    solve_pb1_system,
    zero_inclusive_penalty_grid,
)
from ar_raphu.spectral.pb1_selection import (
    H2HistoryScore,
    H2ResolutionScore,
    select_h2_history_one_se,
    select_h2_resolution_one_se,
)
from ar_raphu.spectral.penalty_interval import (
    LogPenaltyInterval,
    expand_penalty_interval,
)


def test_zero_penalty_candidate_is_exact() -> None:
    grid = zero_inclusive_penalty_grid(
        LogPenaltyInterval(1.0e-4, 1.0e2), 7
    )
    assert grid.shape == (8,)
    assert grid[0] == 0.0
    assert np.all(grid[1:] > 0.0)


def test_zero_penalty_does_not_expand_lower_boundary() -> None:
    assert not positive_lower_expansion_required(
        selected_index=0,
        axis_zero_best_loss=1.0,
        global_minimum_loss=1.0,
        global_minimum_se=0.01,
    )


def test_positive_lower_boundary_can_expand() -> None:
    assert positive_lower_expansion_required(
        selected_index=1,
        axis_zero_best_loss=1.2,
        global_minimum_loss=1.0,
        global_minimum_se=0.05,
    )
    assert not positive_lower_expansion_required(
        selected_index=1,
        axis_zero_best_loss=1.04,
        global_minimum_loss=1.0,
        global_minimum_se=0.05,
    )


def test_upper_boundary_expands_at_most_twice() -> None:
    interval = LogPenaltyInterval(1.0e-4, 1.0e2)
    interval = expand_penalty_interval(interval, boundary="upper")
    interval = expand_penalty_interval(interval, boundary="upper")
    assert interval.expansion_count == 2


def test_numerical_jitter_is_not_scientific_ridge() -> None:
    system = np.diag([2.0, 3.0])
    rhs = np.array([1.0, -1.0])
    result = solve_pb1_system(system, rhs)
    np.testing.assert_allclose(
        result.coefficients,
        np.linalg.solve(system, rhs),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert result.numerical_jitter > 0.0


def test_equilibrated_solver_matches_direct_well_conditioned() -> None:
    system = np.array([[3.0, 0.2], [0.2, 2.0]], dtype=np.float64)
    rhs = np.array([1.0, -0.5], dtype=np.float64)
    result = solve_pb1_system(system, rhs)
    np.testing.assert_allclose(
        result.coefficients, np.linalg.solve(system, rhs), atol=1.0e-13
    )
    assert result.relative_kkt_residual <= 1.0e-8


def test_iterative_refinement_reduces_original_coordinate_kkt() -> None:
    system = np.array([[3.0, 0.2], [0.2, 2.0]], dtype=np.float64)
    rhs = np.array([1.0, -0.5], dtype=np.float64)
    coefficients, before, after, steps = iterative_refine_pb1(
        system,
        rhs,
        np.zeros(2, dtype=np.float64),
    )
    assert steps >= 1
    assert after < before
    np.testing.assert_allclose(
        coefficients, np.linalg.solve(system, rhs), atol=1.0e-13
    )


def test_svd_fallback_handles_zero_penalty_semidefinite_case() -> None:
    system = np.diag([2.0, 0.0])
    rhs = np.array([4.0, 0.0])
    result = solve_pb1_system(system, rhs)
    assert result.solver_stage == "SVD_MINIMUM_NORM"
    assert result.solution_is_minimum_norm
    np.testing.assert_allclose(result.coefficients, [2.0, 0.0], atol=1.0e-14)
    assert result.relative_kkt_residual <= 1.0e-8


def test_svd_cutoff_tightens_until_original_coordinate_kkt_passes() -> None:
    matrix = np.diag([1.0, 1.0e-16, 0.0])
    rhs = np.array([0.0, 1.0e-16, 0.0])
    result = solve_pb1_system(matrix, rhs)
    assert result.solver_stage == "SVD_MINIMUM_NORM"
    assert result.converged is True
    assert result.relative_kkt_residual <= 1.0e-8
    np.testing.assert_allclose(matrix @ result.coefficients, rhs)


def test_solver_does_not_change_selected_penalties() -> None:
    gram = np.array([[2.0, 0.1], [0.1, 1.0]])
    rhs = np.array([1.0, 0.5])
    penalty = np.diag([0.0, 0.25])
    result = solve_pb1_system(gram + penalty, rhs)
    np.testing.assert_allclose(
        result.coefficients,
        np.linalg.solve(gram + penalty, rhs),
        atol=1.0e-13,
    )


def test_one_se_excludes_candidates_that_fail_frozen_kkt() -> None:
    def candidate(loss: float, converged: bool, order: int) -> PB1PenaltyCandidate:
        return PB1PenaltyCandidate(
            lag_weight=float(order),
            amplitude_weight=0.0,
            ridge_weight=0.0,
            validation_mse_mean=loss,
            validation_mse_by_group=(loss, loss),
            validation_mse_se=0.0,
            effective_df=float(order + 1),
            relative_kkt_residual=1.0e-12 if converged else 1.0e-4,
            numerical_jitter=0.0,
            configuration_order=order,
            index_lag=order,
            index_amplitude=0,
            index_ridge=0,
            coefficients=np.zeros(1),
            intercept=0.0,
            solver_diagnostics={"converged": converged},
        )

    selected = _select_one_se(
        [candidate(0.1, False, 0), candidate(0.2, True, 1)]
    )
    assert selected.validation_mse_mean == 0.2


def test_kkt_is_recomputed_in_original_coordinates() -> None:
    system = np.diag([1.0e-9, 1.0e9])
    rhs = np.array([1.0e-9, 1.0])
    result = solve_pb1_system(system, rhs)
    expected = np.linalg.norm(system @ result.coefficients - rhs) / np.linalg.norm(rhs)
    assert result.relative_kkt_residual == expected


def _single_record(dataset_id: str, length: int) -> DynamicDataset:
    return DynamicDataset(
        x=np.arange(length, dtype=np.float64)[:, None],
        y=np.arange(length, dtype=np.float64)[:, None],
        timestamps=None,
        split=np.full(length, "train", dtype=object),
        sequence_id=np.full(length, f"0000:{dataset_id}", dtype=object),
        label_mask=np.ones((length, 1), dtype=bool),
        quality_mask=np.ones((length, 2), dtype=bool),
        feature_names=("input",),
        target_names=("output",),
        metadata={"dataset_id": dataset_id, "test_records_exposed": False},
    )


def test_tanks_literature_partition_is_700_rest() -> None:
    dataset = _single_record("cascaded_tanks", 1024)
    config = {
        "schema_version": 7,
        "dataset": {
            "id": "cascaded_tanks",
            "development_split": {
                "status": "FROZEN_LITERATURE_SPLIT",
                "source": "Champneys2024_associated_code",
                "train_rows": [0, 700],
                "validation_rows": [700, "end"],
            },
        },
    }
    result = apply_pb1_repair_v2_partition(dataset, config)
    assert np.sum(result.split == "train") == 700
    assert np.sum(result.split == "validation") == 324
    assert result.metadata["official_test_access_count"] == 0


def test_silverbox_literature_partition_is_half_half() -> None:
    dataset = _single_record("silverbox", 101)
    config = {
        "schema_version": 7,
        "dataset": {
            "id": "silverbox",
            "development_split": {
                "status": "FROZEN_LITERATURE_SPLIT",
                "source": "Champneys2024_associated_code",
                "train_fraction": [0.0, 0.5],
                "validation_fraction": [0.5, 1.0],
            },
        },
    }
    result = apply_pb1_repair_v2_partition(dataset, config)
    assert np.sum(result.split == "train") == 50
    assert np.sum(result.split == "validation") == 51


def test_h2_history_one_se_uses_frozen_complexity_key() -> None:
    selected = select_h2_history_one_se(
        [
            H2HistoryScore(16, 8, 1.00, 0.05),
            H2HistoryScore(8, 8, 1.03, 0.02),
            H2HistoryScore(8, 4, 1.06, 0.01),
        ]
    )
    assert (selected.L_x, selected.L_y) == (8, 8)


def test_h2_resolution_requires_representation_and_lepski() -> None:
    selected = select_h2_resolution_one_se(
        [
            H2ResolutionScore("cubic_bspline", 16, 12, 1.0, 0.05, False, True),
            H2ResolutionScore("cubic_bspline", 16, 16, 1.01, 0.05, True, True),
            H2ResolutionScore("cubic_bspline", 32, 12, 0.99, 0.05, True, False),
            H2ResolutionScore("cubic_bspline", 32, 16, 1.0, 0.05, True, True),
        ]
    )
    assert (selected.lag_count, selected.amplitude_count) == (16, 16)
