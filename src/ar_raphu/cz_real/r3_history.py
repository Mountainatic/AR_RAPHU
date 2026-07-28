"""R3-A native history search with a shared anchored spectral representation."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import time

import numpy as np
import scipy.linalg

from ar_raphu.spectral.amplitude_domain import AmplitudeDomain
from ar_raphu.spectral.design import build_ar_nuisance_design, build_spectral_design
from ar_raphu.spectral.penalties import tensor_penalty
from ar_raphu.spectral.penalty_interval import (
    expand_penalty_interval,
    positive_generalized_eigenvalues,
)
from ar_raphu.spectral.spline_basis import (
    CenteredSplineBasis,
    clamped_knots,
    evaluate_basis,
)

from .linear import target_indices
from .protocol import DIRECT_HORIZONS, DevelopmentFold, build_development_folds


@dataclass(slots=True)
class FoldSystem:
    fold: int
    gram: np.ndarray
    rhs: np.ndarray
    normalized_penalties: tuple[np.ndarray, np.ndarray, np.ndarray]
    normalized_penalty_values: tuple[np.ndarray, np.ndarray, np.ndarray]
    feature_mean: np.ndarray
    target_mean: float
    validation_matrix: np.ndarray
    validation_target: np.ndarray


def _tensor_penalty_components(
    lag_gram: np.ndarray,
    amplitude_grams: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        tensor_penalty(
            lag_gram,
            amplitude_grams,
            lag_smoothness=1.0,
            amplitude_smoothness=0.0,
            ridge_weight=0.0,
        ),
        tensor_penalty(
            lag_gram,
            amplitude_grams,
            lag_smoothness=0.0,
            amplitude_smoothness=1.0,
            ridge_weight=0.0,
        ),
        tensor_penalty(
            lag_gram,
            amplitude_grams,
            lag_smoothness=0.0,
            amplitude_smoothness=0.0,
            ridge_weight=1.0,
        ),
    )


def _ar_penalty_components(
    y: np.ndarray,
    *,
    train_stop: int,
    L_y: int,
    M_tau: int,
    M_x: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lag_knots = clamped_knots(0.0, float(L_y - 1), M_tau, 3)
    lag_basis = evaluate_basis(np.arange(L_y), lag_knots, 3)
    lag_gram = lag_basis.T @ lag_basis / L_y
    y_train = np.asarray(y[:train_stop], dtype=np.float64)
    domain = AmplitudeDomain.fit(y_train, padding_fraction=0.10)
    amplitude = CenteredSplineBasis.fit(
        y_train,
        n_basis=M_x,
        degree=3,
        domain=domain,
    )
    evaluated = amplitude.transform(y_train)
    amplitude_gram = evaluated.T @ evaluated / len(evaluated)
    return _tensor_penalty_components(lag_gram, [amplitude_gram])


def _normalize_penalty(
    penalty: np.ndarray,
    gram: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = positive_generalized_eigenvalues(penalty, gram)
    if not len(values):
        raise RuntimeError("Penalty normalization has no positive generalized modes.")
    scale = float(np.median(values))
    normalized = penalty / scale
    return normalized, values / scale, scale


def _automatic_bounds(values: np.ndarray) -> tuple[float, float]:
    lower = (1.0 / 0.999 - 1.0) / float(np.max(values))
    upper = (1.0 / 0.001 - 1.0) / float(np.min(values))
    return max(lower, np.finfo(np.float64).tiny), upper


def _solve_system(
    gram: np.ndarray,
    rhs: np.ndarray,
    scientific_penalty: np.ndarray,
) -> tuple[np.ndarray, float, float, int]:
    """Solve in original coordinates; jitter is numerical and separately logged."""

    system = 0.5 * (gram + gram.T) + scientific_penalty
    scale = np.sqrt(np.maximum(np.diag(system), np.finfo(np.float64).eps))
    equilibrated = system / np.outer(scale, scale)
    equilibrated_rhs = rhs / scale
    best: tuple[np.ndarray, float, float, int] | None = None
    for jitter in (0.0, 1.0e-14, 1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10):
        try:
            factor = scipy.linalg.cho_factor(
                equilibrated + jitter * np.eye(len(equilibrated)),
                lower=True,
                check_finite=True,
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        coefficients = scipy.linalg.cho_solve(
            factor, equilibrated_rhs, check_finite=False
        ) / scale
        steps = 0
        for step in range(6):
            residual = rhs - system @ coefficients
            kkt = float(
                np.linalg.norm(residual)
                / max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
            )
            if kkt <= 1.0e-8 or step == 5:
                break
            correction = scipy.linalg.cho_solve(
                factor, residual / scale, check_finite=False
            ) / scale
            coefficients += correction
            steps += 1
        if best is None or kkt < best[1]:
            best = coefficients, kkt, jitter, steps
        if kkt <= 1.0e-8:
            break
    if best is None:
        raise np.linalg.LinAlgError("Penalty solver rescue produced no solution.")
    return best


def _build_fold_system(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fold: DevelopmentFold,
    horizon: int,
    L_x: int,
    L_y: int,
    M_tau: int,
    M_x: int,
    continuation_scale_coefficient: float,
) -> FoldSystem:
    train_targets = target_indices(
        start=0,
        stop=fold.effective_train_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    validation_targets = target_indices(
        start=fold.validation_start,
        stop=fold.validation_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    train_x = build_spectral_design(
        x,
        target_indices=train_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_x=L_x,
        lag_basis_count=M_tau,
        amplitude_basis_count=M_x,
        continuation_scale_factor=continuation_scale_coefficient,
    )
    validation_x = build_spectral_design(
        x,
        target_indices=validation_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_x=L_x,
        lag_basis_count=M_tau,
        amplitude_basis_count=M_x,
        continuation_scale_factor=continuation_scale_coefficient,
    )
    train_ar = build_ar_nuisance_design(
        y,
        target_indices=train_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_y=L_y,
        lag_basis_count=M_tau,
        amplitude_basis_count=M_x,
        continuation_scale_factor=continuation_scale_coefficient,
    )
    validation_ar = build_ar_nuisance_design(
        y,
        target_indices=validation_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_y=L_y,
        lag_basis_count=M_tau,
        amplitude_basis_count=M_x,
        continuation_scale_factor=continuation_scale_coefficient,
    )
    train_matrix = np.column_stack((train_x.matrix, train_ar))
    validation_matrix = np.column_stack((validation_x.matrix, validation_ar))
    train_target = y[train_targets]
    feature_mean = train_matrix.mean(axis=0)
    target_mean = float(train_target.mean())
    centered_matrix = train_matrix - feature_mean
    centered_target = train_target - target_mean
    gram = centered_matrix.T @ centered_matrix / len(centered_matrix)
    rhs = centered_matrix.T @ centered_target / len(centered_matrix)
    external_penalties = _tensor_penalty_components(
        train_x.lag_gram,
        train_x.amplitude_grams,
    )
    ar_penalties = _ar_penalty_components(
        y,
        train_stop=fold.effective_train_stop,
        L_y=L_y,
        M_tau=M_tau,
        M_x=M_x,
    )
    penalties = tuple(
        scipy.linalg.block_diag(external, ar)
        for external, ar in zip(
            external_penalties, ar_penalties, strict=True
        )
    )
    normalized_rows = tuple(
        _normalize_penalty(penalty, gram) for penalty in penalties
    )
    return FoldSystem(
        fold=fold.fold,
        gram=gram,
        rhs=rhs,
        normalized_penalties=tuple(row[0] for row in normalized_rows),
        normalized_penalty_values=tuple(row[1] for row in normalized_rows),
        feature_mean=feature_mean,
        target_mean=target_mean,
        validation_matrix=validation_matrix,
        validation_target=y[validation_targets],
    )


def _candidate_rows(
    systems: list[FoldSystem],
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    names = ("lag", "amplitude", "ridge")
    for order, indices in enumerate(
        itertools.product(*(range(len(grid)) for grid in grids))
    ):
        weights = tuple(
            float(grids[axis][indices[axis]]) for axis in range(3)
        )
        fold_rows: list[dict[str, float | int]] = []
        for system in systems:
            scientific_penalty = sum(
                weight * component
                for weight, component in zip(
                    weights, system.normalized_penalties, strict=True
                )
            )
            coefficients, kkt, jitter, refinement = _solve_system(
                system.gram,
                system.rhs,
                scientific_penalty,
            )
            intercept = system.target_mean - float(
                system.feature_mean @ coefficients
            )
            prediction = system.validation_matrix @ coefficients + intercept
            mse = float(np.mean((prediction - system.validation_target) ** 2))
            fold_rows.append(
                {
                    "fold": system.fold,
                    "validation_MSE_mm2": mse,
                    "relative_kkt_residual": kkt,
                    "numerical_jitter_relative": jitter,
                    "iterative_refinement_steps": refinement,
                }
            )
        losses = np.array(
            [row["validation_MSE_mm2"] for row in fold_rows], dtype=np.float64
        )
        rows.append(
            {
                "configuration_order": order,
                "indices": {
                    name: int(indices[axis]) for axis, name in enumerate(names)
                },
                "scientific_penalty_weights": {
                    name: float(weights[axis]) for axis, name in enumerate(names)
                },
                "validation_MSE_mean": float(losses.mean()),
                "validation_MSE_SE": float(losses.std(ddof=1) / np.sqrt(len(losses))),
                "folds": fold_rows,
                "maximum_relative_kkt_residual": max(
                    float(row["relative_kkt_residual"]) for row in fold_rows
                ),
                "numerical_jitter_is_scientific_ridge": False,
            }
        )
    return rows


def _select_penalty(rows: list[dict[str, object]]) -> dict[str, object]:
    valid = [
        row
        for row in rows
        if float(row["maximum_relative_kkt_residual"]) <= 1.0e-8
    ]
    if not valid:
        raise RuntimeError("KKT_FAILED_AFTER_RESCUE")
    minimum = min(
        valid,
        key=lambda row: (
            float(row["validation_MSE_mean"]),
            int(row["configuration_order"]),
        ),
    )
    threshold = float(minimum["validation_MSE_mean"]) + float(
        minimum["validation_MSE_SE"]
    )
    eligible = [
        row for row in valid if float(row["validation_MSE_mean"]) <= threshold
    ]
    # Deterministic PB1 lexical tie after one-SE: stronger roughness control,
    # then less scientific ridge, then the predeclared configuration order.
    return max(
        eligible,
        key=lambda row: (
            float(row["scientific_penalty_weights"]["lag"])
            + float(row["scientific_penalty_weights"]["amplitude"]),
            -float(row["scientific_penalty_weights"]["ridge"]),
            -int(row["configuration_order"]),
        ),
    )


def run_history_candidate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    L_x: int,
    L_y: int,
    horizon: int,
    M_tau: int = 16,
    M_x: int = 32,
    continuation_scale_coefficient: float = 1.0,
    positive_grid_points: int = 7,
    maximum_edge_expansions: int = 2,
) -> dict[str, object]:
    started = time.perf_counter()
    folds = build_development_folds(
        L_x=L_x,
        L_y=L_y,
        h_max=max(DIRECT_HORIZONS),
    )
    systems = [
        _build_fold_system(
            x,
            y,
            fold=fold,
            horizon=horizon,
            L_x=L_x,
            L_y=L_y,
            M_tau=M_tau,
            M_x=M_x,
            continuation_scale_coefficient=continuation_scale_coefficient,
        )
        for fold in folds
    ]
    names = ("lag", "amplitude", "ridge")
    bounds_by_axis = [
        [
            _automatic_bounds(system.normalized_penalty_values[axis])
            for system in systems
        ]
        for axis in range(3)
    ]
    lowers = [min(item[0] for item in bounds) for bounds in bounds_by_axis]
    uppers = [max(item[1] for item in bounds) for bounds in bounds_by_axis]
    interval_history: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    status = "PENALTY_INTERVAL_NOT_CERTIFIED"
    for expansion in range(maximum_edge_expansions + 1):
        grids = tuple(
            np.concatenate(
                (
                    np.zeros(1, dtype=np.float64),
                    np.geomspace(lowers[axis], uppers[axis], positive_grid_points),
                )
            )
            for axis in range(3)
        )
        rows = _candidate_rows(systems, grids)
        all_rows.extend(rows)
        selected = _select_penalty(rows)
        selected_indices = [
            int(selected["indices"][name]) for name in names
        ]
        minimum = min(rows, key=lambda row: float(row["validation_MSE_mean"]))
        boundaries: list[dict[str, int | str]] = []
        for axis, index in enumerate(selected_indices):
            if index == len(grids[axis]) - 1:
                boundaries.append({"axis": axis, "side": "upper"})
            elif index == 1:
                zero_best = min(
                    (
                        float(row["validation_MSE_mean"])
                        for row in rows
                        if int(row["indices"][names[axis]]) == 0
                    ),
                    default=float("inf"),
                )
                if zero_best > float(minimum["validation_MSE_mean"]) + float(
                    minimum["validation_MSE_SE"]
                ):
                    boundaries.append({"axis": axis, "side": "lower"})
        interval_history.append(
            {
                "expansion_round": expansion,
                "lower_by_axis": {
                    name: lowers[axis] for axis, name in enumerate(names)
                },
                "upper_by_axis": {
                    name: uppers[axis] for axis, name in enumerate(names)
                },
                "selected_indices": {
                    name: selected_indices[axis]
                    for axis, name in enumerate(names)
                },
                "selected_weights": selected["scientific_penalty_weights"],
                "boundaries": boundaries,
            }
        )
        if not boundaries:
            status = "PENALTY_INTERVAL_CERTIFIED"
            break
        if expansion == maximum_edge_expansions:
            break
        for boundary in boundaries:
            axis = int(boundary["axis"])
            width = np.log(uppers[axis]) - np.log(lowers[axis])
            if boundary["side"] == "upper":
                uppers[axis] = float(np.exp(np.log(uppers[axis]) + width))
            else:
                lowers[axis] = float(np.exp(np.log(lowers[axis]) - width))
    if selected is None:
        raise AssertionError("Penalty search produced no selected row.")
    return {
        "schema": "CZ_R3A_HISTORY_CANDIDATE_V1",
        "status": (
            "COMPLETED"
            if status == "PENALTY_INTERVAL_CERTIFIED"
            and float(selected["maximum_relative_kkt_residual"]) <= 1.0e-8
            else "FAILED"
        ),
        "history": {"L_x": L_x, "L_y": L_y},
        "horizon_samples": horizon,
        "anchor": {
            "M_tau": M_tau,
            "M_x": M_x,
            "CONTINUATION_SCALE_COEFFICIENT": continuation_scale_coefficient,
        },
        "penalty": {
            "exact_zero_endpoint": True,
            "positive_grid_points": positive_grid_points,
            "automatic_interval": True,
            "maximum_edge_expansions": maximum_edge_expansions,
            "independent_penalty_axes": list(names),
            "one_se_tie": (
                "STRONGER_LAG_PLUS_AMPLITUDE_THEN_SMALLER_RIDGE_"
                "THEN_CONFIGURATION_ORDER"
            ),
            "normalization": "POSITIVE_GENERALIZED_EIGENVALUE_MEDIAN_RELATIVE_TO_TRAIN_GRAM",
            "selected": selected,
            "interval_history": interval_history,
            "candidate_rows": all_rows,
            "status": status,
            "numerical_jitter_is_separate": True,
        },
        "validation_loss": selected["validation_MSE_mean"],
        "validation_se": selected["validation_MSE_SE"],
        "elapsed_seconds": time.perf_counter() - started,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_accessed": False,
    }


def history_complexity_key(L_x: int, L_y: int) -> tuple[int, int, int, int, int]:
    return L_x + L_y, L_x * L_y, max(L_x, L_y), L_x, L_y
