"""CZ R3 development tasks backed by the matrix-free ORSS solver."""

from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
import torch

from ar_raphu.orss.operator import build_urysohn_operator
from ar_raphu.orss.penalties import PenaltyWeights, SeparablePenalty
from ar_raphu.orss.sweep import (
    PenaltyNormalization,
    aggregate_folds,
    candidate_grid,
    diagonal_spectral_normalization,
    reduced_sweep,
    select_one_se,
    solve_full,
)

from .linear import target_indices
from .protocol import DIRECT_HORIZONS, build_development_folds


def _fold_operators(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fold,
    horizon: int,
    L_x: int,
    L_y: int,
    M_tau: int,
    M_x: int,
    c_rho: float,
    device: torch.device,
    dtype: torch.dtype,
    chunk_time: int,
):
    train_indices = target_indices(
        start=0,
        stop=fold.effective_train_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    validation_indices = target_indices(
        start=fold.validation_start,
        stop=fold.validation_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    train_operator, state = build_urysohn_operator(
        x,
        y,
        target_indices=train_indices,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        lag_basis_count=M_tau,
        amplitude_basis_count=M_x,
        continuation_scale_coefficient=c_rho,
        device=device,
        dtype=dtype,
        chunk_time=chunk_time,
    )
    validation_operator, _ = build_urysohn_operator(
        x,
        y,
        target_indices=validation_indices,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        lag_basis_count=M_tau,
        amplitude_basis_count=M_x,
        continuation_scale_coefficient=c_rho,
        device=device,
        dtype=dtype,
        chunk_time=chunk_time,
        basis_state=state,
        feature_mean=train_operator.feature_mean,
    )
    target_mean = float(np.mean(y[train_indices]))
    train_target = torch.as_tensor(
        y[train_indices] - target_mean, device=device, dtype=dtype
    )
    validation_target = torch.as_tensor(
        y[validation_indices], device=device, dtype=dtype
    )
    return (
        train_operator,
        validation_operator,
        train_target,
        validation_target,
        target_mean,
    )


def _normalizations(
    x: np.ndarray,
    y: np.ndarray,
    *,
    folds,
    horizon: int,
    L_x: int,
    L_y: int,
    M_tau: int,
    M_x: int,
    c_rho: float,
    device: torch.device,
    dtype: torch.dtype,
    chunk_time: int,
) -> list[PenaltyNormalization]:
    rows: list[PenaltyNormalization] = []
    for fold in folds:
        train, validation, *_ = _fold_operators(
            x,
            y,
            fold=fold,
            horizon=horizon,
            L_x=L_x,
            L_y=L_y,
            M_tau=M_tau,
            M_x=M_x,
            c_rho=c_rho,
            device=device,
            dtype=dtype,
            chunk_time=chunk_time,
        )
        penalty = SeparablePenalty(
            channels=train.channels,
            m_tau=train.m_tau,
            m_x=train.m_x,
            device=device,
            dtype=dtype,
        )
        rows.append(diagonal_spectral_normalization(train, penalty))
        del train, validation, penalty
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _expanded_bounds(
    lower: list[float],
    upper: list[float],
    boundaries: list[tuple[int, str]],
) -> tuple[list[float], list[float]]:
    for axis, side in boundaries:
        width = np.log(upper[axis]) - np.log(lower[axis])
        if side == "upper":
            upper[axis] = float(np.exp(np.log(upper[axis]) + width))
        else:
            lower[axis] = float(np.exp(np.log(lower[axis]) - width))
    return lower, upper


def _boundary_requests(
    selection: dict[str, object],
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[tuple[int, str]]:
    selected = selection["selected"]["normalized_weights"]
    minimum = selection["minimum"]
    threshold = float(selection["threshold"])
    requests: list[tuple[int, str]] = []
    for axis, name in enumerate(("lag", "amplitude", "ridge")):
        value = float(selected[name])
        if value == float(grids[axis][-1]):
            requests.append((axis, "upper"))
        elif value == float(grids[axis][1]):
            # Expand downward only when zero on this axis is outside one-SE.
            zero_best = min(
                (
                    float(row["validation_MSE_mean"])
                    for row in selection["_all_rows"]
                    if float(row["normalized_weights"][name]) == 0.0
                ),
                default=float("inf"),
            )
            if zero_best > threshold and float(
                minimum["validation_MSE_mean"]
            ) < zero_best:
                requests.append((axis, "lower"))
    return requests


def run_development_task(
    x: np.ndarray,
    y: np.ndarray,
    *,
    horizon: int,
    L_x: int,
    L_y: int,
    M_tau: int,
    M_x: int,
    c_rho: float,
    device: torch.device,
    primary_dtype: torch.dtype,
    chunk_time: int,
    positive_grid_points: int,
    maximum_edge_expansions: int,
    rb_tolerance: float,
    rb_maximum_dimension: int,
    krylov_tolerance: float,
    krylov_maximum_iterations: int,
) -> dict[str, object]:
    started = time.perf_counter()
    folds = build_development_folds(
        L_x=L_x, L_y=L_y, h_max=max(DIRECT_HORIZONS)
    )
    normalizations = _normalizations(
        x,
        y,
        folds=folds,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        M_tau=M_tau,
        M_x=M_x,
        c_rho=c_rho,
        device=device,
        dtype=primary_dtype,
        chunk_time=chunk_time,
    )
    lower = [
        min(row.lower[axis] for row in normalizations) for axis in range(3)
    ]
    upper = [
        max(row.upper[axis] for row in normalizations) for axis in range(3)
    ]
    interval_history: list[dict[str, object]] = []
    final_folds: list[dict[str, object]] = []
    final_candidates: list[PenaltyWeights] = []
    final_selection: dict[str, object] | None = None
    interval_status = "PENALTY_INTERVAL_NOT_CERTIFIED"
    for expansion in range(maximum_edge_expansions + 1):
        candidates, grids = candidate_grid(
            lower,
            upper,
            positive_points=positive_grid_points,
        )
        fold_results: list[dict[str, object]] = []
        for fold, normalization in zip(folds, normalizations, strict=True):
            (
                train,
                validation,
                train_target,
                validation_target,
                target_mean,
            ) = _fold_operators(
                x,
                y,
                fold=fold,
                horizon=horizon,
                L_x=L_x,
                L_y=L_y,
                M_tau=M_tau,
                M_x=M_x,
                c_rho=c_rho,
                device=device,
                dtype=primary_dtype,
                chunk_time=chunk_time,
            )
            result = reduced_sweep(
                train,
                validation,
                train_target,
                validation_target,
                target_mean=target_mean,
                candidates=candidates,
                grids=grids,
                normalization=normalization,
                residual_tolerance=rb_tolerance,
                maximum_dimension=rb_maximum_dimension,
                krylov_tolerance=krylov_tolerance,
                maximum_iterations=krylov_maximum_iterations,
            )
            result["fold"] = fold.fold
            result["penalty_normalization"] = asdict(normalization)
            fold_results.append(result)
            del train, validation, train_target, validation_target
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows = aggregate_folds(fold_results, candidates)
        selection = select_one_se(rows)
        selection["_all_rows"] = rows
        requests = _boundary_requests(selection, grids)
        interval_history.append(
            {
                "expansion_round": expansion,
                "lower": dict(zip(("lag", "amplitude", "ridge"), lower)),
                "upper": dict(zip(("lag", "amplitude", "ridge"), upper)),
                "selected": selection["selected"],
                "boundary_requests": [
                    {"axis": ("lag", "amplitude", "ridge")[axis], "side": side}
                    for axis, side in requests
                ],
            }
        )
        final_folds = fold_results
        final_candidates = candidates
        final_selection = selection
        if not requests:
            interval_status = "PENALTY_INTERVAL_CERTIFIED"
            break
        if expansion < maximum_edge_expansions:
            lower, upper = _expanded_bounds(lower, upper, requests)
    if final_selection is None:
        raise AssertionError("Penalty sweep produced no selection.")
    all_rows = final_selection.pop("_all_rows")
    selected_weights = PenaltyWeights(
        **final_selection["selected"]["normalized_weights"]
    )

    # Refit the frozen candidate in FP64 on every fold and certify the KKT.
    certifications: list[dict[str, object]] = []
    for fold, normalization in zip(folds, normalizations, strict=True):
        (
            train,
            validation,
            train_target,
            validation_target,
            target_mean,
        ) = _fold_operators(
            x,
            y,
            fold=fold,
            horizon=horizon,
            L_x=L_x,
            L_y=L_y,
            M_tau=M_tau,
            M_x=M_x,
            c_rho=c_rho,
            device=device,
            dtype=torch.float64,
            chunk_time=chunk_time,
        )
        penalty = SeparablePenalty(
            channels=train.channels,
            m_tau=train.m_tau,
            m_x=train.m_x,
            device=device,
            dtype=torch.float64,
        )
        actual = normalization.actual(selected_weights)
        fitted = solve_full(
            train,
            train_target,
            penalty,
            actual,
            relative_tolerance=1.0e-10,
            maximum_iterations=max(krylov_maximum_iterations, 2500),
        )
        prediction = validation.forward(fitted.coefficients) + target_mean
        certifications.append(
            {
                "fold": fold.fold,
                "actual_weights": asdict(actual),
                "relative_kkt_residual": fitted.relative_kkt_residual,
                "iterations": fitted.iterations,
                "method": fitted.method,
                "validation_MSE_mm2": float(
                    torch.mean((prediction - validation_target) ** 2).item()
                ),
            }
        )
        del train, validation, train_target, validation_target, penalty
        if device.type == "cuda":
            torch.cuda.empty_cache()

    kkt_pass = all(
        float(row["relative_kkt_residual"]) <= 1.0e-8
        for row in certifications
    )
    rb_pass = all(bool(row["RB_RESIDUAL_CERTIFIED"]) for row in final_folds)
    return {
        "schema": "CZ_R3_ORSS_DEVELOPMENT_TASK_V1",
        "status": (
            "COMPLETED"
            if interval_status == "PENALTY_INTERVAL_CERTIFIED"
            and kkt_pass
            and rb_pass
            else "FAILED"
        ),
        "history": {"L_x": L_x, "L_y": L_y},
        "resolution": {"M_tau": M_tau, "M_x": M_x},
        "CONTINUATION_SCALE_COEFFICIENT": c_rho,
        "horizon": horizon,
        "penalty": {
            "normalization_method": (
                "MATRIX_FREE_DATA_NORMAL_DIAGONAL_GENERALIZED_SPECTRAL_PROXY"
            ),
            "exact_zero_endpoint": True,
            "positive_grid_points": positive_grid_points,
            "maximum_edge_expansions": maximum_edge_expansions,
            "interval_status": interval_status,
            "interval_history": interval_history,
            "selection": final_selection,
            "candidate_rows": all_rows,
        },
        "fold_solver_diagnostics": final_folds,
        "final_fp64_certifications": certifications,
        "validation_loss": float(
            final_selection["selected"]["validation_MSE_mean"]
        ),
        "validation_se": float(
            final_selection["selected"]["validation_MSE_SE"]
        ),
        "RB_RESIDUAL_CERTIFIED": rb_pass,
        "FINAL_KKT_PASS": kkt_pass,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_access_count": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
