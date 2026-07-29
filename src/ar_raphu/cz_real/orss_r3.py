"""CZ R3 development tasks backed by the matrix-free ORSS solver."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Callable, TypeVar

import numpy as np
import torch

from ar_raphu.orss.diagnostics import write_json
from ar_raphu.orss.operator import (
    OperatorTimelineCache,
    _bounded_c1_basis,
    build_operator_timeline_cache,
    build_urysohn_operator,
)
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


T = TypeVar("T")


def _parallel_folds(
    rows,
    worker: Callable[[object], T],
    *,
    maximum_workers: int,
    device: torch.device,
) -> list[T]:
    if maximum_workers <= 1:
        return [worker(row) for row in rows]

    def streamed(row):
        if device.type != "cuda":
            return worker(row)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            result = worker(row)
        stream.synchronize()
        return result

    ordered: dict[int, T] = {}
    with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = {
            executor.submit(streamed, row): index
            for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
    return [ordered[index] for index in range(len(rows))]


def prepare_history_group_timeline_caches(
    x: np.ndarray,
    y: np.ndarray,
    *,
    L_x: int,
    L_y: int,
    M_tau: int,
    M_x: int,
    c_rho: float,
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
) -> tuple[
    dict[torch.dtype, dict[int, OperatorTimelineCache]],
    dict[str, object],
]:
    """Create fold-local timeline bases once for every horizon in a group."""

    folds = build_development_folds(
        L_x=L_x, L_y=L_y, h_max=max(DIRECT_HORIZONS)
    )
    started = time.perf_counter()
    caches: dict[torch.dtype, dict[int, OperatorTimelineCache]] = {}
    per_dtype: dict[str, float] = {}
    for dtype in dtypes:
        dtype_started = time.perf_counter()
        fold_caches: dict[int, OperatorTimelineCache] = {}
        for fold in folds:
            fold_caches[fold.fold] = build_operator_timeline_cache(
                x,
                y,
                train_target_stop=fold.effective_train_stop,
                L_x=L_x,
                L_y=L_y,
                lag_basis_count=M_tau,
                amplitude_basis_count=M_x,
                continuation_scale_coefficient=c_rho,
                device=device,
                dtype=dtype,
            )
        caches[dtype] = fold_caches
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_dtype[str(dtype)] = time.perf_counter() - dtype_started
    return caches, {
        "basis_build_seconds": time.perf_counter() - started,
        "basis_build_seconds_by_dtype": per_dtype,
        "horizon_shared_timeline_cache": True,
        "fold_cache_count": len(folds),
    }


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
    timeline_cache: OperatorTimelineCache | None = None,
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
        timeline_cache=timeline_cache,
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
        timeline_cache=timeline_cache,
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
        state,
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
    timeline_caches: dict[int, OperatorTimelineCache] | None = None,
    concurrent_folds: int = 1,
) -> list[PenaltyNormalization]:
    def normalize(fold) -> PenaltyNormalization:
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
            timeline_cache=(
                timeline_caches.get(fold.fold)
                if timeline_caches is not None
                else None
            ),
        )
        penalty = SeparablePenalty(
            channels=train.channels,
            m_tau=train.m_tau,
            m_x=train.m_x,
            device=device,
            dtype=dtype,
        )
        result = diagonal_spectral_normalization(train, penalty)
        del train, validation, penalty
        return result

    rows = _parallel_folds(
        list(folds),
        normalize,
        maximum_workers=concurrent_folds,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _expanded_bounds(
    lower: list[float],
    upper: list[float],
    boundaries: list[tuple[int, str]],
    *,
    positive_grid_points: int,
) -> tuple[list[float], list[float]]:
    if positive_grid_points < 2:
        raise ValueError("Edge expansion requires at least two positive points.")
    for axis, side in boundaries:
        step = (
            np.log(upper[axis]) - np.log(lower[axis])
        ) / (positive_grid_points - 1)
        if side == "upper":
            upper[axis] = float(np.exp(np.log(upper[axis]) + step))
        else:
            lower[axis] = float(np.exp(np.log(lower[axis]) - step))
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
    candidate_batch_size: int = 512,
    concurrent_folds: int = 1,
    timeline_caches: dict[int, OperatorTimelineCache] | None = None,
    certification_timeline_caches: (
        dict[int, OperatorTimelineCache] | None
    ) = None,
    checkpoint_root: Path | None = None,
    checkpoint_identity: dict[str, object] | None = None,
    resume: bool = False,
    shared_cache_profiler: dict[str, object] | None = None,
) -> dict[str, object]:
    if concurrent_folds < 1:
        raise ValueError("concurrent_folds must be positive.")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    folds = build_development_folds(
        L_x=L_x, L_y=L_y, h_max=max(DIRECT_HORIZONS)
    )
    normalization_started = time.perf_counter()
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
        timeline_caches=timeline_caches,
        concurrent_folds=concurrent_folds,
    )
    normalization_seconds = time.perf_counter() - normalization_started
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
        def run_fold(pair) -> dict[str, object]:
            fold, normalization = pair
            fold_started = time.perf_counter()
            fold_root = (
                checkpoint_root
                / f"fold_{fold.fold}"
                / f"expansion_{expansion}"
                if checkpoint_root is not None
                else None
            )
            identity = {
                **(checkpoint_identity or {}),
                "fold": int(fold.fold),
                "expansion": int(expansion),
                "horizon": int(horizon),
                "L_x": int(L_x),
                "L_y": int(L_y),
                "M_tau": int(M_tau),
                "M_x": int(M_x),
                "c_rho": float(c_rho),
                "normalization": asdict(normalization),
                "candidate_count": len(candidates),
            }
            fold_result_path = (
                fold_root / "fold_result.json"
                if fold_root is not None
                else None
            )
            if fold_root is not None:
                identity_path = fold_root / "checkpoint_identity.json"
                if identity_path.exists():
                    saved_identity = json.loads(
                        identity_path.read_text(encoding="utf-8")
                    )
                    if saved_identity != identity:
                        raise RuntimeError(
                            "FOLD_CHECKPOINT_IDENTITY_MISMATCH"
                        )
                else:
                    write_json(identity_path, identity)
            if (
                resume
                and fold_result_path is not None
                and fold_result_path.exists()
            ):
                existing = json.loads(
                    fold_result_path.read_text(encoding="utf-8")
                )
                if existing.get("checkpoint_identity") != identity:
                    raise RuntimeError(
                        "FOLD_CHECKPOINT_IDENTITY_MISMATCH"
                    )
                return existing
            window_started = time.perf_counter()
            (
                train,
                validation,
                train_target,
                validation_target,
                target_mean,
                _,
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
                timeline_cache=(
                    timeline_caches.get(fold.fold)
                    if timeline_caches is not None
                    else None
                ),
            )
            (
                fallback_train,
                fallback_validation,
                fallback_train_target,
                fallback_validation_target,
                _,
                _,
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
                timeline_cache=(
                    certification_timeline_caches.get(fold.fold)
                    if certification_timeline_caches is not None
                    else None
                ),
            )
            window_build_seconds = time.perf_counter() - window_started
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
                candidate_batch_size=candidate_batch_size,
                checkpoint_dir=fold_root,
                resume=resume,
                fallback_train_operator=fallback_train,
                fallback_validation_operator=fallback_validation,
                fallback_train_target=fallback_train_target,
                fallback_validation_target=fallback_validation_target,
            )
            result["fold"] = fold.fold
            result["penalty_normalization"] = asdict(normalization)
            result["checkpoint_identity"] = identity
            result["profiler"]["window_build_seconds"] = (
                window_build_seconds
            )
            result["profiler"]["per_fold_seconds"] = (
                time.perf_counter() - fold_started
            )
            if fold_result_path is not None:
                write_json(fold_result_path, result)
                (fold_root / "DONE").write_text(
                    "COMPLETED\n", encoding="utf-8"
                )
            del (
                train,
                validation,
                train_target,
                validation_target,
                fallback_train,
                fallback_validation,
                fallback_train_target,
                fallback_validation_target,
            )
            return result

        fold_results = _parallel_folds(
            list(zip(folds, normalizations, strict=True)),
            run_fold,
            maximum_workers=concurrent_folds,
            device=device,
        )
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
        if checkpoint_root is not None:
            write_json(
                checkpoint_root / f"expansion_{expansion:02d}.json",
                interval_history[-1],
            )
        final_folds = fold_results
        final_candidates = candidates
        final_selection = selection
        if not requests:
            interval_status = "PENALTY_INTERVAL_CERTIFIED"
            break
        if expansion < maximum_edge_expansions:
            lower, upper = _expanded_bounds(
                lower,
                upper,
                requests,
                positive_grid_points=positive_grid_points,
            )
    if final_selection is None:
        raise AssertionError("Penalty sweep produced no selection.")
    all_rows = final_selection.pop("_all_rows")
    selected_weights = PenaltyWeights(
        **final_selection["selected"]["normalized_weights"]
    )

    # Refit the frozen candidate in FP64 on every fold and certify the KKT.
    def certify(pair) -> dict[str, object]:
        fold, normalization = pair
        fold_started = time.perf_counter()
        certification_root = (
            checkpoint_root / "certification" / f"fold_{fold.fold}"
            if checkpoint_root is not None
            else None
        )
        result_path = (
            certification_root / "fold_result.json"
            if certification_root is not None
            else None
        )
        identity = {
            **(checkpoint_identity or {}),
            "fold": int(fold.fold),
            "stage": "FINAL_FP64_CERTIFICATION",
            "horizon": int(horizon),
            "L_x": int(L_x),
            "L_y": int(L_y),
            "M_tau": int(M_tau),
            "M_x": int(M_x),
            "selected_weights": asdict(selected_weights),
            "normalization": asdict(normalization),
        }
        if certification_root is not None:
            identity_path = (
                certification_root / "checkpoint_identity.json"
            )
            if identity_path.exists():
                saved_identity = json.loads(
                    identity_path.read_text(encoding="utf-8")
                )
                if saved_identity != identity:
                    raise RuntimeError(
                        "CERTIFICATION_CHECKPOINT_IDENTITY_MISMATCH"
                    )
            else:
                write_json(identity_path, identity)
        if resume and result_path is not None and result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("checkpoint_identity") != identity:
                raise RuntimeError(
                    "CERTIFICATION_CHECKPOINT_IDENTITY_MISMATCH"
                )
            return existing
        window_started = time.perf_counter()
        (
            train,
            validation,
            train_target,
            validation_target,
            target_mean,
            _,
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
            timeline_cache=(
                certification_timeline_caches.get(fold.fold)
                if certification_timeline_caches is not None
                else None
            ),
        )
        window_build_seconds = time.perf_counter() - window_started
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
            preconditioner_kind="channel_block",
        )
        prediction = validation.forward(fitted.coefficients) + target_mean
        result = {
            "fold": fold.fold,
            "actual_weights": asdict(actual),
            "relative_kkt_residual": fitted.relative_kkt_residual,
            "iterations": fitted.iterations,
            "method": fitted.method,
            "validation_MSE_mm2": float(
                torch.mean((prediction - validation_target) ** 2).item()
            ),
            "checkpoint_identity": identity,
            "profiler": {
                "window_build_seconds": window_build_seconds,
                "final_refinement_seconds": fitted.elapsed_seconds,
                "per_fold_seconds": time.perf_counter() - fold_started,
            },
        }
        if result_path is not None:
            write_json(result_path, result)
            (certification_root / "DONE").write_text(
                "COMPLETED\n", encoding="utf-8"
            )
        del train, validation, train_target, validation_target, penalty
        return result

    certifications = _parallel_folds(
        list(zip(folds, normalizations, strict=True)),
        certify,
        maximum_workers=concurrent_folds,
        device=device,
    )
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
        "profiler": {
            **(shared_cache_profiler or {}),
            "normalization_seconds": normalization_seconds,
            "concurrent_folds": concurrent_folds,
            "candidate_batch_size": candidate_batch_size,
            "per_fold_seconds": [
                float(row["profiler"]["per_fold_seconds"])
                for row in final_folds
            ],
            "full_anchor_count": int(
                sum(
                    int(row["profiler"]["full_anchor_count"])
                    for row in final_folds
                )
            ),
            "full_anchor_seconds": float(
                sum(
                    float(row["profiler"]["full_anchor_seconds"])
                    for row in final_folds
                )
            ),
            "krylov_iterations_per_anchor": [
                int(value)
                for row in final_folds
                for value in row["profiler"][
                    "krylov_iterations_per_anchor"
                ]
            ],
            "rb_dimension": [
                int(row["profiler"]["rb_dimension"])
                for row in final_folds
            ],
            "rb_greedy_iterations": [
                int(row["profiler"]["rb_greedy_iterations"])
                for row in final_folds
            ],
            "reduced_batch_seconds": float(
                sum(
                    float(row["profiler"]["reduced_batch_seconds"])
                    for row in final_folds
                )
            ),
            "preconditioner_build_seconds": float(
                sum(
                    float(
                        row["profiler"][
                            "preconditioner_build_seconds"
                        ]
                    )
                    for row in final_folds
                )
            ),
            "operator_forward_calls": int(
                sum(
                    int(row["profiler"]["operator_forward_calls"])
                    for row in final_folds
                )
            ),
            "operator_adjoint_calls": int(
                sum(
                    int(row["profiler"]["operator_adjoint_calls"])
                    for row in final_folds
                )
            ),
            "dr_build_seconds": 0.0,
            "dr_build_status": "NOT_USED_BY_ORSS_V2_BATCHED",
            "cpu_gpu_sync_count": 0,
            "cpu_gpu_sync_count_status": "NOT_INSTRUMENTED",
            "gpu_kernel_active_fraction": None,
            "gpu_kernel_active_fraction_status": (
                "EXTERNAL_TELEMETRY_REQUIRED"
            ),
            "effective_df_seconds": 0.0,
            "final_refinement_seconds": float(
                sum(
                    float(
                        row["profiler"]["final_refinement_seconds"]
                    )
                    for row in certifications
                )
            ),
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
    }


def _free_run_fold(
    operator,
    state,
    coefficients: torch.Tensor,
    *,
    target_mean: float,
    y: np.ndarray,
    validation_indices: np.ndarray,
    horizon: int,
    validation_start: int,
    c_rho: float,
) -> dict[str, object]:
    shaped = operator.reshape_theta(coefficients)
    external = shaped.clone()
    external[-1] = 0.0
    external_raw = operator._raw_forward(external.reshape(-1))
    centered_offset = torch.sum(operator.feature_mean * shaped)
    y_work = torch.as_tensor(
        np.asarray(y, dtype=np.float64),
        device=operator.device,
        dtype=operator.dtype,
    ).clone()
    offsets = torch.arange(
        operator.branches[-1].amplitude.shape[1],
        device=operator.device,
        dtype=torch.int64,
    )
    lag_basis = operator.branches[-1].lag_basis
    predictions = torch.empty(
        len(validation_indices),
        device=operator.device,
        dtype=operator.dtype,
    )
    out_calls = 0
    total_calls = 0
    maximum_distance = 0.0
    first_exit: int | None = None
    lower = float(state.ar_basis.lower)
    upper = float(state.ar_basis.upper)
    scale = max(upper - lower, np.finfo(np.float64).eps)
    for row, target in enumerate(validation_indices):
        origin = int(target) - int(horizon)
        window = y_work[origin - offsets]
        amplitude, inside = _bounded_c1_basis(
            state.ar_basis, window, scale_factor=c_rho
        )
        outside = ~inside
        count = int(outside.sum().item())
        if count and first_exit is None:
            first_exit = int(target)
        out_calls += count
        total_calls += int(window.numel())
        if count:
            distance = torch.maximum(
                torch.clamp(
                    torch.as_tensor(lower, device=window.device, dtype=window.dtype)
                    - window,
                    min=0.0,
                ),
                torch.clamp(
                    window
                    - torch.as_tensor(
                        upper, device=window.device, dtype=window.dtype
                    ),
                    min=0.0,
                ),
            )
            maximum_distance = max(
                maximum_distance, float(distance.max().item()) / scale
            )
        ar_value = torch.einsum(
            "lb,la,ab->",
            amplitude,
            lag_basis,
            shaped[-1],
        )
        prediction = (
            external_raw[row] + ar_value - centered_offset + target_mean
        )
        predictions[row] = prediction
        y_work[int(target)] = prediction
    target = torch.as_tensor(
        np.asarray(y, dtype=np.float64)[validation_indices],
        device=operator.device,
        dtype=operator.dtype,
    )
    residual = predictions - target
    external_total = sum(
        branch.amplitude.shape[0] * branch.amplitude.shape[1]
        for branch in operator.branches[:-1]
    )
    external_out = sum(
        int(round(branch.out_of_domain_fraction * (
            branch.amplitude.shape[0] * branch.amplitude.shape[1]
        )))
        for branch in operator.branches[:-1]
    )
    return {
        "free_run_MSE_mm2": float(torch.mean(residual.square()).item()),
        "free_run_RMSE_mm": float(
            torch.sqrt(torch.mean(residual.square())).item()
        ),
        "finite_complete_free_run": bool(
            torch.isfinite(predictions).all().item()
        ),
        "first_exit_target_index": first_exit,
        "AR_continuation_calls": out_calls,
        "AR_total_calls": total_calls,
        "external_continuation_calls": external_out,
        "external_total_calls": external_total,
        "continuation_usage_fraction": (
            (out_calls + external_out)
            / max(total_calls + external_total, 1)
        ),
        "maximum_normalized_extrapolation_distance": maximum_distance,
        "drift_slope_mm_per_sample": float(
            np.polyfit(
                np.arange(len(predictions), dtype=np.float64),
                residual.detach().cpu().numpy(),
                1,
            )[0]
        ),
        "maximum_absolute_bias_mm": float(residual.abs().max().item()),
    }


def evaluate_frozen_configuration(
    x: np.ndarray,
    y: np.ndarray,
    *,
    horizon: int,
    L_x: int,
    L_y: int,
    M_tau: int,
    M_x: int,
    normalized_weights: PenaltyWeights,
    c_rho: float,
    device: torch.device,
    chunk_time: int,
    maximum_iterations: int,
    include_predictions: bool = False,
) -> dict[str, object]:
    """Refit a frozen model and evaluate teacher-forced and free-run folds."""

    folds = build_development_folds(
        L_x=L_x, L_y=L_y, h_max=max(DIRECT_HORIZONS)
    )
    records: list[dict[str, object]] = []
    for fold in folds:
        validation_indices = target_indices(
            start=fold.validation_start,
            stop=fold.validation_stop,
            horizon=horizon,
            max_history=max(L_x, L_y),
        )
        (
            train,
            validation,
            train_target,
            validation_target,
            target_mean,
            state,
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
        normalization = diagonal_spectral_normalization(train, penalty)
        actual = normalization.actual(normalized_weights)
        fitted = solve_full(
            train,
            train_target,
            penalty,
            actual,
            relative_tolerance=1.0e-10,
            maximum_iterations=max(2500, maximum_iterations),
            preconditioner_kind="channel_block",
        )
        teacher_prediction = validation.forward(
            fitted.coefficients
        ) + target_mean
        teacher_mse = float(
            torch.mean(
                (teacher_prediction - validation_target).square()
            ).item()
        )
        free_run = _free_run_fold(
            validation,
            state,
            fitted.coefficients,
            target_mean=target_mean,
            y=y,
            validation_indices=validation_indices,
            horizon=horizon,
            validation_start=fold.validation_start,
            c_rho=c_rho,
        )
        record = {
                "fold": fold.fold,
                "teacher_forced_validation_MSE_mm2": teacher_mse,
                "relative_kkt_residual": fitted.relative_kkt_residual,
                "actual_weights": asdict(actual),
                **free_run,
            }
        if include_predictions:
            record["teacher_forced_prediction"] = (
                teacher_prediction.detach().cpu().numpy()
            )
            record["validation_target"] = (
                validation_target.detach().cpu().numpy()
            )
            record["validation_target_indices"] = validation_indices
        records.append(record)
        del train, validation, train_target, validation_target, penalty
        if device.type == "cuda":
            torch.cuda.empty_cache()
    free_losses = np.asarray(
        [float(row["free_run_MSE_mm2"]) for row in records],
        dtype=np.float64,
    )
    teacher_losses = np.asarray(
        [float(row["teacher_forced_validation_MSE_mm2"]) for row in records],
        dtype=np.float64,
    )
    return {
        "status": (
            "COMPLETED"
            if all(
                bool(row["finite_complete_free_run"])
                and float(row["relative_kkt_residual"]) <= 1.0e-8
                for row in records
            )
            else "FAILED"
        ),
        "folds": records,
        "free_run_MSE_mean": float(free_losses.mean()),
        "free_run_MSE_SE": float(
            free_losses.std(ddof=1) / np.sqrt(len(free_losses))
        ),
        "teacher_forced_MSE_mean": float(teacher_losses.mean()),
        "maximum_normalized_extrapolation_distance": max(
            float(row["maximum_normalized_extrapolation_distance"])
            for row in records
        ),
        "continuation_usage_fraction": float(
            np.mean(
                [float(row["continuation_usage_fraction"]) for row in records]
            )
        ),
        "finite_complete_free_run": all(
            bool(row["finite_complete_free_run"]) for row in records
        ),
        "FINAL_KKT_PASS": all(
            float(row["relative_kkt_residual"]) <= 1.0e-8
            for row in records
        ),
    }


def _gram_whitened_rank_reconstructions(
    operator,
    coefficients: torch.Tensor,
) -> tuple[list[list[float]], list[torch.Tensor]]:
    shaped = operator.reshape_theta(coefficients)
    spectra: list[list[float]] = []
    per_rank = [
        torch.zeros_like(shaped)
        for _ in range(min(operator.m_tau, operator.m_x))
    ]
    for channel, branch in enumerate(operator.branches):
        lag_gram = (
            branch.lag_basis.T @ branch.lag_basis
        ) / branch.lag_basis.shape[0]
        amplitude_flat = branch.amplitude.reshape(-1, operator.m_x)
        amplitude_gram = (
            amplitude_flat.T @ amplitude_flat
        ) / amplitude_flat.shape[0]
        lag_floor = (
            torch.linalg.eigvalsh(lag_gram).abs().max().clamp_min(1.0)
            * 1.0e-12
        )
        amplitude_floor = (
            torch.linalg.eigvalsh(amplitude_gram)
            .abs()
            .max()
            .clamp_min(1.0)
            * 1.0e-12
        )
        lag_factor = torch.linalg.cholesky(
            lag_gram
            + lag_floor
            * torch.eye(
                operator.m_tau,
                device=operator.device,
                dtype=operator.dtype,
            )
        ).T
        amplitude_factor = torch.linalg.cholesky(
            amplitude_gram
            + amplitude_floor
            * torch.eye(
                operator.m_x,
                device=operator.device,
                dtype=operator.dtype,
            )
        ).T
        whitened = (
            lag_factor @ shaped[channel] @ amplitude_factor.T
        )
        left, singular, right_t = torch.linalg.svd(
            whitened, full_matrices=False
        )
        spectra.append([float(value) for value in singular.tolist()])
        for rank in range(1, len(per_rank) + 1):
            truncated = (
                left[:, :rank]
                @ torch.diag(singular[:rank])
                @ right_t[:rank]
            )
            unlagged = torch.linalg.solve(lag_factor, truncated)
            restored = torch.linalg.solve(
                amplitude_factor,
                unlagged.T,
            ).T
            per_rank[rank - 1][channel] = restored
    return spectra, [row.reshape(-1) for row in per_rank]


def rank_profile_configuration(
    x: np.ndarray,
    y: np.ndarray,
    *,
    horizon: int,
    L_x: int,
    L_y: int,
    M_tau: int,
    M_x: int,
    normalized_weights: PenaltyWeights,
    c_rho: float,
    rank_budgets: list[float],
    device: torch.device,
    chunk_time: int,
    maximum_iterations: int,
) -> dict[str, object]:
    folds = build_development_folds(
        L_x=L_x, L_y=L_y, h_max=max(DIRECT_HORIZONS)
    )
    fold_rows: list[dict[str, object]] = []
    for fold in folds:
        (
            train,
            validation,
            train_target,
            validation_target,
            target_mean,
            _,
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
        normalization = diagonal_spectral_normalization(train, penalty)
        actual = normalization.actual(normalized_weights)
        fitted = solve_full(
            train,
            train_target,
            penalty,
            actual,
            relative_tolerance=1.0e-10,
            maximum_iterations=max(2500, maximum_iterations),
            preconditioner_kind="channel_block",
        )
        full_prediction = validation.forward(
            fitted.coefficients
        ) + target_mean
        full_mse = float(
            torch.mean(
                (full_prediction - validation_target).square()
            ).item()
        )
        spectra, reconstructions = _gram_whitened_rank_reconstructions(
            train, fitted.coefficients
        )
        rank_rows: list[dict[str, float | int]] = []
        for rank, coefficients in enumerate(reconstructions, start=1):
            prediction = validation.forward(coefficients) + target_mean
            mse = float(
                torch.mean((prediction - validation_target).square()).item()
            )
            rank_rows.append(
                {
                    "rank": rank,
                    "validation_MSE_mm2": mse,
                    "relative_to_full_inflation": (
                        (mse - full_mse) / max(full_mse, 1.0e-15)
                    ),
                }
            )
        predictive = {
            str(budget): next(
                (
                    int(row["rank"])
                    for row in rank_rows
                    if float(row["relative_to_full_inflation"]) <= budget
                ),
                len(rank_rows),
            )
            for budget in rank_budgets
        }
        fold_rows.append(
            {
                "fold": fold.fold,
                "full_validation_MSE_mm2": full_mse,
                "rank1_validation_MSE_mm2": rank_rows[0][
                    "validation_MSE_mm2"
                ],
                "rank2_validation_MSE_mm2": rank_rows[
                    min(1, len(rank_rows) - 1)
                ]["validation_MSE_mm2"],
                "predictive_rank_by_budget": predictive,
                "rank_curve": rank_rows,
                "singular_spectra_by_channel": spectra,
                "relative_kkt_residual": fitted.relative_kkt_residual,
            }
        )
        del train, validation, train_target, validation_target, penalty
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "status": (
            "COMPLETED"
            if all(
                float(row["relative_kkt_residual"]) <= 1.0e-8
                for row in fold_rows
            )
            else "FAILED"
        ),
        "folds": fold_rows,
        "predictive_rank_by_budget": {
            str(budget): int(
                np.ceil(
                    np.mean(
                        [
                            int(row["predictive_rank_by_budget"][str(budget)])
                            for row in fold_rows
                        ]
                    )
                )
            )
            for budget in rank_budgets
        },
        "structural_rank_status": "K_LEVEL_NOT_IDENTIFIED",
        "structural_rank_reason": (
            "Finite-sieve coercivity and Schur identification were not "
            "pre-certified; predictive rank is reported without a K-level claim."
        ),
        "FINAL_KKT_PASS": all(
            float(row["relative_kkt_residual"]) <= 1.0e-8
            for row in fold_rows
        ),
    }
