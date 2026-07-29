"""Certified matrix-free penalty sweeps for CZ ORSS experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch

from .augmented import AugmentedRegularizedOperator
from .krylov import KrylovResult, lsqr, pcg_normal
from .operator import UrysohnLinearOperator
from .penalties import PenaltyWeights, SeparablePenalty
from .preconditioner import (
    build_diagonal_preconditioner,
    data_normal_diagonal,
)
from .reduced_basis import ParametricReducedBasis, ReducedCandidate


AXES = ("lag", "amplitude", "ridge")


@dataclass(frozen=True, slots=True)
class PenaltyNormalization:
    lag: float
    amplitude: float
    ridge: float
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def actual(self, normalized: PenaltyWeights) -> PenaltyWeights:
        return PenaltyWeights(
            normalized.lag / self.lag,
            normalized.amplitude / self.amplitude,
            normalized.ridge / self.ridge,
        )


@dataclass(frozen=True, slots=True)
class FullSolve:
    coefficients: torch.Tensor
    relative_kkt_residual: float
    iterations: int
    method: str
    elapsed_seconds: float
    preconditioner_build_seconds: float
    operator_forward_calls: int
    operator_adjoint_calls: int


def _positive_ratio(
    penalty_diagonal: torch.Tensor, data_diagonal: torch.Tensor
) -> torch.Tensor:
    ratio = penalty_diagonal / data_diagonal.clamp_min(
        torch.finfo(data_diagonal.dtype).eps
    )
    scale = ratio.abs().max().clamp_min(1.0)
    return ratio[ratio > 1.0e-10 * scale]


def diagonal_spectral_normalization(
    operator: UrysohnLinearOperator,
    penalty: SeparablePenalty,
) -> PenaltyNormalization:
    """Matrix-free automatic interval using the normal diagonal spectrum.

    This is the ORSS replacement for dense generalized-eigenvalue interval
    construction.  The approximation is fixed before looking at validation
    loss and its scale and extrema are recorded in every result.
    """

    data_diagonal = data_normal_diagonal(operator).to(torch.float64)
    unit = (
        PenaltyWeights(1.0, 0.0, 0.0),
        PenaltyWeights(0.0, 1.0, 0.0),
        PenaltyWeights(0.0, 0.0, 1.0),
    )
    scales: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for weights in unit:
        diagonal = penalty.diagonal(weights).to(torch.float64)
        ratio = _positive_ratio(diagonal, data_diagonal)
        if not len(ratio):
            raise RuntimeError("PENALTY_NORMALIZATION_HAS_NO_POSITIVE_MODES")
        scale = float(torch.median(ratio).item())
        normalized = ratio / scale
        scales.append(scale)
        lower.append(
            max(
                (1.0 / 0.999 - 1.0) / float(normalized.max().item()),
                np.finfo(np.float64).tiny,
            )
        )
        upper.append(
            (1.0 / 0.001 - 1.0) / float(normalized.min().item())
        )
    return PenaltyNormalization(
        *scales,
        lower=tuple(lower),
        upper=tuple(upper),
    )


def candidate_grid(
    lower: Iterable[float],
    upper: Iterable[float],
    *,
    positive_points: int,
) -> tuple[list[PenaltyWeights], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grids = tuple(
        np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                np.geomspace(float(lo), float(hi), positive_points),
            )
        )
        for lo, hi in zip(lower, upper, strict=True)
    )
    candidates = [
        PenaltyWeights(float(a), float(b), float(c))
        for a, b, c in itertools.product(*grids)
    ]
    return candidates, grids  # type: ignore[return-value]


def anchor_weights(
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[PenaltyWeights]:
    low = [float(grid[1]) for grid in grids]
    high = [float(grid[-1]) for grid in grids]
    middle = [float(grid[1 + (len(grid) - 2) // 2]) for grid in grids]
    rows = [
        PenaltyWeights(0.0, 0.0, 0.0),
        PenaltyWeights(*middle),
        PenaltyWeights(high[0], middle[1], middle[2]),
        PenaltyWeights(middle[0], high[1], middle[2]),
        PenaltyWeights(middle[0], middle[1], high[2]),
        PenaltyWeights(high[0], high[1], low[2]),
        PenaltyWeights(high[0], low[1], high[2]),
        PenaltyWeights(low[0], high[1], high[2]),
    ]
    unique: list[PenaltyWeights] = []
    for row in rows:
        if row not in unique:
            unique.append(row)
    return unique


def solve_full(
    operator: UrysohnLinearOperator,
    centered_target: torch.Tensor,
    penalty: SeparablePenalty,
    weights: PenaltyWeights,
    *,
    relative_tolerance: float,
    maximum_iterations: int,
    initial: torch.Tensor | None = None,
) -> FullSolve:
    started = time.perf_counter()
    forward_before = operator.operator_forward_calls
    adjoint_before = operator.operator_adjoint_calls
    preconditioner_seconds = 0.0
    if weights.exact_zero:
        augmented = AugmentedRegularizedOperator(operator, penalty, weights)
        result: KrylovResult = lsqr(
            augmented,
            augmented.augmented_rhs(centered_target),
            relative_tolerance=relative_tolerance,
            maximum_iterations=maximum_iterations,
            initial=initial,
        )
        preconditioner_started = time.perf_counter()
        preconditioner = build_diagonal_preconditioner(
            operator, penalty, weights
        )
        preconditioner_seconds += (
            time.perf_counter() - preconditioner_started
        )
        refinement = pcg_normal(
            operator.normal,
            operator.rhs(centered_target),
            initial=result.coefficients,
            preconditioner=preconditioner,
            relative_tolerance=relative_tolerance,
            maximum_iterations=max(maximum_iterations, 2500),
        )
        coefficients = refinement.coefficients
        relative = refinement.relative_residual
        iterations = result.iterations + refinement.iterations
        method = "ZERO_ENDPOINT_LSQR_MINIMUM_NORM_PLUS_PCG_REFINEMENT"
    else:
        rhs = operator.rhs(centered_target)
        preconditioner_started = time.perf_counter()
        preconditioner = build_diagonal_preconditioner(
            operator, penalty, weights
        )
        preconditioner_seconds += (
            time.perf_counter() - preconditioner_started
        )
        result_cg = pcg_normal(
            lambda vector: operator.normal(vector)
            + penalty.normal(vector, weights),
            rhs,
            initial=initial,
            preconditioner=preconditioner,
            relative_tolerance=relative_tolerance,
            maximum_iterations=maximum_iterations,
        )
        coefficients = result_cg.coefficients
        relative = result_cg.relative_residual
        iterations = result_cg.iterations
        method = "DIAGONAL_PRECONDITIONED_NORMAL_PCG"
    return FullSolve(
        coefficients=coefficients,
        relative_kkt_residual=float(relative),
        iterations=int(iterations),
        method=method,
        elapsed_seconds=time.perf_counter() - started,
        preconditioner_build_seconds=preconditioner_seconds,
        operator_forward_calls=(
            operator.operator_forward_calls - forward_before
        ),
        operator_adjoint_calls=(
            operator.operator_adjoint_calls - adjoint_before
        ),
    )


def _actual(
    normalized: PenaltyWeights, normalization: PenaltyNormalization
) -> PenaltyWeights:
    return normalization.actual(normalized)


def reduced_sweep(
    train_operator: UrysohnLinearOperator,
    validation_operator: UrysohnLinearOperator,
    train_target: torch.Tensor,
    validation_target: torch.Tensor,
    *,
    target_mean: float,
    candidates: list[PenaltyWeights],
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    normalization: PenaltyNormalization,
    residual_tolerance: float,
    maximum_dimension: int,
    krylov_tolerance: float,
    maximum_iterations: int,
    candidate_batch_size: int = 512,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, object]:
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be positive.")
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_state() -> None:
        if checkpoint_dir is None:
            return
        torch.save(
            {
                "basis": reduced.basis.detach().cpu(),
                "anchor_weights": [
                    asdict(row) for row in reduced.anchor_weights
                ],
            },
            checkpoint_dir / "rb_state.pt",
        )

    def batched_scan(
        rows: list[PenaltyWeights],
    ) -> list[ReducedCandidate]:
        resolved: list[ReducedCandidate] = []
        for start in range(0, len(rows), candidate_batch_size):
            resolved.extend(
                reduced.solve_many(rows[start : start + candidate_batch_size])
            )
        return resolved

    penalty = SeparablePenalty(
        channels=train_operator.channels,
        m_tau=train_operator.m_tau,
        m_x=train_operator.m_x,
        device=train_operator.device,
        dtype=train_operator.dtype,
    )
    right = train_operator.rhs(train_target)
    reduced = ParametricReducedBasis(
        train_operator,
        penalty,
        right,
        maximum_dimension=maximum_dimension,
    )
    anchors: list[dict[str, object]] = []
    cached: dict[PenaltyWeights, FullSolve] = {}
    anchor_root = checkpoint_dir / "anchors" if checkpoint_dir else None
    for anchor_index, normalized in enumerate(anchor_weights(grids)):
        actual = _actual(normalized, normalization)
        anchor_file = (
            anchor_root / f"anchor_{anchor_index:03d}.pt"
            if anchor_root is not None
            else None
        )
        was_resumed = bool(
            resume and anchor_file is not None and anchor_file.exists()
        )
        if was_resumed:
            saved = torch.load(
                anchor_file,
                map_location=train_operator.device,
                weights_only=True,
            )
            saved_weights = PenaltyWeights(**saved["normalized_weights"])
            if saved_weights != normalized:
                raise RuntimeError("ANCHOR_CHECKPOINT_IDENTITY_MISMATCH")
            solved = FullSolve(
                coefficients=saved["coefficients"].to(
                    device=train_operator.device,
                    dtype=train_operator.dtype,
                ),
                relative_kkt_residual=float(
                    saved["relative_kkt_residual"]
                ),
                iterations=int(saved["iterations"]),
                method=str(saved["method"]),
                elapsed_seconds=float(saved["elapsed_seconds"]),
                preconditioner_build_seconds=float(
                    saved.get("preconditioner_build_seconds", 0.0)
                ),
                operator_forward_calls=int(
                    saved.get("operator_forward_calls", 0)
                ),
                operator_adjoint_calls=int(
                    saved.get("operator_adjoint_calls", 0)
                ),
            )
        else:
            solved = solve_full(
                train_operator,
                train_target,
                penalty,
                actual,
                relative_tolerance=krylov_tolerance,
                maximum_iterations=maximum_iterations,
            )
            if anchor_file is not None:
                anchor_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "normalized_weights": asdict(normalized),
                        "actual_weights": asdict(actual),
                        "coefficients": solved.coefficients.detach().cpu(),
                        "relative_kkt_residual": (
                            solved.relative_kkt_residual
                        ),
                        "iterations": solved.iterations,
                        "method": solved.method,
                        "elapsed_seconds": solved.elapsed_seconds,
                        "preconditioner_build_seconds": (
                            solved.preconditioner_build_seconds
                        ),
                        "operator_forward_calls": (
                            solved.operator_forward_calls
                        ),
                        "operator_adjoint_calls": (
                            solved.operator_adjoint_calls
                        ),
                    },
                    anchor_file,
                )
                (anchor_file.parent / f"anchor_{anchor_index:03d}.DONE").write_text(
                    "COMPLETED\n", encoding="utf-8"
                )
        cached[normalized] = solved
        added = reduced.add_anchor(normalized, solved.coefficients)
        anchors.append(
            {
                "normalized_weights": asdict(normalized),
                "actual_weights": asdict(actual),
                "iterations": solved.iterations,
                "relative_kkt_residual": solved.relative_kkt_residual,
                "elapsed_seconds": solved.elapsed_seconds,
                "method": solved.method,
                "basis_vector_added": added,
                "preconditioner_build_seconds": (
                    solved.preconditioner_build_seconds
                ),
                "operator_forward_calls": solved.operator_forward_calls,
                "operator_adjoint_calls": solved.operator_adjoint_calls,
                "resumed": was_resumed,
            }
        )
        checkpoint_state()

    greedy: list[dict[str, object]] = []
    fallback = False
    greedy_root = checkpoint_dir / "greedy" if checkpoint_dir else None
    if resume and greedy_root is not None and greedy_root.exists():
        for greedy_file in sorted(greedy_root.glob("greedy_*.pt")):
            saved = torch.load(
                greedy_file,
                map_location=train_operator.device,
                weights_only=True,
            )
            normalized = PenaltyWeights(**saved["normalized_weights"])
            added = reduced.add_anchor(
                normalized,
                saved["coefficients"].to(
                    device=train_operator.device,
                    dtype=train_operator.dtype,
                ),
            )
            if not added:
                raise RuntimeError("GREEDY_CHECKPOINT_BASIS_DUPLICATE")
            greedy.append(
                {
                    **saved["diagnostics"],
                    "basis_vector_added": True,
                    "dimension": reduced.dimension,
                    "resumed": True,
                }
            )
        checkpoint_state()

    reduced_batch_seconds = 0.0
    while True:
        actual_candidates = [
            _actual(candidate, normalization) for candidate in candidates
        ]
        batch_started = time.perf_counter()
        rows = batched_scan(actual_candidates)
        reduced_batch_seconds += time.perf_counter() - batch_started
        worst_index = max(
            range(len(rows)), key=lambda index: rows[index].relative_residual
        )
        worst = rows[worst_index]
        maximum_residual = float(worst.relative_residual)
        if maximum_residual <= residual_tolerance:
            break
        if reduced.dimension >= maximum_dimension:
            fallback = True
            break
        normalized = candidates[worst_index]
        actual = _actual(normalized, normalization)
        solved = solve_full(
            train_operator,
            train_target,
            penalty,
            actual,
            relative_tolerance=krylov_tolerance,
            maximum_iterations=maximum_iterations,
            initial=worst.coefficients.to(train_operator.dtype),
        )
        added = reduced.add_anchor(normalized, solved.coefficients)
        diagnostics = {
            "normalized_weights": asdict(normalized),
            "actual_weights": asdict(actual),
            "estimated_relative_residual": maximum_residual,
            "full_relative_kkt_residual": solved.relative_kkt_residual,
            "iterations": solved.iterations,
            "elapsed_seconds": solved.elapsed_seconds,
            "preconditioner_build_seconds": (
                solved.preconditioner_build_seconds
            ),
            "operator_forward_calls": solved.operator_forward_calls,
            "operator_adjoint_calls": solved.operator_adjoint_calls,
        }
        greedy.append(
            {
                **diagnostics,
                "basis_vector_added": added,
                "dimension": reduced.dimension,
                "resumed": False,
            }
        )
        if greedy_root is not None:
            greedy_root.mkdir(parents=True, exist_ok=True)
            greedy_index = len(greedy) - 1
            greedy_file = greedy_root / f"greedy_{greedy_index:03d}.pt"
            torch.save(
                {
                    "normalized_weights": asdict(normalized),
                    "actual_weights": asdict(actual),
                    "coefficients": solved.coefficients.detach().cpu(),
                    "diagnostics": diagnostics,
                },
                greedy_file,
            )
            (greedy_root / f"greedy_{greedy_index:03d}.DONE").write_text(
                "COMPLETED\n", encoding="utf-8"
            )
        checkpoint_state()
        if not added:
            fallback = True
            break

    candidate_rows: list[dict[str, object]] = []
    if fallback:
        full_rows: list[tuple[PenaltyWeights, FullSolve]] = []
        warm: torch.Tensor | None = None
        for normalized in candidates:
            solved = cached.get(normalized)
            if solved is None:
                solved = solve_full(
                    train_operator,
                    train_target,
                    penalty,
                    _actual(normalized, normalization),
                    relative_tolerance=krylov_tolerance,
                    maximum_iterations=maximum_iterations,
                    initial=warm,
                )
            warm = solved.coefficients
            full_rows.append((normalized, solved))
        resolved = [
            (normalized, solved.coefficients, solved.relative_kkt_residual)
            for normalized, solved in full_rows
        ]
    else:
        batch_started = time.perf_counter()
        rb_rows = batched_scan(
            [_actual(candidate, normalization) for candidate in candidates]
        )
        reduced_batch_seconds += time.perf_counter() - batch_started
        resolved = [
            (normalized, row.coefficients, row.relative_residual)
            for normalized, row in zip(candidates, rb_rows, strict=True)
        ]

    validation_started = time.perf_counter()
    losses: list[float] = []
    for start in range(0, len(resolved), candidate_batch_size):
        block = resolved[start : start + candidate_batch_size]
        coefficients = torch.stack(
            [row[1] for row in block], dim=0
        ).to(
            device=validation_operator.device,
            dtype=validation_operator.dtype,
        )
        prediction = (
            validation_operator.forward_batch(coefficients) + target_mean
        )
        block_losses = torch.mean(
            (prediction - validation_target[:, None]) ** 2, dim=0
        )
        losses.extend(block_losses.detach().cpu().tolist())
    validation_batch_seconds = time.perf_counter() - validation_started
    for order, ((normalized, _, residual), loss) in enumerate(
        zip(resolved, losses, strict=True)
    ):
        candidate_rows.append(
            {
                "configuration_order": order,
                "normalized_weights": asdict(normalized),
                "actual_weights": asdict(_actual(normalized, normalization)),
                "validation_MSE_mm2": float(loss),
                "relative_residual_certificate": float(residual),
            }
        )
    return {
        "candidate_rows": candidate_rows,
        "anchors": anchors,
        "greedy_iterations": greedy,
        "reduced_basis_dimension": reduced.dimension,
        "maximum_relative_residual": max(
            float(row["relative_residual_certificate"])
            for row in candidate_rows
        ),
        "RB_RESIDUAL_CERTIFIED": (
            max(
                float(row["relative_residual_certificate"])
                for row in candidate_rows
            )
            <= residual_tolerance
        ),
        "full_orss_fallback": fallback,
        "profiler": {
            "full_anchor_count": len(anchors),
            "full_anchor_seconds": float(
                sum(float(row["elapsed_seconds"]) for row in anchors)
            ),
            "krylov_iterations_per_anchor": [
                int(row["iterations"]) for row in anchors
            ],
            "rb_dimension": reduced.dimension,
            "rb_greedy_iterations": len(greedy),
            "reduced_batch_seconds": reduced_batch_seconds,
            "validation_batch_seconds": validation_batch_seconds,
            "candidate_batch_size": candidate_batch_size,
            "preconditioner_build_seconds": float(
                sum(
                    float(row["preconditioner_build_seconds"])
                    for row in anchors + greedy
                )
            ),
            "operator_forward_calls": int(
                sum(
                    int(row["operator_forward_calls"])
                    for row in anchors + greedy
                    if bool(row.get("resumed", False))
                )
                + train_operator.operator_forward_calls
            ),
            "operator_adjoint_calls": int(
                sum(
                    int(row["operator_adjoint_calls"])
                    for row in anchors + greedy
                    if bool(row.get("resumed", False))
                )
                + train_operator.operator_adjoint_calls
            ),
        },
    }


def aggregate_folds(
    fold_results: list[dict[str, object]],
    candidates: list[PenaltyWeights],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        fold_rows = [
            result["candidate_rows"][index] for result in fold_results
        ]
        losses = np.asarray(
            [float(row["validation_MSE_mm2"]) for row in fold_rows],
            dtype=np.float64,
        )
        rows.append(
            {
                "configuration_order": index,
                "normalized_weights": asdict(candidate),
                "validation_MSE_mean": float(losses.mean()),
                "validation_MSE_SE": float(
                    losses.std(ddof=1) / math.sqrt(len(losses))
                ),
                "folds": [
                    {
                        "fold": int(fold_results[fold]["fold"]),
                        **fold_rows[fold],
                    }
                    for fold in range(len(fold_results))
                ],
            }
        )
    return rows


def select_one_se(rows: list[dict[str, object]]) -> dict[str, object]:
    minimum = min(
        rows,
        key=lambda row: (
            float(row["validation_MSE_mean"]),
            int(row["configuration_order"]),
        ),
    )
    threshold = float(minimum["validation_MSE_mean"]) + float(
        minimum["validation_MSE_SE"]
    )
    eligible = [
        row for row in rows if float(row["validation_MSE_mean"]) <= threshold
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["normalized_weights"]["lag"])
            + float(row["normalized_weights"]["amplitude"]),
            -float(row["normalized_weights"]["ridge"]),
            -int(row["configuration_order"]),
        ),
    )
    return {
        "minimum": minimum,
        "threshold": threshold,
        "eligible_orders": [
            int(row["configuration_order"]) for row in eligible
        ],
        "selected": selected,
    }
