"""Monotone-restart FISTA for the frozen standardized diagnostic gate path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GateSolution:
    lambda_value: float
    intercept: float
    gates: np.ndarray
    objective: float
    iterations: int
    converged: bool
    kkt_residual: float


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def lambda_maximum(design: np.ndarray, target: np.ndarray) -> float:
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    centered_design = design - design.mean(axis=0)
    centered_target = target - target.mean()
    return float(
        np.max(np.abs(centered_design.T @ centered_target)) / len(target)
    )


def _objective(
    design: np.ndarray,
    target: np.ndarray,
    intercept: float,
    gates: np.ndarray,
    lambda_value: float,
) -> float:
    residual = target - intercept - design @ gates
    return float(
        0.5 * np.mean(residual**2)
        + lambda_value * np.abs(gates).sum()
    )


def _kkt_residual(
    design: np.ndarray,
    target: np.ndarray,
    intercept: float,
    gates: np.ndarray,
    lambda_value: float,
    *,
    zero_threshold: float = 1.0e-12,
) -> float:
    gradient = design.T @ (intercept + design @ gates - target) / len(target)
    active = np.abs(gates) > zero_threshold
    residuals = [abs(float(np.mean(intercept + design @ gates - target)))]
    if active.any():
        residuals.append(
            float(
                np.max(
                    np.abs(
                        gradient[active]
                        + lambda_value * np.sign(gates[active])
                    )
                )
            )
        )
    if (~active).any():
        residuals.append(
            float(
                np.max(
                    np.maximum(
                        np.abs(gradient[~active]) - lambda_value,
                        0.0,
                    )
                )
            )
        )
    return max(residuals)


def solve_gate_fista(
    design: np.ndarray,
    target: np.ndarray,
    lambda_value: float,
    *,
    initial_gates: np.ndarray | None = None,
    max_iterations: int = 10000,
    tolerance: float = 1.0e-9,
) -> GateSolution:
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if design.ndim != 2 or target.shape != (design.shape[0],):
        raise ValueError("Expected design [samples,variables] and target [samples].")
    if not np.isfinite(design).all() or not np.isfinite(target).all():
        raise ValueError("FISTA inputs must be finite.")
    if lambda_value < 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("Invalid FISTA control value.")

    design_mean = design.mean(axis=0)
    target_mean = float(target.mean())
    centered_design = design - design_mean
    centered_target = target - target_mean
    gram = centered_design.T @ centered_design / len(target)
    lipschitz = float(np.linalg.eigvalsh(gram).max())
    if lipschitz <= 0:
        gates = np.zeros(design.shape[1], dtype=np.float64)
        return GateSolution(
            lambda_value=lambda_value,
            intercept=target_mean,
            gates=gates,
            objective=_objective(design, target, target_mean, gates, lambda_value),
            iterations=0,
            converged=True,
            kkt_residual=0.0,
        )

    gates = (
        np.zeros(design.shape[1], dtype=np.float64)
        if initial_gates is None
        else np.asarray(initial_gates, dtype=np.float64).copy()
    )
    if gates.shape != (design.shape[1],):
        raise ValueError("initial_gates has the wrong shape.")
    extrapolated = gates.copy()
    momentum = 1.0
    intercept = target_mean - float(design_mean @ gates)
    previous_objective = _objective(
        design, target, intercept, gates, lambda_value
    )
    converged = False

    for iteration in range(1, max_iterations + 1):
        gradient = (
            centered_design.T
            @ (centered_design @ extrapolated - centered_target)
            / len(target)
        )
        candidate = soft_threshold(
            extrapolated - gradient / lipschitz,
            lambda_value / lipschitz,
        )
        candidate_intercept = target_mean - float(design_mean @ candidate)
        candidate_objective = _objective(
            design,
            target,
            candidate_intercept,
            candidate,
            lambda_value,
        )
        if candidate_objective > previous_objective + 1.0e-14:
            momentum = 1.0
            extrapolated = gates.copy()
            gradient = (
                centered_design.T
                @ (centered_design @ extrapolated - centered_target)
                / len(target)
            )
            candidate = soft_threshold(
                extrapolated - gradient / lipschitz,
                lambda_value / lipschitz,
            )
            candidate_intercept = target_mean - float(design_mean @ candidate)
            candidate_objective = _objective(
                design,
                target,
                candidate_intercept,
                candidate,
                lambda_value,
            )

        relative_change = float(
            np.linalg.norm(candidate - gates)
            / max(1.0, np.linalg.norm(gates))
        )
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        extrapolated = candidate + (
            (momentum - 1.0) / next_momentum
        ) * (candidate - gates)
        gates = candidate
        intercept = candidate_intercept
        momentum = next_momentum
        previous_objective = candidate_objective
        if relative_change < tolerance:
            converged = True
            break

    return GateSolution(
        lambda_value=float(lambda_value),
        intercept=float(intercept),
        gates=gates,
        objective=float(previous_objective),
        iterations=iteration,
        converged=converged,
        kkt_residual=_kkt_residual(
            design, target, intercept, gates, lambda_value
        ),
    )


def solve_gate_path(
    design: np.ndarray,
    target: np.ndarray,
    lambda_ratios: list[float],
    *,
    max_iterations: int = 10000,
    tolerance: float = 1.0e-9,
) -> list[GateSolution]:
    maximum = lambda_maximum(design, target)
    previous = np.zeros(np.asarray(design).shape[1], dtype=np.float64)
    solutions: list[GateSolution] = []
    for ratio in lambda_ratios:
        solution = solve_gate_fista(
            design,
            target,
            float(ratio) * maximum,
            initial_gates=previous,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        solutions.append(solution)
        previous = solution.gates
    return solutions
