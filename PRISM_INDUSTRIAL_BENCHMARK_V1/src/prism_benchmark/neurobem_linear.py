"""FP64 linear operators and ERA realization for the NeuroBEM PRISM audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import linalg

from .neurobem_data import SegmentData, body_context, generalized_targets, motor_thrust_proxy


@dataclass(frozen=True)
class RidgeContract:
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    coefficient_standardized: np.ndarray
    condition_number: float
    relative_kkt_residual: float
    fit_rows: int

    @property
    def coefficient_physical(self) -> np.ndarray:
        return self.coefficient_standardized / self.feature_scale[:, None]

    @property
    def intercept_physical(self) -> np.ndarray:
        return self.target_mean - self.feature_mean @ self.coefficient_physical


@dataclass(frozen=True)
class EraContract:
    order: int
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    singular_values: np.ndarray
    spectral_radius: float


@dataclass(frozen=True)
class ContextContract:
    candidate: str
    ridge: RidgeContract
    speed_knots: np.ndarray | None


def k_design(segment: SegmentData, history_steps: int, support_start: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = motor_thrust_proxy(segment)
    n = u.shape[0]
    start = max(history_steps, history_steps if support_start is None else support_start)
    if start >= n:
        return np.empty((0, history_steps * 4)), np.empty((0, 4)), np.empty(0, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(u, history_steps, axis=0)
    offset = start - history_steps
    count = n - start
    selected = windows[offset : offset + count]
    x = selected[:, :, ::-1].transpose(0, 2, 1).reshape(count, history_steps * 4)
    rows = np.arange(start, n, dtype=np.int64)
    return x, rows, rows


def segment_k_xy(
    segment: SegmentData,
    history_steps: int,
    mass: float,
    inertia: Sequence[float],
    support_start: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, _, rows = k_design(segment, history_steps, support_start=support_start)
    y = generalized_targets(segment, mass, inertia)[rows]
    return x, y, rows


def concatenate_k_xy(
    segments: Iterable[SegmentData],
    history_steps: int,
    mass: float,
    inertia: Sequence[float],
    support_start: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    provenance: list[tuple[str, np.ndarray]] = []
    for segment in segments:
        x, y, rows = segment_k_xy(segment, history_steps, mass, inertia, support_start)
        if len(rows):
            xs.append(x)
            ys.append(y)
            provenance.append((segment.record.segment_id, rows))
    if not xs:
        raise ValueError("no legal K rows")
    return np.concatenate(xs), np.concatenate(ys), provenance


def fit_numerical_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha_grid: Sequence[float],
    maximum_condition_number: float,
    maximum_relative_kkt_residual: float,
) -> RidgeContract:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("invalid ridge arrays")
    feature_mean = x.mean(axis=0, dtype=np.float64)
    feature_scale = np.sqrt(np.mean(np.square(x - feature_mean), axis=0, dtype=np.float64))
    feature_scale = np.where(feature_scale > np.finfo(np.float64).eps, feature_scale, 1.0)
    target_mean = y.mean(axis=0, dtype=np.float64)
    xs = (x - feature_mean) / feature_scale
    ys = y - target_mean
    gram = (xs.T @ xs) / x.shape[0]
    rhs = (xs.T @ ys) / x.shape[0]
    identity = np.eye(gram.shape[0], dtype=np.float64)
    failures: list[tuple[float, float, float]] = []
    for alpha in alpha_grid:
        penalized = gram + float(alpha) * identity
        condition = float(np.linalg.cond(penalized))
        try:
            coefficient = linalg.solve(penalized, rhs, assume_a="pos", check_finite=False)
        except linalg.LinAlgError:
            failures.append((float(alpha), condition, float("inf")))
            continue
        residual = penalized @ coefficient - rhs
        relative_kkt = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).eps))
        if condition <= maximum_condition_number and relative_kkt <= maximum_relative_kkt_residual:
            return RidgeContract(
                alpha=float(alpha),
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                target_mean=target_mean,
                coefficient_standardized=coefficient,
                condition_number=condition,
                relative_kkt_residual=relative_kkt,
                fit_rows=int(x.shape[0]),
            )
        failures.append((float(alpha), condition, relative_kkt))
    raise np.linalg.LinAlgError(f"NO_NUMERICALLY_CERTIFIED_RIDGE:{failures}")


def predict_ridge(contract: RidgeContract, x: np.ndarray) -> np.ndarray:
    return contract.target_mean + ((x - contract.feature_mean) / contract.feature_scale) @ contract.coefficient_standardized


def normalized_mse(y: np.ndarray, prediction: np.ndarray, fit_variance: np.ndarray) -> float:
    variance = np.maximum(np.asarray(fit_variance, dtype=np.float64), np.finfo(float).eps)
    return float(np.mean(np.mean(np.square(y - prediction), axis=0) / variance))


def guarded_one_se(
    losses: dict[object, Sequence[float]],
    complexity_order: Sequence[object],
    maximum_relative_regret: float,
) -> dict[str, object]:
    means = {key: float(np.mean(value)) for key, value in losses.items()}
    ses = {
        key: float(np.std(value, ddof=1) / np.sqrt(len(value))) if len(value) > 1 else 0.0
        for key, value in losses.items()
    }
    best = min(means, key=means.get)
    threshold = means[best] + ses[best]
    regret = means[best] * (1.0 + maximum_relative_regret)
    eligible = {key for key in means if means[key] <= threshold and means[key] <= regret}
    selected = next(key for key in complexity_order if key in eligible)
    return {
        "selected": selected,
        "best_mean": best,
        "threshold": threshold,
        "mean_losses": means,
        "standard_errors": ses,
        "eligible": [key for key in complexity_order if key in eligible],
    }


def markov_parameters(contract: RidgeContract, history_steps: int, input_count: int = 4) -> np.ndarray:
    coefficient = contract.coefficient_physical
    if coefficient.shape[0] != history_steps * input_count:
        raise ValueError("coefficient does not match history")
    return coefficient.reshape(history_steps, input_count, -1).transpose(0, 2, 1)


def era_from_markov(markov: np.ndarray, order: int, block_rows: int) -> EraContract:
    if markov.ndim != 3:
        raise ValueError("markov must be lag x output x input")
    lag_count, output_count, input_count = markov.shape
    block_columns = lag_count - block_rows
    if block_rows < 1 or block_columns < 1 or block_rows + block_columns > lag_count:
        raise ValueError("invalid block Hankel dimensions")
    h0 = np.block([[markov[i + j] for j in range(block_columns)] for i in range(block_rows)])
    shifted = np.concatenate((markov[1:], np.zeros_like(markov[:1])), axis=0)
    h1 = np.block([[shifted[i + j] for j in range(block_columns)] for i in range(block_rows)])
    u, singular, vt = linalg.svd(h0, full_matrices=False, check_finite=False)
    if order > len(singular) or singular[order - 1] <= np.finfo(float).eps * singular[0]:
        raise np.linalg.LinAlgError("ERA_ORDER_EXCEEDS_NUMERICAL_RANK")
    ur = u[:, :order]
    sr = singular[:order]
    vr = vt[:order, :].T
    sqrt_s = np.sqrt(sr)
    inv_sqrt = 1.0 / sqrt_s
    a = (inv_sqrt[:, None] * (ur.T @ h1 @ vr)) * inv_sqrt[None, :]
    observability = ur * sqrt_s[None, :]
    controllability = sqrt_s[:, None] * vr.T
    c = observability[:output_count, :]
    b = controllability[:, :input_count]
    d = np.zeros((output_count, input_count), dtype=np.float64)
    spectral_radius = float(np.max(np.abs(linalg.eigvals(a))))
    return EraContract(order, a, b, c, d, singular, spectral_radius)


def simulate_era(contract: EraContract, u: np.ndarray) -> np.ndarray:
    state = np.zeros(contract.order, dtype=np.float64)
    output = np.empty((u.shape[0], contract.C.shape[0]), dtype=np.float64)
    for index in range(u.shape[0]):
        output[index] = contract.C @ state + contract.D @ u[index]
        state = contract.A @ state + contract.B @ u[index]
    return output


def signed_quadratic_context(context: np.ndarray, k_prediction: np.ndarray) -> np.ndarray:
    velocity = context[:, :3]
    rate = context[:, 3:6]
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    cross = np.column_stack((velocity[:, 0] * velocity[:, 1], velocity[:, 0] * velocity[:, 2], velocity[:, 1] * velocity[:, 2]))
    return np.column_stack(
        (
            velocity * np.abs(velocity),
            speed * velocity,
            rate * np.abs(rate),
            k_prediction * np.abs(k_prediction),
            cross,
        )
    )


def restricted_cubic_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    knots = np.unique(np.asarray(knots, dtype=np.float64))
    if len(knots) < 4 or knots[-1] <= knots[-2]:
        raise ValueError("insufficient distinct spline knots")
    scale = max(knots[-1] - knots[0], np.finfo(float).eps) ** 2
    columns = [x]
    for knot in knots[:-2]:
        d = np.maximum(x - knot, 0.0) ** 3
        d -= np.maximum(x - knots[-2], 0.0) ** 3 * (knots[-1] - knot) / (knots[-1] - knots[-2])
        d += np.maximum(x - knots[-1], 0.0) ** 3 * (knots[-2] - knot) / (knots[-1] - knots[-2])
        columns.append(d / scale)
    return np.column_stack(columns)


def w_features(
    candidate: str,
    context: np.ndarray,
    k_prediction: np.ndarray,
    *,
    speed_knots: np.ndarray | None = None,
) -> np.ndarray:
    if candidate == "SIGNED_QUADRATIC_AERO_CONTEXT":
        return signed_quadratic_context(context, k_prediction)
    if candidate.startswith("NATURAL_CUBIC_SPEED_CONTEXT_K"):
        if speed_knots is None:
            raise ValueError("natural cubic candidate requires speed knots")
        speed = np.linalg.norm(context[:, :3], axis=1)
        return np.column_stack((context, k_prediction, restricted_cubic_basis(speed, speed_knots)))
    raise ValueError(f"unknown W candidate: {candidate}")


def fit_w_candidate(
    candidate: str,
    context: np.ndarray,
    k_prediction: np.ndarray,
    residual_target: np.ndarray,
    alpha_grid: Sequence[float],
    maximum_condition_number: float,
    maximum_relative_kkt_residual: float,
) -> ContextContract:
    knots: np.ndarray | None = None
    if candidate.startswith("NATURAL_CUBIC_SPEED_CONTEXT_K"):
        count = int(candidate.rsplit("K", 1)[1])
        quantiles = np.linspace(0.0, 1.0, count + 2)
        knots = np.unique(np.quantile(np.linalg.norm(context[:, :3], axis=1), quantiles))
        if len(knots) < 4:
            raise ValueError("W_SPEED_SUPPORT_DEGENERATE")
    features = w_features(candidate, context, k_prediction, speed_knots=knots)
    ridge = fit_numerical_ridge(
        features,
        residual_target,
        alpha_grid,
        maximum_condition_number,
        maximum_relative_kkt_residual,
    )
    return ContextContract(candidate=candidate, ridge=ridge, speed_knots=knots)


def predict_w(contract: ContextContract, context: np.ndarray, k_prediction: np.ndarray) -> np.ndarray:
    features = w_features(contract.candidate, context, k_prediction, speed_knots=contract.speed_knots)
    return predict_ridge(contract.ridge, features)


def context_at_targets(segment: SegmentData, target_rows: np.ndarray) -> np.ndarray:
    if np.any(target_rows < 1):
        raise ValueError("context requires target row at least one")
    return body_context(segment)[target_rows - 1]
