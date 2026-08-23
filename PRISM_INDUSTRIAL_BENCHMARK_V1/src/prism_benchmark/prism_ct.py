from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping
import math

import numpy as np


@dataclass(frozen=True)
class CTBasisConfig:
    dt_seconds: float = 2.0
    taus_seconds: tuple[float, ...] = (
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
        1200.0,
        2400.0,
        4800.0,
        7200.0,
    )
    ridge_lambda: float = 1.0
    condition_hard_fail: float = 1.0e8
    support_z_limit: float = 5.0
    support_fraction_min: float = 0.95

    def validate(self) -> None:
        if self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        if not self.taus_seconds:
            raise ValueError("taus_seconds cannot be empty")
        taus = np.asarray(self.taus_seconds, dtype=np.float64)
        if np.any(~np.isfinite(taus)) or np.any(taus <= 0):
            raise ValueError("all time constants must be finite and positive")
        if np.any(np.diff(taus) <= 0):
            raise ValueError("taus_seconds must be strictly increasing")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")


@dataclass(frozen=True)
class FeatureAudit:
    n_rows: int
    n_features: int
    matrix_rank: int
    standardized_condition_number: float
    max_abs_offdiag_correlation: float
    finite: bool
    passed_conditioning: bool


@dataclass(frozen=True)
class SupportAudit:
    n_rows: int
    fraction_within_z_limit: float
    worst_feature_fraction_within_z_limit: float
    max_abs_standardized_value: float
    passed: bool


@dataclass(frozen=True)
class RidgeState:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float


@dataclass(frozen=True)
class CandidateResult:
    name: str
    validation_block_mse: tuple[float, ...]
    validation_mse_mean: float
    validation_mse_se: float
    complexity: int
    feature_audit: FeatureAudit | None
    eligible: bool


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("expected 1D or 2D array")
    if len(x) < 2:
        raise ValueError("at least two rows are required")
    if not np.all(np.isfinite(x)):
        raise ValueError("input contains non-finite values")
    return x


def stable_ct_states(x: np.ndarray, config: CTBasisConfig) -> np.ndarray:
    """Exact ZOH discretization of independent stable first-order CT modes.

    For every channel x_j and time constant tau_r:
        dz/dt = -(1/tau_r) z + (1/tau_r) x

    The exact zero-order-hold update is:
        z[k] = a_r z[k-1] + (1-a_r) x[k], a_r = exp(-dt/tau_r)

    Returns shape (n_rows, n_channels, n_taus).
    """
    config.validate()
    x = _as_2d(x)
    n, p = x.shape
    taus = np.asarray(config.taus_seconds, dtype=np.float64)
    a = np.exp(-config.dt_seconds / taus)
    z = np.empty((n, p, len(taus)), dtype=np.float64)
    z[0, :, :] = x[0, :, None]
    for t in range(1, n):
        z[t] = a[None, :] * z[t - 1] + (1.0 - a)[None, :] * x[t, :, None]
    return z


def ct_absolute_features(
    x: np.ndarray,
    config: CTBasisConfig,
    include_current: bool = True,
) -> np.ndarray:
    x = _as_2d(x)
    z = stable_ct_states(x, config)
    blocks = [x] if include_current else []
    blocks.extend(z[:, :, r] for r in range(z.shape[2]))
    return np.concatenate(blocks, axis=1)


def ct_multires_features(
    x: np.ndarray,
    config: CTBasisConfig,
    include_current: bool = True,
) -> np.ndarray:
    """Causal scale-increment representation.

    With include_current=True:
        [x, x-z_tau1, z_tau1-z_tau2, ..., z_tau(R-1)-z_tauR]

    With include_current=False only the scale increments are returned. The increment-only
    form is the default inside ``aligned_temporal_blocks`` because it avoids repeating the
    instantaneous state and materially improves conditioning on the silicon data.
    """
    x = _as_2d(x)
    z = stable_ct_states(x, config)
    blocks = [x] if include_current else []
    blocks.append(x - z[:, :, 0])
    blocks.extend(z[:, :, r] - z[:, :, r + 1] for r in range(z.shape[2] - 1))
    return np.concatenate(blocks, axis=1)


def delay_steps(config: CTBasisConfig) -> np.ndarray:
    return np.unique(
        np.maximum(
            1,
            np.rint(np.asarray(config.taus_seconds) / config.dt_seconds).astype(int),
        )
    )


def delay_features(
    x: np.ndarray,
    config: CTBasisConfig,
    include_current: bool = True,
) -> tuple[np.ndarray, int]:
    x = _as_2d(x)
    lags = delay_steps(config)
    start = int(lags.max())
    blocks = [x[start:]] if include_current else []
    for lag in lags:
        blocks.append(x[start - lag : len(x) - lag])
    return np.concatenate(blocks, axis=1), start


def aligned_temporal_blocks(
    x: np.ndarray,
    config: CTBasisConfig,
) -> tuple[dict[str, np.ndarray], int]:
    """Build all v0.2 temporal branches on one shared causal support.

    Branch semantics:
      delay: instantaneous state + discrete lags matched to the CT time constants.
      ct_absolute: stable CT low-pass states only.
      ct_multires: increment-only stable CT scale differences.
      delay_ct_multires: early-fusion ablation, retained for conditioning experiments.

    The early-fusion branch is intentionally auditable rather than automatically trusted;
    on the silicon data it can approach the conditioning threshold and should not bypass A.
    """
    x = _as_2d(x)
    delay, start = delay_features(x, config)
    absolute = ct_absolute_features(x, config, include_current=False)[start:]
    multires = ct_multires_features(x, config, include_current=False)[start:]
    blocks = {
        "delay": delay,
        "ct_absolute": absolute,
        "ct_multires": multires,
        "delay_ct_multires": np.concatenate([delay, multires], axis=1),
    }
    return blocks, start


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _as_2d(x)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (x - mean) / scale, mean, scale


def feature_audit(x: np.ndarray, config: CTBasisConfig) -> FeatureAudit:
    """Numerical certificate for a candidate temporal branch.

    Conditioning is measured on standardized features via SVD. This deliberately audits the
    representation before Ridge regularization can hide a badly conditioned basis.
    """
    x = _as_2d(x)
    finite = bool(np.all(np.isfinite(x)))
    if not finite:
        return FeatureAudit(len(x), x.shape[1], 0, math.inf, math.inf, False, False)
    xs, _, _ = _standardize(x)
    singular = np.linalg.svd(xs, compute_uv=False)
    tol = np.finfo(np.float64).eps * max(xs.shape) * (singular[0] if len(singular) else 0.0)
    rank = int(np.sum(singular > tol))
    condition = (
        math.inf
        if len(singular) == 0 or singular[-1] <= tol
        else float(singular[0] / singular[-1])
    )
    if xs.shape[1] <= 1:
        max_corr = 0.0
    else:
        corr = np.corrcoef(xs, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)
        max_corr = float(np.max(np.abs(corr)))
    passed = (
        finite
        and rank == xs.shape[1]
        and condition <= config.condition_hard_fail
    )
    return FeatureAudit(
        len(x),
        x.shape[1],
        rank,
        condition,
        max_corr,
        finite,
        passed,
    )


def support_audit(
    reference_x: np.ndarray,
    candidate_x: np.ndarray,
    config: CTBasisConfig,
) -> SupportAudit:
    """Input-only support audit suitable for OOD/cross-run diagnostics.

    No target value from the candidate domain is used. A candidate row is in support only if
    every standardized feature stays within ``support_z_limit`` of the reference fit domain.
    """
    reference_x = _as_2d(reference_x)
    candidate_x = _as_2d(candidate_x)
    if reference_x.shape[1] != candidate_x.shape[1]:
        raise ValueError("reference/candidate feature dimensions differ")
    _, mean, scale = _standardize(reference_x)
    z = np.abs((candidate_x - mean) / scale)
    within = z <= config.support_z_limit
    row_fraction = float(np.mean(np.all(within, axis=1)))
    per_feature = np.mean(within, axis=0)
    worst_feature = float(np.min(per_feature))
    max_abs = float(np.max(z))
    passed = (
        row_fraction >= config.support_fraction_min
        and worst_feature >= config.support_fraction_min
    )
    return SupportAudit(
        len(candidate_x),
        row_fraction,
        worst_feature,
        max_abs,
        passed,
    )


def fit_ridge(
    x: np.ndarray,
    target: np.ndarray,
    config: CTBasisConfig,
) -> RidgeState:
    x = _as_2d(x)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(x) != len(target):
        raise ValueError("feature/target length mismatch")
    xs, mean, scale = _standardize(x)
    intercept = float(target.mean())
    centered = target - intercept
    alpha = float(config.ridge_lambda * len(xs))
    gram = xs.T @ xs + alpha * np.eye(xs.shape[1], dtype=np.float64)
    coef = np.linalg.solve(gram, xs.T @ centered)
    return RidgeState(mean, scale, coef, intercept, alpha)


def predict_ridge(model: RidgeState, x: np.ndarray) -> np.ndarray:
    x = _as_2d(x)
    return ((x - model.mean) / model.scale) @ model.coef + model.intercept


def contiguous_block_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_blocks: int = 5,
) -> tuple[float, ...]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    blocks = np.array_split(np.arange(len(y_true)), n_blocks)
    return tuple(
        float(np.mean((y_true[index] - y_pred[index]) ** 2))
        for index in blocks
        if len(index)
    )


def candidate_summary(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    complexity: int,
    audit: FeatureAudit | None,
    eligible: bool = True,
) -> CandidateResult:
    losses = contiguous_block_mse(y_true, y_pred)
    array = np.asarray(losses, dtype=np.float64)
    mean = float(array.mean())
    se = (
        float(array.std(ddof=1) / math.sqrt(len(array)))
        if len(array) > 1
        else 0.0
    )
    return CandidateResult(name, losses, mean, se, complexity, audit, eligible)


def select_one_se(candidates: Iterable[CandidateResult]) -> CandidateResult:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.eligible and np.isfinite(candidate.validation_mse_mean)
    ]
    if not eligible:
        raise ValueError("no eligible candidate")
    best = min(eligible, key=lambda candidate: candidate.validation_mse_mean)
    threshold = best.validation_mse_mean + best.validation_mse_se
    within = [
        candidate
        for candidate in eligible
        if candidate.validation_mse_mean <= threshold
    ]
    return min(
        within,
        key=lambda candidate: (
            candidate.complexity,
            candidate.validation_mse_mean,
            candidate.name,
        ),
    )


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    persistence = np.asarray(persistence, dtype=np.float64)
    error = y_true - y_pred
    persistence_error = y_true - persistence
    mse = float(np.mean(error ** 2))
    persistence_mse = float(np.mean(persistence_error ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = (
        float(1.0 - np.sum(error ** 2) / denominator)
        if denominator > 0
        else math.nan
    )
    skill = (
        float(1.0 - mse / persistence_mse)
        if persistence_mse > 0
        else math.nan
    )
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "persistence_skill_mse": skill,
    }


def config_dict(config: CTBasisConfig) -> Mapping[str, object]:
    return asdict(config)
