"""Thin PB1 adapter for exact CPU-FP64 spectral development fits."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re
import time

import numpy as np
import scipy.linalg

from ar_raphu.datasets.base import DynamicDataset
from ar_raphu.datasets.scaling import TrainOnlyStandardizer
from ar_raphu.datasets.windowing import build_windowed_task

from .amplitude_domain import AmplitudeDomain
from .gram_svd import gram_whitened_svd
from .penalties import normalized_second_difference
from .penalty_interval import (
    LogPenaltyInterval,
    automatic_penalty_interval,
    expand_penalty_interval,
    normalize_penalty_relative_to_gram,
    numerical_jitter,
)
from .predictive_rank import predictive_rank_profile
from .spline_basis import CenteredSplineBasis, clamped_knots, evaluate_basis


@dataclass(frozen=True, slots=True)
class PB1TensorBlock:
    train_matrix: np.ndarray
    validation_matrix: np.ndarray
    lag_basis: np.ndarray
    amplitude_basis: CenteredSplineBasis
    lag_gram: np.ndarray
    amplitude_gram: np.ndarray
    coefficient_slice: slice


@dataclass(frozen=True, slots=True)
class PB1PenaltyCandidate:
    lag_weight: float
    amplitude_weight: float
    ridge_weight: float
    validation_mse_mean: float
    validation_mse_by_group: tuple[float, ...]
    validation_mse_se: float
    effective_df: float
    relative_kkt_residual: float
    numerical_jitter: float
    configuration_order: int
    index_lag: int
    index_amplitude: int
    index_ridge: int
    coefficients: np.ndarray
    intercept: float


@dataclass(frozen=True, slots=True)
class PB1SpectralDevelopmentFit:
    selected: PB1PenaltyCandidate
    candidates: tuple[PB1PenaltyCandidate, ...]
    interval_history: tuple[dict[str, object], ...]
    penalty_status: str
    x_block: PB1TensorBlock
    ar_block: PB1TensorBlock
    train_target: np.ndarray
    validation_target: np.ndarray
    validation_groups: np.ndarray
    rank_audit: dict[str, object]
    history: tuple[int, int]
    horizon: int
    elapsed_seconds: float


def _lag_basis(history: int, kind: str, count: int | None) -> np.ndarray:
    if kind == "discrete_identity":
        return np.eye(history, dtype=np.float64)
    if kind != "cubic_bspline" or count is None:
        raise ValueError("Unsupported PB1 lag basis.")
    knots = clamped_knots(0.0, float(history - 1), count, 3)
    return evaluate_basis(np.arange(history), knots, 3)


def _tensor_matrix(
    histories: np.ndarray,
    *,
    fit_values: np.ndarray,
    lag_kind: str,
    lag_count: int | None,
    amplitude_count: int,
) -> tuple[np.ndarray, np.ndarray, CenteredSplineBasis, np.ndarray, np.ndarray]:
    values = np.asarray(histories, dtype=np.float64)
    fit = np.asarray(fit_values, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or not len(values) or not len(fit):
        raise ValueError("PB1 tensor block requires histories and train values.")
    lag = _lag_basis(values.shape[1], lag_kind, lag_count)
    domain = AmplitudeDomain.fit(
        fit,
        padding_fraction=0.10,
        core_quantiles=(0.01, 0.99),
    )
    amplitude = CenteredSplineBasis.fit(
        fit,
        n_basis=amplitude_count,
        degree=3,
        domain=domain,
        quantiles=(0.01, 0.99),
    )
    evaluation = amplitude.transform(values.reshape(-1)).reshape(
        len(values), values.shape[1], amplitude_count
    )
    matrix = np.einsum("la,nlb->nab", lag, evaluation, optimize=True).reshape(
        len(values), lag.shape[1] * amplitude_count
    )
    lag_gram = lag.T @ lag / len(lag)
    fit_amplitude = amplitude.transform(fit)
    amplitude_gram = fit_amplitude.T @ fit_amplitude / len(fit_amplitude)
    return matrix, lag, amplitude, lag_gram, amplitude_gram


def _block_penalties(blocks: tuple[PB1TensorBlock, ...]) -> tuple[np.ndarray, ...]:
    width = sum(block.coefficient_slice.stop - block.coefficient_slice.start for block in blocks)
    lag_penalty = np.zeros((width, width), dtype=np.float64)
    amplitude_penalty = np.zeros_like(lag_penalty)
    ridge_penalty = np.eye(width, dtype=np.float64)
    for block in blocks:
        lag_count = block.lag_gram.shape[0]
        amplitude_count = block.amplitude_gram.shape[0]
        section = block.coefficient_slice
        lag_penalty[section, section] = np.kron(
            normalized_second_difference(lag_count),
            np.eye(amplitude_count),
        )
        amplitude_penalty[section, section] = np.kron(
            np.eye(lag_count),
            normalized_second_difference(amplitude_count),
        )
    return lag_penalty, amplitude_penalty, ridge_penalty


def _group_labels(dataset_id: str, sequence_ids: np.ndarray) -> np.ndarray:
    labels: list[str] = []
    for sequence in map(str, sequence_ids):
        record = sequence.split(":", 1)[-1]
        if dataset_id == "pwh":
            match = re.match(r"Est-phase-(\d+)-amp-\d+$", record)
            if match is None:
                raise ValueError(f"Unexpected PWH record {record!r}.")
            labels.append(f"phase-{match.group(1)}")
        elif dataset_id == "whpn":
            labels.append(record)
        else:
            raise ValueError("PB1 grouped risk is implemented for PWH/WHPN.")
    return np.asarray(labels, dtype=object)


def _group_mse(
    target: np.ndarray, prediction: np.ndarray, groups: np.ndarray
) -> tuple[tuple[float, ...], float, float]:
    scores = tuple(
        float(np.mean((target[groups == group] - prediction[groups == group]) ** 2))
        for group in np.unique(groups)
    )
    mean = float(np.mean(scores))
    se = (
        float(np.std(scores, ddof=1) / math.sqrt(len(scores)))
        if len(scores) > 1
        else 0.0
    )
    return scores, mean, se


def _fit_candidate(
    *,
    train_matrix_centered: np.ndarray,
    validation_matrix: np.ndarray,
    train_target_centered: np.ndarray,
    validation_target: np.ndarray,
    validation_groups: np.ndarray,
    gram: np.ndarray,
    rhs: np.ndarray,
    feature_mean: np.ndarray,
    target_mean: float,
    normalized_penalties: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: tuple[float, float, float],
    indices: tuple[int, int, int],
    configuration_order: int,
) -> PB1PenaltyCandidate:
    penalty = sum(
        weight * component
        for weight, component in zip(weights, normalized_penalties, strict=True)
    )
    system_without_jitter = gram + penalty
    jitter = numerical_jitter(system_without_jitter)
    system = system_without_jitter + jitter * np.eye(len(gram))
    factor = scipy.linalg.cho_factor(system, lower=True, check_finite=True)
    coefficients = scipy.linalg.cho_solve(factor, rhs, check_finite=False)
    intercept = target_mean - float(feature_mean @ coefficients)
    validation_prediction = validation_matrix @ coefficients + intercept
    by_group, mean, se = _group_mse(
        validation_target, validation_prediction, validation_groups
    )
    residual = system @ coefficients - rhs
    relative_kkt = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
    )
    influence = scipy.linalg.cho_solve(factor, gram, check_finite=False)
    effective_df = float(np.trace(influence))
    return PB1PenaltyCandidate(
        lag_weight=weights[0],
        amplitude_weight=weights[1],
        ridge_weight=weights[2],
        validation_mse_mean=mean,
        validation_mse_by_group=by_group,
        validation_mse_se=se,
        effective_df=effective_df,
        relative_kkt_residual=relative_kkt,
        numerical_jitter=jitter,
        configuration_order=configuration_order,
        index_lag=indices[0],
        index_amplitude=indices[1],
        index_ridge=indices[2],
        coefficients=coefficients,
        intercept=intercept,
    )


def _select_one_se(
    candidates: list[PB1PenaltyCandidate],
) -> PB1PenaltyCandidate:
    minimum = min(
        candidates,
        key=lambda row: (row.validation_mse_mean, row.configuration_order),
    )
    threshold = minimum.validation_mse_mean + minimum.validation_mse_se
    eligible = [
        row for row in candidates if row.validation_mse_mean <= threshold
    ]
    return min(
        eligible,
        key=lambda row: (
            row.effective_df,
            -(row.lag_weight + row.amplitude_weight),
            row.ridge_weight,
            row.configuration_order,
        ),
    )


def _rank_audit(
    selected: PB1PenaltyCandidate,
    *,
    x_block: PB1TensorBlock,
    ar_block: PB1TensorBlock,
    validation_target: np.ndarray,
) -> dict[str, object]:
    x_width = x_block.coefficient_slice.stop - x_block.coefficient_slice.start
    theta_x = selected.coefficients[x_block.coefficient_slice].reshape(
        x_block.lag_gram.shape[0], x_block.amplitude_gram.shape[0]
    )
    spectrum = gram_whitened_svd(
        theta_x, x_block.lag_gram, x_block.amplitude_gram
    )
    ar_prediction = (
        ar_block.validation_matrix
        @ selected.coefficients[ar_block.coefficient_slice]
    )
    full_prediction = (
        x_block.validation_matrix @ theta_x.reshape(x_width)
        + ar_prediction
        + selected.intercept
    )
    variance = max(float(np.var(validation_target)), np.finfo(float).eps)
    full_mse = float(np.mean((validation_target - full_prediction) ** 2))
    ranks = min(len(spectrum.singular_values), 16)
    curves: list[float] = []
    rank_mse: list[float] = []
    for rank in range(1, ranks + 1):
        truncated = spectrum.truncate(rank)
        prediction = (
            x_block.validation_matrix @ truncated.reshape(x_width)
            + ar_prediction
            + selected.intercept
        )
        mse = float(np.mean((validation_target - prediction) ** 2))
        rank_mse.append(mse)
        curves.append(math.sqrt(max(mse - full_mse, 0.0) / variance))
    return {
        "structural_rank_claim_allowed": False,
        "predictive_svd_rank_claim_allowed": True,
        "external_kernel_only": True,
        "singular_values": spectrum.singular_values.tolist(),
        "normalized_spectral_energy": (
            spectrum.singular_values**2
            / max(
                float(np.sum(spectrum.singular_values**2)),
                np.finfo(float).eps,
            )
        ).tolist(),
        "full_validation_mse": full_mse,
        "rank_validation_mse": rank_mse,
        "normalized_excess_rmse_curve": curves,
        "predictive_effective_ranks": {
            str(budget): rank
            for budget, rank in predictive_rank_profile(
                np.asarray(curves), (0.10, 0.05, 0.02)
            ).items()
        },
    }


def fit_pb1_shared_history_spectral(
    dataset: DynamicDataset,
    *,
    L_x: int,
    L_y: int,
    horizon: int = 1,
    lag_kind: str = "discrete_identity",
    lag_count: int | None = None,
    amplitude_count: int = 16,
    grid_points: int = 7,
    maximum_expansions: int = 2,
) -> PB1SpectralDevelopmentFit:
    """Fit H3 full spectral XAR at one preregistered resolution."""

    started = time.perf_counter()
    if np.any(dataset.split == "test"):
        raise PermissionError("PB1 spectral development refuses test rows.")
    dataset_id = str(dataset.metadata.get("dataset_id", ""))
    standardizer = TrainOnlyStandardizer.fit(dataset)
    scaled = standardizer.transform(dataset)
    task = build_windowed_task(
        scaled,
        target=0,
        horizon=horizon,
        L_x=L_x,
        L_y=L_y,
        include_splits=("train", "validation"),
    )
    train_mask = task.split == "train"
    validation_mask = task.split == "validation"
    fit_x = scaled.x[scaled.split == "train", 0]
    fit_y = scaled.y[scaled.split == "train", 0]
    x_matrix, x_lag, x_amp, x_lag_gram, x_amp_gram = _tensor_matrix(
        task.x_history[:, 0, :],
        fit_values=fit_x,
        lag_kind=lag_kind,
        lag_count=lag_count,
        amplitude_count=amplitude_count,
    )
    ar_matrix, ar_lag, ar_amp, ar_lag_gram, ar_amp_gram = _tensor_matrix(
        task.y_history,
        fit_values=fit_y,
        lag_kind=lag_kind,
        lag_count=lag_count,
        amplitude_count=amplitude_count,
    )
    x_width = x_matrix.shape[1]
    ar_width = ar_matrix.shape[1]
    x_block = PB1TensorBlock(
        train_matrix=x_matrix[train_mask],
        validation_matrix=x_matrix[validation_mask],
        lag_basis=x_lag,
        amplitude_basis=x_amp,
        lag_gram=x_lag_gram,
        amplitude_gram=x_amp_gram,
        coefficient_slice=slice(0, x_width),
    )
    ar_block = PB1TensorBlock(
        train_matrix=ar_matrix[train_mask],
        validation_matrix=ar_matrix[validation_mask],
        lag_basis=ar_lag,
        amplitude_basis=ar_amp,
        lag_gram=ar_lag_gram,
        amplitude_gram=ar_amp_gram,
        coefficient_slice=slice(x_width, x_width + ar_width),
    )
    train_matrix = np.column_stack(
        (x_block.train_matrix, ar_block.train_matrix)
    )
    validation_matrix = np.column_stack(
        (x_block.validation_matrix, ar_block.validation_matrix)
    )
    train_target = task.target[train_mask]
    validation_target = task.target[validation_mask]
    validation_groups = _group_labels(
        dataset_id, task.sequence_id[validation_mask]
    )
    feature_mean = train_matrix.mean(axis=0)
    target_mean = float(train_target.mean())
    centered_matrix = train_matrix - feature_mean
    centered_target = train_target - target_mean
    gram = centered_matrix.T @ centered_matrix / len(centered_matrix)
    rhs = centered_matrix.T @ centered_target / len(centered_matrix)
    raw_penalties = _block_penalties((x_block, ar_block))
    normalized = tuple(
        normalize_penalty_relative_to_gram(component, gram).normalized
        for component in raw_penalties
    )
    intervals = [
        automatic_penalty_interval(component, gram)
        for component in normalized
    ]
    all_rows: list[PB1PenaltyCandidate] = []
    interval_history: list[dict[str, object]] = []
    selected: PB1PenaltyCandidate | None = None
    status = "PENALTY_INTERVAL_NOT_CERTIFIED"
    configuration_order = 0
    for expansion_round in range(maximum_expansions + 1):
        grids = [interval.grid(grid_points) for interval in intervals]
        round_rows: list[PB1PenaltyCandidate] = []
        for indices in itertools.product(range(grid_points), repeat=3):
            weights = tuple(
                float(grids[axis][indices[axis]]) for axis in range(3)
            )
            row = _fit_candidate(
                train_matrix_centered=centered_matrix,
                validation_matrix=validation_matrix,
                train_target_centered=centered_target,
                validation_target=validation_target,
                validation_groups=validation_groups,
                gram=gram,
                rhs=rhs,
                feature_mean=feature_mean,
                target_mean=target_mean,
                normalized_penalties=normalized,
                weights=weights,
                indices=indices,
                configuration_order=configuration_order,
            )
            round_rows.append(row)
            all_rows.append(row)
            configuration_order += 1
        selected = _select_one_se(round_rows)
        boundary_axes = []
        selected_indices = (
            selected.index_lag,
            selected.index_amplitude,
            selected.index_ridge,
        )
        for axis, index in enumerate(selected_indices):
            if index == 0:
                boundary_axes.append((axis, "lower"))
            elif index == grid_points - 1:
                boundary_axes.append((axis, "upper"))
        interval_history.append(
            {
                "round": expansion_round,
                "intervals": [
                    {
                        "lower": interval.lower,
                        "upper": interval.upper,
                        "expansion_count": interval.expansion_count,
                    }
                    for interval in intervals
                ],
                "selected_indices": list(selected_indices),
                "boundary_axes": [
                    {"axis": axis, "side": side}
                    for axis, side in boundary_axes
                ],
            }
        )
        if not boundary_axes:
            status = "PENALTY_INTERVAL_CERTIFIED"
            break
        if expansion_round == maximum_expansions:
            break
        for axis, side in boundary_axes:
            intervals[axis] = expand_penalty_interval(
                intervals[axis], boundary=side
            )
    if selected is None:
        raise AssertionError("PB1 spectral penalty search produced no candidate.")
    rank_audit = _rank_audit(
        selected,
        x_block=x_block,
        ar_block=ar_block,
        validation_target=validation_target,
    )
    return PB1SpectralDevelopmentFit(
        selected=selected,
        candidates=tuple(all_rows),
        interval_history=tuple(interval_history),
        penalty_status=status,
        x_block=x_block,
        ar_block=ar_block,
        train_target=train_target,
        validation_target=validation_target,
        validation_groups=validation_groups,
        rank_audit=rank_audit,
        history=(L_x, L_y),
        horizon=horizon,
        elapsed_seconds=time.perf_counter() - started,
    )
