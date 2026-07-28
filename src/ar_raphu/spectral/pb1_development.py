"""Thin PB1 adapter for exact CPU-FP64 spectral development fits."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re
import time
import warnings

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
)
from .predictive_rank import predictive_rank_profile
from .pb1_repair import (
    automatic_time_block_length,
    positive_lower_expansion_required,
    solve_pb1_system,
    zero_inclusive_penalty_grid,
)
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
    solver_diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class PB1SpectralDevelopmentFit:
    selected: PB1PenaltyCandidate
    candidates: tuple[PB1PenaltyCandidate, ...]
    interval_history: tuple[dict[str, object], ...]
    penalty_status: str
    x_block: PB1TensorBlock | None
    ar_block: PB1TensorBlock | None
    train_target: np.ndarray
    validation_target: np.ndarray
    train_matrix: np.ndarray
    train_groups: np.ndarray
    validation_groups: np.ndarray
    selected_penalty: np.ndarray
    rank_audit: dict[str, object]
    history: tuple[int, int]
    track: str
    horizon: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PB1RankBootstrap:
    replicates: int
    seed: int
    singular_value_quantiles: dict[str, np.ndarray]
    normalized_energy_quantiles: dict[str, np.ndarray]
    spectral_tail_budget_rank_frequencies: dict[str, dict[str, int]]
    maximum_relative_kkt_residual: float
    elapsed_seconds: float
    automatic_block_length: int | None


@dataclass(frozen=True, slots=True)
class PB1FreeRunMetrics:
    initialization_length: int
    scored_samples: int
    mse_standardized: float
    rmse_standardized: float
    mse_original_units: float
    rmse_original_units: float
    mse_by_sequence_standardized: dict[str, float]
    status: str


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


def _group_labels(
    dataset_id: str,
    sequence_ids: np.ndarray,
    target: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(sequence_ids, dtype=object)
    labels = np.empty(len(values), dtype=object)
    if dataset_id in {"pwh", "whpn"}:
        for index, sequence in enumerate(map(str, values)):
            record = sequence.split(":", 1)[-1]
            if dataset_id == "pwh":
                match = re.match(r"Est-phase-(\d+)-amp-\d+$", record)
                if match is None:
                    raise ValueError(f"Unexpected PWH record {record!r}.")
                labels[index] = f"phase-{match.group(1)}"
            else:
                labels[index] = record
        return labels
    if dataset_id not in {"cascaded_tanks", "silverbox"}:
        raise ValueError(f"Unsupported PB1 grouped-risk dataset {dataset_id!r}.")
    if target is None:
        raise ValueError("Time-series grouped risk requires the aligned target.")
    aligned_target = np.asarray(target, dtype=np.float64)
    if aligned_target.shape != (len(values),):
        raise ValueError("Grouped-risk target is not aligned.")
    for sequence in np.unique(values):
        indices = np.flatnonzero(values == sequence)
        block_length = automatic_time_block_length(aligned_target[indices])
        local_blocks = np.arange(len(indices)) // block_length
        labels[indices] = [
            f"{sequence}:blocked-risk-{block}" for block in local_blocks
        ]
    return labels


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
    solution = solve_pb1_system(system_without_jitter, rhs)
    coefficients = solution.coefficients
    intercept = target_mean - float(feature_mean @ coefficients)
    validation_prediction = validation_matrix @ coefficients + intercept
    by_group, mean, se = _group_mse(
        validation_target, validation_prediction, validation_groups
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", scipy.linalg.LinAlgWarning)
            influence = scipy.linalg.solve(
                system_without_jitter,
                gram,
                assume_a="sym",
                check_finite=False,
            )
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
        influence = scipy.linalg.pinvh(
            system_without_jitter, check_finite=False
        ) @ gram
    effective_df = float(np.trace(influence))
    return PB1PenaltyCandidate(
        lag_weight=weights[0],
        amplitude_weight=weights[1],
        ridge_weight=weights[2],
        validation_mse_mean=mean,
        validation_mse_by_group=by_group,
        validation_mse_se=se,
        effective_df=effective_df,
        relative_kkt_residual=solution.relative_kkt_residual,
        numerical_jitter=solution.numerical_jitter,
        configuration_order=configuration_order,
        index_lag=indices[0],
        index_amplitude=indices[1],
        index_ridge=indices[2],
        coefficients=coefficients,
        intercept=intercept,
        solver_diagnostics={
            "solver_stage": solution.solver_stage,
            "converged": solution.converged,
            "diag_equilibration_used": solution.diag_equilibration_used,
            "iterative_refinement_steps": solution.iterative_refinement_steps,
            "condition_estimate_before": solution.condition_estimate_before,
            "condition_estimate_after": solution.condition_estimate_after,
            "effective_rank": solution.effective_rank,
            "smallest_retained_singular_value": (
                solution.smallest_retained_singular_value
            ),
            "svd_rcond": solution.svd_rcond,
            "solution_is_minimum_norm": solution.solution_is_minimum_norm,
        },
    )


def _select_one_se(
    candidates: list[PB1PenaltyCandidate],
) -> PB1PenaltyCandidate:
    candidates = [
        row
        for row in candidates
        if bool(row.solver_diagnostics["converged"])
        and row.relative_kkt_residual <= 1.0e-8
    ]
    if not candidates:
        raise RuntimeError(
            "No penalty candidate passed the frozen original-coordinate KKT."
        )
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
    ar_block: PB1TensorBlock | None,
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
        np.zeros(len(validation_target), dtype=np.float64)
        if ar_block is None
        else ar_block.validation_matrix
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
    effective_ranks = predictive_rank_profile(
        np.asarray(curves), (0.10, 0.05, 0.02)
    )
    relative_inflation = [
        (mse - full_mse) / max(full_mse, np.finfo(float).eps)
        for mse in rank_mse
    ]
    primary_rank = effective_ranks[0.05]
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
        "full_mse": full_mse,
        "rank1_mse": rank_mse[0],
        "rank2_mse": rank_mse[1] if len(rank_mse) > 1 else rank_mse[0],
        "adaptive_mse": rank_mse[primary_rank - 1],
        "rank_validation_mse": rank_mse,
        "normalized_excess_rmse_curve": curves,
        "output_standardized_predictive_loss": curves,
        "relative_loss_inflation": relative_inflation,
        "selected_rank_10pct": effective_ranks[0.10],
        "selected_rank_5pct": effective_ranks[0.05],
        "selected_rank_2pct": effective_ranks[0.02],
        "predictive_effective_ranks": {
            str(budget): rank
            for budget, rank in effective_ranks.items()
        },
    }


def _block_history_vector(
    block: PB1TensorBlock,
    history: np.ndarray,
) -> np.ndarray:
    values = np.asarray(history, dtype=np.float64)
    if values.shape != (block.lag_basis.shape[0],):
        raise ValueError("Free-run history length differs from fitted block.")
    amplitude = block.amplitude_basis.transform(values)
    return np.einsum(
        "la,lb->ab", block.lag_basis, amplitude, optimize=True
    ).reshape(-1)


def simulate_pb1_free_run(
    dataset: DynamicDataset,
    fit: PB1SpectralDevelopmentFit,
    *,
    official_initialization: int = 0,
) -> PB1FreeRunMetrics:
    """Run one-step spectral simulation without intermediate true outputs."""

    if np.any(dataset.split == "test"):
        raise PermissionError("PB1 development free-run refuses test rows.")
    if fit.horizon != 1:
        raise ValueError("Free-run simulation requires the frozen one-step model.")
    standardizer = TrainOnlyStandardizer.fit(dataset)
    scaled = standardizer.transform(dataset)
    L_x, L_y = fit.history
    initialization = max(L_y, int(official_initialization))
    simulated = np.asarray(scaled.y[:, 0], dtype=np.float64).copy()
    scored_target: list[float] = []
    scored_prediction: list[float] = []
    scored_sequence: list[str] = []
    for sequence in np.unique(scaled.sequence_id):
        indices = np.flatnonzero(scaled.sequence_id == sequence)
        validation_local = np.flatnonzero(scaled.split[indices] == "validation")
        if not len(validation_local):
            continue
        first_validation = int(validation_local[0])
        start = first_validation + initialization
        start = max(start, L_x, L_y)
        for local_target in range(start, int(validation_local[-1]) + 1):
            target_index = int(indices[local_target])
            if scaled.split[target_index] != "validation":
                continue
            origin_local = local_target - 1
            x_history = scaled.x[
                indices[origin_local - np.arange(L_x)], 0
            ]
            y_history = simulated[
                indices[origin_local - np.arange(L_y)]
            ]
            prediction = fit.selected.intercept
            if fit.x_block is not None:
                prediction += float(
                    _block_history_vector(fit.x_block, x_history)
                    @ fit.selected.coefficients[fit.x_block.coefficient_slice]
                )
            if fit.ar_block is not None:
                prediction += float(
                    _block_history_vector(fit.ar_block, y_history)
                    @ fit.selected.coefficients[fit.ar_block.coefficient_slice]
                )
            simulated[target_index] = prediction
            scored_target.append(float(scaled.y[target_index, 0]))
            scored_prediction.append(prediction)
            scored_sequence.append(str(sequence))
    if not scored_target:
        raise ValueError("Free-run simulation has no scorable validation targets.")
    target = np.asarray(scored_target, dtype=np.float64)
    prediction = np.asarray(scored_prediction, dtype=np.float64)
    sequences = np.asarray(scored_sequence, dtype=object)
    mse = float(np.mean((target - prediction) ** 2))
    by_sequence = {
        str(sequence): float(
            np.mean(
                (
                    target[sequences == sequence]
                    - prediction[sequences == sequence]
                )
                ** 2
            )
        )
        for sequence in np.unique(sequences)
    }
    scale = float(standardizer.y_scale[0])
    return PB1FreeRunMetrics(
        initialization_length=initialization,
        scored_samples=len(target),
        mse_standardized=mse,
        rmse_standardized=math.sqrt(mse),
        mse_original_units=mse * scale**2,
        rmse_original_units=math.sqrt(mse) * scale,
        mse_by_sequence_standardized=by_sequence,
        status="COMPLETED",
    )


def bootstrap_external_rank_spectrum(
    fit: PB1SpectralDevelopmentFit,
    *,
    replicates: int = 250,
    seed: int = 20240727,
) -> PB1RankBootstrap:
    """Cluster bootstrap the frozen external spectrum without retuning."""

    started = time.perf_counter()
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive.")
    if fit.x_block is None:
        raise ValueError("External-rank bootstrap is not applicable to AR-only.")
    matrix = np.asarray(fit.train_matrix, dtype=np.float64)
    target = np.asarray(fit.train_target, dtype=np.float64)
    groups = np.asarray(fit.train_groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Cluster bootstrap needs at least two training groups.")
    summaries: dict[
        object, tuple[int, np.ndarray, float, np.ndarray, np.ndarray]
    ] = {}
    for group in unique_groups:
        mask = groups == group
        local_x = matrix[mask]
        local_y = target[mask]
        summaries[group] = (
            int(np.sum(mask)),
            local_x.sum(axis=0),
            float(local_y.sum()),
            local_x.T @ local_x,
            local_x.T @ local_y,
        )
    rng = np.random.default_rng(seed)
    x_slice = fit.x_block.coefficient_slice
    lag_count = fit.x_block.lag_gram.shape[0]
    amplitude_count = fit.x_block.amplitude_gram.shape[0]
    spectrum_width = min(lag_count, amplitude_count)
    singular_values = np.empty((replicates, spectrum_width))
    normalized_energy = np.empty_like(singular_values)
    rank_counts = {
        "0.1": {},
        "0.05": {},
        "0.02": {},
    }
    maximum_kkt = 0.0
    for replicate in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        multiplicities = {
            group: int(np.sum(sampled == group)) for group in unique_groups
        }
        n = sum(
            multiplicities[group] * summaries[group][0]
            for group in unique_groups
        )
        sum_x = sum(
            (
                multiplicities[group] * summaries[group][1]
                for group in unique_groups
            ),
            start=np.zeros(matrix.shape[1], dtype=np.float64),
        )
        sum_y = sum(
            multiplicities[group] * summaries[group][2]
            for group in unique_groups
        )
        sum_xx = sum(
            (
                multiplicities[group] * summaries[group][3]
                for group in unique_groups
            ),
            start=np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64),
        )
        sum_xy = sum(
            (
                multiplicities[group] * summaries[group][4]
                for group in unique_groups
            ),
            start=np.zeros(matrix.shape[1], dtype=np.float64),
        )
        mean_x = sum_x / n
        mean_y = sum_y / n
        gram = sum_xx / n - np.outer(mean_x, mean_x)
        rhs = sum_xy / n - mean_x * mean_y
        system_without_jitter = gram + fit.selected_penalty
        solution = solve_pb1_system(system_without_jitter, rhs)
        coefficients = solution.coefficients
        relative_kkt = solution.relative_kkt_residual
        maximum_kkt = max(maximum_kkt, relative_kkt)
        theta_x = coefficients[x_slice].reshape(
            lag_count, amplitude_count
        )
        spectrum = gram_whitened_svd(
            theta_x,
            fit.x_block.lag_gram,
            fit.x_block.amplitude_gram,
        )
        values = spectrum.singular_values
        energy = values**2
        energy /= max(float(energy.sum()), np.finfo(float).eps)
        singular_values[replicate] = values
        normalized_energy[replicate] = energy
        for budget in (0.10, 0.05, 0.02):
            rank = spectrum_width + 1
            for candidate_rank in range(1, spectrum_width + 1):
                tail = math.sqrt(float(energy[candidate_rank:].sum()))
                if tail <= budget:
                    rank = candidate_rank
                    break
            key = str(budget)
            rank_key = str(rank)
            rank_counts[key][rank_key] = rank_counts[key].get(rank_key, 0) + 1
    quantiles = (0.025, 0.5, 0.975)
    return PB1RankBootstrap(
        replicates=replicates,
        seed=seed,
        singular_value_quantiles={
            str(quantile): np.quantile(singular_values, quantile, axis=0)
            for quantile in quantiles
        },
        normalized_energy_quantiles={
            str(quantile): np.quantile(normalized_energy, quantile, axis=0)
            for quantile in quantiles
        },
        spectral_tail_budget_rank_frequencies=rank_counts,
        maximum_relative_kkt_residual=maximum_kkt,
        elapsed_seconds=time.perf_counter() - started,
        automatic_block_length=(
            automatic_time_block_length(target)
            if len(np.unique(fit.train_groups)) > 1
            and fit.track in {"X", "XAR"}
            and all("blocked-risk" in str(group) for group in unique_groups)
            else None
        ),
    )


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
    track: str = "XAR",
) -> PB1SpectralDevelopmentFit:
    """Fit H3 full spectral XAR at one preregistered resolution."""

    started = time.perf_counter()
    if np.any(dataset.split == "test"):
        raise PermissionError("PB1 spectral development refuses test rows.")
    if track not in {"X", "AR", "XAR"}:
        raise ValueError("PB1 spectral track must be X, AR, or XAR.")
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
    offset = 0
    x_block: PB1TensorBlock | None = None
    ar_block: PB1TensorBlock | None = None
    active_blocks: list[PB1TensorBlock] = []
    if track in {"X", "XAR"}:
        x_width = x_matrix.shape[1]
        x_block = PB1TensorBlock(
            train_matrix=x_matrix[train_mask],
            validation_matrix=x_matrix[validation_mask],
            lag_basis=x_lag,
            amplitude_basis=x_amp,
            lag_gram=x_lag_gram,
            amplitude_gram=x_amp_gram,
            coefficient_slice=slice(offset, offset + x_width),
        )
        active_blocks.append(x_block)
        offset += x_width
    if track in {"AR", "XAR"}:
        ar_width = ar_matrix.shape[1]
        ar_block = PB1TensorBlock(
            train_matrix=ar_matrix[train_mask],
            validation_matrix=ar_matrix[validation_mask],
            lag_basis=ar_lag,
            amplitude_basis=ar_amp,
            lag_gram=ar_lag_gram,
            amplitude_gram=ar_amp_gram,
            coefficient_slice=slice(offset, offset + ar_width),
        )
        active_blocks.append(ar_block)
    train_matrix = np.column_stack(
        [block.train_matrix for block in active_blocks]
    )
    validation_matrix = np.column_stack(
        [block.validation_matrix for block in active_blocks]
    )
    train_target = task.target[train_mask]
    validation_target = task.target[validation_mask]
    validation_groups = _group_labels(
        dataset_id,
        task.sequence_id[validation_mask],
        task.target[validation_mask],
    )
    feature_mean = train_matrix.mean(axis=0)
    target_mean = float(train_target.mean())
    centered_matrix = train_matrix - feature_mean
    centered_target = train_target - target_mean
    gram = centered_matrix.T @ centered_matrix / len(centered_matrix)
    rhs = centered_matrix.T @ centered_target / len(centered_matrix)
    raw_penalties = _block_penalties(tuple(active_blocks))
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
        grids = [
            zero_inclusive_penalty_grid(interval, grid_points)
            for interval in intervals
        ]
        round_rows: list[PB1PenaltyCandidate] = []
        axis_size = grid_points + 1
        for indices in itertools.product(range(axis_size), repeat=3):
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
        axis_statuses: list[str] = []
        selected_indices = (
            selected.index_lag,
            selected.index_amplitude,
            selected.index_ridge,
        )
        valid_round_rows = [
            row
            for row in round_rows
            if bool(row.solver_diagnostics["converged"])
            and row.relative_kkt_residual <= 1.0e-8
        ]
        global_minimum = min(
            valid_round_rows,
            key=lambda row: (row.validation_mse_mean, row.configuration_order),
        )
        for axis, index in enumerate(selected_indices):
            if index == 0:
                axis_statuses.append("ZERO_PENALTY_ENDPOINT_CERTIFIED")
            elif index == axis_size - 1:
                boundary_axes.append((axis, "upper"))
                axis_statuses.append("UPPER_BOUNDARY_SELECTED")
            elif index == 1:
                zero_rows = [
                    row
                    for row in valid_round_rows
                    if (
                        row.index_lag,
                        row.index_amplitude,
                        row.index_ridge,
                    )[axis]
                    == 0
                ]
                zero_best = min(
                    (row.validation_mse_mean for row in zero_rows),
                    default=float("inf"),
                )
                if positive_lower_expansion_required(
                    selected_index=index,
                    axis_zero_best_loss=zero_best,
                    global_minimum_loss=global_minimum.validation_mse_mean,
                    global_minimum_se=global_minimum.validation_mse_se,
                ):
                    boundary_axes.append((axis, "lower"))
                    axis_statuses.append("POSITIVE_LOWER_EXPANSION_REQUIRED")
                else:
                    axis_statuses.append("POSITIVE_LOWER_CERTIFIED_WITH_ZERO")
            else:
                axis_statuses.append("POSITIVE_INTERIOR_CERTIFIED")
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
                "selected_weights": [
                    selected.lag_weight,
                    selected.amplitude_weight,
                    selected.ridge_weight,
                ],
                "axis_statuses": axis_statuses,
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
            if any(side == "upper" for _, side in boundary_axes):
                status = "PENALTY_UPPER_INTERVAL_NOT_CERTIFIED"
            break
        for axis, side in boundary_axes:
            intervals[axis] = expand_penalty_interval(
                intervals[axis], boundary=side
            )
    if selected is None:
        raise AssertionError("PB1 spectral penalty search produced no candidate.")
    selected_penalty = sum(
        weight * component
        for weight, component in zip(
            (
                selected.lag_weight,
                selected.amplitude_weight,
                selected.ridge_weight,
            ),
            normalized,
            strict=True,
        )
    )
    rank_audit = (
        _rank_audit(
            selected,
            x_block=x_block,
            ar_block=ar_block,
            validation_target=validation_target,
        )
        if x_block is not None
        else {
            "status": "NOT_APPLICABLE",
            "reason": "AR_ONLY_TRACK_HAS_NO_EXTERNAL_KERNEL_RANK",
            "structural_rank_claim_allowed": False,
            "predictive_svd_rank_claim_allowed": False,
            "full_validation_mse": selected.validation_mse_mean,
        }
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
        train_matrix=train_matrix,
        train_groups=_group_labels(
            dataset_id,
            task.sequence_id[train_mask],
            task.target[train_mask],
        ),
        validation_groups=validation_groups,
        selected_penalty=selected_penalty,
        rank_audit=rank_audit,
        history=(L_x, L_y),
        track=track,
        horizon=horizon,
        elapsed_seconds=time.perf_counter() - started,
    )
