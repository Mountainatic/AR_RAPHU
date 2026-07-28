"""ARX lag selection adapted from Champneys et al. 2024 to frozen PB1 splits."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import scipy.linalg
import scipy.signal

from ar_raphu.datasets.base import DynamicDataset


@dataclass(frozen=True, slots=True)
class ARXHistoryCandidate:
    nx: int
    ny: int
    validation_aic_mean: float
    validation_aic_by_record: tuple[float, ...]
    effective_rank: int
    condition_number: float
    stable_simulation: bool


@dataclass(frozen=True, slots=True)
class ARXHistorySelection:
    selected_nx: int
    selected_ny: int
    coefficients_y: np.ndarray
    coefficients_x: np.ndarray
    x_mean: float
    x_scale: float
    y_mean: float
    y_scale: float
    candidates: tuple[ARXHistoryCandidate, ...]
    elapsed_seconds: float


def _record_arrays(
    dataset: DynamicDataset, split: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    records: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sequence in np.unique(dataset.sequence_id):
        indices = np.flatnonzero(dataset.sequence_id == sequence)
        selected = indices[dataset.split[indices] == split]
        if not len(selected):
            continue
        if len(selected) > 1 and np.any(np.diff(selected) != 1):
            raise ValueError(f"{sequence}: {split} rows are not contiguous.")
        records.append(
            (
                f"{sequence}:{split}",
                np.asarray(dataset.x[selected, 0], dtype=np.float64),
                np.asarray(dataset.y[selected, 0], dtype=np.float64),
            )
        )
    if not records:
        raise ValueError(f"No {split} records.")
    return records


def _fit_scaler(
    records: list[tuple[str, np.ndarray, np.ndarray]]
) -> tuple[float, float, float, float]:
    x = np.concatenate([record[1] for record in records])
    y = np.concatenate([record[2] for record in records])
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_scale = float(np.std(x))
    y_scale = float(np.std(y))
    if x_scale <= 0.0 or y_scale <= 0.0:
        raise ValueError("ARX training scaler has a constant channel.")
    return x_mean, x_scale, y_mean, y_scale


def _maximal_arx_design(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_nx: int,
    max_ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    history = max(max_nx, max_ny)
    if len(x) <= history:
        raise ValueError("Record is shorter than maximum ARX history.")
    y_window = np.lib.stride_tricks.sliding_window_view(y, max_ny)
    y_lags = y_window[history - max_ny : -1, ::-1]
    x_window = np.lib.stride_tricks.sliding_window_view(x, max_nx)
    x_lags = x_window[history - max_nx : -1, ::-1]
    target = y[history:]
    if len(y_lags) != len(target) or len(x_lags) != len(target):
        raise AssertionError("ARX maximal design alignment failed.")
    return np.column_stack((y_lags, x_lags)), target


def _streaming_qr(
    records: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    max_nx: int,
    max_ny: int,
    x_mean: float,
    x_scale: float,
    y_mean: float,
    y_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    r_factor: np.ndarray | None = None
    projected_target: np.ndarray | None = None
    for _, x_raw, y_raw in records:
        x = (x_raw - x_mean) / x_scale
        y = (y_raw - y_mean) / y_scale
        design, target = _maximal_arx_design(
            x, y, max_nx=max_nx, max_ny=max_ny
        )
        if r_factor is not None and projected_target is not None:
            design = np.vstack((r_factor, design))
            target = np.concatenate((projected_target, target))
        q_factor, r_factor = scipy.linalg.qr(
            design,
            mode="economic",
            overwrite_a=False,
            check_finite=True,
        )
        projected_target = q_factor.T @ target
    if r_factor is None or projected_target is None:
        raise AssertionError("Streaming QR received no training records.")
    return r_factor, projected_target


def _fit_subset(
    r_factor: np.ndarray,
    projected_target: np.ndarray,
    *,
    nx: int,
    ny: int,
    max_ny: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    columns = np.r_[np.arange(ny), max_ny + np.arange(nx)]
    matrix = r_factor[:, columns]
    coefficients, _, rank, singular_values = scipy.linalg.lstsq(
        matrix,
        projected_target,
        cond=None,
        lapack_driver="gelsd",
        check_finite=True,
    )
    condition = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) and singular_values[-1] > 0.0
        else float("inf")
    )
    return coefficients[:ny], coefficients[ny:], int(rank), condition


def simulate_arx(
    x: np.ndarray,
    y_initialization: np.ndarray,
    *,
    coefficients_y: np.ndarray,
    coefficients_x: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Free-run ARX simulation using only the initial observed output history."""

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y_initialization, dtype=np.float64)
    ny = len(coefficients_y)
    nx = len(coefficients_x)
    burn = max(nx, ny)
    if len(x_values) != len(y_values) or len(x_values) <= burn:
        raise ValueError("Invalid ARX simulation record.")
    denominator = np.r_[1.0, -np.asarray(coefficients_y, dtype=np.float64)]
    # PB1 target y[t+1] may only use x through t, hence the leading zero.
    numerator = np.r_[0.0, np.asarray(coefficients_x, dtype=np.float64)]
    past_y = y_values[burn - 1 :: -1][:ny]
    past_x = x_values[burn - 1 :: -1][:nx]
    initial_state = scipy.signal.lfiltic(
        numerator, denominator, y=past_y, x=past_x
    )
    simulated_tail, _ = scipy.signal.lfilter(
        numerator,
        denominator,
        x_values[burn:],
        zi=initial_state,
    )
    simulated = np.full_like(y_values, np.nan)
    simulated[:burn] = y_values[:burn]
    simulated[burn:] = simulated_tail
    return simulated, burn


def _validation_aic(
    records: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    coefficients_y: np.ndarray,
    coefficients_x: np.ndarray,
    x_mean: float,
    x_scale: float,
    y_mean: float,
    y_scale: float,
) -> tuple[tuple[float, ...], bool]:
    scores: list[float] = []
    stable = True
    parameters = len(coefficients_y) + len(coefficients_x)
    for _, x_raw, y_raw in records:
        x = (x_raw - x_mean) / x_scale
        y = (y_raw - y_mean) / y_scale
        prediction, burn = simulate_arx(
            x,
            y,
            coefficients_y=coefficients_y,
            coefficients_x=coefficients_x,
        )
        residual = y[burn:] - prediction[burn:]
        n = len(residual)
        rss = float(residual @ residual)
        if (
            not np.isfinite(rss)
            or rss < 0.0
            or n <= parameters
        ):
            score = float("inf")
            stable = False
        else:
            score = float(
                2 * parameters
                + n
                * np.log(
                    max(rss, np.finfo(np.float64).tiny)
                    / (n - parameters)
                )
            )
        scores.append(score)
    return tuple(scores), stable


def _complexity_key(nx: int, ny: int) -> tuple[int, int, int, int]:
    return nx + ny, max(nx, ny), nx, ny


def fit_and_select_arx_history(
    dataset: DynamicDataset,
    *,
    max_nx: int = 20,
    max_ny: int = 20,
) -> ARXHistorySelection:
    """Select ARX history by mean record-level validation AIC, without test data."""

    started = time.perf_counter()
    if dataset.n_features != 1 or dataset.n_targets != 1:
        raise ValueError("Champneys 2024 ARX profile is SISO.")
    if np.any(dataset.split == "test"):
        raise PermissionError("ARX development selection refuses test rows.")
    train_records = _record_arrays(dataset, "train")
    validation_records = _record_arrays(dataset, "validation")
    x_mean, x_scale, y_mean, y_scale = _fit_scaler(train_records)
    r_factor, projected_target = _streaming_qr(
        train_records,
        max_nx=max_nx,
        max_ny=max_ny,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    candidates: list[ARXHistoryCandidate] = []
    fitted: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    selected: ARXHistoryCandidate | None = None
    for nx in range(1, max_nx + 1):
        for ny in range(1, max_ny + 1):
            coefficients_y, coefficients_x, rank, condition = _fit_subset(
                r_factor,
                projected_target,
                nx=nx,
                ny=ny,
                max_ny=max_ny,
            )
            scores, stable = _validation_aic(
                validation_records,
                coefficients_y=coefficients_y,
                coefficients_x=coefficients_x,
                x_mean=x_mean,
                x_scale=x_scale,
                y_mean=y_mean,
                y_scale=y_scale,
            )
            mean_score = float(np.mean(scores))
            candidate = ARXHistoryCandidate(
                nx=nx,
                ny=ny,
                validation_aic_mean=mean_score,
                validation_aic_by_record=scores,
                effective_rank=rank,
                condition_number=condition,
                stable_simulation=stable,
            )
            candidates.append(candidate)
            fitted[(nx, ny)] = (coefficients_y, coefficients_x)
            if selected is None:
                selected = candidate
                continue
            tolerance = (
                64.0
                * np.finfo(np.float64).eps
                * max(1.0, abs(selected.validation_aic_mean))
            )
            if mean_score < selected.validation_aic_mean - tolerance:
                selected = candidate
            elif (
                abs(mean_score - selected.validation_aic_mean) <= tolerance
                and _complexity_key(nx, ny)
                < _complexity_key(selected.nx, selected.ny)
            ):
                selected = candidate
    if selected is None or not np.isfinite(selected.validation_aic_mean):
        raise RuntimeError("Every ARX history candidate was unstable.")
    coefficients_y, coefficients_x = fitted[(selected.nx, selected.ny)]
    return ARXHistorySelection(
        selected_nx=selected.nx,
        selected_ny=selected.ny,
        coefficients_y=coefficients_y,
        coefficients_x=coefficients_x,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        candidates=tuple(candidates),
        elapsed_seconds=float(time.perf_counter() - started),
    )


def fit_arx_fixed_history(
    dataset: DynamicDataset,
    *,
    nx: int,
    ny: int,
) -> ARXHistorySelection:
    """Fit the literature-frozen ARX history without searching shorter lags."""

    started = time.perf_counter()
    if nx <= 0 or ny <= 0:
        raise ValueError("Fixed ARX histories must be positive.")
    if dataset.n_features != 1 or dataset.n_targets != 1:
        raise ValueError("Champneys 2024 ARX profile is SISO.")
    if np.any(dataset.split == "test"):
        raise PermissionError("ARX development fit refuses test rows.")
    train_records = _record_arrays(dataset, "train")
    validation_records = _record_arrays(dataset, "validation")
    x_mean, x_scale, y_mean, y_scale = _fit_scaler(train_records)
    r_factor, projected_target = _streaming_qr(
        train_records,
        max_nx=nx,
        max_ny=ny,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    coefficients_y, coefficients_x, rank, condition = _fit_subset(
        r_factor,
        projected_target,
        nx=nx,
        ny=ny,
        max_ny=ny,
    )
    scores, stable = _validation_aic(
        validation_records,
        coefficients_y=coefficients_y,
        coefficients_x=coefficients_x,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    candidate = ARXHistoryCandidate(
        nx=nx,
        ny=ny,
        validation_aic_mean=float(np.mean(scores)),
        validation_aic_by_record=scores,
        effective_rank=rank,
        condition_number=condition,
        stable_simulation=stable,
    )
    return ARXHistorySelection(
        selected_nx=nx,
        selected_ny=ny,
        coefficients_y=coefficients_y,
        coefficients_x=coefficients_x,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        candidates=(candidate,),
        elapsed_seconds=float(time.perf_counter() - started),
    )
