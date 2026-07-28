"""Literature-faithful monomial Legendre pNARX for PB1 development."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import scipy.linalg

from ar_raphu.datasets.base import DynamicDataset


@dataclass(frozen=True, slots=True)
class PNARXCandidate:
    order: int
    validation_aic_mean: float
    validation_aic_by_record: tuple[float, ...]
    effective_rank: int
    condition_number: float
    stable_simulation: bool


@dataclass(frozen=True, slots=True)
class PNARXSelection:
    order: int
    coefficients: np.ndarray
    x_mean: float
    x_scale: float
    y_mean: float
    y_scale: float
    candidates: tuple[PNARXCandidate, ...]
    elapsed_seconds: float


def _records(
    dataset: DynamicDataset, split: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sequence in np.unique(dataset.sequence_id):
        indices = np.flatnonzero(dataset.sequence_id == sequence)
        selected = indices[dataset.split[indices] == split]
        if not len(selected):
            continue
        if len(selected) > 1 and np.any(np.diff(selected) != 1):
            raise ValueError(f"{sequence}: {split} rows are not contiguous.")
        result.append(
            (
                f"{sequence}:{split}",
                np.asarray(dataset.x[selected, 0], dtype=np.float64),
                np.asarray(dataset.y[selected, 0], dtype=np.float64),
            )
        )
    if not result:
        raise ValueError(f"No {split} records.")
    return result


def _fit_scaler(
    records: list[tuple[str, np.ndarray, np.ndarray]]
) -> tuple[float, float, float, float]:
    x = np.concatenate([record[1] for record in records])
    y = np.concatenate([record[2] for record in records])
    x_mean, y_mean = float(x.mean()), float(y.mean())
    x_scale, y_scale = float(x.std()), float(y.std())
    if x_scale <= 0.0 or y_scale <= 0.0:
        raise ValueError("pNARX training scaler has a constant channel.")
    return x_mean, x_scale, y_mean, y_scale


def _history_design(
    x: np.ndarray, y: np.ndarray, *, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return histories through t and target y[t+1]."""

    history = max(nx, ny)
    if len(x) <= history:
        raise ValueError("Record is shorter than the pNARX history.")
    y_windows = np.lib.stride_tricks.sliding_window_view(y, ny)
    y_lags = y_windows[history - ny : -1, ::-1]
    x_windows = np.lib.stride_tricks.sliding_window_view(x, nx)
    x_lags = x_windows[history - nx : -1, ::-1]
    target = y[history:]
    if len(y_lags) != len(target) or len(x_lags) != len(target):
        raise AssertionError("pNARX history alignment failed.")
    return np.column_stack((y_lags, x_lags)), target


def legendre_monomial_design(history: np.ndarray, order: int) -> np.ndarray:
    """Apply degrees 1..order separately to each regressor, without cross terms."""

    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 2 or order < 1:
        raise ValueError("Expected a 2D history and positive order.")
    vandermonde = np.polynomial.legendre.legvander(values, order)
    return vandermonde[..., 1:].reshape(len(values), values.shape[1] * order)


def _streaming_qr(
    records: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    nx: int,
    ny: int,
    maximum_order: int,
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
        histories, target = _history_design(x, y, nx=nx, ny=ny)
        design = legendre_monomial_design(histories, maximum_order)
        if r_factor is not None and projected_target is not None:
            design = np.vstack((r_factor, design))
            target = np.concatenate((projected_target, target))
        q_factor, r_factor = scipy.linalg.qr(
            design, mode="economic", check_finite=True
        )
        projected_target = q_factor.T @ target
    if r_factor is None or projected_target is None:
        raise AssertionError("Streaming pNARX QR received no records.")
    return r_factor, projected_target


def _fit_order(
    r_factor: np.ndarray,
    projected_target: np.ndarray,
    *,
    n_regressors: int,
    maximum_order: int,
    order: int,
) -> tuple[np.ndarray, int, float]:
    columns = np.concatenate(
        [
            regressor * maximum_order + np.arange(order)
            for regressor in range(n_regressors)
        ]
    )
    coefficients, _, rank, singular_values = scipy.linalg.lstsq(
        r_factor[:, columns],
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
    return coefficients, int(rank), condition


def simulate_pnarx(
    x: np.ndarray,
    y_initialization: np.ndarray,
    *,
    nx: int,
    ny: int,
    order: int,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Free-run simulation using observed outputs only for the initial history."""

    x_values = np.asarray(x, dtype=np.float64)
    observed_y = np.asarray(y_initialization, dtype=np.float64)
    burn = max(nx, ny)
    if len(x_values) != len(observed_y) or len(x_values) <= burn:
        raise ValueError("Invalid pNARX simulation record.")
    prediction = np.full_like(observed_y, np.nan)
    prediction[:burn] = observed_y[:burn]
    for target_index in range(burn, len(prediction)):
        origin = target_index - 1
        history = np.r_[
            prediction[origin - np.arange(ny)],
            x_values[origin - np.arange(nx)],
        ][None, :]
        features = legendre_monomial_design(history, order)
        prediction[target_index] = (features @ coefficients).item()
        if not np.isfinite(prediction[target_index]):
            break
    return prediction, burn


def _validation_aic(
    records: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    nx: int,
    ny: int,
    order: int,
    coefficients: np.ndarray,
    x_mean: float,
    x_scale: float,
    y_mean: float,
    y_scale: float,
) -> tuple[tuple[float, ...], bool]:
    scores: list[float] = []
    stable = True
    parameters = (nx + ny) * order
    for _, x_raw, y_raw in records:
        x = (x_raw - x_mean) / x_scale
        y = (y_raw - y_mean) / y_scale
        prediction, burn = simulate_pnarx(
            x,
            y,
            nx=nx,
            ny=ny,
            order=order,
            coefficients=coefficients,
        )
        residual = y[burn:] - prediction[burn:]
        n = len(residual)
        rss = float(residual @ residual)
        if not np.isfinite(rss) or n <= parameters:
            scores.append(float("inf"))
            stable = False
        else:
            scores.append(
                float(
                    2 * parameters
                    + n
                    * np.log(
                        max(rss, np.finfo(np.float64).tiny)
                        / (n - parameters)
                    )
                )
            )
    return tuple(scores), stable


def fit_and_select_pnarx(
    dataset: DynamicDataset,
    *,
    nx: int,
    ny: int,
    orders: tuple[int, ...] = (2, 3, 4, 5, 6, 7),
) -> PNARXSelection:
    """Select the paper's monomial Legendre order by validation AIC."""

    started = time.perf_counter()
    if dataset.n_features != 1 or dataset.n_targets != 1:
        raise ValueError("Champneys 2024 pNARX profile is SISO.")
    if np.any(dataset.split == "test"):
        raise PermissionError("pNARX development selection refuses test rows.")
    if not orders or min(orders) < 2 or max(orders) > 7:
        raise ValueError("Peer-reviewed pNARX orders must be within 2..7.")
    train_records = _records(dataset, "train")
    validation_records = _records(dataset, "validation")
    x_mean, x_scale, y_mean, y_scale = _fit_scaler(train_records)
    maximum_order = max(orders)
    r_factor, projected_target = _streaming_qr(
        train_records,
        nx=nx,
        ny=ny,
        maximum_order=maximum_order,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    rows: list[PNARXCandidate] = []
    fitted: dict[int, np.ndarray] = {}
    for order in orders:
        coefficients, rank, condition = _fit_order(
            r_factor,
            projected_target,
            n_regressors=nx + ny,
            maximum_order=maximum_order,
            order=order,
        )
        scores, stable = _validation_aic(
            validation_records,
            nx=nx,
            ny=ny,
            order=order,
            coefficients=coefficients,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
        )
        rows.append(
            PNARXCandidate(
                order=order,
                validation_aic_mean=float(np.mean(scores)),
                validation_aic_by_record=scores,
                effective_rank=rank,
                condition_number=condition,
                stable_simulation=stable,
            )
        )
        fitted[order] = coefficients
    selected = min(rows, key=lambda row: (row.validation_aic_mean, row.order))
    if not np.isfinite(selected.validation_aic_mean):
        raise FloatingPointError("Every pNARX order was unstable on validation.")
    return PNARXSelection(
        order=selected.order,
        coefficients=fitted[selected.order],
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        candidates=tuple(rows),
        elapsed_seconds=time.perf_counter() - started,
    )
