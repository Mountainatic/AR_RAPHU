"""CPU FP64 ridge and independent-block ARX solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.linalg


@dataclass(frozen=True, slots=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "Standardizer":
        values = np.asarray(matrix, dtype=np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0.0] = 1.0
        return cls(mean, scale)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (np.asarray(matrix, dtype=np.float64) - self.mean) / self.scale


@dataclass(slots=True)
class RidgeFit:
    coefficients: np.ndarray
    intercept: float
    standardizer: Standardizer
    relative_kkt: float
    train_mse: float
    alpha: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return (
            self.standardizer.transform(matrix) @ self.coefficients
            + self.intercept
        )

    def physical_coefficients(self) -> np.ndarray:
        return self.coefficients / self.standardizer.scale


@dataclass(slots=True)
class BlockRidgeFit:
    ar_coefficients: np.ndarray
    q_coefficients: np.ndarray
    intercept: float
    ar_standardizer: Standardizer
    q_standardizer: Standardizer
    relative_kkt: float
    train_mse: float
    alpha_ar: float
    alpha_q: float | None

    def predict(self, ar: np.ndarray, q: np.ndarray) -> np.ndarray:
        prediction = (
            self.ar_standardizer.transform(ar) @ self.ar_coefficients
            + self.intercept
        )
        if len(self.q_coefficients):
            prediction = (
                prediction
                + self.q_standardizer.transform(q) @ self.q_coefficients
            )
        return prediction

    def physical_q_coefficients(self) -> np.ndarray:
        if not len(self.q_coefficients):
            return self.q_coefficients
        return self.q_coefficients / self.q_standardizer.scale


def _minimum_norm_solve(
    system: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    values, vectors = scipy.linalg.eigh(
        system, check_finite=False, driver="evd"
    )
    threshold = max(
        float(values.max()) * 1.0e-12,
        np.finfo(np.float64).eps,
    )
    inverse = np.where(
        values > threshold,
        1.0 / np.maximum(values, threshold),
        0.0,
    )
    return vectors @ (inverse * (vectors.T @ rhs))


def _solve(system: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        factor = scipy.linalg.cho_factor(
            system, lower=True, check_finite=False
        )
        return scipy.linalg.cho_solve(
            factor, rhs, check_finite=False
        )
    except np.linalg.LinAlgError:
        return _minimum_norm_solve(system, rhs)


def fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
) -> RidgeFit:
    raw = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    standardizer = Standardizer.fit(raw)
    x = standardizer.transform(raw)
    y_mean = float(y.mean())
    yc = y - y_mean
    gram = x.T @ x / len(x)
    rhs = x.T @ yc / len(x)
    system = gram + float(alpha) * np.eye(x.shape[1])
    coefficients = (
        _minimum_norm_solve(system, rhs)
        if float(alpha) == 0.0
        else _solve(system, rhs)
    )
    residual = system @ coefficients - rhs
    relative_kkt = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
    )
    prediction = y_mean + x @ coefficients
    return RidgeFit(
        coefficients=coefficients,
        intercept=y_mean,
        standardizer=standardizer,
        relative_kkt=relative_kkt,
        train_mse=float(np.mean((y - prediction) ** 2)),
        alpha=float(alpha),
    )


def fit_block_ridge(
    ar_matrix: np.ndarray,
    q_matrix: np.ndarray,
    target: np.ndarray,
    *,
    alpha_ar: float,
    alpha_q: float | None,
) -> BlockRidgeFit:
    ar_raw = np.asarray(ar_matrix, dtype=np.float64)
    q_raw = np.asarray(q_matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    ar_scaler = Standardizer.fit(ar_raw)
    q_scaler = Standardizer.fit(q_raw)
    ar = ar_scaler.transform(ar_raw)
    q = q_scaler.transform(q_raw)
    y_mean = float(y.mean())
    yc = y - y_mean
    if alpha_q is None:
        ar_fit = fit_ridge(ar_raw, y, alpha=alpha_ar)
        return BlockRidgeFit(
            ar_coefficients=ar_fit.coefficients,
            q_coefficients=np.empty(0, dtype=np.float64),
            intercept=ar_fit.intercept,
            ar_standardizer=ar_fit.standardizer,
            q_standardizer=q_scaler,
            relative_kkt=ar_fit.relative_kkt,
            train_mse=ar_fit.train_mse,
            alpha_ar=float(alpha_ar),
            alpha_q=None,
        )
    x = np.column_stack((ar, q))
    gram = x.T @ x / len(x)
    rhs = x.T @ yc / len(x)
    penalty = np.concatenate(
        (
            np.full(ar.shape[1], float(alpha_ar)),
            np.full(q.shape[1], float(alpha_q)),
        )
    )
    system = gram + np.diag(penalty)
    coefficients = (
        _minimum_norm_solve(system, rhs)
        if float(alpha_ar) == 0.0 and float(alpha_q) == 0.0
        else _solve(system, rhs)
    )
    residual = system @ coefficients - rhs
    prediction = y_mean + x @ coefficients
    return BlockRidgeFit(
        ar_coefficients=coefficients[: ar.shape[1]],
        q_coefficients=coefficients[ar.shape[1] :],
        intercept=y_mean,
        ar_standardizer=ar_scaler,
        q_standardizer=q_scaler,
        relative_kkt=float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
        ),
        train_mse=float(np.mean((y - prediction) ** 2)),
        alpha_ar=float(alpha_ar),
        alpha_q=float(alpha_q),
    )


def regression_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    residual = y - p
    mse = float(np.mean(residual**2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(residual))),
    }


def relative_improvement(
    baseline_loss: np.ndarray, model_loss: np.ndarray
) -> float:
    baseline = float(np.mean(baseline_loss))
    return float(
        1.0
        - np.mean(model_loss)
        / max(baseline, np.finfo(np.float64).eps)
    )
