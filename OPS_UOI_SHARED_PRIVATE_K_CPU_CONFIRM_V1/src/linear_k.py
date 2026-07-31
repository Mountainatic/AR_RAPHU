from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cholesky, solve_triangular, subspace_angles

from .io_data import inner_folds, load_direction, load_protocol


CHANNELS = (
    "joint_lift",
    "heater_power",
    "crystal_rotation",
    "crucible_rotation",
)


@dataclass(frozen=True)
class MotherBasis:
    name: str
    knots: np.ndarray
    lag_minutes: np.ndarray
    lag_design: np.ndarray
    gram: np.ndarray
    gram_sqrt: np.ndarray
    gram_inv_sqrt: np.ndarray
    roughness: np.ndarray

    @property
    def size(self) -> int:
        return int(self.lag_design.shape[1])


@dataclass
class RidgeFit:
    coefficient: np.ndarray
    prediction: np.ndarray
    kkt_relative: float
    condition_number: float


def mother_basis(
    name: str,
    specification: dict[str, Any],
    *,
    sequence_steps: int,
    cadence_sec: float,
) -> MotherBasis:
    degree = 3
    left, right = [float(value) for value in specification["boundary_min"]]
    internal = np.asarray(specification["internal_knots_min"], dtype=np.float64)
    if np.any(np.diff(internal) <= 0) or internal[0] <= left or internal[-1] >= right:
        raise ValueError(f"INVALID_INTERNAL_KNOTS:{name}")
    knots = np.concatenate(
        (
            np.repeat(left, degree + 1),
            internal,
            np.repeat(right, degree + 1),
        )
    )
    lag_minutes = np.arange(sequence_steps, dtype=np.float64) * cadence_sec / 60.0
    lag_design = BSpline.design_matrix(
        lag_minutes, knots, degree, extrapolate=False
    ).toarray()
    grid = np.linspace(left, right, 4001, dtype=np.float64)
    grid_design = BSpline.design_matrix(
        grid, knots, degree, extrapolate=False
    ).toarray()
    gram = np.trapezoid(
        grid_design[:, :, None] * grid_design[:, None, :],
        grid,
        axis=0,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    floor = max(float(np.max(eigenvalues)) * 1e-12, 1e-15)
    eigenvalues = np.maximum(eigenvalues, floor)
    gram_sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    gram_inv_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    difference = np.diff(np.eye(lag_design.shape[1]), n=2, axis=0)
    roughness = difference.T @ difference
    return MotherBasis(
        name=name,
        knots=knots,
        lag_minutes=lag_minutes,
        lag_design=lag_design,
        gram=gram,
        gram_sqrt=gram_sqrt,
        gram_inv_sqrt=gram_inv_sqrt,
        roughness=roughness,
    )


def verify_nested(v0: MotherBasis, v1: MotherBasis) -> float:
    projection, *_ = np.linalg.lstsq(v1.lag_design, v0.lag_design, rcond=None)
    residual = v0.lag_design - v1.lag_design @ projection
    return float(np.max(np.abs(residual)))


def full_feature_tensor(sequence_u: np.ndarray, basis: MotherBasis, cadence_sec: float) -> np.ndarray:
    values = np.asarray(sequence_u, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(CHANNELS):
        raise ValueError(f"INVALID_SEQUENCE_SHAPE:{values.shape}")
    dt_minutes = float(cadence_sec) / 60.0
    return np.einsum(
        "nlc,la->nca", values, basis.lag_design, optimize=True
    ) * dt_minutes


def flatten_support(tensor: np.ndarray, support: tuple[int, ...]) -> np.ndarray:
    if len(support) == 0:
        return np.zeros((len(tensor), 0), dtype=np.float64)
    return np.asarray(tensor[:, support, :], dtype=np.float64).reshape(len(tensor), -1)


def block_penalty(
    basis: MotherBasis,
    support_count: int,
    lambda_0: float,
    lambda_2: float,
) -> np.ndarray:
    if support_count == 0:
        return np.zeros((0, 0), dtype=np.float64)
    block = float(lambda_0) * np.eye(basis.size) + float(lambda_2) * basis.roughness
    return np.kron(np.eye(support_count), block)


def ridge_fit(
    matrix: np.ndarray,
    target: np.ndarray,
    penalty: np.ndarray,
    *,
    predict_matrix: np.ndarray | None = None,
) -> RidgeFit:
    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if matrix.shape[1] == 0:
        prediction_rows = len(target) if predict_matrix is None else len(predict_matrix)
        return RidgeFit(
            coefficient=np.zeros(0, dtype=np.float64),
            prediction=np.zeros(prediction_rows, dtype=np.float64),
            kkt_relative=0.0,
            condition_number=1.0,
        )
    n_rows = max(len(target), 1)
    gram = matrix.T @ matrix / n_rows + penalty
    rhs = matrix.T @ target / n_rows
    jitter = 0.0
    factor = None
    for _ in range(7):
        try:
            factor = cholesky(
                gram + jitter * np.eye(gram.shape[0]),
                lower=True,
                check_finite=False,
            )
            break
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0.0 else jitter * 100.0
    if factor is None:
        raise RuntimeError("CHOLESKY_FAILED")
    intermediate = solve_triangular(factor, rhs, lower=True, check_finite=False)
    coefficient = solve_triangular(
        factor.T, intermediate, lower=False, check_finite=False
    )
    residual = gram @ coefficient - rhs
    kkt = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-30))
    prediction_source = matrix if predict_matrix is None else np.asarray(
        predict_matrix, dtype=np.float64
    )
    prediction = prediction_source @ coefficient
    return RidgeFit(
        coefficient=coefficient,
        prediction=prediction,
        kkt_relative=kkt,
        condition_number=float(np.linalg.cond(gram)),
    )


def fold_mse(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    mean = float(np.mean(array))
    standard_error = (
        float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    )
    return mean, standard_error


def evaluate_regularization_task(payload: dict[str, Any]) -> dict[str, Any]:
    shared_root = Path(payload["shared_root"])
    direction = str(payload["direction"])
    support = tuple(int(value) for value in payload["support"])
    mother_name = str(payload["mother_name"])
    config = payload["config"]
    data = load_direction(shared_root, direction)
    protocol = load_protocol(shared_root)
    basis = mother_basis(
        mother_name,
        config["mother_spaces"][mother_name],
        sequence_steps=int(data.train["sequence_u"].shape[1]),
        cadence_sec=float(config["cadence_sec"]),
    )
    tensor = full_feature_tensor(
        data.train["sequence_u"], basis, float(config["cadence_sec"])
    )
    matrix = flatten_support(tensor, support)
    purge_raw = int(
        round(float(config["purge_min"]) * 60.0 / float(protocol["sample_period_sec"]))
    )
    folds = inner_folds(
        data.train["origin_raw_index"],
        protocol["inner_folds"],
        purge_raw_samples=purge_raw,
    )
    rows: list[dict[str, Any]] = []
    for lambda_0, lambda_2 in itertools.product(
        config["ridge_grid"], config["smooth_grid"]
    ):
        penalty = block_penalty(
            basis, len(support), float(lambda_0), float(lambda_2)
        )
        errors: list[float] = []
        fold_coefficients: list[list[float]] = []
        maximum_kkt = 0.0
        maximum_condition = 0.0
        for training, validation in folds:
            fit = ridge_fit(
                matrix[training],
                data.train["target_z"][training],
                penalty,
                predict_matrix=matrix[validation],
            )
            error = data.train["target_z"][validation] - fit.prediction
            errors.append(float(np.mean(error * error)))
            fold_coefficients.append(fit.coefficient.tolist())
            maximum_kkt = max(maximum_kkt, fit.kkt_relative)
            maximum_condition = max(maximum_condition, fit.condition_number)
        mean, standard_error = fold_mse(errors)
        rows.append(
            {
                "direction": direction,
                "mother_space": mother_name,
                "support": list(support),
                "support_mask": sum(1 << value for value in support),
                "lambda_0": float(lambda_0),
                "lambda_2": float(lambda_2),
                "fold_mse": errors,
                "oof_mse": mean,
                "oof_se": standard_error,
                "maximum_kkt": maximum_kkt,
                "maximum_condition": maximum_condition,
                "fold_coefficients": fold_coefficients,
                "basis_size": basis.size,
            }
        )
    return {
        "direction": direction,
        "mother_space": mother_name,
        "support": list(support),
        "rows": rows,
    }


def select_one_se(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError("NO_SELECTION_ROWS")
    minimum = min(rows, key=lambda row: (float(row["oof_mse"]), int(row["support_mask"])))
    threshold = float(minimum["oof_mse"]) + float(minimum["oof_se"])
    candidates = [row for row in rows if float(row["oof_mse"]) <= threshold + 1e-15]
    selected = min(
        candidates,
        key=lambda row: (
            len(row["support"]),
            -float(row["lambda_2"]),
            float(row["lambda_0"]),
            int(row["support_mask"]),
        ),
    )
    trace = {
        "minimum": {
            key: minimum[key]
            for key in (
                "mother_space",
                "support",
                "lambda_0",
                "lambda_2",
                "oof_mse",
                "oof_se",
            )
        },
        "one_se_threshold": threshold,
        "candidate_count": len(candidates),
        "selected": {
            key: selected[key]
            for key in (
                "mother_space",
                "support",
                "lambda_0",
                "lambda_2",
                "oof_mse",
                "oof_se",
            )
        },
    }
    return selected, trace


def coefficient_matrix(
    coefficient: np.ndarray,
    support: tuple[int, ...],
    basis_size: int,
) -> np.ndarray:
    matrix = np.zeros((basis_size, len(CHANNELS)), dtype=np.float64)
    if support:
        active = np.asarray(coefficient, dtype=np.float64).reshape(len(support), basis_size)
        for row, channel in enumerate(support):
            matrix[:, channel] = active[row]
    return matrix


def whitened_svd(
    coefficient_by_channel: np.ndarray, basis: MotherBasis
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    whitened = basis.gram_sqrt @ coefficient_by_channel
    return np.linalg.svd(whitened, full_matrices=False)


def mode_coefficients(left_vectors: np.ndarray, basis: MotherBasis) -> np.ndarray:
    return basis.gram_inv_sqrt @ left_vectors


def filtered_mode_features(
    tensor: np.ndarray,
    mode_coefficient: np.ndarray,
    support: tuple[int, ...],
) -> np.ndarray:
    if len(support) == 0 or mode_coefficient.shape[1] == 0:
        return np.zeros((len(tensor), 0), dtype=np.float64)
    columns = []
    for channel in support:
        columns.append(tensor[:, channel, :] @ mode_coefficient)
    return np.column_stack(columns)


def absolute_gram_correlation(
    first: np.ndarray, second: np.ndarray, gram: np.ndarray
) -> float:
    numerator = float(first.T @ gram @ second)
    denominator = math.sqrt(
        max(float(first.T @ gram @ first), 0.0)
        * max(float(second.T @ gram @ second), 0.0)
    )
    return abs(numerator) / denominator if denominator else 0.0


def principal_angle_degrees(
    first_whitened: np.ndarray, second_whitened: np.ndarray
) -> np.ndarray:
    if first_whitened.shape[1] == 0 or second_whitened.shape[1] == 0:
        return np.array([], dtype=np.float64)
    return np.degrees(subspace_angles(first_whitened, second_whitened))


def participation(loadings: np.ndarray) -> dict[str, Any]:
    squared = np.asarray(loadings, dtype=np.float64) ** 2
    total = float(np.sum(squared))
    fractions = squared / total if total else np.zeros_like(squared)
    ratio = 1.0 / float(np.sum(fractions * fractions)) if total else 0.0
    return {
        "fractions": fractions.tolist(),
        "participation_ratio": ratio,
        "channels_at_least_10pct": int(np.sum(fractions >= 0.1)),
    }


def support_subsets() -> list[tuple[int, ...]]:
    return [
        tuple(index for index in range(len(CHANNELS)) if mask & (1 << index))
        for mask in range(1 << len(CHANNELS))
    ]
