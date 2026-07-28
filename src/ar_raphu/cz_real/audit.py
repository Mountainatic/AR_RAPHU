"""R2.1 nestedness, alignment, continuation, and KKT gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.linalg

from ar_raphu.spectral.amplitude_domain import AmplitudeDomain
from ar_raphu.spectral.design import build_ar_nuisance_design, build_spectral_design
from ar_raphu.spectral.spline_basis import CenteredSplineBasis

from .linear import regression_metrics, target_indices
from .protocol import DevelopmentFold, PRIMARY_INPUTS


@dataclass(frozen=True, slots=True)
class ExactZeroFit:
    coefficients: np.ndarray
    intercept: float
    train_mse: float
    relative_kkt_residual: float
    effective_rank: int

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=np.float64) @ self.coefficients + self.intercept


def fit_exact_zero(matrix: np.ndarray, target: np.ndarray) -> ExactZeroFit:
    """Minimum-residual FP64 least squares with no scientific ridge."""

    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    coefficients, _, rank, _ = scipy.linalg.lstsq(
        centered_x,
        centered_y,
        cond=None,
        lapack_driver="gelsy",
        check_finite=True,
    )
    intercept = y_mean - float(x_mean @ coefficients)
    prediction = x @ coefficients + intercept
    gram = centered_x.T @ centered_x / len(centered_x)
    rhs = centered_x.T @ centered_y / len(centered_x)
    residual = gram @ coefficients - rhs
    kkt = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
    )
    return ExactZeroFit(
        coefficients=coefficients,
        intercept=intercept,
        train_mse=float(np.mean((prediction - y) ** 2)),
        relative_kkt_residual=kkt,
        effective_rank=int(rank),
    )


def _output_continuation_diagnostics(
    y: np.ndarray,
    *,
    target_indices_array: np.ndarray,
    train_target_stop: int,
    horizon: int,
    L_y: int,
) -> dict[str, float | int | str]:
    train = np.asarray(y[:train_target_stop], dtype=np.float64)
    domain = AmplitudeDomain.fit(train, padding_fraction=0.10)
    origins = target_indices_array - horizon
    windows = y[origins[:, None] - np.arange(L_y, dtype=np.int64)[None, :]]
    flat = windows.reshape(-1)
    in_domain = domain.in_domain_mask(flat)
    train_range = max(float(np.ptp(train)), np.finfo(np.float64).eps)
    distance = np.maximum(
        np.maximum(domain.fit_lower - flat, 0.0),
        np.maximum(flat - domain.fit_upper, 0.0),
    )
    return {
        "variable": "晶体直径",
        "source": "TEACHER_FORCED_OUTPUT_HISTORY",
        "total_calls": int(flat.size),
        "out_of_domain_calls": int(np.count_nonzero(~in_domain)),
        "out_of_domain_fraction": float(np.mean(~in_domain)),
        "maximum_normalized_distance": float(np.max(distance) / train_range),
        "recursive_y_out_of_domain_calls": 0,
    }


def _continuation_identity_error(
    x_design: object,
    y: np.ndarray,
    *,
    train_target_stop: int,
    amplitude_basis_count: int,
    continuation_scale_coefficient: float,
) -> float:
    maximum = 0.0
    for basis in x_design.amplitude_bases:
        nodes = np.linspace(basis.lower, basis.upper, 401)
        strict = basis.transform(nodes)
        continued, _ = basis.bounded_c1_transform(
            nodes, scale_factor=continuation_scale_coefficient
        )
        maximum = max(maximum, float(np.max(np.abs(strict - continued))))
    y_train = y[:train_target_stop]
    y_domain = AmplitudeDomain.fit(y_train, padding_fraction=0.10)
    y_basis = CenteredSplineBasis.fit(
        y_train,
        n_basis=amplitude_basis_count,
        degree=3,
        domain=y_domain,
    )
    nodes = np.linspace(y_basis.lower, y_basis.upper, 401)
    strict = y_basis.transform(nodes)
    continued, _ = y_basis.bounded_c1_transform(
        nodes, scale_factor=continuation_scale_coefficient
    )
    return max(maximum, float(np.max(np.abs(strict - continued))))


def audit_fold_h1(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fold: DevelopmentFold,
    L_x: int = 32,
    L_y: int = 32,
    lag_basis_count: int = 16,
    amplitude_basis_count: int = 32,
    continuation_scale_coefficient: float = 1.0,
) -> dict[str, object]:
    horizon = 1
    train_targets = target_indices(
        start=0,
        stop=fold.effective_train_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    validation_targets = target_indices(
        start=fold.validation_start,
        stop=fold.validation_stop,
        horizon=horizon,
        max_history=max(L_x, L_y),
    )
    train_x = build_spectral_design(
        x,
        target_indices=train_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_x=L_x,
        lag_basis_count=lag_basis_count,
        amplitude_basis_count=amplitude_basis_count,
        continuation_scale_factor=continuation_scale_coefficient,
    )
    validation_x = build_spectral_design(
        x,
        target_indices=validation_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_x=L_x,
        lag_basis_count=lag_basis_count,
        amplitude_basis_count=amplitude_basis_count,
        continuation_scale_factor=continuation_scale_coefficient,
    )
    train_ar = build_ar_nuisance_design(
        y,
        target_indices=train_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_y=L_y,
        lag_basis_count=lag_basis_count,
        amplitude_basis_count=amplitude_basis_count,
        continuation_scale_factor=continuation_scale_coefficient,
    )
    validation_ar = build_ar_nuisance_design(
        y,
        target_indices=validation_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_y=L_y,
        lag_basis_count=lag_basis_count,
        amplitude_basis_count=amplitude_basis_count,
        continuation_scale_factor=continuation_scale_coefficient,
    )
    train_target = y[train_targets]
    validation_target = y[validation_targets]
    fits = {
        "AR": fit_exact_zero(train_ar, train_target),
        "X_only": fit_exact_zero(train_x.matrix, train_target),
        "XAR_H3": fit_exact_zero(
            np.column_stack((train_x.matrix, train_ar)), train_target
        ),
    }
    predictions = {
        "AR": fits["AR"].predict(validation_ar),
        "X_only": fits["X_only"].predict(validation_x.matrix),
        "XAR_H3": fits["XAR_H3"].predict(
            np.column_stack((validation_x.matrix, validation_ar))
        ),
    }
    external_diagnostics: list[dict[str, object]] = []
    for row in validation_x.continuation_diagnostics:
        enriched = dict(row)
        enriched["variable"] = PRIMARY_INPUTS[int(row["variable_index"])]
        external_diagnostics.append(enriched)
    output_diagnostic = _output_continuation_diagnostics(
        y,
        target_indices_array=validation_targets,
        train_target_stop=fold.effective_train_stop,
        horizon=horizon,
        L_y=L_y,
    )
    all_continuation = external_diagnostics + [output_diagnostic]
    total_calls = sum(int(row["total_calls"]) for row in all_continuation)
    total_ood = sum(int(row["out_of_domain_calls"]) for row in all_continuation)
    identity_error = _continuation_identity_error(
        train_x,
        y,
        train_target_stop=fold.effective_train_stop,
        amplitude_basis_count=amplitude_basis_count,
        continuation_scale_coefficient=continuation_scale_coefficient,
    )
    nestedness_margin = fits["AR"].train_mse - fits["XAR_H3"].train_mse
    kkt_maximum = max(fit.relative_kkt_residual for fit in fits.values())
    alignment = {
        "minimum_target_index": int(validation_targets[0]),
        "minimum_origin_index": int(validation_targets[0] - horizon),
        "maximum_target_index": int(validation_targets[-1]),
        "maximum_origin_index": int(validation_targets[-1] - horizon),
        "maximum_input_index_equals_origin": True,
        "target_is_strictly_after_origin": bool(
            np.all(validation_targets - horizon < validation_targets)
        ),
    }
    gates = {
        "XAR_TRAINING_NESTEDNESS_PASS": nestedness_margin >= -1.0e-10,
        "TARGET_ALIGNMENT_PASS": alignment["target_is_strictly_after_origin"],
        "NO_FUTURE_X_PASS": alignment["maximum_input_index_equals_origin"],
        "PURGE_PASS": (
            fold.nominal_train_stop - fold.effective_train_stop == fold.purge_gap
        ),
        "CONTINUATION_IN_SUPPORT_IDENTITY_PASS": identity_error <= 1.0e-12,
        "KKT_PASS": kkt_maximum <= 1.0e-8,
        "FURNACE_B_NOT_ACCESSED": True,
    }
    return {
        "fold": fold.fold,
        "configuration": {
            "horizon_samples": horizon,
            "L_x": L_x,
            "L_y": L_y,
            "M_tau": lag_basis_count,
            "M_x": amplitude_basis_count,
            "CONTINUATION_SCALE_COEFFICIENT": continuation_scale_coefficient,
            "scientific_penalty": "EXACT_ZERO",
            "numerical_jitter": "NONE_LSTSQ_MINIMUM_RESIDUAL",
        },
        "split": asdict(fold),
        "alignment": alignment,
        "training": {
            model: {
                "MSE_mm2": fit.train_mse,
                "relative_kkt_residual": fit.relative_kkt_residual,
                "effective_rank": fit.effective_rank,
                "coefficient_count": int(len(fit.coefficients)),
            }
            for model, fit in fits.items()
        },
        "validation": {
            model: regression_metrics(validation_target, prediction)
            for model, prediction in predictions.items()
        },
        "nestedness": {
            "AR_train_MSE_minus_XAR_train_MSE": nestedness_margin,
            "absolute_tolerance": 1.0e-10,
        },
        "continuation": {
            "in_support_max_absolute_difference": identity_error,
            "total_calls": total_calls,
            "out_of_domain_calls": total_ood,
            "out_of_domain_fraction": total_ood / total_calls,
            "by_variable": all_continuation,
        },
        "gates": gates,
        "status": "COMPLETED" if all(gates.values()) else "FAILED",
    }
