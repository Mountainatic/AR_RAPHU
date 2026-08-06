from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Hashable, Mapping, Sequence, TypeVar

import numpy as np


T = TypeVar("T")


@dataclass(frozen=True)
class ProfileRegretSelection:
    best_profile: Any
    best_mean: float
    best_standard_error: float
    one_se_threshold: float
    one_se_candidates: tuple[Any, ...]
    regret_guarded_candidates: tuple[Any, ...]
    retained_profiles: tuple[Any, ...]
    maximum_relative_regret: float
    usable_fold_count: dict[Any, int]
    means: dict[Any, float]
    standard_errors: dict[Any, float]

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["best_profile"] = str(self.best_profile)
        for key in (
            "one_se_candidates",
            "regret_guarded_candidates",
            "retained_profiles",
        ):
            value[key] = [str(item) for item in value[key]]
        for key in ("usable_fold_count", "means", "standard_errors"):
            value[key] = {str(item): result for item, result in value[key].items()}
        return value


def profile_one_se_regret_guard(
    fold_losses: Mapping[Hashable, Sequence[float]],
    complexity_key: Callable[[Hashable], tuple[Any, ...]],
    *,
    maximum_relative_regret: float = 0.02,
    maximum_retained_profiles: int = 2,
    minimum_usable_folds: int = 3,
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> ProfileRegretSelection:
    """Apply the registered one-SE and 2% relative-regret profile guard."""
    if maximum_relative_regret < 0:
        raise ValueError("maximum profile regret must be nonnegative")
    if maximum_retained_profiles not in {1, 2}:
        raise ValueError("v2.1.1 retains at most two K profiles")
    usable: dict[Hashable, np.ndarray] = {}
    for profile, values in fold_losses.items():
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if len(finite) >= minimum_usable_folds:
            usable[profile] = finite
    if not usable:
        raise ValueError("no K profile has the minimum usable folds")
    means = {
        profile: float(np.mean(values, dtype=np.float64))
        for profile, values in usable.items()
    }
    errors = {
        profile: float(np.std(values, ddof=1) / math.sqrt(len(values)))
        for profile, values in usable.items()
    }
    best = min(usable, key=lambda profile: (means[profile], complexity_key(profile)))
    threshold = means[best] + errors[best]
    one_se = tuple(
        profile
        for profile in usable
        if means[profile] < threshold
        or np.isclose(means[profile], threshold, rtol=rtol, atol=atol)
    )
    denominator = max(abs(means[best]), np.finfo(np.float64).eps)
    guarded = tuple(
        profile
        for profile in one_se
        if (means[profile] - means[best]) / denominator
        <= maximum_relative_regret + rtol
    )
    if best not in guarded:
        raise RuntimeError("best K profile was lost from its own regret guard")
    retained: list[Hashable] = [best]
    if maximum_retained_profiles == 2:
        distinct = [profile for profile in guarded if profile != best]
        if distinct:
            retained.append(min(distinct, key=complexity_key))
    return ProfileRegretSelection(
        best_profile=best,
        best_mean=means[best],
        best_standard_error=errors[best],
        one_se_threshold=threshold,
        one_se_candidates=one_se,
        regret_guarded_candidates=guarded,
        retained_profiles=tuple(retained),
        maximum_relative_regret=float(maximum_relative_regret),
        usable_fold_count={profile: len(values) for profile, values in usable.items()},
        means=means,
        standard_errors=errors,
    )


def numerical_contract_passes(contract: Mapping[str, Any]) -> bool:
    certificate = contract.get("numerical_certificate", contract.get("certificate", {}))
    status = certificate.get("status")
    if status not in {"PASS", "PASS_WITH_WARNING", "EXACT_ZERO"}:
        return False
    coefficient = np.asarray(contract.get("coefficient", contract.get("theta", [])), dtype=np.float64)
    if not np.isfinite(coefficient).all():
        return False
    if status == "EXACT_ZERO":
        return True
    relative_kkt = certificate.get("relative_kkt")
    condition = certificate.get("condition_number")
    rank = certificate.get("numerical_rank")
    coefficient_l2 = certificate.get("coefficient_l2")
    if relative_kkt is not None and float(relative_kkt) > 1e-8:
        return False
    if condition is not None and float(condition) > 1e14:
        return False
    if rank is not None and coefficient.size and int(rank) < 1:
        return False
    if coefficient_l2 is not None and float(coefficient_l2) > 1e6:
        return False
    return True


def select_smallest_stable(
    candidates: Sequence[T],
    fit_candidate: Callable[[T], Mapping[str, Any]],
    *,
    valid_candidate: Callable[[Mapping[str, Any]], bool] = numerical_contract_passes,
) -> tuple[T, Mapping[str, Any], list[dict[str, Any]]]:
    """Return the first registered candidate that passes numerical certificates."""
    audits: list[dict[str, Any]] = []
    for candidate in candidates:
        contract = fit_candidate(candidate)
        passed = bool(valid_candidate(contract))
        audits.append(
            {
                "candidate": candidate,
                "pass": passed,
                "certificate": contract.get(
                    "numerical_certificate", contract.get("certificate", {})
                ),
            }
        )
        if passed:
            return candidate, contract, audits
    raise RuntimeError("no registered ridge candidate passed numerical certificates")


def input_path_preservation_gate(
    target: np.ndarray,
    candidate_prediction: np.ndarray,
    best_active_k_prediction: np.ndarray,
    *,
    input_prediction: np.ndarray | None = None,
    nonintercept_coefficients: np.ndarray | Sequence[float] = (),
    numerical_certificate_passed: bool,
    minimum_variance_ratio_to_target: float = 1e-8,
    minimum_fraction_of_best_active_k_variance_ratio: float = 0.10,
    maximum_mse_ratio_vs_best_active_k: float = 1.02,
    minimum_nonintercept_coefficient_abs: float = 1e-10,
) -> dict[str, Any]:
    """Shared scale-independent PF/Joint input-path preservation gate."""
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(candidate_prediction, dtype=np.float64).reshape(-1)
    best_k = np.asarray(best_active_k_prediction, dtype=np.float64).reshape(-1)
    input_values = (
        prediction
        if input_prediction is None
        else np.asarray(input_prediction, dtype=np.float64).reshape(-1)
    )
    if not (len(y) == len(prediction) == len(best_k) == len(input_values)):
        raise ValueError("input-path gate arrays have different row counts")
    finite = (
        np.isfinite(y)
        & np.isfinite(prediction)
        & np.isfinite(best_k)
        & np.isfinite(input_values)
    )
    if int(finite.sum()) < 3:
        return {
            "status": "INPUT_PATH_COLLAPSED",
            "pass": False,
            "reason": "INSUFFICIENT_FINITE_ROWS",
            "finite_rows": int(finite.sum()),
        }
    y = y[finite]
    prediction = prediction[finite]
    best_k = best_k[finite]
    input_values = input_values[finite]
    target_variance = float(np.var(y, dtype=np.float64))
    if target_variance <= np.finfo(np.float64).tiny:
        return {
            "status": "INPUT_PATH_COLLAPSED",
            "pass": False,
            "reason": "TARGET_VARIANCE_ZERO",
            "finite_rows": len(y),
        }
    input_variance_ratio = float(np.var(input_values, dtype=np.float64) / target_variance)
    best_k_variance_ratio = float(np.var(best_k, dtype=np.float64) / target_variance)
    required_variance_ratio = max(
        float(minimum_variance_ratio_to_target),
        float(minimum_fraction_of_best_active_k_variance_ratio)
        * best_k_variance_ratio,
    )
    candidate_mse = float(np.mean(np.square(y - prediction), dtype=np.float64))
    best_k_mse = float(np.mean(np.square(y - best_k), dtype=np.float64))
    mse_limit = float(maximum_mse_ratio_vs_best_active_k) * best_k_mse
    coefficients = np.asarray(nonintercept_coefficients, dtype=np.float64).reshape(-1)
    coefficient_nonzero = bool(
        coefficients.size
        and np.isfinite(coefficients).all()
        and np.max(np.abs(coefficients), initial=0.0)
        > float(minimum_nonintercept_coefficient_abs)
    )
    checks = {
        "variance": input_variance_ratio >= required_variance_ratio,
        "mse": candidate_mse <= mse_limit,
        "coefficient": coefficient_nonzero,
        "numerical_certificate": bool(numerical_certificate_passed),
    }
    passed = all(checks.values())
    return {
        "status": "INPUT_PATH_PRESERVED" if passed else "INPUT_PATH_COLLAPSED",
        "pass": bool(passed),
        "checks": checks,
        "finite_rows": len(y),
        "target_variance": target_variance,
        "input_prediction_variance_ratio_to_target": input_variance_ratio,
        "best_active_k_variance_ratio_to_target": best_k_variance_ratio,
        "required_variance_ratio_to_target": required_variance_ratio,
        "candidate_mse": candidate_mse,
        "best_active_k_mse": best_k_mse,
        "maximum_candidate_mse": mse_limit,
        "maximum_nonintercept_coefficient_abs": float(
            np.max(np.abs(coefficients), initial=0.0)
        ),
    }


def attach_nonselecting_validation_confirmation(
    formal_oof_gate: Mapping[str, Any],
    validation_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Record materialized-validation behavior without changing the OOF decision."""
    result = dict(formal_oof_gate)
    result["oof_confirmation_passed"] = bool(formal_oof_gate.get("pass", False))
    result["validation_confirmation"] = {
        **dict(validation_gate),
        "selection_eligible": False,
        "role": "POST_SELECTION_MATERIALIZATION_DIAGNOSTIC",
    }
    return result
