from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .c2_models import (
    _folds,
    _pls_predictions,
    _prediction_frame,
    run_job as run_legacy_c2_job,
)
from .c3_models import (
    _arx_features,
    _hammerstein_profiles,
    _narx_expand,
    _nonlinear_features,
    _ridge_block_predict,
    run_job as run_legacy_c3_job,
)
from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    deterministic_subsample,
    input_columns,
    realized_state_profiles,
    sha256_file,
)
from .cpu_selection import mse, regression_metrics, select_one_se
from .stage0 import write_json
from .v2_runtime import release_process_memory, run_parallel
from .v211_public_all_views import (
    public_all_dynamic_views,
    public_all_input_views,
)
from .v211_support import (
    SUPPORT_CONTRACT,
    fold_evaluation_causal_floor,
    load_native_samples,
    require_native_support_contract,
    support_id_hash,
)
from .v211_history_override import load_tep_history_override


STATIC_INPUT_MODELS = ("RIDGE", "PLS", "RBF_SVR", "XGBOOST")
NATIVE_INPUT_MODELS = (
    "DPLS",
    "PARALLEL_HAMMERSTEIN",
    "HAMMERSTEIN_WIENER",
)
DYNAMIC_MODELS = ("AR", "ARX", "LINEAR_NARX", "N4SID")
SIMPLE_MODELS = ("MEAN", "PERSISTENCE", "SEASONAL_PERSISTENCE")
COMPLETED_STATUSES = {
    "PASS",
    "FAILED_RETAINED",
    "NOT_RUN_IMPLEMENTATION_ABSENT",
    "NOT_RUN_PROTOCOL_INCOMPATIBLE",
}
SUCCESSFUL_COMPLETION_STATUSES = {
    "PASS",
    "NOT_RUN_IMPLEMENTATION_ABSENT",
    "NOT_RUN_PROTOCOL_INCOMPATIBLE",
}


@dataclass(frozen=True, order=True)
class SupportRequirement:
    input_history_steps: int = 0
    target_delta_steps: int = 0
    target_history_steps: int = 0

    def __post_init__(self) -> None:
        if self.input_history_steps < 0:
            raise ValueError("input history must be nonnegative")
        if (self.target_delta_steps == 0) != (self.target_history_steps == 0):
            raise ValueError("target delta/history must both be zero or positive")
        if self.target_delta_steps < 0 or self.target_history_steps < 0:
            raise ValueError("target history values must be nonnegative")

    def to_json(self) -> dict[str, int]:
        return {
            "input_history_steps": int(self.input_history_steps),
            "target_delta_steps": int(self.target_delta_steps),
            "target_history_steps": int(self.target_history_steps),
        }


@dataclass(frozen=True)
class FoldSupport:
    fit: pd.DataFrame
    evaluation: pd.DataFrame
    fit_native_rows: int
    fit_support_hash: str
    evaluation_common_rows: int
    evaluation_support_hash: str


def _freeze(project: Path) -> dict[str, Any]:
    return json.loads(
        (project / "configs/cpu_model_freeze_v1.json").read_text(encoding="utf-8")
    )


def _additional_floor(
    samples: pd.DataFrame,
    additional_causal_history_floor: int | Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    floor = samples["causal_history_floor"].to_numpy(dtype=np.int64)
    if additional_causal_history_floor is None:
        return floor
    additional = np.asarray(additional_causal_history_floor, dtype=np.int64)
    if additional.ndim == 0:
        additional = np.full(len(samples), int(additional), dtype=np.int64)
    if additional.shape != floor.shape:
        raise ValueError("additional causal floor does not match sample rows")
    return np.maximum(floor, additional)


def requirement_mask(
    samples: pd.DataFrame,
    requirement: SupportRequirement,
    additional_causal_history_floor: int | Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    require_native_support_contract(samples)
    floor = _additional_floor(samples, additional_causal_history_floor)
    mask = np.ones(len(samples), dtype=bool)
    if requirement.input_history_steps:
        origins = samples["origin"].to_numpy(dtype=np.int64)
        mask &= origins - int(requirement.input_history_steps) >= floor
    if requirement.target_history_steps:
        delta = int(requirement.target_delta_steps)
        history = int(requirement.target_history_steps)
        count = max(1, history // delta)
        oldest_offset = (count - 1) * delta
        latest = samples["latest_available_target_index"].to_numpy(dtype=np.int64)
        mask &= latest - oldest_offset >= floor
    return mask


def apply_requirement(
    samples: pd.DataFrame,
    requirement: SupportRequirement,
    additional_causal_history_floor: int | Sequence[int] | np.ndarray | None = None,
) -> pd.DataFrame:
    return samples.loc[
        requirement_mask(samples, requirement, additional_causal_history_floor)
    ].copy()


def apply_common_requirements(
    samples: pd.DataFrame,
    requirements: Iterable[SupportRequirement],
    additional_causal_history_floor: int | Sequence[int] | np.ndarray | None = None,
) -> pd.DataFrame:
    registered = tuple(dict.fromkeys(requirements))
    require_native_support_contract(samples)
    mask = np.ones(len(samples), dtype=bool)
    for requirement in registered:
        mask &= requirement_mask(
            samples,
            requirement,
            additional_causal_history_floor,
        )
    return samples.loc[mask].copy()


def _cap_after_support(samples: pd.DataFrame, cap: int) -> pd.DataFrame:
    index = deterministic_subsample(samples, int(cap))
    return samples.iloc[index].reset_index(drop=True)


def candidate_fold_supports(
    train: pd.DataFrame,
    view: ViewSpec,
    requirements: Iterable[SupportRequirement],
    *,
    fit_cap: int,
    evaluation_cap: int,
) -> tuple[dict[SupportRequirement, list[FoldSupport]], dict[str, Any]]:
    registered = tuple(sorted(set(requirements)))
    if not registered:
        raise ValueError("at least one support requirement is required")
    raw_folds = list(_folds(train, view))
    active = list(registered)
    unavailable: list[SupportRequirement] = []
    selected_folds: list[
        tuple[pd.DataFrame, pd.DataFrame, dict[SupportRequirement, pd.DataFrame]]
    ] = []

    def severity(requirement: SupportRequirement) -> tuple[int, int, int]:
        return (
            requirement.input_history_steps
            + requirement.target_history_steps
            + requirement.target_delta_steps,
            requirement.input_history_steps,
            requirement.target_history_steps,
        )

    while active:
        candidate_folds = []
        invalid_requirements: set[SupportRequirement] = set()
        for fit_index, evaluation_index in raw_folds:
            fit_raw = train.iloc[fit_index]
            evaluation_raw = train.iloc[evaluation_index]
            evaluation_floor = fold_evaluation_causal_floor(
                fit_raw, evaluation_raw
            )
            fit_by_requirement = {
                requirement: apply_requirement(fit_raw, requirement)
                for requirement in active
            }
            common = apply_common_requirements(
                evaluation_raw,
                active,
                evaluation_floor,
            )
            missing = {
                requirement
                for requirement, fit in fit_by_requirement.items()
                if fit.empty
            }
            if missing or common.empty:
                invalid_requirements.update(missing or set(active))
                continue
            candidate_folds.append((fit_raw, common, fit_by_requirement))
        if len(candidate_folds) == len(raw_folds) and len(candidate_folds) >= 2:
            selected_folds = candidate_folds
            break
        if len(active) == 1:
            if len(candidate_folds) < 2:
                raise ValueError(
                    "no support-compatible inner folds for baseline candidates"
                )
            selected_folds = candidate_folds
            break
        remove = max(invalid_requirements or set(active), key=severity)
        active.remove(remove)
        unavailable.append(remove)

    if not active or not selected_folds:
        raise ValueError("no support-compatible baseline candidates")
    active_requirements = tuple(active)
    result = {requirement: [] for requirement in active_requirements}
    common_rows: list[int] = []
    common_hashes: list[str] = []
    for fit_raw, evaluation_common, fit_by_requirement in selected_folds:
        evaluation_common = _cap_after_support(
            evaluation_common,
            evaluation_cap,
        )
        evaluation_hash = support_id_hash(evaluation_common)
        common_rows.append(len(evaluation_common))
        common_hashes.append(evaluation_hash)
        for requirement in active_requirements:
            fit_native = fit_by_requirement[requirement]
            fit = _cap_after_support(fit_native, fit_cap)
            result[requirement].append(
                FoldSupport(
                    fit=fit,
                    evaluation=evaluation_common,
                    fit_native_rows=len(fit_native),
                    fit_support_hash=support_id_hash(fit),
                    evaluation_common_rows=len(evaluation_common),
                    evaluation_support_hash=evaluation_hash,
                )
            )
    audit = {
        "support_contract": SUPPORT_CONTRACT,
        "candidate_native_fitting": True,
        "common_validation_scoring": True,
        "row_cap_applied_after_native_mask": True,
        "fit_cap": int(fit_cap),
        "evaluation_cap": int(evaluation_cap),
        "available_requirements": [
            requirement.to_json() for requirement in active_requirements
        ],
        "unavailable_requirements": [
            requirement.to_json() for requirement in unavailable
        ],
        "raw_fold_count": len(raw_folds),
        "selected_fold_count": len(selected_folds),
        "common_validation_rows_by_fold": common_rows,
        "common_validation_support_hash_by_fold": common_hashes,
        "requirements": [
            requirement.to_json() for requirement in active_requirements
        ],
        "fit_rows_by_requirement": {
            json.dumps(requirement.to_json(), sort_keys=True): [
                fold.fit_native_rows for fold in result[requirement]
            ]
            for requirement in active_requirements
        },
        "fit_support_hash_by_requirement": {
            json.dumps(requirement.to_json(), sort_keys=True): [
                fold.fit_support_hash for fold in result[requirement]
            ]
            for requirement in active_requirements
        },
    }
    return result, audit


def _result_root(output: Path, family: str, model: str, view: ViewSpec) -> Path:
    return output / family / "PREDICTIONS" / model / view.relative_root


def _completed(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if value.get("status") in COMPLETED_STATUSES else None


def _write_prediction_result(
    *,
    output: Path,
    family: str,
    stage: str,
    model: str,
    view: ViewSpec,
    samples: pd.DataFrame,
    prediction: np.ndarray,
    selection: Mapping[str, Any],
    parameter_count: int,
    started: float,
) -> dict[str, Any]:
    destination = _result_root(output, family, model, view)
    destination.mkdir(parents=True, exist_ok=True)
    frame = _prediction_frame(samples, view, model, prediction, parameter_count)
    prediction_path = destination / "validation.parquet"
    frame.to_parquet(prediction_path, index=False, compression="zstd")
    result = {
        "status": "PASS",
        "stage": stage,
        "support_contract": SUPPORT_CONTRACT,
        "model": model,
        "dataset": view.head.dataset,
        "task": view.head.task_id,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "split": "validation",
        "rows": len(frame),
        "parameter_count": int(parameter_count),
        "dtype": "float64",
        "test_accessed": False,
        "ood_accessed": False,
        "selection": dict(selection),
        "prediction_path": str(prediction_path.relative_to(output)),
        "prediction_sha256": sha256_file(prediction_path),
        "elapsed_seconds": time.time() - started,
        **regression_metrics(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        ),
    }
    write_json(destination / "RESULT.json", result)
    return result


def _failure_result(
    *,
    output: Path,
    family: str,
    stage: str,
    model: str,
    view: ViewSpec,
    started: float,
    error: Exception,
) -> dict[str, Any]:
    destination = _result_root(output, family, model, view)
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "FAILED_RETAINED",
        "stage": stage,
        "support_contract": SUPPORT_CONTRACT,
        "model": model,
        "dataset": view.head.dataset,
        "task": view.head.task_id,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "test_accessed": False,
        "ood_accessed": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "elapsed_seconds": time.time() - started,
    }
    write_json(destination / "RESULT.json", result)
    return result


def run_simple_job(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    model: str,
) -> dict[str, Any]:
    del project
    started = time.time()
    destination = _result_root(output, "C2", model, view)
    destination.mkdir(parents=True, exist_ok=True)
    if model == "SEASONAL_PERSISTENCE":
        result = {
            "status": "NOT_RUN_PROTOCOL_INCOMPATIBLE",
            "stage": "B1_PUBLIC_ALL_SIMPLE_DEVELOPMENT",
            "support_contract": SUPPORT_CONTRACT,
            "model": model,
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "information_set": view.information_set,
            "availability_scenario": view.availability_scenario,
            "proxy_policy": view.proxy_policy,
            "reason": "NO_TRAIN_ONLY_REGISTERED_PERIOD",
            "test_accessed": False,
            "ood_accessed": False,
            "elapsed_seconds": time.time() - started,
        }
        write_json(destination / "RESULT.json", result)
        return result
    try:
        train = load_native_samples(shared, view, "train")
        validation = load_native_samples(shared, view, "validation")
        if model == "MEAN":
            value = float(np.mean(train["y_true"].to_numpy(dtype=np.float64)))
            parameter_count = 1
            selection = {
                "fit_partition": "train_only",
                "support_contract": SUPPORT_CONTRACT,
                "native_history_steps": 0,
                "train_rows": len(train),
                "train_support_hash": support_id_hash(train),
                "validation_support_hash": support_id_hash(validation),
            }
        elif model == "PERSISTENCE":
            value = 0.0
            parameter_count = 0
            selection = {
                "formula": "registered_change_equals_zero",
                "support_contract": SUPPORT_CONTRACT,
                "native_history_steps": 0,
                "validation_support_hash": support_id_hash(validation),
            }
        else:
            raise KeyError(model)
        prediction = np.full(len(validation), value, dtype=np.float64)
        return _write_prediction_result(
            output=output,
            family="C2",
            stage="B1_PUBLIC_ALL_SIMPLE_DEVELOPMENT",
            model=model,
            view=view,
            samples=validation,
            prediction=prediction,
            selection=selection,
            parameter_count=parameter_count,
            started=started,
        )
    except Exception as error:
        return _failure_result(
            output=output,
            family="C2",
            stage="B1_PUBLIC_ALL_SIMPLE_DEVELOPMENT",
            model=model,
            view=view,
            started=started,
            error=error,
        )


def run_static_input_job(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    model: str,
) -> dict[str, Any]:
    started = time.time()
    try:
        result = run_legacy_c2_job(shared, project, output / "C2", view, model)
        path = _result_root(output, "C2", model, view) / "RESULT.json"
        if result.get("status") != "PASS":
            return result
        train = load_native_samples(shared, view, "train")
        validation = load_native_samples(shared, view, "validation")
        requirement = SupportRequirement(input_history_steps=1)
        native_train = apply_requirement(train, requirement)
        native_validation = apply_requirement(validation, requirement)
        if len(native_train) != len(train) or len(native_validation) != len(
            validation
        ):
            raise RuntimeError(
                "static strict-past snapshot escaped the registered anchor support"
            )
        selection = dict(result["selection"])
        selection["native_support_audit"] = {
            "support_contract": SUPPORT_CONTRACT,
            "requirement": requirement.to_json(),
            "anchor_train_rows": len(train),
            "native_train_rows": len(native_train),
            "anchor_validation_rows": len(validation),
            "native_validation_rows": len(native_validation),
            "native_train_support_hash": support_id_hash(native_train),
            "native_validation_support_hash": support_id_hash(native_validation),
            "row_cap_applied_after_native_mask": True,
            "mask_is_identity_on_registered_anchor_universe": True,
        }
        result["selection"] = selection
        result["support_contract"] = SUPPORT_CONTRACT
        result["ood_accessed"] = False
        write_json(path, result)
        return result
    except Exception as error:
        return _failure_result(
            output=output,
            family="C2",
            stage="B1_PUBLIC_ALL_STATIC_INPUT_DEVELOPMENT",
            model=model,
            view=view,
            started=started,
            error=error,
        )


def _fit_pls_selected(
    x_train: np.ndarray,
    y_train: np.ndarray,
    components: int,
) -> Any:
    from sklearn.cross_decomposition import PLSRegression

    model = PLSRegression(
        n_components=int(components),
        scale=True,
        max_iter=1000,
        tol=1e-8,
    )
    model.fit(x_train, y_train)
    return model


def _chunked_features_prediction(
    samples: pd.DataFrame,
    feature_function: Callable[[pd.DataFrame], np.ndarray],
    prediction_function: Callable[[np.ndarray], np.ndarray],
    chunk_rows: int = 50_000,
) -> np.ndarray:
    prediction = np.empty(len(samples), dtype=np.float64)
    for start in range(0, len(samples), chunk_rows):
        stop = min(len(samples), start + chunk_rows)
        features = feature_function(samples.iloc[start:stop])
        prediction[start:stop] = np.asarray(
            prediction_function(features), dtype=np.float64
        ).reshape(-1)
    return prediction


def run_dpls_job(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    history_override_config: Path | str | None = None,
) -> dict[str, Any]:
    started = time.time()
    model_name = "DPLS"
    try:
        freeze = _freeze(project)
        config = freeze["c2"]["dpls"]
        history_override = load_tep_history_override(history_override_config)
        if history_override is not None:
            history_override.require_view(view)
            if str(view.information_set) != "input_only":
                raise RuntimeError("TEP DPLS history override requires input_only")
        train = load_native_samples(shared, view, "train")
        validation = load_native_samples(shared, view, "validation")
        columns = input_columns(shared, view.head.task_id, view.proxy_policy)
        multipliers = (
            config["lag_coverage_for_positive_h"]
            if history_override is None
            else history_override.positive_h_history_multipliers
        )
        histories = [
            max(1, int(multiplier) * view.head.h_steps)
            for multiplier in multipliers
        ]
        histories = sorted(set(histories))
        requirements = {
            history: SupportRequirement(input_history_steps=history)
            for history in histories
        }
        fold_supports, support_audit = candidate_fold_supports(
            train,
            view,
            requirements.values(),
            fit_cap=int(freeze["selection"]["fit_row_cap_default"]),
            evaluation_cap=int(
                freeze["selection"]["selection_validation_row_cap_default"]
            ),
        )
        if history_override is not None:
            unavailable = {
                int(item["input_history_steps"])
                for item in support_audit["unavailable_requirements"]
            }
            forbidden = unavailable.intersection(
                history_override.fail_if_history_unavailable
            )
            if forbidden:
                raise RuntimeError(
                    f"required TEP DPLS histories are unavailable: {sorted(forbidden)}"
                )
        histories = [
            history
            for history in histories
            if requirements[history] in fold_supports
        ]
        requirements = {
            history: requirements[history] for history in histories
        }
        if not histories:
            raise ValueError("no support-compatible DPLS histories")
        candidates = [
            (history, int(components))
            for history in histories
            for components in config["components"]
        ]
        if (
            history_override is not None
            and len(candidates)
            != history_override.dpls_maximum_joint_configurations
        ):
            raise RuntimeError(
                "TEP DPLS joint grid must contain exactly "
                f"{history_override.dpls_maximum_joint_configurations} "
                f"configurations, got {len(candidates)}"
            )
        losses = {candidate: [] for candidate in candidates}
        accessor = BaseAccessor(shared, view.head.dataset, "train", columns)
        for fold_index in range(len(next(iter(fold_supports.values())))):
            evaluation = next(iter(fold_supports.values()))[fold_index].evaluation
            y_evaluation = evaluation["y_true"].to_numpy(dtype=np.float64)
            for history in histories:
                fit = fold_supports[requirements[history]][fold_index].fit
                x_fit = accessor.input_lags(
                    fit,
                    columns,
                    history,
                    int(
                        config["maximum_lags_per_channel"]
                        if history_override is None
                        else history_override.dpls_maximum_lags_per_channel
                    ),
                )
                x_evaluation = accessor.input_lags(
                    evaluation,
                    columns,
                    history,
                    int(
                        config["maximum_lags_per_channel"]
                        if history_override is None
                        else history_override.dpls_maximum_lags_per_channel
                    ),
                )
                y_fit = fit["y_true"].to_numpy(dtype=np.float64)
                for components in [
                    value for value in config["components"]
                ]:
                    candidate = (history, int(components))
                    if components > min(x_fit.shape[1], len(x_fit) - 1):
                        losses[candidate].append(float("inf"))
                        continue
                    predicted, _ = _pls_predictions(
                        x_fit,
                        y_fit,
                        x_evaluation,
                        int(components),
                    )
                    losses[candidate].append(mse(y_evaluation, predicted))
                del x_fit, x_evaluation, y_fit
                release_process_memory()
        selected = select_one_se(losses, lambda value: (value[0], value[1]))
        history, components = selected.candidate
        requirement = requirements[int(history)]
        native_train = apply_requirement(train, requirement)
        fit = _cap_after_support(
            native_train,
            int(freeze["selection"]["fit_row_cap_default"]),
        )
        native_validation = apply_requirement(validation, requirement).reset_index(
            drop=True
        )
        x_fit = accessor.input_lags(
            fit,
            columns,
            int(history),
            int(
                config["maximum_lags_per_channel"]
                if history_override is None
                else history_override.dpls_maximum_lags_per_channel
            ),
        )
        estimator = _fit_pls_selected(
            x_fit,
            fit["y_true"].to_numpy(dtype=np.float64),
            int(components),
        )
        validation_accessor = BaseAccessor(
            shared, view.head.dataset, "validation", columns
        )
        prediction = _chunked_features_prediction(
            native_validation,
            lambda chunk: validation_accessor.input_lags(
                chunk,
                columns,
                int(history),
                int(
                    config["maximum_lags_per_channel"]
                    if history_override is None
                    else history_override.dpls_maximum_lags_per_channel
                ),
            ),
            lambda matrix: estimator.predict(matrix).reshape(-1),
        )
        parameter_count = int(components) * (x_fit.shape[1] + 2) + 1
        selection = {
            "selected_history": int(history),
            "selected_components": int(components),
            "history_override": None
            if history_override is None
            else history_override.audit(),
            "fold_losses": {str(key): value for key, value in losses.items()},
            "one_se": selected.__dict__,
            "native_support_audit": {
                **support_audit,
                "selected_requirement": requirement.to_json(),
                "anchor_train_rows": len(train),
                "selected_native_train_rows": len(native_train),
                "selected_fit_rows_after_cap": len(fit),
                "selected_native_train_support_hash": support_id_hash(native_train),
                "selected_fit_support_hash": support_id_hash(fit),
                "anchor_validation_rows": len(validation),
                "selected_native_validation_rows": len(native_validation),
                "selected_native_validation_support_hash": support_id_hash(
                    native_validation
                ),
            },
        }
        return _write_prediction_result(
            output=output,
            family="C2",
            stage="B1_PUBLIC_ALL_DPLS_DEVELOPMENT",
            model=model_name,
            view=view,
            samples=native_validation,
            prediction=prediction,
            selection=selection,
            parameter_count=parameter_count,
            started=started,
        )
    except Exception as error:
        return _failure_result(
            output=output,
            family="C2",
            stage="B1_PUBLIC_ALL_DPLS_DEVELOPMENT",
            model=model_name,
            view=view,
            started=started,
            error=error,
        )


def _profile_requirement(profile: tuple[int, int]) -> SupportRequirement:
    delta, history = (int(value) for value in profile)
    return SupportRequirement(
        target_delta_steps=delta,
        target_history_steps=history,
    )


def _arx_requirement(profile: tuple[int, int]) -> SupportRequirement:
    delta, history = (int(value) for value in profile)
    return SupportRequirement(
        input_history_steps=history,
        target_delta_steps=delta,
        target_history_steps=history,
    )


def run_ar_job(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    history_override_config: Path | str | None = None,
) -> dict[str, Any]:
    started = time.time()
    model_name = "AR"
    try:
        freeze = _freeze(project)
        config = freeze["c3"]["state_profile"]
        history_override = load_tep_history_override(history_override_config)
        if history_override is not None:
            history_override.require_view(view)
            if str(view.information_set) != "dynamic":
                raise RuntimeError("TEP AR history override requires dynamic")
        train = load_native_samples(shared, view, "train")
        validation = load_native_samples(shared, view, "validation")
        profiles = [
            tuple(profile)
            for profile in realized_state_profiles(
                view.head,
                positive_h_history_multipliers=None
                if history_override is None
                else history_override.positive_h_history_multipliers,
                delta_steps_override=None
                if history_override is None
                else history_override.state_delta_steps,
            )
        ]
        requirements = {profile: _profile_requirement(profile) for profile in profiles}
        fold_supports, support_audit = candidate_fold_supports(
            train,
            view,
            requirements.values(),
            fit_cap=int(freeze["selection"]["fit_row_cap_default"]),
            evaluation_cap=int(
                freeze["selection"]["selection_validation_row_cap_default"]
            ),
        )
        if history_override is not None:
            unavailable = {
                int(item["target_history_steps"])
                for item in support_audit["unavailable_requirements"]
                if int(item.get("target_history_steps", 0)) > 0
            }
            forbidden = unavailable.intersection(
                history_override.fail_if_history_unavailable
            )
            if forbidden:
                raise RuntimeError(
                    f"required TEP AR histories are unavailable: {sorted(forbidden)}"
                )
        profiles = [
            profile
            for profile in profiles
            if requirements[profile] in fold_supports
        ]
        requirements = {
            profile: requirements[profile] for profile in profiles
        }
        if not profiles:
            raise ValueError("no support-compatible AR profiles")
        accessor = BaseAccessor(
            shared, view.head.dataset, "train", [view.head.target]
        )
        profile_losses = {profile: [] for profile in profiles}
        for profile in profiles:
            requirement = requirements[profile]
            for fold in fold_supports[requirement]:
                x_fit = accessor.target_state(
                    fold.fit, view.head.target, *profile
                )
                x_evaluation = accessor.target_state(
                    fold.evaluation, view.head.target, *profile
                )
                prediction, _ = _ridge_block_predict(
                    x_fit,
                    fold.fit["y_true"].to_numpy(dtype=np.float64),
                    x_evaluation,
                    float(config["pilot_alpha"]),
                )
                profile_losses[profile].append(
                    mse(
                        fold.evaluation["y_true"].to_numpy(dtype=np.float64),
                        prediction,
                    )
                )
        selected_profile = select_one_se(
            profile_losses, lambda value: (value[1], -value[0])
        )
        profile = tuple(selected_profile.candidate)
        requirement = requirements[profile]
        penalty_losses = {
            float(alpha): []
            for alpha in config["penalty_scan_after_profile_freeze"]
        }
        for fold in fold_supports[requirement]:
            x_fit = accessor.target_state(fold.fit, view.head.target, *profile)
            x_evaluation = accessor.target_state(
                fold.evaluation, view.head.target, *profile
            )
            y_fit = fold.fit["y_true"].to_numpy(dtype=np.float64)
            y_evaluation = fold.evaluation["y_true"].to_numpy(dtype=np.float64)
            for alpha in penalty_losses:
                prediction, _ = _ridge_block_predict(
                    x_fit, y_fit, x_evaluation, float(alpha)
                )
                penalty_losses[alpha].append(mse(y_evaluation, prediction))
        selected_penalty = select_one_se(
            penalty_losses, lambda value: (-value,)
        )
        native_train = apply_requirement(train, requirement)
        fit = _cap_after_support(
            native_train,
            int(freeze["selection"]["fit_row_cap_default"]),
        )
        native_validation = apply_requirement(
            validation, requirement
        ).reset_index(drop=True)
        x_fit = accessor.target_state(fit, view.head.target, *profile)
        y_fit = fit["y_true"].to_numpy(dtype=np.float64)
        validation_accessor = BaseAccessor(
            shared, view.head.dataset, "validation", [view.head.target]
        )
        alpha = float(selected_penalty.candidate)
        prediction = _chunked_features_prediction(
            native_validation,
            lambda chunk: validation_accessor.target_state(
                chunk, view.head.target, *profile
            ),
            lambda matrix: _ridge_block_predict(
                x_fit, y_fit, matrix, alpha
            )[0],
        )
        _, certificate = _ridge_block_predict(x_fit, y_fit, x_fit[:1], alpha)
        selection = {
            "profile": selected_profile.__dict__,
            "penalty": selected_penalty.__dict__,
            "profile_fold_losses": {
                str(key): value for key, value in profile_losses.items()
            },
            "penalty_fold_losses": penalty_losses,
            "selected_profile": list(profile),
            "selected_alpha": alpha,
            "history_override": None
            if history_override is None
            else history_override.audit(),
            "numerical_certificate": certificate,
            "native_support_audit": {
                **support_audit,
                "selected_requirement": requirement.to_json(),
                "anchor_train_rows": len(train),
                "selected_native_train_rows": len(native_train),
                "selected_fit_rows_after_cap": len(fit),
                "selected_fit_support_hash": support_id_hash(fit),
                "anchor_validation_rows": len(validation),
                "selected_native_validation_rows": len(native_validation),
                "selected_native_validation_support_hash": support_id_hash(
                    native_validation
                ),
            },
        }
        return _write_prediction_result(
            output=output,
            family="C3",
            stage="B1_PUBLIC_ALL_AR_DEVELOPMENT",
            model=model_name,
            view=view,
            samples=native_validation,
            prediction=prediction,
            selection=selection,
            parameter_count=x_fit.shape[1] + 1,
            started=started,
        )
    except Exception as error:
        return _failure_result(
            output=output,
            family="C3",
            stage="B1_PUBLIC_ALL_AR_DEVELOPMENT",
            model=model_name,
            view=view,
            started=started,
            error=error,
        )


def _ar_selection(output: Path, view: ViewSpec) -> dict[str, Any]:
    path = _result_root(output, "C3", "AR", view) / "RESULT.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError(f"AR prerequisite is not PASS: {path}")
    return result["selection"]


def run_arx_job(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
) -> dict[str, Any]:
    started = time.time()
    model_name = "ARX"
    try:
        freeze = _freeze(project)
        config = freeze["c3"]["arx"]
        ar = _ar_selection(output, view)
        profile = tuple(int(value) for value in ar["selected_profile"])
        alpha = float(ar["selected_alpha"])
        requirement = _arx_requirement(profile)
        train = load_native_samples(shared, view, "train")
        validation = load_native_samples(shared, view, "validation")
        fold_supports, support_audit = candidate_fold_supports(
            train,
            view,
            [requirement],
            fit_cap=int(freeze["selection"]["fit_row_cap_default"]),
            evaluation_cap=int(
                freeze["selection"]["selection_validation_row_cap_default"]
            ),
        )
        columns = input_columns(shared, view.head.task_id, view.proxy_policy)
        all_columns = [view.head.target, *columns]
        accessor = BaseAccessor(shared, view.head.dataset, "train", all_columns)
        candidates: list[Any] = [
            "EXACT_X_ZERO",
            *[float(value) for value in config["block_penalty_ratios_x_over_ar"]],
        ]
        losses = {candidate: [] for candidate in candidates}
        for fold in fold_supports[requirement]:
            x_fit, ar_width = _arx_features(
                accessor,
                fold.fit,
                view,
                columns,
                profile,
                int(config["maximum_input_lags_per_channel"]),
            )
            x_evaluation, _ = _arx_features(
                accessor,
                fold.evaluation,
                view,
                columns,
                profile,
                int(config["maximum_input_lags_per_channel"]),
            )
            y_fit = fold.fit["y_true"].to_numpy(dtype=np.float64)
            y_evaluation = fold.evaluation["y_true"].to_numpy(dtype=np.float64)
            zero_prediction, _ = _ridge_block_predict(
                x_fit[:, :ar_width],
                y_fit,
                x_evaluation[:, :ar_width],
                alpha,
            )
            losses["EXACT_X_ZERO"].append(mse(y_evaluation, zero_prediction))
            for ratio in candidates[1:]:
                penalties = np.concatenate(
                    [
                        np.full(ar_width, alpha),
                        np.full(x_fit.shape[1] - ar_width, alpha * float(ratio)),
                    ]
                )
                prediction, _ = _ridge_block_predict(
                    x_fit, y_fit, x_evaluation, penalties
                )
                losses[ratio].append(mse(y_evaluation, prediction))
            release_process_memory()
        selected = select_one_se(
            losses,
            lambda value: (0,) if value == "EXACT_X_ZERO" else (1, -float(value)),
        )
        native_train = apply_requirement(train, requirement)
        fit = _cap_after_support(
            native_train,
            int(freeze["selection"]["fit_row_cap_default"]),
        )
        native_validation = apply_requirement(
            validation, requirement
        ).reset_index(drop=True)
        validation_accessor = BaseAccessor(
            shared, view.head.dataset, "validation", all_columns
        )
        x_fit, ar_width = _arx_features(
            accessor,
            fit,
            view,
            columns,
            profile,
            int(config["maximum_input_lags_per_channel"]),
        )
        y_fit = fit["y_true"].to_numpy(dtype=np.float64)
        if selected.candidate == "EXACT_X_ZERO":
            fit_matrix = x_fit[:, :ar_width]

            def features(chunk: pd.DataFrame) -> np.ndarray:
                matrix, _ = _arx_features(
                    validation_accessor,
                    chunk,
                    view,
                    columns,
                    profile,
                    int(config["maximum_input_lags_per_channel"]),
                )
                return matrix[:, :ar_width]

            penalties: np.ndarray | float = alpha
            parameter_count = ar_width + 1
        else:
            ratio = float(selected.candidate)
            fit_matrix = x_fit

            def features(chunk: pd.DataFrame) -> np.ndarray:
                return _arx_features(
                    validation_accessor,
                    chunk,
                    view,
                    columns,
                    profile,
                    int(config["maximum_input_lags_per_channel"]),
                )[0]

            penalties = np.concatenate(
                [
                    np.full(ar_width, alpha),
                    np.full(x_fit.shape[1] - ar_width, alpha * ratio),
                ]
            )
            parameter_count = x_fit.shape[1] + 1
        prediction = _chunked_features_prediction(
            native_validation,
            features,
            lambda matrix: _ridge_block_predict(
                fit_matrix, y_fit, matrix, penalties
            )[0],
        )
        _, certificate = _ridge_block_predict(
            fit_matrix, y_fit, fit_matrix[:1], penalties
        )
        selection = {
            "selected_profile": list(profile),
            "ar_alpha": alpha,
            "selected_x_penalty_ratio": selected.candidate,
            "fold_losses": {str(key): value for key, value in losses.items()},
            "one_se": selected.__dict__,
            "numerical_certificate": certificate,
            "native_support_audit": {
                **support_audit,
                "selected_requirement": requirement.to_json(),
                "anchor_train_rows": len(train),
                "selected_native_train_rows": len(native_train),
                "selected_fit_rows_after_cap": len(fit),
                "selected_fit_support_hash": support_id_hash(fit),
                "anchor_validation_rows": len(validation),
                "selected_native_validation_rows": len(native_validation),
                "selected_native_validation_support_hash": support_id_hash(
                    native_validation
                ),
            },
        }
        return _write_prediction_result(
            output=output,
            family="C3",
            stage="B1_PUBLIC_ALL_ARX_DEVELOPMENT",
            model=model_name,
            view=view,
            samples=native_validation,
            prediction=prediction,
            selection=selection,
            parameter_count=parameter_count,
            started=started,
        )
    except Exception as error:
        return _failure_result(
            output=output,
            family="C3",
            stage="B1_PUBLIC_ALL_ARX_DEVELOPMENT",
            model=model_name,
            view=view,
            started=started,
            error=error,
        )


def run_narx_job(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
) -> dict[str, Any]:
    started = time.time()
    model_name = "LINEAR_NARX"
    try:
        freeze = _freeze(project)
        config = freeze["c3"]["linear_narx"]
        arx_config = freeze["c3"]["arx"]
        state_config = freeze["c3"]["state_profile"]
        ar = _ar_selection(output, view)
        profile = tuple(int(value) for value in ar["selected_profile"])
        requirement = _arx_requirement(profile)
        train = load_native_samples(shared, view, "train")
        validation = load_native_samples(shared, view, "validation")
        fold_supports, support_audit = candidate_fold_supports(
            train,
            view,
            [requirement],
            fit_cap=int(freeze["selection"]["fit_row_cap_default"]),
            evaluation_cap=int(
                freeze["selection"]["selection_validation_row_cap_default"]
            ),
        )
        columns = input_columns(shared, view.head.task_id, view.proxy_policy)
        all_columns = [view.head.target, *columns]
        accessor = BaseAccessor(shared, view.head.dataset, "train", all_columns)
        alphas = [
            float(value)
            for value in state_config["penalty_scan_after_profile_freeze"]
        ]
        losses = {alpha: [] for alpha in alphas}
        for fold in fold_supports[requirement]:
            raw_fit, _ = _arx_features(
                accessor,
                fold.fit,
                view,
                columns,
                profile,
                int(arx_config["maximum_input_lags_per_channel"]),
            )
            raw_evaluation, _ = _arx_features(
                accessor,
                fold.evaluation,
                view,
                columns,
                profile,
                int(arx_config["maximum_input_lags_per_channel"]),
            )
            y_fit = fold.fit["y_true"].to_numpy(dtype=np.float64)
            y_evaluation = fold.evaluation["y_true"].to_numpy(dtype=np.float64)
            x_fit, x_evaluation, _ = _narx_expand(
                raw_fit,
                y_fit,
                raw_evaluation,
                int(config["maximum_linear_state_features_before_expansion"]),
            )
            for alpha in losses:
                prediction, _ = _ridge_block_predict(
                    x_fit, y_fit, x_evaluation, alpha
                )
                losses[alpha].append(mse(y_evaluation, prediction))
            release_process_memory()
        selected = select_one_se(losses, lambda value: (-value,))
        native_train = apply_requirement(train, requirement)
        fit = _cap_after_support(
            native_train,
            int(freeze["selection"]["fit_row_cap_default"]),
        )
        native_validation = apply_requirement(
            validation, requirement
        ).reset_index(drop=True)
        validation_accessor = BaseAccessor(
            shared, view.head.dataset, "validation", all_columns
        )
        raw_fit, _ = _arx_features(
            accessor,
            fit,
            view,
            columns,
            profile,
            int(arx_config["maximum_input_lags_per_channel"]),
        )
        y_fit = fit["y_true"].to_numpy(dtype=np.float64)
        x_fit, _, selected_features = _narx_expand(
            raw_fit,
            y_fit,
            raw_fit[:1],
            int(config["maximum_linear_state_features_before_expansion"]),
        )
        alpha = float(selected.candidate)

        def predict_chunk(chunk: pd.DataFrame) -> np.ndarray:
            raw_evaluation, _ = _arx_features(
                validation_accessor,
                chunk,
                view,
                columns,
                profile,
                int(arx_config["maximum_input_lags_per_channel"]),
            )
            chunk_fit, chunk_evaluation, chunk_features = _narx_expand(
                raw_fit,
                y_fit,
                raw_evaluation,
                int(config["maximum_linear_state_features_before_expansion"]),
            )
            if not np.array_equal(chunk_features, selected_features):
                raise RuntimeError("NARX train-only feature selection drifted")
            return _ridge_block_predict(
                chunk_fit, y_fit, chunk_evaluation, alpha
            )[0]

        prediction = np.empty(len(native_validation), dtype=np.float64)
        for start_index in range(0, len(native_validation), 50_000):
            stop_index = min(len(native_validation), start_index + 50_000)
            prediction[start_index:stop_index] = predict_chunk(
                native_validation.iloc[start_index:stop_index]
            )
        _, certificate = _ridge_block_predict(
            x_fit, y_fit, x_fit[:1], alpha
        )
        selection = {
            "selected_profile": list(profile),
            "selected_alpha": alpha,
            "selected_linear_feature_indices": selected_features.tolist(),
            "fold_losses": losses,
            "one_se": selected.__dict__,
            "numerical_certificate": certificate,
            "native_support_audit": {
                **support_audit,
                "selected_requirement": requirement.to_json(),
                "anchor_train_rows": len(train),
                "selected_native_train_rows": len(native_train),
                "selected_fit_rows_after_cap": len(fit),
                "selected_fit_support_hash": support_id_hash(fit),
                "anchor_validation_rows": len(validation),
                "selected_native_validation_rows": len(native_validation),
                "selected_native_validation_support_hash": support_id_hash(
                    native_validation
                ),
            },
        }
        return _write_prediction_result(
            output=output,
            family="C3",
            stage="B1_PUBLIC_ALL_NARX_DEVELOPMENT",
            model=model_name,
            view=view,
            samples=native_validation,
            prediction=prediction,
            selection=selection,
            parameter_count=x_fit.shape[1] + 1,
            started=started,
        )
    except Exception as error:
        return _failure_result(
            output=output,
            family="C3",
            stage="B1_PUBLIC_ALL_NARX_DEVELOPMENT",
            model=model_name,
            view=view,
            started=started,
            error=error,
        )


def run_hammerstein_job(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    model_name: str,
    history_override_config: Path | str | None = None,
) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression

    started = time.time()
    try:
        wiener = model_name == "HAMMERSTEIN_WIENER"
        freeze = _freeze(project)
        section = "hammerstein_wiener" if wiener else "parallel_hammerstein"
        config = freeze["c3"][section]
        history_override = load_tep_history_override(history_override_config)
        if history_override is not None:
            history_override.require_view(view)
            if str(view.information_set) != "input_only":
                raise RuntimeError(
                    "TEP Hammerstein history override requires input_only"
                )
        train = load_native_samples(shared, view, "train")
        validation = load_native_samples(shared, view, "validation")
        columns = input_columns(shared, view.head.task_id, view.proxy_policy)
        profiles = [
            tuple(profile)
            for profile in _hammerstein_profiles(
                view,
                positive_h_history_multipliers=None
                if history_override is None
                else history_override.positive_h_history_multipliers,
                delta_steps_override=None
                if history_override is None
                else history_override.state_delta_steps,
                profile_cap=6
                if history_override is None
                else history_override.hammerstein_profile_cap,
                require_every_registered_history=history_override is not None,
            )
        ]
        if history_override is not None and {
            int(profile[1]) for profile in profiles
        } != set(history_override.history_steps):
            raise RuntimeError(
                "TEP Hammerstein deterministic profile cap did not cover every history"
            )
        requirements = {
            profile: SupportRequirement(input_history_steps=int(profile[1]))
            for profile in profiles
        }
        fold_supports, support_audit = candidate_fold_supports(
            train,
            view,
            requirements.values(),
            fit_cap=int(freeze["selection"]["fit_row_cap_default"]),
            evaluation_cap=int(
                freeze["selection"]["selection_validation_row_cap_default"]
            ),
        )
        if history_override is not None:
            unavailable = {
                int(item["input_history_steps"])
                for item in support_audit["unavailable_requirements"]
                if int(item.get("input_history_steps", 0)) > 0
            }
            forbidden = unavailable.intersection(
                history_override.fail_if_history_unavailable
            )
            if forbidden:
                raise RuntimeError(
                    "required TEP Hammerstein histories are unavailable: "
                    f"{sorted(forbidden)}"
                )
        profiles = [
            profile
            for profile in profiles
            if requirements[profile] in fold_supports
        ]
        requirements = {
            profile: requirements[profile] for profile in profiles
        }
        if not profiles:
            raise ValueError("no support-compatible Hammerstein profiles")
        if wiener:
            candidates = [
                (profile, nonlinearity, output_map)
                for profile in profiles
                for nonlinearity in config["input_nonlinearities"]
                for output_map in config["output_map"]
            ]
        else:
            candidates = [
                (profile, nonlinearity, "identity")
                for profile in profiles
                for nonlinearity in config["nonlinearities"]
            ]
        if (
            history_override is not None
            and len(candidates)
            != history_override.hammerstein_profile_cap
            * (
                len(config["input_nonlinearities"])
                * len(config["output_map"])
                if wiener
                else len(config["nonlinearities"])
            )
        ):
            raise RuntimeError(
                "TEP Hammerstein extension candidate grid size drifted: "
                f"joint configurations, got {len(candidates)}"
            )
        losses = {candidate: [] for candidate in candidates}
        accessor = BaseAccessor(shared, view.head.dataset, "train", columns)
        for profile in profiles:
            requirement = requirements[profile]
            delta, history = profile
            for fold in fold_supports[requirement]:
                raw_fit = accessor.input_regular_lags(
                    fold.fit, columns, delta, history, 8
                )
                raw_evaluation = accessor.input_regular_lags(
                    fold.evaluation, columns, delta, history, 8
                )
                y_fit = fold.fit["y_true"].to_numpy(dtype=np.float64)
                y_evaluation = fold.evaluation["y_true"].to_numpy(dtype=np.float64)
                nonlinearities = (
                    config["input_nonlinearities"]
                    if wiener
                    else config["nonlinearities"]
                )
                output_maps = config["output_map"] if wiener else ["identity"]
                for nonlinearity in nonlinearities:
                    x_fit, x_evaluation = _nonlinear_features(
                        raw_fit, raw_evaluation, nonlinearity
                    )
                    prediction, _ = _ridge_block_predict(
                        x_fit, y_fit, x_evaluation, 0.001
                    )
                    train_prediction = None
                    for output_map in output_maps:
                        current = prediction
                        if output_map == "isotonic_train_only":
                            if train_prediction is None:
                                train_prediction, _ = _ridge_block_predict(
                                    x_fit, y_fit, x_fit, 0.001
                                )
                            current = IsotonicRegression(
                                out_of_bounds="clip"
                            ).fit(train_prediction, y_fit).predict(prediction)
                        losses[(profile, nonlinearity, output_map)].append(
                            mse(y_evaluation, current)
                        )
                release_process_memory()
        selected = select_one_se(
            losses,
            lambda value: (
                value[0][1],
                -value[0][0],
                0 if value[1] == "linear" else 1,
                0 if value[2] == "identity" else 1,
            ),
        )
        profile, nonlinearity, output_map = selected.candidate
        requirement = requirements[profile]
        native_train = apply_requirement(train, requirement)
        fit = _cap_after_support(
            native_train,
            int(freeze["selection"]["fit_row_cap_default"]),
        )
        native_validation = apply_requirement(
            validation, requirement
        ).reset_index(drop=True)
        raw_fit = accessor.input_regular_lags(fit, columns, *profile, 8)
        y_fit = fit["y_true"].to_numpy(dtype=np.float64)
        x_fit, _ = _nonlinear_features(raw_fit, raw_fit[:1], nonlinearity)
        train_prediction = None
        calibrator = None
        if output_map == "isotonic_train_only":
            train_prediction, _ = _ridge_block_predict(
                x_fit, y_fit, x_fit, 0.001
            )
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                train_prediction, y_fit
            )
        validation_accessor = BaseAccessor(
            shared, view.head.dataset, "validation", columns
        )

        def predict_chunk(chunk: pd.DataFrame) -> np.ndarray:
            raw_evaluation = validation_accessor.input_regular_lags(
                chunk, columns, *profile, 8
            )
            chunk_fit, chunk_evaluation = _nonlinear_features(
                raw_fit, raw_evaluation, nonlinearity
            )
            current = _ridge_block_predict(
                chunk_fit, y_fit, chunk_evaluation, 0.001
            )[0]
            if calibrator is not None:
                current = calibrator.predict(current)
            return np.asarray(current, dtype=np.float64)

        prediction = np.empty(len(native_validation), dtype=np.float64)
        for start_index in range(0, len(native_validation), 50_000):
            stop_index = min(len(native_validation), start_index + 50_000)
            prediction[start_index:stop_index] = predict_chunk(
                native_validation.iloc[start_index:stop_index]
            )
        _, certificate = _ridge_block_predict(
            x_fit, y_fit, x_fit[:1], 0.001
        )
        selection = {
            "selected_profile": list(profile),
            "selected_input_nonlinearity": nonlinearity,
            "selected_output_map": output_map,
            "history_override": None
            if history_override is None
            else history_override.audit(),
            "registered_profiles": [list(value) for value in profiles],
            "fold_losses": {str(key): value for key, value in losses.items()},
            "one_se": selected.__dict__,
            "numerical_certificate": certificate,
            "native_support_audit": {
                **support_audit,
                "selected_requirement": requirement.to_json(),
                "anchor_train_rows": len(train),
                "selected_native_train_rows": len(native_train),
                "selected_fit_rows_after_cap": len(fit),
                "selected_fit_support_hash": support_id_hash(fit),
                "anchor_validation_rows": len(validation),
                "selected_native_validation_rows": len(native_validation),
                "selected_native_validation_support_hash": support_id_hash(
                    native_validation
                ),
            },
        }
        return _write_prediction_result(
            output=output,
            family="C3",
            stage="B1_PUBLIC_ALL_HAMMERSTEIN_DEVELOPMENT",
            model=model_name,
            view=view,
            samples=native_validation,
            prediction=prediction,
            selection=selection,
            parameter_count=x_fit.shape[1] + 1,
            started=started,
        )
    except Exception as error:
        return _failure_result(
            output=output,
            family="C3",
            stage="B1_PUBLIC_ALL_HAMMERSTEIN_DEVELOPMENT",
            model=model_name,
            view=view,
            started=started,
            error=error,
        )


def _not_run_result(
    *,
    output: Path,
    family: str,
    stage: str,
    model: str,
    view: ViewSpec,
    reason: str,
    started: float,
) -> dict[str, Any]:
    destination = _result_root(output, family, model, view)
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "NOT_RUN_PROTOCOL_INCOMPATIBLE",
        "stage": stage,
        "support_contract": SUPPORT_CONTRACT,
        "model": model,
        "dataset": view.head.dataset,
        "task": view.head.task_id,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "reason": reason,
        "test_accessed": False,
        "ood_accessed": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(destination / "RESULT.json", result)
    return result


def _has_contiguous_entity_sequence(frame: pd.DataFrame) -> bool:
    """Return whether a baseline can use one uninterrupted entity sequence."""
    if frame.empty or frame["entity_id"].nunique() != 1:
        return False
    rows = np.sort(frame["row_in_entity"].to_numpy(dtype=np.int64))
    return np.array_equal(rows, np.arange(int(rows[0]), int(rows[-1]) + 1))


def run_n4sid_job(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
) -> dict[str, Any]:
    started = time.time()
    model = "N4SID"
    train = load_native_samples(shared, view, "train")
    validation = load_native_samples(shared, view, "validation")
    if train["entity_id"].nunique() != 1:
        return _not_run_result(
            output=output,
            family="C3",
            stage="B1_PUBLIC_ALL_N4SID_DEVELOPMENT",
            model=model,
            view=view,
            reason="EXPLICIT_GROUPED_STATE_RESET_IMPLEMENTATION_ABSENT",
            started=started,
        )
    for samples in (train, validation):
        if not np.array_equal(
            samples["latest_available_target_index"].to_numpy(dtype=np.int64),
            samples["origin"].to_numpy(dtype=np.int64) - 1,
        ):
            return _not_run_result(
                output=output,
                family="C3",
                stage="B1_PUBLIC_ALL_N4SID_DEVELOPMENT",
                model=model,
                view=view,
                reason="REGISTERED_TARGET_AVAILABILITY_DELAY_NOT_SUPPORTED",
                started=started,
            )
    train_sequence = pd.read_parquet(
        shared / "base_data" / view.head.dataset / "train.parquet",
        columns=["entity_id", "row_in_entity"],
    )
    if not _has_contiguous_entity_sequence(train_sequence):
        reason = (
            "EXPLICIT_GROUPED_STATE_RESET_IMPLEMENTATION_ABSENT"
            if train_sequence["entity_id"].nunique() != 1
            else "CONTIGUOUS_SEQUENCE_REQUIRED_GAPS_PRESENT"
        )
        return _not_run_result(
            output=output,
            family="C3",
            stage="B1_PUBLIC_ALL_N4SID_DEVELOPMENT",
            model=model,
            view=view,
            reason=reason,
            started=started,
        )
    result = run_legacy_c3_job(
        shared, project, output / "C3", view, model
    )
    path = _result_root(output, "C3", model, view) / "RESULT.json"
    if result.get("status") != "PASS":
        return result
    selection = dict(result["selection"])
    selection["native_support_audit"] = {
        "support_contract": SUPPORT_CONTRACT,
        "candidate_orders_share_identical_sequences": True,
        "anchor_train_rows": len(train),
        "anchor_validation_rows": len(validation),
        "train_support_hash": support_id_hash(train),
        "validation_support_hash": support_id_hash(validation),
        "row_cap_applied": False,
        "future_input_policy": selection.get("future_input_policy"),
        "target_availability_is_record_time": True,
    }
    result["selection"] = selection
    result["support_contract"] = SUPPORT_CONTRACT
    result["ood_accessed"] = False
    write_json(path, result)
    return result


def _development_summary_status(results: Sequence[Mapping[str, Any]]) -> str:
    statuses = [result.get("status") for result in results]
    if any(status not in COMPLETED_STATUSES for status in statuses):
        return "FAILED"
    if any(status == "FAILED_RETAINED" for status in statuses):
        return "PASS_WITH_RETAINED_FAILURES"
    if all(status in SUCCESSFUL_COMPLETION_STATUSES for status in statuses):
        return "PASS"
    return "FAILED"


def _run_resumable(
    function: Callable[..., dict[str, Any]],
    jobs: list[tuple[Path, tuple[Any, ...]]],
    workers: int,
    *,
    per_worker_gib: float,
    label: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pending: list[tuple[Any, ...]] = []
    for result_path, arguments in jobs:
        previous = _completed(result_path)
        if previous is None:
            pending.append(arguments)
        else:
            results.append(previous)
    if pending:
        results.extend(
            run_parallel(
                function,
                pending,
                max(1, min(int(workers), len(pending))),
                per_worker_gib=float(per_worker_gib),
                label=label,
            )
        )
    return results


def _jobs(
    root: Path,
    family: str,
    model: str,
    views: Iterable[ViewSpec],
    arguments: Callable[[ViewSpec], tuple[Any, ...]],
) -> list[tuple[Path, tuple[Any, ...]]]:
    return [
        (
            _result_root(root, family, model, view) / "RESULT.json",
            arguments(view),
        )
        for view in views
    ]


def run_public_all_baseline_development(
    shared: Path,
    project: Path,
    output: Path,
    workers: int,
    *,
    input_views_override: Sequence[ViewSpec] | None = None,
    dynamic_views_override: Sequence[ViewSpec] | None = None,
    stage_name: str = "B1_PUBLIC_ALL_CPU_BASELINE_DEVELOPMENT",
) -> dict[str, Any]:
    root = output / "BASELINE_DEVELOPMENT"
    root.mkdir(parents=True, exist_ok=True)
    input_views = (
        public_all_input_views(shared)
        if input_views_override is None
        else list(input_views_override)
    )
    dynamic_views = (
        public_all_dynamic_views(shared)
        if dynamic_views_override is None
        else list(dynamic_views_override)
    )
    if not input_views or not dynamic_views:
        raise RuntimeError("baseline development requires non-empty registered views")
    all_views = [*input_views, *dynamic_views]
    results: list[dict[str, Any]] = []

    for model in SIMPLE_MODELS:
        results.extend(
            _run_resumable(
                run_simple_job,
                _jobs(
                    root,
                    "C2",
                    model,
                    all_views,
                    lambda view, selected=model: (
                        shared,
                        project,
                        root,
                        view,
                        selected,
                    ),
                ),
                min(workers, 6),
                per_worker_gib=1.0,
                label=f"PUBLIC_ALL_BASELINE:{model}",
            )
        )

    for model in STATIC_INPUT_MODELS:
        results.extend(
            _run_resumable(
                run_static_input_job,
                _jobs(
                    root,
                    "C2",
                    model,
                    input_views,
                    lambda view, selected=model: (
                        shared,
                        project,
                        root,
                        view,
                        selected,
                    ),
                ),
                min(workers, 4),
                per_worker_gib=3.0,
                label=f"PUBLIC_ALL_BASELINE:{model}",
            )
        )

    results.extend(
        _run_resumable(
            run_dpls_job,
            _jobs(
                root,
                "C2",
                "DPLS",
                input_views,
                lambda view: (shared, project, root, view),
            ),
            min(workers, 3),
            per_worker_gib=5.0,
            label="PUBLIC_ALL_BASELINE:DPLS",
        )
    )

    for model in ("PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"):
        for dataset in sorted({view.head.dataset for view in input_views}):
            dataset_views = [
                view for view in input_views if view.head.dataset == dataset
            ]
            results.extend(
                _run_resumable(
                    run_hammerstein_job,
                    _jobs(
                        root,
                        "C3",
                        model,
                        dataset_views,
                        lambda view, selected=model: (
                            shared,
                            project,
                            root,
                            view,
                            selected,
                        ),
                    ),
                    min(workers, 2),
                    per_worker_gib=5.0,
                    label=f"PUBLIC_ALL_BASELINE:{model}:{dataset}",
                )
            )

    results.extend(
        _run_resumable(
            run_ar_job,
            _jobs(
                root,
                "C3",
                "AR",
                dynamic_views,
                lambda view: (shared, project, root, view),
            ),
            min(workers, 4),
            per_worker_gib=3.0,
            label="PUBLIC_ALL_BASELINE:AR",
        )
    )
    for model, function in (
        ("ARX", run_arx_job),
        ("LINEAR_NARX", run_narx_job),
        ("N4SID", run_n4sid_job),
    ):
        results.extend(
            _run_resumable(
                function,
                _jobs(
                    root,
                    "C3",
                    model,
                    dynamic_views,
                    lambda view: (shared, project, root, view),
                ),
                min(workers, 3),
                per_worker_gib=4.0,
                label=f"PUBLIC_ALL_BASELINE:{model}",
            )
        )

    status_counts = pd.Series(
        [str(result.get("status")) for result in results], dtype="string"
    ).value_counts()
    flat = [
        {
            key: value
            for key, value in result.items()
            if key not in {"selection", "traceback"}
        }
        for result in results
    ]
    pd.DataFrame(flat).sort_values(
        ["target_head", "information_set", "availability_scenario", "model"],
        na_position="last",
    ).to_csv(root / "PUBLIC_ALL_BASELINE_DEVELOPMENT.csv", index=False)
    summary = {
        "status": _development_summary_status(results),
        "stage": stage_name,
        "support_contract": SUPPORT_CONTRACT,
        "jobs": len(results),
        "status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "candidate_native_fitting": True,
        "common_validation_scoring": True,
        "test_accessed": False,
        "ood_accessed": False,
        "cpu_model_freeze_sha256": sha256_file(
            project / "configs/cpu_model_freeze_v1.json"
        ),
    }
    write_json(root / "SUMMARY.json", summary)
    return summary
