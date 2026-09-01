from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .c4_prism import (
    _fit_candidate,
    _numerical_status,
    channel_class,
    channel_profiles,
    profile_values,
)
from .cpu_data import (
    BaseAccessor,
    ViewSpec,
    deterministic_subsample,
    inner_folds,
    input_columns,
    main_views,
    sha256_file,
)
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v211_support import (
    SUPPORT_CONTRACT,
    apply_native_support,
    load_native_samples,
    registered_fold_native_masks,
    support_audit,
    support_id_hash,
)
from .v2_selection import one_se_select, practical_activation


PRIMARY_HEAD_ID = "PMSM_SW__H600__W60"
EXPECTED_INPUTS = [
    "ambient",
    "coolant",
    "u_d",
    "u_q",
    "i_d",
    "i_q",
    "motor_speed",
    "torque",
]
D_LADDER = [
    "exact_zero",
    "linear",
    "rank_1",
    "rank_2",
    "rank_3",
    "rank_4",
    "full",
]
IMPLEMENTATION_FREEZE = "configs/prism_v22_pmsm_sw_implementation_semantics_freeze_20260901.json"
CPU_FREEZE = "configs/cpu_model_freeze_v1.json"
SPLIT_REGISTRY = "dataset_registry/pmsm/SPLIT_REGISTRY.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pilot_tuple(config: dict[str, Any]) -> tuple[float, float, float]:
    pilot = config["penalties"]["pilot"]
    return (
        float(pilot["lambda_0"]),
        float(pilot["lambda_tau"]),
        float(pilot["lambda_x"]),
    )


def assert_freeze_consistency(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail before target fitting if the preregistered D mapping drifts from C4."""
    implementation = _json(project / IMPLEMENTATION_FREEZE)
    cpu = _json(project / CPU_FREEZE)
    c4 = cpu["c4"]
    d = implementation["D_branch"]
    selection = implementation["selection_partition"]

    if implementation["status"] != "PRE_MODEL_FIT_IMPLEMENTATION_SEMANTICS_FROZEN":
        raise RuntimeError("PMSM SW implementation freeze status drift")
    if implementation["model_target_results_seen_before_this_freeze"] is not False:
        raise RuntimeError("implementation freeze was not pre-result")
    if implementation["test_target_access_before_this_freeze"] is not False:
        raise RuntimeError("test target was accessed before implementation freeze")
    if d["inherit_source"] != "configs/cpu_model_freeze_v1.json::c4":
        raise RuntimeError("D inheritance source drift")
    if d["complexity_ladder"] != c4["complexity_ladder"] or d["complexity_ladder"] != D_LADDER:
        raise RuntimeError("D complexity ladder drift")
    if d["delta_ratio_by_class"] != c4["delta_ratio_by_class"]:
        raise RuntimeError("D delta grid drift")
    if d["history_for_positive_h"] != c4["history_for_positive_h"]:
        raise RuntimeError("D positive-h history grid drift")
    if d["lag_basis"]["candidate_m_tau"] != c4["lag_basis"]["candidate_m_tau"]:
        raise RuntimeError("D m_tau grid drift")
    if d["amplitude_basis"]["candidate_m_x"] != c4["amplitude_basis"]["candidate_m_x"]:
        raise RuntimeError("D m_x grid drift")
    if d["rank_candidates"] != c4["rank_candidates"]:
        raise RuntimeError("D rank grid drift")
    for name in ("lambda_0", "lambda_tau", "lambda_x"):
        if d["penalties"][name] != c4["penalties"][name]:
            raise RuntimeError(f"D {name} grid drift")
    if [float(value) for value in d["penalties"]["pilot"]] != list(_pilot_tuple(c4)):
        raise RuntimeError("D pilot penalty drift")
    if d["selection_order"] != c4["selection_order"]:
        raise RuntimeError("D selection order drift")
    if int(selection["inner_fold_count"]) != 4:
        raise RuntimeError("D inner fold count drift")
    if int(selection["minimum_usable_folds"]) != 3:
        raise RuntimeError("D minimum usable folds drift")
    if float(selection["minimum_relative_improvement"]) != 0.01:
        raise RuntimeError("D activation threshold drift")
    if float(selection["minimum_positive_fold_fraction"]) != 0.75:
        raise RuntimeError("D positive-fold threshold drift")
    if int(selection["row_caps"]["D_C4_fit_row_cap"]) != int(c4["fit_row_cap"]):
        raise RuntimeError("D fit cap drift")
    if int(selection["row_caps"]["D_C4_selection_validation_row_cap"]) != int(
        c4["selection_validation_row_cap"]
    ):
        raise RuntimeError("D selection validation cap drift")
    if implementation["sample_support"]["contract"] != SUPPORT_CONTRACT:
        raise RuntimeError("D support contract drift")
    if d["channel_classes"] != {channel: channel_class("pmsm", channel) for channel in EXPECTED_INPUTS}:
        raise RuntimeError("D channel class mapping drift")
    return implementation, c4


def assert_prelockbox(shared: Path, project: Path) -> dict[str, Any]:
    lockbox = _json(shared / "LOCKBOX.json")
    if lockbox.get("test_profiles_physically_absent") is not True:
        raise RuntimeError("PMSM SW test profiles are not physically locked out")
    if lockbox.get("test_target_rows_materialized") is not False:
        raise RuntimeError("PMSM SW test target rows were materialized")
    if (shared / "base_data/pmsm/test.parquet").exists():
        raise RuntimeError("test base-data partition exists in development shared data")
    if list((shared / "sample_ids").rglob("test.parquet")):
        raise RuntimeError("test sample-id partition exists in development shared data")

    views = _json(shared / "dataset_views/VIEW_REGISTRY.json")
    relevant = [
        item
        for item in views
        if item.get("task_id") == "PMSM_SW" and item.get("proxy_policy") == "proxy_excluded"
    ]
    if len(relevant) != 1 or relevant[0].get("input_columns") != EXPECTED_INPUTS:
        raise RuntimeError("PMSM SW primary input contract drift")

    split = _json(project / SPLIT_REGISTRY)
    test_profiles = {str(int(value)) for value in split["test_profile_ids"]}
    for base_split in ("train", "validation"):
        path = shared / "base_data/pmsm" / f"{base_split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        entities = set(pd.read_parquet(path, columns=["entity_id"])["entity_id"].astype(str).unique())
        leaked = entities & test_profiles
        if leaked:
            raise RuntimeError(f"test profiles leaked into {base_split}: {sorted(leaked)}")
    return lockbox


def primary_view(shared: Path) -> ViewSpec:
    views = main_views(shared, "dynamic")
    matches = [view for view in views if view.head.head_id == PRIMARY_HEAD_ID]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one primary PMSM SW view, got {matches}")
    view = matches[0]
    if view.head.dataset != "pmsm" or view.head.task_id != "PMSM_SW":
        raise RuntimeError("primary PMSM SW view identity drift")
    if view.proxy_policy != "proxy_excluded" or view.availability_scenario != "record_time":
        raise RuntimeError("primary PMSM SW view contract drift")
    if view.head.h_steps != 600 or view.head.w_steps != 60 or view.head.cadence_seconds != 0.5:
        raise RuntimeError("primary PMSM SW time contract drift")
    return view


def _complexity_profile(profile: tuple[int, int]) -> tuple[int, int]:
    return int(profile[1]), -int(profile[0])


def _loss_json(losses: dict[Any, list[float]]) -> dict[str, list[float]]:
    return {str(key): [float(value) for value in values] for key, values in losses.items()}


def _candidate_losses(
    *,
    accessor: BaseAccessor,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    channel: str,
    profile: tuple[int, int],
    m_tau: int,
    kind: str,
    m_x: int,
    lambdas: tuple[float, float, float],
    c4: dict[str, Any],
    scoring_history_steps: int,
) -> list[float]:
    records = registered_fold_native_masks(
        train,
        folds,
        fit_history_steps=int(profile[1]),
        scoring_history_steps=int(scoring_history_steps),
        fit_cap=int(c4["fit_row_cap"]),
        evaluation_cap=int(c4["selection_validation_row_cap"]),
    )
    losses: list[float] = []
    for record in records:
        fit = record["fit"]
        evaluation = record["evaluation"]
        try:
            if kind == "exact_zero":
                prediction = np.zeros(len(evaluation), dtype=np.float64)
            else:
                fit_values, _ = profile_values(accessor, fit, channel, profile, int(m_tau))
                evaluation_values, _ = profile_values(
                    accessor, evaluation, channel, profile, int(m_tau)
                )
                prediction, _, _ = _fit_candidate(
                    fit_values,
                    fit["y_true"].to_numpy(dtype=np.float64),
                    evaluation_values,
                    kind,
                    int(m_x),
                    lambdas,
                    c4["solver"],
                )
            losses.append(
                mse(evaluation["y_true"].to_numpy(dtype=np.float64), prediction)
            )
        except Exception:
            losses.append(float("inf"))
    return losses


def _zero_losses(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    scoring_history_steps: int,
    c4: dict[str, Any],
) -> tuple[list[float], list[str], list[int]]:
    records = registered_fold_native_masks(
        train,
        folds,
        fit_history_steps=int(scoring_history_steps),
        scoring_history_steps=int(scoring_history_steps),
        fit_cap=int(c4["fit_row_cap"]),
        evaluation_cap=int(c4["selection_validation_row_cap"]),
    )
    losses = []
    for record in records:
        target = record["evaluation"]["y_true"].to_numpy(dtype=np.float64)
        losses.append(float(np.mean(np.square(target), dtype=np.float64)))
    return (
        losses,
        [support_id_hash(record["evaluation"]) for record in records],
        [int(len(record["evaluation"])) for record in records],
    )


def run_d_channel(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    channel: str,
) -> dict[str, Any]:
    """Run preregistered D-only selection using train only; evaluate validation once frozen."""
    started = time.time()
    destination = output / "D_ONLY" / view.head.head_id / view.proxy_policy / channel
    destination.mkdir(parents=True, exist_ok=True)
    try:
        implementation, c4 = assert_freeze_consistency(project)
        assert_prelockbox(shared, project)
        registered_inputs = input_columns(shared, view.head.task_id, view.proxy_policy)
        if registered_inputs != EXPECTED_INPUTS or channel not in registered_inputs:
            raise RuntimeError("requested D channel is outside preregistered input contract")

        # Selection boundary: only train samples and a train-only accessor exist here.
        train = load_native_samples(shared, view, "train")
        train_accessor = BaseAccessor(shared, view.head.dataset, "train", [channel])
        folds = inner_folds(train, int(implementation["selection_partition"]["inner_fold_count"]))
        profiles = channel_profiles(view, channel, c4)
        if not profiles:
            raise RuntimeError("empty preregistered D profile universe")
        profile_comparison_history = max(int(profile[1]) for profile in profiles)
        pilot_lambdas = _pilot_tuple(c4)
        pilot_m_tau = 8

        profile_losses = {
            profile: _candidate_losses(
                accessor=train_accessor,
                train=train,
                folds=folds,
                channel=channel,
                profile=profile,
                m_tau=pilot_m_tau,
                kind="linear",
                m_x=1,
                lambdas=pilot_lambdas,
                c4=c4,
                scoring_history_steps=profile_comparison_history,
            )
            for profile in profiles
        }
        profile_selection = one_se_select(
            profile_losses,
            _complexity_profile,
            minimum_usable_folds=int(implementation["selection_partition"]["minimum_usable_folds"]),
        )
        selected_profile = tuple(profile_selection.selected)
        retained_profiles = sorted(profile_selection.acceptable, key=_complexity_profile)[
            : int(c4["channel_candidate_gate"]["maximum_profiles_retained_per_channel"])
        ]
        local_history = int(selected_profile[1])

        # Re-score selected linear and exact-zero on identical selected-profile support.
        linear_activation_losses = _candidate_losses(
            accessor=train_accessor,
            train=train,
            folds=folds,
            channel=channel,
            profile=selected_profile,
            m_tau=pilot_m_tau,
            kind="linear",
            m_x=1,
            lambdas=pilot_lambdas,
            c4=c4,
            scoring_history_steps=local_history,
        )
        zero_losses, local_scoring_hashes, local_scoring_rows = _zero_losses(
            train, folds, scoring_history_steps=local_history, c4=c4
        )
        activation_one_se = one_se_select(
            {"exact_zero": zero_losses, "linear": linear_activation_losses},
            lambda value: (0,) if value == "exact_zero" else (1,),
            neutral="exact_zero",
            minimum_usable_folds=int(implementation["selection_partition"]["minimum_usable_folds"]),
        )
        activation = practical_activation(
            zero_losses,
            linear_activation_losses,
            minimum_relative_improvement=float(
                implementation["selection_partition"]["minimum_relative_improvement"]
            ),
            minimum_positive_fraction=float(
                implementation["selection_partition"]["minimum_positive_fold_fraction"]
            ),
        )

        resolution_selection = None
        ladder_selection = None
        penalty_audit: dict[str, Any] = {}
        ladder_losses: dict[Any, list[float]] = {"exact_zero": zero_losses}
        if activation_one_se.selected == "exact_zero" or not activation["pass"]:
            selected_kind = "exact_zero"
            selected_m_tau = pilot_m_tau
            selected_m_x = 1
            selected_lambdas = pilot_lambdas
            final_activation = activation
        else:
            resolution_losses: dict[Any, list[float]] = {}
            for m_tau in c4["lag_basis"]["candidate_m_tau"]:
                for m_x in c4["amplitude_basis"]["candidate_m_x"]:
                    resolution_losses[(int(m_tau), int(m_x))] = _candidate_losses(
                        accessor=train_accessor,
                        train=train,
                        folds=folds,
                        channel=channel,
                        profile=selected_profile,
                        m_tau=int(m_tau),
                        kind="full",
                        m_x=int(m_x),
                        lambdas=pilot_lambdas,
                        c4=c4,
                        scoring_history_steps=local_history,
                    )
            resolution_selection = one_se_select(
                resolution_losses,
                lambda value: (int(value[0]), int(value[1])),
                minimum_usable_folds=int(implementation["selection_partition"]["minimum_usable_folds"]),
            )
            selected_m_tau, selected_m_x = map(int, resolution_selection.selected)

            for kind in D_LADDER[1:]:
                ladder_losses[kind] = _candidate_losses(
                    accessor=train_accessor,
                    train=train,
                    folds=folds,
                    channel=channel,
                    profile=selected_profile,
                    m_tau=selected_m_tau,
                    kind=kind,
                    m_x=selected_m_x,
                    lambdas=pilot_lambdas,
                    c4=c4,
                    scoring_history_steps=local_history,
                )
            order = {name: index for index, name in enumerate(D_LADDER)}
            ladder_selection = one_se_select(
                ladder_losses,
                lambda value: (order[str(value)],),
                neutral="exact_zero",
                minimum_usable_folds=int(implementation["selection_partition"]["minimum_usable_folds"]),
            )
            selected_kind = str(ladder_selection.selected)
            if selected_kind == "exact_zero":
                selected_lambdas = pilot_lambdas
                final_activation = practical_activation(
                    zero_losses,
                    zero_losses,
                    minimum_relative_improvement=float(
                        implementation["selection_partition"]["minimum_relative_improvement"]
                    ),
                    minimum_positive_fraction=float(
                        implementation["selection_partition"]["minimum_positive_fold_fraction"]
                    ),
                )
            else:
                final_activation = practical_activation(
                    zero_losses,
                    ladder_losses[selected_kind],
                    minimum_relative_improvement=float(
                        implementation["selection_partition"]["minimum_relative_improvement"]
                    ),
                    minimum_positive_fraction=float(
                        implementation["selection_partition"]["minimum_positive_fold_fraction"]
                    ),
                )
                if not final_activation["pass"]:
                    selected_kind = "exact_zero"
                    selected_lambdas = pilot_lambdas
                else:
                    lambdas = list(pilot_lambdas)
                    for position, name in enumerate(("lambda_0", "lambda_tau", "lambda_x")):
                        scan: dict[float, list[float]] = {}
                        for value in c4["penalties"][name]:
                            candidate_lambdas = list(lambdas)
                            candidate_lambdas[position] = float(value)
                            scan[float(value)] = _candidate_losses(
                                accessor=train_accessor,
                                train=train,
                                folds=folds,
                                channel=channel,
                                profile=selected_profile,
                                m_tau=selected_m_tau,
                                kind=selected_kind,
                                m_x=selected_m_x,
                                lambdas=tuple(candidate_lambdas),
                                c4=c4,
                                scoring_history_steps=local_history,
                            )
                        choice = one_se_select(
                            scan,
                            lambda value: (-float(value),),
                            minimum_usable_folds=int(
                                implementation["selection_partition"]["minimum_usable_folds"]
                            ),
                        )
                        lambdas[position] = float(choice.selected)
                        penalty_audit[name] = {
                            "selection": choice.to_json(),
                            "fold_losses": _loss_json(scan),
                        }
                    selected_lambdas = tuple(lambdas)

        # This dictionary is complete before validation targets/accessor are loaded.
        selection_frozen = {
            "selection_partition": "train_only",
            "validation_used_for_selection": False,
            "test_accessed": False,
            "selected_profile": list(selected_profile),
            "retained_profiles": [list(value) for value in retained_profiles],
            "selected_kind": selected_kind,
            "selected_m_tau": int(selected_m_tau),
            "selected_m_x": int(selected_m_x),
            "selected_lambdas": [float(value) for value in selected_lambdas],
            "profile_selection": profile_selection.to_json(),
            "activation_one_se": activation_one_se.to_json(),
            "linear_activation": activation,
            "resolution_selection": None
            if resolution_selection is None
            else resolution_selection.to_json(),
            "ladder_selection": None if ladder_selection is None else ladder_selection.to_json(),
            "final_activation": final_activation,
            "penalty_audit": penalty_audit,
        }

        # Development holdout boundary: validation is materialized only after selection_frozen.
        validation = load_native_samples(shared, view, "validation")
        validation_accessor = BaseAccessor(shared, view.head.dataset, "validation", [channel])
        selected_history = int(selected_profile[1])
        final_train_native = apply_native_support(train, selected_history)
        final_train = final_train_native.iloc[
            deterministic_subsample(final_train_native, int(c4["fit_row_cap"]))
        ].reset_index(drop=True)
        selected_validation = apply_native_support(validation, selected_history).reset_index(drop=True)
        train_values, intervals = profile_values(
            train_accessor, final_train, channel, selected_profile, int(selected_m_tau)
        )
        validation_values, _ = profile_values(
            validation_accessor,
            selected_validation,
            channel,
            selected_profile,
            int(selected_m_tau),
        )
        prediction, model_contract, parameter_count = _fit_candidate(
            train_values,
            final_train["y_true"].to_numpy(dtype=np.float64),
            validation_values,
            selected_kind,
            int(selected_m_x),
            tuple(float(value) for value in selected_lambdas),
            c4["solver"],
        )
        numerical_status, relative_kkt, condition_number = _numerical_status(
            model_contract, c4["numerical_thresholds"]
        )
        frame = selected_validation[
            ["base_origin_id", "view_sample_id", "entity_id", "origin", "y_true"]
        ].copy()
        frame["y_pred"] = prediction
        frame["model"] = f"PRISM_V2_2_PMSM_SW_D_{channel}"
        frame["dtype"] = "float64"
        prediction_path = destination / "validation.parquet"
        frame.to_parquet(prediction_path, index=False, compression="zstd")

        result = {
            "status": "PASS",
            "stage": "PMSM_SW_D_ONLY_IMPLEMENTATION_AUDIT",
            "evidence_role": "DEVELOPMENT_IMPLEMENTATION_AUDIT_NOT_FINAL_CONFIRMATION",
            "dataset": view.head.dataset,
            "task": view.head.task_id,
            "target_head": view.head.head_id,
            "proxy_policy": view.proxy_policy,
            "channel": channel,
            "channel_class": channel_class("pmsm", channel),
            "support_contract": SUPPORT_CONTRACT,
            "profile_comparison_history_steps": int(profile_comparison_history),
            "local_scoring_history_steps": int(local_history),
            "local_scoring_rows_by_fold": local_scoring_rows,
            "exact_zero_scoring_support_hash": local_scoring_hashes,
            "nonzero_scoring_support_hash": local_scoring_hashes,
            "row_cap_applied_after_native_mask": True,
            "selection_frozen_before_validation_materialization": True,
            "selection": selection_frozen,
            "selected_intervals": [list(value) for value in intervals],
            "parameter_count": int(parameter_count),
            "model_contract": model_contract,
            "numerical_status": numerical_status,
            "relative_kkt": float(relative_kkt),
            "condition_number": float(condition_number),
            "active": bool(selected_kind != "exact_zero" and numerical_status == "PASS"),
            "profile_fold_losses": _loss_json(profile_losses),
            "linear_activation_fold_losses": [float(value) for value in linear_activation_losses],
            "ladder_fold_losses": _loss_json(ladder_losses),
            "selected_native_train_rows_before_cap": int(len(final_train_native)),
            "selected_fit_rows_after_cap": int(len(final_train)),
            "selected_native_validation_rows": int(len(selected_validation)),
            "selected_native_support_audit": {
                "train": support_audit(final_train_native),
                "validation": support_audit(selected_validation),
            },
            "prediction_path": str(prediction_path.relative_to(output)),
            "prediction_sha256": sha256_file(prediction_path),
            "validation_used_for_selection": False,
            "test_accessed": False,
            "elapsed_seconds": float(time.time() - started),
            **regression_metrics(frame["y_true"].to_numpy(dtype=np.float64), prediction),
        }
    except Exception as error:
        result = {
            "status": "SOLVER_FAILED_RETAINED",
            "stage": "PMSM_SW_D_ONLY_IMPLEMENTATION_AUDIT",
            "evidence_role": "DEVELOPMENT_IMPLEMENTATION_AUDIT_NOT_FINAL_CONFIRMATION",
            "target_head": view.head.head_id,
            "proxy_policy": view.proxy_policy,
            "channel": channel,
            "validation_used_for_selection": False,
            "test_accessed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": float(time.time() - started),
        }
    write_json(destination / "RESULT.json", result)
    return result


def run_primary_d_audit(
    shared: Path,
    project: Path,
    output: Path,
    channels: Sequence[str] | None = None,
) -> dict[str, Any]:
    assert_freeze_consistency(project)
    assert_prelockbox(shared, project)
    view = primary_view(shared)
    registered = input_columns(shared, view.head.task_id, view.proxy_policy)
    requested = list(registered if channels is None else channels)
    if len(requested) != len(set(requested)) or any(channel not in registered for channel in requested):
        raise RuntimeError(f"invalid requested channels: {requested}")
    results = [run_d_channel(shared, project, output, view, channel) for channel in requested]
    summary = {
        "status": "PASS" if all(item.get("status") == "PASS" for item in results) else "FAILED",
        "stage": "PMSM_SW_D_ONLY_IMPLEMENTATION_AUDIT",
        "evidence_role": "DEVELOPMENT_IMPLEMENTATION_AUDIT_NOT_FINAL_CONFIRMATION",
        "target_head": PRIMARY_HEAD_ID,
        "channels": requested,
        "all_registered_channels_run": requested == registered,
        "validation_used_for_selection": False,
        "test_accessed": False,
        "results": [
            {
                "channel": item.get("channel"),
                "status": item.get("status"),
                "active": item.get("active"),
                "selected_kind": item.get("selection", {}).get("selected_kind"),
                "selected_profile": item.get("selection", {}).get("selected_profile"),
                "mse": item.get("mse"),
                "rmse": item.get("rmse"),
                "mae": item.get("mae"),
                "r2": item.get("r2"),
                "error_type": item.get("error_type"),
                "error": item.get("error"),
            }
            for item in results
        ],
    }
    write_json(output / "D_ONLY_SUMMARY.json", summary)
    return summary
