from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import v211_k
from .cpu_data import ViewSpec, inner_folds
from .v2_k import (
    CHANNEL_SAMPLE_COLUMNS,
    channel_profiles as registered_channel_profiles,
)


def _profile_support_availability(
    anchor_train: pd.DataFrame,
    folds: list[tuple[Any, Any]],
    profiles: list[tuple[int, int]],
    *,
    fit_cap: int,
    evaluation_cap: int,
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for profile in profiles:
        history = int(profile[1])
        records = registered_fold_native_masks(
            anchor_train,
            folds,
            fit_history_steps=history,
            scoring_history_steps=history,
            fit_cap=int(fit_cap),
            evaluation_cap=int(evaluation_cap),
        )
        fit_rows = [len(record["fit_native"]) for record in records]
        scoring_rows = [
            len(record["evaluation_common"]) for record in records
        ]
        available = bool(records) and all(
            fit_rows[index] > 0 and scoring_rows[index] > 0
            for index in range(len(records))
        )
        audits.append(
            {
                "profile": list(profile),
                "history_steps": history,
                "status": (
                    "AVAILABLE" if available else "UNAVAILABLE_BY_SUPPORT"
                ),
                "required_folds": len(folds),
                "fit_rows_by_fold": fit_rows,
                "scoring_rows_by_fold": scoring_rows,
                "zero_fit_folds": [
                    index for index, rows in enumerate(fit_rows) if rows == 0
                ],
                "zero_scoring_folds": [
                    index
                    for index, rows in enumerate(scoring_rows)
                    if rows == 0
                ],
            }
        )
    return audits

from .v211_config import load_v211_configs
from .v211_support import load_native_samples, registered_fold_native_masks


def _available_profiles(
    shared: Path,
    project: Path,
    view: ViewSpec,
    channel: str,
    protocol: str,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    v211, v21, v2 = load_v211_configs(project, protocol=protocol)
    del v211
    train = load_native_samples(
        shared,
        view,
        "train",
        columns=CHANNEL_SAMPLE_COLUMNS,
    )
    folds = inner_folds(train, int(v21["selection"]["inner_folds"]))
    # run_cz_k_channel temporarily replaces v211_k.channel_profiles with the
    # support-filtering closure. Calling that mutable symbol here would recurse.
    profiles = registered_channel_profiles(view, channel, v2)
    fit_cap = int(v2["row_caps"]["single_channel_k_fit"])
    evaluation_cap = int(v2["row_caps"]["validation_selection_per_fold"])
    available: list[tuple[int, int]] = []
    audit: list[dict[str, Any]] = []
    for profile in profiles:
        history = int(profile[1])
        records = registered_fold_native_masks(
            train,
            folds,
            fit_history_steps=history,
            scoring_history_steps=history,
            fit_cap=fit_cap,
            evaluation_cap=evaluation_cap,
        )
        fit_rows = [len(record["fit_native"]) for record in records]
        scoring_rows = [
            len(record["evaluation_common"]) for record in records
        ]
        usable = bool(records) and all(
            fit_rows[index] > 0 and scoring_rows[index] > 0
            for index in range(len(records))
        )
        item = {
            "profile": list(profile),
            "history_steps": history,
            "status": "AVAILABLE" if usable else "UNAVAILABLE_BY_SUPPORT",
            "required_folds": len(folds),
            "fit_rows_by_fold": fit_rows,
            "scoring_rows_by_fold": scoring_rows,
            "zero_fit_folds": [
                index for index, rows in enumerate(fit_rows) if rows == 0
            ],
            "zero_scoring_folds": [
                index
                for index, rows in enumerate(scoring_rows)
                if rows == 0
            ],
        }
        audit.append(item)
        if usable:
            available.append(tuple(int(value) for value in profile))
    if not available:
        raise RuntimeError(
            "no registered CZ K profile has support in every inner fold: "
            f"{audit}"
        )
    return available, audit


def run_cz_k_channel(
    shared: Path,
    project: Path,
    output: Path,
    view: ViewSpec,
    channel: str,
    protocol: str = "public_all",
) -> dict[str, Any]:
    if view.head.dataset != "cz_czochralski":
        return v211_k.run_k_channel(
            shared,
            project,
            output,
            view,
            channel,
            protocol,
        )

    cache: dict[str, Any] = {}
    original_profiles = v211_k.channel_profiles

    def filtered_profiles(
        candidate_view: ViewSpec,
        candidate_channel: str,
        config: dict[str, Any],
    ) -> list[tuple[int, int]]:
        del config
        key = f"{candidate_view.relative_root}|{candidate_channel}"
        if key not in cache:
            profiles, audit = _available_profiles(
                shared,
                project,
                candidate_view,
                candidate_channel,
                protocol,
            )
            cache[key] = (profiles, audit)
        return list(cache[key][0])

    v211_k.channel_profiles = filtered_profiles
    try:
        result = v211_k.run_k_channel(
            shared,
            project,
            output,
            view,
            channel,
            protocol,
        )
    finally:
        v211_k.channel_profiles = original_profiles

    if isinstance(result, dict):
        result["native_profile_support_adapter"] = "CZ_SUPPORT_FILTER_R1"
        key = f"{view.relative_root}|{channel}"
        result["profile_support_availability"] = (
            cache[key][1] if key in cache else []
        )
    return result
