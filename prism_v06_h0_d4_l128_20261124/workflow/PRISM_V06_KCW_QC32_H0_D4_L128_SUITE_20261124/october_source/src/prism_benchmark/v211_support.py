from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .cpu_data import SAMPLE_RUNTIME_COLUMNS, ViewSpec, load_samples
from .v2_k import _cap


SUPPORT_CONTRACT = "NATIVE_K_COMMON_ASSEMBLY_R1"
LEGACY_SUPPORT_ERROR = "NATIVE_SUPPORT_REQUIRES_REBUILT_C1_ANCHOR_UNIVERSE"
SUPPORT_COLUMNS = (
    "causal_history_floor",
    "anchor_history_steps",
    "sample_support_contract",
)


def require_native_support_contract(samples: pd.DataFrame) -> None:
    missing = [column for column in SUPPORT_COLUMNS if column not in samples.columns]
    if missing:
        raise RuntimeError(f"{LEGACY_SUPPORT_ERROR}: missing columns {missing}")
    contracts = set(samples["sample_support_contract"].astype(str).unique())
    if contracts != {SUPPORT_CONTRACT}:
        raise RuntimeError(
            f"{LEGACY_SUPPORT_ERROR}: observed support contracts {sorted(contracts)}"
        )


def load_native_samples(
    shared: Path,
    view: ViewSpec,
    split: str,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    requested = list(
        dict.fromkeys(
            [*(SAMPLE_RUNTIME_COLUMNS if columns is None else columns), *SUPPORT_COLUMNS]
        )
    )
    path = shared / "sample_ids" / view.relative_root / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    missing = set(SUPPORT_COLUMNS).difference(pq.ParquetFile(path).schema.names)
    if missing:
        raise RuntimeError(f"{LEGACY_SUPPORT_ERROR}: missing columns {sorted(missing)}")
    samples = load_samples(shared, view, split, columns=requested)
    require_native_support_contract(samples)
    return samples


def native_support_mask(
    samples: pd.DataFrame,
    history_steps: int,
    additional_causal_history_floor: int | Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    require_native_support_contract(samples)
    history = int(history_steps)
    if history < 0:
        raise ValueError("history_steps must be nonnegative")
    origin = samples["origin"].to_numpy(dtype=np.int64)
    floor = samples["causal_history_floor"].to_numpy(dtype=np.int64)
    if additional_causal_history_floor is not None:
        additional = np.asarray(
            additional_causal_history_floor, dtype=np.int64
        )
        if additional.ndim == 0:
            additional = np.full(len(samples), int(additional), dtype=np.int64)
        if additional.shape != floor.shape:
            raise ValueError("additional causal floor does not match sample rows")
        floor = np.maximum(floor, additional)
    return origin - history >= floor


def apply_native_support(
    samples: pd.DataFrame,
    history_steps: int,
    additional_causal_history_floor: int | Sequence[int] | np.ndarray | None = None,
) -> pd.DataFrame:
    return samples.loc[
        native_support_mask(
            samples, history_steps, additional_causal_history_floor
        )
    ].copy()


def fold_evaluation_causal_floor(
    fit_raw: pd.DataFrame,
    evaluation_raw: pd.DataFrame,
) -> np.ndarray | None:
    """Preserve a registered temporal fold's purge boundary for long histories."""
    fit_entities = set(fit_raw["entity_id"].astype(str))
    evaluation_entities = set(evaluation_raw["entity_id"].astype(str))
    overlapping = fit_entities.intersection(evaluation_entities)
    if not overlapping:
        return None
    boundaries = (
        evaluation_raw.assign(_entity=evaluation_raw["entity_id"].astype(str))
        .groupby("_entity", sort=False)["dependency_start"]
        .min()
        .to_dict()
    )
    return np.asarray(
        [boundaries[str(entity)] for entity in evaluation_raw["entity_id"]],
        dtype=np.int64,
    )


def support_id_hash(samples: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for base_origin_id, view_sample_id in zip(
        samples["base_origin_id"].astype(str),
        samples["view_sample_id"].astype(str),
        strict=True,
    ):
        digest.update(base_origin_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(view_sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def base_origin_support_hash(samples: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for base_origin_id in samples["base_origin_id"].astype(str):
        digest.update(base_origin_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def support_audit(samples: pd.DataFrame) -> dict[str, Any]:
    require_native_support_contract(samples)
    origins = samples["origin"].to_numpy(dtype=np.int64)
    floors = samples["causal_history_floor"].to_numpy(dtype=np.int64)
    return {
        "rows": int(len(samples)),
        "support_hash": support_id_hash(samples),
        "minimum_origin": None if len(origins) == 0 else int(origins.min()),
        "maximum_origin": None if len(origins) == 0 else int(origins.max()),
        "minimum_causal_history_floor": None
        if len(floors) == 0
        else int(floors.min()),
        "maximum_causal_history_floor": None
        if len(floors) == 0
        else int(floors.max()),
        "support_contract": SUPPORT_CONTRACT,
    }


def intersection_support_by_base_origin_id(
    samples: pd.DataFrame,
    histories: Iterable[int],
    additional_causal_history_floor: int | Sequence[int] | np.ndarray | None = None,
) -> pd.DataFrame:
    registered = [int(value) for value in histories]
    if not registered:
        require_native_support_contract(samples)
        return samples.copy()
    mask = np.ones(len(samples), dtype=bool)
    for history in registered:
        mask &= native_support_mask(
            samples, history, additional_causal_history_floor
        )
    return samples.loc[mask].copy()


def apply_row_cap_after_support(
    samples: pd.DataFrame,
    history_steps: int,
    cap: int,
) -> pd.DataFrame:
    native = apply_native_support(samples, history_steps)
    return _cap(native, int(cap)).reset_index(drop=True)


def selected_k_histories(active: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    histories: dict[str, int] = {}
    for item in active:
        if item.get("support_contract") != SUPPORT_CONTRACT:
            raise RuntimeError(LEGACY_SUPPORT_ERROR)
        channel = str(item["channel"])
        history = int(
            item.get(
                "selected_profile_history_steps",
                item.get("selected_profile", [None, None])[1],
            )
        )
        histories[channel] = history
    return histories


def apply_assembly_support(
    samples: pd.DataFrame,
    active: Sequence[Mapping[str, Any]],
    additional_causal_history_floor: int | Sequence[int] | np.ndarray | None = None,
) -> pd.DataFrame:
    return intersection_support_by_base_origin_id(
        samples,
        selected_k_histories(active).values(),
        additional_causal_history_floor,
    )


def registered_fold_native_masks(
    anchor_train: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    fit_history_steps: int,
    scoring_history_steps: int,
    fit_cap: int,
    evaluation_cap: int,
) -> list[dict[str, Any]]:
    result = []
    for fold_index, (fit_index, evaluation_index) in enumerate(folds):
        fit_raw = anchor_train.iloc[fit_index]
        evaluation_raw = anchor_train.iloc[evaluation_index]
        fit_native = apply_native_support(fit_raw, fit_history_steps)
        evaluation_floor = fold_evaluation_causal_floor(
            fit_raw, evaluation_raw
        )
        evaluation_common = apply_native_support(
            evaluation_raw,
            scoring_history_steps,
            evaluation_floor,
        )
        fit = _cap(fit_native, int(fit_cap)).reset_index(drop=True)
        evaluation = _cap(
            evaluation_common, int(evaluation_cap)
        ).reset_index(drop=True)
        result.append(
            {
                "fold_index": int(fold_index),
                "fit_raw": fit_raw.reset_index(drop=True),
                "evaluation_raw": evaluation_raw.reset_index(drop=True),
                "fit_native": fit_native.reset_index(drop=True),
                "evaluation_common": evaluation_common.reset_index(drop=True),
                "fit": fit,
                "evaluation": evaluation,
                "fit_support_hash": support_id_hash(fit),
                "evaluation_support_hash": support_id_hash(evaluation),
            }
        )
    return result
