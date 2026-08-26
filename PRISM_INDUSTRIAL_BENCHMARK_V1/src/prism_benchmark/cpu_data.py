from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .rust_kernels import block_means_prefix as rust_block_means_prefix


@dataclass(frozen=True)
class HeadSpec:
    head_id: str
    task_id: str
    dataset: str
    target: str
    cadence_seconds: float
    h_steps: int
    w_steps: int
    w0_steps: int
    primary: bool


@dataclass(frozen=True)
class ViewSpec:
    head: HeadSpec
    information_set: str
    availability_scenario: str
    proxy_policy: str

    @property
    def relative_root(self) -> Path:
        return Path(
            self.head.head_id,
            self.information_set,
            self.availability_scenario,
            self.proxy_policy,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_heads(shared: Path, primary_only: bool = True) -> list[HeadSpec]:
    payload = json.loads((shared / "TASK_REGISTRY.json").read_text(encoding="utf-8"))
    result = []
    for item in payload["heads"]:
        if primary_only and not item["primary"]:
            continue
        result.append(
            HeadSpec(
                head_id=item["head_id"],
                task_id=item["task_id"],
                dataset=item["dataset"],
                target=item["target"],
                cadence_seconds=float(item["cadence_seconds"]),
                h_steps=int(item["h_steps"]),
                w_steps=int(item["w_steps"]),
                w0_steps=int(item["w0_steps"]),
                primary=bool(item["primary"]),
            )
        )
    return result


def input_columns(shared: Path, task_id: str, proxy_policy: str) -> list[str]:
    payload = json.loads((shared / "dataset_views/VIEW_REGISTRY.json").read_text(encoding="utf-8"))
    for item in payload:
        if item["task_id"] == task_id and item["proxy_policy"] == proxy_policy:
            return list(item["input_columns"])
    raise KeyError((task_id, proxy_policy))


def primary_proxy_policy(shared: Path, task_id: str) -> str:
    protocol = json.loads((shared / "PROTOCOL.json").read_text(encoding="utf-8"))
    for task in protocol["tasks"]:
        if task["task_id"] == task_id:
            return str(task["proxy_policies"][0])
    raise KeyError(task_id)


def main_views(shared: Path, information_set: str) -> list[ViewSpec]:
    result = []
    for head in load_heads(shared, primary_only=True):
        result.append(
            ViewSpec(
                head=head,
                information_set=information_set,
                availability_scenario="record_time",
                proxy_policy=primary_proxy_policy(shared, head.task_id),
            )
        )
    return result


SAMPLE_RUNTIME_COLUMNS = [
    "base_origin_id",
    "view_sample_id",
    "dataset",
    "entity_id",
    "task_id",
    "target_head",
    "split",
    "origin",
    "dependency_start",
    "dependency_stop_exclusive",
    "latest_available_target_index",
    "y_true",
]


def load_samples(
    shared: Path,
    view: ViewSpec,
    split: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    path = shared / "sample_ids" / view.relative_root / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns or SAMPLE_RUNTIME_COLUMNS)


def _partitions_for(split: str) -> list[str]:
    if split == "train":
        return ["train"]
    if split == "validation":
        return ["train", "validation"]
    if split == "test":
        return ["train", "validation", "test"]
    if split == "ood":
        return ["train", "validation", "test", "ood"]
    raise KeyError(split)


class BaseAccessor:
    def __init__(self, shared: Path, dataset: str, split: str, columns: Iterable[str]) -> None:
        self.dataset = dataset
        requested = list(dict.fromkeys([*columns, "entity_id", "row_in_entity"]))
        frames = []
        for partition in _partitions_for(split):
            path = shared / "base_data" / dataset / f"{partition}.parquet"
            if path.is_file():
                frames.append(pd.read_parquet(path, columns=requested))
        if not frames:
            raise FileNotFoundError((dataset, split))
        frame = pd.concat(frames, ignore_index=True)
        self.columns = frozenset(
            column for column in requested if column not in {"entity_id", "row_in_entity"}
        )
        self.entities: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        for entity_id, group in frame.groupby("entity_id", sort=False):
            group = group.sort_values("row_in_entity")
            rows = group["row_in_entity"].to_numpy(dtype=np.int64)
            arrays = {
                column: group[column].to_numpy(dtype=np.float64)
                for column in requested
                if column not in {"entity_id", "row_in_entity"}
            }
            self.entities[str(entity_id)] = (rows, arrays)
        self._count_prefix: dict[str, tuple[int, np.ndarray]] = {}
        self._value_prefix: dict[tuple[str, str], tuple[int, np.ndarray]] = {}

    def has_columns(self, columns: Iterable[str]) -> bool:
        return set(columns).issubset(self.columns)

    def _prefixes(self, entity_id: str, column: str) -> tuple[int, np.ndarray, np.ndarray]:
        # A few protocol tests construct a minimal accessor with object.__new__;
        # keep the cache lazy so that reference fixture remains supported.
        if not hasattr(self, "_value_prefix"):
            self._value_prefix = {}
        if not hasattr(self, "_count_prefix"):
            self._count_prefix = {}
        value_key = (entity_id, column)
        cached_value = self._value_prefix.get(value_key)
        cached_count = self._count_prefix.get(entity_id)
        if cached_value is not None and cached_count is not None:
            if cached_value[0] != cached_count[0]:
                raise AssertionError("prefix origins disagree")
            return cached_value[0], cached_value[1], cached_count[1]
        rows, arrays = self.entities[entity_id]
        dense_min = int(rows.min())
        dense_max = int(rows.max())
        length = dense_max - dense_min + 1
        present = np.zeros(length, dtype=np.int64)
        present[rows - dense_min] = 1
        count_prefix = np.concatenate([[0], np.cumsum(present, dtype=np.int64)])
        dense = np.zeros(length, dtype=np.float64)
        dense[rows - dense_min] = arrays[column]
        value_prefix = np.concatenate([[0.0], np.cumsum(dense, dtype=np.float64)])
        self._count_prefix[entity_id] = (dense_min, count_prefix)
        self._value_prefix[value_key] = (dense_min, value_prefix)
        return dense_min, value_prefix, count_prefix

    def warm_prefixes(self, columns: Iterable[str]) -> None:
        selected = list(columns)
        if not self.has_columns(selected):
            raise KeyError(set(selected) - set(self.columns))
        for entity_id in self.entities:
            for column in selected:
                self._prefixes(entity_id, column)

    def gather(self, samples: pd.DataFrame, columns: list[str], indices: np.ndarray) -> np.ndarray:
        if indices.ndim == 1:
            indices = indices[:, None]
        result = np.empty((len(samples), indices.shape[1] * len(columns)), dtype=np.float64)
        entities = samples["entity_id"].astype(str).to_numpy()
        codes, labels = pd.factorize(entities, sort=False)
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes, minlength=len(labels))
        groups = np.split(order, np.cumsum(counts)[:-1])
        for entity_id, mask in zip(labels, groups, strict=True):
            rows, arrays = self.entities[entity_id]
            wanted = indices[mask]
            positions = np.searchsorted(rows, wanted)
            valid = positions < len(rows)
            clipped = np.minimum(positions, max(len(rows) - 1, 0))
            valid &= rows[clipped] == wanted
            if not bool(np.all(valid)):
                bad = wanted[~valid][:5]
                raise ValueError(f"missing causal rows for {entity_id}: {bad.tolist()}")
            blocks = [arrays[column][positions] for column in columns]
            result[mask] = np.concatenate(blocks, axis=1)
        return result

    def snapshot(self, samples: pd.DataFrame, columns: list[str]) -> np.ndarray:
        indices = samples["origin"].to_numpy(dtype=np.int64) - 1
        return self.gather(samples, columns, indices)

    def input_lags(
        self,
        samples: pd.DataFrame,
        columns: list[str],
        history_steps: int,
        maximum_lags: int,
    ) -> np.ndarray:
        lag_count = min(maximum_lags, max(1, history_steps))
        offsets = np.unique(np.rint(np.linspace(1, history_steps, lag_count)).astype(np.int64))
        indices = samples["origin"].to_numpy(dtype=np.int64)[:, None] - offsets[None, :]
        return self.gather(samples, columns, indices)

    def input_regular_lags(
        self,
        samples: pd.DataFrame,
        columns: list[str],
        delta_steps: int,
        history_steps: int,
        maximum_lags: int,
    ) -> np.ndarray:
        delta_steps = max(1, delta_steps)
        available = max(1, history_steps // delta_steps)
        lag_count = min(maximum_lags, available)
        offsets = np.unique(
            np.rint(np.linspace(delta_steps, history_steps, lag_count)).astype(np.int64)
        )
        indices = samples["origin"].to_numpy(dtype=np.int64)[:, None] - offsets[None, :]
        return self.gather(samples, columns, indices)

    def target_state(
        self,
        samples: pd.DataFrame,
        target: str,
        delta_steps: int,
        history_steps: int,
    ) -> np.ndarray:
        count = max(1, history_steps // max(delta_steps, 1))
        offsets = np.arange(count, dtype=np.int64) * max(delta_steps, 1)
        latest = samples["latest_available_target_index"].to_numpy(dtype=np.int64)
        indices = latest[:, None] - offsets[None, :]
        levels = self.gather(samples, [target], indices)
        changes = np.empty_like(levels)
        changes[:, 0] = 0.0
        if levels.shape[1] > 1:
            changes[:, 1:] = levels[:, :-1] - levels[:, 1:]
        return np.concatenate([levels, changes], axis=1)

    def block_means(
        self,
        samples: pd.DataFrame,
        column: str,
        intervals: list[tuple[int, int]],
    ) -> np.ndarray:
        result = np.empty((len(samples), len(intervals)), dtype=np.float64)
        entities = samples["entity_id"].astype(str).to_numpy()
        origins = samples["origin"].to_numpy(dtype=np.int64)
        codes, labels = pd.factorize(entities, sort=False)
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes, minlength=len(labels))
        groups = np.split(order, np.cumsum(counts)[:-1])
        for entity_id, mask in zip(labels, groups, strict=True):
            dense_min, prefix, count_prefix = self._prefixes(entity_id, column)
            dense_length = len(count_prefix) - 1
            rust_result = rust_block_means_prefix(
                origins[mask], dense_min, prefix, count_prefix, intervals
            )
            if rust_result is not None:
                result[mask] = rust_result
                continue
            for block_index, (near, far) in enumerate(intervals):
                starts = origins[mask] - far - dense_min
                stops = origins[mask] - near - dense_min
                if np.any(starts < 0) or np.any(stops > dense_length):
                    raise ValueError(f"block outside entity support: {entity_id}")
                block_counts = count_prefix[stops] - count_prefix[starts]
                expected = far - near
                if np.any(block_counts != expected):
                    raise ValueError(f"block crosses missing rows: {entity_id}")
                result[mask, block_index] = (prefix[stops] - prefix[starts]) / expected
        return result


def geometric_intervals(history_steps: int, blocks: int) -> list[tuple[int, int]]:
    if history_steps < 1 or blocks < 1:
        raise ValueError((history_steps, blocks))
    blocks = min(blocks, history_steps)
    raw = np.geomspace(1, history_steps + 1, blocks + 1) - 1
    edges = np.rint(raw).astype(np.int64)
    edges[0] = 0
    edges[-1] = history_steps
    edges = np.unique(edges)
    if len(edges) - 1 < blocks:
        edges = np.unique(np.rint(np.linspace(0, history_steps, blocks + 1)).astype(np.int64))
    intervals = [(int(edges[index]), int(edges[index + 1])) for index in range(len(edges) - 1)]
    if not intervals or intervals[-1][1] != history_steps:
        raise AssertionError((history_steps, blocks, intervals))
    return intervals


def deterministic_subsample(samples: pd.DataFrame, cap: int) -> np.ndarray:
    if len(samples) <= cap:
        return np.arange(len(samples), dtype=np.int64)
    hashes = np.fromiter(
        (int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) for value in samples["base_origin_id"].astype(str)),
        dtype=np.uint64,
        count=len(samples),
    )
    return np.sort(np.argpartition(hashes, cap - 1)[:cap])


def _temporal_entity_folds(
    samples: pd.DataFrame,
    groups: pd.Series,
    unique_entities: Sequence[str],
    count: int,
    extra_buffer_steps: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build causal time folds independently inside each short entity set."""
    parts: list[tuple[list[np.ndarray], list[np.ndarray]]] = [
        ([], []) for _ in range(count)
    ]
    group_values = groups.to_numpy(dtype=object)
    for entity in unique_entities:
        positions = np.flatnonzero(group_values == entity)
        subset = samples.iloc[positions]
        minimum = int(subset["dependency_start"].min())
        maximum = int(subset["dependency_stop_exclusive"].max())
        boundaries = np.rint(
            np.linspace(minimum, maximum, count + 2)
        ).astype(np.int64)
        starts = subset["dependency_start"].to_numpy(dtype=np.int64)
        stops = subset["dependency_stop_exclusive"].to_numpy(dtype=np.int64)
        for index in range(count):
            boundary = int(boundaries[index + 1])
            next_boundary = int(boundaries[index + 2])
            train_local = np.flatnonzero(stops <= boundary)
            validation_local = np.flatnonzero(
                (starts >= boundary + extra_buffer_steps)
                & (stops <= next_boundary)
            )
            if len(train_local) and len(validation_local):
                parts[index][0].append(positions[train_local])
                parts[index][1].append(positions[validation_local])
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_parts, validation_parts in parts:
        if train_parts and validation_parts:
            folds.append(
                (
                    np.concatenate(train_parts).astype(np.int64),
                    np.concatenate(validation_parts).astype(np.int64),
                )
            )
    return folds


def inner_folds(samples: pd.DataFrame, count: int = 4, extra_buffer_steps: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    entities = samples["entity_id"].astype(str)
    dataset = str(samples["dataset"].iloc[0])
    if dataset == "tep":
        groups = entities.str.extract(r"run=(\d+)$", expand=False).astype(int).astype(str)
    else:
        groups = entities
    unique_entities = sorted(groups.unique(), key=lambda value: (len(value), value))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if 1 < len(unique_entities) <= count:
        folds = _temporal_entity_folds(
            samples,
            groups,
            unique_entities,
            count,
            extra_buffer_steps,
        )
    elif len(unique_entities) > 1:
        chunks = [chunk.tolist() for chunk in np.array_split(np.asarray(unique_entities, dtype=object), count + 1)]
        for index in range(count):
            train_entities = set(value for chunk in chunks[: index + 1] for value in chunk)
            validation_entities = set(chunks[index + 1])
            train_index = np.flatnonzero(groups.isin(train_entities).to_numpy())
            validation_index = np.flatnonzero(groups.isin(validation_entities).to_numpy())
            if len(train_index) and len(validation_index):
                folds.append((train_index, validation_index))
    else:
        minimum = int(samples["dependency_start"].min())
        maximum = int(samples["dependency_stop_exclusive"].max())
        boundaries = np.rint(np.linspace(minimum, maximum, count + 2)).astype(np.int64)
        for index in range(count):
            boundary = int(boundaries[index + 1])
            next_boundary = int(boundaries[index + 2])
            train_mask = samples["dependency_stop_exclusive"].to_numpy(dtype=np.int64) <= boundary
            validation_mask = (
                (samples["dependency_start"].to_numpy(dtype=np.int64) >= boundary + extra_buffer_steps)
                & (samples["dependency_stop_exclusive"].to_numpy(dtype=np.int64) <= next_boundary)
            )
            train_index = np.flatnonzero(train_mask)
            validation_index = np.flatnonzero(validation_mask)
            if len(train_index) and len(validation_index):
                folds.append((train_index, validation_index))
    if len(folds) != count:
        raise ValueError(f"expected {count} nonempty inner folds, got {len(folds)}")
    return folds


def realized_state_profiles(
    head: HeadSpec,
    positive_h_history_multipliers: Sequence[int] | None = None,
    delta_steps_override: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    window = head.w_steps
    if delta_steps_override is None:
        deltas = sorted(
            {
                max(1, int(math.floor(window * ratio + 0.5)))
                for ratio in (0.25, 0.5, 1.0, 2.0)
            }
        )
    else:
        from .v211_history_override import (
            validate_positive_h_history_multipliers,
        )

        deltas = list(
            validate_positive_h_history_multipliers(delta_steps_override)
        )
    result = []
    for delta in deltas:
        if head.h_steps > 0:
            if positive_h_history_multipliers is None:
                multipliers = (2, 4, 8)
            else:
                from .v211_history_override import (
                    validate_positive_h_history_multipliers,
                )

                multipliers = validate_positive_h_history_multipliers(
                    positive_h_history_multipliers
                )
            histories = [
                max(delta, multiplier * head.h_steps)
                for multiplier in multipliers
            ]
        else:
            if positive_h_history_multipliers is not None:
                raise ValueError(
                    "positive-h history override cannot be used for h=0"
                )
            histories = [multiplier * delta for multiplier in (4, 16, 64)]
        result.extend((delta, history) for history in histories)
    # Rounding and the ``max(delta, multiplier * h)`` maturity floor can map
    # different registered grid points to the same realized profile.  A
    # realized profile is a semantic candidate, so preserve registration order
    # while evaluating each unique candidate exactly once.
    return list(dict.fromkeys(result))
