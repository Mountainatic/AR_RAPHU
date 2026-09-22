"""Executable H/W/W0 contracts for the unified PRISM benchmark release.

The module intentionally contains no dataset loader.  It makes the registered
index arithmetic and reporting semantics reusable without exposing private CZ
data or silently changing model selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .level_reconstruction import metric_bundle_delta_and_level


REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "unified_hw_r2_protocols_20260921.json"
)


@dataclass(frozen=True)
class HWProtocol:
    head_id: str
    dataset: str
    cadence_seconds: float
    h_steps: int
    w_steps: int
    w0_steps: int | None
    target_kind: str
    history_steps: int | None = None

    @property
    def horizon_seconds(self) -> float:
        return self.cadence_seconds * self.h_steps


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_registry(value)
    return value


def protocols(path: Path = REGISTRY_PATH) -> dict[str, HWProtocol]:
    result: dict[str, HWProtocol] = {}
    for item in load_registry(path)["heads"]:
        protocol = HWProtocol(
            head_id=str(item["head_id"]),
            dataset=str(item["dataset"]),
            cadence_seconds=float(item["cadence_seconds"]),
            h_steps=int(item["h_steps"]),
            w_steps=int(item["w_steps"]),
            w0_steps=(
                None if item.get("w0_steps") is None else int(item["w0_steps"])
            ),
            target_kind=str(item["target_kind"]),
            history_steps=(
                None
                if item.get("history_steps") is None
                else int(item["history_steps"])
            ),
        )
        result[protocol.head_id] = protocol
    return result


def validate_registry(value: dict[str, Any]) -> None:
    if value.get("schema_version") != "1.0":
        raise ValueError("unsupported unified H/W registry schema")
    heads = value.get("heads")
    if not isinstance(heads, list) or not heads:
        raise ValueError("registry must contain heads")
    ids = [str(item.get("head_id")) for item in heads]
    if len(ids) != len(set(ids)):
        raise ValueError("head_id values must be unique")
    for item in heads:
        if float(item["cadence_seconds"]) <= 0:
            raise ValueError(f"non-positive cadence: {item['head_id']}")
        if int(item["h_steps"]) < 0 or int(item["w_steps"]) <= 0:
            raise ValueError(f"invalid H/W: {item['head_id']}")
        kind = str(item["target_kind"])
        if kind not in {"window_delta", "cz_raw2s_delta", "direct_level"}:
            raise ValueError(f"unknown target kind: {kind}")
        if kind != "direct_level" and int(item["w0_steps"]) <= 0:
            raise ValueError(f"delta target requires W0: {item['head_id']}")
        if kind == "cz_raw2s_delta":
            if item["w_steps"] != 1 or item["w0_steps"] != 1:
                raise ValueError("CZ raw-2s requires W/W0=1/1")
            if int(item.get("history_steps", 0)) != 256:
                raise ValueError("CZ raw-2s requires L=256")


def registered_levels(
    series: Iterable[float], origin: int, protocol: HWProtocol
) -> tuple[float | None, float]:
    """Return (current level, target level) using left-closed/right-open slices.

    ``origin`` is ``t`` for window-delta/CZ heads.  For Tanks it is the last
    observed sample index, matching ``run_cascaded_tanks_w_experiment.py``.
    """

    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("series must be a finite one-dimensional array")
    try:
        numeric_origin = float(origin)
    except (TypeError, ValueError) as exc:
        raise TypeError("origin must be an integer sample index") from exc
    if not np.isfinite(numeric_origin) or not numeric_origin.is_integer():
        raise ValueError("origin must be an exact finite integer sample index")
    t = int(numeric_origin)
    if protocol.target_kind == "window_delta":
        assert protocol.w0_steps is not None
        if (
            t - protocol.w0_steps < 0
            or t + protocol.h_steps < 0
            or t + protocol.h_steps + protocol.w_steps > len(values)
        ):
            raise IndexError("origin does not have complete registered windows")
        current = values[t - protocol.w0_steps : t]
        future = values[
            t + protocol.h_steps : t + protocol.h_steps + protocol.w_steps
        ]
        if len(current) != protocol.w0_steps or len(future) != protocol.w_steps:
            raise IndexError("origin does not have complete registered windows")
        return float(np.mean(current)), float(np.mean(future))
    if protocol.target_kind == "cz_raw2s_delta":
        history = int(protocol.history_steps or 0)
        if t < history or t + protocol.h_steps - 1 >= len(values):
            raise IndexError("origin does not have a complete CZ target")
        return float(values[t - 1]), float(values[t + protocol.h_steps - 1])
    target_index = t + protocol.h_steps
    if t < 0 or target_index >= len(values):
        raise IndexError("origin does not have a complete direct-level target")
    return None, float(values[target_index])


def registered_target(
    series: Iterable[float], origin: int, protocol: HWProtocol
) -> float:
    current, future = registered_levels(series, origin, protocol)
    return future if current is None else future - current


def report_delta_predictions(
    delta_true: Iterable[float],
    delta_pred: Iterable[float],
    current_level: Iterable[float],
) -> dict[str, float | str]:
    """Report delta and reconstructed-level metrics without reselection."""

    return metric_bundle_delta_and_level(delta_true, delta_pred, current_level)


def report_direct_level_predictions(
    level_true: Iterable[float], level_pred: Iterable[float]
) -> dict[str, float | None]:
    truth = np.asarray(level_true, dtype=np.float64)
    prediction = np.asarray(level_pred, dtype=np.float64)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction):
        raise ValueError("level arrays must be one-dimensional and equally sized")
    if not len(truth) or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("level arrays must be non-empty and finite")
    residual = truth - prediction
    denominator = float(np.sum(np.square(truth - np.mean(truth))))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2_level": None
        if denominator == 0.0
        else 1.0 - float(np.sum(np.square(residual))) / denominator,
    }
