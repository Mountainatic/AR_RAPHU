from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from prism_benchmark.unified_hw_protocol import (
    protocols,
    registered_levels,
    report_delta_predictions,
    report_direct_level_predictions,
)


def _columns(path: Path, required: Iterable[str]) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("prediction CSV is empty")
    required = set(required)
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"prediction CSV is missing columns: {sorted(missing)}")
    columns: dict[str, list[float]] = {}
    for name in required:
        try:
            columns[name] = [float(row[name]) for row in rows]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"column {name!r} must be numeric") from exc
    return columns


def _series(path: Path) -> np.ndarray:
    columns = _columns(path, {"value"})
    values = np.asarray(columns["value"], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError("series CSV value column must be non-empty and finite")
    return values


def _verify_targets(protocol: Any, columns: dict[str, list[float]], values: np.ndarray) -> None:
    for row, origin in enumerate(columns["origin"]):
        current, future = registered_levels(values, origin, protocol)
        if protocol.target_kind == "direct_level":
            if not np.isclose(columns["y_true"][row], future, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"row {row} y_true violates registered H/W target")
        else:
            assert current is not None
            expected_delta = future - current
            if not np.isclose(columns["current_level"][row], current, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"row {row} current_level violates registered W0 anchor")
            if not np.isclose(columns["delta_true"][row], expected_delta, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"row {row} delta_true violates registered H/W target")


def _serializable(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report the registered delta/level R2 pair for a unified H/W head."
    )
    parser.add_argument("--head-id", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--series",
        type=Path,
        help="one-column CSV named value; when supplied, H/W targets are reconstructed and verified",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    registered = protocols()
    if args.head_id not in registered:
        raise SystemExit(f"unknown registered head: {args.head_id}")
    protocol = registered[args.head_id]
    if protocol.target_kind == "direct_level":
        required = {"y_true", "y_pred"}
        if args.series is not None:
            required.add("origin")
        columns = _columns(args.predictions, required)
        metrics = report_direct_level_predictions(
            columns["y_true"], columns["y_pred"]
        )
    else:
        required = {"delta_true", "delta_pred", "current_level"}
        if args.series is not None:
            required.add("origin")
        columns = _columns(args.predictions, required)
        metrics = report_delta_predictions(
            columns["delta_true"],
            columns["delta_pred"],
            columns["current_level"],
        )
    target_verification = "NOT_REQUESTED_METRIC_REPORT_ONLY"
    if args.series is not None:
        _verify_targets(protocol, columns, _series(args.series))
        target_verification = "PASS_RECONSTRUCTED_FROM_SERIES"
    report = {
        "head_id": protocol.head_id,
        "target_kind": protocol.target_kind,
        "h_steps": protocol.h_steps,
        "w_steps": protocol.w_steps,
        "w0_steps": protocol.w0_steps,
        "target_verification": target_verification,
        "metrics": _serializable(metrics),
    }
    rendered = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
