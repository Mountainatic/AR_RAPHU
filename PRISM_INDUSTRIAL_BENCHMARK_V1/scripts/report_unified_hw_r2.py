from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from prism_benchmark.unified_hw_protocol import (
    protocols,
    report_delta_predictions,
    report_direct_level_predictions,
)


def _columns(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("prediction CSV is empty")
    return {
        name: [float(row[name]) for row in rows]
        for name in rows[0]
        if name is not None
    }


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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    registered = protocols()
    if args.head_id not in registered:
        raise SystemExit(f"unknown registered head: {args.head_id}")
    protocol = registered[args.head_id]
    columns = _columns(args.predictions)
    if protocol.target_kind == "direct_level":
        required = {"y_true", "y_pred"}
        if not required.issubset(columns):
            raise SystemExit("direct-level CSV requires y_true,y_pred")
        metrics = report_direct_level_predictions(
            columns["y_true"], columns["y_pred"]
        )
    else:
        required = {"delta_true", "delta_pred", "current_level"}
        if not required.issubset(columns):
            raise SystemExit(
                "delta-head CSV requires delta_true,delta_pred,current_level"
            )
        metrics = report_delta_predictions(
            columns["delta_true"],
            columns["delta_pred"],
            columns["current_level"],
        )
    report = {
        "head_id": protocol.head_id,
        "target_kind": protocol.target_kind,
        "h_steps": protocol.h_steps,
        "w_steps": protocol.w_steps,
        "w0_steps": protocol.w0_steps,
        "metrics": _serializable(metrics),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
