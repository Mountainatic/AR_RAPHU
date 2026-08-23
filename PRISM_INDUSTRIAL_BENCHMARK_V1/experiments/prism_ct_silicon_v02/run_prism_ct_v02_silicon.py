#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from prism_benchmark.prism_ct import (
    CTBasisConfig,
    aligned_temporal_blocks,
    config_dict,
    feature_audit,
    fit_ridge,
    predict_ridge,
    regression_metrics,
    support_audit,
)

HORIZONS = (1, 5, 15, 30, 60, 120, 300, 600)
COMPLEXITY = {
    "persistence": 0,
    "delay": 2,
    "ct_multires": 2,
    "ct_absolute": 3,
    "delay_ct_multires": 4,
}


def _fold_summary(losses: list[float]) -> tuple[float, float]:
    array = np.asarray(losses, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
    )


def _make_inner_folds(
    start: int,
    stop: int,
    horizon: int,
    count: int = 4,
    buffer_steps: int = 300,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Four expanding-regime folds inside the source/train interval.

    The 300-step buffer is 600 s at the frozen 2 s cadence, matching the ten-minute
    separation used by the current PRISM CPU inner-selection policy.
    """
    edges = np.rint(np.linspace(start, stop, count + 2)).astype(int)
    folds = []
    for index in range(count):
        boundary = int(edges[index + 1])
        next_boundary = int(edges[index + 2])
        train = np.arange(start, max(start, boundary - horizon))
        validation_start = max(boundary + buffer_steps, start)
        validation = np.arange(
            min(next_boundary, validation_start),
            max(validation_start, next_boundary - horizon),
        )
        if len(train) and len(validation):
            folds.append((train, validation))
    if len(folds) != count:
        raise ValueError(
            f"expected {count} nonempty inner folds, got {len(folds)}"
        )
    return folds


def _rows(indices: np.ndarray, start: int) -> np.ndarray:
    return indices - start


def _fit_predict(
    block: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    start: int,
    horizon: int,
    config: CTBasisConfig,
) -> np.ndarray:
    model = fit_ridge(
        block[_rows(train, start)],
        y[train + horizon] - y[train],
        config,
    )
    return y[evaluation] + predict_ridge(
        model,
        block[_rows(evaluation, start)],
    )


def _select_route(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    start: int,
    selection_stop: int,
    horizon: int,
    config: CTBasisConfig,
) -> tuple[str, dict[str, tuple[float, float]], dict[str, list[float]]]:
    folds = _make_inner_folds(start, selection_stop, horizon)
    losses: dict[str, list[float]] = {"persistence": []}
    for name in blocks:
        losses[name] = []

    for train, validation in folds:
        losses["persistence"].append(
            float(np.mean((y[validation + horizon] - y[validation]) ** 2))
        )
        for name, block in blocks.items():
            audit = feature_audit(block[_rows(train, start)], config)
            if not audit.passed_conditioning:
                losses[name].append(float("inf"))
                continue
            prediction = _fit_predict(
                block,
                y,
                train,
                validation,
                start,
                horizon,
                config,
            )
            losses[name].append(
                float(np.mean((y[validation + horizon] - prediction) ** 2))
            )

    stats = {
        name: _fold_summary(values)
        for name, values in losses.items()
        if np.all(np.isfinite(values))
    }
    best = min(stats, key=lambda name: stats[name][0])
    threshold = stats[best][0] + stats[best][1]
    within = [name for name in stats if stats[name][0] <= threshold]
    selected = min(
        within,
        key=lambda name: (COMPLEXITY[name], stats[name][0], name),
    )
    return selected, stats, losses


def _evaluate_domain(
    domain: str,
    x: np.ndarray,
    y: np.ndarray,
    config: CTBasisConfig,
    records: list[dict[str, object]],
    ablations: list[dict[str, object]],
) -> None:
    blocks, start = aligned_temporal_blocks(x, config)
    n = len(y)
    train_end = int(0.60 * n)
    validation_end = int(0.80 * n)

    for horizon in HORIZONS:
        selected, stats, fold_losses = _select_route(
            blocks,
            y,
            start,
            train_end,
            horizon,
            config,
        )

        train = np.arange(start, train_end - horizon)
        validation = np.arange(train_end + 300, validation_end - horizon)
        test = np.arange(validation_end, n - horizon)

        if selected == "persistence":
            validation_prediction = y[validation]
        else:
            validation_prediction = _fit_predict(
                blocks[selected],
                y,
                train,
                validation,
                start,
                horizon,
                config,
            )
        validation_metrics = regression_metrics(
            y[validation + horizon],
            validation_prediction,
            y[validation],
        )

        fit = np.arange(start, validation_end - horizon)
        if selected == "persistence":
            test_prediction = y[test]
        else:
            test_prediction = _fit_predict(
                blocks[selected],
                y,
                fit,
                test,
                start,
                horizon,
                config,
            )
        metrics = regression_metrics(
            y[test + horizon],
            test_prediction,
            y[test],
        )
        records.append(
            {
                "evaluation": "within_inner4_route_outer_holdout",
                "domain": domain,
                "horizon_steps": horizon,
                "horizon_seconds": horizon * config.dt_seconds,
                "selected_branch": selected,
                "outer_validation_skill": validation_metrics["persistence_skill_mse"],
                **metrics,
            }
        )

        # Descriptive only. These rows are never consulted by _select_route.
        for name, block in blocks.items():
            audit = feature_audit(block[_rows(fit, start)], config)
            if not audit.passed_conditioning:
                continue
            prediction = _fit_predict(
                block,
                y,
                fit,
                test,
                start,
                horizon,
                config,
            )
            branch_metrics = regression_metrics(
                y[test + horizon],
                prediction,
                y[test],
            )
            ablations.append(
                {
                    "evaluation": "descriptive_test_ablation",
                    "domain": domain,
                    "horizon_steps": horizon,
                    "branch": name,
                    "selected": name == selected,
                    **branch_metrics,
                    **audit.__dict__,
                }
            )


def _evaluate_cross(
    label: str,
    source_x: np.ndarray,
    source_y: np.ndarray,
    destination_x: np.ndarray,
    destination_y: np.ndarray,
    config: CTBasisConfig,
    records: list[dict[str, object]],
) -> None:
    source_blocks, start = aligned_temporal_blocks(source_x, config)
    destination_blocks, destination_start = aligned_temporal_blocks(
        destination_x,
        config,
    )
    if start != destination_start:
        raise AssertionError("shared time basis must have identical support")

    selection_stop = int(0.80 * len(source_y))
    for horizon in HORIZONS:
        selected, stats, fold_losses = _select_route(
            source_blocks,
            source_y,
            start,
            selection_stop,
            horizon,
            config,
        )
        source_fit = np.arange(start, len(source_y) - horizon)
        destination = np.arange(start, len(destination_y) - horizon)

        if selected == "persistence":
            prediction = destination_y[destination]
            support = None
        else:
            model = fit_ridge(
                source_blocks[selected][_rows(source_fit, start)],
                source_y[source_fit + horizon] - source_y[source_fit],
                config,
            )
            prediction = destination_y[destination] + predict_ridge(
                model,
                destination_blocks[selected][_rows(destination, start)],
            )
            support = support_audit(
                source_blocks[selected][_rows(source_fit, start)],
                destination_blocks[selected][_rows(destination, start)],
                config,
            )

        metrics = regression_metrics(
            destination_y[destination + horizon],
            prediction,
            destination_y[destination],
        )
        records.append(
            {
                "evaluation": "cross_source_inner4_route_frozen",
                "domain": label,
                "horizon_steps": horizon,
                "horizon_seconds": horizon * config.dt_seconds,
                "selected_branch": selected,
                "support_passed": None if support is None else support.passed,
                "support_fraction": (
                    None if support is None else support.fraction_within_z_limit
                ),
                **metrics,
            }
        )


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    parser.add_argument("--out-dir", default="prism_ct_v02_results")
    args = parser.parse_args()

    data = np.load(args.npz)
    columns = [str(value) for value in data["columns"]]
    sheet1 = data["sheet1"]
    sheet2 = data["sheet2"]
    if columns[-1] != "晶体直径":
        raise ValueError(f"expected final target 晶体直径, got {columns[-1]}")

    # Current diameter y_t is causally available and is intentionally included in the
    # temporal state bank. The supervised label remains future diameter y[t+h].
    x1, y1 = sheet1, sheet1[:, -1]
    x2, y2 = sheet2, sheet2[:, -1]

    config = CTBasisConfig()
    records: list[dict[str, object]] = []
    ablations: list[dict[str, object]] = []

    _evaluate_domain("Sheet1", x1, y1, config, records, ablations)
    _evaluate_domain("Sheet2", x2, y2, config, records, ablations)
    _evaluate_cross("Sheet1->Sheet2", x1, y1, x2, y2, config, records)
    _evaluate_cross("Sheet2->Sheet1", x2, y2, x1, y1, config, records)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "metrics.csv", records)
    _write_csv(output / "descriptive_ablation.csv", ablations)
    (output / "freeze.json").write_text(
        json.dumps(
            {
                "version": "PRISM-CT-v0.2",
                "columns": columns,
                "horizons_steps": HORIZONS,
                "config": config_dict(config),
                "inner_folds": 4,
                "selection_buffer_steps": 300,
                "selection_buffer_seconds": 600,
                "route": (
                    "one-SE across train/source-only expanding regime folds; "
                    "complexity tie-break; outer holdout never selects route"
                ),
                "descriptive_test_ablation": "NOT_FOR_SELECTION",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
