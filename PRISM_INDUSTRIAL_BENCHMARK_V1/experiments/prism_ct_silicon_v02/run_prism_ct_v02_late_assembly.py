#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
from prism_benchmark.prism_ct_assembly import (
    fit_simplex_assembly,
    predict_simplex_assembly,
)

HORIZONS = (1, 5, 15, 30, 60, 120, 300, 600)
DYNAMIC_BRANCHES = ("delay", "ct_multires", "ct_absolute")


def _rows(indices: np.ndarray, start: int) -> np.ndarray:
    return indices - start


def _fit_branch_predictions(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    start: int,
    horizon: int,
    config: CTBasisConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    predictions: dict[str, np.ndarray] = {}
    audits: dict[str, object] = {}
    for name in DYNAMIC_BRANCHES:
        block = blocks[name]
        audit = feature_audit(block[_rows(train, start)], config)
        audits[name] = audit
        if not audit.passed_conditioning:
            continue
        model = fit_ridge(
            block[_rows(train, start)],
            y[train + horizon] - y[train],
            config,
        )
        predictions[name] = predict_ridge(
            model,
            block[_rows(evaluation, start)],
        )
    if not predictions:
        raise ValueError("no numerically admissible dynamic branch")
    return predictions, audits


def _evaluate_within(
    domain: str,
    x: np.ndarray,
    y: np.ndarray,
    config: CTBasisConfig,
    records: list[dict[str, object]],
) -> None:
    blocks, start = aligned_temporal_blocks(x, config)
    n_rows = len(y)
    train_end = int(0.60 * n_rows)
    validation_end = int(0.80 * n_rows)

    for horizon in HORIZONS:
        train = np.arange(start, train_end - horizon)
        validation = np.arange(train_end + 300, validation_end - horizon)
        test = np.arange(validation_end, n_rows - horizon)

        validation_predictions, audits = _fit_branch_predictions(
            blocks,
            y,
            train,
            validation,
            start,
            horizon,
            config,
        )
        assembly = fit_simplex_assembly(
            validation_predictions,
            y[validation + horizon] - y[validation],
            ridge=1e-3,
        )

        fit = np.arange(start, validation_end - horizon)
        test_predictions, fit_audits = _fit_branch_predictions(
            blocks,
            y,
            fit,
            test,
            start,
            horizon,
            config,
        )
        test_predictions = {
            name: test_predictions[name]
            for name in assembly.branch_names
        }
        delta = predict_simplex_assembly(assembly, test_predictions)
        prediction = y[test] + delta
        metrics = regression_metrics(
            y[test + horizon],
            prediction,
            y[test],
        )
        records.append(
            {
                "evaluation": "within_60_20_20_late_assembly",
                "domain": domain,
                "horizon_steps": horizon,
                "horizon_seconds": horizon * config.dt_seconds,
                "branch_names": "|".join(assembly.branch_names),
                "dynamic_weights": "|".join(
                    f"{value:.12g}" for value in assembly.weights
                ),
                "persistence_weight": assembly.persistence_weight,
                "assembly_iterations": assembly.iterations,
                **metrics,
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
        raise AssertionError("shared CT basis must have identical support")

    train_end = int(0.60 * len(source_y))
    validation_end = int(0.80 * len(source_y))

    for horizon in HORIZONS:
        train = np.arange(start, train_end - horizon)
        validation = np.arange(train_end + 300, validation_end - horizon)
        validation_predictions, audits = _fit_branch_predictions(
            source_blocks,
            source_y,
            train,
            validation,
            start,
            horizon,
            config,
        )
        assembly = fit_simplex_assembly(
            validation_predictions,
            source_y[validation + horizon] - source_y[validation],
            ridge=1e-3,
        )

        source_fit = np.arange(start, len(source_y) - horizon)
        destination = np.arange(start, len(destination_y) - horizon)
        destination_predictions: dict[str, np.ndarray] = {}
        support_flags: dict[str, bool] = {}
        for name in assembly.branch_names:
            source_block = source_blocks[name]
            destination_block = destination_blocks[name]
            model = fit_ridge(
                source_block[_rows(source_fit, start)],
                source_y[source_fit + horizon] - source_y[source_fit],
                config,
            )
            destination_predictions[name] = predict_ridge(
                model,
                destination_block[_rows(destination, start)],
            )
            support = support_audit(
                source_block[_rows(source_fit, start)],
                destination_block[_rows(destination, start)],
                config,
            )
            support_flags[name] = support.passed

        delta = predict_simplex_assembly(assembly, destination_predictions)
        prediction = destination_y[destination] + delta
        metrics = regression_metrics(
            destination_y[destination + horizon],
            prediction,
            destination_y[destination],
        )
        records.append(
            {
                "evaluation": "cross_sheet_source_validation_late_assembly",
                "domain": label,
                "horizon_steps": horizon,
                "horizon_seconds": horizon * config.dt_seconds,
                "branch_names": "|".join(assembly.branch_names),
                "dynamic_weights": "|".join(
                    f"{value:.12g}" for value in assembly.weights
                ),
                "persistence_weight": assembly.persistence_weight,
                "support_flags": "|".join(
                    f"{name}:{support_flags[name]}"
                    for name in assembly.branch_names
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
    parser.add_argument("--out-dir", default="prism_ct_v02_late_assembly_results")
    args = parser.parse_args()

    data = np.load(args.npz)
    columns = [str(value) for value in data["columns"]]
    sheet1 = data["sheet1"]
    sheet2 = data["sheet2"]
    if columns[-1] != "晶体直径":
        raise ValueError(f"expected final target 晶体直径, got {columns[-1]}")

    # y_t is causally known and remains part of the temporal state bank.
    x1, y1 = sheet1, sheet1[:, -1]
    x2, y2 = sheet2, sheet2[:, -1]
    config = CTBasisConfig()
    records: list[dict[str, object]] = []

    _evaluate_within("Sheet1", x1, y1, config, records)
    _evaluate_within("Sheet2", x2, y2, config, records)
    _evaluate_cross("Sheet1->Sheet2", x1, y1, x2, y2, config, records)
    _evaluate_cross("Sheet2->Sheet1", x2, y2, x1, y1, config, records)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "metrics.csv", records)
    (output / "freeze.json").write_text(
        json.dumps(
            {
                "version": "PRISM-CT-v0.2-late-assembly",
                "columns": columns,
                "horizons_steps": HORIZONS,
                "config": config_dict(config),
                "assembly": {
                    "dynamic_branches": DYNAMIC_BRANCHES,
                    "ridge": 1e-3,
                    "constraints": "nonnegative simplex with explicit zero-delta persistence anchor",
                    "weights_fit_on": "60-80% source/validation only",
                    "test_or_destination_target_used_for_weights": False,
                },
                "support_audit": (
                    "diagnostic-only in v0.2; hard support gating is not frozen yet"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
