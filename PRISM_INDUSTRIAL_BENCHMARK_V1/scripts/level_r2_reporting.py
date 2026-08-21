"""Reporting-only Level-R2 reconstruction for the six-dataset extension."""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from prism_benchmark.cpu_data import BaseAccessor, load_heads, sha256_file
from prism_benchmark.level_reconstruction import (
    metric_bundle_delta_and_level,
    support_hash,
)
from prism_benchmark.six_dataset_reporting import PredictionSpec, prediction_specs
from prism_benchmark.stage0 import write_json


def _shared_root(
    run_root: Path,
    public_root: Path,
    spec: PredictionSpec,
) -> Path:
    if spec.scope == "public5":
        return public_root / "shared"
    if not spec.direction:
        raise ValueError("CZ prediction is missing direction")
    return run_root / "shared" / spec.direction


def _target_and_window(
    shared: Path,
    spec: PredictionSpec,
) -> tuple[str, int, int, str]:
    if spec.scope == "public5":
        for head in load_heads(shared, primary_only=False):
            if head.head_id == spec.target_head:
                return (
                    head.target,
                    int(head.w0_steps),
                    int(head.w_steps),
                    head.dataset,
                )
        raise KeyError(spec.target_head)
    return "crystal_diameter", 12, 12, "cz_czochralski"


def _metadata(shared: Path, spec: PredictionSpec) -> pd.DataFrame:
    path = (
        shared
        / "sample_ids"
        / spec.target_head
        / spec.information_set
        / spec.availability_scenario
        / spec.proxy_policy
        / f"{spec.split}.parquet"
    )
    columns = [
        "base_origin_id",
        "view_sample_id",
        "entity_id",
        "origin",
        "current_start",
        "current_stop_exclusive",
        "target_start",
        "target_stop_exclusive",
    ]
    return pd.read_parquet(path, columns=columns)


def _prediction(spec: PredictionSpec, dataset: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        spec.path,
        columns=[
            "sample_id",
            "base_origin_id",
            "dataset",
            "target_head",
            "split",
            "y_true",
            "y_pred",
            "model",
            "information_set",
            "availability_scenario",
            "proxy_policy",
        ],
    )
    expected = {
        "dataset": dataset,
        "target_head": spec.target_head,
        "split": spec.split,
        "model": spec.model,
        "information_set": spec.information_set,
        "availability_scenario": spec.availability_scenario,
        "proxy_policy": spec.proxy_policy,
    }
    for column, value in expected.items():
        observed = frame[column].astype(str).unique().tolist()
        if observed != [str(value)]:
            raise RuntimeError(
                f"STOP_PREDICTION_IDENTITY_MISMATCH: {column}={observed}, "
                f"expected={value!r}, path={spec.path}"
            )
    return frame.drop(columns=list(expected))


def _window_mean(
    accessor: BaseAccessor,
    samples: pd.DataFrame,
    target: str,
    start_column: str,
    stop_column: str,
    expected_width: int,
) -> np.ndarray:
    start = samples[start_column].to_numpy(dtype=np.int64)
    stop = samples[stop_column].to_numpy(dtype=np.int64)
    widths = stop - start
    if (
        len(widths) == 0
        or int(widths[0]) <= 0
        or not np.all(widths == widths[0])
        or not np.all(widths == expected_width)
    ):
        raise ValueError(
            f"registered window width mismatch: {start_column}; "
            f"observed={sorted(set(widths.tolist()))}, expected={expected_width}"
        )

    # Frozen C1 targets were materialized by target_change(), which computes
    # every window from a full-entity FP64 prefix sum.  Reuse that exact
    # numerical path: a direct gather(...).mean() is mathematically equivalent
    # but can differ by several e-9 after a long cumulative history.
    result = np.empty(len(samples), dtype=np.float64)
    entities = samples["entity_id"].astype(str).to_numpy()
    codes, labels = pd.factorize(entities, sort=False)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(labels))
    groups = np.split(order, np.cumsum(counts)[:-1])
    for entity_id, mask in zip(labels, groups, strict=True):
        dense_min, value_prefix, count_prefix = accessor._prefixes(
            str(entity_id), target
        )
        if dense_min != 0:
            raise ValueError(
                "registered window cannot reproduce the frozen full-entity "
                f"prefix path: entity={entity_id!r}, dense_min={dense_min}"
            )
        starts = start[mask] - dense_min
        stops = stop[mask] - dense_min
        dense_length = len(count_prefix) - 1
        if np.any(starts < 0) or np.any(stops > dense_length):
            raise ValueError(
                "registered window outside entity support: "
                f"entity={entity_id!r}, window={start_column}"
            )
        window_counts = count_prefix[stops] - count_prefix[starts]
        if np.any(window_counts != expected_width):
            raise ValueError(
                "registered window contains missing entity rows: "
                f"entity={entity_id!r}, window={start_column}"
            )
        if np.any(count_prefix[stops] != stops):
            raise ValueError(
                "entity prefix contains gaps before a registered window: "
                f"entity={entity_id!r}, window={start_column}"
            )
        result[mask] = (
            value_prefix[stops] - value_prefix[starts]
        ) / expected_width
    return result


def _registered_levels(
    accessor: BaseAccessor,
    metadata: pd.DataFrame,
    target: str,
    current_width: int,
    future_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    current = _window_mean(
        accessor,
        metadata,
        target,
        "current_start",
        "current_stop_exclusive",
        current_width,
    )
    future = _window_mean(
        accessor,
        metadata,
        target,
        "target_start",
        "target_stop_exclusive",
        future_width,
    )
    return current, future


def collect_level_r2(
    run_root: Path,
    public_root: Path,
    *,
    specs: Sequence[PredictionSpec] | None = None,
    known_prediction_sha256: Mapping[str, str] | None = None,
    require_common_support: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_specs = list(
        prediction_specs(run_root, public_root=public_root)
        if specs is None
        else specs
    )
    accessor_groups: dict[
        tuple[str, str, str, str, str, str, int, int], list[PredictionSpec]
    ] = defaultdict(list)
    for spec in selected_specs:
        shared = _shared_root(run_root, public_root, spec)
        target, w0_steps, w_steps, dataset = _target_and_window(shared, spec)
        accessor_groups[
            (
                str(shared.resolve()),
                target,
                dataset,
                spec.split,
                spec.scope,
                spec.target_head,
                w0_steps,
                w_steps,
            )
        ].append(spec)

    for accessor_key in sorted(accessor_groups):
        (
            shared_value,
            target,
            dataset,
            split,
            _scope,
            _target_head,
            w0_steps,
            w_steps,
        ) = accessor_key
        shared = Path(shared_value)
        # The frozen target_change() prefix runs over the complete entity, not
        # only the chronological partitions permitted as model inputs for this
        # evaluation split.  Reading every frozen base-data partition here is
        # reporting-only and is required to reproduce that numerical path;
        # model features, predictions, and selection remain untouched.
        accessor = BaseAccessor(shared, dataset, "ood", [target])
        view_groups: dict[tuple[str, str, str, str, str], list[PredictionSpec]] = (
            defaultdict(list)
        )
        for spec in accessor_groups[accessor_key]:
            view_groups[
                (
                    spec.direction or "",
                    spec.target_head,
                    spec.information_set,
                    spec.availability_scenario,
                    spec.proxy_policy,
                )
            ].append(spec)
        for view_key in sorted(view_groups):
            group = sorted(view_groups[view_key], key=lambda item: item.model)
            metadata = _metadata(shared, group[0])
            current, future = _registered_levels(
                accessor,
                metadata,
                target,
                w0_steps,
                w_steps,
            )
            metadata = metadata.assign(
                _current_level=current,
                _future_level_true=future,
            )
            for spec in group:
                prediction = _prediction(spec, dataset)
                if prediction.empty:
                    continue
                if prediction["sample_id"].duplicated().any():
                    raise RuntimeError(f"duplicate prediction sample IDs: {spec.path}")
                joined = prediction.merge(
                    metadata,
                    on="base_origin_id",
                    how="inner",
                    sort=False,
                    validate="one_to_one",
                )
                if len(joined) != len(prediction):
                    raise RuntimeError(f"prediction metadata mismatch: {spec.path}")
                if not np.array_equal(
                    joined["sample_id"].astype(str).to_numpy(),
                    joined["view_sample_id"].astype(str).to_numpy(),
                ):
                    raise RuntimeError(f"STOP_SAMPLE_ID_MISMATCH: {spec.path}")
                delta_true = joined["y_true"].to_numpy(dtype=np.float64)
                current_level = joined["_current_level"].to_numpy(dtype=np.float64)
                future_level_true = joined["_future_level_true"].to_numpy(
                    dtype=np.float64
                )
                try:
                    bundle = metric_bundle_delta_and_level(
                        delta_true,
                        joined["y_pred"].to_numpy(dtype=np.float64),
                        current_level,
                        future_level_true=future_level_true,
                    )
                except AssertionError as error:
                    target_error = (future_level_true - current_level) - delta_true
                    absolute_error = np.abs(target_error)
                    scale = np.maximum(np.abs(delta_true), 1e-300)
                    relative_error = absolute_error / scale
                    worst = int(np.argmax(absolute_error))
                    raise AssertionError(
                        f"{error}; path={spec.path}; target_head={spec.target_head}; "
                        f"information_set={spec.information_set}; split={spec.split}; "
                        f"model={spec.model}; max_abs_error={absolute_error[worst]:.17g}; "
                        f"relative_error_at_max_abs={relative_error[worst]:.17g}; "
                        f"sample_id={joined.iloc[worst]['sample_id']}"
                    ) from error
                sample_ids = joined["sample_id"].astype(str).tolist()
                hashed_support = support_hash(sample_ids)
                prediction_sha256 = (
                    known_prediction_sha256 or {}
                ).get(str(spec.path.resolve())) or sha256_file(spec.path)
                rows.append(
                    {
                        "scope": spec.scope,
                        "direction": spec.direction or "",
                        "dataset": dataset,
                        "target_head": spec.target_head,
                        "information_set": spec.information_set,
                        "availability_scenario": spec.availability_scenario,
                        "proxy_policy": spec.proxy_policy,
                        "split": spec.split,
                        "model": spec.model,
                        "rows": int(len(joined)),
                        "support_hash": hashed_support,
                        "sample_support_hash": hashed_support,
                        "prediction_path": str(spec.path),
                        "prediction_sha256": prediction_sha256,
                        "prediction_target_semantics": (
                            "REGISTERED_CHANGE_TARGET_VERIFIED"
                        ),
                        "level_target_semantics": bundle[
                            "level_target_semantics"
                        ],
                        "current_window_steps": w0_steps,
                        "future_window_steps": w_steps,
                        "same_prediction_error": True,
                        "different_target_variance": True,
                        "model_retrained": False,
                        "model_reselected": False,
                        "sample_support_changed": False,
                        "r2_level_reporting": "R2_LEVEL_RECONSTRUCTED",
                        "r2_delta": bundle["r2_delta"],
                        "r2_level_reconstructed": bundle[
                            "r2_level_reconstructed"
                        ],
                        "mse": bundle["mse"],
                        "rmse": bundle["rmse"],
                        "mae": bundle["mae"],
                        "r2_level_persistence": bundle[
                            "r2_level_persistence"
                        ],
                        "persistence_skill": bundle["persistence_skill"],
                        "std_level_target": bundle["std_level_target"],
                        "std_delta_target": bundle["std_delta_target"],
                        "variance_ratio": bundle["variance_ratio"],
                        "target_identity_max_abs_error": bundle[
                            "target_identity_max_abs_error"
                        ],
                        "residual_identity_max_abs_error": bundle[
                            "residual_identity_max_abs_error"
                        ],
                        "mse_identity_max_abs_error": abs(
                            bundle["mse"] - bundle["mse_delta"]
                        ),
                        "rmse_identity_max_abs_error": abs(
                            bundle["rmse"] - bundle["rmse_delta"]
                        ),
                        "mae_identity_max_abs_error": abs(
                            bundle["mae"] - bundle["mae_delta"]
                        ),
                    }
                )
                del prediction, joined
            del metadata, current, future
        del accessor
    if not rows:
        raise RuntimeError("no prediction rows available for Level-R2 reporting")
    frame = pd.DataFrame(rows).sort_values(
        ["scope", "direction", "target_head", "information_set", "split", "model"]
    )
    support_groups = [
        "scope",
        "direction",
        "target_head",
        "information_set",
        "availability_scenario",
        "proxy_policy",
        "split",
    ]
    support_audit = frame.groupby(support_groups, dropna=False).agg(
        support_hashes=("support_hash", "nunique"),
        row_counts=("rows", "nunique"),
    )
    support_mismatch = (support_audit["support_hashes"] != 1).any() or (
        support_audit["row_counts"] != 1
    ).any()
    if require_common_support and support_mismatch:
        raise RuntimeError("STOP_COMPARABLE_MODEL_SUPPORT_MISMATCH")
    audit = {
        "status": "PASS",
        "reporting_only": True,
        "model_retrained": False,
        "model_reselected": False,
        "test_rerun": False,
        "rows": int(len(frame)),
        "datasets": sorted(frame["dataset"].unique().tolist()),
        "tasks": sorted(frame["target_head"].unique().tolist()),
        "models": sorted(frame["model"].unique().tolist()),
        "identity_checks_passed": True,
        "common_support_groups": int(len(support_audit)),
        "common_support_contract_enforced": require_common_support,
        "common_support_checks_passed": (
            not support_mismatch if require_common_support else "NOT_REQUESTED"
        ),
        "target_identity_max_abs_error": float(
            frame["target_identity_max_abs_error"].max()
        ),
        "residual_identity_max_abs_error": float(
            frame["residual_identity_max_abs_error"].max()
        ),
        "identity_max_mse": float(frame["mse_identity_max_abs_error"].max()),
        "identity_max_rmse": float(frame["rmse_identity_max_abs_error"].max()),
        "identity_max_mae": float(frame["mae_identity_max_abs_error"].max()),
    }
    return frame.reset_index(drop=True), audit


def write_level_r2_outputs(run_root: Path, public_root: Path) -> dict[str, Any]:
    final = run_root / "final"
    final.mkdir(parents=True, exist_ok=True)
    frame, audit = collect_level_r2(run_root, public_root)
    path = final / "SIX_DATASET_LEVEL_R2_METRICS.csv"
    frame.to_csv(path, index=False)
    frame.to_csv(final / "PUBLIC_ALL_LEVEL_R2_METRICS.csv", index=False)
    audit["metrics_path"] = str(path)
    audit["metrics_sha256"] = sha256_file(path)
    write_json(final / "LEVEL_R2_RECONSTRUCTION_AUDIT.json", audit)
    return audit


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_level_r2_outputs(args.run_root, args.public_root)))
