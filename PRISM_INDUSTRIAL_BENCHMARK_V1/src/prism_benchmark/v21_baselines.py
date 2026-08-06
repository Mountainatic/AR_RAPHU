from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cpu_data import ViewSpec, load_samples, sha256_file
from .cpu_selection import mse, regression_metrics
from .stage0 import write_json
from .v21_config import V21Paths, load_v21_config
from .v21_views import sru_dynamic_views, sru_input_views


INVENTORY_NAME = "FROZEN_BASELINE_INVENTORY.json"


def _views(paths: V21Paths) -> list[ViewSpec]:
    return [*sru_input_views(paths.shared), *sru_dynamic_views(paths.shared)]


def _registered_models(config: dict[str, Any], information_set: str) -> list[str]:
    key = "input_only" if information_set == "input_only" else "dynamic"
    return [
        str(model)
        for model in config["baselines"][key]
        if not str(model).startswith("PRISM_V2_1_")
    ]


def _prediction_candidates(
    root: Path,
    view: ViewSpec,
    split: str,
    model: str,
) -> list[Path]:
    candidates = []
    for path in root.rglob(f"{model}.parquet"):
        parts = set(path.parts)
        if (
            view.head.head_id in parts
            and split in parts
            and view.information_set in parts
            and view.availability_scenario in parts
            and view.proxy_policy in parts
        ):
            candidates.append(path)
    return sorted(candidates)


def _unique_prediction(
    root: Path,
    view: ViewSpec,
    split: str,
    model: str,
) -> Path:
    candidates = _prediction_candidates(root, view, split, model)
    if len(candidates) != 1:
        raise RuntimeError(
            "baseline prediction lookup must be unique: "
            f"head={view.head.head_id} information_set={view.information_set} "
            f"split={split} model={model} candidates={candidates}"
        )
    return candidates[0]


def _validate_prediction_frame(
    source: Path,
    expected: pd.DataFrame,
    model: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(source)
    required = {"sample_id", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"baseline frame lacks columns {sorted(missing)}: {source}")
    expected_ids = expected["view_sample_id"].astype(str).to_numpy()
    observed_ids = frame["sample_id"].astype(str).to_numpy()
    if not np.array_equal(observed_ids, expected_ids):
        raise RuntimeError(f"baseline sample_id mismatch: {source}")
    expected_y = expected["y_true"].to_numpy(dtype=np.float64)
    observed_y = frame["y_true"].to_numpy(dtype=np.float64)
    if not np.array_equal(observed_y, expected_y, equal_nan=True):
        raise RuntimeError(f"baseline y_true mismatch: {source}")
    if frame["y_pred"].isna().any():
        raise RuntimeError(f"baseline y_pred contains missing values: {source}")
    audit = {
        "model": model,
        "rows": len(frame),
        "source_sha256": sha256_file(source),
        "parameter_count": (
            None
            if "parameter_count" not in frame.columns
            else int(frame["parameter_count"].max())
        ),
        **regression_metrics(
            observed_y,
            frame["y_pred"].to_numpy(dtype=np.float64),
        ),
    }
    return frame, audit


def freeze_baseline_inventory(paths: V21Paths) -> dict[str, Any]:
    """Freeze baseline inclusion using validation only and opaque test hashes.

    Test parquet contents are not opened here.  Their paths and byte hashes are
    frozen so E7 can later validate sample IDs after the test guard opens.
    """
    if paths.baseline_root is None:
        raise RuntimeError("E6 requires --baseline-root with frozen predictions")
    root = paths.baseline_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    config = load_v21_config(paths.project)
    entries: list[dict[str, Any]] = []
    best: dict[str, str] = {}
    for view in _views(paths):
        validation = load_samples(paths.shared, view, "validation")
        view_entries = []
        for model in _registered_models(config, view.information_set):
            validation_path = _unique_prediction(root, view, "validation", model)
            test_path = _unique_prediction(root, view, "test", model)
            _, validation_audit = _validate_prediction_frame(
                validation_path,
                validation,
                model,
            )
            entry = {
                "target_head": view.head.head_id,
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "model": model,
                "validation_path": str(validation_path.relative_to(root)),
                "validation_sha256": sha256_file(validation_path),
                "validation_mse": float(validation_audit["mse"]),
                "test_path": str(test_path.relative_to(root)),
                "test_sha256": sha256_file(test_path),
                "test_contents_accessed": False,
            }
            entries.append(entry)
            view_entries.append(entry)
        key = "|".join(
            (
                view.head.head_id,
                view.information_set,
                view.availability_scenario,
                view.proxy_policy,
            )
        )
        best[key] = min(
            view_entries,
            key=lambda item: (float(item["validation_mse"]), str(item["model"])),
        )["model"]
    inventory = {
        "status": "BASELINE_INCLUSION_FROZEN",
        "baseline_root": str(root),
        "entries": entries,
        "best_by_validation": best,
        "test_accessed": False,
    }
    write_json(paths.output / "BASELINES" / INVENTORY_NAME, inventory)
    return inventory


def load_frozen_baseline_inventory(paths: V21Paths) -> dict[str, Any]:
    path = paths.output / "BASELINES" / INVENTORY_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if inventory.get("status") != "BASELINE_INCLUSION_FROZEN":
        raise RuntimeError("baseline inventory is not frozen")
    return inventory


def materialize_test_baselines(paths: V21Paths) -> list[dict[str, Any]]:
    inventory = load_frozen_baseline_inventory(paths)
    root = Path(inventory["baseline_root"])
    views = {
        (
            view.head.head_id,
            view.information_set,
            view.availability_scenario,
            view.proxy_policy,
        ): view
        for view in _views(paths)
    }
    audits = []
    for entry in inventory["entries"]:
        key = (
            entry["target_head"],
            entry["information_set"],
            entry["availability_scenario"],
            entry["proxy_policy"],
        )
        view = views[key]
        source = root / entry["test_path"]
        if sha256_file(source) != entry["test_sha256"]:
            raise RuntimeError(f"frozen baseline changed before E7: {source}")
        test = load_samples(paths.shared, view, "test")
        frame, audit = _validate_prediction_frame(source, test, entry["model"])
        destination = (
            paths.output
            / "BASELINES"
            / "test_predictions"
            / view.head.head_id
            / view.information_set
            / view.availability_scenario
            / view.proxy_policy
            / f"{entry['model']}.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False, compression="zstd")
        audits.append(
            {
                "status": "PASS",
                "target_head": view.head.head_id,
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "model": entry["model"],
                "prediction_path": str(destination.relative_to(paths.output)),
                "prediction_sha256": sha256_file(destination),
                "test_accessed": True,
                **audit,
            }
        )
    write_json(
        paths.output / "BASELINES" / "TEST_BASELINE_AUDIT.json",
        {"status": "PASS", "models": audits, "test_accessed": True},
    )
    return audits
