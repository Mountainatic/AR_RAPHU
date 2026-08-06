from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .c2_models import run_job as run_c2_job
from .c3_models import run_job as run_c3_job
from .c4_prism import _run_channel, _run_joint_and_ablations
from .c5_models import _run_view as run_c5_view
from .c6_full_final import (
    _audit_path,
    _core_history,
    _evaluate_baseline_job,
    _evaluate_prism_job,
    _prediction_path,
)
from .cpu_data import ViewSpec, input_columns, load_samples, sha256_file
from .cpu_selection import regression_metrics
from .stage0 import write_json
from .v21_audit import compare_data_base_audits, data_base_audit, write_pre_audit
from .v21_config import (
    ACTIVE_HEADS,
    V21Paths,
    load_baseline_replay_amendment,
    load_v21_config,
)
from .v21_views import assert_only_sru, sru_dynamic_views, sru_input_views


INVENTORY_NAME = "FROZEN_BASELINE_INVENTORY.json"
REPLAY_MANIFEST_NAME = "BASELINE_REPLAY_MANIFEST.json"
REPLAY_STATUS = "BASELINE_REPLAY_FROZEN"
DEFAULT_BASELINE_WORKERS = 8

C2_INPUT_MODELS = ("RIDGE", "PLS", "DPLS", "RBF_SVR", "XGBOOST")
C2_DYNAMIC_MODELS = ("LOCAL_LINEAR_TREND",)
C3_DYNAMIC_MODELS = (
    "AR",
    "ARX",
    "LINEAR_NARX",
    "N4SID",
    "PARALLEL_HAMMERSTEIN",
    "HAMMERSTEIN_WIENER",
)
PRISM_INPUT_MODEL = "PRISM_CHANNEL_SPECIFIC"
PRISM_DYNAMIC_MODELS = ("PRISM_PHYSICS_FIRST", "PRISM_K_JOINT_AR")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head(project: Path) -> str:
    completed = subprocess.run(
        ["git", "-c", "core.filemode=false", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _worker_count() -> int:
    raw = os.environ.get("PRISM_V21_BASELINE_WORKERS", str(DEFAULT_BASELINE_WORKERS))
    try:
        workers = int(raw)
    except ValueError as error:
        raise RuntimeError("PRISM_V21_BASELINE_WORKERS must be an integer") from error
    if workers < 1 or workers > 24:
        raise RuntimeError("PRISM_V21_BASELINE_WORKERS must be in [1, 24]")
    return workers


def _view_key(view: ViewSpec) -> tuple[str, str, str, str]:
    return (
        view.head.head_id,
        view.information_set,
        view.availability_scenario,
        view.proxy_policy,
    )


def _head_pairs(paths: V21Paths) -> list[tuple[ViewSpec, ViewSpec]]:
    input_views = sru_input_views(paths.shared)
    dynamic_views = sru_dynamic_views(paths.shared)
    assert_only_sru(input_views)
    assert_only_sru(dynamic_views)
    if len(input_views) != len(ACTIVE_HEADS) or len(dynamic_views) != len(ACTIVE_HEADS):
        raise RuntimeError("baseline replay requires exactly two registered SRU head pairs")
    dynamic_by_context = {
        (view.head.head_id, view.availability_scenario, view.proxy_policy): view
        for view in dynamic_views
    }
    pairs = []
    for input_view in input_views:
        if input_view.head.h_steps != 5 or input_view.head.w_steps != 1:
            raise RuntimeError("baseline replay is restricted to SRU H5/W1 heads")
        key = (
            input_view.head.head_id,
            input_view.availability_scenario,
            input_view.proxy_policy,
        )
        dynamic_view = dynamic_by_context.get(key)
        if dynamic_view is None:
            raise RuntimeError(f"missing dynamic SRU view for {key}")
        pairs.append((input_view, dynamic_view))
    if {pair[0].head.head_id for pair in pairs} != ACTIVE_HEADS:
        raise RuntimeError("baseline replay SRU head set mismatch")
    return pairs


def baseline_replay_scope(paths: V21Paths) -> dict[str, Any]:
    """Return the immutable B0 scope without opening any prediction artifact."""
    pairs = _head_pairs(paths)
    return {
        "datasets": ["sru"],
        "heads": sorted(view.head.head_id for view, _ in pairs),
        "head_pairs": len(pairs),
        "information_sets": ["input_only", "dynamic"],
        "development_splits": ["train", "validation"],
        "materialized_splits": ["validation", "test"],
        "ood_splits": [],
        "historical_prediction_search": False,
        "external_baseline_root": False,
    }


def _registered_models(config: dict[str, Any], information_set: str) -> list[str]:
    key = "input_only" if information_set == "input_only" else "dynamic"
    return [str(model) for model in config["baselines"][key]]


def _validate_frozen_baseline_matrix(
    project: Path,
    v21_config: dict[str, Any],
) -> dict[str, Any]:
    c6_path = project / "configs" / "c6_full_final_v2.json"
    c6 = _read_json(c6_path)
    if c6.get("status") != "FROZEN_BEFORE_REPAIRED_BASELINE_TEST_ACCESS":
        raise RuntimeError("C6 full-final baseline contract is not frozen")
    for information_set, c6_key in (
        ("input_only", "input_only"),
        ("dynamic", "dynamic"),
    ):
        registered = _registered_models(v21_config, information_set)
        frozen = [str(model) for model in c6["leaderboards"][c6_key]]
        if registered != frozen:
            raise RuntimeError(
                f"v2.1 {information_set} baselines differ from frozen C6"
            )
    return c6


def _completed_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = _read_json(path)
    return result if result.get("status") in {"PASS", "FAILED_RETAINED"} else None


def _execute_resumable(
    function: Callable[..., dict[str, Any]],
    jobs: Iterable[tuple[Path, tuple[Any, ...]]],
    workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pending: list[tuple[Path, tuple[Any, ...]]] = []
    for result_path, arguments in jobs:
        previous = _completed_result(result_path)
        if previous is None:
            pending.append((result_path, arguments))
        else:
            results.append(previous)
    if not pending:
        return results
    if workers == 1:
        results.extend(function(*arguments) for _, arguments in pending)
        return results
    with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
        futures = {
            executor.submit(function, *arguments): result_path
            for result_path, arguments in pending
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _execute_calls(
    function: Callable[[tuple[Any, ...]], Any],
    jobs: list[tuple[Any, ...]],
    workers: int,
) -> list[Any]:
    if not jobs:
        return []
    if workers == 1:
        return [function(job) for job in jobs]
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = [executor.submit(function, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _development_manifest(
    path: Path,
    stage: str,
    results: list[dict[str, Any]],
    project: Path,
    workers: int,
) -> dict[str, Any]:
    passed = sum(result.get("status") == "PASS" for result in results)
    manifest = {
        "status": (
            "PASS" if passed == len(results) else "PASS_WITH_RETAINED_FAILURES"
        ),
        "stage": stage,
        "scope": "SRU_H5_W1_ONLY",
        "jobs": len(results),
        "passed": passed,
        "failed_retained": len(results) - passed,
        "workers": workers,
        "test_accessed": False,
        "cpu_model_freeze_sha256": sha256_file(
            project / "configs" / "cpu_model_freeze_v1.json"
        ),
    }
    write_json(path, manifest)
    return manifest


def _run_frozen_development(
    paths: V21Paths,
    pairs: list[tuple[ViewSpec, ViewSpec]],
    workers: int,
) -> dict[str, Path]:
    development = paths.baseline_replay_root / "DEVELOPMENT"
    c2_output = development / "C2"
    c3_output = development / "C3"
    c4_output = development / "C4"
    c5_output = development / "C5"

    c2_jobs = []
    for input_view, dynamic_view in pairs:
        for model in C2_INPUT_MODELS:
            result = c2_output / "PREDICTIONS" / model / input_view.relative_root / "RESULT.json"
            c2_jobs.append(
                (result, (paths.shared, paths.project, c2_output, input_view, model))
            )
        for model in C2_DYNAMIC_MODELS:
            result = c2_output / "PREDICTIONS" / model / dynamic_view.relative_root / "RESULT.json"
            c2_jobs.append(
                (result, (paths.shared, paths.project, c2_output, dynamic_view, model))
            )
    c2_results = _execute_resumable(run_c2_job, c2_jobs, workers)
    _development_manifest(
        c2_output / "C2_CLASSICAL_MANIFEST.json",
        "B0_C2_SRU_DEVELOPMENT",
        c2_results,
        paths.project,
        workers,
    )

    ar_jobs = []
    for _, dynamic_view in pairs:
        result = c3_output / "PREDICTIONS" / "AR" / dynamic_view.relative_root / "RESULT.json"
        ar_jobs.append(
            (result, (paths.shared, paths.project, c3_output, dynamic_view, "AR"))
        )
    c3_results = _execute_resumable(run_c3_job, ar_jobs, workers)
    remaining_c3_jobs = []
    for _, dynamic_view in pairs:
        for model in C3_DYNAMIC_MODELS[1:]:
            result = c3_output / "PREDICTIONS" / model / dynamic_view.relative_root / "RESULT.json"
            remaining_c3_jobs.append(
                (result, (paths.shared, paths.project, c3_output, dynamic_view, model))
            )
    c3_results.extend(
        _execute_resumable(run_c3_job, remaining_c3_jobs, workers)
    )
    _development_manifest(
        c3_output / "C3_MANIFEST.json",
        "B0_C3_SRU_DEVELOPMENT",
        c3_results,
        paths.project,
        workers,
    )

    channel_jobs = []
    for input_view, _ in pairs:
        for channel in input_columns(
            paths.shared,
            input_view.head.task_id,
            input_view.proxy_policy,
        ):
            result = (
                c4_output
                / "CHANNELS"
                / input_view.head.head_id
                / input_view.proxy_policy
                / channel
                / "RESULT.json"
            )
            channel_jobs.append(
                (
                    result,
                    (paths.shared, paths.project, c4_output, input_view, channel),
                )
            )
    c4_results = _execute_resumable(_run_channel, channel_jobs, workers)
    joint_jobs = []
    for input_view, _ in pairs:
        result = (
            c4_output
            / "JOINT"
            / input_view.head.head_id
            / input_view.proxy_policy
            / "RESULT.json"
        )
        joint_jobs.append(
            (
                result,
                (paths.shared, paths.project, c4_output, input_view),
            )
        )
    c4_results.extend(
        _execute_resumable(
            _run_joint_and_ablations,
            joint_jobs,
            min(workers, len(pairs)),
        )
    )
    _development_manifest(
        c4_output / "C4_MANIFEST.json",
        "B0_C4_SRU_PRISM_DEVELOPMENT",
        c4_results,
        paths.project,
        workers,
    )

    c5_jobs = []
    for _, dynamic_view in pairs:
        result = (
            c5_output
            / dynamic_view.head.head_id
            / dynamic_view.proxy_policy
            / "RESULT.json"
        )
        c5_jobs.append(
            (
                result,
                (
                    paths.shared,
                    paths.project,
                    c3_output,
                    c4_output,
                    c5_output,
                    dynamic_view,
                    1,
                ),
            )
        )
    c5_results = _execute_resumable(
        run_c5_view,
        c5_jobs,
        min(workers, len(pairs)),
    )
    _development_manifest(
        c5_output / "C5_MANIFEST.json",
        "B0_C5_SRU_PRISM_DYNAMIC_DEVELOPMENT",
        c5_results,
        paths.project,
        workers,
    )
    return {
        "c2": c2_output,
        "c3": c3_output,
        "c4": c4_output,
        "c5": c5_output,
    }


def _prediction_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _target_digest(values: np.ndarray) -> str:
    normalized = np.asarray(values, dtype="<f8")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _baseline_frame(
    samples: pd.DataFrame,
    view: ViewSpec,
    model: str,
    prediction: np.ndarray,
    parameter_count: int,
    split: str,
    core_history_steps: int,
) -> pd.DataFrame:
    required = [
        "view_sample_id",
        "base_origin_id",
        "dataset",
        "entity_id",
        "task_id",
        "target_head",
        "origin",
        "y_true",
    ]
    missing = set(required) - set(samples.columns)
    if missing:
        raise RuntimeError(f"sample frame lacks columns {sorted(missing)}")
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if len(prediction) != len(samples) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("baseline prediction length/non-finite mismatch")
    frame = samples[required].copy().rename(columns={"view_sample_id": "sample_id"})
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["base_origin_id"] = frame["base_origin_id"].astype(str)
    frame["entity_id"] = frame["entity_id"].astype(str)
    frame["split"] = split
    frame["y_pred"] = prediction
    frame["model"] = model
    frame["information_set"] = view.information_set
    frame["availability_scenario"] = view.availability_scenario
    frame["proxy_policy"] = view.proxy_policy
    frame["dtype"] = "float64"
    frame["parameter_count"] = int(parameter_count)
    frame["h_steps"] = int(view.head.h_steps)
    frame["w_steps"] = int(view.head.w_steps)
    frame["core_history_steps"] = int(core_history_steps)
    return frame


def _aligned_prediction(source: Path, expected: pd.DataFrame) -> tuple[np.ndarray, int]:
    frame = pd.read_parquet(source)
    required = {"sample_id", "base_origin_id", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"development prediction lacks {sorted(missing)}: {source}")
    expected_sample_ids = expected["view_sample_id"].astype(str).to_numpy()
    observed_sample_ids = frame["sample_id"].astype(str).to_numpy()
    if np.array_equal(observed_sample_ids, expected_sample_ids):
        aligned = frame.reset_index(drop=True)
    else:
        source_ids = frame["base_origin_id"].astype(str)
        expected_ids = expected["base_origin_id"].astype(str)
        if source_ids.duplicated().any() or expected_ids.duplicated().any():
            raise RuntimeError(f"non-unique base_origin_id prevents alignment: {source}")
        indexed = frame.assign(base_origin_id=source_ids).set_index("base_origin_id")
        if set(indexed.index) != set(expected_ids):
            raise RuntimeError(f"base_origin_id mismatch: {source}")
        aligned = indexed.loc[expected_ids].reset_index()
    expected_y = expected["y_true"].to_numpy(dtype=np.float64)
    observed_y = aligned["y_true"].to_numpy(dtype=np.float64)
    if not np.array_equal(observed_y, expected_y, equal_nan=True):
        raise RuntimeError(f"y_true mismatch after alignment: {source}")
    if aligned["y_pred"].isna().any():
        raise RuntimeError(f"y_pred contains missing values: {source}")
    parameter_count = 0
    if "parameter_count" in aligned.columns and aligned["parameter_count"].notna().any():
        parameter_count = int(aligned["parameter_count"].max())
    return aligned["y_pred"].to_numpy(dtype=np.float64), parameter_count


def _result_prediction(
    result_path: Path,
    result_root: Path,
    model: str,
) -> tuple[Path, dict[str, Any]]:
    result = _read_json(result_path)
    if result.get("status") != "PASS":
        raise RuntimeError(
            f"development contract unavailable: {model}: "
            f"{result.get('error', result.get('status'))}"
        )
    if result.get("prediction_path"):
        return result_root / result["prediction_path"], result
    for item in result.get("prediction_files", ()):
        if item.get("model") == model:
            return result_root / item["path"], result
    raise RuntimeError(f"development prediction missing for {model}: {result_path}")


def _development_prediction_source(
    roots: dict[str, Path],
    input_view: ViewSpec,
    dynamic_view: ViewSpec,
    view: ViewSpec,
    model: str,
) -> tuple[Path, dict[str, Any]]:
    if view.information_set == "input_only":
        if model in C2_INPUT_MODELS:
            result = roots["c2"] / "PREDICTIONS" / model / view.relative_root / "RESULT.json"
            return _result_prediction(result, roots["c2"], model)
        if model in {"PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"}:
            result = (
                roots["c3"]
                / "PREDICTIONS"
                / model
                / dynamic_view.relative_root
                / "RESULT.json"
            )
            return _result_prediction(result, roots["c3"], model)
        if model == PRISM_INPUT_MODEL:
            result = (
                roots["c4"]
                / "JOINT"
                / input_view.head.head_id
                / input_view.proxy_policy
                / "RESULT.json"
            )
            return _result_prediction(result, roots["c4"], model)
    else:
        if model in C2_DYNAMIC_MODELS:
            result = roots["c2"] / "PREDICTIONS" / model / view.relative_root / "RESULT.json"
            return _result_prediction(result, roots["c2"], model)
        if model in C3_DYNAMIC_MODELS:
            result = roots["c3"] / "PREDICTIONS" / model / view.relative_root / "RESULT.json"
            return _result_prediction(result, roots["c3"], model)
        if model in PRISM_DYNAMIC_MODELS:
            result = (
                roots["c5"]
                / dynamic_view.head.head_id
                / dynamic_view.proxy_policy
                / "RESULT.json"
            )
            return _result_prediction(result, roots["c5"], model)
    raise KeyError((view.information_set, model))


def _prediction_audit(
    path: Path,
    frame: pd.DataFrame,
    view: ViewSpec,
    model: str,
    split: str,
    *,
    include_metrics: bool,
    source_path: Path | None = None,
) -> dict[str, Any]:
    audit = {
        "status": "PASS",
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "model": model,
        "split": split,
        "rows": len(frame),
        "parameter_count": int(frame["parameter_count"].max()),
        "prediction_path": str(path),
        "prediction_sha256": sha256_file(path),
        "sample_id_sha256": _prediction_digest(frame["sample_id"].astype(str)),
        "y_true_sha256": _target_digest(frame["y_true"].to_numpy(dtype=np.float64)),
        "test_metrics_computed": False,
    }
    if source_path is not None:
        audit["development_source_path"] = str(source_path)
        audit["development_source_sha256"] = sha256_file(source_path)
    if include_metrics:
        audit.update(
            regression_metrics(
                frame["y_true"].to_numpy(dtype=np.float64),
                frame["y_pred"].to_numpy(dtype=np.float64),
            )
        )
    return audit


def _failure_audit(
    view: ViewSpec,
    model: str,
    split: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "status": "FAILED_RETAINED",
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
        "model": model,
        "split": split,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "test_metrics_computed": False,
    }


def _write_validation_predictions(
    paths: V21Paths,
    pairs: list[tuple[ViewSpec, ViewSpec]],
    roots: dict[str, Path],
    final_root: Path,
    core_histories: dict[str, int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    audits = []
    for input_view, dynamic_view in pairs:
        for view in (input_view, dynamic_view):
            validation = load_samples(paths.shared, view, "validation")
            train = load_samples(paths.shared, view, "train")
            for model in _registered_models(config, view.information_set):
                try:
                    source = None
                    if model == "PERSISTENCE":
                        prediction = np.zeros(len(validation), dtype=np.float64)
                        parameters = 0
                    elif model == "MEAN":
                        prediction = np.full(
                            len(validation),
                            float(train["y_true"].mean()),
                            dtype=np.float64,
                        )
                        parameters = 1
                    else:
                        source, _ = _development_prediction_source(
                            roots,
                            input_view,
                            dynamic_view,
                            view,
                            model,
                        )
                        prediction, parameters = _aligned_prediction(source, validation)
                    frame = _baseline_frame(
                        validation,
                        view,
                        model,
                        prediction,
                        parameters,
                        "validation",
                        core_histories[view.head.head_id],
                    )
                    path = _prediction_path(final_root, view, "validation", model)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(path, index=False, compression="zstd")
                    audit = _prediction_audit(
                        path,
                        frame,
                        view,
                        model,
                        "validation",
                        include_metrics=True,
                        source_path=source,
                    )
                    write_json(_audit_path(final_root, view, "validation", model), audit)
                    audits.append(audit)
                except Exception as error:
                    audit = _failure_audit(view, model, "validation", error)
                    write_json(_audit_path(final_root, view, "validation", model), audit)
                    audits.append(audit)
    return audits


def _canonicalize_test_result(
    paths: V21Paths,
    final_root: Path,
    view: ViewSpec,
    result: dict[str, Any],
    core_history_steps: int,
) -> dict[str, Any]:
    model = str(result["model"])
    if result.get("status") != "PASS":
        result = dict(result)
        result.update(
            {
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "test_metrics_computed": False,
            }
        )
        write_json(_audit_path(final_root, view, "test", model), result)
        return result
    samples = load_samples(paths.shared, view, "test")
    source = Path(result["prediction_path"])
    if not source.is_absolute():
        source = final_root / source
    prediction, parameters_from_frame = _aligned_prediction(source, samples)
    parameters = int(result.get("parameter_count", parameters_from_frame))
    frame = _baseline_frame(
        samples,
        view,
        model,
        prediction,
        parameters,
        "test",
        core_history_steps,
    )
    frame.to_parquet(source, index=False, compression="zstd")
    audit = _prediction_audit(
        source,
        frame,
        view,
        model,
        "test",
        include_metrics=False,
    )
    audit.update(
        {
            "fit_partition": "train_plus_validation_only",
            "evaluation_partition": "test",
            "baseline_replay_test_accessed": True,
            "v21_candidate_test_accessed": False,
            "test_metrics_computed": False,
            "test_metrics_exposed_to_selection": False,
        }
    )
    write_json(_audit_path(final_root, view, "test", model), audit)
    return audit


def _write_test_predictions(
    paths: V21Paths,
    pairs: list[tuple[ViewSpec, ViewSpec]],
    roots: dict[str, Path],
    final_root: Path,
    core_histories: dict[str, int],
    config: dict[str, Any],
    workers: int,
) -> list[dict[str, Any]]:
    regular_jobs: list[tuple[Any, ...]] = []
    prism_jobs: list[tuple[Any, ...]] = []
    for input_view, dynamic_view in pairs:
        core_history = core_histories[input_view.head.head_id]
        for model in _registered_models(config, "input_only"):
            if model != PRISM_INPUT_MODEL:
                regular_jobs.append(
                    (
                        paths.shared,
                        paths.project,
                        roots["c2"],
                        roots["c3"],
                        final_root,
                        input_view,
                        "test",
                        model,
                        core_history,
                    )
                )
        for model in _registered_models(config, "dynamic"):
            if model not in PRISM_DYNAMIC_MODELS:
                regular_jobs.append(
                    (
                        paths.shared,
                        paths.project,
                        roots["c2"],
                        roots["c3"],
                        final_root,
                        dynamic_view,
                        "test",
                        model,
                        core_history,
                    )
                )
        prism_jobs.append(
            (
                paths.shared,
                paths.project,
                roots["c3"],
                roots["c4"],
                roots["c5"],
                final_root,
                input_view,
                "test",
                core_history,
            )
        )
    results = _execute_calls(_evaluate_baseline_job, regular_jobs, workers)
    for value in _execute_calls(
        _evaluate_prism_job,
        prism_jobs,
        min(workers, len(prism_jobs)),
    ):
        results.extend(value)
    views = {
        (view.head.head_id, view.information_set): view
        for pair in pairs
        for view in pair
    }
    audits = []
    for result in results:
        key = (str(result["target_head"]), str(result["information_set"]))
        view = views[key]
        audits.append(
            _canonicalize_test_result(
                paths,
                final_root,
                view,
                result,
                core_histories[view.head.head_id],
            )
        )
    return audits


def _entry_key(value: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(value["target_head"]),
        str(value["information_set"]),
        str(value["availability_scenario"]),
        str(value["proxy_policy"]),
        str(value["model"]),
    )


def _relative_prediction(path: str, root: Path) -> str:
    value = Path(path).resolve()
    try:
        return value.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"baseline prediction escaped replay root: {value}") from error


def _build_inventory(
    paths: V21Paths,
    final_root: Path,
    validation_audits: list[dict[str, Any]],
    test_audits: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    validation = {_entry_key(item): item for item in validation_audits}
    test = {_entry_key(item): item for item in test_audits}
    entries = []
    failures = []
    best: dict[str, str] = {}
    for view in [*sru_input_views(paths.shared), *sru_dynamic_views(paths.shared)]:
        view_entries = []
        for model in _registered_models(config, view.information_set):
            key = (*_view_key(view), model)
            validation_item = validation.get(key)
            test_item = test.get(key)
            if (
                validation_item is None
                or validation_item.get("status") != "PASS"
                or test_item is None
                or test_item.get("status") != "PASS"
            ):
                failures.append(
                    {
                        "target_head": view.head.head_id,
                        "information_set": view.information_set,
                        "availability_scenario": view.availability_scenario,
                        "proxy_policy": view.proxy_policy,
                        "model": model,
                        "validation": validation_item,
                        "test": test_item,
                    }
                )
                continue
            entry = {
                "target_head": view.head.head_id,
                "information_set": view.information_set,
                "availability_scenario": view.availability_scenario,
                "proxy_policy": view.proxy_policy,
                "model": model,
                "validation_path": _relative_prediction(
                    validation_item["prediction_path"], final_root
                ),
                "validation_sha256": validation_item["prediction_sha256"],
                "validation_sample_id_sha256": validation_item["sample_id_sha256"],
                "validation_mse": float(validation_item["mse"]),
                "test_path": _relative_prediction(
                    test_item["prediction_path"], final_root
                ),
                "test_sha256": test_item["prediction_sha256"],
                "test_sample_id_sha256": test_item["sample_id_sha256"],
                "baseline_test_contents_accessed_during_b0": True,
                "test_metrics_computed": False,
                "test_metrics_exposed_to_selection": False,
            }
            entries.append(entry)
            view_entries.append(entry)
        if not view_entries:
            raise RuntimeError(f"no reproduced baseline survived for {_view_key(view)}")
        identifier = "|".join(_view_key(view))
        best[identifier] = min(
            view_entries,
            key=lambda item: (float(item["validation_mse"]), str(item["model"])),
        )["model"]
    return {
        "status": "BASELINE_INCLUSION_FROZEN",
        "provenance": "FROZEN_CODE_REPLAY_NO_HISTORICAL_CACHE",
        "baseline_root_relative_to_output": final_root.relative_to(paths.output).as_posix(),
        "entries": entries,
        "failures_retained": failures,
        "best_by_validation": best,
        "selection_inputs": "VALIDATION_ONLY",
        "baseline_replay_test_accessed": True,
        "v21_candidate_test_accessed": False,
        "test_metrics_computed": False,
        "test_metrics_exposed_to_selection": False,
    }


def run_baseline_replay(paths: V21Paths, workers: int | None = None) -> dict[str, Any]:
    """Run the frozen SRU baseline stack and freeze aligned validation/test files.

    This is the B0 subprocess. It never searches for historical predictions and
    never computes a test metric. Test parquet is generated only so its bytes can
    be frozen before v2.1 development starts.
    """
    started = time.time()
    workers = _worker_count() if workers is None else int(workers)
    if workers < 1 or workers > 24:
        raise RuntimeError("baseline replay workers must be in [1, 24]")
    config = load_v21_config(paths.project)
    amendment = load_baseline_replay_amendment(paths.project)
    _validate_frozen_baseline_matrix(paths.project, config)
    scope = baseline_replay_scope(paths)
    pairs = _head_pairs(paths)

    pre_path = paths.output / "DATA_AUDIT" / "V21_DATA_BASE_PRE_AUDIT.json"
    if pre_path.is_file():
        pre_audit = _read_json(pre_path)
    else:
        pre_audit = write_pre_audit(paths.shared, paths.output)

    roots = _run_frozen_development(paths, pairs, workers)
    core_histories = {
        input_view.head.head_id: _core_history(
            roots["c2"],
            roots["c3"],
            roots["c4"],
            roots["c5"],
            input_view,
            dynamic_view,
        )
        for input_view, dynamic_view in pairs
    }
    final_root = paths.baseline_replay_root / "FINAL_FIT"
    validation_audits = _write_validation_predictions(
        paths,
        pairs,
        roots,
        final_root,
        core_histories,
        config,
    )
    test_audits = _write_test_predictions(
        paths,
        pairs,
        roots,
        final_root,
        core_histories,
        config,
        workers,
    )

    post_baseline = data_base_audit(paths.shared)
    comparison = compare_data_base_audits(pre_audit, post_baseline)
    post_baseline["comparison_to_pre"] = comparison
    post_baseline_path = (
        paths.output / "DATA_AUDIT" / "V21_DATA_BASE_POST_BASELINE_AUDIT.json"
    )
    write_json(post_baseline_path, post_baseline)
    if comparison["status"] != "PASS":
        raise RuntimeError("STOP_DATA_BASE_MUTATED")

    inventory = _build_inventory(
        paths,
        final_root,
        validation_audits,
        test_audits,
        config,
    )
    inventory_path = paths.output / "BASELINES" / INVENTORY_NAME
    write_json(inventory_path, inventory)
    manifest = {
        "status": REPLAY_STATUS,
        "stage": "B0_FROZEN_SRU_BASELINE_REPLAY",
        "scope": scope,
        "historical_baseline_parquet": "NOT_AVAILABLE_NOT_SEARCHED_NOT_REQUIRED",
        "historical_prediction_search_performed": False,
        "external_baseline_root_used": False,
        "source_code_commit": _git_head(paths.project),
        "cpu_model_freeze_sha256": sha256_file(
            paths.project / "configs" / "cpu_model_freeze_v1.json"
        ),
        "c6_full_final_freeze_sha256": sha256_file(
            paths.project / "configs" / "c6_full_final_v2.json"
        ),
        "amendment_id": amendment["amendment_id"],
        "amendment_sha256": sha256_file(paths.baseline_amendment_path),
        "data_pre_audit_sha256": sha256_file(pre_path),
        "data_post_baseline_audit_sha256": sha256_file(post_baseline_path),
        "development_manifests": {
            name: {
                "path": str(path.relative_to(paths.output)),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "C2": roots["c2"] / "C2_CLASSICAL_MANIFEST.json",
                "C3": roots["c3"] / "C3_MANIFEST.json",
                "C4": roots["c4"] / "C4_MANIFEST.json",
                "C5": roots["c5"] / "C5_MANIFEST.json",
            }.items()
        },
        "inventory_path": str(inventory_path.relative_to(paths.output)),
        "inventory_sha256": sha256_file(inventory_path),
        "successful_baseline_contracts": len(inventory["entries"]),
        "failed_retained_contracts": len(inventory["failures_retained"]),
        "validation_prediction_files": sum(
            item.get("status") == "PASS" for item in validation_audits
        ),
        "test_prediction_files": sum(
            item.get("status") == "PASS" for item in test_audits
        ),
        "baseline_replay_test_accessed": True,
        "baseline_test_metrics_computed": False,
        "baseline_test_metrics_exposed_to_selection": False,
        "v21_candidate_test_accessed": False,
        "workers": workers,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = paths.output / "BASELINES" / REPLAY_MANIFEST_NAME
    write_json(manifest_path, manifest)
    return manifest


def _resolve_inside(base: Path, relative: str) -> Path:
    root = base.resolve()
    value = (root / relative).resolve()
    try:
        value.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"baseline path escaped frozen root: {relative}") from error
    return value


def freeze_baseline_inventory(paths: V21Paths) -> dict[str, Any]:
    """Verify the internal B0 inventory without searching or parsing test rows."""
    load_baseline_replay_amendment(paths.project)
    manifest_path = paths.output / "BASELINES" / REPLAY_MANIFEST_NAME
    inventory_path = paths.output / "BASELINES" / INVENTORY_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    manifest = _read_json(manifest_path)
    if manifest.get("status") != REPLAY_STATUS:
        raise RuntimeError("baseline replay manifest is not frozen")
    if manifest.get("historical_prediction_search_performed") is not False:
        raise RuntimeError("historical baseline search is forbidden")
    if manifest.get("external_baseline_root_used") is not False:
        raise RuntimeError("external baseline roots are forbidden")
    if manifest.get("inventory_sha256") != sha256_file(inventory_path):
        raise RuntimeError("frozen baseline inventory changed after B0")
    if manifest.get("amendment_sha256") != sha256_file(paths.baseline_amendment_path):
        raise RuntimeError("baseline replay amendment changed after B0")
    if manifest.get("cpu_model_freeze_sha256") != sha256_file(
        paths.project / "configs" / "cpu_model_freeze_v1.json"
    ):
        raise RuntimeError("CPU model freeze changed after B0")
    if manifest.get("c6_full_final_freeze_sha256") != sha256_file(
        paths.project / "configs" / "c6_full_final_v2.json"
    ):
        raise RuntimeError("C6 full-final freeze changed after B0")
    inventory = _read_json(inventory_path)
    if inventory.get("status") != "BASELINE_INCLUSION_FROZEN":
        raise RuntimeError("baseline inventory is not frozen")
    if inventory.get("test_metrics_computed") is not False:
        raise RuntimeError("B0 must not compute baseline test metrics")
    if inventory.get("test_metrics_exposed_to_selection") is not False:
        raise RuntimeError("B0 test values must not reach selection")
    root = _resolve_inside(
        paths.output,
        str(inventory["baseline_root_relative_to_output"]),
    )
    for entry in inventory.get("entries", ()):
        if entry.get("target_head") not in ACTIVE_HEADS:
            raise RuntimeError("non-SRU baseline escaped the frozen inventory")
        if entry.get("test_metrics_computed") is not False:
            raise RuntimeError("test metric leaked into baseline inventory")
        for split in ("validation", "test"):
            path = _resolve_inside(root, str(entry[f"{split}_path"]))
            if sha256_file(path) != entry[f"{split}_sha256"]:
                raise RuntimeError(f"frozen baseline changed after B0: {path}")
    return inventory


def load_frozen_baseline_inventory(paths: V21Paths) -> dict[str, Any]:
    return freeze_baseline_inventory(paths)


def _validate_prediction_frame(
    source: Path,
    expected: pd.DataFrame,
    model: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(source)
    required = {"sample_id", "entity_id", "origin", "y_true", "y_pred"}
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


def materialize_test_baselines(paths: V21Paths) -> list[dict[str, Any]]:
    inventory = load_frozen_baseline_inventory(paths)
    root = _resolve_inside(
        paths.output,
        str(inventory["baseline_root_relative_to_output"]),
    )
    views = {
        _view_key(view): view
        for view in [*sru_input_views(paths.shared), *sru_dynamic_views(paths.shared)]
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
        source = _resolve_inside(root, str(entry["test_path"]))
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
                "baseline_replay_test_accessed": True,
                "v21_candidate_test_accessed": True,
                "test_accessed": True,
                **audit,
            }
        )
    write_json(
        paths.output / "BASELINES" / "TEST_BASELINE_AUDIT.json",
        {
            "status": "PASS",
            "models": audits,
            "baseline_replay_test_accessed": True,
            "v21_candidate_test_accessed": True,
            "test_accessed": True,
        },
    )
    return audits
