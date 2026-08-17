from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq

from .cpu_data import ViewSpec, sha256_file
from .cpu_selection import regression_metrics
from .v2_runtime import release_process_memory
from .v211_public_all_baseline_materialization import (
    _cap_after_support,
    _development,
    _freeze,
    _not_run,
    _result,
    baseline_candidates,
    materialize_baseline_view,
)
from .v211_public_all_baselines import SupportRequirement
from .v211_public_all_closure import (
    FINAL_SUCCESS_STATUSES,
    _has_registered_split,
    common_support_record,
)
from .v211_public_all_config import PublicAllPaths, load_public_all_descriptor
from .v211_public_all_materialization import (
    materialize_dynamic_prism_view,
    materialize_input_prism_view,
    preflight_public_all_materialization,
)
from .v211_public_all_views import (
    public_all_dynamic_views,
    public_all_input_views,
)
from .v211_support import support_id_hash


_REQUIRED_PREDICTION_COLUMNS = {
    "sample_id",
    "base_origin_id",
    "dataset",
    "entity_id",
    "task_id",
    "target_head",
    "split",
    "origin",
    "y_true",
    "y_pred",
    "model",
    "information_set",
    "availability_scenario",
    "proxy_policy",
    "parameter_count",
    "dtype",
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git(project: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=project,
        text=True,
    ).strip()


def _hardlink(source: Path, destination: Path) -> dict[str, Any]:
    _require(source.is_file(), f"partial-resume source is not a file: {source}")
    _require(
        not destination.exists() and not destination.is_symlink(),
        f"partial-resume destination already exists: {destination}",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)
    source_stat = source.stat()
    destination_stat = destination.stat()
    _require(
        source_stat.st_dev == destination_stat.st_dev
        and source_stat.st_ino == destination_stat.st_ino,
        f"partial-resume artifact is not a hardlink: {destination}",
    )
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": source_stat.st_size,
        "sha256": sha256_file(source),
        "device": source_stat.st_dev,
        "inode": source_stat.st_ino,
        "destination_link_count": destination_stat.st_nlink,
    }


def _link_regular_files(
    source: Path,
    destination: Path,
    *,
    excluded_names: frozenset[str] = frozenset(),
    recursive: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not source.is_dir():
        return records
    candidates = source.rglob("*") if recursive else source.iterdir()
    for path in sorted(candidates):
        if not path.is_file() or path.name in excluded_names:
            continue
        records.append(_hardlink(path, destination / path.relative_to(source)))
    return records


def _historical_first_access(parent: PublicAllPaths) -> float:
    timestamps: list[float] = []
    repair_path = parent.freeze / "POST_FREEZE_MATERIALIZATION_REPAIR.json"
    if repair_path.is_file():
        repair = _read_json(repair_path)
        for item in repair.get("lockbox_failure_history", []):
            audit_path = Path(str(item.get("access_audit_path", "")))
            if audit_path.is_file():
                audit = _read_json(audit_path)
                if audit.get("first_access_timestamp") is not None:
                    timestamps.append(float(audit["first_access_timestamp"]))
    if parent.test_access_audit_path.is_file():
        audit = _read_json(parent.test_access_audit_path)
        if audit.get("first_access_timestamp") is not None:
            timestamps.append(float(audit["first_access_timestamp"]))
    _require(bool(timestamps), "partial-resume parent has no lockbox access timestamp")
    return min(timestamps)


def _parent_failure_entry(parent: PublicAllPaths) -> dict[str, Any]:
    access_path = parent.test_access_audit_path
    failure_path = parent.final / "LOCKBOX_ACCESSED_RUNTIME_FAILURE.json"
    access = _read_json(access_path)
    failure = _read_json(failure_path)
    _require(
        failure.get("status") == "LOCKBOX_ACCESSED_RUNTIME_FAILURE",
        "partial-resume parent does not retain a lockbox runtime failure",
    )
    _require(
        failure.get("ood_accessed") is False
        and failure.get("ood_y_read") is False,
        "partial-resume is only legal when parent OOD was not accessed",
    )
    return {
        "attempt": int(failure.get("lockbox_access_attempt", 4)),
        "namespace": str(parent.run_root),
        "materialization_commit": failure.get("materialization_commit"),
        "access_audit_path": str(access_path),
        "access_audit_sha256": sha256_file(access_path),
        "failure_path": str(failure_path),
        "failure_sha256": sha256_file(failure_path),
        "error_type": failure.get("error_type"),
        "error": failure.get("error"),
        "failed_model": failure.get("failed_model"),
        "failed_view": failure.get("failed_view"),
        "prism_test_views_materialized": failure.get(
            "prism_test_views_materialized"
        ),
        "baseline_test_parquets_materialized": failure.get(
            "baseline_test_parquets_materialized"
        ),
        "test_y_read": failure.get("test_y_read"),
        "ood_y_read": failure.get("ood_y_read"),
    }


def prepare_partial_resume(
    paths: PublicAllPaths,
    parent: PublicAllPaths,
    artifact_root: Path,
    *,
    materialization_commit: str,
    memory_repair_commit: str,
) -> dict[str, Any]:
    _require(
        not paths.run_root.exists(),
        f"partial-resume run root already exists: {paths.run_root}",
    )
    _require(
        not artifact_root.exists(),
        f"partial-resume artifact root already exists: {artifact_root}",
    )
    _require(
        _git(paths.project, "rev-parse", "HEAD") == materialization_commit,
        "partial-resume materialization commit is not current HEAD",
    )
    _require(
        not _git(paths.project, "status", "--short"),
        "partial-resume requires a clean Git worktree",
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", memory_repair_commit, "HEAD"],
        cwd=paths.project,
        check=False,
    )
    _require(
        ancestor.returncode == 0,
        "partial-resume memory repair commit is not an ancestor of HEAD",
    )
    parent_failure = _parent_failure_entry(parent)
    parent_freeze = _read_json(parent.development_freeze_path)
    _require(parent_freeze.get("status") == "FROZEN", "parent freeze is not FROZEN")
    _require(
        sha256_file(parent.development_freeze_path)
        == "e1d040c170da495888c37e3d45b39273b8bf6f3599b223868f141ab51e192ce7e",
        "partial-resume development freeze SHA mismatch",
    )

    paths.run_root.mkdir(parents=True)
    paths.freeze.mkdir()
    paths.final.mkdir()
    paths.logs.mkdir()
    paths.return_root.mkdir()
    artifact_root.mkdir(parents=True)

    os.symlink((parent.run_root / "shared").resolve(), paths.run_root / "shared")
    os.symlink((parent.run_root / "results").resolve(), paths.run_root / "results")
    (paths.final / "test_predictions").mkdir()
    routing: dict[str, Any] = {}
    for name in (
        "baseline_test_predictions",
        "ood_predictions",
        "baseline_ood_predictions",
    ):
        target = artifact_root / name
        target.mkdir()
        link = paths.final / name
        os.symlink(target, link)
        routing[name] = {
            "path": str(link),
            "target": str(target),
            "is_symlink": link.is_symlink(),
        }

    root_records = _link_regular_files(
        parent.run_root, paths.run_root, recursive=False
    )
    freeze_records = _link_regular_files(
        parent.freeze,
        paths.freeze,
        excluded_names=frozenset(
            {
                "POST_FREEZE_MATERIALIZATION_REPAIR.json",
                "R3_PRE_LOCKBOX_GATE.json",
            }
        ),
    )
    log_records = _link_regular_files(parent.logs, paths.logs)
    _hardlink(
        parent.freeze / "POST_FREEZE_MATERIALIZATION_REPAIR.json",
        paths.freeze / "PARENT_R3_POST_FREEZE_MATERIALIZATION_REPAIR.json",
    )
    _hardlink(
        parent.test_access_audit_path,
        paths.freeze / "ATTEMPT_4_PUBLIC_ALL_TEST_OOD_ACCESS_AUDIT.json",
    )
    _hardlink(
        parent.final / "LOCKBOX_ACCESSED_RUNTIME_FAILURE.json",
        paths.freeze / "ATTEMPT_4_LOCKBOX_ACCESSED_RUNTIME_FAILURE.json",
    )

    parent_repair = _read_json(
        parent.freeze / "POST_FREEZE_MATERIALIZATION_REPAIR.json"
    )
    history = list(parent_repair.get("lockbox_failure_history", []))
    history.append(parent_failure)
    repair = {
        **parent_repair,
        "status": "ACCEPTED_AUDITED_PARTIAL_RESUME",
        "evidence_class": (
            "POST_LOCKBOX_MATERIALIZATION_REPAIR_WITH_FROZEN_DEVELOPMENT_"
            "AND_VALIDATED_PARTIAL_ARTIFACT_REUSE"
        ),
        "repair_generation": 4,
        "lockbox_access_attempts": 5,
        "repair_parent_run_root": str(parent.run_root),
        "repair_run_root": str(paths.run_root),
        "materialization_repair_commit": materialization_commit,
        "previous_materialization_repair_commit": parent_repair.get(
            "materialization_repair_commit"
        ),
        "memory_materialization_fix_commit": memory_repair_commit,
        "partial_resume_orchestration_commit": materialization_commit,
        "lockbox_failure_history": history,
        "failed_attempt_predictions_reused": True,
        "failed_attempt_prediction_reuse_policy": (
            "SHA256_ROWS_SCHEMA_VIEW_SPLIT_MODEL_SUPPORT_METRICS_AND_"
            "FROZEN_DEVELOPMENT_CONTRACT_VALIDATION"
        ),
        "user_authorized_partial_resume": True,
        "user_authorized_evidence_downgrade": True,
        "post_test_reselection": False,
        "current_repair_test_accessed": False,
        "current_repair_ood_accessed": False,
        "created_at_unix": time.time(),
    }
    _write_json(paths.freeze / "POST_FREEZE_MATERIALIZATION_REPAIR.json", repair)
    routing_payload = {
        "status": "PASS",
        "reason": "PREDICTION_ARTIFACT_CAPACITY_ROUTING_ONLY",
        "created_at_unix": time.time(),
        "protocol_or_model_changed": False,
        "sample_support_changed": False,
        "failed_attempt_predictions_reused": True,
        "links": routing,
    }
    _write_json(paths.final / "ARTIFACT_STORAGE_ROUTING.json", routing_payload)

    gate = {
        "status": "PASS",
        "stage": "R4_PARTIAL_RESUME_PRE_LOCKBOX_GATE",
        "generated_at_unix": time.time(),
        "development_freeze_sha256": sha256_file(
            paths.development_freeze_path
        ),
        "materialization_commit": materialization_commit,
        "memory_repair_commit": memory_repair_commit,
        "repair_manifest_sha256": sha256_file(
            paths.freeze / "POST_FREEZE_MATERIALIZATION_REPAIR.json"
        ),
        "checks": {
            "access_attempt_is_five": True,
            "development_freeze_frozen": True,
            "parent_runtime_failure_retained": True,
            "parent_ood_not_accessed": True,
            "post_test_reselection_false": True,
            "user_authorized_partial_resume": True,
            "destination_prediction_roots_empty": True,
            "hardlink_only_reuse_required": True,
        },
        "metadata_hardlinks": {
            "root": len(root_records),
            "freeze": len(freeze_records) + 3,
            "logs": len(log_records),
        },
        "test_accessed": False,
        "ood_accessed": False,
    }
    _write_json(paths.freeze / "R4_PARTIAL_RESUME_PRE_LOCKBOX_GATE.json", gate)
    return gate


def _view_identity(view: ViewSpec) -> dict[str, str]:
    return {
        "dataset": view.head.dataset,
        "target_head": view.head.head_id,
        "information_set": view.information_set,
        "availability_scenario": view.availability_scenario,
        "proxy_policy": view.proxy_policy,
    }


def _expected_support(
    paths: PublicAllPaths, view: ViewSpec, split: str
) -> tuple[int, str]:
    record = common_support_record(paths, view)
    split_record = record.get("splits", {}).get(split)
    _require(
        isinstance(split_record, dict),
        f"leaderboard support lacks {split}: {view.relative_root}",
    )
    return int(split_record["rows"]), str(split_record["support_hash"])


def _logical_support_hash(path: Path) -> str:
    digest = hashlib.sha256()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=131_072,
        columns=["base_origin_id", "sample_id"],
        use_threads=False,
    ):
        origins = batch.column(0).to_pylist()
        samples = batch.column(1).to_pylist()
        for base_origin_id, sample_id in zip(origins, samples, strict=True):
            digest.update(str(base_origin_id).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(sample_id).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _physical_support_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    parquet = pq.ParquetFile(path)
    column_indices = {
        name: parquet.schema.names.index(name)
        for name in ("sample_id", "base_origin_id")
    }
    with path.open("rb") as handle:
        for row_group_index in range(parquet.metadata.num_row_groups):
            row_group = parquet.metadata.row_group(row_group_index)
            digest.update(str(row_group.num_rows).encode("ascii"))
            digest.update(b"\n")
            for name in ("sample_id", "base_origin_id"):
                chunk = row_group.column(column_indices[name])
                offsets = [
                    value
                    for value in (
                        chunk.dictionary_page_offset,
                        chunk.data_page_offset,
                    )
                    if value is not None and value >= 0
                ]
                _require(bool(offsets), f"parquet column has no page offset: {path}")
                handle.seek(min(offsets))
                payload = handle.read(chunk.total_compressed_size)
                _require(
                    len(payload) == chunk.total_compressed_size,
                    f"short parquet column read: {path}",
                )
                digest.update(name.encode("ascii"))
                digest.update(b"\0")
                digest.update(payload)
    return digest.hexdigest()


def _constant_column(path: Path, name: str) -> Any:
    parquet = pq.ParquetFile(path)
    index = parquet.schema.names.index(name)
    observed: Any = None
    initialized = False
    for row_group_index in range(parquet.metadata.num_row_groups):
        column = parquet.metadata.row_group(row_group_index).column(index)
        statistics = column.statistics
        if statistics is None or not statistics.has_min_max:
            values = (
                parquet.read_row_group(row_group_index, columns=[name])
                .column(0)
                .unique()
                .to_pylist()
            )
            _require(len(values) == 1, f"{name} is not constant in {path}")
            value = values[0]
        else:
            _require(
                statistics.min == statistics.max,
                f"{name} is not constant in {path}",
            )
            value = statistics.min
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not initialized:
            observed = value
            initialized = True
        else:
            _require(observed == value, f"{name} changes across row groups: {path}")
    _require(initialized, f"empty parquet has no {name}: {path}")
    return observed


def _prediction_metrics(path: Path) -> dict[str, float]:
    table = pq.read_table(
        path,
        columns=["y_true", "y_pred"],
        use_threads=False,
    )
    y_true = table.column("y_true").combine_chunks().to_numpy(
        zero_copy_only=False
    )
    y_pred = table.column("y_pred").combine_chunks().to_numpy(
        zero_copy_only=False
    )
    metrics = regression_metrics(y_true, y_pred)
    del table, y_true, y_pred
    release_process_memory()
    return metrics



def _metric_matches(expected: Any, observed: float) -> bool:
    try:
        expected_value = float(expected)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isclose(
            expected_value,
            observed,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    )


def _validate_prediction(
    path: Path,
    view: ViewSpec,
    model: str,
    split: str,
    expected_rows: int,
    expected_support_hash: str,
    *,
    expected_sha256: str | None = None,
    expected_audit: Mapping[str, Any] | None = None,
    canonical_support_fingerprint: str | None = None,
    compute_logical_support: bool = False,
) -> dict[str, Any]:
    _require(path.is_file(), f"prediction artifact is absent: {path}")
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema.names)
    _require(
        not _REQUIRED_PREDICTION_COLUMNS.difference(columns),
        f"prediction schema mismatch: {path}",
    )
    _require(
        parquet.metadata.num_rows == expected_rows,
        f"prediction row mismatch: {path}",
    )
    identity = {
        **_view_identity(view),
        "model": model,
        "split": split,
        "dtype": "float64",
    }
    for name, expected in identity.items():
        _require(
            str(_constant_column(path, name)) == str(expected),
            f"prediction {name} mismatch: {path}",
        )
    parameter_count = int(_constant_column(path, "parameter_count"))
    observed_sha256 = sha256_file(path)
    if expected_sha256 is not None:
        _require(
            observed_sha256 == expected_sha256,
            f"prediction SHA256 mismatch: {path}",
        )

    support_fingerprint = _physical_support_fingerprint(path)
    support_verification = "PHYSICAL_COLUMN_FINGERPRINT"
    logical_support_hash: str | None = None
    if compute_logical_support:
        logical_support_hash = _logical_support_hash(path)
        _require(
            logical_support_hash == expected_support_hash,
            f"prediction logical support hash mismatch: {path}",
        )
        support_verification = "LOGICAL_SUPPORT_HASH"
    elif (
        canonical_support_fingerprint is not None
        and support_fingerprint != canonical_support_fingerprint
    ):
        logical_support_hash = _logical_support_hash(path)
        _require(
            logical_support_hash == expected_support_hash,
            f"prediction support fingerprint and logical hash mismatch: {path}",
        )
        support_verification = "LOGICAL_SUPPORT_HASH_FALLBACK"

    metrics = _prediction_metrics(path)
    if expected_audit is not None:
        for field, expected in _view_identity(view).items():
            _require(
                str(expected_audit.get(field)) == expected,
                f"prediction audit {field} mismatch: {path}",
            )
        _require(
            expected_audit.get("model") == model
            and expected_audit.get("split") == split
            and int(expected_audit.get("rows", -1)) == expected_rows,
            f"prediction audit identity mismatch: {path}",
        )
        _require(
            expected_audit.get("scoring_support_hash")
            == expected_support_hash,
            f"prediction audit support mismatch: {path}",
        )
        _require(
            int(expected_audit.get("parameter_count", -1)) == parameter_count,
            f"prediction audit parameter count mismatch: {path}",
        )
        for name, value in metrics.items():
            _require(
                _metric_matches(expected_audit.get(name), value),
                f"prediction audit metric mismatch for {name}: {path}",
            )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": observed_sha256,
        "rows": expected_rows,
        "parameter_count": parameter_count,
        "support_hash": expected_support_hash,
        "logical_support_hash": logical_support_hash,
        "support_fingerprint": support_fingerprint,
        "support_verification": support_verification,
        "metrics": metrics,
    }


def _prism_result_name(view: ViewSpec) -> str:
    return (
        "PRISM_INPUT_RESULT.json"
        if view.information_set == "input_only"
        else "PRISM_DYNAMIC_RESULT.json"
    )


def _reuse_prism_test(
    paths: PublicAllPaths,
    parent: PublicAllPaths,
    views: list[ViewSpec],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    audits: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for view in views:
        expected_rows, expected_support_hash = _expected_support(
            paths, view, "test"
        )
        source_root = parent.final / "test_predictions" / view.relative_root
        source_result = source_root / _prism_result_name(view)
        payload = _read_json(source_result)
        _require(
            payload.get("status") == "PASS",
            f"parent PRISM result is not PASS: {source_result}",
        )
        model_audits = payload.get("models")
        _require(
            isinstance(model_audits, list) and bool(model_audits),
            f"parent PRISM result has no models: {source_result}",
        )
        canonical: str | None = None
        for index, audit in enumerate(model_audits):
            _require(
                isinstance(audit, dict) and audit.get("status") == "PASS",
                f"parent PRISM audit is not PASS: {source_result}",
            )
            model = str(audit.get("model"))
            expected_relative = (
                Path("final")
                / "test_predictions"
                / view.relative_root
                / f"{model}.parquet"
            )
            _require(
                Path(str(audit.get("prediction_path"))) == expected_relative,
                f"parent PRISM prediction path drift: {source_result}",
            )
            source = parent.run_root / expected_relative
            validation = _validate_prediction(
                source,
                view,
                model,
                "test",
                expected_rows,
                expected_support_hash,
                expected_sha256=str(audit.get("prediction_sha256")),
                expected_audit=audit,
                canonical_support_fingerprint=canonical,
                compute_logical_support=index == 0,
            )
            if canonical is None:
                canonical = str(validation["support_fingerprint"])
            destination = paths.run_root / expected_relative
            link = _hardlink(source, destination)
            records.append(
                {
                    **_view_identity(view),
                    "model": model,
                    "validation": validation,
                    "hardlink": link,
                }
            )
            audits.append(dict(audit))
        _require(canonical is not None, f"no PRISM support canonical: {view}")
        fingerprints[view.relative_root] = canonical
        _hardlink(
            source_result,
            paths.final
            / "test_predictions"
            / view.relative_root
            / _prism_result_name(view),
        )
    return audits, fingerprints, records


def _baseline_requirement_and_cap(
    paths: PublicAllPaths,
    model: str,
    result: Mapping[str, Any],
) -> tuple[SupportRequirement, int | None]:
    selection = result.get("selection", {})
    freeze = _freeze(paths.project)
    default_cap = int(freeze["selection"]["fit_row_cap_default"])
    if model in {"MEAN", "PERSISTENCE"}:
        return SupportRequirement(), None
    if model in {"RIDGE", "PLS", "RBF_SVR", "XGBOOST"}:
        cap_key = {
            "RBF_SVR": "fit_row_cap_svr",
            "XGBOOST": "fit_row_cap_xgboost",
        }.get(model, "fit_row_cap_default")
        return (
            SupportRequirement(input_history_steps=1),
            int(freeze["selection"][cap_key]),
        )
    if model == "DPLS":
        return (
            SupportRequirement(
                input_history_steps=int(selection["selected_history"])
            ),
            default_cap,
        )
    if model in {"PARALLEL_HAMMERSTEIN", "HAMMERSTEIN_WIENER"}:
        profile = tuple(int(value) for value in selection["selected_profile"])
        return SupportRequirement(input_history_steps=profile[1]), default_cap
    if model == "AR":
        profile = tuple(int(value) for value in selection["selected_profile"])
        return (
            SupportRequirement(
                target_delta_steps=profile[0],
                target_history_steps=profile[1],
            ),
            default_cap,
        )
    if model in {"ARX", "LINEAR_NARX"}:
        profile = tuple(int(value) for value in selection["selected_profile"])
        return (
            SupportRequirement(
                input_history_steps=profile[1],
                target_delta_steps=profile[0],
                target_history_steps=profile[1],
            ),
            default_cap,
        )
    raise KeyError(model)


def _native_fit_audit(
    paths: PublicAllPaths,
    view: ViewSpec,
    model: str,
    result: Mapping[str, Any],
    cache: dict[tuple[str, int, int, int, int | None], dict[str, Any]],
) -> dict[str, Any]:
    requirement, cap = _baseline_requirement_and_cap(paths, model, result)
    key = (
        view.relative_root,
        requirement.input_history_steps,
        requirement.target_delta_steps,
        requirement.target_history_steps,
        cap,
    )
    if key not in cache:
        fit = _development(paths.shared, view, [requirement])
        if cap is not None:
            fit = _cap_after_support(fit, cap)
        cache[key] = {
            "rows": len(fit),
            "support_hash": support_id_hash(fit),
            "requirement": {
                "input_history_steps": requirement.input_history_steps,
                "target_delta_steps": requirement.target_delta_steps,
                "target_history_steps": requirement.target_history_steps,
            },
            "cap": cap,
        }
        del fit
        release_process_memory()
    return cache[key]


def _baseline_selection(
    model: str, result: Mapping[str, Any]
) -> dict[str, Any]:
    selection = dict(result.get("selection", {}))
    if model in {"MEAN", "PERSISTENCE"}:
        selection["final_refit_partition"] = "train_plus_validation"
    return selection


def _reuse_baseline_test(
    paths: PublicAllPaths,
    parent: PublicAllPaths,
    views: list[ViewSpec],
    support_fingerprints: Mapping[str, str],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, set[str]],
    list[dict[str, Any]],
]:
    slots: dict[tuple[str, str], dict[str, Any]] = {}
    pending: dict[str, set[str]] = {}
    records: list[dict[str, Any]] = []
    native_cache: dict[
        tuple[str, int, int, int, int | None], dict[str, Any]
    ] = {}
    stable_parameter_models = {
        "MEAN",
        "PERSISTENCE",
        "RIDGE",
        "PLS",
        "DPLS",
        "AR",
        "ARX",
        "LINEAR_NARX",
        "PARALLEL_HAMMERSTEIN",
        "HAMMERSTEIN_WIENER",
    }
    for view in views:
        expected_rows, expected_support_hash = _expected_support(
            paths, view, "test"
        )
        for family, model, output_model in baseline_candidates(view):
            result = _result(paths, family, model, view)
            key = (view.relative_root, output_model)
            if result is None:
                slots[key] = _not_run(
                    view,
                    output_model,
                    "DEVELOPMENT_RESULT_ABSENT",
                    status="FAILED_RETAINED",
                    split="test",
                )
                continue
            development_status = str(result.get("status"))
            if development_status in {
                "NOT_RUN_IMPLEMENTATION_ABSENT",
                "NOT_RUN_PROTOCOL_INCOMPATIBLE",
            }:
                slots[key] = _not_run(
                    view,
                    output_model,
                    str(
                        result.get(
                            "reason",
                            f"DEVELOPMENT_STATUS_{development_status}",
                        )
                    ),
                    status=development_status,
                    split="test",
                )
                continue
            if development_status != "PASS":
                slots[key] = _not_run(
                    view,
                    output_model,
                    f"DEVELOPMENT_STATUS_{development_status}",
                    status="FAILED_RETAINED",
                    split="test",
                )
                continue
            if model == "N4SID":
                slots[key] = _not_run(
                    view,
                    output_model,
                    "N4SID_FINAL_REFIT_NOT_IMPLEMENTED",
                    status="NOT_RUN_PROTOCOL_INCOMPATIBLE",
                    split="test",
                )
                continue
            source = (
                parent.final
                / "baseline_test_predictions"
                / view.relative_root
                / f"{output_model}.parquet"
            )
            if not source.is_file():
                pending.setdefault(view.relative_root, set()).add(output_model)
                continue
            validation = _validate_prediction(
                source,
                view,
                output_model,
                "test",
                expected_rows,
                expected_support_hash,
                canonical_support_fingerprint=support_fingerprints[
                    view.relative_root
                ],
            )
            parameter_count = int(validation["parameter_count"])
            if (
                output_model in stable_parameter_models
                and result.get("parameter_count") is not None
            ):
                _require(
                    parameter_count == int(result["parameter_count"]),
                    f"baseline parameter count drift: {source}",
                )
            native = _native_fit_audit(
                paths, view, model, result, native_cache
            )
            destination = (
                paths.final
                / "baseline_test_predictions"
                / view.relative_root
                / f"{output_model}.parquet"
            )
            link = _hardlink(source, destination)
            audit = {
                "status": "PASS",
                **_view_identity(view),
                "model": output_model,
                "split": "test",
                "rows": expected_rows,
                "native_fit_rows": int(native["rows"]),
                "native_fit_support_hash": native["support_hash"],
                "scoring_support_hash": expected_support_hash,
                "parameter_count": parameter_count,
                "selection": _baseline_selection(model, result),
                "prediction_path": str(
                    destination.relative_to(paths.run_root)
                ),
                "prediction_sha256": validation["sha256"],
                "test_accessed": True,
                "ood_accessed": False,
                "fit_and_prediction_seconds": None,
                "elapsed_seconds": None,
                "recovered_materialization_timing": (
                    "NOT_RECORDED_BEFORE_PARENT_RUNTIME_FAILURE"
                ),
                "development_result_sha256": sha256_file(
                    paths.output
                    / "BASELINE_DEVELOPMENT"
                    / family
                    / "PREDICTIONS"
                    / model
                    / view.relative_root
                    / "RESULT.json"
                ),
                **validation["metrics"],
            }
            slots[key] = audit
            records.append(
                {
                    **_view_identity(view),
                    "model": output_model,
                    "validation": validation,
                    "native_fit": native,
                    "hardlink": link,
                }
            )
    return slots, pending, records


def _canonical_baseline_audits(
    views: list[ViewSpec],
    slots: Mapping[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for view in views:
        for _, _, output_model in baseline_candidates(view):
            key = (view.relative_root, output_model)
            _require(key in slots, f"baseline audit slot is absent: {key}")
            audits.append(dict(slots[key]))
    return audits


def _write_baseline_summary(
    paths: PublicAllPaths,
    split: str,
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    split_upper = split.upper()
    result = {
        "status": (
            "PASS"
            if audits
            and all(
                item.get("status") in FINAL_SUCCESS_STATUSES
                for item in audits
            )
            else "FAILED"
        ),
        "stage": f"T1_PUBLIC_ALL_BASELINE_{split_upper}_ACCESS",
        "models": audits,
        "test_accessed": split == "test",
        "ood_accessed": split == "ood",
    }
    _write_json(
        paths.final
        / f"baseline_{split}_predictions"
        / f"BASELINE_{split_upper}_RESULT.json",
        result,
    )
    return result



def _access_started_payload(
    paths: PublicAllPaths,
    parent: PublicAllPaths,
    *,
    materialization_preflight: Mapping[str, Any],
    registered_ood_views: int,
    materialization_commit: str,
) -> dict[str, Any]:
    freeze = _read_json(paths.development_freeze_path)
    descriptor = load_public_all_descriptor(paths.project)
    return {
        "status": "TEST_OOD_PARTIAL_RESUME_STARTED",
        "stage": "T1_PUBLIC_ALL_TEST_OOD_ACCESS",
        "first_access_timestamp": _historical_first_access(parent),
        "current_access_timestamp": time.time(),
        "generating_commit": freeze.get("generating_commit"),
        "materialization_commit": materialization_commit,
        "freeze_sha256": sha256_file(paths.development_freeze_path),
        "shared_sha256": freeze.get("shared_development_metadata_sha256"),
        "config_sha": descriptor.get("config_sha256"),
        "theory_sha": freeze.get("canonical_theory_sha256"),
        "repair_generation": 4,
        "lockbox_access_attempt": 5,
        "partial_resume": True,
        "parent_run_root": str(parent.run_root),
        "registered_ood_views": registered_ood_views,
        "test_accessed": True,
        "ood_accessed": False,
        "test_y_read": False,
        "ood_y_read": False,
        "post_test_reselection": False,
        "materialization_contract_preflight": dict(
            materialization_preflight
        ),
    }


def _status_counts(audits: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for audit in audits:
        status = str(audit.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def run_partial_resume(
    paths: PublicAllPaths,
    parent: PublicAllPaths,
    *,
    materialization_commit: str,
    expected_prism_views: int = 20,
    expected_reused_baselines: int = 129,
    expected_pending_baselines: int = 7,
) -> dict[str, Any]:
    freeze = _read_json(paths.development_freeze_path)
    _require(freeze.get("status") == "FROZEN", "test resume requires FROZEN")
    _require(
        _git(paths.project, "rev-parse", "HEAD") == materialization_commit,
        "partial-resume materialization commit is not current HEAD",
    )
    _require(
        not _git(paths.project, "status", "--short"),
        "partial-resume requires a clean Git worktree",
    )
    _require(
        not paths.test_access_audit_path.exists(),
        "partial-resume access audit already exists",
    )
    gate_path = paths.freeze / "R4_PARTIAL_RESUME_PRE_LOCKBOX_GATE.json"
    gate = _read_json(gate_path)
    _require(gate.get("status") == "PASS", "partial-resume gate is not PASS")

    input_views = list(public_all_input_views(paths.shared))
    dynamic_views = list(public_all_dynamic_views(paths.shared))
    all_views = [*input_views, *dynamic_views]
    ood_input_views = [
        view
        for view in input_views
        if _has_registered_split(paths, view, "ood")
    ]
    ood_dynamic_views = [
        view
        for view in dynamic_views
        if _has_registered_split(paths, view, "ood")
    ]
    ood_views = [*ood_input_views, *ood_dynamic_views]
    preflight = preflight_public_all_materialization(paths, dynamic_views)
    access = _access_started_payload(
        paths,
        parent,
        materialization_preflight=preflight,
        registered_ood_views=len(ood_views),
        materialization_commit=materialization_commit,
    )
    _write_json(paths.test_access_audit_path, access)

    progress: dict[str, Any] = {
        "status": "RUNNING",
        "stage": "R4_PARTIAL_RESUME",
        "started_at_unix": time.time(),
        "reused_prism_views": 0,
        "reused_prism_predictions": 0,
        "reused_baseline_predictions": 0,
        "pending_baseline_predictions": None,
        "completed_pending_baseline_predictions": 0,
        "ood_prism_predictions": 0,
        "ood_baseline_predictions": 0,
        "post_test_reselection": False,
    }
    progress_path = paths.logs / "R4_PARTIAL_RESUME_STATUS.json"
    _write_json(progress_path, progress)

    test_y_read = False
    ood_accessed = False
    ood_y_read = False
    prism_test_audits: list[dict[str, Any]] = []
    baseline_test_audits: list[dict[str, Any]] = []
    ood_audits: list[dict[str, Any]] = []
    reused_prism_records: list[dict[str, Any]] = []
    reused_baseline_records: list[dict[str, Any]] = []
    pending_count = 0
    try:
        test_y_read = True
        access["test_y_read"] = True
        _write_json(paths.test_access_audit_path, access)

        (
            prism_test_audits,
            support_fingerprints,
            reused_prism_records,
        ) = _reuse_prism_test(paths, parent, all_views)
        _require(
            len(all_views) == expected_prism_views,
            "partial-resume PRISM view count mismatch",
        )
        progress["reused_prism_views"] = len(all_views)
        progress["reused_prism_predictions"] = len(reused_prism_records)
        _write_json(progress_path, progress)

        slots, pending, reused_baseline_records = _reuse_baseline_test(
            paths,
            parent,
            all_views,
            support_fingerprints,
        )
        pending_count = sum(len(models) for models in pending.values())
        _require(
            len(reused_baseline_records) == expected_reused_baselines,
            "partial-resume reused baseline count mismatch: "
            f"expected={expected_reused_baselines} "
            f"observed={len(reused_baseline_records)}",
        )
        _require(
            pending_count == expected_pending_baselines,
            "partial-resume pending baseline count mismatch: "
            f"expected={expected_pending_baselines} observed={pending_count}",
        )
        progress["reused_baseline_predictions"] = len(
            reused_baseline_records
        )
        progress["pending_baseline_predictions"] = pending_count
        _write_json(progress_path, progress)

        reuse_audit = {
            "status": "PASS",
            "stage": "R4_VALIDATED_PARTIAL_ARTIFACT_REUSE",
            "parent_run_root": str(parent.run_root),
            "destination_run_root": str(paths.run_root),
            "validation_contract": (
                "SHA256_ROWS_SCHEMA_VIEW_SPLIT_MODEL_SUPPORT_METRICS_AND_"
                "FROZEN_DEVELOPMENT_CONTRACT_VALIDATION"
            ),
            "hardlink_only": True,
            "reused_prism_views": len(all_views),
            "reused_prism_predictions": len(reused_prism_records),
            "reused_baseline_predictions": len(reused_baseline_records),
            "pending_baseline_predictions": pending_count,
            "pending": {
                relative_root: sorted(models)
                for relative_root, models in sorted(pending.items())
            },
            "prism": reused_prism_records,
            "baselines": reused_baseline_records,
            "test_accessed": True,
            "ood_accessed": False,
            "post_test_reselection": False,
        }
        reuse_audit_path = (
            paths.freeze / "R4_PARTIAL_RESUME_ARTIFACT_REAUDIT.json"
        )
        _write_json(reuse_audit_path, reuse_audit)
        access["partial_resume_artifact_reaudit_sha256"] = sha256_file(
            reuse_audit_path
        )
        _write_json(paths.test_access_audit_path, access)

        view_map = {view.relative_root: view for view in all_views}
        for relative_root, models in pending.items():
            view = view_map[relative_root]
            new_audits = materialize_baseline_view(
                paths,
                view,
                split="test",
                models=models,
            )
            _require(
                len(new_audits) == len(models)
                and all(
                    audit.get("status") == "PASS"
                    for audit in new_audits
                ),
                f"pending baseline materialization failed: {relative_root}",
            )
            for audit in new_audits:
                slots[(relative_root, str(audit["model"]))] = audit
            progress["completed_pending_baseline_predictions"] += len(
                new_audits
            )
            progress["last_completed_pending_view"] = relative_root
            _write_json(progress_path, progress)

        _require(
            progress["completed_pending_baseline_predictions"]
            == pending_count,
            "not all pending baseline predictions completed",
        )
        baseline_test_audits = _canonical_baseline_audits(
            all_views, slots
        )
        baseline_test_summary = _write_baseline_summary(
            paths, "test", baseline_test_audits
        )
        _require(
            baseline_test_summary["status"] == "PASS",
            "baseline test summary is not PASS",
        )

        ood_accessed = bool(ood_views)
        ood_y_read = bool(ood_views)
        access["ood_accessed"] = ood_accessed
        access["ood_y_read"] = ood_y_read
        _write_json(paths.test_access_audit_path, access)
        for view in ood_input_views:
            values = materialize_input_prism_view(
                paths, view, split="ood"
            )
            ood_audits.extend(values)
            progress["ood_prism_predictions"] += len(values)
            progress["last_completed_ood_view"] = view.relative_root
            _write_json(progress_path, progress)
        for view in ood_dynamic_views:
            values = materialize_dynamic_prism_view(
                paths, view, split="ood"
            )
            ood_audits.extend(values)
            progress["ood_prism_predictions"] += len(values)
            progress["last_completed_ood_view"] = view.relative_root
            _write_json(progress_path, progress)

        baseline_ood_audits: list[dict[str, Any]] = []
        for view in ood_views:
            values = materialize_baseline_view(
                paths, view, split="ood"
            )
            baseline_ood_audits.extend(values)
            progress["ood_baseline_predictions"] += sum(
                audit.get("status") == "PASS" for audit in values
            )
            progress["last_completed_baseline_ood_view"] = (
                view.relative_root
            )
            _write_json(progress_path, progress)
        baseline_ood_summary = _write_baseline_summary(
            paths, "ood", baseline_ood_audits
        )
        _require(
            not ood_views or baseline_ood_summary["status"] == "PASS",
            "baseline OOD summary is not PASS",
        )
        ood_audits.extend(baseline_ood_audits)
    except Exception as error:
        failure = {
            "status": "LOCKBOX_ACCESSED_RUNTIME_FAILURE",
            "stage": "T1_PUBLIC_ALL_TEST_OOD_ACCESS",
            "repair_generation": 4,
            "lockbox_access_attempt": 5,
            "materialization_commit": materialization_commit,
            "error_type": type(error).__name__,
            "error": str(error),
            "test_accessed": True,
            "ood_accessed": ood_accessed,
            "test_y_read": test_y_read,
            "ood_y_read": ood_y_read,
            "reused_prism_views": progress["reused_prism_views"],
            "reused_prism_predictions": progress[
                "reused_prism_predictions"
            ],
            "reused_baseline_predictions": progress[
                "reused_baseline_predictions"
            ],
            "pending_baseline_predictions": pending_count,
            "completed_pending_baseline_predictions": progress[
                "completed_pending_baseline_predictions"
            ],
            "ood_prism_predictions": progress[
                "ood_prism_predictions"
            ],
            "ood_baseline_predictions": progress[
                "ood_baseline_predictions"
            ],
            "post_test_reselection": False,
        }
        _write_json(
            paths.final / "LOCKBOX_ACCESSED_RUNTIME_FAILURE.json",
            failure,
        )
        progress["status"] = "FAILED"
        progress["error_type"] = type(error).__name__
        progress["error"] = str(error)
        progress["finished_at_unix"] = time.time()
        _write_json(progress_path, progress)
        access.update(failure)
        _write_json(paths.test_access_audit_path, access)
        raise

    audits = [
        *prism_test_audits,
        *baseline_test_audits,
        *ood_audits,
    ]
    counts = _status_counts(audits)
    result = {
        **access,
        "status": (
            "PASS"
            if audits
            and all(
                audit.get("status") in FINAL_SUCCESS_STATUSES
                for audit in audits
            )
            else "FAILED"
        ),
        "models": audits,
        "status_counts": counts,
        "test_accessed": True,
        "ood_accessed": ood_accessed,
        "test_y_read": test_y_read,
        "ood_y_read": ood_y_read,
        "reused_prism_views": len(all_views),
        "reused_prism_predictions": len(reused_prism_records),
        "reused_baseline_predictions": len(reused_baseline_records),
        "new_baseline_test_predictions": pending_count,
        "post_test_reselection": False,
    }
    _require(result["status"] == "PASS", "partial resume result is not PASS")
    _write_json(paths.test_access_audit_path, result)

    repair_path = paths.freeze / "POST_FREEZE_MATERIALIZATION_REPAIR.json"
    repair = _read_json(repair_path)
    repair.update(
        {
            "status": "COMPLETED_AUDITED_PARTIAL_RESUME",
            "current_repair_test_accessed": True,
            "current_repair_ood_accessed": ood_accessed,
            "partial_resume_access_audit_sha256": sha256_file(
                paths.test_access_audit_path
            ),
            "partial_resume_reused_prism_views": len(all_views),
            "partial_resume_reused_baseline_predictions": len(
                reused_baseline_records
            ),
            "partial_resume_new_baseline_test_predictions": pending_count,
            "post_test_reselection": False,
            "completed_at_unix": time.time(),
        }
    )
    _write_json(repair_path, repair)

    progress["status"] = "PASS"
    progress["finished_at_unix"] = time.time()
    progress["access_audit_sha256"] = sha256_file(
        paths.test_access_audit_path
    )
    _write_json(progress_path, progress)
    return result
