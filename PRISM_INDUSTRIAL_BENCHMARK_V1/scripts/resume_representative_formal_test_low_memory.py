from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

import prism_benchmark.representative_baseline_checkpoints as baseline_module
from prism_benchmark.cz_l256_nowcast import DIRECTIONS, materialize_target_direction
from prism_benchmark.portable_checkpoints import (
    INFERENCE_ONLY_ENV,
    activate_inference_fit_guard,
)
from prism_benchmark.representative_formal import (
    ACTIVE_DATASETS,
    CHECKPOINT_MANIFEST_NAME,
    GLOBAL_FREEZE_NAME,
    PROTOCOL_ID,
    RESERVED_DATASETS,
    _assert_no_out_of_scope_artifacts,
    _formal_path_views,
    _rankings,
    _support_acceptance,
    _verify_checkpoint_inventory,
    checkpoint_namespace_root,
    free_gib,
    require_checkpoint_manifest,
    require_global_freeze,
)
from prism_benchmark.representative_prism_checkpoints import (
    predict_prism_checkpoint_for_view,
)
from prism_benchmark.cpu_data import sha256_file
from prism_benchmark.stage0 import write_json
from prism_benchmark.v211_representative_stage1_config import (
    load_representative_stage1_descriptor,
)


RECOVERY_SCHEMA = 1


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _view_key(view: Any) -> str:
    return "__".join(
        (
            view.head.head_id,
            view.information_set,
            view.availability_scenario,
            view.proxy_policy,
        )
    )


def _git_commit(project: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_output(recovery_root: Path, unit_id: str) -> Path:
    return recovery_root / "units" / f"{_safe_name(unit_id)}.json"


def _locate_view(project: Path, run_root: Path, namespace: str, view_key: str) -> tuple[Any, Any]:
    for candidate_namespace, paths, views in _formal_path_views(project, run_root):
        if candidate_namespace != namespace:
            continue
        for view in views:
            if _view_key(view) == view_key:
                return paths, view
    raise RuntimeError(f"STOP_LOW_MEMORY_UNKNOWN_VIEW:{namespace}:{view_key}")


def _worker(args: argparse.Namespace) -> None:
    if os.environ.get(INFERENCE_ONLY_ENV) != "1":
        raise RuntimeError("STOP_LOW_MEMORY_WORKER_NOT_INFERENCE_ONLY")
    activate_inference_fit_guard()
    require_global_freeze(args.run_root)
    require_checkpoint_manifest(args.run_root)
    paths, view = _locate_view(args.project, args.run_root, args.namespace, args.view_key)
    checkpoint_root = checkpoint_namespace_root(args.run_root / "checkpoints", args.namespace)
    if args.family == "prism":
        records = predict_prism_checkpoint_for_view(
            paths, view, checkpoint_root, split="test"
        )
    elif args.family == "baseline":
        candidates = list(baseline_module.baseline_candidates(view))
        selected = [item for item in candidates if str(item[2]) == args.model]
        if len(selected) != 1:
            raise RuntimeError(
                f"STOP_LOW_MEMORY_BASELINE_SELECTION:{args.namespace}:{args.view_key}:{args.model}"
            )
        original = baseline_module.baseline_candidates
        baseline_module.baseline_candidates = lambda _view: selected
        try:
            records = baseline_module.predict_baseline_checkpoints_for_view(
                paths, view, checkpoint_root, split="test"
            )
        finally:
            baseline_module.baseline_candidates = original
    else:
        raise AssertionError(args.family)
    write_json(
        args.output,
        {
            "status": "INFERENCE_PASS",
            "schema_version": RECOVERY_SCHEMA,
            "namespace": args.namespace,
            "namespace_run_root": str(paths.run_root.resolve()),
            "view_key": args.view_key,
            "family": args.family,
            "model": args.model,
            "records": records,
        },
    )


def _target_worker(args: argparse.Namespace) -> None:
    if os.environ.get(INFERENCE_ONLY_ENV) != "1":
        raise RuntimeError("STOP_LOW_MEMORY_TARGET_WORKER_NOT_INFERENCE_ONLY")
    activate_inference_fit_guard()
    require_global_freeze(args.run_root)
    require_checkpoint_manifest(args.run_root)
    audit = materialize_target_direction(
        args.raw_cz,
        args.run_root / "cz" / "shared",
        args.direction,
        global_freeze_path=args.run_root / "freeze" / GLOBAL_FREEZE_NAME,
        checkpoint_manifest_path=args.run_root / "freeze" / CHECKPOINT_MANIFEST_NAME,
    )
    write_json(args.output, {"status": "PASS", "audit": audit})


def _compact_prediction(
    *, formal_root: Path, namespace_root: Path, record: dict[str, Any]
) -> dict[str, Any]:
    relative = record.get("prediction_path")
    if record.get("status") != "PASS" or not relative:
        return record
    parquet = (namespace_root / str(relative)).resolve()
    allowed = (formal_root / ("public/final" if "public" in parquet.parts else "cz")).resolve()
    if not parquet.is_file() or allowed not in parquet.parents:
        raise RuntimeError(f"STOP_LOW_MEMORY_PREDICTION_PATH:{parquet}")
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet)
    rows = int(parquet_file.metadata.num_rows)
    if rows != int(record["rows"]):
        raise RuntimeError(f"STOP_LOW_MEMORY_ROW_MISMATCH:{parquet}")
    compact = parquet.with_suffix(".y_pred.fp64.npy")
    temporary = compact.with_suffix(compact.suffix + ".tmp")
    values = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float64, shape=(rows,)
    )
    offset = 0
    for batch in parquet_file.iter_batches(columns=["y_pred"], batch_size=262_144):
        current = batch.column(0).to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False
        )
        values[offset : offset + len(current)] = current
        offset += len(current)
    if offset != rows:
        raise RuntimeError(f"STOP_LOW_MEMORY_COMPACT_ROWS:{parquet}:{offset}:{rows}")
    values.flush()
    del values
    os.replace(temporary, compact)
    replay = np.load(compact, mmap_mode="r")
    if replay.dtype != np.float64 or replay.shape != (rows,):
        raise RuntimeError(f"STOP_LOW_MEMORY_COMPACT_RELOAD:{compact}")
    del replay
    original = {
        "path": str(parquet.relative_to(formal_root)),
        "bytes": parquet.stat().st_size,
        "sha256": _sha256(parquet),
    }
    compact_record = {
        **record,
        "original_prediction_artifact": original,
        "prediction_path": str(compact.relative_to(formal_root)),
        "prediction_sha256": _sha256(compact),
        "prediction_bytes": compact.stat().st_size,
        "prediction_storage": "NPY_FP64_Y_PRED_ONLY",
        "sample_identity_storage": "ROWS_AND_ORDER_HASH_IN_METHOD_RECORD",
    }
    audit = compact.with_suffix(compact.suffix + ".audit.json")
    write_json(
        audit,
        {
            "status": "PASS",
            "schema_version": RECOVERY_SCHEMA,
            "original": original,
            "compact": {
                "path": str(compact.relative_to(formal_root)),
                "bytes": compact.stat().st_size,
                "sha256": compact_record["prediction_sha256"],
                "dtype": "float64",
                "rows": rows,
            },
            "original_deleted_after_verified_compaction": True,
        },
    )
    parquet.unlink()
    gc.collect()
    return compact_record


def _run_child(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=PROJECT, env=env, check=True)


def _completed_unit_is_valid(formal_root: Path, output: Path) -> bool:
    if not output.is_file():
        return False
    try:
        envelope = json.loads(output.read_text(encoding="utf-8"))
        if envelope.get("status") != "COMPACT_PASS":
            return False
        for record in envelope.get("records", []):
            if record.get("status") != "PASS":
                continue
            artifact = formal_root / str(record["prediction_path"])
            if not artifact.is_file() or _sha256(artifact) != record["prediction_sha256"]:
                return False
        return True
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _main(args: argparse.Namespace) -> None:
    freeze = require_global_freeze(args.run_root)
    manifest = require_checkpoint_manifest(args.run_root)
    descriptor = load_representative_stage1_descriptor(args.project)
    minimum_free = float(descriptor["minimum_runtime_free_gib"])
    recovery_root = args.run_root / "final" / "low_memory_recovery"
    recovery_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[INFERENCE_ONLY_ENV] = "1"
    env["MALLOC_ARENA_MAX"] = "1"
    env["MALLOC_TRIM_THRESHOLD_"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    target_audits: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        output = recovery_root / "targets" / f"{direction}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.is_file():
            _run_child(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--target-worker",
                    "--project",
                    str(args.project),
                    "--run-root",
                    str(args.run_root),
                    "--raw-cz",
                    str(args.raw_cz),
                    "--direction",
                    direction,
                    "--output",
                    str(output),
                ],
                env,
            )
        target_audits.append(json.loads(output.read_text(encoding="utf-8"))["audit"])

    unit_files: list[Path] = []
    compaction_audits: list[dict[str, Any]] = []
    for namespace, paths, views in _formal_path_views(args.project, args.run_root):
        for view in views:
            view_key = _view_key(view)
            units: list[tuple[str, str]] = [("prism", "ALL")]
            units.extend(
                ("baseline", str(candidate[2]))
                for candidate in baseline_module.baseline_candidates(view)
            )
            for family, model in units:
                if free_gib(args.run_root.parent) < minimum_free + 1.0:
                    raise RuntimeError("STOP_LOW_STORAGE_DURING_LOW_MEMORY_TEST")
                unit_id = "__".join((namespace, view_key, family, model))
                output = _unit_output(recovery_root, unit_id)
                output.parent.mkdir(parents=True, exist_ok=True)
                if _completed_unit_is_valid(args.run_root, output):
                    unit_files.append(output)
                    continue
                _run_child(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--project",
                        str(args.project),
                        "--run-root",
                        str(args.run_root),
                        "--namespace",
                        namespace,
                        "--view-key",
                        view_key,
                        "--family",
                        family,
                        "--model",
                        model,
                        "--output",
                        str(output),
                    ],
                    env,
                )
                envelope = json.loads(output.read_text(encoding="utf-8"))
                compacted = [
                    _compact_prediction(
                        formal_root=args.run_root,
                        namespace_root=Path(envelope["namespace_run_root"]),
                        record=record,
                    )
                    for record in envelope["records"]
                ]
                envelope["records"] = compacted
                envelope["status"] = "COMPACT_PASS"
                write_json(output, envelope)
                for record in compacted:
                    if record.get("original_prediction_artifact"):
                        compaction_audits.append(
                            {
                                "namespace": namespace,
                                "view_key": view_key,
                                "model": record.get("model"),
                                "original": record["original_prediction_artifact"],
                                "compact_path": record["prediction_path"],
                                "compact_sha256": record["prediction_sha256"],
                            }
                        )
                unit_files.append(output)

    records: list[dict[str, Any]] = []
    for path in unit_files:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("status") != "COMPACT_PASS":
            raise RuntimeError(f"STOP_LOW_MEMORY_UNIT_NOT_COMPACT:{path}")
        records.extend(
            {"namespace": envelope["namespace"], **record}
            for record in envelope["records"]
        )
    support = _support_acceptance(records)
    _verify_checkpoint_inventory(args.run_root / "checkpoints", manifest["entries"])
    _assert_no_out_of_scope_artifacts(args.run_root)
    rankings = _rankings(records)
    report = {
        "status": "PASS",
        "stage": "FORMAL_TEST_INFERENCE_COMPLETE_LOW_MEMORY_RECOVERY",
        "protocol_id": PROTOCOL_ID,
        "primary_metric": "R2_LEVEL_RECONSTRUCTED",
        "secondary_metric": "R2_DELTA",
        "active_datasets": list(ACTIVE_DATASETS),
        "reserved_datasets": dict(RESERVED_DATASETS),
        "neural3_status": "NOT_RUN_BY_USER_SCOPE",
        "stage2_status": "NOT_RUN_BY_USER_SCOPE",
        "target_rod_access_audits": target_audits,
        "methods": records,
        "support_acceptance": support,
        "rankings": rankings,
        "checkpoint_entry_count": manifest["entry_count"],
        "checkpoints_unchanged_after_test": True,
        "fit_refit_select_called_in_test": False,
        "test_accessed_after_global_freeze_and_checkpoints": True,
        "ood_accessed": False,
        "low_memory_recovery": {
            "status": "PASS",
            "original_source_commit": manifest["source_commit"],
            "recovery_harness_commit": _git_commit(args.project),
            "unit_count": len(unit_files),
            "one_model_per_baseline_subprocess": True,
            "temporary_parquet_compaction": "NPY_FP64_Y_PRED_ONLY",
            "compaction_audit_count": len(compaction_audits),
            "global_selection_freeze_sha256": sha256_file(
                args.run_root / "freeze" / GLOBAL_FREEZE_NAME
            ),
            "checkpoint_manifest_sha256": sha256_file(
                args.run_root / "freeze" / CHECKPOINT_MANIFEST_NAME
            ),
        },
    }
    final_root = args.run_root / "final"
    write_json(final_root / "FORMAL_LEVEL_DELTA_REPORT.json", report)
    write_json(final_root / "INPUT_ONLY_LEADERBOARD.json", rankings["input_only"])
    write_json(final_root / "DYNAMIC_LEADERBOARD.json", rankings["dynamic"])
    write_json(recovery_root / "COMPACTION_AUDIT.json", compaction_audits)
    acceptance = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "formal_datasets": ["tep", "sru", "cz_czochralski"],
        "formal_dataset_count": 3,
        "neural3_artifacts_created": False,
        "reserved_datasets_accessed": False,
        "stage2_accessed": False,
        "first_target_access_after_freeze": all(
            item.get("target_rod_first_access_after_freeze") is True
            for item in target_audits
        ),
        "all_predictions_checkpoint_traceable": all(
            item.get("checkpoint_hash")
            for item in records
            if item.get("status") == "PASS"
        ),
        "checkpoint_hash_and_mtime_unchanged": True,
        "level_delta_identity_tolerance": 1e-10,
        "support_acceptance_status": "PASS",
        "low_memory_recovery_status": "PASS",
    }
    write_json(final_root / "FINAL_ACCEPTANCE.json", acceptance)
    write_json(
        recovery_root / "RECOVERY_STATUS.json",
        {
            "status": "PASS",
            "safe_stop_source_status": "FAILED_SAFE_STOP",
            "final_acceptance": acceptance,
            "freeze_manifest_hash": freeze["development_manifest_sha256"],
            "free_gib_after": free_gib(args.run_root.parent),
        },
    )
    status_path = args.run_root / "logs" / "FORMAL_LAUNCH_STATUS.json"
    launch_status = json.loads(status_path.read_text(encoding="utf-8"))
    launch_status["prior_safe_stop"] = {
        "status": launch_status.get("status"),
        "error": launch_status.get("error"),
        "failed_utc": launch_status.get("failed_utc"),
    }
    launch_status["status"] = "PASS_RECOVERED_LOW_MEMORY"
    launch_status["active_stage"] = "complete"
    launch_status["low_memory_recovery_status"] = "PASS"
    write_json(status_path, launch_status)
    print(json.dumps(acceptance, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-cz", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--target-worker", action="store_true")
    parser.add_argument("--namespace")
    parser.add_argument("--view-key")
    parser.add_argument("--family", choices=("prism", "baseline"))
    parser.add_argument("--model", default="ALL")
    parser.add_argument("--direction", choices=DIRECTIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.project = args.project.resolve()
    args.run_root = args.run_root.resolve()
    if args.raw_cz is not None:
        args.raw_cz = args.raw_cz.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    if args.worker:
        _worker(args)
    elif args.target_worker:
        _target_worker(args)
    else:
        if args.raw_cz is None or not args.raw_cz.is_file():
            raise RuntimeError("--raw-cz must identify the frozen CZ workbook")
        _main(args)


if __name__ == "__main__":
    main()
