"""Execute the frozen CZ raw-2s H4 authority E1--E6 plan.

This runner deliberately separates the small E1 no-refit completion from the
storage-intensive E2--E6 campaign.  It never writes into the sealed R3 anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pandas as pd


PLAN_RELATIVE_PATH = Path(
    "configs/cz_raw2s_h4_authority_e1_e6_rerun_plan_20260922.json"
)
STREAMING_AMENDMENT_RELATIVE_PATH = Path(
    "configs/cz_raw2s_h4_authority_e2_e6_streaming_amendment_20260924.json"
)
LEGACY_RUNNER_RELATIVE_PATH = Path("scripts/run_cz_raw2s_e1_e6.py")
MINIMUM_FULL_PRIVATE_GIB = 300
MINIMUM_STREAMING_PRIVATE_GIB = 8
MINIMUM_STREAMING_SCRATCH_GIB = 8
MAXIMUM_STREAMING_UNIT_PRIVATE_GIB = 4
MAXIMUM_STREAMING_UNIT_SCRATCH_GIB = 12
DIRECTIONS = ("Rod_1_to_Rod_2", "Rod_2_to_Rod_1")
H_STEPS = 4


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"STOP_CANNOT_IMPORT:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _nearest_existing(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _private_path_audit(path: Path) -> None:
    lowered = str(path.resolve()).replace("\\", "/").lower()
    if "/autodl-pub" in lowered:
        raise RuntimeError("STOP_PRIVATE_CZ_OUTPUT_ON_PUBLIC_MOUNT")


def preflight(project: Path, anchor: Path, run_root: Path) -> dict[str, Any]:
    _private_path_audit(run_root)
    plan_path = project / PLAN_RELATIVE_PATH
    if not plan_path.is_file():
        raise RuntimeError("STOP_FROZEN_PLAN_MISSING")
    if not (anchor / "final" / "INDEPENDENT_EXTENSION_REPORT.json").is_file():
        raise RuntimeError("STOP_SEALED_R3_ANCHOR_REPORT_MISSING")
    legacy = _load_module(
        project / LEGACY_RUNNER_RELATIVE_PATH, "cz_authority_legacy_audit"
    )
    authority = legacy.authority_audit(project)
    usage = shutil.disk_usage(_nearest_existing(run_root))
    free_gib = usage.free / (1024**3)
    result = {
        "status": "PASS",
        "stage": "P0_PLAN_AND_CODE_FREEZE",
        "created_utc": _utc(),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "anchor_run": str(anchor),
        "anchor_report_sha256": _sha256(
            anchor / "final" / "INDEPENDENT_EXTENSION_REPORT.json"
        ),
        "authority": authority,
        "private_output_path": str(run_root),
        "private_path_audit": "PASS",
        "private_free_gib": free_gib,
        "minimum_full_private_gib": MINIMUM_FULL_PRIVATE_GIB,
        "e1_no_refit_execution_allowed": free_gib >= 1.0,
        "e2_e6_full_execution_gate": (
            "PASS" if free_gib >= MINIMUM_FULL_PRIVATE_GIB else "BLOCKED_INSUFFICIENT_STORAGE"
        ),
        "raw_workbook_copy_forbidden": True,
    }
    if not result["e1_no_refit_execution_allowed"]:
        raise RuntimeError("STOP_INSUFFICIENT_STORAGE_FOR_E1_ADAPTER")
    _write_json(run_root / "PLAN_FREEZE.json", result)
    return result


def full_storage_gate(run_root: Path) -> dict[str, Any]:
    _private_path_audit(run_root)
    usage = shutil.disk_usage(_nearest_existing(run_root))
    free_gib = usage.free / (1024**3)
    result = {
        "status": "PASS" if free_gib >= MINIMUM_FULL_PRIVATE_GIB else "BLOCKED",
        "free_gib": free_gib,
        "required_gib": MINIMUM_FULL_PRIVATE_GIB,
        "path": str(run_root),
        "checked_utc": _utc(),
    }
    _write_json(run_root / "FULL_STORAGE_GATE.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(
            f"STOP_INSUFFICIENT_PRIVATE_STORAGE:{free_gib:.3f}GiB<{MINIMUM_FULL_PRIVATE_GIB}GiB"
        )
    return result


def streaming_storage_gate(
    project: Path, run_root: Path, scratch_root: Path
) -> dict[str, Any]:
    """Authorize the separately documented bounded-retention execution mode.

    This is not a silent relaxation of the frozen 300 GiB full-retention gate:
    the immutable parent plan and its gate remain recorded.  The amendment is
    hashed into every result, and only one isolated unit is authorized by this
    check.  Parallel execution has its own worker-equivalence prerequisite.
    """

    _private_path_audit(run_root)
    amendment_path = project / STREAMING_AMENDMENT_RELATIVE_PATH
    if not amendment_path.is_file():
        raise RuntimeError("STOP_STREAMING_EXECUTION_AMENDMENT_MISSING")
    amendment = _read_json(amendment_path)
    if (
        amendment.get("status") != "EXECUTION_AMENDMENT_ACTIVE"
        or amendment.get("statistical_protocol_changed") is not False
        or amendment.get("model_or_selector_changed") is not False
    ):
        raise RuntimeError("STOP_INVALID_STREAMING_EXECUTION_AMENDMENT")
    private_usage = shutil.disk_usage(_nearest_existing(run_root))
    scratch_usage = shutil.disk_usage(_nearest_existing(scratch_root))
    private_free_gib = private_usage.free / (1024**3)
    scratch_free_gib = scratch_usage.free / (1024**3)
    private_required_gib = (
        MINIMUM_STREAMING_PRIVATE_GIB + MAXIMUM_STREAMING_UNIT_PRIVATE_GIB
    )
    scratch_required_gib = (
        MINIMUM_STREAMING_SCRATCH_GIB + MAXIMUM_STREAMING_UNIT_SCRATCH_GIB
    )
    passed = (
        private_free_gib >= private_required_gib
        and scratch_free_gib >= scratch_required_gib
    )
    result = {
        "status": "PASS" if passed else "BLOCKED",
        "mode": "STREAMING_BOUNDED_RETENTION",
        "checked_utc": _utc(),
        "parent_full_retention_gate_gib": MINIMUM_FULL_PRIVATE_GIB,
        "parent_full_retention_gate_status": "NOT_APPLICABLE_TO_STREAMING_MODE",
        "amendment_path": str(amendment_path),
        "amendment_sha256": _sha256(amendment_path),
        "statistical_protocol_changed": False,
        "private_path": str(run_root),
        "private_free_gib": private_free_gib,
        "private_low_watermark_gib": MINIMUM_STREAMING_PRIVATE_GIB,
        "maximum_private_unit_gib": MAXIMUM_STREAMING_UNIT_PRIVATE_GIB,
        "private_required_before_unit_gib": private_required_gib,
        "scratch_path": str(scratch_root),
        "scratch_free_gib": scratch_free_gib,
        "scratch_low_watermark_gib": MINIMUM_STREAMING_SCRATCH_GIB,
        "maximum_scratch_unit_gib": MAXIMUM_STREAMING_UNIT_SCRATCH_GIB,
        "scratch_required_before_unit_gib": scratch_required_gib,
        "authorized_outer_units": 1,
        "two_way_parallelism": "BLOCKED_PENDING_WORKER_EQUIVALENCE",
        "tiny_pilot_parallelism": "BLOCKED_PENDING_RESOURCE_CERTIFICATE",
    }
    _write_json(run_root / "STREAMING_STORAGE_GATE.json", result)
    if not passed:
        raise RuntimeError(
            "STOP_INSUFFICIENT_STREAMING_STORAGE:"
            f"private={private_free_gib:.3f}/{private_required_gib}GiB,"
            f"scratch={scratch_free_gib:.3f}/{scratch_required_gib}GiB"
        )
    status_path = run_root / "RUN_STATUS.json"
    run_status = _read_json(status_path) if status_path.is_file() else {}
    run_status.update(
        {
            "status": "PARTIAL",
            "E2_E6": "STREAMING_EXECUTION_AUTHORIZED",
            "execution_mode": "STREAMING_BOUNDED_RETENTION",
            "next_stage": "P1_E2_AUTHORITY_ADAPTER_TINY_PILOT",
            "reason": (
                "The 300 GiB full-retention gate remains unsatisfied; the "
                "separately hashed execution amendment authorizes one isolated "
                "unit after the bounded-retention gate passes."
            ),
            "updated_utc": _utc(),
        }
    )
    _write_json(status_path, run_status)
    return result


def _authority_runner(project: Path) -> ModuleType:
    legacy = _load_module(project / LEGACY_RUNNER_RELATIVE_PATH, "cz_authority_e1_wrapper")
    return legacy._load_authority_runner(project)


def _anchor_paths(project: Path, anchor: Path, direction: str) -> tuple[Any, Any, Path]:
    from prism_benchmark.v211_public_all_config import PublicAllPaths

    runner = _authority_runner(project)
    shared, _, dynamic_view, checkpoint_root = runner._cz_paths(
        anchor, H_STEPS, direction
    )
    paths = PublicAllPaths(
        project,
        shared,
        anchor / "cz" / "h4" / "directions" / direction,
    )
    return paths, dynamic_view, checkpoint_root


def _checked_directory_symlink(destination: Path, source: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise RuntimeError(f"STOP_CORRECTION_INPUT_BINDING_DRIFT:{destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def _corrected_authority_replay(
    project: Path,
    anchor_paths: Any,
    view: Any,
    direction: str,
    run_root: Path,
) -> tuple[Any, Path, list[dict[str, Any]]]:
    """Refit only the sealed final checkpoint, then replay formal test once."""

    from prism_benchmark.portable_checkpoints import INFERENCE_ONLY_ENV
    from prism_benchmark.representative_prism_checkpoints import (
        _checkpoint_dir,
        fit_prism_checkpoint_for_view,
        predict_prism_checkpoint_for_view,
    )
    from prism_benchmark.v211_public_all_config import PublicAllPaths

    correction_direction = run_root / "CORRECTED_AUTHORITY" / "directions" / direction
    _checked_directory_symlink(correction_direction / "results", anchor_paths.output)
    _checked_directory_symlink(correction_direction / "freeze", anchor_paths.freeze)
    paths = PublicAllPaths(project, anchor_paths.shared, correction_direction)
    checkpoint_root = run_root / "CORRECTED_AUTHORITY" / "checkpoints" / direction
    checkpoint = _checkpoint_dir(checkpoint_root, view)
    if not checkpoint.exists():
        fit_prism_checkpoint_for_view(paths, view, checkpoint_root)

    result_path = paths.final / "test_predictions" / view.relative_root / "PRISM_INFERENCE_RESULT.json"
    if result_path.is_file():
        payload = _read_json(result_path)
        records = [dict(item) for item in payload.get("models", [])]
        if not records:
            raise RuntimeError("STOP_CORRECTED_FORMAL_RESULT_EMPTY")
    else:
        previous = os.environ.get(INFERENCE_ONLY_ENV)
        os.environ[INFERENCE_ONLY_ENV] = "1"
        try:
            records = [
                dict(item)
                for item in predict_prism_checkpoint_for_view(
                    paths, view, checkpoint_root, split="test"
                )
            ]
        finally:
            if previous is None:
                os.environ.pop(INFERENCE_ONLY_ENV, None)
            else:
                os.environ[INFERENCE_ONLY_ENV] = previous
    return paths, checkpoint_root, records


def _prediction_difference(left: Path, right: Path) -> dict[str, Any]:
    a = pd.read_parquet(left)[["base_origin_id", "y_true", "y_pred"]]
    b = pd.read_parquet(right)[["base_origin_id", "y_true", "y_pred"]]
    merged = a.rename(columns={"y_true": "a_true", "y_pred": "a_pred"}).merge(
        b.rename(columns={"y_true": "b_true", "y_pred": "b_pred"}),
        on="base_origin_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(a) or len(merged) != len(b):
        raise RuntimeError("STOP_E1_STAGE_SUPPORT_MISMATCH")
    return {
        "rows": int(len(merged)),
        "maximum_absolute_truth_error": float(
            np.max(
                np.abs(
                    merged["a_true"].to_numpy(dtype=np.float64)
                    - merged["b_true"].to_numpy(dtype=np.float64)
                ),
                initial=0.0,
            )
        ),
        "maximum_absolute_prediction_error": float(
            np.max(
                np.abs(
                    merged["a_pred"].to_numpy(dtype=np.float64)
                    - merged["b_pred"].to_numpy(dtype=np.float64)
                ),
                initial=0.0,
            )
        ),
    }


def run_e1(project: Path, anchor: Path, run_root: Path) -> dict[str, Any]:
    from prism_benchmark.cz_authority_pure_k_checkpoint import (
        derive_pure_k_checkpoint_for_view,
        pure_k_formal_identity_certificate,
        pure_k_oof_identity_certificate,
        verify_pure_k_checkpoint_reload,
    )

    freeze = preflight(project, anchor, run_root)
    pure_k_root = run_root / "checkpoints"
    e1_root = run_root / "E1_STAGEWISE"
    formal_records: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    zero_audits: list[dict[str, Any]] = []
    stage_map = {
        "PRISM_V2_1_1_K_C_DYNAMIC": "K+C",
        "PRISM_V2_1_1_K_C_W_DYNAMIC": "K+C+DELTA_W",
        "PRISM_V2_1_1_PHYSICS_FIRST": "K+C+DELTA_W+A",
        "PRISM_V2_1_1_JOINT_KWA": "JOINT",
    }

    for direction in DIRECTIONS:
        anchor_paths, view, _ = _anchor_paths(
            project, anchor, direction
        )
        paths, authority_checkpoint_root, corrected_records = _corrected_authority_replay(
            project, anchor_paths, view, direction, run_root
        )
        for record in corrected_records:
            model = str(record.get("model"))
            if model in stage_map:
                formal_records.append(
                    {
                        "direction": direction,
                        "h_steps": H_STEPS,
                        "stage": stage_map[model],
                        **record,
                    }
                )
        derivation = derive_pure_k_checkpoint_for_view(
            paths, view, authority_checkpoint_root, pure_k_root / direction
        )
        reload_audit = verify_pure_k_checkpoint_reload(
            Path(str(derivation["checkpoint_dir"]))
        )
        oof = pure_k_oof_identity_certificate(paths, view, pure_k_root / direction)
        formal = pure_k_formal_identity_certificate(
            paths, view, pure_k_root / direction, e1_root / direction
        )
        formal_records.append({"direction": direction, "stage": "K", **formal})
        certificates.append(
            {
                "direction": direction,
                "derivation": derivation,
                "reload": reload_audit,
                "development_identity": oof,
                "formal_identity": formal,
            }
        )
        pure_path = Path(str(formal["prediction_path"]))
        final_root = paths.final / "test_predictions" / view.relative_root
        k_c = final_root / "PRISM_V2_1_1_K_C_DYNAMIC.parquet"
        k_c_w = final_root / "PRISM_V2_1_1_K_C_W_DYNAMIC.parquet"
        c_identity = _prediction_difference(pure_path, k_c)
        w_identity = _prediction_difference(k_c, k_c_w)
        if c_identity["maximum_absolute_prediction_error"] > 1e-10:
            raise RuntimeError("STOP_E1_C_ZERO_IDENTITY_FAILED")
        if w_identity["maximum_absolute_prediction_error"] > 1e-10:
            raise RuntimeError("STOP_E1_W_ZERO_IDENTITY_FAILED")
        zero_audits.append(
            {
                "direction": direction,
                "C": {"status": "PASS", **c_identity},
                "W": {"status": "PASS", **w_identity},
            }
        )

    order = {"K": 0, "K+C": 1, "K+C+DELTA_W": 2, "K+C+DELTA_W+A": 3, "JOINT": 4}
    formal_records.sort(key=lambda row: (str(row["direction"]), order[str(row["stage"])]))
    frame = pd.DataFrame(formal_records)
    if len(frame) != len(DIRECTIONS) * len(order):
        raise RuntimeError("STOP_E1_INCOMPLETE_STAGE_LADDER")
    for direction, group in frame.groupby("direction", sort=False):
        if group["sample_id_order_hash"].nunique() != 1:
            raise RuntimeError(f"STOP_E1_SAMPLE_ORDER_HASH_MISMATCH:{direction}")
        if group["scoring_support_hash"].nunique() != 1:
            raise RuntimeError(f"STOP_E1_SCORING_SUPPORT_HASH_MISMATCH:{direction}")
    e1_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(e1_root / "stagewise_metrics.csv", index=False)
    gains: list[dict[str, Any]] = []
    for direction, group in frame.groupby("direction", sort=False):
        group = group.sort_values("stage", key=lambda values: values.map(order))
        previous: Mapping[str, Any] | None = None
        for row in group.to_dict("records"):
            gains.append(
                {
                    "direction": direction,
                    "stage": row["stage"],
                    "rmse_delta": row["rmse_delta"],
                    "r2_delta": row["r2_delta"],
                    "r2_level_reconstructed": row["r2_level_reconstructed"],
                    "rmse_delta_gain_vs_parent": (
                        None
                        if previous is None
                        else float(previous["rmse_delta"]) - float(row["rmse_delta"])
                    ),
                }
            )
            previous = row
    pd.DataFrame(gains).to_csv(e1_root / "incremental_gains.csv", index=False)
    _write_json(e1_root / "pure_k_replay_certificate.json", {"status": "PASS", "directions": certificates})
    _write_json(e1_root / "exact_zero_audit.json", {"status": "PASS", "directions": zero_audits})
    result = {
        "status": "PASS",
        "stage": "P2_E1_PURE_K_REPLAY",
        "protocol": "CZ_RAW2S_CURRENT_L256_H4_W1_W0_1",
        "sampling_period_seconds": 2,
        "forecast_seconds": 8,
        "directions": list(DIRECTIONS),
        "stage_rows": int(len(frame)),
        "pure_k_refit_performed": False,
        "sealed_anchor_modified": False,
        "test_metrics_used_for_selection": False,
        "full_storage_gate": freeze["e2_e6_full_execution_gate"],
        "completed_utc": _utc(),
    }
    _write_json(e1_root / "STATUS.json", result)
    _write_json(
        run_root / "RUN_STATUS.json",
        {
            "status": "PARTIAL",
            "completed": ["P0_PLAN_AND_CODE_FREEZE", "P1_ADAPTER_UNIT_TESTS", "P2_E1_PURE_K_REPLAY"],
            "E1_STAGEWISE": "PASS",
            "E2_E6": freeze["e2_e6_full_execution_gate"],
            "reason": "E2-E6 require the frozen 300 GiB private-storage gate",
            "updated_utc": _utc(),
        },
    )
    return result


def status(run_root: Path) -> dict[str, Any]:
    return {
        name: (
            _read_json(run_root / name)
            if (run_root / name).is_file()
            else {"status": "NOT_YET_RUN"}
        )
        for name in (
            "PLAN_FREEZE.json",
            "FULL_STORAGE_GATE.json",
            "STREAMING_STORAGE_GATE.json",
            "RUN_STATUS.json",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "e1",
            "full-storage-gate",
            "streaming-storage-gate",
            "status",
        ),
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--anchor-run-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/dev/shm"))
    args = parser.parse_args()
    project = args.project.resolve()
    anchor = args.anchor_run_root.resolve()
    run_root = args.run_root.resolve()
    sys.path.insert(0, str(project / "src"))
    if args.stage == "preflight":
        result = preflight(project, anchor, run_root)
    elif args.stage == "e1":
        result = run_e1(project, anchor, run_root)
    elif args.stage == "full-storage-gate":
        result = full_storage_gate(run_root)
    elif args.stage == "streaming-storage-gate":
        result = streaming_storage_gate(project, run_root, args.scratch_root.resolve())
    else:
        result = status(run_root)
    print(json.dumps({"stage": args.stage, "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
