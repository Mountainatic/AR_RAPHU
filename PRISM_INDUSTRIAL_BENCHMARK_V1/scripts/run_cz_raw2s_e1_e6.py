"""Authoritative CZ raw-2s H4 orchestration for the E1--E6 correction.

This entry point deliberately contains no surrogate implementation of PRISM.
It delegates C1 materialization, K/C/W/A/Joint development, strict nested-OOF
selection, portable checkpoint fitting, and inference-only formal evaluation
to the implementation descended from ``prism-strict-oof-finalization-20260915``.

The previous version of this file implemented rolling statistics, PCA and
Ridge blocks under the names K/C/W/A. Those were not the authoritative PRISM
operators. E2--E6 remain ``NOT_YET_RUN`` until an experiment-specific adapter
can exercise the same authoritative modules without changing their semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROTOCOL_ID = "CZ_RAW2S_H4_AUTHORITATIVE_PRISM_E1_E6_CORRECTION_20260922_R1"
AUTHORITY_BRANCH = "prism-strict-oof-finalization-20260915"
AUTHORITY_COMMIT = "2ee6273b8f915cbcdff2f46d56bc80047ddae4a7"
CONFIG_RELATIVE_PATH = Path("configs/cz_raw2s_h4_authoritative_e1_e6_20260922.json")
AUTHORITY_RUNNER_RELATIVE_PATH = Path("scripts/run_independent_extension_20260825.py")

RAW_PERIOD_SECONDS = 2
HISTORY_LENGTH = 256
H = 4
W = 1
W0 = 1
PURGE = HISTORY_LENGTH + H
DIRECTIONS = ("Rod_1_to_Rod_2", "Rod_2_to_Rod_1")

# Every unmodified model/selector module must be byte-identical to the authority
# commit. Reviewed exceptions are pinned separately below.
AUTHORITY_BLOBS = {
    "src/prism_benchmark/cz_k_support.py": "5853b3157a90b8b0fc84e8bee497aa861fa4a443",
    "src/prism_benchmark/v211_c.py": "f2ca61ac43ece6dfda6e65379bcfb4015b67e2ae",
    "src/prism_benchmark/v211_w.py": "2113e4ae119969e63569206a4e6ca9eda4cb0a1f",
    "src/prism_benchmark/v211_a.py": "5adbeeeeb9fa804fef71f88706c58e34608b0229",
    "src/prism_benchmark/strict_oof_selection.py": "a399fd6740447585a1a3c0853c5493ccbe0b99e8",
    "src/prism_benchmark/cz_l256_nowcast.py": "cc2a9ef5251ed8eb292c114b19f8d9528cddbf46",
}
CHECKPOINT_RELATIVE_PATH = "src/prism_benchmark/representative_prism_checkpoints.py"
CHECKPOINT_AUTHORITY_BLOB = "71d9c729f69dbead855a5a2ef70e04cd024a6719"
CHECKPOINT_BEST_K_PATCHED_BLOB = "5ecc15b6b7b3199c4b9442922955a5c39bbfd32a"
JOINT_RELATIVE_PATH = "src/prism_benchmark/v211_joint_stability.py"
JOINT_AUTHORITY_BLOB = "ba3b5d2783fe94c1e46158878560c28a721c2958"
JOINT_ZERO_C_PATCHED_BLOB = "52719bb7c2b29ac242be0fa927e9ffc80e8144bb"
METRIC_RELATIVE_PATH = "src/prism_benchmark/level_reconstruction.py"
METRIC_AUTHORITY_BLOB = "511c00d6f88e021618d018c3a111ffc1dddb1c89"
METRIC_LABEL_PATCHED_BLOB = "abca993161b7650fdf4c7f67a8d80e2ab38bf609"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def validate_protocol_contract() -> None:
    if (RAW_PERIOD_SECONDS, HISTORY_LENGTH, H, W, W0, PURGE) != (2, 256, 4, 1, 1, 260):
        raise RuntimeError("STOP_CZ_H4_PROTOCOL_DRIFT")


def chronological_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Protocol-only helper retained for dependency-isolation regression tests."""

    if len(frame) < 3:
        raise RuntimeError("at least three samples are required")
    cut = max(1, min(len(frame) - 1, int(np.floor(0.8 * len(frame)))))
    validation_start = int(frame.iloc[cut]["absolute_dependency_start"])
    train = np.flatnonzero(
        frame["absolute_dependency_stop_exclusive"].to_numpy(dtype=np.int64)
        <= validation_start
    )
    validation = np.arange(cut, len(frame), dtype=np.int64)
    if not len(train) or not len(validation):
        raise RuntimeError("dependency-purged split is empty")
    if (
        int(frame.loc[train, "absolute_dependency_stop_exclusive"].max())
        > int(frame.loc[validation, "absolute_dependency_start"].min())
    ):
        raise RuntimeError("dependency intervals overlap")
    return train, validation


def metrics(y: np.ndarray, prediction: np.ndarray, current: np.ndarray) -> dict[str, float]:
    """Small metric identity helper; model fitting is intentionally absent."""

    truth = np.asarray(y, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    anchor = np.asarray(current, dtype=np.float64)
    residual = truth - estimate
    model_mse = float(np.mean(np.square(residual)))
    rmse = float(np.sqrt(model_mse))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum(np.square(truth - truth.mean(dtype=np.float64))))
    r2_delta = (
        float(1.0 - np.sum(np.square(residual)) / denominator)
        if denominator
        else float("nan")
    )
    level_truth = anchor + truth
    level_prediction = anchor + estimate
    level_denominator = float(
        np.sum(np.square(level_truth - level_truth.mean(dtype=np.float64)))
    )
    level_r2 = (
        float(
            1.0
            - np.sum(np.square(level_truth - level_prediction)) / level_denominator
        )
        if level_denominator
        else float("nan")
    )
    persistence_mse = float(np.mean(np.square(truth)))
    persistence_rmse = float(np.sqrt(persistence_mse))
    return {
        "rmse_delta": rmse,
        "mae_delta": mae,
        "r2_delta": r2_delta,
        "r2_level_reconstructed": level_r2,
        "persistence_skill": (
            float(1.0 - model_mse / persistence_mse)
            if persistence_mse
            else float("nan")
        ),
        "persistence_skill_mse": (
            float(1.0 - model_mse / persistence_mse)
            if persistence_mse
            else float("nan")
        ),
        "persistence_skill_rmse": (
            float(1.0 - rmse / persistence_rmse)
            if persistence_rmse
            else float("nan")
        ),
    }


def authority_audit(project: Path) -> dict[str, Any]:
    validate_protocol_contract()
    repo = project.parent
    if not (project / AUTHORITY_RUNNER_RELATIVE_PATH).is_file():
        raise RuntimeError("STOP_AUTHORITY_RUNNER_MISSING")
    ancestor = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        AUTHORITY_COMMIT,
        "HEAD",
        check=False,
    ).returncode == 0
    if not ancestor:
        raise RuntimeError("STOP_STRICT_OOF_AUTHORITY_NOT_ANCESTOR")

    modules: list[dict[str, Any]] = []
    for relative, expected_blob in AUTHORITY_BLOBS.items():
        path = project / relative
        if not path.is_file():
            raise RuntimeError(f"STOP_AUTHORITY_MODULE_MISSING:{relative}")
        authority_blob = _git(
            repo,
            "rev-parse",
            f"{AUTHORITY_COMMIT}:PRISM_INDUSTRIAL_BENCHMARK_V1/{relative}",
        ).stdout.strip()
        current_blob = _git(repo, "hash-object", str(path)).stdout.strip()
        if authority_blob != expected_blob or current_blob != expected_blob:
            raise RuntimeError(f"STOP_AUTHORITY_MODULE_DRIFT:{relative}")
        modules.append(
            {
                "path": relative,
                "authority_blob": authority_blob,
                "current_blob": current_blob,
                "sha256": sha256_file(path),
                "status": "BYTE_IDENTICAL_TO_AUTHORITY",
            }
        )

    checkpoint = project / CHECKPOINT_RELATIVE_PATH
    authority_checkpoint = _git(
        repo,
        "rev-parse",
        f"{AUTHORITY_COMMIT}:PRISM_INDUSTRIAL_BENCHMARK_V1/{CHECKPOINT_RELATIVE_PATH}",
    ).stdout.strip()
    current_checkpoint = _git(repo, "hash-object", str(checkpoint)).stdout.strip()
    if authority_checkpoint != CHECKPOINT_AUTHORITY_BLOB:
        raise RuntimeError("STOP_CHECKPOINT_AUTHORITY_BLOB_DRIFT")
    if current_checkpoint != CHECKPOINT_BEST_K_PATCHED_BLOB:
        raise RuntimeError("STOP_CHECKPOINT_BEST_K_PATCH_DRIFT")
    modules.append(
        {
            "path": CHECKPOINT_RELATIVE_PATH,
            "authority_blob": authority_checkpoint,
            "current_blob": current_checkpoint,
            "sha256": sha256_file(checkpoint),
            "status": "AUTHORITY_PLUS_REVIEWED_BEST_K_ENUM_PATCH",
        }
    )

    joint = project / JOINT_RELATIVE_PATH
    authority_joint = _git(
        repo,
        "rev-parse",
        f"{AUTHORITY_COMMIT}:PRISM_INDUSTRIAL_BENCHMARK_V1/{JOINT_RELATIVE_PATH}",
    ).stdout.strip()
    current_joint = _git(repo, "hash-object", str(joint)).stdout.strip()
    if authority_joint != JOINT_AUTHORITY_BLOB:
        raise RuntimeError("STOP_JOINT_AUTHORITY_BLOB_DRIFT")
    if current_joint != JOINT_ZERO_C_PATCHED_BLOB:
        raise RuntimeError("STOP_JOINT_ZERO_C_PATCH_DRIFT")
    modules.append(
        {
            "path": JOINT_RELATIVE_PATH,
            "authority_blob": authority_joint,
            "current_blob": current_joint,
            "sha256": sha256_file(joint),
            "status": "AUTHORITY_PLUS_REVIEWED_ZERO_C_PATCH",
        }
    )
    metric = project / METRIC_RELATIVE_PATH
    authority_metric = _git(
        repo,
        "rev-parse",
        f"{AUTHORITY_COMMIT}:PRISM_INDUSTRIAL_BENCHMARK_V1/{METRIC_RELATIVE_PATH}",
    ).stdout.strip()
    current_metric = _git(repo, "hash-object", str(metric)).stdout.strip()
    if authority_metric != METRIC_AUTHORITY_BLOB:
        raise RuntimeError("STOP_METRIC_AUTHORITY_BLOB_DRIFT")
    if current_metric != METRIC_LABEL_PATCHED_BLOB:
        raise RuntimeError("STOP_METRIC_LABEL_PATCH_DRIFT")
    modules.append(
        {
            "path": METRIC_RELATIVE_PATH,
            "authority_blob": authority_metric,
            "current_blob": current_metric,
            "sha256": sha256_file(metric),
            "status": "AUTHORITY_PLUS_REVIEWED_METRIC_LABEL_PATCH",
        }
    )
    return {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "authority_branch": AUTHORITY_BRANCH,
        "authority_commit": AUTHORITY_COMMIT,
        "authority_is_ancestor": True,
        "execution_commit": _git(repo, "rev-parse", "HEAD").stdout.strip(),
        "modules": modules,
        "joint_patch_scope": "zero-C routing guard and nullable best-active-K reference only",
        "checkpoint_patch_scope": (
            "bind the BEST_ACTIVE_K_CHANNEL enum during final checkpoint refit "
            "and replay; no estimator or selection change"
        ),
        "metric_patch_scope": "report both MSE-relative and RMSE-relative persistence skill names",
        "custom_prism_feature_or_estimator_code_present": False,
        "created_utc": utc(),
    }


def _load_authority_runner(project: Path) -> ModuleType:
    path = project / AUTHORITY_RUNNER_RELATIVE_PATH
    name = "cz_h4_authority_runner_20260922"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("STOP_CANNOT_IMPORT_AUTHORITY_RUNNER")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.PROJECT = project
    module.REPO_ROOT = project.parent
    module.CONFIG_PATH = project / CONFIG_RELATIVE_PATH
    module.BASELINE_COMMIT = AUTHORITY_COMMIT
    return module


def _e1_authority_ladder(run_root: Path) -> dict[str, Any]:
    report_path = run_root / "final" / "INDEPENDENT_EXTENSION_REPORT.json"
    report = read_json(report_path)
    stage_by_model = {
        "PRISM_V2_1_1_K_C_DYNAMIC": "K+C",
        "PRISM_V2_1_1_K_C_W_DYNAMIC": "K+C+DELTA_W",
        "PRISM_V2_1_1_K_C_A_ABLATION": "K+C+A_ABLATION",
        "PRISM_V2_1_1_PHYSICS_FIRST": "K+C+DELTA_W+A",
        "PRISM_V2_1_1_JOINT_KWA": "J_KWA",
    }
    rows: list[dict[str, Any]] = []
    for record in report.get("cz", []):
        model = str(record.get("model"))
        if (
            int(record.get("h_steps", -1)) == H
            and record.get("information_set") == "dynamic"
            and model in stage_by_model
        ):
            rows.append({"stage": stage_by_model[model], **record})
    expected_per_direction = set(stage_by_model.values())
    observed = {
        direction: {str(row["stage"]) for row in rows if row.get("direction") == direction}
        for direction in DIRECTIONS
    }
    missing = {
        direction: sorted(expected_per_direction - stages)
        for direction, stages in observed.items()
        if stages != expected_per_direction
    }
    destination = run_root / "E1_STAGEWISE"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "formal_authority_ladder.csv", index=False)
    result = {
        "status": "PARTIAL" if not missing else "FAILED",
        "protocol_id": PROTOCOL_ID,
        "h_steps": H,
        "forecast_seconds": H * RAW_PERIOD_SECONDS,
        "rows": len(rows),
        "directions": list(DIRECTIONS),
        "stages": sorted(expected_per_direction),
        "missing": missing,
        "pure_k_formal_stage": "NOT_YET_RUN",
        "pure_k_reason": (
            "the authoritative portable final-checkpoint codec begins at the "
            "registered K+C route; a separate authority-preserving K-only final "
            "checkpoint adapter is required"
        ),
        "model_semantics": "authority K/C/W/A/Joint modules; no surrogate features",
        "test_accessed_after_checkpoint_seal": True,
    }
    write_json(destination / "STATUS.json", result)
    return result


def _write_supplementary_status(run_root: Path, e1: Mapping[str, Any]) -> dict[str, Any]:
    experiments = {
        "E1_STAGEWISE": {
            "status": str(e1["status"]),
            "reason": "authority ladder emitted; pure K formal adapter remains NOT_YET_RUN",
        },
        "E2_SEMISYNTHETIC": {
            "status": "NOT_YET_RUN",
            "reason": "authority-compatible target injection adapter is not implemented",
        },
        "E3_MULTISCALE": {
            "status": "NOT_YET_RUN",
            "reason": "must rerun registered K profiles and downstream authority modules at equal fit budget",
        },
        "E4_CANDIDATE_SENSITIVITY": {
            "status": "NOT_YET_RUN",
            "reason": "candidate-universe changes require separately frozen authority-module configurations",
        },
        "E5_STRUCTURAL_STABILITY": {
            "status": "NOT_YET_RUN",
            "reason": "must aggregate authority-module reruns rather than proxy route signatures",
        },
        "E6_MEASUREMENT_ROBUSTNESS": {
            "status": "NOT_YET_RUN",
            "reason": "raw perturbation must occur before C1 followed by full authority refit/replay",
        },
    }
    result = {
        "status": "PARTIAL",
        "protocol_id": PROTOCOL_ID,
        "experiments": experiments,
        "invalidated_predecessor": "CZ_RAW2S_H4_CORRECTED_R2 proxy Ridge/PCA runner",
        "completed_utc": utc(),
    }
    write_json(run_root / "SUPPLEMENTARY_STATUS.json", result)
    write_json(
        run_root / "RUN_STATUS.json",
        {
            "status": "PARTIAL",
            "reason": "authoritative E1 ladder complete except pure K; E2-E6 not yet rerun",
            "raw_workbook_copied_to_results": False,
            "github_upload": False,
            "completed_utc": utc(),
        },
    )
    return result


def _run_one(stage: str, run_root: Path, project: Path) -> Any:
    audit = authority_audit(project)
    runner = _load_authority_runner(project)
    if stage == "scope":
        result = runner.scope(run_root)
        write_json(run_root / "freeze" / "AUTHORITY_IMPLEMENTATION_AUDIT.json", audit)
        return result
    audit_path = run_root / "freeze" / "AUTHORITY_IMPLEMENTATION_AUDIT.json"
    if not audit_path.is_file() or read_json(audit_path).get("status") != "PASS":
        raise RuntimeError("STOP_AUTHORITY_IMPLEMENTATION_AUDIT_NOT_SEALED")
    if stage == "environment":
        return runner.environment(run_root)
    if stage == "pilot":
        return runner.run_cz_pilot(run_root)
    if stage == "pilot-accept":
        return runner.formal_pilot_accept(run_root)
    if stage == "development":
        return runner.run_cz_development(run_root)
    if stage == "development-reconcile":
        return runner.reconcile_cz_development(run_root)
    if stage == "freeze":
        return runner.formal_freeze(run_root)
    if stage == "support-freeze":
        return runner.freeze_cz_common_support(run_root)
    if stage == "checkpoints":
        return runner.fit_checkpoints(run_root)
    if stage == "formal-test":
        result = runner.run_test(run_root)
        e1 = _e1_authority_ladder(run_root)
        _write_supplementary_status(run_root, e1)
        return result
    if stage == "report":
        return runner.report(run_root)
    if stage == "status":
        return {
            "authority_audit": read_json(audit_path),
            "supplementary": (
                read_json(run_root / "SUPPLEMENTARY_STATUS.json")
                if (run_root / "SUPPLEMENTARY_STATUS.json").is_file()
                else {"status": "NOT_YET_RUN"}
            ),
        }
    raise ValueError(stage)


def _run_all(run_root: Path, project: Path) -> dict[str, Any]:
    stages = (
        "scope",
        "environment",
        "pilot",
        "pilot-accept",
        "development",
        "development-reconcile",
        "freeze",
        "support-freeze",
        "checkpoints",
        "formal-test",
        "report",
    )
    results: dict[str, Any] = {}
    for stage in stages:
        results[stage] = _run_one(stage, run_root, project)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=(
            "all",
            "scope",
            "environment",
            "pilot",
            "pilot-accept",
            "development",
            "development-reconcile",
            "freeze",
            "support-freeze",
            "checkpoints",
            "formal-test",
            "report",
            "status",
        ),
    )
    parser.add_argument("--output", "--run-root", dest="run_root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    project = args.project.resolve()
    sys.path.insert(0, str(project / "src"))
    result = (
        _run_all(run_root, project)
        if args.stage == "all"
        else _run_one(args.stage, run_root, project)
    )
    print(
        json.dumps(
            {
                "status": (
                    result.get("status", "PASS")
                    if isinstance(result, Mapping)
                    else "PASS"
                ),
                "stage": args.stage,
                "output": str(run_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
