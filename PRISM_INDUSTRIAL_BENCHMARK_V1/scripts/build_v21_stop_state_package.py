from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism_benchmark.cpu_data import sha256_file
from prism_benchmark.v21_audit import write_post_audit
from prism_benchmark.v21_config import V21Paths


PACKAGE_NAME = "PRISM_V2_1_SRU_STAGEWISE_ROUTED_RESULTS_bundle"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _status(raw: str) -> str:
    if raw in {"PASS", "BASELINE_REPLAY_FROZEN", "ASSEMBLY_FROZEN"}:
        return "COMPLETED"
    if raw in {
        "SOLVER_FAILED_RETAINED",
        "JOINT_INPUT_PATH_COLLAPSED",
        "COMPLETED_WITH_RETAINED_FAILURES",
    }:
        return "FAILED"
    return raw


def _model(stage: str) -> str:
    return {
        "E2_K": "PRISM_V2_1_K",
        "E2_C": "PRISM_V2_1_K_C",
        "E3_W": "PRISM_V2_1_K_C_W",
        "E4_A": "PRISM_V2_1_PHYSICS_FIRST",
        "E5_JOINT": "PRISM_V2_1_JOINT_KWA",
    }.get(stage, stage)


def _information_set(stage: str) -> str:
    return "dynamic" if stage in {"E4_A", "E5_JOINT"} else "input_only"


def _git(project: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.filemode=false", *args],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _records(root: Path, excluded: set[Path]) -> list[dict[str, Any]]:
    excluded_resolved = {path.resolve() for path in excluded}
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded_resolved:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=np.float64)
    y_pred = frame["y_pred"].to_numpy(dtype=np.float64)
    error = y_true - y_pred
    mse = float(np.mean(error * error, dtype=np.float64))
    variance = float(np.var(y_true, dtype=np.float64))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "r2": float(
            1.0 - np.sum(error * error) / np.sum((y_true - np.mean(y_true)) ** 2)
        )
        if variance > 0
        else float("nan"),
        "nrmse": float(np.sqrt(mse) / np.sqrt(variance))
        if variance > 0
        else float("nan"),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_stop_state(paths: V21Paths) -> dict[str, Any]:
    output = paths.output
    stop_audit_path = output / "RUN_LOG" / "V21_CHAIN_FINAL_STOP_AUDIT.json"
    stop_audit = _json(stop_audit_path)
    if stop_audit.get("status") != "STOPPED_BY_FROZEN_E5_GATE":
        raise RuntimeError("expected frozen E5 stop audit is missing")
    if paths.final_freeze_path.exists():
        raise RuntimeError("stop-state packaging refuses an existing E6 freeze")

    post_audit = write_post_audit(paths.shared, output)
    reports = output / "REPORTS"
    stop_state = reports / "STOP_STATE"
    reports.mkdir(parents=True, exist_ok=True)
    stop_state.mkdir(parents=True, exist_ok=True)

    result_paths = sorted((output / "DEVELOPMENT").rglob("RESULT.json"))
    results = [(path, _json(path)) for path in result_paths]

    metric_fields = [
        "target_head",
        "information_set",
        "availability_scenario",
        "proxy_policy",
        "model",
        "stage",
        "split",
        "status",
        "raw_status",
        "mse",
        "rmse",
        "mae",
        "r2",
        "nrmse",
        "rows",
        "selected_candidate",
        "prediction_path",
        "prediction_sha256",
        "artifact_path",
        "test_accessed",
    ]
    metric_rows: list[dict[str, Any]] = []
    for path, result in results:
        stage = str(result.get("stage", ""))
        metric_rows.append(
            {
                "target_head": result.get("target_head", ""),
                "information_set": _information_set(stage),
                "availability_scenario": result.get("availability_scenario", ""),
                "proxy_policy": result.get("proxy_policy", ""),
                "model": _model(stage),
                "stage": stage,
                "split": "validation",
                "status": _status(str(result.get("status", "NOT_YET_RUN"))),
                "raw_status": result.get("status", "NOT_YET_RUN"),
                "mse": result.get("mse"),
                "rmse": result.get("rmse"),
                "mae": result.get("mae"),
                "r2": result.get("r2"),
                "nrmse": result.get("nrmse"),
                "rows": result.get("rows"),
                "selected_candidate": result.get("final_selected_candidate"),
                "prediction_path": result.get("prediction_path"),
                "prediction_sha256": result.get("prediction_sha256"),
                "artifact_path": path.relative_to(output).as_posix(),
                "test_accessed": bool(result.get("test_accessed", False)),
            }
        )
    for stage, split in (
        ("E6_FINAL_FREEZE", ""),
        ("E7_TEST", "test"),
        ("E8_REPORT_AND_PACKAGE", ""),
    ):
        metric_rows.append(
            {
                "target_head": "",
                "information_set": "",
                "availability_scenario": "",
                "proxy_policy": "",
                "model": "",
                "stage": stage,
                "split": split,
                "status": "NOT_YET_RUN",
                "raw_status": "NOT_YET_RUN",
                "mse": None,
                "rmse": None,
                "mae": None,
                "r2": None,
                "nrmse": None,
                "rows": None,
                "selected_candidate": None,
                "prediction_path": None,
                "prediction_sha256": None,
                "artifact_path": "",
                "test_accessed": False,
            }
        )
    _write_csv(reports / "V21_SRU_FINAL_METRICS.csv", metric_fields, metric_rows)

    entity_fields = [
        "target_head",
        "information_set",
        "model",
        "split",
        "entity_id",
        "rows",
        "status",
        "mse",
        "rmse",
        "mae",
        "r2",
        "nrmse",
        "test_accessed",
    ]
    entity_rows: list[dict[str, Any]] = []
    for _, result in results:
        prediction_rel = result.get("prediction_path")
        if not prediction_rel:
            continue
        prediction_path = output / str(prediction_rel)
        if not prediction_path.is_file():
            continue
        frame = pd.read_parquet(prediction_path)
        if not {"entity_id", "y_true", "y_pred"}.issubset(frame.columns):
            continue
        stage = str(result.get("stage", ""))
        for entity, group in frame.groupby("entity_id", sort=False):
            entity_rows.append(
                {
                    "target_head": result.get("target_head", ""),
                    "information_set": _information_set(stage),
                    "model": _model(stage),
                    "split": "validation",
                    "entity_id": str(entity),
                    "rows": int(len(group)),
                    "status": _status(str(result.get("status", "NOT_YET_RUN"))),
                    **_prediction_metrics(group),
                    "test_accessed": bool(result.get("test_accessed", False)),
                }
            )
    _write_csv(reports / "V21_SRU_ENTITY_METRICS.csv", entity_fields, entity_rows)

    _write_csv(
        reports / "V21_SRU_BOOTSTRAP.csv",
        ["status", "reason", "split", "replicates", "test_accessed"],
        [
            {
                "status": "NOT_YET_RUN",
                "reason": "E5_JOINT_STOPPED_BEFORE_E6_FREEZE",
                "split": "test",
                "replicates": "",
                "test_accessed": False,
            }
        ],
    )

    with (reports / "V21_SRU_MODEL_AUDIT.jsonl").open("w", encoding="utf-8") as handle:
        for path, result in results:
            handle.write(
                json.dumps(
                    {
                        "artifact_path": path.relative_to(output).as_posix(),
                        "status": _status(str(result.get("status", "NOT_YET_RUN"))),
                        "raw_status": result.get("status"),
                        "target_head": result.get("target_head"),
                        "stage": result.get("stage"),
                        "test_accessed": bool(result.get("test_accessed", False)),
                        "result": result,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    with (reports / "V21_SRU_ASSEMBLY_CARDS.jsonl").open("w", encoding="utf-8") as handle:
        for path in sorted((output / "ASSEMBLY_CARDS").rglob("*_ASSEMBLY_CARD.json")):
            handle.write(path.read_text(encoding="utf-8").strip() + "\n")

    runtime = _json(output / "RUN_LOG" / "V21_RUNTIME_PROFILE.json")
    _write_json(
        reports / "V21_SRU_RESOURCE_AUDIT.json",
        {
            "status": "COMPLETED",
            "runtime_manager": "uv",
            "runtime_profile": runtime,
            "development_result_count": len(results),
            "development_elapsed_seconds": sum(
                float(result.get("elapsed_seconds", 0.0) or 0.0)
                for _, result in results
            ),
            "test_accessed": False,
        },
    )

    code_head = _git(paths.project, "rev-parse", "HEAD")
    code_status = _git(paths.project, "status", "--short")
    theory = paths.plan / "PRISM_Theory_v2_1_Stagewise_Routed_Modular_Assembly_Theory_Only.md"
    plan = paths.plan / "PRISM_V2_1_SRU_EXPERIMENT_AND_IMPLEMENTATION_PLAN.md"
    _write_json(
        stop_state / "V21_SRU_FINAL_FREEZE_MANIFEST.json",
        {
            "status": "NOT_YET_RUN",
            "stage": "E6_FINAL_FREEZE",
            "reason": "E5_JOINT_STOPPED_BY_FROZEN_INPUT_PATH_GATE",
            "theory_sha256": sha256_file(theory),
            "plan_sha256": sha256_file(plan),
            "config_sha256": sha256_file(paths.config_path),
            "code_commit": code_head,
            "dirty_status": code_status,
            "data_post_audit_sha256": sha256_file(
                output / "DATA_AUDIT" / "V21_DATA_BASE_POST_AUDIT.json"
            ),
            "completed_stages": [
                "B0",
                "E0",
                "E1",
                "E2-K",
                "E2-C",
                "E3-W",
                "E4-A",
            ],
            "stopped_stage": "E5-JOINT",
            "test_accessed": False,
            "baseline_replay_test_accessed": True,
            "v21_candidate_test_accessed": False,
            "bootstrap_status": "NOT_YET_RUN",
            "final_metrics_scope": "validation_and_stage_status_only",
        },
    )

    report_lines = [
        "# PRISM v2.1 SRU final report (stop-state)",
        "",
        "Overall status: FAILED.",
        "",
        "This is a stop-state evidence report, not an E8-completed scientific release.",
        "The report and package are generated because the registered plan requires a",
        "report/package even when a frozen stage gate stops the chain.",
        "",
        "## Scope",
        "",
        "- Dataset: SRU only.",
        "- Heads: SRU_H2S__H5__W1 and SRU_SO2__H5__W1.",
        "- Historical baseline parquet: NOT_AVAILABLE_NOT_SEARCHED_NOT_REQUIRED.",
        "- Baseline replay test access: authorized exception; test metrics were not computed or exposed.",
        "- v2.1 candidate test access: false.",
        "",
        "## Stage status",
        "",
        "| Stage | Status |",
        "|---|---|",
    ]
    stage_status = {
        "B0": _status(
            str(_json(output / "BASELINES" / "BASELINE_REPLAY_MANIFEST.json").get("status", "NOT_YET_RUN"))
        ),
        "E0": _status(
            str(_json(output / "FREEZE" / "E0_INHERITANCE_AUDIT.json").get("status", "NOT_YET_RUN"))
        ),
        "E1": _status(
            str(_json(output / "FREEZE" / "E1_REGRESSION_TESTS.json").get("status", "NOT_YET_RUN"))
        ),
        **{
            str(result.get("stage")): _status(str(result.get("status", "NOT_YET_RUN")))
            for _, result in results
        },
    }
    for stage in (
        "B0",
        "E0",
        "E1",
        "E2_K",
        "E2_C",
        "E3_W",
        "E4_A",
        "E5_JOINT",
        "E6_FINAL_FREEZE",
        "E7_TEST",
        "E8_REPORT_AND_PACKAGE",
    ):
        report_lines.append(f"| {stage} | {stage_status.get(stage, 'NOT_YET_RUN')} |")
    report_lines.extend(
        [
            "",
            "## Frozen stop reason",
            "",
            "- H2S: JOINT_INPUT_PATH_COLLAPSED because K_C_INPUT_PATH_EXACT_ZERO; no AR-only fallback.",
            "- SO2: JOINT_INPUT_PATH_COLLAPSED; relative gain over AR diagnostic = -2.8232861222521232e-05, positive-fold fraction = 0, input prediction variance = 1.971571991679871e-14.",
            "- Frozen gates were relative gain >= 0.01, positive-fold fraction >= 0.75, and input variance > 1e-12.",
            "- Automatic resume stops at the non-PASS E5 marker. E6/E7 are not bypassed.",
            "",
            "## Validation and test boundary",
            "",
            "V21_SRU_FINAL_METRICS.csv contains validation metrics and explicit NOT_YET_RUN rows only.",
            "V21_SRU_ENTITY_METRICS.csv contains validation entities only.",
            "V21_SRU_BOOTSTRAP.csv is NOT_YET_RUN because E5 stopped before freeze.",
            "No v2.1 candidate test metric is present.",
            "",
            "## Runtime and provenance",
            "",
            f"- Server source commit: {code_head}.",
            "- Runtime manager: uv.",
            "- Runtime profile: 32 CPUs, 60 GiB cgroup memory, 8 workers x 4 threads, 40 GiB conservative budget.",
            f"- Data post-audit comparison: {post_audit.get('comparison_to_pre', {}).get('status')}.",
            "",
            "## Interpretation boundary",
            "",
            "This package records a single-dataset SRU development stop. It does not establish benchmark-wide superiority, OOD performance, or a completed test comparison.",
            "",
        ]
    )
    (reports / "V21_SRU_FINAL_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    output_manifest = output / "MANIFEST.json"
    output_sums = output / "SHA256SUMS.txt"
    output_records = _records(output, {output_manifest, output_sums})
    _write_json(output_manifest, {"status": "STOP_STATE_PACKAGE_INPUTS", "files": output_records})
    output_sums.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in output_records),
        encoding="utf-8",
    )

    package_root = paths.project / PACKAGE_NAME
    zip_path = paths.project / f"{PACKAGE_NAME}.zip"
    sidecar_path = paths.project / f"{PACKAGE_NAME}.zip.sha256"
    if package_root.exists() or zip_path.exists() or sidecar_path.exists():
        raise RuntimeError("refusing to overwrite an existing package")
    package_root.mkdir(parents=True)
    shutil.copytree(paths.plan, package_root / "theory_and_plan")
    result_root = package_root / "results"
    result_root.mkdir()
    for name in (
        "FREEZE",
        "DATA_AUDIT",
        "BASELINES",
        "DEVELOPMENT",
        "ASSEMBLY_CARDS",
        "RUN_LOG",
        "REPORTS",
    ):
        source = output / name
        if source.is_dir():
            shutil.copytree(source, result_root / name)
    shutil.copy2(output_manifest, result_root / output_manifest.name)
    shutil.copy2(output_sums, result_root / output_sums.name)
    (package_root / "GIT_HEAD.txt").write_text(code_head + "\n", encoding="utf-8")
    (package_root / "GIT_STATUS.txt").write_text(code_status + "\n", encoding="utf-8")
    (package_root / "STOP_STATE.txt").write_text(
        "E6 freeze, E7 candidate test, and E8 bootstrap were NOT_YET_RUN after the frozen E5 stop.\n",
        encoding="utf-8",
    )

    stage_manifest = package_root / "MANIFEST.json"
    stage_sums = package_root / "SHA256SUMS.txt"
    stage_records = _records(package_root, {stage_manifest, stage_sums})
    _write_json(stage_manifest, {"status": "STOP_STATE_PACKAGE", "files": stage_records})
    stage_sums.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in stage_records),
        encoding="utf-8",
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(package_root.parent))

    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP_INTEGRITY_FAILED: {bad}")
        required = (
            "MANIFEST.json",
            "SHA256SUMS.txt",
            "V21_SRU_FINAL_REPORT.md",
            "V21_SRU_FINAL_METRICS.csv",
            "V21_SRU_FINAL_FREEZE_MANIFEST.json",
        )
        missing = [
            name
            for name in required
            if not any(item.endswith(name) for item in archive.namelist())
        ]
        if missing:
            raise RuntimeError(f"MISSING_REQUIRED_ARTIFACT: {missing}")

    digest = sha256_file(zip_path)
    sidecar_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    summary = {
        "status": "FAILED",
        "stage": "STOP_STATE_REPORT_AND_PACKAGE",
        "zip": zip_path.name,
        "zip_sha256": digest,
        "sidecar": sidecar_path.name,
        "zip_integrity": "PASS",
        "e6_e7_e8": "NOT_YET_RUN",
        "v21_candidate_test_accessed": False,
    }
    _write_json(stop_state / "PACKAGE_SUMMARY.json", summary)
    return summary


def main() -> None:
    project = Path("/root/autodl-tmp/PRISM_V21_SRU_81fe505/PRISM_INDUSTRIAL_BENCHMARK_V1")
    paths = V21Paths(
        project=project,
        shared=Path("/root/autodl-tmp/PRISM_SHARED_DATA_C1"),
        output=project / "results_prism_v2_1_sru",
    )
    print(json.dumps(build_stop_state(paths), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
