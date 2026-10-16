from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cpu_data import sha256_file
from .cpu_selection import regression_metrics
from .stage0 import write_json
from .v21_baselines import load_frozen_baseline_inventory
from .v21_reporting import _align_pair, _entity_metrics, _holm, _moving_block_means
from .v21_views import sru_dynamic_views, sru_input_views
from .v211_config import V211Paths, load_v211_configs


PACKAGE_NAME = "PRISM_V2_1_1_SRU_IMPLEMENTATION_CORRECTION_RESULTS_bundle"
BOOTSTRAP_SEED = 20260806


def _read_predictions(paths: V211Paths, audit: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(paths.output / audit["prediction_path"])
    required = {"sample_id", "entity_id", "origin", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"prediction artifact lacks columns {sorted(missing)}: "
            f"{audit['prediction_path']}"
        )
    return frame


def _key(audit: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(audit["target_head"]),
        str(audit["information_set"]),
        str(audit["availability_scenario"]),
        str(audit["proxy_policy"]),
        str(audit["model"]),
    )


def _stable_seed(identifier: str) -> int:
    suffix = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16)
    return (BOOTSTRAP_SEED + suffix) % (2**32)


def _block_lengths(paths: V211Paths) -> dict[tuple[str, str], int]:
    result = {}
    for view in sru_input_views(paths.shared):
        result[(view.head.head_id, view.information_set)] = max(
            1, view.head.h_steps + view.head.w_steps
        )
    for view in sru_dynamic_views(paths.shared):
        path = (
            paths.output
            / "DEVELOPMENT"
            / "A"
            / view.head.head_id
            / view.availability_scenario
            / view.proxy_policy
            / "RESULT.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        history = int(value.get("a_contract", {}).get("profile", [1, 1])[1])
        result[(view.head.head_id, view.information_set)] = max(
            1, view.head.h_steps + view.head.w_steps, history
        )
    return result


def _comparison_specs(
    inventory: dict[str, Any],
    audits: list[dict[str, Any]],
) -> list[dict[str, str]]:
    available = {_key(audit) for audit in audits}
    specs: list[dict[str, str]] = []
    for head, information_set, availability, proxy in sorted(
        {key[:4] for key in available}
    ):
        prefix = (head, information_set, availability, proxy)
        if information_set == "input_only":
            candidate = "PRISM_V2_1_1_K_C_W"
            comparisons = (
                ("PRISM_CHANNEL_SPECIFIC", "PRIMARY_INPUT"),
                ("HAMMERSTEIN_WIENER", "PRIMARY_INPUT"),
                ("PARALLEL_HAMMERSTEIN", "PRIMARY_INPUT"),
            )
            mechanisms = (("PRISM_V2_1_1_K_C", "MECHANISM_INPUT"),)
        else:
            candidate = "PRISM_V2_1_1_PHYSICS_FIRST"
            comparisons = (
                ("PRISM_PHYSICS_FIRST", "PRIMARY_DYNAMIC"),
                ("ARX", "PRIMARY_DYNAMIC"),
                ("LINEAR_NARX", "PRIMARY_DYNAMIC"),
            )
            mechanisms = ()
        if (*prefix, candidate) not in available:
            continue
        for comparator, family in (*comparisons, *mechanisms):
            if (*prefix, comparator) not in available:
                raise RuntimeError(
                    f"required comparison prediction is missing: {(*prefix, comparator)}"
                )
            specs.append(
                {
                    "target_head": head,
                    "information_set": information_set,
                    "availability_scenario": availability,
                    "proxy_policy": proxy,
                    "candidate": candidate,
                    "comparator": comparator,
                    "comparison_family": family,
                }
            )
        best_key = "|".join(prefix)
        best = str(inventory["best_by_validation"][best_key])
        if best not in {value[0] for value in (*comparisons, *mechanisms)}:
            if (*prefix, best) not in available:
                raise RuntimeError(f"strongest baseline prediction is missing: {best_key}")
            specs.append(
                {
                    "target_head": head,
                    "information_set": information_set,
                    "availability_scenario": availability,
                    "proxy_policy": proxy,
                    "candidate": candidate,
                    "comparator": best,
                    "comparison_family": (
                        "PRIMARY_INPUT"
                        if information_set == "input_only"
                        else "PRIMARY_DYNAMIC"
                    ),
                }
            )
        if information_set == "dynamic":
            joint = "PRISM_V2_1_1_JOINT_KWA"
            for comparator in ("PRISM_K_JOINT_AR", best):
                if (*prefix, joint) in available and (*prefix, comparator) in available:
                    specs.append(
                        {
                            "target_head": head,
                            "information_set": information_set,
                            "availability_scenario": availability,
                            "proxy_policy": proxy,
                            "candidate": joint,
                            "comparator": comparator,
                            "comparison_family": "PRIMARY_DYNAMIC",
                        }
                    )
            dynamic_mechanisms = (
                (
                    "PRISM_V2_1_1_K_C_A_ABLATION",
                    "PRISM_V2_1_1_K_C_DYNAMIC",
                    "K_VS_K_A",
                ),
                (
                    "PRISM_V2_1_1_PHYSICS_FIRST",
                    "PRISM_V2_1_1_K_C_W_DYNAMIC",
                    "K_W_VS_K_W_A",
                ),
                (
                    "PRISM_V2_1_1_J_KWA",
                    "PRISM_V2_1_1_J_KA",
                    "JOINT_KA_VS_JOINT_KWA",
                ),
                (
                    "PRISM_V2_1_1_PHYSICS_FIRST",
                    "PRISM_V2_1_1_PF_A_MU0_ABLATION",
                    "A_SOFT_MU_SELECTED_VS_ZERO",
                ),
            )
            for mechanism_candidate, comparator, family in dynamic_mechanisms:
                if (
                    (*prefix, mechanism_candidate) in available
                    and (*prefix, comparator) in available
                ):
                    specs.append(
                        {
                            "target_head": head,
                            "information_set": information_set,
                            "availability_scenario": availability,
                            "proxy_policy": proxy,
                            "candidate": mechanism_candidate,
                            "comparator": comparator,
                            "comparison_family": family,
                        }
                    )
        else:
            mu0 = "PRISM_V2_1_1_K_C_W_MU0_ABLATION"
            if (*prefix, mu0) in available:
                specs.append(
                    {
                        "target_head": head,
                        "information_set": information_set,
                        "availability_scenario": availability,
                        "proxy_policy": proxy,
                        "candidate": candidate,
                        "comparator": mu0,
                        "comparison_family": "W_SOFT_MU_SELECTED_VS_ZERO",
                    }
                )
    unique = {"|".join(spec.values()): spec for spec in specs}
    return [unique[key] for key in sorted(unique)]


def build_bootstrap(
    paths: V211Paths,
    audits: list[dict[str, Any]],
) -> pd.DataFrame:
    _, v21, _ = load_v211_configs(paths.project)
    replicates = int(v21["statistics"]["paired_block_bootstrap_replicates"])
    if replicates != 500:
        raise RuntimeError("v2.1.1 paired block bootstrap must use 500 replicates")
    inventory = load_frozen_baseline_inventory(paths)
    frames = {_key(audit): _read_predictions(paths, audit) for audit in audits}
    lengths = _block_lengths(paths)
    rows: list[dict[str, Any]] = []
    for spec in _comparison_specs(inventory, audits):
        prefix = (
            spec["target_head"],
            spec["information_set"],
            spec["availability_scenario"],
            spec["proxy_policy"],
        )
        paired = _align_pair(
            frames[(*prefix, spec["candidate"])],
            frames[(*prefix, spec["comparator"])],
        )
        identifier = "|".join(spec.values())
        block_length = lengths[(spec["target_head"], spec["information_set"])]
        samples = _moving_block_means(
            paired,
            block_length,
            replicates,
            _stable_seed(identifier),
        )
        observed = float(
            np.mean(
                paired["loss_difference"].to_numpy(dtype=np.float64),
                dtype=np.float64,
            )
        )
        below = float(np.mean(samples <= 0.0))
        above = float(np.mean(samples >= 0.0))
        rows.append(
            {
                **spec,
                "split": "test",
                "comparison_id": identifier,
                "loss": "squared_error",
                "mean_loss_difference_candidate_minus_comparator": observed,
                "ci_lower": float(np.quantile(samples, 0.025)),
                "ci_upper": float(np.quantile(samples, 0.975)),
                "probability_candidate_better": below,
                "raw_p_value": min(1.0, 2.0 * min(below, above)),
                "holm_p_value": np.nan,
                "bootstrap_replicates": replicates,
                "bootstrap_seed": _stable_seed(identifier),
                "block_length": block_length,
                "paired_rows": len(paired),
            }
        )
    _holm(rows)
    return pd.DataFrame(rows)


def _git(project: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.filemode=false", *args],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _write_output_manifest(paths: V211Paths) -> tuple[Path, Path]:
    manifest_path = paths.output / "MANIFEST.json"
    sums_path = paths.output / "SHA256SUMS.txt"
    excluded = {manifest_path.resolve(), sums_path.resolve()}
    records = []
    for path in sorted(item for item in paths.output.rglob("*") if item.is_file()):
        if path.resolve() in excluded:
            continue
        records.append(
            {
                "path": str(path.relative_to(paths.output)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(manifest_path, {"status": "PASS", "files": records})
    sums_path.write_text(
        "".join(
            f"{record['sha256']}  {record['path']}\n" for record in records
        ),
        encoding="utf-8",
    )
    return manifest_path, sums_path


def package_results(paths: V211Paths, *, completed: bool) -> dict[str, Any]:
    manifest, sums = _write_output_manifest(paths)
    zip_path = paths.project / f"{PACKAGE_NAME}.zip"
    sidecar = paths.project / f"{PACKAGE_NAME}.zip.sha256"
    if zip_path.is_file():
        try:
            with zipfile.ZipFile(zip_path) as existing:
                existing_valid = existing.testzip() is None
        except zipfile.BadZipFile:
            existing_valid = False
        if existing_valid:
            digest = sha256_file(zip_path)
            expected = sidecar.read_text(encoding="utf-8").split()[0] if sidecar.is_file() else None
            if expected in {None, digest}:
                if expected is None:
                    sidecar.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
                return {
                    "status": "ZIP_INTEGRITY_PASS",
                    "zip_path": str(zip_path),
                    "zip_sha256": digest,
                    "sidecar_path": str(sidecar),
                    "resumed_existing_package": True,
                }
    temporary_zip = zip_path.with_suffix(".zip.tmp")
    with tempfile.TemporaryDirectory(
        prefix=f"{PACKAGE_NAME}_stage_", dir=paths.project
    ) as temporary:
        stage = Path(temporary) / PACKAGE_NAME
        (stage / "theory_and_plan").mkdir(parents=True)
        shutil.copytree(paths.plan, stage / "theory_and_plan" / "v2_1_1")
        shutil.copytree(
            paths.inherited_plan, stage / "theory_and_plan" / "inherited_v2_1"
        )
        result_root = stage / "results"
        result_root.mkdir(parents=True)
        for name in (
            "FREEZE",
            "DATA_AUDIT",
            "BASELINES",
            "DEVELOPMENT",
            "ASSEMBLY_CARDS",
            "FINAL",
            "REPORTS",
            "RUN_LOG",
        ):
            source = paths.output / name
            if source.exists():
                shutil.copytree(source, result_root / name)
        shutil.copy2(manifest, result_root / manifest.name)
        shutil.copy2(sums, result_root / sums.name)
        (stage / "GIT_HEAD.txt").write_text(
            _git(paths.project, "rev-parse", "HEAD") + "\n", encoding="utf-8"
        )
        (stage / "GIT_STATUS.txt").write_text(
            _git(paths.project, "status", "--short") + "\n", encoding="utf-8"
        )
        stage_records = []
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            stage_records.append(
                {
                    "path": str(path.relative_to(stage)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        write_json(stage / "MANIFEST.json", {"status": "PASS", "files": stage_records})
        (stage / "SHA256SUMS.txt").write_text(
            "".join(
                f"{record['sha256']}  {record['path']}\n"
                for record in stage_records
            ),
            encoding="utf-8",
        )
        with zipfile.ZipFile(
            temporary_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(item for item in stage.rglob("*") if item.is_file()):
                archive.write(path, path.relative_to(stage.parent))
    with zipfile.ZipFile(temporary_zip) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP_INTEGRITY_FAILED: {bad}")
        names = set(archive.namelist())
        required = ["MANIFEST.json", "SHA256SUMS.txt"]
        if completed:
            required.extend(
                [
                    "V211_SRU_FINAL_REPORT.md",
                    "V211_SRU_FINAL_METRICS.csv",
                    "V211_SRU_FINAL_FREEZE_MANIFEST.json",
                ]
            )
        else:
            required.extend(
                ["V211_SRU_DEVELOPMENT_STOP_REPORT.md", "V211_DEVELOPMENT_DECISION.json"]
            )
        for name in required:
            if not any(value.endswith(name) for value in names):
                raise RuntimeError(f"MISSING_REQUIRED_ARTIFACT: {name}")
    temporary_zip.replace(zip_path)
    digest = sha256_file(zip_path)
    sidecar.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return {
        "status": "ZIP_INTEGRITY_PASS",
        "zip_path": str(zip_path),
        "zip_sha256": digest,
        "sidecar_path": str(sidecar),
    }


def build_stop_report_and_package(
    paths: V211Paths,
    decision: dict[str, Any],
) -> dict[str, Any]:
    final = paths.output / "FINAL"
    reports = paths.output / "REPORTS"
    final.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    write_json(
        final / "V211_SRU_STOP_STATUS.json",
        {
            "status": "V2_1_1_DEVELOPMENT_STOP",
            "reason": "NO_SUPPORTED_INCREMENT_AFTER_IMPLEMENTATION_REPAIR",
            "test_status": "TEST_NOT_ACCESSED",
            "test_accessed": False,
        },
    )
    pd.DataFrame(decision.get("candidate_comparisons", [])).to_csv(
        final / "V211_SRU_DEVELOPMENT_GATE_CANDIDATES.csv", index=False
    )
    report = reports / "V211_SRU_DEVELOPMENT_STOP_REPORT.md"
    failed = [key for key, value in decision.get("checks", {}).items() if not value]
    report.write_text(
        "\n".join(
            (
                "# PRISM v2.1.1 SRU implementation-correction report",
                "",
                "Status: `V2_1_1_DEVELOPMENT_STOP`.",
                "",
                "Reason: `NO_SUPPORTED_INCREMENT_AFTER_IMPLEMENTATION_REPAIR`.",
                "",
                "Test status: `TEST_NOT_ACCESSED`.",
                "",
                "The registered implementation repairs were executed on the two SRU heads.",
                "The frozen development continue gate did not pass, so E6R/E7R/E8R test",
                "evaluation was not entered and no post-test retuning was performed.",
                "",
                "Failed frozen checks: " + (", ".join(failed) if failed else "none"),
                "",
            )
        ),
        encoding="utf-8",
    )
    package = package_results(paths, completed=False)
    return {
        "status": "V2_1_1_DEVELOPMENT_STOP",
        "stage": "E5_5_DEVELOPMENT_GATE_AND_STOP_PACKAGE",
        "report": str(report.relative_to(paths.output)),
        "package": package,
        "test_accessed": False,
    }


def _resource_summary(paths: V211Paths) -> dict[str, Any]:
    records = []
    for path in sorted((paths.output / "DEVELOPMENT").rglob("RESULT.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "path": str(path.relative_to(paths.output)),
                "stage": value.get("stage"),
                "target_head": value.get("target_head"),
                "elapsed_seconds": value.get("elapsed_seconds"),
                "status": value.get("status"),
            }
        )
    return {
        "status": "PASS",
        "runtime_manager": "uv",
        "development_records": records,
        "total_recorded_elapsed_seconds": float(
            sum(
                float(record["elapsed_seconds"])
                for record in records
                if record["elapsed_seconds"] is not None
            )
        ),
        "test_accessed": True,
    }


def build_report_and_package(
    paths: V211Paths,
    access_audit: dict[str, Any],
) -> dict[str, Any]:
    audits = list(access_audit["models"])
    metrics = []
    entities = []
    for audit in audits:
        frame = _read_predictions(paths, audit)
        metrics.append(
            {
                key: audit.get(key)
                for key in (
                    "target_head",
                    "information_set",
                    "availability_scenario",
                    "proxy_policy",
                    "model",
                    "mse",
                    "rmse",
                    "mae",
                    "r2",
                    "nrmse",
                    "rows",
                    "parameter_count",
                    "effective_df",
                    "fit_and_prediction_seconds",
                    "prediction_path",
                    "prediction_sha256",
                )
            }
        )
        entities.extend(
            {
                "target_head": audit["target_head"],
                "information_set": audit["information_set"],
                "model": audit["model"],
                **row,
            }
            for row in _entity_metrics(frame)
        )
    metric_index = {
        (row["target_head"], row["information_set"], row["model"]): row
        for row in metrics
    }
    for row in metrics:
        persistence = metric_index.get(
            (row["target_head"], row["information_set"], "PERSISTENCE")
        )
        ar = metric_index.get((row["target_head"], "dynamic", "AR"))
        row["relative_persistence_skill"] = (
            np.nan
            if persistence is None or float(persistence["mse"]) == 0.0
            else 1.0 - float(row["mse"]) / float(persistence["mse"])
        )
        row["dynamic_relative_ar_skill"] = (
            np.nan
            if row["information_set"] != "dynamic"
            or ar is None
            or float(ar["mse"]) == 0.0
            else 1.0 - float(row["mse"]) / float(ar["mse"])
        )
    final = paths.output / "FINAL"
    final.mkdir(parents=True, exist_ok=True)
    metrics_path = final / "V211_SRU_FINAL_METRICS.csv"
    entities_path = final / "V211_SRU_ENTITY_METRICS.csv"
    bootstrap_path = final / "V211_SRU_BOOTSTRAP.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(entities).to_csv(entities_path, index=False)
    build_bootstrap(paths, audits).to_csv(bootstrap_path, index=False)
    (final / "V211_SRU_MODEL_AUDIT.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audits),
        encoding="utf-8",
    )
    cards = sorted((paths.output / "ASSEMBLY_CARDS").rglob("*_ASSEMBLY_CARD.json"))
    (final / "V211_SRU_ASSEMBLY_CARDS.jsonl").write_text(
        "".join(path.read_text(encoding="utf-8").strip() + "\n" for path in cards),
        encoding="utf-8",
    )
    write_json(final / "V211_SRU_RESOURCE_AUDIT.json", _resource_summary(paths))
    report = paths.output / "REPORTS" / "V211_SRU_FINAL_REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            (
                "# PRISM v2.1.1 SRU implementation-correction final report",
                "",
                "Status: `COMPLETED`.",
                "",
                "The scope is limited to the two frozen SRU heads. Candidate test data",
                "was accessed only after the v2.1.1 development gate and E6R freeze.",
                "Primary comparisons use paired per-sample predictions, the frozen",
                "500-replicate moving-block bootstrap, and within-family Holm correction.",
                "",
                "Summary label: `SRU_WITHIN_DATASET_SUMMARY`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    package = package_results(paths, completed=True)
    return {
        "status": "COMPLETED",
        "stage": "E8R_REPORT_AND_PACKAGE",
        "within_dataset_summary_label": "SRU_WITHIN_DATASET_SUMMARY",
        "test_accessed": True,
        "metrics_rows": len(metrics),
        "entity_metric_rows": len(entities),
        "report": str(report.relative_to(paths.output)),
        "package": package,
    }
