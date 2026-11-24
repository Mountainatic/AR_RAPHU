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
from .v21_config import V21Paths, load_v21_config
from .v21_views import sru_dynamic_views, sru_input_views


PACKAGE_NAME = "PRISM_V2_1_SRU_STAGEWISE_ROUTED_RESULTS_bundle"
BOOTSTRAP_SEED = 20260804


def _read_predictions(paths: V21Paths, audit: dict[str, Any]) -> pd.DataFrame:
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


def _prediction_index(
    paths: V21Paths,
    audits: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str, str], pd.DataFrame]:
    result = {}
    for audit in audits:
        key = _key(audit)
        if key in result:
            raise RuntimeError(f"duplicate final prediction key: {key}")
        result[key] = _read_predictions(paths, audit)
    return result


def _align_pair(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if not np.array_equal(
        left["sample_id"].astype(str).to_numpy(),
        right["sample_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError("paired comparison sample_id mismatch")
    if not np.array_equal(
        left["y_true"].to_numpy(dtype=np.float64),
        right["y_true"].to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise RuntimeError("paired comparison y_true mismatch")
    return pd.DataFrame(
        {
            "sample_id": left["sample_id"].astype(str),
            "entity_id": left["entity_id"].astype(str),
            "origin": left["origin"].to_numpy(dtype=np.int64),
            "loss_difference": (
                np.square(
                    left["y_true"].to_numpy(dtype=np.float64)
                    - left["y_pred"].to_numpy(dtype=np.float64)
                )
                - np.square(
                    right["y_true"].to_numpy(dtype=np.float64)
                    - right["y_pred"].to_numpy(dtype=np.float64)
                )
            ),
        }
    )


def _moving_block_means(
    paired: pd.DataFrame,
    block_length: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    if block_length < 1 or replicates < 1:
        raise ValueError("invalid block bootstrap settings")
    rng = np.random.default_rng(seed)
    entities = []
    for _, group in paired.sort_values(["entity_id", "origin"]).groupby(
        "entity_id",
        sort=False,
    ):
        values = group["loss_difference"].to_numpy(dtype=np.float64)
        length = min(block_length, len(values))
        starts = np.arange(max(1, len(values) - length + 1), dtype=np.int64)
        offsets = np.arange(length, dtype=np.int64)
        blocks = (len(values) + length - 1) // length
        entities.append((values, starts, offsets, blocks))
    result = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        pieces = []
        for values, starts, offsets, blocks in entities:
            sampled_starts = rng.choice(starts, size=blocks)
            indices = (
                sampled_starts[:, np.newaxis] + offsets[np.newaxis, :]
            ).reshape(-1)[: len(values)]
            pieces.append(values[indices])
        result[replicate] = float(np.mean(np.concatenate(pieces), dtype=np.float64))
    return result


def _stable_seed(identifier: str) -> int:
    suffix = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16)
    return (BOOTSTRAP_SEED + suffix) % (2**32)


def _holm(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (
            str(row["target_head"]),
            str(row["information_set"]),
            str(row["comparison_family"]),
        )
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: float(rows[index]["raw_p_value"]))
        running = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * float(rows[index]["raw_p_value"]))
            running = max(running, adjusted)
            rows[index]["holm_p_value"] = running


def _block_lengths(paths: V21Paths) -> dict[tuple[str, str], int]:
    result = {}
    for view in sru_input_views(paths.shared):
        result[(view.head.head_id, view.information_set)] = max(
            1,
            view.head.h_steps + view.head.w_steps,
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
            1,
            view.head.h_steps + view.head.w_steps,
            history,
        )
    return result


def _comparison_specs(
    inventory: dict[str, Any],
    audits: list[dict[str, Any]],
) -> list[dict[str, str]]:
    available = {_key(audit) for audit in audits}
    specs = []
    view_keys = sorted({key[:4] for key in available})
    for head, information_set, availability, proxy in view_keys:
        prefix = (head, information_set, availability, proxy)
        if information_set == "input_only":
            candidate = "PRISM_V2_1_K_C_W"
            comparisons = (
                ("PRISM_CHANNEL_SPECIFIC", "PRIMARY_INPUT"),
                ("HAMMERSTEIN_WIENER", "PRIMARY_INPUT"),
                ("PARALLEL_HAMMERSTEIN", "PRIMARY_INPUT"),
            )
            mechanism = (("PRISM_V2_1_K_C", "MECHANISM_INPUT"),)
        else:
            candidate = "PRISM_V2_1_PHYSICS_FIRST"
            comparisons = (
                ("PRISM_PHYSICS_FIRST", "PRIMARY_DYNAMIC"),
                ("ARX", "PRIMARY_DYNAMIC"),
                ("LINEAR_NARX", "PRIMARY_DYNAMIC"),
            )
            mechanism = ()
        candidate_key = (head, information_set, availability, proxy, candidate)
        if candidate_key not in available:
            continue
        for comparator, family in (*comparisons, *mechanism):
            comparator_key = (head, information_set, availability, proxy, comparator)
            if comparator_key not in available:
                raise RuntimeError(f"required comparison prediction is missing: {comparator_key}")
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
        best_key = "|".join((head, information_set, availability, proxy))
        best = str(inventory["best_by_validation"][best_key])
        if best not in {item[0] for item in comparisons}:
            specs.append(
                {
                    "target_head": head,
                    "information_set": information_set,
                    "availability_scenario": availability,
                    "proxy_policy": proxy,
                    "candidate": candidate,
                    "comparator": best,
                    "comparison_family": (
                        "PRIMARY_INPUT" if information_set == "input_only" else "PRIMARY_DYNAMIC"
                    ),
                }
            )
        if information_set == "dynamic":
            joint_key = (head, information_set, availability, proxy, "PRISM_V2_1_JOINT_KWA")
            old_joint_key = (head, information_set, availability, proxy, "PRISM_K_JOINT_AR")
            if joint_key in available and old_joint_key in available:
                specs.append(
                    {
                        "target_head": head,
                        "information_set": information_set,
                        "availability_scenario": availability,
                        "proxy_policy": proxy,
                        "candidate": "PRISM_V2_1_JOINT_KWA",
                        "comparator": "PRISM_K_JOINT_AR",
                        "comparison_family": "PRIMARY_DYNAMIC",
                    }
                )
            best_joint_key = (head, information_set, availability, proxy, best)
            if joint_key in available and best_joint_key in available:
                specs.append(
                    {
                        "target_head": head,
                        "information_set": information_set,
                        "availability_scenario": availability,
                        "proxy_policy": proxy,
                        "candidate": "PRISM_V2_1_JOINT_KWA",
                        "comparator": best,
                        "comparison_family": "PRIMARY_DYNAMIC",
                    }
                )
            dynamic_mechanisms = (
                (
                    "PRISM_V2_1_K_C_A_ABLATION",
                    "PRISM_V2_1_K_C_DYNAMIC",
                    "K_VS_K_A",
                ),
                (
                    "PRISM_V2_1_PHYSICS_FIRST",
                    "PRISM_V2_1_K_C_W_DYNAMIC",
                    "K_W_VS_K_W_A",
                ),
                (
                    "PRISM_V2_1_J_KWA",
                    "PRISM_V2_1_J_KA",
                    "JOINT_KA_VS_JOINT_KWA",
                ),
                (
                    "PRISM_V2_1_PHYSICS_FIRST",
                    "PRISM_V2_1_PF_A_MU0_ABLATION",
                    "A_SOFT_MU_SELECTED_VS_ZERO",
                ),
            )
            for mechanism_candidate, mechanism_comparator, label in dynamic_mechanisms:
                if (
                    (*prefix, mechanism_candidate) in available
                    and (*prefix, mechanism_comparator) in available
                ):
                    specs.append(
                        {
                            "target_head": head,
                            "information_set": information_set,
                            "availability_scenario": availability,
                            "proxy_policy": proxy,
                            "candidate": mechanism_candidate,
                            "comparator": mechanism_comparator,
                            "comparison_family": label,
                        }
                    )
        elif information_set == "input_only":
            mu0_key = (*prefix, "PRISM_V2_1_K_C_W_MU0_ABLATION")
            if mu0_key in available:
                specs.append(
                    {
                        "target_head": head,
                        "information_set": information_set,
                        "availability_scenario": availability,
                        "proxy_policy": proxy,
                        "candidate": "PRISM_V2_1_K_C_W",
                        "comparator": "PRISM_V2_1_K_C_W_MU0_ABLATION",
                        "comparison_family": "W_SOFT_MU_SELECTED_VS_ZERO",
                    }
                )
    unique = {}
    for spec in specs:
        identifier = "|".join(spec.values())
        unique[identifier] = spec
    return [unique[key] for key in sorted(unique)]


def build_bootstrap(
    paths: V21Paths,
    audits: list[dict[str, Any]],
) -> pd.DataFrame:
    config = load_v21_config(paths.project)
    replicates = int(config["statistics"]["paired_block_bootstrap_replicates"])
    inventory = load_frozen_baseline_inventory(paths)
    frames = _prediction_index(paths, audits)
    lengths = _block_lengths(paths)
    rows: list[dict[str, Any]] = []
    for spec in _comparison_specs(inventory, audits):
        prefix = (
            spec["target_head"],
            spec["information_set"],
            spec["availability_scenario"],
            spec["proxy_policy"],
        )
        candidate = frames[(*prefix, spec["candidate"])]
        comparator = frames[(*prefix, spec["comparator"])]
        paired = _align_pair(candidate, comparator)
        identifier = "|".join(spec.values())
        block_length = lengths[(spec["target_head"], spec["information_set"])]
        samples = _moving_block_means(
            paired,
            block_length,
            replicates,
            _stable_seed(identifier),
        )
        observed = float(np.mean(paired["loss_difference"], dtype=np.float64))
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


def _entity_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for entity, group in frame.groupby("entity_id", sort=False):
        rows.append(
            {
                "entity_id": entity,
                **regression_metrics(
                    group["y_true"].to_numpy(dtype=np.float64),
                    group["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return rows


def _resource_summary(paths: V21Paths) -> dict[str, Any]:
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
    finite = [
        float(record["elapsed_seconds"])
        for record in records
        if record["elapsed_seconds"] is not None
    ]
    return {
        "status": "PASS",
        "runtime_manager": "uv",
        "development_records": records,
        "total_recorded_elapsed_seconds": float(sum(finite)),
        "test_accessed": True,
    }


def _write_output_manifest(paths: V21Paths) -> tuple[Path, Path]:
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
        "".join(f"{record['sha256']}  {record['path']}\n" for record in records),
        encoding="utf-8",
    )
    return manifest_path, sums_path


def _git(project: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.filemode=false", *args],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def package_results(paths: V21Paths) -> dict[str, Any]:
    manifest, sums = _write_output_manifest(paths)
    zip_path = paths.project / f"{PACKAGE_NAME}.zip"
    sidecar = paths.project / f"{PACKAGE_NAME}.zip.sha256"
    if zip_path.exists() or sidecar.exists():
        raise RuntimeError("refusing to overwrite an existing v2.1 result package")
    with tempfile.TemporaryDirectory(prefix=f"{PACKAGE_NAME}_stage_", dir=paths.project) as temporary:
        stage = Path(temporary) / PACKAGE_NAME
        shutil.copytree(paths.plan, stage / "theory_and_plan")
        result_root = stage / "results"
        result_root.mkdir(parents=True)
        for name in ("FREEZE", "DATA_AUDIT", "BASELINES", "ASSEMBLY_CARDS", "FINAL", "REPORTS"):
            source = paths.output / name
            if not source.exists():
                raise RuntimeError(f"required package directory is missing: {source}")
            shutil.copytree(source, result_root / name)
        shutil.copy2(manifest, result_root / manifest.name)
        shutil.copy2(sums, result_root / sums.name)
        (stage / "GIT_HEAD.txt").write_text(
            _git(paths.project, "rev-parse", "HEAD") + "\n",
            encoding="utf-8",
        )
        (stage / "GIT_STATUS.txt").write_text(
            _git(paths.project, "status", "--short") + "\n",
            encoding="utf-8",
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
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(item for item in stage.rglob("*") if item.is_file()):
                archive.write(path, path.relative_to(stage.parent))
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP_INTEGRITY_FAILED: {bad}")
        names = set(archive.namelist())
        for required in (
            "MANIFEST.json",
            "SHA256SUMS.txt",
            "V21_SRU_FINAL_REPORT.md",
            "V21_SRU_FINAL_METRICS.csv",
            "V21_SRU_FINAL_FREEZE_MANIFEST.json",
        ):
            if not any(name.endswith(required) for name in names):
                raise RuntimeError(f"MISSING_REQUIRED_ARTIFACT: {required}")
    digest = sha256_file(zip_path)
    sidecar.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return {
        "status": "ZIP_INTEGRITY_PASS",
        "zip_path": str(zip_path),
        "zip_sha256": digest,
        "sidecar_path": str(sidecar),
    }


def build_report_and_package(
    paths: V21Paths,
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
    metrics_path = final / "V21_SRU_FINAL_METRICS.csv"
    entities_path = final / "V21_SRU_ENTITY_METRICS.csv"
    bootstrap_path = final / "V21_SRU_BOOTSTRAP.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(entities).to_csv(entities_path, index=False)
    build_bootstrap(paths, audits).to_csv(bootstrap_path, index=False)
    (final / "V21_SRU_MODEL_AUDIT.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audits),
        encoding="utf-8",
    )
    cards = sorted((paths.output / "ASSEMBLY_CARDS").rglob("*_ASSEMBLY_CARD.json"))
    (final / "V21_SRU_ASSEMBLY_CARDS.jsonl").write_text(
        "".join(path.read_text(encoding="utf-8").strip() + "\n" for path in cards),
        encoding="utf-8",
    )
    resources = _resource_summary(paths)
    write_json(final / "V21_SRU_RESOURCE_AUDIT.json", resources)
    report_path = paths.output / "REPORTS" / "V21_SRU_FINAL_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            (
                "# PRISM v2.1 SRU final report",
                "",
                "Status: COMPLETED.",
                "",
                "The result scope is limited to the two registered SRU heads.",
                "No benchmark-wide rank or cross-dataset generalization claim is reported.",
                "All primary comparisons use paired sample-level predictions and the frozen",
                "500-replicate moving-block bootstrap with within-family Holm correction.",
                "",
                "Summary label: `SRU_WITHIN_DATASET_SUMMARY`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    package = package_results(paths)
    return {
        "status": "COMPLETED",
        "stage": "E8_REPORT_AND_PACKAGE",
        "within_dataset_summary_label": "SRU_WITHIN_DATASET_SUMMARY",
        "test_accessed": True,
        "metrics_rows": len(metrics),
        "entity_metric_rows": len(entities),
        "report": str(report_path.relative_to(paths.output)),
        "package": package,
    }
