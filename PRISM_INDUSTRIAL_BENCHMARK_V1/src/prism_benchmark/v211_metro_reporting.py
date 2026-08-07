from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cpu_data import ViewSpec, sha256_file
from .stage0 import write_json
from .v2_runtime import run_parallel
from .v211_a import EXACT_ZERO
from .v211_joint import J_K, J_KA, J_KW, J_KWA
from .v211_metro_config import (
    ACTIVE_HEAD,
    EVIDENCE_CLASS,
    PROTOCOL_ID,
    SOURCE_COMMIT,
    MetroV211Paths,
    effective_worker_count,
    git_value,
    load_metro_config,
    runtime_parallelism_audit,
)
from .v211_metro_views import metro_p60_dynamic_views
from .v211_w import IDENTITY


PACKAGE_NAME = "PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_RESULTS_bundle"
BOOTSTRAP_SEED = 20260807
PF_CANDIDATES = ("KC", "KCW", "KCA", "KCWA", "PF_SELECTED")
JOINT_CANDIDATES = (J_K, J_KW, J_KA, J_KWA, "J_SELECTED")
COMPARISONS = (
    ("KCW", "KC", "PF_W_MARGINAL"),
    ("KCWA", "KCA", "PF_W_MARGINAL"),
    ("PF_SELECTED", "KC", "PF_SELECTED_INCREMENT"),
    (J_KW, J_K, "JOINT_W_MARGINAL"),
    (J_KWA, J_KA, "JOINT_W_MARGINAL"),
    ("J_SELECTED", J_K, "JOINT_SELECTED_INCREMENT"),
)
PRIMARY_W_COMPARISONS = {
    "KCW_vs_KC",
    "KCWA_vs_KCA",
    "J_KW_vs_J_K",
    "J_KWA_vs_J_KA",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_seed(identifier: str) -> int:
    suffix = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16)
    return (BOOTSTRAP_SEED + suffix) % (2**32)


def _result_path(paths: MetroV211Paths, stage: str, view: ViewSpec) -> Path:
    if stage in {"C", "W"}:
        return (
            paths.output
            / "DEVELOPMENT"
            / stage
            / view.head.head_id
            / view.proxy_policy
            / "RESULT.json"
        )
    return (
        paths.output
        / "DEVELOPMENT"
        / stage
        / view.head.head_id
        / view.availability_scenario
        / view.proxy_policy
        / "RESULT.json"
    )


def _contract_path(paths: MetroV211Paths, view: ViewSpec) -> Path:
    return (
        paths.output
        / "FINAL"
        / "contracts"
        / view.proxy_policy
        / "FINAL_MODEL_CONTRACT.json"
    )


def _normalize_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in frame and "view_sample_id" in frame:
        frame = frame.rename(columns={"view_sample_id": "sample_id"})
    required = {"sample_id", "entity_id", "origin", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"prediction artifact lacks columns: {sorted(missing)}")
    if not np.isfinite(frame["y_pred"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("prediction artifact contains non-finite predictions")
    return frame


def _development_predictions(
    paths: MetroV211Paths, view: ViewSpec
) -> dict[str, pd.DataFrame]:
    a_result = _read(_result_path(paths, "A", view))
    joint_result = _read(_result_path(paths, "JOINT", view))
    pf = pd.read_parquet(paths.output / a_result["nested_validation_prediction_path"])
    result = {
        candidate: _normalize_prediction(group.reset_index(drop=True))
        for candidate, group in pf.groupby("candidate", sort=False)
    }
    for route in (J_K, J_KW, J_KA, J_KWA):
        payload = joint_result["route_materializations"][route]
        frame = pd.read_parquet(paths.output / payload["prediction_path"])
        result[route] = _normalize_prediction(frame)
    result["J_SELECTED"] = result[str(joint_result["final_selected_candidate"])].copy()
    missing = set((*PF_CANDIDATES, *JOINT_CANDIDATES)) - set(result)
    if missing:
        raise RuntimeError(f"development prediction routes are missing: {sorted(missing)}")
    return result


def _final_predictions(
    paths: MetroV211Paths, view: ViewSpec, split: str
) -> dict[str, pd.DataFrame]:
    contract = _read(_contract_path(paths, view))
    return {
        candidate: _normalize_prediction(
            pd.read_parquet(paths.output / contract["prediction_paths"][split][candidate])
        )
        for candidate in (*PF_CANDIDATES, *JOINT_CANDIDATES)
    }


def _aligned_loss_matrices(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference = frames["KC"]
    sample_ids = reference["sample_id"].astype(str).to_numpy()
    target = reference["y_true"].to_numpy(dtype=np.float64)
    entity = reference["entity_id"].astype(str).to_numpy()
    origin = reference["origin"].to_numpy(dtype=np.int64)
    differences = []
    comparators = []
    for candidate, comparator, _ in COMPARISONS:
        left = frames[candidate]
        right = frames[comparator]
        if not np.array_equal(sample_ids, left["sample_id"].astype(str).to_numpy()):
            raise RuntimeError(f"sample_id mismatch for {candidate}")
        if not np.array_equal(sample_ids, right["sample_id"].astype(str).to_numpy()):
            raise RuntimeError(f"sample_id mismatch for {comparator}")
        left_target = left["y_true"].to_numpy(dtype=np.float64)
        right_target = right["y_true"].to_numpy(dtype=np.float64)
        if not np.array_equal(target, left_target, equal_nan=True):
            raise RuntimeError(f"target mismatch for {candidate}")
        if not np.array_equal(target, right_target, equal_nan=True):
            raise RuntimeError(f"target mismatch for {comparator}")
        candidate_loss = np.square(
            target - left["y_pred"].to_numpy(dtype=np.float64)
        )
        comparator_loss = np.square(
            target - right["y_pred"].to_numpy(dtype=np.float64)
        )
        differences.append(candidate_loss - comparator_loss)
        comparators.append(comparator_loss)
    return (
        entity,
        origin,
        np.column_stack(differences),
        np.column_stack(comparators),
    )


def moving_block_matrix_means(
    entity: np.ndarray,
    origin: np.ndarray,
    values: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Moving-block bootstrap for several paired statistics on shared draws."""
    if block_length < 1 or replicates < 1:
        raise ValueError("invalid moving-block bootstrap settings")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(matrix) != len(entity) or len(matrix) != len(origin):
        raise ValueError("bootstrap arrays have different row counts")
    order = np.lexsort((np.asarray(origin, dtype=np.int64), entity.astype(str)))
    sorted_entity = entity.astype(str)[order]
    matrix = matrix[order]
    boundaries = np.flatnonzero(sorted_entity[1:] != sorted_entity[:-1]) + 1
    groups = np.split(np.arange(len(matrix), dtype=np.int64), boundaries)
    prefixes: list[tuple[np.ndarray, int, int, int]] = []
    for indices in groups:
        count = len(indices)
        length = min(int(block_length), count)
        prefix = np.vstack(
            [
                np.zeros((1, matrix.shape[1]), dtype=np.float64),
                np.cumsum(matrix[indices], axis=0, dtype=np.float64),
            ]
        )
        block_count = int(math.ceil(count / length))
        prefixes.append((prefix, count, length, block_count))
    rng = np.random.default_rng(seed)
    output = np.empty((replicates, matrix.shape[1]), dtype=np.float64)
    total_rows = max(1, len(matrix))
    for replicate in range(replicates):
        total = np.zeros(matrix.shape[1], dtype=np.float64)
        for prefix, count, length, block_count in prefixes:
            maximum_start = count - length
            starts = rng.integers(0, maximum_start + 1, size=block_count)
            takes = np.full(block_count, length, dtype=np.int64)
            takes[-1] = count - length * (block_count - 1)
            total += np.sum(prefix[starts + takes] - prefix[starts], axis=0)
        output[replicate] = total / total_rows
    return output


def _block_length(paths: MetroV211Paths, view: ViewSpec) -> int:
    a_result = _read(_result_path(paths, "A", view))
    profile = a_result.get("a_contract", {}).get("profile")
    history = int(profile[1]) if isinstance(profile, list) and len(profile) == 2 else 0
    return max(1, int(view.head.h_steps + view.head.w_steps), history)


def _bootstrap_view(
    paths: MetroV211Paths,
    view: ViewSpec,
    replicates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_length = _block_length(paths, view)
    split_frames = {"validation": _development_predictions(paths, view)}
    split_frames.update(
        {split: _final_predictions(paths, view, split) for split in ("test", "ood")}
    )
    for split, frames in split_frames.items():
        entity, origin, difference, comparator = _aligned_loss_matrices(frames)
        seed = _stable_seed(f"{view.relative_root.as_posix()}|{split}")
        samples = moving_block_matrix_means(
            entity,
            origin,
            np.concatenate([difference, comparator], axis=1),
            block_length=block_length,
            replicates=replicates,
            seed=seed,
        )
        difference_samples = samples[:, : len(COMPARISONS)]
        comparator_samples = samples[:, len(COMPARISONS) :]
        observed_difference = np.mean(difference, axis=0, dtype=np.float64)
        observed_comparator = np.mean(comparator, axis=0, dtype=np.float64)
        observed_relative = -observed_difference / np.maximum(
            observed_comparator, np.finfo(np.float64).tiny
        )
        relative_samples = -difference_samples / np.maximum(
            comparator_samples, np.finfo(np.float64).tiny
        )
        for index, (candidate, comparator_name, family) in enumerate(COMPARISONS):
            below = float(np.mean(difference_samples[:, index] <= 0.0))
            above = float(np.mean(difference_samples[:, index] >= 0.0))
            rows.append(
                {
                    "target_head": view.head.head_id,
                    "view": view.relative_root.as_posix(),
                    "availability_scenario": view.availability_scenario,
                    "proxy_policy": view.proxy_policy,
                    "split": split,
                    "candidate": candidate,
                    "comparator": comparator_name,
                    "comparison_id": f"{candidate}_vs_{comparator_name}",
                    "comparison_family": family,
                    "mean_mse_difference_candidate_minus_comparator": float(
                        observed_difference[index]
                    ),
                    "relative_mse_improvement": float(observed_relative[index]),
                    "mse_difference_ci_lower": float(
                        np.quantile(difference_samples[:, index], 0.025)
                    ),
                    "mse_difference_ci_upper": float(
                        np.quantile(difference_samples[:, index], 0.975)
                    ),
                    "relative_improvement_ci_lower": float(
                        np.quantile(relative_samples[:, index], 0.025)
                    ),
                    "relative_improvement_ci_upper": float(
                        np.quantile(relative_samples[:, index], 0.975)
                    ),
                    "probability_candidate_better": below,
                    "raw_p_value": min(1.0, 2.0 * min(below, above)),
                    "holm_p_value": float("nan"),
                    "holm_alpha": 0.05,
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                    "block_length": block_length,
                    "paired_rows": len(difference),
                }
            )
    return rows


def _holm(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((str(row["proxy_policy"]), str(row["split"])), []).append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: float(rows[index]["raw_p_value"]))
        running = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(
                1.0, (count - rank) * float(rows[index]["raw_p_value"])
            )
            running = max(running, adjusted)
            rows[index]["holm_p_value"] = running
            rows[index]["holm_reject_0_05"] = bool(running <= 0.05)
            lower = float(rows[index]["mse_difference_ci_lower"])
            upper = float(rows[index]["mse_difference_ci_upper"])
            rows[index]["longest_block_conclusion"] = (
                "CANDIDATE_BETTER_SUPPORTED"
                if upper < 0.0 and running <= 0.05
                else "CANDIDATE_WORSE_SUPPORTED"
                if lower > 0.0 and running <= 0.05
                else "NEUTRAL_OR_TRANSFER_UNSTABLE"
            )


def _metric_rows(paths: MetroV211Paths, views: list[ViewSpec]) -> list[dict[str, Any]]:
    rows = []
    for view in views:
        contract = _read(_contract_path(paths, view))
        for metric in contract["metrics"]:
            candidate = str(metric["candidate"])
            rows.append(
                {
                    "target_head": view.head.head_id,
                    "view": view.relative_root.as_posix(),
                    "availability_scenario": view.availability_scenario,
                    "proxy_policy": view.proxy_policy,
                    "selected_pf_route": contract["selected_pf"],
                    "selected_joint_route": contract["selected_joint"],
                    "selection_role": (
                        "FORMAL_SELECTED_ALIAS"
                        if candidate in {"PF_SELECTED", "J_SELECTED"}
                        else "PRE_REGISTERED_NESTED_ROUTE"
                    ),
                    **metric,
                }
            )
    return rows


def _supported(row: Mapping[str, Any], *, better: bool) -> bool:
    boundary = (
        float(row["mse_difference_ci_upper"])
        if better
        else float(row["mse_difference_ci_lower"])
    )
    return bool(
        float(row["holm_p_value"]) <= 0.05
        and (boundary < 0.0 if better else boundary > 0.0)
    )


def _selection_transfer_audit(
    paths: MetroV211Paths,
    views: list[ViewSpec],
    bootstrap: pd.DataFrame,
) -> dict[str, Any]:
    audited = []
    for view in views:
        w_result = _read(_result_path(paths, "W", view))
        a_result = _read(_result_path(paths, "A", view))
        joint_result = _read(_result_path(paths, "JOINT", view))
        pf_w_active = w_result["w_contract"]["family"] != IDENTITY
        a_active = a_result["a_contract"]["family"] != EXACT_ZERO
        selected_joint = str(joint_result["final_selected_candidate"])
        joint_w_active = selected_joint in {J_KW, J_KWA}
        pf_comparison = "KCWA_vs_KCA" if a_active else "KCW_vs_KC"
        joint_comparison = (
            "J_KWA_vs_J_KA"
            if selected_joint == J_KWA
            else "J_KW_vs_J_K"
            if selected_joint == J_KW
            else "J_KWA_vs_J_KA"
            if selected_joint == J_KA
            else "J_KW_vs_J_K"
        )

        def row(split: str, comparison: str) -> dict[str, Any]:
            subset = bootstrap[
                (bootstrap["proxy_policy"] == view.proxy_policy)
                & (bootstrap["split"] == split)
                & (bootstrap["comparison_id"] == comparison)
            ]
            if len(subset) != 1:
                raise RuntimeError((view.proxy_policy, split, comparison, len(subset)))
            return subset.iloc[0].to_dict()

        pf_transfer = bool(
            pf_w_active
            and _supported(row("test", pf_comparison), better=True)
            and _supported(row("ood", pf_comparison), better=True)
        )
        joint_transfer = bool(
            joint_w_active
            and _supported(row("test", joint_comparison), better=True)
            and _supported(row("ood", joint_comparison), better=True)
        )
        selected_ood_risk = bool(
            (pf_w_active and _supported(row("ood", pf_comparison), better=False))
            or (
                joint_w_active
                and _supported(row("ood", joint_comparison), better=False)
            )
        )
        formal_label: str | None
        if selected_ood_risk:
            formal_label = "W_ID_GAIN_OOD_RISK"
        elif pf_w_active and not pf_transfer:
            formal_label = "PF_W_DEVELOPMENT_ONLY_TRANSFER_UNSTABLE"
        elif not pf_w_active and joint_w_active and joint_transfer:
            formal_label = "W_INTERACTION_DEPENDENT_JOINT_ONLY"
        elif pf_w_active and not joint_w_active and pf_transfer:
            formal_label = "PF_STATIC_CURVATURE_ONLY"
        elif (pf_w_active or joint_w_active) and (
            (not pf_w_active or pf_transfer)
            and (not joint_w_active or joint_transfer)
        ):
            formal_label = "W_TRIGGER_TRANSFER_SUPPORTED"
        elif not pf_w_active and not joint_w_active:
            formal_label = "NORMAL_NO_W_DEGRADATION"
        else:
            formal_label = None
        unselected_test_signal = bool(
            (
                not pf_w_active
                and float(row("test", pf_comparison)["relative_mse_improvement"]) > 0.0
            )
            or (
                not joint_w_active
                and float(
                    row("test", joint_comparison)["relative_mse_improvement"]
                )
                > 0.0
            )
        )
        audited.append(
            {
                "view": view.relative_root.as_posix(),
                "proxy_policy": view.proxy_policy,
                "pf_w_active_on_development": pf_w_active,
                "pf_a_active_on_development": a_active,
                "pf_selected_route": a_result["pf_selected_route"],
                "joint_selected_route": selected_joint,
                "joint_w_active_on_development": joint_w_active,
                "pf_formal_w_comparison": pf_comparison,
                "joint_formal_w_comparison": joint_comparison,
                "pf_w_test_and_ood_transfer_supported": pf_transfer,
                "joint_w_test_and_ood_transfer_supported": joint_transfer,
                "selected_w_ood_risk_supported": selected_ood_risk,
                "formal_acceptance_label": formal_label,
                "acceptance_matrix_status": (
                    "MAPPED" if formal_label is not None else "UNMAPPED_MATRIX_STATE"
                ),
                "post_freeze_ablation_signal_only": unselected_test_signal,
                "secondary_label": (
                    "POST_FREEZE_ABLATION_SIGNAL_ONLY"
                    if unselected_test_signal
                    else None
                ),
                "test_did_not_change_formal_selection": True,
            }
        )
    labels = sorted(
        {str(item["formal_acceptance_label"]) for item in audited if item["formal_acceptance_label"]}
    )
    return {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "evidence_class": EVIDENCE_CLASS,
        "historical_aggregates_used_for_selection": False,
        "test_or_ood_used_to_rename_formal_model": False,
        "views": audited,
        "formal_acceptance_labels": labels,
    }


def _resource_audit(
    paths: MetroV211Paths,
    views: list[ViewSpec],
    config: dict[str, Any],
) -> dict[str, Any]:
    development = []
    failures = []
    for path in sorted((paths.output / "DEVELOPMENT").rglob("RESULT.json")):
        value = _read(path)
        record = {
            "path": str(path.relative_to(paths.output)),
            "stage": value.get("stage"),
            "status": value.get("status"),
            "elapsed_seconds": value.get("elapsed_seconds"),
            "row_cap_audit": value.get("row_cap_audit"),
        }
        development.append(record)
        if value.get("status") != "PASS":
            failures.append(record)
    final = []
    for view in views:
        contract = _read(_contract_path(paths, view))
        final.append(
            {
                "view": contract["view"],
                "fit_row_cap": contract["fit_row_cap"],
                "fit_rows": contract["fit_rows"],
                "prediction_chunk_rows": contract["prediction_chunk_rows"],
                "wall_seconds": contract["wall_seconds"],
                "peak_rss_kib": contract["peak_rss_kib"],
            }
        )
    return {
        "status": "PASS",
        "runtime_manager": "uv",
        "runtime_parallelism": runtime_parallelism_audit(config),
        "blas_threads_per_worker": 1,
        "dtype": "float64",
        "development": development,
        "final_materialization": final,
        "report_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "solver_failure_inventory": failures,
    }


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(
    paths: MetroV211Paths,
    metrics: pd.DataFrame,
    transfer: Mapping[str, Any],
) -> Path:
    selected = metrics[metrics["candidate"].isin(["PF_SELECTED", "J_SELECTED"])]
    metric_rows = [
        [
            row.proxy_policy,
            row.split,
            row.candidate,
            f"{float(row.mse):.12g}",
            f"{float(row.r2):.8g}",
        ]
        for row in selected.itertuples(index=False)
    ]
    decision_rows = [
        [
            item["proxy_policy"],
            item["pf_selected_route"],
            item["joint_selected_route"],
            item["formal_acceptance_label"],
            item["secondary_label"],
        ]
        for item in transfer["views"]
    ]
    report = paths.output / "FINAL" / "METRO_P60_V211_REPORT.md"
    report.write_text(
        "\n".join(
            [
                "# PRISM v2.1.1 Metro-P60 Wiener degradation/activation transfer audit",
                "",
                "Status: `COMPLETED`.",
                "",
                f"Evidence class: `{EVIDENCE_CLASS}`.",
                "",
                "The historical Metro-P60 test/OOD aggregates were known before this run. ",
                "They are shown only as retrospective context and were not used for development selection.",
                "The formal PF and Joint names remain the M6 development selections; post-freeze",
                "ablations were not promoted after test/OOD access.",
                "",
                "## Frozen selections and acceptance-matrix labels",
                "",
                _markdown_table(
                    ["proxy", "PF selected", "Joint selected", "formal label", "secondary label"],
                    decision_rows,
                ),
                "",
                "## Formal selected-model metrics",
                "",
                _markdown_table(["proxy", "split", "candidate", "MSE", "R2"], metric_rows),
                "",
                "Paired comparisons use 500 shared-draw moving-block bootstrap replicates,",
                "the inherited dynamic block-length rule, and Holm correction across all six",
                "registered M8 comparisons within each proxy view and split.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def _write_output_manifest(paths: MetroV211Paths) -> tuple[Path, Path]:
    manifest_path = paths.output / "MANIFEST.json"
    sums_path = paths.output / "SHA256SUMS.txt"
    excluded = {manifest_path.resolve(), sums_path.resolve()}
    records = []
    for path in sorted(item for item in paths.output.rglob("*") if item.is_file()):
        if path.resolve() in excluded:
            continue
        records.append(
            {
                "path": path.relative_to(paths.output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(manifest_path, {"status": "PASS", "files": records})
    sums_path.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in records),
        encoding="utf-8",
    )
    return manifest_path, sums_path


def _bundle_sources(paths: MetroV211Paths) -> list[tuple[Path, str]]:
    sources: list[tuple[Path, str]] = []

    def add_tree(root: Path, prefix: str) -> None:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            sources.append((path, f"{prefix}/{path.relative_to(root).as_posix()}"))

    add_tree(paths.output, "results")
    add_tree(paths.plan, "theory_and_plan/metro_audit")
    add_tree(paths.project / "src" / "prism_benchmark", "code_snapshot/src/prism_benchmark")
    add_tree(paths.project / "scripts", "code_snapshot/scripts")
    add_tree(paths.project / "tests", "code_snapshot/tests")
    sources.append((paths.project / ".gitignore", "code_snapshot/.gitignore"))
    return sources


def package_results(paths: MetroV211Paths) -> dict[str, Any]:
    _write_output_manifest(paths)
    zip_path = paths.project / f"{PACKAGE_NAME}.zip"
    sidecar = paths.project / f"{PACKAGE_NAME}.zip.sha256"
    temporary_zip = paths.project / f"{PACKAGE_NAME}.zip.tmp"
    for path in (zip_path, sidecar, temporary_zip):
        if path.exists():
            path.unlink()
    sources = _bundle_sources(paths)
    records = [
        {
            "path": archive_path,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        for source, archive_path in sources
    ]
    with tempfile.TemporaryDirectory(prefix="metro_v211_bundle_", dir=paths.project) as temp:
        temporary = Path(temp)
        bundle_manifest = temporary / "MANIFEST.json"
        bundle_sums = temporary / "SHA256SUMS.txt"
        write_json(
            bundle_manifest,
            {
                "status": "PASS",
                "protocol_id": PROTOCOL_ID,
                "evidence_class": EVIDENCE_CLASS,
                "files": records,
            },
        )
        bundle_sums.write_text(
            "".join(f"{item['sha256']}  {item['path']}\n" for item in records),
            encoding="utf-8",
        )
        root = PACKAGE_NAME
        with zipfile.ZipFile(temporary_zip, "w", allowZip64=True) as archive:
            for source, archive_path in sources:
                compression = (
                    zipfile.ZIP_STORED
                    if source.suffix.lower() in {".parquet", ".zip", ".gz", ".npy", ".npz"}
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(
                    source,
                    f"{root}/{archive_path}",
                    compress_type=compression,
                    compresslevel=None if compression == zipfile.ZIP_STORED else 6,
                )
            archive.write(bundle_manifest, f"{root}/MANIFEST.json", compress_type=zipfile.ZIP_DEFLATED)
            archive.write(bundle_sums, f"{root}/SHA256SUMS.txt", compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(temporary_zip) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP_INTEGRITY_FAILED: {bad}")
        names = set(archive.namelist())
        required = (
            "results/FINAL/METRO_P60_V211_FINAL_METRICS.csv",
            "results/FINAL/METRO_P60_V211_W_MARGINALS.csv",
            "results/FINAL/METRO_P60_V211_BOOTSTRAP.csv",
            "results/FINAL/METRO_P60_V211_SELECTION_TRANSFER_AUDIT.json",
            "results/FINAL/METRO_P60_V211_REPORT.md",
            "MANIFEST.json",
            "SHA256SUMS.txt",
        )
        for suffix in required:
            if f"{PACKAGE_NAME}/{suffix}" not in names:
                raise RuntimeError(f"MISSING_REQUIRED_ARTIFACT: {suffix}")
    temporary_zip.replace(zip_path)
    digest = sha256_file(zip_path)
    sidecar.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return {
        "status": "ZIP_INTEGRITY_PASS",
        "zip_path": str(zip_path),
        "zip_sha256": digest,
        "sidecar_path": str(sidecar),
        "zip_bytes": zip_path.stat().st_size,
    }


def run_m8(paths: MetroV211Paths) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_metro_config(paths.project)
    access = _read(paths.test_access_audit_path)
    if access.get("status") != "PASS":
        raise RuntimeError("M8 requires a completed M7 test/OOD access audit")
    freeze = _read(paths.development_freeze_path)
    if freeze.get("code_commit") != git_value(paths.project, "rev-parse", "HEAD"):
        raise RuntimeError("code changed after development freeze")
    if git_value(paths.project, "status", "--porcelain=v1"):
        raise RuntimeError("worktree is dirty after development freeze")
    registered_primary = set(config["statistics"]["primary_internal_comparisons"])
    if registered_primary != PRIMARY_W_COMPARISONS:
        raise RuntimeError("primary W comparison registry changed")
    replicates = int(config["statistics"]["paired_moving_block_bootstrap_replicates"])
    if replicates != 500:
        raise RuntimeError("Metro-P60 bootstrap replicate count must remain 500")
    views = metro_p60_dynamic_views(paths.shared)
    bootstrap_parts = run_parallel(
        _bootstrap_view,
        [(paths, view, replicates) for view in views],
        effective_worker_count(config),
        per_worker_gib=float(os.environ.get("PRISM_V211_MEMORY_GIB_PER_WORKER", "20")),
        label="PRISM_V211_METRO_M8_BOOTSTRAP",
    )
    bootstrap_rows = [row for part in bootstrap_parts for row in part]
    _holm(bootstrap_rows)
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values(
        ["proxy_policy", "split", "comparison_id"]
    )
    metrics = pd.DataFrame(_metric_rows(paths, views)).sort_values(
        ["proxy_policy", "split", "candidate"]
    )
    transfer = _selection_transfer_audit(paths, views, bootstrap)
    final = paths.output / "FINAL"
    final.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(final / "METRO_P60_V211_FINAL_METRICS.csv", index=False)
    bootstrap.to_csv(final / "METRO_P60_V211_BOOTSTRAP.csv", index=False)
    bootstrap[bootstrap["comparison_id"].isin(PRIMARY_W_COMPARISONS)].to_csv(
        final / "METRO_P60_V211_W_MARGINALS.csv", index=False
    )
    write_json(final / "METRO_P60_V211_SELECTION_TRANSFER_AUDIT.json", transfer)
    historical = _read(paths.historical_reference_path)
    if historical.get("selection_use_forbidden") is not True:
        raise RuntimeError("historical aggregate selection-use guard is missing")
    pd.DataFrame(historical["records"]).to_csv(
        final / "METRO_P60_V211_HISTORICAL_CONTEXT_ONLY.csv", index=False
    )
    resource_audit = _resource_audit(paths, views, config)
    write_json(final / "METRO_P60_V211_RESOURCE_AUDIT.json", resource_audit)
    write_json(
        final / "METRO_P60_V211_SOLVER_FAILURE_INVENTORY.json",
        {
            "status": "PASS" if not resource_audit["solver_failure_inventory"] else "FAILURES_RETAINED",
            "records": resource_audit["solver_failure_inventory"],
        },
    )
    report = _write_report(paths, metrics, transfer)
    status_path = paths.output / "RUN_STATUS.json"
    run_status = _read(status_path)
    run_status.update(
        {
            "status": "COMPLETED",
            "stage": "M8",
            "development_frozen": True,
            "test_accessed": True,
            "ood_accessed": True,
            "evidence_class": EVIDENCE_CLASS,
        }
    )
    write_json(status_path, run_status)
    result = {
        "status": "COMPLETED",
        "stage": "M8_STATISTICS_REPORT_PACKAGE",
        "protocol_id": PROTOCOL_ID,
        "evidence_class": EVIDENCE_CLASS,
        "metrics_rows": len(metrics),
        "bootstrap_rows": len(bootstrap),
        "formal_acceptance_labels": transfer["formal_acceptance_labels"],
        "report": str(report.relative_to(paths.output)),
        "wall_seconds_before_packaging": time.perf_counter() - started,
        "test_accessed": True,
        "ood_accessed": True,
    }
    write_json(final / "M8_RESULT.json", result)
    package = package_results(paths)
    result["package"] = package
    result["wall_seconds"] = time.perf_counter() - started
    write_json(final / "M8_RESULT_WITH_PACKAGE.json", result)
    return result
