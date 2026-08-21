"""Public-five CPU/PRISM Level-R2 reporting from frozen predictions only."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cpu_data import sha256_file

if TYPE_CHECKING:
    from .six_dataset_reporting import PredictionSpec


PUBLIC5_DATASETS = 5
PUBLIC5_PRIMARY_HEADS = 7
PUBLIC5_SUPPORT_CONTRACT = "NATIVE_K_COMMON_ASSEMBLY_R1"
EXPECTED_DATASETS = {"debutanizer", "metropt", "pmsm", "sru", "tep"}
EXPECTED_HEADS = {
    "DEB_C4__H5__W1",
    "METRO_OIL20__H120__W12",
    "METRO_P60__H6__W1",
    "PMSM_PM5__H600__W60",
    "SRU_H2S__H5__W1",
    "SRU_SO2__H5__W1",
    "TEP_G12__H4__W2",
}
PACKAGE_ROOT_NAME = "PRISM_V211_PUBLIC5_CPU_LEVEL_R2_REPORT"
PACKAGE_NAME = f"{PACKAGE_ROOT_NAME}_bundle.zip"
REPORT_NAME = "PRISM_V211_PUBLIC5_CPU_LEVEL_R2_REINTERPRETATION_REPORT.md"

KEY_COLUMNS = [
    "target_head",
    "information_set",
    "availability_scenario",
    "proxy_policy",
    "split",
    "model",
]
GROUP_COLUMNS = [
    "target_head",
    "information_set",
    "availability_scenario",
    "proxy_policy",
    "split",
]
PRISM_INPUT_MODELS = ("PRISM_V2_1_1_K_C_W", "PRISM_V2_1_1_K_C")
PRISM_DYNAMIC_MODELS = (
    "PRISM_V2_1_1_PHYSICS_FIRST",
    "PRISM_V2_1_1_JOINT_KWA",
)
TRIVIAL_BASELINES = {"MEAN", "PERSISTENCE", "SEASONAL_PERSISTENCE"}
FORBIDDEN_PACKAGE_SUFFIXES = {
    ".xlsx",
    ".xls",
    ".parquet",
    ".pt",
    ".pth",
    ".ckpt",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def verify_public5_source(public_root: Path) -> dict[str, Any]:
    """Verify that the frozen public-five benchmark is safe to reinterpret."""

    final = public_root / "final"
    evidence_path = final / "PUBLIC_ALL_FINAL_EVIDENCE_SUMMARY.json"
    metrics_path = final / "PUBLIC_ALL_METRICS.csv"
    access_path = final / "PUBLIC_ALL_TEST_OOD_ACCESS_AUDIT.json"
    repro_path = final / "FULL_REPRO_MANIFEST.json"
    for path in (evidence_path, metrics_path, access_path, repro_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    evidence = _read_json(evidence_path)
    failures: list[str] = []
    if evidence.get("status") not in {"PASS", "EXPERIMENT_COMPLETED"}:
        failures.append(f"status={evidence.get('status')!r}")
    if evidence.get("support_contract") != PUBLIC5_SUPPORT_CONTRACT:
        failures.append(f"support_contract={evidence.get('support_contract')!r}")
    if int(evidence.get("datasets", -1)) != PUBLIC5_DATASETS:
        failures.append(f"datasets={evidence.get('datasets')!r}")
    if int(evidence.get("primary_heads", -1)) != PUBLIC5_PRIMARY_HEADS:
        failures.append(f"primary_heads={evidence.get('primary_heads')!r}")
    if evidence.get("post_test_reselection") is not False:
        failures.append(
            f"post_test_reselection={evidence.get('post_test_reselection')!r}"
        )
    access_sha256 = sha256_file(access_path)
    if evidence.get("test_access_audit_sha") != access_sha256:
        failures.append("test access audit SHA256 does not match final evidence")
    package = evidence.get("package")
    if not isinstance(package, dict):
        failures.append("final evidence is missing package provenance")
        package = {}
    bundle_path = Path(str(package.get("path", "")))
    if not bundle_path.is_file():
        bundle_path = public_root / "return" / Path(str(package.get("path", ""))).name
    if not bundle_path.is_file():
        failures.append("original public-five result bundle is missing")
    elif sha256_file(bundle_path) != package.get("sha256"):
        failures.append("original public-five result bundle SHA256 mismatch")
    else:
        repro_bytes = repro_path.read_bytes()
        with zipfile.ZipFile(bundle_path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.endswith("/results/FULL_REPRO_MANIFEST.json")
            ]
            if len(candidates) != 1 or archive.read(candidates[0]) != repro_bytes:
                failures.append(
                    "FULL_REPRO_MANIFEST does not match the original result bundle"
                )
    if failures:
        raise RuntimeError("STOP_PUBLIC5_NOT_COMPLETED: " + "; ".join(failures))
    return {
        "status": "PASS",
        "source_status": evidence.get("status"),
        "support_contract": evidence["support_contract"],
        "datasets": int(evidence["datasets"]),
        "primary_heads": int(evidence["primary_heads"]),
        "post_test_reselection": False,
        "source_branch": evidence.get("source_branch")
        or evidence.get("execution_branch"),
        "source_commit": evidence.get("source_commit"),
        "generating_commit": evidence.get("generating_commit"),
        "reporting_commit": evidence.get("reporting_commit"),
        "development_freeze_sha": evidence.get("development_freeze_sha"),
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "access_audit_path": str(access_path),
        "access_audit_sha256": access_sha256,
        "full_repro_manifest_path": str(repro_path),
        "full_repro_manifest_sha256": sha256_file(repro_path),
        "original_bundle_path": str(bundle_path),
        "original_bundle_sha256": package.get("sha256"),
    }


def load_source_metrics(public_root: Path) -> pd.DataFrame:
    path = public_root / "final" / "PUBLIC_ALL_METRICS.csv"
    frame = pd.read_csv(path)
    missing = set(
        [
            "status",
            "dataset",
            "task_id",
            "target_head",
            "information_set",
            "availability_scenario",
            "proxy_policy",
            "view_role",
            "model",
            "model_source",
            "split",
            "prediction_path",
            "prediction_sha256",
            "rows",
            "mse",
            "rmse",
            "mae",
            "r2",
            "persistence_skill",
            "rank",
        ]
    ) - set(frame.columns)
    if missing:
        raise KeyError(f"PUBLIC_ALL_METRICS.csv is missing columns: {sorted(missing)}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise RuntimeError("duplicate source metric identity")
    if set(frame["dataset"].dropna().astype(str).unique()) != EXPECTED_DATASETS:
        raise RuntimeError("public-five metrics do not cover the exact five datasets")
    if set(frame["target_head"].dropna().astype(str).unique()) != EXPECTED_HEADS:
        raise RuntimeError("public-five metrics do not cover the exact seven heads")
    if set(frame["model_source"].dropna().unique()) - {"PRISM", "CPU_BASELINE"}:
        raise RuntimeError("unexpected non-CPU model source in public-five metrics")
    return frame


def prediction_specs_from_source(
    public_root: Path,
    source: pd.DataFrame,
) -> list["PredictionSpec"]:
    from .six_dataset_reporting import PredictionSpec

    specs: list[PredictionSpec] = []
    passed = source.loc[source["status"].eq("PASS")].copy()
    if passed["prediction_path"].isna().any():
        raise RuntimeError("PASS row is missing its frozen prediction path")
    for row in passed.itertuples(index=False):
        path = Path(str(row.prediction_path))
        if not path.is_absolute():
            path = public_root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        specs.append(
            PredictionSpec(
                path=path,
                scope="public5",
                direction=None,
                split=str(row.split),
                model=str(row.model),
                target_head=str(row.target_head),
                information_set=str(row.information_set),
                availability_scenario=str(row.availability_scenario),
                proxy_policy=str(row.proxy_policy),
            )
        )
    return specs


def verify_frozen_reporting_inputs(
    public_root: Path,
    source: pd.DataFrame,
    specs: Sequence["PredictionSpec"],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Bind every prediction and shared level source read to the R5 repro manifest."""

    manifest_path = public_root / "final" / "FULL_REPRO_MANIFEST.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "PASS" or not isinstance(manifest.get("files"), list):
        raise RuntimeError("STOP_INVALID_FULL_REPRO_MANIFEST")
    index: dict[str, dict[str, Any]] = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        raw_path = str(entry["path"])
        if Path(raw_path).is_absolute():
            continue
        previous = index.get(raw_path)
        if previous is not None and previous.get("sha256") != entry.get("sha256"):
            raise RuntimeError(f"conflicting repro manifest entries: {raw_path}")
        index[raw_path] = entry

    dataset_by_key = {
        tuple(row[column] for column in KEY_COLUMNS): str(row["dataset"])
        for _, row in source.loc[source["status"].eq("PASS")].iterrows()
    }
    expected: set[str] = {"shared/TASK_REGISTRY.json"}
    prediction_relatives: dict[str, str] = {}
    # Level reconstruction uses the complete frozen entity to reproduce the
    # FP64 prefix-sum path that materialized target_change().  This is a
    # reporting read only; it does not expose rows to model fitting/inference.
    frozen_entity_partitions = ("train", "validation", "test", "ood")
    partitions = {
        "test": frozen_entity_partitions,
        "ood": frozen_entity_partitions,
    }
    for spec in specs:
        key = (
            spec.target_head,
            spec.information_set,
            spec.availability_scenario,
            spec.proxy_policy,
            spec.split,
            spec.model,
        )
        dataset = dataset_by_key.get(key)
        if dataset is None:
            raise RuntimeError(f"prediction spec is absent from source metrics: {key}")
        sample_relative = (
            Path("shared")
            / "sample_ids"
            / spec.target_head
            / spec.information_set
            / spec.availability_scenario
            / spec.proxy_policy
            / f"{spec.split}.parquet"
        ).as_posix()
        expected.add(sample_relative)
        for partition in partitions[spec.split]:
            relative = (
                Path("shared") / "base_data" / dataset / f"{partition}.parquet"
            ).as_posix()
            if (public_root / relative).is_file():
                expected.add(relative)
        try:
            prediction_relative = spec.path.relative_to(public_root).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"prediction path escapes public root: {spec.path}"
            ) from error
        expected.add(prediction_relative)
        prediction_relatives[str(spec.path.resolve())] = prediction_relative

    verified_files: list[dict[str, Any]] = []
    actual_hashes: dict[str, str] = {}
    for relative in sorted(expected):
        entry = index.get(relative)
        if entry is None:
            raise RuntimeError(f"repro manifest is missing reporting input: {relative}")
        path = public_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        if int(entry.get("bytes", -1)) != actual_size:
            raise RuntimeError(f"repro manifest byte count mismatch: {relative}")
        actual_hash = sha256_file(path)
        if entry.get("sha256") != actual_hash:
            raise RuntimeError(f"repro manifest SHA256 mismatch: {relative}")
        actual_hashes[relative] = actual_hash
        verified_files.append(
            {
                "path": relative,
                "bytes": actual_size,
                "sha256": actual_hash,
                "role": entry.get("role"),
            }
        )
    prediction_hashes = {
        absolute: actual_hashes[relative]
        for absolute, relative in prediction_relatives.items()
    }
    digest = hashlib.sha256()
    for item in verified_files:
        digest.update(str(item["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return (
        {
            "status": "PASS",
            "full_repro_manifest": str(manifest_path),
            "full_repro_manifest_sha256": sha256_file(manifest_path),
            "verified_file_count": len(verified_files),
            "verified_prediction_count": len(prediction_hashes),
            "verified_input_set_sha256": digest.hexdigest(),
            "files": verified_files,
        },
        prediction_hashes,
    )


def add_dual_rankings(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank Level R2 and persistence skill without changing frozen selection."""

    result = frame.copy()
    passed = result["reconstruction_status"].eq("PASS")
    result["rank_by_level_r2"] = np.nan
    result["rank_by_persistence_skill"] = np.nan
    result.loc[passed, "rank_by_level_r2"] = (
        result.loc[passed]
        .groupby(GROUP_COLUMNS, dropna=False)["r2_level_reconstructed"]
        .rank(method="min", ascending=False)
    )
    numeric_skill = pd.to_numeric(result["persistence_skill"], errors="coerce")
    skill_rows = passed & numeric_skill.notna()
    ranked = result.loc[skill_rows, GROUP_COLUMNS].copy()
    ranked["_skill"] = numeric_skill.loc[skill_rows]
    result.loc[skill_rows, "rank_by_persistence_skill"] = (
        ranked.groupby(GROUP_COLUMNS, dropna=False)["_skill"]
        .rank(method="min", ascending=False)
        .to_numpy()
    )
    return result


def merge_reconstructed_metrics(
    source: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve every formal source status while attaching reconstructed metrics."""

    source_pass_keys = {
        tuple(row)
        for row in source.loc[source["status"].eq("PASS"), KEY_COLUMNS].itertuples(
            index=False, name=None
        )
    }
    reconstructed_keys = {
        tuple(row)
        for row in reconstructed.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
    }
    if source_pass_keys != reconstructed_keys:
        raise RuntimeError("source and reconstructed prediction key sets differ")
    renamed = source.rename(
        columns={
            "status": "source_status",
            "dataset": "source_dataset",
            "rows": "source_rows",
            "prediction_path": "source_prediction_path",
            "prediction_sha256": "source_prediction_sha256",
            "mse": "source_delta_mse",
            "rmse": "source_delta_rmse",
            "mae": "source_delta_mae",
            "r2": "source_delta_r2",
            "persistence_skill": "source_persistence_skill",
            "rank": "frozen_delta_rank",
        }
    )
    merged = renamed.merge(
        reconstructed,
        on=KEY_COLUMNS,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    passed = merged["source_status"].eq("PASS")
    if merged.loc[passed, "r2_level_reconstructed"].isna().any():
        raise RuntimeError("PASS source row is missing Level-R2 reconstruction")
    if merged.loc[~passed, "r2_level_reconstructed"].notna().any():
        raise RuntimeError("non-run source row unexpectedly has reconstructed metrics")
    if not np.array_equal(
        merged.loc[passed, "source_dataset"].astype(str).to_numpy(),
        merged.loc[passed, "dataset"].astype(str).to_numpy(),
    ):
        raise RuntimeError("source and task-registry dataset identities differ")
    if not np.array_equal(
        merged.loc[passed, "source_rows"].astype(int).to_numpy(),
        merged.loc[passed, "rows"].astype(int).to_numpy(),
    ):
        raise RuntimeError("source and reconstructed row counts differ")
    source_hashes = merged.loc[passed, "source_prediction_sha256"].astype(str)
    rebuilt_hashes = merged.loc[passed, "prediction_sha256"].astype(str)
    if not np.array_equal(source_hashes.to_numpy(), rebuilt_hashes.to_numpy()):
        raise RuntimeError("frozen prediction SHA256 mismatch")
    for source_column, rebuilt_column, label in (
        ("source_delta_mse", "mse", "MSE"),
        ("source_delta_rmse", "rmse", "RMSE"),
        ("source_delta_mae", "mae", "MAE"),
        ("source_delta_r2", "r2_delta", "R2_DELTA"),
    ):
        if not np.allclose(
            merged.loc[passed, source_column].to_numpy(dtype=np.float64),
            merged.loc[passed, rebuilt_column].to_numpy(dtype=np.float64),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        ):
            raise RuntimeError(f"frozen source {label} does not reproduce")
    source_skill = pd.to_numeric(
        merged.loc[passed, "source_persistence_skill"], errors="coerce"
    )
    rebuilt_skill = pd.to_numeric(
        merged.loc[passed, "persistence_skill"], errors="coerce"
    )
    comparable_skill = source_skill.notna() & rebuilt_skill.notna()
    if not np.allclose(
        source_skill.loc[comparable_skill].to_numpy(dtype=np.float64),
        rebuilt_skill.loc[comparable_skill].to_numpy(dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError("frozen source persistence skill does not reproduce")
    merged["status"] = merged["source_status"]
    merged["reconstruction_status"] = np.where(
        passed,
        "PASS",
        merged["source_status"],
    )
    merged["dataset"] = merged["source_dataset"]
    merged["task"] = merged["task_id"]
    merged["direction"] = ""
    merged["view"] = (
        merged["information_set"].astype(str)
        + "/"
        + merged["availability_scenario"].astype(str)
        + "/"
        + merged["proxy_policy"].astype(str)
    )
    merged["historical_results_overwritten"] = False
    merged["hyperparameters_changed"] = False
    merged["test_rerun"] = False
    merged["ood_rerun"] = False
    merged = add_dual_rankings(merged)
    preferred = [
        "status",
        "reconstruction_status",
        "dataset",
        "task",
        "target_head",
        "direction",
        "view",
        "information_set",
        "availability_scenario",
        "proxy_policy",
        "view_role",
        "model",
        "model_source",
        "split",
        "rows",
        "support_hash",
        "prediction_target_semantics",
        "level_target_semantics",
        "current_window_steps",
        "future_window_steps",
        "r2_level_reconstructed",
        "r2_delta",
        "mse",
        "rmse",
        "mae",
        "r2_level_persistence",
        "persistence_skill",
        "std_level_target",
        "std_delta_target",
        "variance_ratio",
        "rank_by_level_r2",
        "rank_by_persistence_skill",
        "frozen_delta_rank",
        "same_prediction_error",
        "different_target_variance",
        "target_identity_max_abs_error",
        "residual_identity_max_abs_error",
        "mse_identity_max_abs_error",
        "rmse_identity_max_abs_error",
        "mae_identity_max_abs_error",
        "model_retrained",
        "model_reselected",
        "hyperparameters_changed",
        "sample_support_changed",
        "test_rerun",
        "ood_rerun",
        "historical_results_overwritten",
        "prediction_sha256",
        "prediction_path",
        "source_status",
        "source_rows",
        "source_delta_r2",
        "source_delta_mse",
        "source_delta_rmse",
        "source_delta_mae",
        "source_persistence_skill",
        "source_prediction_sha256",
        "source_prediction_path",
    ]
    ordered = [column for column in preferred if column in merged.columns]
    remainder = [column for column in merged.columns if column not in ordered]
    return merged.loc[:, ordered + remainder]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if _finite(value) else "—"


def _display_model(model: str) -> str:
    names = {
        "PRISM_V2_1_1_K_C_W": "PRISM input K+C+W",
        "PRISM_V2_1_1_K_C": "PRISM input K+C",
        "PRISM_V2_1_1_PHYSICS_FIRST": "PRISM-PF",
        "PRISM_V2_1_1_JOINT_KWA": "PRISM-Joint",
        "PERSISTENCE": "Persistence",
    }
    return names.get(model, model.replace("_", " "))


def select_frozen_comparison_rows(
    frame: pd.DataFrame,
    target_head: str,
    information_set: str,
) -> pd.DataFrame:
    """Select plot rows using only pre-existing formal identities/ranks."""

    group = frame.loc[
        frame["target_head"].eq(target_head)
        & frame["information_set"].eq(information_set)
        & frame["split"].eq("test")
        & frame["view_role"].eq("primary")
        & frame["reconstruction_status"].eq("PASS")
    ].copy()
    if group.empty:
        return group
    selected: list[pd.DataFrame] = []
    model_order = (
        PRISM_INPUT_MODELS
        if information_set == "input_only"
        else PRISM_DYNAMIC_MODELS
    )
    for model in model_order:
        row = group.loc[group["model"].eq(model)]
        if not row.empty:
            selected.append(row.iloc[[0]])
    classical = group.loc[
        group["model_source"].eq("CPU_BASELINE")
        & ~group["model"].isin(TRIVIAL_BASELINES)
    ].sort_values(["frozen_delta_rank", "model"], na_position="last")
    if not classical.empty:
        selected.append(classical.iloc[[0]])
    persistence = group.loc[group["model"].eq("PERSISTENCE")]
    if not persistence.empty:
        selected.append(persistence.iloc[[0]])
    if not selected:
        return group.iloc[0:0]
    return pd.concat(selected, ignore_index=True).drop_duplicates("model")


def write_plots(frame: pd.DataFrame, plot_dir: Path) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    chart_map: list[dict[str, Any]] = []
    heads = sorted(frame["target_head"].dropna().unique())
    for head in heads:
        figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), squeeze=False)
        for axis, information_set in zip(
            axes[0], ("input_only", "dynamic"), strict=True
        ):
            selected = select_frozen_comparison_rows(frame, head, information_set)
            if selected.empty:
                axis.text(0.5, 0.5, "No formal primary test rows", ha="center")
                axis.set_axis_off()
                continue
            labels = [_display_model(str(value)) for value in selected["model"]]
            values = selected["r2_level_reconstructed"].to_numpy(dtype=float)
            colors = []
            for row in selected.itertuples(index=False):
                if row.model == "PERSISTENCE":
                    colors.append("#777777")
                elif row.model_source == "PRISM":
                    colors.append("#3B6FB6")
                else:
                    colors.append("#C99724")
            positions = np.arange(len(selected))
            bars = axis.barh(
                positions,
                values,
                color=colors,
                edgecolor="#2F343A",
                linewidth=0.7,
            )
            axis.set_yticks(positions, labels)
            axis.invert_yaxis()
            axis.axvline(0.0, color="#2F343A", linewidth=0.9)
            axis.grid(axis="x", color="#D9DDE3", linewidth=0.7)
            axis.set_axisbelow(True)
            axis.set_xlabel("Reconstructed Level R²")
            axis.set_title(information_set.replace("_", " ").title())
            left = min(0.0, float(np.min(values)))
            right = max(0.0, float(np.max(values)))
            span = max(right - left, max(abs(values).max(), 1.0) * 0.1)
            padding = span * 0.08
            axis.set_xlim(left - padding, right + padding)
            for bar, value in zip(bars, values, strict=True):
                offset = span * 0.025
                axis.text(
                    value + (offset if value >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=8.5,
                    color="#20242A",
                )
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
        figure.suptitle(f"{head} — frozen primary test model comparison")
        figure.text(
            0.5,
            0.01,
            "Best classical CPU baseline is identified by its original frozen Delta-target rank; no Level-R² reselection.",
            ha="center",
            fontsize=9,
            color="#4A4F57",
        )
        figure.tight_layout(rect=(0, 0.055, 1, 0.94))
        path = plot_dir / f"LEVEL_R2_MODEL_COMPARISON_{head}.svg"
        figure.savefig(path, format="svg", bbox_inches="tight")
        plt.close(figure)
        chart_map.append(
            {
                "section": f"Task evidence: {head}",
                "analytical_question": (
                    "How do frozen formal PRISM routes and the originally ranked "
                    "best compatible CPU baseline compare with persistence?"
                ),
                "takeaway": "Compare reconstructed Level R2 on the primary test support.",
                "family": "Comparison & Ranking",
                "chart_type": "horizontal bar, two information-set panels",
                "fields": ["model", "r2_level_reconstructed"],
                "selection_basis": "FROZEN_ORIGINAL_DELTA_RANK",
                "palette_policy": "hard two-root cap plus neutral persistence",
                "artifact": f"plots/{path.name}",
            }
        )

    scatter = frame.loc[
        frame["reconstruction_status"].eq("PASS")
        & frame["split"].eq("test")
        & frame["view_role"].eq("primary")
    ].copy()
    scatter = scatter.loc[
        np.isfinite(scatter["r2_delta"].to_numpy(dtype=float))
        & np.isfinite(scatter["r2_level_reconstructed"].to_numpy(dtype=float))
    ]
    figure, axis = plt.subplots(figsize=(8.8, 7.2))
    palette = {"PRISM": "#3B6FB6", "CPU_BASELINE": "#C99724"}
    markers = {"PRISM": "o", "CPU_BASELINE": "s"}
    for source, group in scatter.groupby("model_source", sort=True):
        axis.scatter(
            group["r2_delta"],
            group["r2_level_reconstructed"],
            label=source.replace("_", " ").title(),
            color=palette.get(str(source), "#777777"),
            marker=markers.get(str(source), "o"),
            alpha=0.72,
            s=34,
            edgecolor="#2F343A",
            linewidth=0.35,
        )
    combined = np.concatenate(
        [
            scatter["r2_delta"].to_numpy(dtype=float),
            scatter["r2_level_reconstructed"].to_numpy(dtype=float),
        ]
    )
    low = float(np.min(combined))
    high = float(np.max(combined))
    use_symlog = low < -20.0 or high > 20.0
    if use_symlog:
        axis.set_xscale("symlog", linthresh=1.0)
        axis.set_yscale("symlog", linthresh=1.0)
    axis.plot([low, high], [low, high], color="#33383F", linestyle="--", linewidth=1.0)
    axis.axhline(0.0, color="#9AA0A8", linewidth=0.7)
    axis.axvline(0.0, color="#9AA0A8", linewidth=0.7)
    axis.grid(color="#D9DDE3", linewidth=0.65)
    axis.set_axisbelow(True)
    axis.set_xlabel("Delta R²")
    axis.set_ylabel("Reconstructed Level R²")
    axis.set_title("Delta R² and reconstructed Level R² — primary test rows")
    subtitle = "Dashed line: equal R² in both representations"
    if use_symlog:
        subtitle += "; symmetric-log axes preserve extreme negative values"
    axis.text(0.5, 1.01, subtitle, transform=axis.transAxes, ha="center", fontsize=9)
    axis.legend(frameon=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    scatter_path = plot_dir / "R2_DELTA_VS_R2_LEVEL.svg"
    figure.savefig(scatter_path, format="svg", bbox_inches="tight")
    plt.close(figure)
    chart_map.append(
        {
            "section": "Representation effect",
            "analytical_question": (
                "How does the same frozen residual score under Delta and Level target variance?"
            ),
            "takeaway": "R2 changes because the denominator changes; the residual is identical.",
            "family": "Relationship",
            "chart_type": "scatter with equality reference",
            "fields": ["r2_delta", "r2_level_reconstructed", "model_source"],
            "rows": int(len(scatter)),
            "scale": "symlog" if use_symlog else "linear",
            "palette_policy": "hard two-root cap",
            "artifact": f"plots/{scatter_path.name}",
        }
    )
    return chart_map


def write_report(frame: pd.DataFrame, audit: Mapping[str, Any], path: Path) -> None:
    passed = frame.loc[frame["reconstruction_status"].eq("PASS")].copy()
    legal_nonruns = frame.loc[~frame["reconstruction_status"].eq("PASS")]
    appearance = passed.assign(
        _appearance_shift=(
            passed["r2_level_reconstructed"] - passed["r2_delta"]
        )
    ).sort_values("_appearance_shift", ascending=False)
    largest = appearance.iloc[0]
    skill = pd.to_numeric(passed["persistence_skill"], errors="coerce")
    positive_skill = int((skill > 0.0).sum())
    negative_skill = int((skill < 0.0).sum())
    lines = [
        "# Derived Level-R² Reinterpretation from Frozen Change-Target Predictions — Public5 CPU Methods",
        "",
        "## Technical summary",
        "",
        (
            f"- **Reconstruction passed for {len(passed)} frozen prediction artifacts** "
            f"covering {passed['dataset'].nunique()} public datasets and "
            f"{passed['target_head'].nunique()} registered heads; no model was refit, "
            "rerun for inference, or reselected. The reporting pass only reread frozen "
            "predictions and manifest-verified shared target windows."
        ),
        (
            f"- **All registered-window and residual identities passed.** The maximum "
            f"target identity error was {_fmt(audit['target_identity_max_abs_error'], 12)} "
            f"and the maximum residual identity error was "
            f"{_fmt(audit['residual_identity_max_abs_error'], 12)}."
        ),
        (
            f"- **All {len(legal_nonruns)} legal non-run states remain visible** "
            "(`NOT_RUN_PROTOCOL_INCOMPATIBLE` or `NOT_APPLICABLE_JOINT_NOT_FROZEN`); "
            "they were not silently dropped from the result tables."
        ),
        (
            f"- **Persistence skill remains the operational guardrail.** Of the "
            f"{len(passed)} reconstructed rows, {positive_skill} beat persistence and "
            f"{negative_skill} were worse than persistence on the identical support."
        ),
        "",
        "## The R² change is a representation effect, not a smaller forecast error",
        "",
        (
            "The level and change representations use identical sample-wise prediction "
            "errors. Their MSE, RMSE, and MAE are therefore identical on the same support. "
            "The difference in R² arises solely from the variance of the quantity against "
            "which the same residual error is normalized."
        ),
        "",
        (
            "中文解释：Level 与 change 表示使用完全相同的逐样本预测残差，因此在同一支持集上 "
            "MSE、RMSE 和 MAE 完全相同。两种 R² 的差别只来自同一残差所除以的目标方差不同，"
            "不能解释为模型误差变小。"
        ),
        "",
        (
            f"本次外观增幅最大的单行结果是 `{largest['target_head']}` / "
            f"`{largest['information_set']}` / `{largest['split']}` / "
            f"`{largest['model']}`：Delta R²={_fmt(largest['r2_delta'])}，"
            f"Level R²={_fmt(largest['r2_level_reconstructed'])}，差值="
            f"{_fmt(largest['_appearance_shift'])}。这只表示归一化分母改变，不是新的实验增益。"
        ),
        "",
        "![Delta R² versus reconstructed Level R²](../plots/R2_DELTA_VS_R2_LEVEL.svg)",
        "",
        "散点图以相同冻结预测逐行比较两种 R²；虚线只是两种表示相等的参照，不是性能目标。",
        "",
        "## Frozen primary-test comparisons by task",
        "",
        (
            "主表分开 input-only 与 dynamic，并同时显示 Level R²、Persistence Level R²、"
            "Persistence Skill、Delta R² 与 RMSE。`best CPU` 按原冻结 Delta-target 排名识别，"
            "不按新的 Level R² 重选；缺失的 Joint 表示原实验未正式冻结，不是本次重构失败。"
        ),
    ]
    for information_set in ("input_only", "dynamic"):
        lines.extend(
            [
                "",
                f"### {information_set.replace('_', ' ').title()} primary test table",
                "",
                "| Dataset | Task | Model | Level R² | Persistence Level R² | Persistence Skill | Delta R² | RMSE |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for head in sorted(frame["target_head"].dropna().unique()):
            selected = select_frozen_comparison_rows(frame, head, information_set)
            for row in selected.itertuples(index=False):
                lines.append(
                    f"| {row.dataset} | {row.target_head} | {_display_model(str(row.model))} | "
                    f"{_fmt(row.r2_level_reconstructed)} | {_fmt(row.r2_level_persistence)} | "
                    f"{_fmt(row.persistence_skill)} | {_fmt(row.r2_delta)} | {_fmt(row.rmse)} |"
                )
    for head in sorted(frame["target_head"].dropna().unique()):
        lines.extend(
            [
                "",
                f"### {head}: frozen primary test routes",
                "",
                (
                    "该图分开 input-only 与 dynamic，展示正式 PRISM 路线、按原冻结排名识别的 "
                    "best classical CPU baseline 以及 persistence。条形位置只表示重构后的 Level R²；"
                    "精确 Delta R²、Persistence Skill 与支持哈希请查 CSV。"
                ),
                "",
                f"![{head} Level R² model comparison](../plots/LEVEL_R2_MODEL_COMPARISON_{head}.svg)",
            ]
        )
    lines.extend(
        [
            "",
            "## Scope, data, and metric definitions",
            "",
            "- Scope: TEP, Debutanizer, SRU, PMSM, and MetroPT-3; seven registered heads; PRISM plus existing CPU baselines only.",
            "- Evidence: frozen test and registered OOD sample-level prediction artifacts from the completed R5 public-five benchmark.",
            "- Level target semantics: `REGISTERED_FUTURE_WINDOW_LEVEL`; no assumption that `W=1` is made.",
            "- `current_level = mean(y[current_start:current_stop_exclusive])` and `future_level_true = mean(y[target_start:target_stop_exclusive])`.",
            "- `future_level_pred = current_level + delta_pred`; persistence is `delta_pred=0`, equivalently `future_level_pred=current_level`.",
            "- Input-only and dynamic leaderboards are separate. Test and OOD rows remain explicitly labeled and are never pooled into one support.",
            "",
            "## Reporting-only reconstruction method",
            "",
            "The reporting pass independently reads both registered current and future target windows from the shared frozen target data. It then verifies `future_level_true-current_level == delta_true` before calculating Level R². Prediction sample IDs are matched one-to-one to the registered view sample IDs; row counts, support hashes, and frozen prediction SHA256 values are retained per result row.",
            "",
            "Both rank columns are reported: `rank_by_level_r2` and `rank_by_persistence_skill`. They are descriptive reporting ranks and do not change the original model selection, route freeze, hyperparameters, split, target, horizon, or support.",
            "",
            "## Limitations, uncertainty, and robustness checks",
            "",
            "- Level R² can look much larger when the registered level target has substantially more variance than the change target; it must be read beside Persistence Level R² and Persistence Skill.",
            "- This public-five CPU report is an interim scoped deliverable. It is not the canonical six-dataset R2 report and contains no CZ or Neural-3 results.",
            "- Legal protocol incompatibilities and Joint-not-frozen states have no prediction artifact and therefore no reconstructed metric; their source statuses remain in every all-results table.",
            "- Test/OOD values are descriptive frozen outcomes. This reconstruction creates no new confirmatory evidence and cannot support causal or mechanistic claims.",
            "",
            "## Recommended next steps",
            "",
            "1. Keep the Neural-3 8-worker recovery unchanged until all frozen prediction artifacts finish.",
            "2. After Neural-3 and CZ are complete, append them through the same identity-audited reconstruction code.",
            "3. Only then generate the canonical `SIX_DATASET_*` Level-R² tables and six-dataset report; do not overwrite this public-five scoped evidence.",
            "",
            "## Further questions",
            "",
            "- Which conclusions remain stable when models are ordered by Persistence Skill instead of Level R²?",
            "- Which registered heads have a large level/change variance ratio but little or negative persistence skill?",
            "- After Neural-3 completes, do its comparative ranks change materially between the two R² representations on identical common support?",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_value(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_provenance(repository_root: Path) -> dict[str, Any]:
    status = _git_value(repository_root, "status", "--short")
    return {
        "branch": _git_value(repository_root, "branch", "--show-current"),
        "commit": _git_value(repository_root, "rev-parse", "HEAD"),
        "status_clean": status == "",
        "status_short": status,
    }


def reporting_preflight(
    repository_root: Path,
    pytest_log: Path | None,
) -> dict[str, Any]:
    provenance = git_provenance(repository_root)
    if not provenance["status_clean"]:
        raise RuntimeError("STOP_DIRTY_REPORTING_WORKTREE")
    if pytest_log is None or not pytest_log.is_file():
        raise RuntimeError("STOP_MISSING_PYTEST_LOG")
    pytest_text = pytest_log.read_text(encoding="utf-8", errors="replace")
    if " passed" not in pytest_text or " failed" in pytest_text:
        raise RuntimeError("STOP_PYTEST_NOT_PASSING")
    return {
        "status": "PASS",
        "git": provenance,
        "pytest_log": str(pytest_log),
        "pytest_log_sha256": sha256_file(pytest_log),
    }


def _write_checksum_file(root: Path) -> Path:
    checksum_path = root / "SHA256SUMS.txt"
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == checksum_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def verify_checksum_file(root: Path) -> None:
    checksum_path = root / "SHA256SUMS.txt"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"package checksum failed: {relative}")


def verify_zip_checksums(zip_path: Path) -> None:
    """Independently read every packaged member and verify internal SHA256SUMS."""

    checksum_member = f"{PACKAGE_ROOT_NAME}/SHA256SUMS.txt"
    with zipfile.ZipFile(zip_path) as archive:
        members = {name for name in archive.namelist() if not name.endswith("/")}
        if checksum_member not in members:
            raise RuntimeError("ZIP is missing its internal SHA256SUMS.txt")
        checksum_text = archive.read(checksum_member).decode("utf-8")
        listed: set[str] = set()
        for line in checksum_text.splitlines():
            digest, relative = line.split("  ", 1)
            member = f"{PACKAGE_ROOT_NAME}/{relative}"
            if member not in members:
                raise RuntimeError(f"ZIP checksum references missing member: {member}")
            actual = hashlib.sha256(archive.read(member)).hexdigest()
            if actual != digest:
                raise RuntimeError(f"ZIP internal checksum failed: {member}")
            listed.add(member)
        expected = members - {checksum_member}
        if listed != expected:
            missing = sorted(expected - listed)
            extra = sorted(listed - expected)
            raise RuntimeError(
                f"ZIP checksum coverage mismatch: missing={missing}, extra={extra}"
            )


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def package_outputs(
    output_root: Path,
    public_root: Path,
    repository_root: Path,
    artifact_paths: Sequence[Path],
    pytest_log: Path | None,
) -> dict[str, Any]:
    preflight = reporting_preflight(repository_root, pytest_log)
    provenance = preflight["git"]
    package_dir = output_root / "package" / PACKAGE_ROOT_NAME
    if package_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing package staging directory: {package_dir}"
        )
    package_dir.mkdir(parents=True)
    for path in artifact_paths:
        relative = path.relative_to(output_root)
        _copy_file(path, package_dir / relative)
    evidence_files = [
        public_root / "final" / "PUBLIC_ALL_FINAL_EVIDENCE_SUMMARY.json",
        public_root / "final" / "PUBLIC_ALL_TEST_OOD_ACCESS_AUDIT.json",
        public_root / "final" / "PUBLIC_ALL_METRICS.csv",
        public_root / "final" / "FULL_REPRO_MANIFEST.json",
    ]
    for path in evidence_files:
        _copy_file(path, package_dir / "source_evidence" / path.name)
    code_files = [
        Path("src/prism_benchmark/level_reconstruction.py"),
        Path("src/prism_benchmark/public5_level_r2_reporting.py"),
        Path("scripts/level_r2_reporting.py"),
        Path("scripts/report_public5_cpu_level_r2.py"),
        Path("tests/test_level_reconstruction.py"),
        Path("tests/test_public5_level_r2_reporting.py"),
    ]
    for relative in code_files:
        _copy_file(repository_root / relative, package_dir / "code" / relative)
    _copy_file(pytest_log, package_dir / "validation" / pytest_log.name)
    write_json(package_dir / "GIT_PROVENANCE.json", provenance)
    readme = "\n".join(
        [
            "# PRISM v2.1.1 Public5 CPU Level-R² report bundle",
            "",
            "This is a reporting-only reconstruction from frozen public-five PRISM/CPU predictions.",
            "It is not the canonical six-dataset bundle and contains no CZ or Neural-3 result.",
            "No model was refit, rerun for inference, or reselected. The reporting pass only reread frozen prediction and shared-target artifacts.",
            "",
            f"Open `report/{REPORT_NAME}` first.",
            "Use `results/PUBLIC5_CPU_LEVEL_R2_RESULTS.csv` for the complete row-level audit.",
            "",
        ]
    )
    (package_dir / "README.md").write_text(readme, encoding="utf-8")
    manifest_entries = []
    for path in sorted(item for item in package_dir.rglob("*") if item.is_file()):
        manifest_entries.append(
            {
                "path": path.relative_to(package_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "status": "PASS",
        "scope": "PUBLIC5_CPU_PRISM_REPORTING_ONLY",
        "canonical_six_dataset_report": False,
        "model_retrained": False,
        "model_reselected": False,
        "test_rerun": False,
        "ood_rerun": False,
        "raw_data_included": False,
        "prediction_parquet_included": False,
        "checkpoint_included": False,
        "git": provenance,
        "files": manifest_entries,
    }
    write_json(package_dir / "MANIFEST.json", manifest)
    _write_checksum_file(package_dir)
    verify_checksum_file(package_dir)
    forbidden = [
        path
        for path in package_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_PACKAGE_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"forbidden package files: {forbidden}")
    return_dir = output_root / "return"
    return_dir.mkdir(parents=True, exist_ok=True)
    zip_path = return_dir / PACKAGE_NAME
    if zip_path.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {zip_path}")
    with zipfile.ZipFile(
        zip_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(item for item in package_dir.rglob("*") if item.is_file()):
            archive.write(
                path,
                arcname=(Path(PACKAGE_ROOT_NAME) / path.relative_to(package_dir)).as_posix(),
            )
    with zipfile.ZipFile(zip_path) as archive:
        failed = archive.testzip()
        if failed is not None:
            raise RuntimeError(f"zip integrity failed: {failed}")
    verify_zip_checksums(zip_path)
    size = zip_path.stat().st_size
    if size >= 15 * 1024 * 1024:
        raise RuntimeError(f"bundle exceeds 15 MiB: {size}")
    digest = sha256_file(zip_path)
    sha_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    validation = {
        "status": "PASS",
        "zip_test": "PASS",
        "internal_sha256s": "PASS",
        "forbidden_files": [],
        "bytes": size,
        "sha256": digest,
        "bundle": str(zip_path),
        "bundle_sha256_file": str(sha_path),
    }
    validation_path = return_dir / "PACKAGE_VALIDATION.json"
    write_json(validation_path, validation)
    return validation


def generate_public5_cpu_level_r2_report(
    public_root: Path,
    output_root: Path,
    repository_root: Path,
    collect_level_r2_fn: Callable[..., tuple[pd.DataFrame, dict[str, Any]]],
    *,
    pytest_log: Path | None = None,
) -> dict[str, Any]:
    """Create the complete public-five CPU report without touching frozen inputs."""

    public_root = public_root.resolve()
    output_root = output_root.resolve()
    repository_root = repository_root.resolve()
    preflight = reporting_preflight(repository_root, pytest_log)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_audit = verify_public5_source(public_root)
    source = load_source_metrics(public_root)
    specs = prediction_specs_from_source(public_root, source)
    frozen_input_audit, prediction_hashes = verify_frozen_reporting_inputs(
        public_root,
        source,
        specs,
    )
    reconstructed, identity_audit = collect_level_r2_fn(
        output_root,
        public_root,
        specs=specs,
        known_prediction_sha256=prediction_hashes,
        require_common_support=True,
    )
    frame = merge_reconstructed_metrics(source, reconstructed)
    results_dir = output_root / "results"
    report_dir = output_root / "report"
    plot_dir = output_root / "plots"
    logs_dir = output_root / "logs"
    for directory in (results_dir, report_dir, plot_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "PUBLIC5_CPU_LEVEL_R2_RESULTS.csv"
    input_path = results_dir / "PUBLIC5_CPU_INPUT_ONLY_LEVEL_R2_LEADERBOARD.csv"
    dynamic_path = results_dir / "PUBLIC5_CPU_DYNAMIC_LEVEL_R2_LEADERBOARD.csv"
    frame.to_csv(results_path, index=False)
    frame.loc[frame["information_set"].eq("input_only")].sort_values(
        GROUP_COLUMNS + ["rank_by_level_r2", "model"],
        na_position="last",
    ).to_csv(input_path, index=False)
    frame.loc[frame["information_set"].eq("dynamic")].sort_values(
        GROUP_COLUMNS + ["rank_by_level_r2", "model"],
        na_position="last",
    ).to_csv(dynamic_path, index=False)
    frozen_input_audit_path = (
        results_dir / "PUBLIC5_CPU_FROZEN_REPORTING_INPUT_AUDIT.json"
    )
    write_json(frozen_input_audit_path, frozen_input_audit)
    source_status_counts = {
        str(key): int(value)
        for key, value in frame["source_status"].value_counts(dropna=False).items()
    }
    audit = {
        "status": "PASS",
        "scope": "PUBLIC5_CPU_PRISM_REPORTING_ONLY",
        "canonical_six_dataset_report": False,
        "model_retrained": False,
        "model_reselected": False,
        "hyperparameters_changed": False,
        "sample_support_changed": False,
        "test_rerun": False,
        "ood_rerun": False,
        "historical_results_overwritten": False,
        "datasets": sorted(frame["dataset"].dropna().unique().tolist()),
        "dataset_count": int(frame["dataset"].nunique()),
        "tasks": sorted(frame["target_head"].dropna().unique().tolist()),
        "task_count": int(frame["target_head"].nunique()),
        "models": sorted(frame["model"].dropna().unique().tolist()),
        "model_count": int(frame["model"].nunique()),
        "formal_source_rows": int(len(frame)),
        "reconstructed_prediction_artifacts": int(
            frame["reconstruction_status"].eq("PASS").sum()
        ),
        "source_status_counts": source_status_counts,
        "identity_checks_passed": True,
        "registered_window_width_checks_passed": True,
        "common_support_checks_passed": identity_audit[
            "common_support_checks_passed"
        ],
        "common_support_groups": identity_audit["common_support_groups"],
        "target_identity_max_abs_error": identity_audit[
            "target_identity_max_abs_error"
        ],
        "residual_identity_max_abs_error": identity_audit[
            "residual_identity_max_abs_error"
        ],
        "mse_identity_max_abs_error": identity_audit["identity_max_mse"],
        "rmse_identity_max_abs_error": identity_audit["identity_max_rmse"],
        "mae_identity_max_abs_error": identity_audit["identity_max_mae"],
        "r2_primary_reporting": "R2_LEVEL_RECONSTRUCTED",
        "r2_secondary_reporting": "R2_DELTA",
        "level_target_semantics": "REGISTERED_FUTURE_WINDOW_LEVEL",
        "same_prediction_error": True,
        "different_target_variance": True,
        "rankings": ["RANK_BY_LEVEL_R2", "RANK_BY_PERSISTENCE_SKILL"],
        "source": source_audit,
        "frozen_reporting_inputs": {
            "status": frozen_input_audit["status"],
            "full_repro_manifest_sha256": frozen_input_audit[
                "full_repro_manifest_sha256"
            ],
            "verified_file_count": frozen_input_audit["verified_file_count"],
            "verified_prediction_count": frozen_input_audit[
                "verified_prediction_count"
            ],
            "verified_input_set_sha256": frozen_input_audit[
                "verified_input_set_sha256"
            ],
            "audit_path": str(frozen_input_audit_path),
        },
        "reporting_preflight": preflight,
        "results": {
            "all": str(results_path),
            "input_only": str(input_path),
            "dynamic": str(dynamic_path),
        },
    }
    audit_path = results_dir / "PUBLIC5_CPU_LEVEL_R2_RECONSTRUCTION_AUDIT.json"
    write_json(audit_path, audit)
    chart_map = write_plots(frame, plot_dir)
    chart_map_path = output_root / "CHART_MAP.json"
    write_json(chart_map_path, {"status": "PASS", "charts": chart_map})
    report_path = report_dir / REPORT_NAME
    write_report(frame, audit, report_path)
    provenance_path = output_root / "GIT_PROVENANCE.json"
    write_json(provenance_path, git_provenance(repository_root))
    artifact_paths = [
        results_path,
        input_path,
        dynamic_path,
        frozen_input_audit_path,
        audit_path,
        report_path,
        chart_map_path,
        provenance_path,
        *sorted(plot_dir.glob("*.svg")),
    ]
    package = package_outputs(
        output_root,
        public_root,
        repository_root,
        artifact_paths,
        pytest_log,
    )
    summary = {
        "status": "PASS",
        "scope": audit["scope"],
        "datasets_processed": audit["dataset_count"],
        "tasks_processed": audit["task_count"],
        "models_recorded": audit["model_count"],
        "formal_source_rows": audit["formal_source_rows"],
        "reconstructed_prediction_artifacts": audit[
            "reconstructed_prediction_artifacts"
        ],
        "identity_audit": {
            "status": "PASS",
            "target_max_abs_error": audit["target_identity_max_abs_error"],
            "residual_max_abs_error": audit[
                "residual_identity_max_abs_error"
            ],
            "mse_max_abs_error": audit["mse_identity_max_abs_error"],
            "rmse_max_abs_error": audit["rmse_identity_max_abs_error"],
            "mae_max_abs_error": audit["mae_identity_max_abs_error"],
        },
        "report": str(report_path),
        "package": package,
    }
    summary_path = output_root / "PUBLIC5_CPU_LEVEL_R2_RUN_SUMMARY.json"
    write_json(summary_path, summary)
    return summary
