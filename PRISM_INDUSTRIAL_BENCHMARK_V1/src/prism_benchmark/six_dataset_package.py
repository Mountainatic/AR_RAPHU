from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .cpu_data import sha256_file
from .six_dataset_extension import (
    EXTENSION_BRANCH,
    EXTENSION_EVIDENCE_CLASS,
    public5_root,
)
from .six_dataset_materialization import ACCESS_AUDIT_NAME
from .six_dataset_reporting import (
    NEURAL_MODELS,
    PRISM_DYNAMIC,
    PRISM_INPUT,
    PRISM_JOINT,
)
from .stage0 import write_json
from .v211_support import SUPPORT_CONTRACT


PACKAGE_STEM = "PRISM_V2_1_1_CZ_NEURAL3_SIX_DATASET_RESULTS_bundle"
SMALL_FILE_LIMIT = 2 * 1024 * 1024
EXCLUDED_SUFFIXES = {".xlsx", ".parquet", ".pt", ".pth", ".ckpt", ".pyc"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _git(project: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=project, text=True
    ).strip()


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No rows available.\n"
    values = frame[columns].fillna("").astype(str)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in values.itertuples(index=False, name=None)
    )
    return "\n".join(lines) + "\n"


def _overall(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        (metrics["split"] == "test")
        & (metrics["status"] == "PASS")
        & metrics["coverage_complete"].astype(bool)
    ]
    return (
        selected.groupby(["information_set", "model"], as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            wins=("rank", lambda values: int((values == 1.0).sum())),
            tasks=("target_head", "count"),
        )
        .sort_values(["information_set", "mean_rank", "model"])
    )


def _best(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        (metrics["split"] == "test")
        & (metrics["status"] == "PASS")
        & metrics["coverage_complete"].astype(bool)
    ].sort_values(
        ["scope", "direction", "target_head", "information_set", "mse", "model"]
    )
    return selected.groupby(
        ["scope", "direction", "target_head", "information_set"],
        as_index=False,
        dropna=False,
    ).first()


def write_extension_documents(
    run_root: Path,
    project: Path,
    *,
    public_root: Path | None = None,
    generating_commit: str | None = None,
    reporting_commit: str | None = None,
) -> dict[str, Any]:
    public_root = public5_root() if public_root is None else public_root.resolve()
    final = run_root / "final"
    metrics = pd.read_csv(final / "SIX_DATASET_METRICS.csv")
    overall = _overall(metrics)
    best = _best(metrics)
    overall.to_csv(final / "SIX_DATASET_OVERALL_MODEL_RANK.csv", index=False)
    freeze_path = (
        run_root / "freeze" / "SIX_DATASET_CZ_NEURAL3_DEVELOPMENT_FREEZE.json"
    )
    freeze = _read_json(freeze_path)
    access_path = final / ACCESS_AUDIT_NAME
    access = _read_json(access_path)
    public_summary = _read_json(
        public_root / "final" / "PUBLIC_ALL_FINAL_EVIDENCE_SUMMARY.json"
    )

    neural_report = final / "NEURAL3_SIX_DATASET_SUMMARY.md"
    neural_report.write_text(
        "\n".join(
            [
                "# Neural-3 Six-Dataset Summary",
                "",
                "Evidence class: " + EXTENSION_EVIDENCE_CLASS,
                "",
                "Neural selection used development validation only. "
                "Public-five PRISM and CPU models were not retrained.",
                "",
                "## Overall rank",
                "",
                _table(
                    overall[overall["model"].isin(NEURAL_MODELS)],
                    [
                        "information_set",
                        "model",
                        "mean_rank",
                        "median_rank",
                        "wins",
                        "tasks",
                    ],
                ),
                "## Task winners",
                "",
                _table(
                    best,
                    [
                        "scope",
                        "direction",
                        "target_head",
                        "information_set",
                        "model",
                        "mse",
                        "rmse",
                        "r2",
                    ],
                ),
                "## Interpretation",
                "",
                "These results are predictive transfer evidence, not causal proof.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cz = metrics[
        (metrics["scope"] == "cz") & (metrics["split"] == "test")
    ]
    cz_report = final / "CZ_TWO_ROD_PRISM_NEURAL3_REPORT.md"
    cz_report.write_text(
        "\n".join(
            [
                "# CZ Two-Rod PRISM and Neural-3 Report",
                "",
                "Task CZ_D20 uses 10 s cadence, h=120, W=12, and W0=12.",
                "The two transfer directions remain separate.",
                "",
                _table(
                    cz.sort_values(["direction", "information_set", "mse"]),
                    [
                        "direction",
                        "information_set",
                        "model",
                        "rows",
                        "mse",
                        "rmse",
                        "mae",
                        "r2",
                        "persistence_skill",
                    ],
                ),
                "",
                "Channel activation and cross-rod performance are conditional "
                "predictive evidence, not causal proof.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    final_report = final / "PRISM_V211_CZ_NEURAL3_SIX_DATASET_FINAL_REPORT.md"
    final_report.write_text(
        "\n".join(
            [
                "# PRISM v2.1.1 CZ + Neural-3 Six-Dataset Extension",
                "",
                "Evidence class: " + EXTENSION_EVIDENCE_CLASS,
                "",
                "The completed public-five experiment is preserved. This "
                "extension adds CZ two-rod transfer and LSTM, iTransformer, "
                "and TimeMixer.",
                "",
                "## Input-only top models",
                "",
                _table(
                    overall[overall["information_set"] == "input_only"].head(5),
                    ["model", "mean_rank", "median_rank", "wins", "tasks"],
                ),
                "## Dynamic top models",
                "",
                _table(
                    overall[overall["information_set"] == "dynamic"].head(5),
                    ["model", "mean_rank", "median_rank", "wins", "tasks"],
                ),
                "## Controls",
                "",
                "Neural test metrics used for tuning: false.",
                "CZ target rod used for selection: false.",
                "Public-five PRISM/CPU retrained: false.",
                "Post-test reselection: false.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    generation = generating_commit or freeze.get("extension_generating_commit")
    reporting = reporting_commit or _git(project, "rev-parse", "HEAD")
    rankings = {
        info: {
            str(row.model): {
                "mean_rank": float(row.mean_rank),
                "median_rank": float(row.median_rank),
                "wins": int(row.wins),
            }
            for row in values.itertuples(index=False)
        }
        for info, values in overall.groupby("information_set")
    }
    evidence = {
        "status": "EXPERIMENT_COMPLETED",
        "evidence_class": EXTENSION_EVIDENCE_CLASS,
        "public5_source_branch": public_summary.get("source_branch"),
        "public5_source_commit": public_summary.get("source_commit"),
        "public5_reporting_commit": public_summary.get("reporting_commit"),
        "public5_original_freeze_sha": public_summary.get(
            "development_freeze_sha"
        ),
        "extension_branch": _git(project, "branch", "--show-current"),
        "extension_generating_commit": generation,
        "extension_reporting_commit": reporting,
        "support_contract": SUPPORT_CONTRACT,
        "cz_raw_sha256": freeze.get("cz_raw_sha256"),
        "cz_registry_sha256": freeze.get("cz_registry_sha256"),
        "datasets": 6,
        "public_primary_heads": 7,
        "cz_primary_heads": 1,
        "cz_transfer_directions": 2,
        "neural_models": list(NEURAL_MODELS),
        "input_only": True,
        "dynamic": True,
        "rankings": rankings,
        "test_accessed": access.get("test_accessed"),
        "ood_accessed": access.get("ood_accessed"),
        "neural_development_test_accessed": False,
        "cz_target_rod_used_for_selection": False,
        "public5_prism_cpu_retrained": False,
        "post_test_reselection": False,
        "development_freeze_sha256": sha256_file(freeze_path),
        "test_access_audit_sha256": sha256_file(access_path),
    }
    evidence_path = (
        final / "PRISM_V211_CZ_NEURAL3_SIX_DATASET_FINAL_EVIDENCE.json"
    )
    write_json(evidence_path, evidence)
    return {
        "status": "PASS",
        "neural_report": str(neural_report),
        "cz_report": str(cz_report),
        "final_report": str(final_report),
        "evidence": str(evidence_path),
        "overall_rank": str(final / "SIX_DATASET_OVERALL_MODEL_RANK.csv"),
    }


def _role(path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        return "PREDICTION_OR_SHARED_PARQUET"
    if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return "MODEL_CHECKPOINT"
    if path.suffix.lower() == ".xlsx":
        return "PRIVATE_RAW_DATA_REFERENCE"
    if "sample_ids" in path.parts:
        return "SAMPLE_REGISTRY"
    if "base_data" in path.parts:
        return "SHARED_BASE_DATA"
    return "REPRODUCTION_ARTIFACT"


def _stage(path: Path) -> str:
    text = path.as_posix()
    if "/shared/" in text:
        return "E4_CZ_SHARED_BUILD"
    if "/DEVELOPMENT/" in text or "/BASELINE_DEVELOPMENT/" in text:
        return "D_DEVELOPMENT"
    if "/FINAL/" in text or "/final/" in text:
        return "T_FINAL_OR_REPORT"
    return "EXTENSION"


def write_full_repro_manifest(
    run_root: Path,
    *,
    extra_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    final = run_root / "final"
    manifest_path = final / "SIX_DATASET_FULL_REPRO_MANIFEST.json"
    candidates: set[Path] = set()
    for base, _directories, files in os.walk(run_root):
        for name in files:
            path = Path(base) / name
            if path == manifest_path or "return" in path.parts:
                continue
            if (
                path.suffix.lower() in EXCLUDED_SUFFIXES
                or path.stat().st_size > SMALL_FILE_LIMIT
            ):
                candidates.add(path.resolve())
    for path in extra_paths:
        if path.is_file():
            candidates.add(path.resolve())
    records = [
        {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "role": _role(path),
            "generated_by_stage": _stage(path),
        }
        for path in sorted(candidates)
    ]
    payload = {"status": "PASS", "files": len(records), "records": records}
    write_json(manifest_path, payload)
    return payload


def _small_files(run_root: Path) -> list[Path]:
    result: list[Path] = []
    for root_name in ("final", "freeze", "logs"):
        root = run_root / root_name
        if not root.is_dir():
            continue
        for base, _directories, files in os.walk(root):
            for name in files:
                path = Path(base) / name
                if path.suffix.lower() in EXCLUDED_SUFFIXES:
                    continue
                if path.stat().st_size <= SMALL_FILE_LIMIT:
                    result.append(path)
    return sorted(set(result))


def _source_files(project: Path) -> list[Path]:
    relative_names = [
        "scripts/run_cz_neural3_extension.py",
        "src/prism_benchmark/cz_baselines.py",
          "src/prism_benchmark/cz_k_support.py",
        "src/prism_benchmark/cz_extension.py",
        "src/prism_benchmark/neural3.py",
        "src/prism_benchmark/six_dataset_extension.py",
        "src/prism_benchmark/six_dataset_materialization.py",
        "src/prism_benchmark/six_dataset_reporting.py",
        "src/prism_benchmark/six_dataset_package.py",
        "src/prism_benchmark/v211_k.py",
        "src/prism_benchmark/v211_public_all_baselines.py",
        "src/prism_benchmark/cpu_data.py",
        "tests/test_cz_neural3_extension.py",
        "tests/test_v211_native_support.py",
        "tests/test_v211_public_all_baselines.py",
    ]
    result = [
        project / name
        for name in relative_names
        if (project / name).is_file()
    ]
    registry = project / "dataset_registry" / "cz_czochralski"
    if registry.is_dir():
        result.extend(path for path in registry.rglob("*") if path.is_file())
    return sorted(set(result))


def _copy(source: Path, root: Path, relative: Path) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _verify_internal_hashes(package_path: Path) -> int:
    with zipfile.ZipFile(package_path, "r") as archive:
        checksum_text = archive.read("SHA256SUMS.txt").decode("utf-8")
        checked = 0
        for line in checksum_text.splitlines():
            if not line.strip():
                continue
            expected, name = line.split(None, 1)
            name = name.strip()
            observed = hashlib.sha256(archive.read(name)).hexdigest()
            if observed != expected:
                raise RuntimeError(f"internal hash mismatch: {name}")
            checked += 1
    return checked


def package_extension(
    run_root: Path,
    project: Path,
    *,
    public_root: Path | None = None,
    reporting_commit: str | None = None,
) -> dict[str, Any]:
    public_root = public5_root() if public_root is None else public_root.resolve()
    raw_paths = sorted(
        Path(
            "/root/autodl-tmp/PRISM_DATASETS_V1/raw_sources/cz_czochralski"
        ).glob("*.xlsx")
    )
    manifest = write_full_repro_manifest(
        run_root,
        extra_paths=raw_paths,
    )
    reporting = reporting_commit or _git(project, "rev-parse", "HEAD")
    documents = write_extension_documents(
        run_root,
        project,
        public_root=public_root,
        reporting_commit=reporting,
    )

    return_root = run_root / "return"
    return_root.mkdir(parents=True, exist_ok=True)
    staging = return_root / (PACKAGE_STEM + "_contents")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for path in _small_files(run_root):
        _copy(path, staging, Path("run") / path.relative_to(run_root))
    for path in _source_files(project):
        _copy(path, staging, Path("source") / path.relative_to(project))

    freeze = _read_json(
        run_root
        / "freeze"
        / "SIX_DATASET_CZ_NEURAL3_DEVELOPMENT_FREEZE.json"
    )
    provenance = {
        "branch": _git(project, "branch", "--show-current"),
        "generating_commit": freeze.get("extension_generating_commit"),
        "reporting_commit": reporting,
        "git_status": _git(project, "status", "--short"),
        "public5_root": str(public_root),
        "public5_final_evidence_sha256": sha256_file(
            public_root / "final" / "PUBLIC_ALL_FINAL_EVIDENCE_SUMMARY.json"
        ),
        "full_repro_manifest_files": manifest["files"],
        "documents": documents,
    }
    write_json(staging / "GIT_PROVENANCE.json", provenance)
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    (staging / "MANIFEST.txt").write_text(
        "\n".join(
            str(path.relative_to(staging)).replace(os.sep, "/")
            for path in files
        )
        + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    (staging / "SHA256SUMS.txt").write_text(
        "\n".join(
            f"{sha256_file(path)}  "
            f"{str(path.relative_to(staging)).replace(os.sep, '/')}"
            for path in files
            if path.name != "SHA256SUMS.txt"
        )
        + "\n",
        encoding="utf-8",
    )

    package_path = return_root / (PACKAGE_STEM + ".zip")
    sidecar = return_root / (PACKAGE_STEM + ".zip.sha256")
    package_path.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    with zipfile.ZipFile(
        package_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=str(path.relative_to(staging)).replace(os.sep, "/"),
                )
    with zipfile.ZipFile(package_path, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("zip test failed")
    internal_hash_files = _verify_internal_hashes(package_path)
    package_sha = sha256_file(package_path)
    sidecar.write_text(
        f"{package_sha}  {package_path.name}\n",
        encoding="utf-8",
    )
    result = {
        "status": "PASS",
        "package_path": str(package_path),
        "package_bytes": int(package_path.stat().st_size),
        "package_sha256": package_sha,
        "sidecar_path": str(sidecar),
        "unzip_test": "PASS",
        "internal_hash_files": internal_hash_files,
        "internal_hashes": "PASS",
        "within_target": package_path.stat().st_size < 15 * 1024 * 1024,
        "reporting_commit": reporting,
        "created_at_unix": time.time(),
    }
    write_json(run_root / "final" / "PACKAGE_STATUS.json", result)
    return result
