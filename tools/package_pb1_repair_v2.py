#!/usr/bin/env python3
"""Build and validate the self-contained PB1 repair-v2 development package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "OPS_UOI_PB1_DEVELOPMENT_REPAIR_V2_20260728"
REQUIRED_INPUTS = (
    "PB1_DEVELOPMENT_REPAIR_V2_REPORT.md",
    "PB1_DEVELOPMENT_REPAIR_V2_STATUS.json",
    "PB1_REPAIR_PREFLIGHT_V2.yaml",
    "configs",
    "src",
    "tools",
    "tests",
    "results/public_benchmarks/pb1_repair_v2",
)
FORBIDDEN_GIT_PATH_FRAGMENTS = (
    "实验数据1.xlsx",
    "id_rsa",
    ".pem",
    ".env",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def assert_safe_target(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    parent = output_root.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise RuntimeError(f"Unsafe package target: {resolved}")


def audit_repository_bundle_scope() -> None:
    object_paths = git("rev-list", "--objects", "--all").lower()
    bad = [
        fragment
        for fragment in FORBIDDEN_GIT_PATH_FRAGMENTS
        if fragment.lower() in object_paths
    ]
    if bad:
        raise RuntimeError(
            "Repository bundle refused because sensitive path fragments occur "
            f"in Git history: {bad}"
        )


def write_environment(path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "runtime_manager": os.environ.get("AR_RAPHU_RUNTIME_MANAGER", "unknown"),
        "numerical_protocol": "CPU_FP64",
        "official_test_access_count": 0,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "environment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def copy_inputs(stage: Path) -> None:
    for relative in REQUIRED_INPUTS:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"Required package input is missing: {source}")
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache", ".git"
                ),
            )
        else:
            shutil.copy2(source, target)


def write_package_docs(stage: Path, source_commit: str) -> None:
    (stage / "README.md").write_text(
        "# PB1 Development Repair V2 Result Package\n\n"
        "Development-only results, source, configuration, tests, and a Git "
        "repository bundle. Official test data were not accessed or included.\n",
        encoding="utf-8",
    )
    (stage / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "- Repair-v2 numerical solver, H3 direct forecasts, literature baselines, "
        "fixed-model every-horizon bootstrap, coverage audit, and explicit blockers.\n",
        encoding="utf-8",
    )
    (stage / "SOURCE_COMMIT.txt").write_text(source_commit + "\n", encoding="utf-8")


def file_manifest(stage: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        if path.name in {"PACKAGE_MANIFEST.json", "SHA256SUMS.txt"}:
            continue
        records.append(
            {
                "path": path.relative_to(stage).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def write_manifest_and_sums(stage: Path, name: str, source_commit: str) -> None:
    records = file_manifest(stage)
    manifest = {
        "schema_version": 2,
        "package": f"{name}.zip",
        "scope": "PB1_DEVELOPMENT_REPAIR_V2_NO_RAW_DATA_NO_OFFICIAL_TEST",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "privacy_guards": {
            "raw_public_data_included": False,
            "private_cz_included": False,
            "credentials_included": False,
            "official_test_rows_included": False,
        },
        "file_count_before_manifest": len(records),
        "files": records,
    }
    (stage / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sum_paths = sorted(
        path
        for path in stage.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [
        f"{sha256(path)}  {path.relative_to(stage).as_posix()}" for path in sum_paths
    ]
    (stage / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_stage(stage: Path) -> None:
    required = (
        "README.md",
        "CHANGELOG.md",
        "PB1_DEVELOPMENT_REPAIR_V2_REPORT.md",
        "PB1_DEVELOPMENT_REPAIR_V2_STATUS.json",
        "PB1_REPAIR_PREFLIGHT_V2.yaml",
        "configs",
        "src",
        "tools",
        "tests",
        "results",
        "environment",
        "repository.bundle",
        "SOURCE_COMMIT.txt",
        "PACKAGE_MANIFEST.json",
        "SHA256SUMS.txt",
    )
    missing = [name for name in required if not (stage / name).exists()]
    if missing:
        raise RuntimeError(f"Missing required package entries: {missing}")
    json.loads((stage / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    status = json.loads(
        (stage / "PB1_DEVELOPMENT_REPAIR_V2_STATUS.json").read_text(encoding="utf-8")
    )
    if status.get("official_test_access_count") != 0:
        raise RuntimeError("Package refused: official test access count is non-zero")
    for line in (stage / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256(stage / relative) != expected:
            raise RuntimeError(f"SHA256 mismatch: {relative}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--output-root", type=Path, default=ROOT / "return")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage = output_root / args.name
    zip_path = output_root / f"{args.name}.zip"
    sha_path = output_root / f"{args.name}.zip.sha256"
    assert_safe_target(stage, output_root)
    assert_safe_target(zip_path, output_root)
    audit_repository_bundle_scope()

    if stage.exists():
        shutil.rmtree(stage)
    if zip_path.exists():
        zip_path.unlink()
    if sha_path.exists():
        sha_path.unlink()
    stage.mkdir(parents=True)

    source_commit = git("rev-parse", "HEAD")
    copy_inputs(stage)
    write_package_docs(stage, source_commit)
    write_environment(stage / "environment")
    subprocess.run(
        ["git", "bundle", "create", str(stage / "repository.bundle"), "--all"],
        cwd=ROOT,
        check=True,
    )
    write_manifest_and_sums(stage, args.name, source_commit)
    verify_stage(stage)

    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            archive.write(path, f"{args.name}/{path.relative_to(stage).as_posix()}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")

    digest = sha256(zip_path)
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "stage": str(stage),
                "zip": str(zip_path),
                "zip_sha256": digest,
                "package_sha256_valid": True,
                "zip_crc_valid": True,
                "source_commit_recorded": True,
                "repository_bundle_included": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
