"""Manifested CPU return package with independent ZIP round-trip validation."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .common import atomic_json, sha256_file


FORBIDDEN_PARTS = {
    ".git", "__pycache__", ".pytest_cache", "cache", "caches", "tmp", "return"
}
FORBIDDEN_SUFFIXES = {".xlsx", ".xls", ".pyc", ".pyo", ".tmp"}


def validate_tree(root: Path) -> dict[str, Any]:
    problems = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            any(part in FORBIDDEN_PARTS for part in relative.parts)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            problems.append(relative.as_posix())
    manifest_path = root / "PACKAGE_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            path = root / entry["path"]
            if not path.is_file():
                problems.append(f"MISSING:{entry['path']}")
            elif sha256_file(path) != entry["sha256"]:
                problems.append(f"HASH:{entry['path']}")
    return {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
    }


def build_cpu_package(
    *,
    project_root: Path,
    results_root: Path,
    shared_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    package_name = "PHYSICS_FIRST_CPU_RESULTS"
    staging = output_root
    archive = output_root.parent / f"{package_name}_bundle.zip"
    checksum = Path(str(archive) + ".sha256")
    if staging.exists():
        shutil.rmtree(staging)
    archive.unlink(missing_ok=True)
    checksum.unlink(missing_ok=True)
    staging.mkdir(parents=True)
    selections = [
        project_root / "configs",
        project_root / "src",
        project_root / "scripts",
        project_root / "tests",
        results_root,
        shared_root,
    ]
    for source in selections:
        if not source.exists():
            continue
        destination = staging / source.relative_to(project_root)
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns(
                    "__pycache__", ".pytest_cache", "*.pyc", "*.tmp", "*.xlsx"
                ),
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for name in (
        "README_CPU.md",
        "RUN_CPU.sh",
        "RESUME_CPU.sh",
        "requirements_cpu.txt",
    ):
        source = project_root / name
        if source.is_file():
            shutil.copy2(source, staging / name)
    entries = []
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            relative = path.relative_to(staging).as_posix()
            entries.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "type": path.suffix.lstrip(".") or "file",
                    "generated_stage": (
                        "SHARED"
                        if relative.startswith("shared/")
                        else "RESULT"
                        if relative.startswith("results_cpu/")
                        else "SOURCE"
                    ),
                }
            )
    atomic_json(
        staging / "PACKAGE_MANIFEST.json",
        {"schema": "PHYSICS_FIRST_CPU_RESULTS_PACKAGE_V1", "files": entries},
    )
    (staging / "SHA256SUMS.txt").write_text(
        "\n".join(f"{entry['sha256']}  {entry['path']}" for entry in entries)
        + "\n",
        encoding="utf-8",
    )
    tree_validation = validate_tree(staging)
    if tree_validation["status"] != "PASS":
        raise RuntimeError(f"PACKAGE_TREE_INVALID:{tree_validation}")
    with zipfile.ZipFile(
        archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(package_name) / path.relative_to(staging))
    with tempfile.TemporaryDirectory(prefix="physics_cpu_zip_verify_") as temp:
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            if any(
                name.lower().endswith((".xlsx", ".xls"))
                or "/.git/" in name
                or "__pycache__" in name
                for name in names
            ):
                raise RuntimeError("FORBIDDEN_FILE_IN_ZIP")
            bundle.extractall(temp)
        validation = validate_tree(Path(temp) / package_name)
        if validation["status"] != "PASS":
            raise RuntimeError(f"ZIP_ROUNDTRIP_INVALID:{validation}")
    archive_sha = sha256_file(archive)
    checksum.write_text(
        f"{archive_sha}  {archive.name}\n", encoding="utf-8"
    )
    return {
        "status": "PASS",
        "archive": str(archive.resolve()),
        "sha256": archive_sha,
        "size": archive.stat().st_size,
        "manifest_file_count": len(entries),
        "checksum_file": str(checksum.resolve()),
    }
