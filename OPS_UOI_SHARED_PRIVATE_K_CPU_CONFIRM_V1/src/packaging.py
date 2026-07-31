from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kind(path: Path) -> str:
    if "results" in path.parts:
        return "generated_result"
    if path.suffix in {".py", ".sh"}:
        return "source"
    if path.suffix in {".yaml", ".json"}:
        return "configuration_or_metadata"
    return "documentation_or_artifact"


def build_bundle(
    root: Path,
    *,
    shared_hash: str,
    cpu_hash: str,
    gpu_hash: str,
    protocol_hash: str,
) -> dict[str, Any]:
    return_root = root / "return"
    package_name = "OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS"
    package_root = return_root / package_name
    zip_path = return_root / f"{package_name}_bundle.zip"
    if package_root.exists():
        shutil.rmtree(package_root)
    if zip_path.exists():
        zip_path.unlink()
    package_root.mkdir(parents=True)
    include = [
        "configs",
        "src",
        "scripts",
        "tests",
        "results",
        "logs",
        "README.md",
        "RUN_CPU_CONFIRM.sh",
        "RESUME_CPU_CONFIRM.sh",
        "OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_EXPERIMENT_PLAN.md",
    ]
    forbidden_parts = {
        "__pycache__",
        ".pytest_cache",
        ".git",
        "work",
        "cache",
    }
    for relative in include:
        source = root / relative
        if not source.exists():
            continue
        if source.is_file():
            shutil.copy2(source, package_root / source.name)
            continue
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if any(part in forbidden_parts for part in rel.parts):
                continue
            if path.suffix.lower() in {".xlsx", ".xls"}:
                raise RuntimeError(f"PRIVATE_DATA_IN_PACKAGE:{path}")
            target = package_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(root.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        git_commit = "UNAVAILABLE"
    manifest_rows = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.name == "PACKAGE_MANIFEST.json":
            continue
        relative = path.relative_to(package_root).as_posix()
        manifest_rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
                "type": _kind(Path(relative)),
                "generated_stage": (
                    "E0-E8" if relative.startswith("results/") else "implementation"
                ),
            }
        )
    manifest = {
        "schema": package_name,
        "git_commit": git_commit,
        "protocol_sha256": protocol_hash,
        "shared_dataset_sha256": shared_hash,
        "cpu_baseline_bundle_sha256": cpu_hash,
        "gpu_baseline_bundle_sha256": gpu_hash,
        "files": manifest_rows,
    }
    (package_root / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checksum_lines = [
        f"{row['sha256']}  {row['path']}" for row in manifest_rows
    ]
    (package_root / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    for row in manifest_rows:
        path = package_root / row["path"]
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"PACKAGE_HASH_FAILED:{row['path']}")
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=f"{package_name}/{path.relative_to(package_root).as_posix()}",
                )
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP_ROUNDTRIP_FAILED:{bad}")
        names = archive.namelist()
        if any(
            name.lower().endswith((".xlsx", ".xls"))
            or "/.git/" in name
            or "__pycache__" in name
            for name in names
        ):
            raise RuntimeError("FORBIDDEN_FILE_IN_ZIP")
    digest = _sha256(zip_path)
    sha_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return {
        "zip": str(zip_path.resolve()),
        "sha256": digest,
        "size": zip_path.stat().st_size,
        "manifest_file_count": len(manifest_rows),
    }
