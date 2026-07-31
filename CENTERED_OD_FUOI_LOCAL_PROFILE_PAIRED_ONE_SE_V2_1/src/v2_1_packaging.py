from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


PACKAGE = "CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_v2_1_package(root: Path) -> None:
    forbidden = {".git", "__pycache__", ".pytest_cache", "work", "cache", "raw_data", "results"}
    for path in root.rglob("*"):
        if any(part in forbidden for part in path.parts) or path.suffix.lower() in {".xlsx", ".xls"}:
            raise RuntimeError(f"FORBIDDEN_PACKAGE_PATH:{path}")


def build_v2_1_bundle(
    source_root: Path,
    *,
    results_root: Path,
    protocol_sha256: str,
    shared_sha256: str,
    cpu_sha256: str,
    gpu_sha256: str,
    v1_sha256: str,
    v2_sha256: str,
    selection_status: str,
    estimator_status: str,
) -> dict[str, Any]:
    return_root = source_root / "return_v2_1"
    package_root = return_root / PACKAGE
    zip_path = return_root / f"{PACKAGE}_bundle.zip"
    sha_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    if package_root.exists():
        shutil.rmtree(package_root)
    for path in (zip_path, sha_path):
        if path.exists():
            path.unlink()
    package_root.mkdir(parents=True)
    include = [
        "configs", "src", "scripts", "tests", "README.md", "RUN_CPU_CONFIRM.sh", "RESUME_CPU_CONFIRM.sh",
        "CENTERED_OD_FUOI_V2_TO_V2_1_LOCAL_PROFILE_PAIRED_ONE_SE_PATCH_PLAN.md",
    ]
    forbidden = {"__pycache__", ".pytest_cache", ".git", "work", "cache", "return", "return_v2_1", "results", "results_v2_1"}
    for relative in include:
        source = source_root / relative
        if not source.exists():
            continue
        if source.is_file():
            shutil.copy2(source, package_root / source.name)
            continue
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(source_root)
            if any(part in forbidden for part in rel.parts):
                continue
            if path.suffix.lower() in {".xlsx", ".xls"}:
                raise RuntimeError(f"PRIVATE_DATA_IN_PACKAGE:{path}")
            target = package_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for path in results_root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = Path("results_v2_1") / path.relative_to(results_root)
        target = package_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    validate_v2_1_package(package_root)
    try:
        git_commit = subprocess.check_output(["git", "-C", str(source_root.parent), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "UNAVAILABLE"
    files = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS.txt"}:
            relative = path.relative_to(package_root).as_posix()
            files.append({
                "path": relative, "size": path.stat().st_size, "sha256": _sha(path),
                "type": "generated_result" if relative.startswith("results_v2_1/") else "source_or_protocol",
                "generated_stage": "E2B-E9" if relative.startswith("results_v2_1/") else "implementation",
            })
    manifest = {
        "schema": PACKAGE, "git_commit": git_commit,
        "protocol_sha256": protocol_sha256, "shared_dataset_sha256": shared_sha256,
        "cpu_baseline_bundle_sha256": cpu_sha256, "gpu_baseline_bundle_sha256": gpu_sha256,
        "v1_results_sha256": v1_sha256, "v2_results_sha256": v2_sha256,
        "selection_status": selection_status, "estimator_status": estimator_status,
        "files": files,
    }
    (package_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (package_root / "SHA256SUMS.txt").write_text("\n".join(f"{row['sha256']}  {row['path']}" for row in files) + "\n", encoding="utf-8")
    for row in files:
        if _sha(package_root / row["path"]) != row["sha256"]:
            raise RuntimeError(f"PACKAGE_HASH_FAILED:{row['path']}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{PACKAGE}/{path.relative_to(package_root).as_posix()}")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP_ROUNDTRIP_FAILED")
        if any(name.lower().endswith((".xlsx", ".xls")) or "/.git/" in name or "__pycache__" in name for name in archive.namelist()):
            raise RuntimeError("ZIP_PRIVACY_FAILED")
    digest = _sha(zip_path)
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return {"zip": str(zip_path.resolve()), "sha256": digest, "size": zip_path.stat().st_size, "manifest_file_count": len(files)}
