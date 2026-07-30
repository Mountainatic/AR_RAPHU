#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting import build_final_report
from src.runtime import atomic_json, environment_snapshot

EXCLUDED_PARTS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", "cache", "caches",
    "tmp", "temporary", "return",
}
EXCLUDED_SUFFIXES = {".xlsx", ".xls", ".pyc", ".pyo", ".tmp"}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        not any(part in EXCLUDED_PARTS for part in relative.parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
        and not path.name.startswith(".")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "return"))
    args = parser.parse_args()
    build_final_report(ROOT)
    atomic_json(ROOT / "results" / "environment.json", environment_snapshot(ROOT))
    output_dir = Path(args.output_dir).resolve()
    package_name = "MULTISCALE_PHYSICS_AUDIT_V1_RESULTS"
    staging = output_dir / package_name
    archive = output_dir / f"{package_name}_bundle.zip"
    checksum_file = Path(str(archive) + ".sha256")
    if staging.exists():
        shutil.rmtree(staging)
    archive.unlink(missing_ok=True)
    checksum_file.unlink(missing_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    source_files = [
        path for path in ROOT.rglob("*")
        if path.is_file() and include(path)
    ]
    for source in source_files:
        relative = source.relative_to(ROOT)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest_entries = []
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            relative = path.relative_to(staging).as_posix()
            stage = (
                "RESULT" if relative.startswith("results/")
                else "LOG" if relative.startswith("logs/")
                else "SOURCE"
            )
            manifest_entries.append({
                "path": relative,
                "size": path.stat().st_size,
                "sha256": digest(path),
                "type": path.suffix.lstrip(".") or "file",
                "generated_stage": stage,
            })
    manifest = {
        "schema": "MULTISCALE_PHYSICS_AUDIT_V1_PACKAGE",
        "files": manifest_entries,
    }
    atomic_json(staging / "PACKAGE_MANIFEST.json", manifest)
    sums = "\n".join(
        f"{entry['sha256']}  {entry['path']}" for entry in manifest_entries
    ) + "\n"
    (staging / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(package_name) / path.relative_to(staging))
    # Independent extraction and full manifest verification.
    with tempfile.TemporaryDirectory(prefix="multiscale_package_verify_") as temp:
        target = Path(temp)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            forbidden = [
                name for name in names
                if name.lower().endswith((".xlsx", ".xls"))
                or "/.git/" in name
                or "__pycache__" in name
            ]
            if forbidden:
                raise RuntimeError(f"PACKAGE_FORBIDDEN_FILES:{forbidden}")
            bundle.extractall(target)
        unpacked = target / package_name
        stored_manifest = json.loads(
            (unpacked / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
        )
        for entry in stored_manifest["files"]:
            candidate = unpacked / entry["path"]
            if not candidate.is_file() or digest(candidate) != entry["sha256"]:
                raise RuntimeError(f"PACKAGE_HASH_MISMATCH:{entry['path']}")
    archive_hash = digest(archive)
    checksum_file.write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8"
    )
    print(f"FINAL_ZIP={archive}")
    print(f"FINAL_SHA256={archive_hash}")
    print(f"ZIP_SIZE={archive.stat().st_size}")
    print(f"MANIFEST_FILE_COUNT={len(manifest_entries)}")
    print("VALIDATION_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
