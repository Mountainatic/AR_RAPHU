#!/usr/bin/env python3
"""Build a privacy-safe, resumable AutoDL transfer archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "AR_RAPHU_AUTODL"
ROOT_FILES = [
    "AGENTS.md",
    "AR_RAPHU_method_v2.md",
    "AR_RAPHU_three_layer_validation_plan_v2.md",
    "AR_RAPHU_v2_revision_notes.md",
    "pyproject.toml",
    "uv.lock",
]
ROOT_DIRECTORIES = [
    "configs",
    "deploy",
    "src",
    "tests",
    "tools",
    "STAGE1_DUAL_SOLVER_V20_bundle",
    "results/phase1",
]
FORBIDDEN = {
    "实验数据1.xlsx",
    "data_manifests/cz",
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excluded(relative: Path) -> bool:
    parts = set(relative.parts)
    if "__pycache__" in parts or ".pytest_cache" in parts:
        return True
    if relative.name.endswith((".pyc", ".tmp", ".lock")):
        return True
    if relative.name == "实验数据1.xlsx":
        return True
    if relative.parts[:2] == ("data_manifests", "cz"):
        return True
    return False


def selected_files() -> list[Path]:
    selected: set[Path] = set()
    for item in ROOT_FILES:
        path = PROJECT_ROOT / item
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.add(path)
    for item in ROOT_DIRECTORIES:
        root = PROJECT_ROOT / item
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("*"):
            if path.is_file() and not excluded(path.relative_to(PROJECT_ROOT)):
                selected.add(path)
    return sorted(selected, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def successful_done_count() -> int:
    count = 0
    root = PROJECT_ROOT / "results" / "phase1" / "job_records"
    for path in root.rglob("DONE.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("returncode") == 0:
                count += 1
        except (OSError, json.JSONDecodeError):
            continue
    return count


def add_file(archive: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = archive.gettarinfo(str(path), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    with path.open("rb") as handle:
        archive.addfile(info, handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / "AR_RAPHU_AUTODL.tar.gz",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = selected_files()
    relative_names = [
        path.relative_to(PROJECT_ROOT).as_posix() for path in files
    ]
    for forbidden in FORBIDDEN:
        if any(name == forbidden or name.startswith(f"{forbidden}/") for name in relative_names):
            raise RuntimeError(f"Forbidden private path selected: {forbidden}")
    entries = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "status": "COMPLETED",
        "created_unix_time": time.time(),
        "archive_root": ARCHIVE_ROOT,
        "file_count": len(entries),
        "total_uncompressed_bytes": sum(item["bytes"] for item in entries),
        "successful_done_records_included": successful_done_count(),
        "private_cz_included": False,
        "forbidden_paths_confirmed_absent": sorted(FORBIDDEN),
        "entries": entries,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            add_file(archive, path, f"{ARCHIVE_ROOT}/{relative}")
        info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/PACKAGE_MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest_bytes))
    digest = sha256_file(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "archive": str(output),
                "sha256": digest,
                "bytes": output.stat().st_size,
                "file_count": len(entries) + 1,
                "private_cz_included": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
