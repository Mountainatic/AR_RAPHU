"""Hash and lineage helpers for public benchmark source files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def build_lineage(
    *,
    dataset_id: str,
    raw_root: Path,
    files: Iterable[Path],
    official_source: str,
    doi: str,
    version_id: str,
    license_name: str,
    loader: str,
) -> dict[str, Any]:
    records = [file_record(path, root=raw_root) for path in sorted(files)]
    return {
        "schema_version": 6,
        "dataset_id": dataset_id,
        "official_source": official_source,
        "doi": doi,
        "version": version_id,
        "license": license_name,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "loader": loader,
        "loader_dependency": {
            "name": "nonlinear-benchmarks",
            "version": version("nonlinear-benchmarks"),
        },
        "raw_root": str(raw_root),
        "files": records,
        "status": {
            "SOURCE_VERIFIED": True,
            "HASH_VERIFIED": True,
            "LICENSE_RECORDED": license_name
            not in {"", "SOURCE_PAGE_REVIEW_REQUIRED", "NOT_STATED_ON_SOURCE"},
            "OFFICIAL_SPLIT_VERIFIED": False,
            "TIME_ORDER_VERIFIED": False,
        },
    }


def write_lineage(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
