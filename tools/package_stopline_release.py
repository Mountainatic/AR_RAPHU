#!/usr/bin/env python3
"""Create one deterministic ZIP containing source, every result, and reports."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "generated" / "AR_RAPHU_STOPLINE_20260725"
OUTPUT = PROJECT_ROOT / "dist" / "AR_RAPHU_STOPLINE_20260725.zip"
PRIVATE_BASENAMES = {"实验数据1.xlsx"}
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "logs",
    "dist",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allowed(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    return (
        path.is_file()
        and path.name not in PRIVATE_BASENAMES
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and not path.name.endswith(".part")
    )


def tracked_files() -> Iterable[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    for value in completed.stdout.decode("utf-8").split("\0"):
        if not value:
            continue
        path = PROJECT_ROOT / value
        if allowed(path) and not value.startswith(("results/", "generated/")):
            yield path


def result_files() -> Iterable[Path]:
    root = PROJECT_ROOT / "results"
    if not root.is_dir():
        raise RuntimeError("results/ is missing.")
    yield from (path for path in root.rglob("*") if allowed(path))


def report_files() -> Iterable[Path]:
    required = {
        "REPORT.md",
        "report.html",
        "artifact.json",
        "evidence_summary.json",
        "scenario_evidence.csv",
        "SHA256SUMS.txt",
    }
    missing = [name for name in required if not (REPORT_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing report artifacts: {missing}")
    yield from (path for path in REPORT_ROOT.rglob("*") if allowed(path))


def archive_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT)
    if relative.parts[0] == "results":
        return relative.as_posix()
    if relative.parts[0] == "generated":
        return "report/" + path.relative_to(REPORT_ROOT).as_posix()
    return "source/" + relative.as_posix()


def main() -> int:
    files = sorted(
        {*tracked_files(), *result_files(), *report_files()},
        key=lambda path: archive_name(path),
    )
    private_matches = [
        str(path)
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and path.name in PRIVATE_BASENAMES
    ]
    if private_matches:
        raise RuntimeError(
            "Private CZ workbook is present in the packaging workspace; "
            f"refusing to package: {private_matches}"
        )
    inventory = [
        {
            "path": archive_name(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "title": "AR-RAPHU v2 stop-line release package",
        "git_commit": git_sha,
        "file_count": len(inventory),
        "uncompressed_bytes": sum(row["bytes"] for row in inventory),
        "private_CZ_included": False,
        "contents": inventory,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=OUTPUT.parent, suffix=".zip.part", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            archive.writestr(
                "PACKAGE_MANIFEST.json",
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
            for path in files:
                archive.write(path, archive_name(path))
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256(OUTPUT)
    sidecar = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {OUTPUT.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "output": str(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
                "sha256": digest,
                "file_count": len(inventory),
                "private_CZ_included": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
