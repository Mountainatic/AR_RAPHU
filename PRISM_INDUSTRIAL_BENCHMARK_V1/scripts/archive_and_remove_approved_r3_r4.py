from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APPROVED_NAMES = {
    "PRISM_V211_REPRESENTATIVE_STAGE1_TEP_SRU_CPU_20260823_R3",
    "PRISM_V211_REPRESENTATIVE_STAGE1_TEP_SRU_CPU_20260823_R4",
}
APPROVED_PARENT = Path("/root/autodl-tmp")
PRESERVE_SUFFIXES = {".json", ".csv", ".log", ".txt", ".md", ".yaml", ".yml"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_target(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.parent != APPROVED_PARENT or resolved.name not in APPROVED_NAMES:
        raise RuntimeError(f"STOP_UNAPPROVED_DELETE_TARGET:{resolved}")
    if resolved.is_symlink() or not resolved.is_dir():
        raise RuntimeError(f"STOP_DELETE_TARGET_NOT_REAL_DIRECTORY:{resolved}")
    return resolved


def _old_recovery_running() -> bool:
    completed = subprocess.run(
        ["pgrep", "-f", "parallel_final_recovery.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in sorted(names):
            path = parent / name
            if path.is_symlink():
                target = os.readlink(path)
                records.append(
                    {
                        "relative_path": path.relative_to(root).as_posix(),
                        "kind": "symlink",
                        "size": int(path.lstat().st_size),
                        "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                        "symlink_target": target,
                    }
                )
        for name in sorted(files):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                target = os.readlink(path)
                records.append(
                    {
                        "relative_path": relative,
                        "kind": "symlink",
                        "size": int(path.lstat().st_size),
                        "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                        "symlink_target": target,
                    }
                )
            else:
                records.append(
                    {
                        "relative_path": relative,
                        "kind": "file",
                        "size": int(path.stat().st_size),
                        "sha256": _sha256(path),
                    }
                )
    return records


def _preserve(relative: str) -> bool:
    path = Path(relative)
    lower = relative.lower()
    return (
        path.suffix.lower() in PRESERVE_SUFFIXES
        or "logs" in {part.lower() for part in path.parts}
        or "acceptance" in lower
        or "audit" in lower
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit, archive, verify and remove only the user-approved R3/R4 directories."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=APPROVED_PARENT / "approved_r3_r4_evidence_20260824.zip",
    )
    parser.add_argument(
        "--allow-concurrent-old-recovery",
        action="store_true",
        help=(
            "Allow the explicitly authorized R3/R4 cleanup while the unrelated "
            "legacy CZ/Neural3 recovery is still running. The default remains fail-closed."
        ),
    )
    parser.add_argument("targets", type=Path, nargs="*")
    args = parser.parse_args()
    targets = args.targets or [APPROVED_PARENT / name for name in sorted(APPROVED_NAMES)]
    resolved = [_validate_target(path) for path in targets]
    if {path.name for path in resolved} != APPROVED_NAMES or len(resolved) != 2:
        raise RuntimeError("STOP_DELETE_SCOPE_MUST_BE_EXACT_APPROVED_R3_AND_R4")
    old_recovery_running = _old_recovery_running()
    if old_recovery_running and not args.allow_concurrent_old_recovery:
        raise RuntimeError("STOP_OLD_CZ_NEURAL3_RECOVERY_STILL_RUNNING")
    archive = args.archive.resolve()
    if archive.exists():
        raise RuntimeError(f"refusing to overwrite evidence archive: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    inventories = {root.name: _inventory(root) for root in resolved}
    manifest = {
        "status": "PRE_DELETE_AUDIT_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "approved_parent": str(APPROVED_PARENT),
        "targets": [str(path) for path in resolved],
        "files": inventories,
        "total_entries": sum(len(value) for value in inventories.values()),
        "total_regular_file_bytes": sum(
            int(item["size"])
            for value in inventories.values()
            for item in value
            if item["kind"] == "file"
        ),
        "preservation_rule": "all logs, JSON/CSV, acceptance/audit, TXT/MD/YAML",
        "old_recovery_running_at_cleanup": old_recovery_running,
        "concurrent_old_recovery_explicitly_authorized": bool(
            args.allow_concurrent_old_recovery
        ),
    }
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for root in resolved:
            for item in inventories[root.name]:
                if item["kind"] != "file" or not _preserve(str(item["relative_path"])):
                    continue
                source = root / str(item["relative_path"])
                bundle.write(source, f"{root.name}/{item['relative_path']}")
        bundle.writestr(
            "DELETE_AUDIT_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        )
    with zipfile.ZipFile(archive, "r") as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("STOP_EVIDENCE_ARCHIVE_CRC_FAILED")
        restored = json.loads(bundle.read("DELETE_AUDIT_MANIFEST.json"))
        if restored != manifest:
            raise RuntimeError("STOP_EVIDENCE_ARCHIVE_MANIFEST_MISMATCH")
        archived_names = set(bundle.namelist())
        for root in resolved:
            for item in inventories[root.name]:
                if item["kind"] == "file" and _preserve(str(item["relative_path"])):
                    expected = f"{root.name}/{item['relative_path']}"
                    if expected not in archived_names:
                        raise RuntimeError(f"STOP_EVIDENCE_FILE_NOT_ARCHIVED:{expected}")
    archive_sha256 = _sha256(archive)
    # The exact resolved targets have already been checked above.  No glob or
    # variable-expanded parent is ever passed to rmtree.
    for root in resolved:
        shutil.rmtree(root)
    remaining = [str(path) for path in resolved if path.exists()]
    if remaining:
        raise RuntimeError(f"STOP_APPROVED_DELETE_INCOMPLETE:{remaining}")
    free_gib = shutil.disk_usage(APPROVED_PARENT).free / (1024**3)
    result = {
        "status": "PASS" if free_gib >= 20.0 else "STOP_INSUFFICIENT_FREE_STORAGE_AFTER_APPROVED_CLEANUP",
        "archive": str(archive),
        "archive_sha256": archive_sha256,
        "deleted_exact_targets": [str(path) for path in resolved],
        "recoverable_from_archive": "evidence files only; large parquet/model-free development payloads were not archived",
        "free_gib_after_cleanup": free_gib,
        "minimum_required_gib": 20.0,
        "no_other_directory_deleted": True,
        "old_recovery_running_at_cleanup": old_recovery_running,
        "concurrent_old_recovery_explicitly_authorized": bool(
            args.allow_concurrent_old_recovery
        ),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
