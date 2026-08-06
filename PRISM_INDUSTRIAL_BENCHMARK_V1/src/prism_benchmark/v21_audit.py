from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .stage0 import write_json


REGISTRY_FILES = (
    "TASK_REGISTRY.json",
    "PROTOCOL.json",
    "dataset_views/VIEW_REGISTRY.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[Path]:
    for relative in REGISTRY_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        yield path
    for directory in (root / "base_data", root / "sample_ids"):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        yield from sorted(path for path in directory.rglob("*.parquet") if path.is_file())


def data_base_audit(shared: Path) -> dict[str, Any]:
    records = []
    for path in sorted(_files(shared), key=lambda value: value.as_posix()):
        relative = path.relative_to(shared).as_posix()
        parts = Path(relative).parts
        dataset = "registry"
        if parts[0] == "base_data" and len(parts) > 1:
            dataset = parts[1]
        elif parts[0] == "sample_ids" and len(parts) > 1:
            dataset = "sru" if parts[1].startswith("SRU_") else "non_sru_or_unknown"
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "dataset_group": "sru" if dataset == "sru" else "non_sru",
            }
        )
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        group = record["dataset_group"]
        item = summary.setdefault(group, {"files": 0, "bytes": 0})
        item["files"] += 1
        item["bytes"] += int(record["bytes"])
    return {
        "status": "PASS",
        "shared_root": str(shared),
        "files": records,
        "summary": summary,
        "total_files": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
    }


def compare_data_base_audits(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def index(value: dict[str, Any]) -> dict[str, tuple[int, str]]:
        return {
            str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
            for item in value.get("files", ())
        }

    left = index(before)
    right = index(after)
    added = sorted(right.keys() - left.keys())
    removed = sorted(left.keys() - right.keys())
    changed = sorted(path for path in left.keys() & right.keys() if left[path] != right[path])
    passed = not added and not removed and not changed
    return {
        "status": "PASS" if passed else "STOP_DATA_BASE_MUTATED",
        "identical": passed,
        "added": added,
        "removed": removed,
        "changed": changed,
        "pre_total_files": len(left),
        "post_total_files": len(right),
    }


def write_pre_audit(shared: Path, output: Path) -> dict[str, Any]:
    audit = data_base_audit(shared)
    path = output / "DATA_AUDIT" / "V21_DATA_BASE_PRE_AUDIT.json"
    write_json(path, audit)
    return audit


def write_post_audit(shared: Path, output: Path) -> dict[str, Any]:
    pre_path = output / "DATA_AUDIT" / "V21_DATA_BASE_PRE_AUDIT.json"
    if not pre_path.is_file():
        raise FileNotFoundError(pre_path)
    before = json.loads(pre_path.read_text(encoding="utf-8"))
    after = data_base_audit(shared)
    comparison = compare_data_base_audits(before, after)
    after["comparison_to_pre"] = comparison
    write_json(output / "DATA_AUDIT" / "V21_DATA_BASE_POST_AUDIT.json", after)
    if comparison["status"] != "PASS":
        raise RuntimeError("STOP_DATA_BASE_MUTATED")
    return after
