from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect commit-safe C1 metadata without raw data or parquet")
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    files = [
        "PROTOCOL.json",
        "DATASET_HASHES.json",
        "TASK_REGISTRY.json",
        "SPLIT_REGISTRY.json",
        "SAMPLE_ID_REGISTRY.json",
        "LOCKBOX.json",
        "VALIDATION_REPORT.md",
        "C1_VALIDATION.json",
        "C1_SAMPLE_COUNTS.csv",
    ]
    for relative in files:
        shutil.copy2(args.shared / relative, args.output / relative)
    for directory in ("dataset_views", "sequence_views", "multiresolution_tabular_views", "graph_views", "masks", "scaler_metadata"):
        shutil.copytree(args.shared / directory, args.output / directory)
    total_bytes = sum(path.stat().st_size for path in args.shared.rglob("*") if path.is_file())
    pointer = {
        "shared_path_on_build_server": str(args.shared.resolve()),
        "shared_total_bytes": total_bytes,
        "shared_file_count": sum(1 for path in args.shared.rglob("*") if path.is_file()),
        "sample_registry_sha256": sha256(args.shared / "SAMPLE_ID_REGISTRY.json"),
        "parquet_in_git_artifact": False,
        "raw_data_in_git_artifact": False,
    }
    (args.output / "SHARED_PACKAGE_POINTER.json").write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "METADATA_MANIFEST.json":
            manifest.append({"path": path.relative_to(args.output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (args.output / "METADATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(manifest), "status": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
