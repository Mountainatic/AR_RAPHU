from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable


FORBIDDEN_PARTS = {".git", "raw_sources", "base_data", "sample_ids", "targets", "purge_masks"}
FORBIDDEN_SUFFIXES = {".xlsx", ".xls", ".rds", ".rdata", ".mat"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path, include: Callable[[Path], bool] | None = None) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or any(part in FORBIDDEN_PARTS for part in path.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        if include is not None and not include(path):
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def build(project: Path, results: Path, shared: Path, output_root: Path) -> dict[str, Any]:
    package = output_root / "PRISM_INDUSTRIAL_CPU_RESULTS_V1"
    if package.exists():
        shutil.rmtree(package)
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_tree(project, package / "SOURCE", lambda path: "__pycache__" not in path.parts and path.suffix != ".pyc")
    _copy_tree(results, package / "RESULTS")
    metadata = package / "SHARED_METADATA"
    metadata.mkdir(parents=True, exist_ok=True)
    for name in ("TASK_REGISTRY.json", "PROTOCOL.json", "C1_VALIDATION.json", "PACKAGE_MANIFEST.json"):
        path = shared / name
        if path.is_file():
            shutil.copy2(path, metadata / name)
    registry = shared / "dataset_views/VIEW_REGISTRY.json"
    if registry.is_file():
        (metadata / "dataset_views").mkdir(parents=True, exist_ok=True)
        shutil.copy2(registry, metadata / "dataset_views/VIEW_REGISTRY.json")
    records = []
    for path in sorted(package.rglob("*")):
        if path.is_file():
            records.append({"path": str(path.relative_to(package)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"package": "PRISM_INDUSTRIAL_CPU_RESULTS_V1", "file_count": len(records), "files": records, "raw_data_included": False}
    (package / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (package / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{record['sha256']}  {record['path']}\n")
    zip_path = output_root / "PRISM_INDUSTRIAL_CPU_RESULTS_V1_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, Path(package.name) / path.relative_to(package))
    with tempfile.TemporaryDirectory(prefix="prism_cpu_roundtrip_") as directory:
        with zipfile.ZipFile(zip_path) as archive:
            archive.testzip()
            archive.extractall(directory)
        extracted = Path(directory) / package.name
        extracted_manifest = json.loads((extracted / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
        for record in extracted_manifest["files"]:
            path = extracted / record["path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"round-trip validation failed: {record['path']}")
    digest = sha256(zip_path)
    sha_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    result = {
        "status": "PASS",
        "zip": str(zip_path.resolve()),
        "sha256": digest,
        "sha256_file": str(sha_path.resolve()),
        "zip_bytes": zip_path.stat().st_size,
        "manifest_file_count": len(records),
        "raw_data_included": False,
        "roundtrip": "PASS",
    }
    (output_root / "PACKAGE_OUTPUT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"FINAL_CPU_ZIP={result['zip']}")
    print(f"FINAL_CPU_SHA256={digest}")
    print(f"ZIP_SIZE={result['zip_bytes']}")
    print(f"MANIFEST_FILE_COUNT={len(records)}")
    print("VALIDATION_STATUS=PASS")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.project, arguments.results, arguments.shared, arguments.output_root)


if __name__ == "__main__":
    main()
