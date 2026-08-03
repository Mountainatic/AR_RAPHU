from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


MAX_PART_BYTES = 1_500_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def upload(gh: str, repository: str, tag: str, path: Path, cwd: Path) -> dict[str, Any]:
    run([gh, "release", "upload", tag, str(path), "--repo", repository, "--clobber"], cwd)
    return {"asset": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def prediction_records(c6: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prediction in sorted((c6 / "PREDICTIONS").rglob("*.parquet")):
        relative = prediction.relative_to(c6)
        audit = c6 / "MODEL_AUDIT" / relative.relative_to("PREDICTIONS").with_suffix(".json")
        if not audit.is_file():
            raise FileNotFoundError(f"prediction audit missing: {audit}")
        metadata = json.loads(audit.read_text(encoding="utf-8"))
        if metadata.get("status") != "PASS" or int(metadata.get("rows", -1)) < 1:
            raise RuntimeError(f"prediction audit did not pass: {audit}")
        records.append(
            {
                "path": str(relative),
                "bytes": prediction.stat().st_size,
                "sha256": metadata["prediction_sha256"],
                "rows": int(metadata["rows"]),
                "model": metadata["model"],
                "split": metadata["split"],
                "target_head": metadata["target_head"],
            }
        )
    return records


def partition(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parts: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for record in records:
        size = int(record["bytes"])
        if size > MAX_PART_BYTES:
            raise RuntimeError(f"single prediction exceeds release part limit: {record['path']}")
        if current and current_bytes + size > MAX_PART_BYTES:
            parts.append(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += size
    if current:
        parts.append(current)
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish complete C6 V2 code, reports, audits and predictions.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--c6", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--repository", default="Mountainatic/AR_RAPHU")
    parser.add_argument("--title", default="PRISM Industrial CPU Full Benchmark C6 V2")
    parser.add_argument("--work", type=Path, required=True)
    arguments = parser.parse_args()

    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError("gh is required")
    decision = json.loads((arguments.c6 / "CPU_FULL_FINAL_DECISION.json").read_text(encoding="utf-8"))
    if decision.get("status") not in {"PASS", "PASS_WITH_RETAINED_FAILURES"}:
        raise RuntimeError("C6 final decision is not publishable")
    if int(decision.get("successful_prediction_files", -1)) != 183:
        raise RuntimeError("expected 183 successful frozen prediction files")

    arguments.work.mkdir(parents=True, exist_ok=True)
    existing = subprocess.run(
        [gh, "release", "view", arguments.release_tag, "--repo", arguments.repository],
        cwd=arguments.repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if existing.returncode:
        run(
            [
                gh,
                "release",
                "create",
                arguments.release_tag,
                "--repo",
                arguments.repository,
                "--target",
                "prism-industrial-benchmark-v1-strict-restart",
                "--title",
                arguments.title,
                "--notes-file",
                str(arguments.c6 / "CPU_FULL_FINAL_REPORT.md"),
                "--latest=false",
            ],
            arguments.repo_root,
        )

    uploaded: list[dict[str, Any]] = []
    source = arguments.work / "PRISM_INDUSTRIAL_CPU_C6_V2_SOURCE.tar.gz"
    with source.open("wb") as handle:
        subprocess.run(
            ["git", "archive", "--format=tar.gz", "--prefix=PRISM_INDUSTRIAL_CPU_C6_V2_SOURCE/", "HEAD", "PRISM_INDUSTRIAL_BENCHMARK_V1"],
            cwd=arguments.repo_root,
            check=True,
            stdout=handle,
        )
    uploaded.append(upload(gh, arguments.repository, arguments.release_tag, source, arguments.repo_root))
    source.unlink()

    summary = arguments.work / "PRISM_INDUSTRIAL_CPU_C6_V2_SUMMARY_AND_AUDIT.tar"
    with tarfile.open(summary, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(arguments.c6.rglob("*")):
            if not path.is_file() or "PREDICTIONS" in path.relative_to(arguments.c6).parts:
                continue
            archive.add(path, arcname=str(Path("C6_FULL_FINAL_V2") / path.relative_to(arguments.c6)), recursive=False)
        for stage in ("C2_CLASSICAL", "C3_SYSTEM_IDENTIFICATION", "C4_PRISM", "C5_PRISM_DYNAMIC"):
            root = arguments.results_root / stage
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or "PREDICTIONS" in path.relative_to(root).parts:
                    continue
                archive.add(path, arcname=str(Path(stage) / path.relative_to(root)), recursive=False)
    uploaded.append(upload(gh, arguments.repository, arguments.release_tag, summary, arguments.repo_root))
    summary.unlink()

    records = prediction_records(arguments.c6)
    parts = partition(records)
    for index, members in enumerate(parts):
        part = arguments.work / f"PRISM_INDUSTRIAL_CPU_C6_V2_PREDICTIONS_{index:03d}.tar"
        payload = {
            "release_tag": arguments.release_tag,
            "part": index,
            "part_count": len(parts),
            "file_count": len(members),
            "files": members,
        }
        with tarfile.open(part, "w", format=tarfile.PAX_FORMAT) as archive:
            encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo(name=f"PART_MANIFEST_{index:03d}.json")
            info.size = len(encoded)
            import io

            archive.addfile(info, io.BytesIO(encoded))
            for record in members:
                path = arguments.c6 / record["path"]
                archive.add(path, arcname=str(Path("C6_FULL_FINAL_V2") / record["path"]), recursive=False)
        asset = upload(gh, arguments.repository, arguments.release_tag, part, arguments.repo_root)
        asset.update({"part": index, "file_count": len(members), "unpacked_bytes": sum(int(row["bytes"]) for row in members)})
        uploaded.append(asset)
        part.unlink()

    manifest = {
        "status": "PASS",
        "release_tag": arguments.release_tag,
        "repository": arguments.repository,
        "prediction_file_count": len(records),
        "prediction_parts": len(parts),
        "predictions": records,
        "assets": uploaded,
        "raw_source_data_included": False,
        "extraction": "Download all *_PREDICTIONS_*.tar assets and extract each into the same destination; paths are disjoint.",
    }
    manifest_path = arguments.work / "FULL_RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upload(gh, arguments.repository, arguments.release_tag, manifest_path, arguments.repo_root)
    report = arguments.c6 / "CPU_FULL_FINAL_REPORT.md"
    upload(gh, arguments.repository, arguments.release_tag, report, arguments.repo_root)
    url = subprocess.check_output(
        [gh, "release", "view", arguments.release_tag, "--repo", arguments.repository, "--json", "url", "--jq", ".url"],
        cwd=arguments.repo_root,
        text=True,
    ).strip()
    print(f"RELEASE_URL={url}")
    print(f"PREDICTION_FILES={len(records)}")
    print(f"PREDICTION_PARTS={len(parts)}")
    print("VALIDATION_STATUS=PASS")


if __name__ == "__main__":
    main()
