#!/usr/bin/env python3
"""Build and verify the frozen Spectral v0.3.3 return package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / "return_v033"
ARCHIVE = ROOT / "SPECTRAL_PS_AR_RAPHU_V033_RESOLUTION_CAPACITY_RESULTS.zip"
HASH = ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256")


def copy_file(relative: str) -> None:
    source = ROOT / relative
    destination = RETURN / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    if ROOT != Path("/root/AR_RAPHU_SPECTRAL_V033"):
        raise RuntimeError(f"Unexpected v0.3.3 worktree: {ROOT}")
    if RETURN.exists():
        shutil.rmtree(RETURN)
    for path in (ARCHIVE, HASH):
        if path.exists():
            path.unlink()
    RETURN.mkdir()

    for relative in (
        "PROTOCOL_REVISION_V033.md",
        "configs/spectral_v033.yaml",
        "src/ar_raphu/synthetic.py",
        "src/ar_raphu/protocol_config.py",
        "tools/run_spectral_suite.py",
        "tools/summarize_spectral_suite.py",
    ):
        copy_file(relative)
    optional = ROOT / "tools" / "run_spectral_job.py"
    if optional.exists():
        copy_file("tools/run_spectral_job.py")
    shutil.copytree(
        ROOT / "src" / "ar_raphu" / "spectral",
        RETURN / "src" / "ar_raphu" / "spectral",
    )
    test_names = {
        path.name for path in (ROOT / "tests").glob("test_spectral_*.py")
    }
    test_names.update(
        {
            "test_amplitude_domain.py",
            "test_scenario_registry.py",
            "test_s4u_generator.py",
            "test_operator_closure.py",
            "test_excitation_design.py",
            "test_weighted_projection.py",
            "test_error_decomposition.py",
            "test_truth_spectrum.py",
            "test_resolution_roles.py",
            "test_capacity_matrix.py",
        }
    )
    for name in sorted(test_names):
        copy_file(f"tests/{name}")
    shutil.copytree(
        ROOT / "results" / "spectral_v033",
        RETURN / "results" / "spectral_v033",
    )

    records = []
    for path in sorted(item for item in RETURN.rglob("*") if item.is_file()):
        data = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(RETURN).as_posix(),
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (RETURN / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(
            {"schema_version": 1, "file_count": len(records), "files": records},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(
        ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(item for item in RETURN.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(RETURN).as_posix())
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    HASH.write_text(f"{digest}  {ARCHIVE.name}\n", encoding="ascii")

    required = {
        "PACKAGE_MANIFEST.json",
        "src/ar_raphu/synthetic.py",
        "results/spectral_v033/V033_RESOLUTION_CAPACITY_DECISION.md",
        "results/spectral_v033/spectral_v033_resolution_capacity_summary.csv",
    }
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        missing = required - names
        if missing:
            raise RuntimeError(f"Package is missing: {sorted(missing)}")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt package member: {corrupt}")
    print(f"FINAL_PACKAGE={ARCHIVE}")
    print(f"FINAL_SHA256={digest}")
    print(f"SIZE_BYTES={ARCHIVE.stat().st_size}")
    print("NEXT_ALLOWED_STAGE=STOP_STRUCTURAL_SPACE_CAPACITY")


if __name__ == "__main__":
    main()
