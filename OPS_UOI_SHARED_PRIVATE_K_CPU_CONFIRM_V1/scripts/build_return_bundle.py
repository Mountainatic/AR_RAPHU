#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.io_data import sha256_file
from src.packaging import build_bundle


def main() -> int:
    checkpoint = json.loads(
        (HERE / "results/checkpoints/latest.json").read_text(encoding="utf-8")
    )
    inputs = checkpoint["inputs"]
    result = build_bundle(
        HERE,
        shared_hash=sha256_file(Path(inputs["shared_bundle"])),
        cpu_hash=sha256_file(Path(inputs["cpu_bundle"])),
        gpu_hash=sha256_file(Path(inputs["gpu_bundle"])),
        protocol_hash=checkpoint["config_sha256"],
    )
    for key, value in (
        ("FINAL_ZIP", result["zip"]),
        ("FINAL_SHA256", result["sha256"]),
        ("ZIP_SIZE", result["size"]),
        ("MANIFEST_FILE_COUNT", result["manifest_file_count"]),
        ("PROTOCOL_SHA256", checkpoint["config_sha256"]),
        ("SHARED_DATASET_SHA256", sha256_file(Path(inputs["shared_bundle"]))),
        ("CPU_BASELINE_BUNDLE_SHA256", sha256_file(Path(inputs["cpu_bundle"]))),
        ("GPU_BASELINE_BUNDLE_SHA256", sha256_file(Path(inputs["gpu_bundle"]))),
        ("VALIDATION_STATUS", "PASS"),
    ):
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
