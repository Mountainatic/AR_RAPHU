#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.packaging import build_cpu_package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--results", default=str(ROOT / "results_cpu"))
    parser.add_argument("--shared", default=str(ROOT / "shared"))
    parser.add_argument(
        "--output-dir", default=str(ROOT / "return" / "PHYSICS_FIRST_CPU_RESULTS")
    )
    args = parser.parse_args()
    result = build_cpu_package(
        project_root=Path(args.source_root),
        results_root=Path(args.results),
        shared_root=Path(args.shared),
        output_root=Path(args.output_dir),
    )
    print(f"FINAL_CPU_ZIP={result['archive']}")
    print(f"FINAL_CPU_SHA256={result['sha256']}")
    print(f"ZIP_SIZE={result['size']}")
    print(f"MANIFEST_FILE_COUNT={result['manifest_file_count']}")
    print(f"VALIDATION_STATUS={result['status']}")
    shared_zip = ROOT / "return" / "SHARED_BENCHMARK_DATASET_bundle.zip"
    from src.common import sha256_file
    print(f"SHARED_DATASET_ZIP={shared_zip.resolve()}")
    print(f"SHARED_DATASET_SHA256={sha256_file(shared_zip)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
