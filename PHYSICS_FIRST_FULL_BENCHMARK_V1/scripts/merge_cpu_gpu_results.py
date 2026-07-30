#!/usr/bin/env python3
"""Hash-gated CPU/GPU bundle merger; does not reinterpret either result set."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def locate(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"EXPECTED_ONE_{name}:{len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-bundle", required=True)
    parser.add_argument("--gpu-bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cpu_gpu_merge_") as temp:
        temp_root = Path(temp)
        cpu_root, gpu_root = temp_root / "cpu", temp_root / "gpu"
        with zipfile.ZipFile(args.cpu_bundle) as bundle:
            bundle.extractall(cpu_root)
        with zipfile.ZipFile(args.gpu_bundle) as bundle:
            bundle.extractall(gpu_root)
        cpu_hashes = json.loads(
            locate(cpu_root, "DATA_AND_SPLIT_HASHES.json").read_text(encoding="utf-8")
        )
        gpu_hashes = json.loads(
            locate(gpu_root, "DATA_AND_SPLIT_HASHES.json").read_text(encoding="utf-8")
        )
        for key in ("data_sha256", "config_sha256", "directions"):
            if cpu_hashes[key] != gpu_hashes[key]:
                raise RuntimeError(f"CPU_GPU_HASH_MISMATCH:{key}")
        shutil.copytree(cpu_root, output / "cpu", dirs_exist_ok=True)
        shutil.copytree(gpu_root, output / "gpu", dirs_exist_ok=True)
        (output / "MERGE_VALIDATION.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "data_sha256": cpu_hashes["data_sha256"],
                    "config_sha256": cpu_hashes["config_sha256"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps({"status": "PASS", "output": str(output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
