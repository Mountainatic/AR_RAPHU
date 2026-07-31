from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io_data import sha256_file
from src.packaging import build_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--results")
    parser.add_argument("--output")
    args = parser.parse_args()
    source = Path(args.source_root).resolve()
    run_args = json.loads((source / "results/run_args.json").read_text())
    result = build_bundle(source, protocol_sha256=sha256_file(source / run_args["protocol"]), shared_sha256=sha256_file(Path(run_args["shared"])), cpu_sha256=sha256_file(Path(run_args["cpu_baselines"])), gpu_sha256=sha256_file(Path(run_args["gpu_baselines"])), v1_sha256=sha256_file(Path(run_args["v1_results"])))
    print(json.dumps(result, indent=2))
