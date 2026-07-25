#!/usr/bin/env python3
"""Create the validation-only, resumable M6 candidate manifest."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.protocol_config import load_protocol_config
from ar_raphu.runtime_environment import require_runtime_environment


def make_manifest() -> Path:
    config = load_protocol_config(require_phase1_frozen=True)
    weights = config["phase1_model_selection"]["M6"][
        "second_difference_smoothness_weights"
    ]
    runner = PROJECT_ROOT / "tools" / "run_phase1_m6.py"
    jobs = []
    for seed in config["training"]["seeds"]["screening"]:
        for smoothness in weights:
            jobs.append(
                {
                    "job_id": (
                        f"m6_seed{int(seed)}_lambda{float(smoothness):.8g}"
                    ),
                    "command": [
                        sys.executable,
                        str(runner),
                        "fit",
                        "--seed",
                        str(seed),
                        "--smoothness",
                        str(smoothness),
                        "--device",
                        "cuda",
                    ],
                }
            )
    output = (
        PROJECT_ROOT
        / "results"
        / "phase1"
        / "manifests"
        / "AR-S2_G2_M6_candidates.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenario": "AR-S2",
                "generator_version": 2,
                "model": "M6",
                "stage": "validation_candidates",
                "device": "cuda",
                "job_count": len(jobs),
                "workers_per_device": 16,
                "cpu_threads_per_worker": 1,
                "test_access": False,
                "jobs": jobs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def main() -> int:
    require_runtime_environment()
    print(make_manifest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
