#!/usr/bin/env python3
"""Create the validation-only, resumable M7 candidate manifest."""

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
    m7 = config["phase1_model_selection"]["M7"]
    runner = PROJECT_ROOT / "tools" / "run_phase1_m7.py"
    jobs = []
    for seed in config["training"]["seeds"]["screening"]:
        for grid in m7["amplitude_spline_grid_sizes"]:
            for smoothness in m7["smoothness_weights"]:
                jobs.append(
                    {
                        "job_id": (
                            f"m7_seed{int(seed)}_gx{int(grid)}_"
                            f"lambda{float(smoothness):.8g}"
                        ),
                        "command": [
                            sys.executable,
                            str(runner),
                            "fit",
                            "--seed",
                            str(seed),
                            "--grid-size",
                            str(grid),
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
        / "AR-S1_G2_M7_candidates.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenario": "AR-S1",
                "generator_version": 2,
                "model": "M7",
                "stage": "validation_candidates",
                "device": "cuda",
                "job_count": len(jobs),
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
