from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v2_runner import V2Experiment


if __name__ == "__main__":
    args = json.loads((ROOT / "results/run_args.json").read_text(encoding="utf-8"))
    experiment = V2Experiment(
        ROOT,
        shared_bundle=Path(args["shared"]),
        cpu_bundle=Path(args["cpu_baselines"]),
        gpu_bundle=Path(args["gpu_baselines"]),
        v1_bundle=Path(args["v1_results"]),
        protocol_path=ROOT / args["protocol"],
        n_jobs=args["n_jobs"],
        bootstrap_jobs=args["bootstrap_jobs"],
    )
    payload = experiment.stage_e9()
    experiment._mark("E9", payload)
    print(f"E9_REGENERATED={payload['registration']}")
