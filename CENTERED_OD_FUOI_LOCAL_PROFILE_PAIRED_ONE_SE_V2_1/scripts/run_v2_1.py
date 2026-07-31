from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v2_1_runner import V21Experiment


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--shared")
    value.add_argument("--cpu-baselines")
    value.add_argument("--gpu-baselines")
    value.add_argument("--v1-results")
    value.add_argument("--v2-results")
    value.add_argument("--protocol", default="configs/frozen_protocol_v2_1.yaml")
    value.add_argument("--profile-workers", type=int, default=8)
    value.add_argument("--bootstrap-workers", type=int, default=16)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--preflight-only", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    invocation = ROOT / "results_v2_1/run_args.json"
    if args.resume:
        stored = json.loads(invocation.read_text(encoding="utf-8"))
        for key, current in vars(args).items():
            if current in (None, False) and key in stored:
                setattr(args, key, stored[key])
    required = (args.shared, args.cpu_baselines, args.gpu_baselines, args.v1_results, args.v2_results)
    if any(value is None for value in required):
        raise SystemExit("shared, cpu-baselines, gpu-baselines, v1-results and v2-results are required")
    invocation.parent.mkdir(parents=True, exist_ok=True)
    invocation.write_text(json.dumps({
        "shared": args.shared, "cpu_baselines": args.cpu_baselines, "gpu_baselines": args.gpu_baselines,
        "v1_results": args.v1_results, "v2_results": args.v2_results, "protocol": args.protocol,
        "profile_workers": args.profile_workers, "bootstrap_workers": args.bootstrap_workers,
    }, indent=2), encoding="utf-8")
    experiment = V21Experiment(
        ROOT,
        shared_bundle=Path(args.shared), cpu_bundle=Path(args.cpu_baselines), gpu_bundle=Path(args.gpu_baselines),
        v1_bundle=Path(args.v1_results), v2_bundle=Path(args.v2_results),
        v2_protocol_path=ROOT / args.protocol,
        n_jobs=args.profile_workers, bootstrap_jobs=args.bootstrap_workers,
    )
    if args.preflight_only:
        payload = experiment.stage_e0(); experiment._mark("E0", payload); print("PRECHECK_STATUS=PASS"); return 0
    return experiment.run()


if __name__ == "__main__":
    raise SystemExit(main())
