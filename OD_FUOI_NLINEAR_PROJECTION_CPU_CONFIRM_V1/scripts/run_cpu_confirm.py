from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runner import Experiment


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--shared")
    value.add_argument("--cpu-baselines")
    value.add_argument("--gpu-baselines")
    value.add_argument("--protocol", default="configs/frozen_protocol.yaml")
    value.add_argument("--n-jobs", type=int, default=12)
    value.add_argument("--bootstrap-jobs", type=int, default=16)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--preflight-only", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    invocation = ROOT / "results/run_args.json"
    if args.resume:
        stored = json.loads(invocation.read_text(encoding="utf-8"))
        for key, value in stored.items():
            if getattr(args, key, None) in (None, False):
                setattr(args, key, value)
    required = (args.shared, args.cpu_baselines, args.gpu_baselines)
    if any(value is None for value in required):
        raise SystemExit("shared, cpu-baselines and gpu-baselines are required on first run")
    invocation.parent.mkdir(parents=True, exist_ok=True)
    invocation.write_text(json.dumps({
        "shared": args.shared, "cpu_baselines": args.cpu_baselines,
        "gpu_baselines": args.gpu_baselines, "protocol": args.protocol,
        "n_jobs": args.n_jobs, "bootstrap_jobs": args.bootstrap_jobs,
    }, indent=2), encoding="utf-8")
    experiment = Experiment(
        ROOT,
        shared_bundle=Path(args.shared), cpu_bundle=Path(args.cpu_baselines),
        gpu_bundle=Path(args.gpu_baselines), protocol_path=(ROOT / args.protocol if not Path(args.protocol).is_absolute() else Path(args.protocol)),
        n_jobs=args.n_jobs, bootstrap_jobs=args.bootstrap_jobs,
    )
    if args.preflight_only:
        payload = experiment.stage_e0()
        experiment._mark("E0", payload)
        print("PRECHECK_STATUS=PASS")
        return 0
    return experiment.run()


if __name__ == "__main__":
    raise SystemExit(main())
