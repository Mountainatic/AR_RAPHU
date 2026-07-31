#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.runner import Experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path)
    parser.add_argument("--cpu-baselines", type=Path)
    parser.add_argument("--gpu-baselines", type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "configs/frozen_l6.yaml")
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--bootstrap-jobs", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=[f"E{i}" for i in range(9)])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = Experiment(
        root=HERE,
        shared_bundle=args.shared,
        cpu_bundle=args.cpu_baselines,
        gpu_bundle=args.gpu_baselines,
        config_path=args.config,
        n_jobs=args.n_jobs,
        bootstrap_jobs=args.bootstrap_jobs,
        resume=args.resume,
    )
    return experiment.run(stop_after=args.stop_after)


if __name__ == "__main__":
    raise SystemExit(main())
