#!/usr/bin/env python3
"""Create resumable task-level GPU manifests for Scheme-A Phase-1 jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ar_raphu.protocol_config import load_protocol_config
from ar_raphu.runtime_environment import require_runtime_environment
from ar_raphu.synthetic import SCENARIOS


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def make_manifest(
    scenario: str,
    stage: str,
    *,
    device: str,
    track: str = "XAR",
    replicate_profile: str = "screening",
) -> Path:
    config = load_protocol_config(require_phase1_frozen=True)
    seeds = config["training"]["seeds"][replicate_profile]
    scales = config["training"]["source_backed_v20_reference"]["pruning_scales"]
    runner = PROJECT_ROOT / "tools" / "run_phase1_scheme_a.py"
    jobs: list[dict] = []
    if track not in {"X", "XAR"} and stage in {"warmup", "fork"}:
        raise ValueError("Sparse stages require Track-X or Track-XAR.")
    if stage == "warmup":
        for seed in seeds:
            jobs.append(
                {
                    "job_id": f"{scenario}_{track.lower()}_warmup_seed{seed}",
                    "command": [
                        sys.executable,
                        str(runner),
                        "warmup",
                        "--scenario",
                        scenario,
                        "--track",
                        track,
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                    ],
                }
            )
    elif stage == "fork":
        for seed in seeds:
            for scale in scales:
                jobs.append(
                    {
                        "job_id": (
                            f"{scenario}_{track.lower()}_fork_seed{seed}_"
                            f"s{float(scale):.6f}"
                        ),
                        "command": [
                            sys.executable,
                            str(runner),
                            "fork",
                            "--scenario",
                            scenario,
                            "--track",
                            track,
                            "--seed",
                            str(seed),
                            "--scale",
                            str(scale),
                            "--device",
                            device,
                        ],
                    }
                )
    elif stage == "dense_ar":
        for seed in seeds:
            jobs.append(
                {
                    "job_id": f"{scenario}_ar_dense_seed{seed}",
                    "command": [
                        sys.executable,
                        str(runner),
                        "dense",
                        "--scenario",
                        scenario,
                        "--track",
                        "AR",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                    ],
                }
            )
    else:
        raise ValueError(stage)
    manifest_track = "AR" if stage == "dense_ar" else track
    profile_suffix = (
        "" if replicate_profile == "screening" else f"_{replicate_profile}"
    )
    output = (
        PROJECT_ROOT
        / "results"
        / "phase1"
        / "manifests"
        / f"{scenario}_G2_{manifest_track}_{stage}{profile_suffix}.json"
    )
    atomic_json(
        output,
        {
            "schema_version": 1,
            "scenario": scenario,
            "generator_version": 2,
            "stage": stage,
            "track": manifest_track,
            "replicate_profile": replicate_profile,
            "device": device,
            "job_count": len(jobs),
            "test_access": stage == "dense_ar",
            "jobs": jobs,
        },
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument(
        "--stage", choices=["warmup", "fork", "dense_ar"], required=True
    )
    parser.add_argument("--track", choices=["X", "XAR"], default="XAR")
    parser.add_argument(
        "--replicate-profile",
        choices=["screening", "critical"],
        default="screening",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    require_runtime_environment()
    args = parse_args()
    print(
        make_manifest(
            args.scenario,
            args.stage,
            device=args.device,
            track=args.track,
            replicate_profile=args.replicate_profile,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
