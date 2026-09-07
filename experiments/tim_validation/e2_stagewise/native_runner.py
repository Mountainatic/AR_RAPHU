"""Run TIM E2 through the released native PRISM v2.1.1 stages."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from prism_benchmark.portable_checkpoints import INFERENCE_ONLY_ENV
from prism_benchmark.representative_prism_checkpoints import (
    fit_prism_checkpoint_for_view,
    predict_prism_checkpoint_for_view,
    verify_prism_checkpoint_reload,
)
from prism_benchmark.v211_a import EXACT_ZERO, run_a_view
from prism_benchmark.v211_c import BEST_ACTIVE_K, run_c_view
from prism_benchmark.v211_config import PUBLIC_ALL_PROTOCOL
from prism_benchmark.v211_k import load_active_channels, run_k_channel
from prism_benchmark.v211_public_all_config import PublicAllPaths
from prism_benchmark.v211_public_all_baselines import apply_common_requirements
from prism_benchmark.v211_public_all_closure import (
    _support_frame,
    view_support_requirements,
)
from prism_benchmark.v211_support import SUPPORT_CONTRACT, support_id_hash
from prism_benchmark.v211_w import IDENTITY, run_w_view

from .synthetic_c1 import (
    CHANNELS,
    HISTORY_BY_SCALE,
    REGIMES,
    build_development,
    materialize_test,
    view,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_paths(output: Path) -> dict[str, Path]:
    current = view("input_only")
    dynamic = view("dynamic")
    return {
        "C": output / "DEVELOPMENT" / "C" / current.head.head_id / current.proxy_policy / "RESULT.json",
        "W": output / "DEVELOPMENT" / "W" / current.head.head_id / current.proxy_policy / "RESULT.json",
        "A": output / "DEVELOPMENT" / "A" / dynamic.head.head_id / dynamic.availability_scenario / dynamic.proxy_policy / "RESULT.json",
    }


def _freeze_common_support(paths: PublicAllPaths) -> dict[str, Any]:
    """Freeze only support identities; test targets are never loaded here."""
    dynamic = view("dynamic")
    requirements = view_support_requirements(paths, dynamic)
    splits: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        source = paths.shared / "sample_ids" / dynamic.relative_root / f"{split}.parquet"
        if not source.is_file():
            continue
        frame = _support_frame(paths.shared, dynamic, split)
        common = apply_common_requirements(frame, requirements)
        splits[split] = {
            "rows": int(len(common)),
            "source_rows": int(len(frame)),
            "support_hash": support_id_hash(common),
            "support_contract": SUPPORT_CONTRACT,
        }
    result = {
        "status": "PASS",
        "stage": "TIM_E2_COMMON_SUPPORT_METADATA_ONLY_FREEZE",
        "support_contract": SUPPORT_CONTRACT,
        "views": [
            {
                "target_head": dynamic.head.head_id,
                "dataset": dynamic.head.dataset,
                "information_set": dynamic.information_set,
                "availability_scenario": dynamic.availability_scenario,
                "proxy_policy": dynamic.proxy_policy,
                "requirements": [item.to_json() for item in requirements],
                "splits": splits,
            }
        ],
        "test_y_read": False,
        "test_accessed": False,
    }
    _write_json(paths.leaderboard_support_path, result)
    return result


def _scale_class(history: int) -> str:
    return min(HISTORY_BY_SCALE, key=lambda name: (abs(history - HISTORY_BY_SCALE[name]), name))


def _structure_signature(paths: PublicAllPaths) -> dict[str, Any]:
    input_view = view("input_only")
    active = load_active_channels(paths.output, input_view)
    stages = {name: _read(path) for name, path in _stage_paths(paths.output).items()}
    c_family = str(stages["C"].get("selected_family"))
    w_family = str(stages["W"].get("w_contract", {}).get("family"))
    a_family = str(stages["A"].get("a_contract", {}).get("family"))
    c_pass = stages["C"].get("status") == "PASS"
    w_pass = stages["W"].get("status") == "PASS"
    a_pass = stages["A"].get("status") == "PASS"
    selected_channels = sorted(str(item["channel"]) for item in active)
    histories = {
        str(item["channel"]): int(item["selected_profile_history_steps"])
        for item in active
    }
    return {
        "implementation": "native_prism_v2.1.1",
        "protocol": PUBLIC_ALL_PROTOCOL,
        "admitted_channels": selected_channels,
        "channel_history": histories,
        "channel_scale_class": {
            channel: _scale_class(history) for channel, history in histories.items()
        },
        "K_admitted": bool(active),
        "C_admitted": c_pass and c_family not in {BEST_ACTIVE_K, "K_EXACT_ZERO"},
        "DeltaW_admitted": w_pass and w_family != IDENTITY,
        "A_admitted": a_pass and a_family != EXACT_ZERO,
        "selected_C_candidate": c_family,
        "selected_W_candidate": w_family,
        "selected_A_candidate": a_family,
        "stage_status": {name: value.get("status") for name, value in stages.items()},
        "test_accessed": False,
    }


def run_development(project: Path, run_root: Path) -> dict[str, Any]:
    paths = PublicAllPaths(project, run_root / "shared", run_root)
    paths.output.mkdir(parents=True, exist_ok=True)
    input_view = view("input_only")
    results = []
    for channel in CHANNELS:
        result = run_k_channel(
            paths.shared,
            paths.project,
            paths.output,
            input_view,
            channel,
            PUBLIC_ALL_PROTOCOL,
        )
        results.append(result)
    c_result = run_c_view(paths.shared, paths.project, paths.output, input_view, PUBLIC_ALL_PROTOCOL)
    w_result = run_w_view(paths.shared, paths.project, paths.output, input_view, PUBLIC_ALL_PROTOCOL)
    a_result = run_a_view(paths.shared, paths.project, paths.output, view("dynamic"), PUBLIC_ALL_PROTOCOL)
    statuses = {
        "K": [item.get("status") for item in results],
        "C": c_result.get("status"),
        "W": w_result.get("status"),
        "A": a_result.get("status"),
    }
    signature = _structure_signature(paths)
    _write_json(run_root / "STRUCTURE_SIGNATURE.json", signature)
    result = {
        "status": "PASS"
        if all(value == "PASS" for value in statuses["K"])
        and statuses["C"] == statuses["W"] == statuses["A"] == "PASS"
        else "COMPLETED_WITH_RETAINED_FAILURES",
        "implementation": "native_prism_v2.1.1",
        "stage_statuses": statuses,
        "structure_signature": signature,
        "test_accessed": False,
    }
    _write_json(run_root / "DEVELOPMENT_COMPLETE.json", result)
    return result


def freeze_and_fit(project: Path, run_root: Path) -> dict[str, Any]:
    paths = PublicAllPaths(project, run_root / "shared", run_root)
    development = _read(run_root / "DEVELOPMENT_COMPLETE.json")
    if development.get("status") not in {"PASS", "COMPLETED_WITH_RETAINED_FAILURES"}:
        raise RuntimeError("native development did not complete")
    if any(development["stage_statuses"].get(stage) != "PASS" for stage in ("C", "W", "A")):
        raise RuntimeError("a downstream native stage did not pass")
    signature = _read(run_root / "STRUCTURE_SIGNATURE.json")
    freeze = {
        "status": "SELECTION_FROZEN",
        "sealed": True,
        "structure_signature": signature,
        "test_accessed": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(run_root / "freeze" / "SELECTION_FREEZE.json", freeze)
    _freeze_common_support(paths)
    checkpoint = fit_prism_checkpoint_for_view(
        paths, view("dynamic"), run_root / "checkpoints"
    )
    seal = {
        "status": "CHECKPOINTS_SEALED",
        "sealed": True,
        "checkpoint": checkpoint,
        "test_accessed": False,
    }
    _write_json(run_root / "freeze" / "CHECKPOINTS_SEALED.json", seal)
    return seal


def run_inference(project: Path, run_root: Path) -> dict[str, Any]:
    if os.environ.get(INFERENCE_ONLY_ENV) != "1":
        raise RuntimeError(f"{INFERENCE_ONLY_ENV}=1 is required")
    paths = PublicAllPaths(project, run_root / "shared", run_root)
    dynamic = view("dynamic")
    checkpoint_root = run_root / "checkpoints"
    checkpoint_dir = next((checkpoint_root / "prism").iterdir())
    reload_audit = verify_prism_checkpoint_reload(checkpoint_dir)
    records = predict_prism_checkpoint_for_view(
        paths,
        dynamic,
        checkpoint_root,
        split="test",
    )
    result = {
        "status": "PASS",
        "implementation": "native_prism_v2.1.1",
        "reload_audit": reload_audit,
        "records": records,
        "fit_called_in_inference": False,
        "test_accessed": True,
    }
    _write_json(run_root / "inference" / "INFERENCE_COMPLETE.json", result)
    return result


def _inference_subprocess(project: Path, run_root: Path) -> list[str]:
    environment = dict(os.environ)
    environment[INFERENCE_ONLY_ENV] = "1"
    command = [
        sys.executable,
        "-m",
        "experiments.tim_validation.e2_stagewise.native_runner",
        "infer",
        "--project",
        str(project),
        "--run-root",
        str(run_root),
    ]
    subprocess.run(command, cwd=project.parent, env=environment, check=True)
    return command


def finalize_existing(
    project: Path,
    run_root: Path,
    *,
    regime: str,
    seed: int,
    n_samples: int,
    noise_level: str,
    started: float | None = None,
) -> None:
    development = _read(run_root / "DEVELOPMENT_COMPLETE.json")
    freeze_and_fit(project, run_root)
    materialize_test(
        run_root / "shared",
        regime,
        seed,
        selection_freeze=run_root / "freeze" / "SELECTION_FREEZE.json",
        checkpoint_seal=run_root / "freeze" / "CHECKPOINTS_SEALED.json",
        n_samples=n_samples,
        noise_level=noise_level,
    )
    paths = PublicAllPaths(project, run_root / "shared", run_root)
    _freeze_common_support(paths)
    command = _inference_subprocess(project, run_root)
    _write_json(
        run_root / "RUN_MANIFEST.json",
        {
            "status": development["status"],
            "implementation": "native_prism_v2.1.1",
            "regime": regime,
            "seed": seed,
            "n_samples": n_samples,
            "noise_level": noise_level,
            "python": platform.python_version(),
            "command": command,
            "wall_seconds": None if started is None else time.time() - started,
            "test_accessed_after_freeze": True,
            "retained_k_failures": [
                index
                for index, status in enumerate(development["stage_statuses"]["K"])
                if status != "PASS"
            ],
        },
    )


def run_all(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    run_root = args.run_root.resolve()
    if args.n_samples < 10000:
        raise ValueError(
            "native frozen 1024-step universe with four expanding folds requires "
            "at least 10000 scored samples"
        )
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"refusing nonempty run root: {run_root}")
    started = time.time()
    build_development(
        run_root / "shared",
        args.regime,
        args.seed,
        n_samples=args.n_samples,
        noise_level=args.noise_level,
    )
    development = run_development(project, run_root)
    if development["status"] not in {"PASS", "COMPLETED_WITH_RETAINED_FAILURES"}:
        return
    finalize_existing(
        project,
        run_root,
        regime=args.regime,
        seed=args.seed,
        n_samples=args.n_samples,
        noise_level=args.noise_level,
        started=started,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--project", type=Path, required=True)
    all_parser.add_argument("--run-root", type=Path, required=True)
    all_parser.add_argument("--regime", choices=REGIMES, required=True)
    all_parser.add_argument("--seed", type=int, required=True)
    all_parser.add_argument("--n-samples", type=int, default=3000)
    all_parser.add_argument(
        "--noise-level",
        choices=("clean", "very_low", "low", "medium", "high"),
        default="very_low",
    )
    infer = subparsers.add_parser("infer")
    infer.add_argument("--project", type=Path, required=True)
    infer.add_argument("--run-root", type=Path, required=True)
    finalize = subparsers.add_parser("finalize-existing")
    finalize.add_argument("--project", type=Path, required=True)
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--regime", choices=REGIMES, required=True)
    finalize.add_argument("--seed", type=int, required=True)
    finalize.add_argument("--n-samples", type=int, default=10000)
    finalize.add_argument("--noise-level", default="very_low")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "all":
        run_all(args)
    elif args.command == "infer":
        run_inference(args.project.resolve(), args.run_root.resolve())
    else:
        finalize_existing(
            args.project.resolve(),
            args.run_root.resolve(),
            regime=args.regime,
            seed=args.seed,
            n_samples=args.n_samples,
            noise_level=args.noise_level,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
