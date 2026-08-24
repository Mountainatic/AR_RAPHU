from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from prism_benchmark.representative_formal import (
    assert_scope_request,
    create_global_selection_freeze,
    fit_and_seal_formal_checkpoints,
    formal_scope,
    run_all_cz_development,
    run_formal_test_inference,
    storage_preflight,
)
from prism_benchmark.v211_representative_stage1_config import (
    ACTIVE_DATASETS,
    load_representative_stage1_descriptor,
)


def _run_public_development(args: argparse.Namespace) -> dict[str, object]:
    if args.raw_public_root is None or args.registry_root is None:
        raise RuntimeError("--raw-public-root and --registry-root are required")
    public_root = args.run_root / "public"
    command = [
        sys.executable,
        str(PROJECT / "scripts" / "launch_representative_stage1_tep_sru_cpu.py"),
        "--raw-root",
        str(args.raw_public_root.resolve()),
        "--registry-root",
        str(args.registry_root.resolve()),
        "--run-root",
        str(public_root.resolve()),
        "--workers",
        str(args.workers),
        "--per-worker-gib",
        str(args.per_worker_gib),
    ]
    subprocess.run(command, cwd=PROJECT, env=os.environ.copy(), check=True)
    status = json.loads(
        (public_root / "logs" / "LAUNCH_STATUS.json").read_text(encoding="utf-8")
    )
    if status.get("status") != "PARTIAL_DEVELOPMENT_CPU_ONLY":
        raise RuntimeError("STOP_TEP_SRU_DEVELOPMENT_ACCEPTANCE_FAILED")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formal sequential TEP/SRU/CZ L256 representative Stage-1 runner."
    )
    parser.add_argument(
        "stage",
        choices=(
            "scope",
            "preflight",
            "public-development",
            "cz-development",
            "freeze",
            "checkpoints",
            "test",
            "neural3",
            "stage2",
        ),
    )
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-public-root", type=Path)
    parser.add_argument("--registry-root", type=Path)
    parser.add_argument("--raw-cz", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--per-worker-gib", type=float, default=4.0)
    args = parser.parse_args()
    args.project = args.project.resolve()
    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    descriptor = load_representative_stage1_descriptor(args.project)
    assert_scope_request(
        datasets=ACTIVE_DATASETS,
        neural3=args.stage == "neural3",
        stage2=args.stage == "stage2",
    )
    if args.stage == "scope":
        result = formal_scope(args.project, args.run_root)
    elif args.stage == "preflight":
        result = storage_preflight(
            args.run_root.parent,
            float(descriptor["minimum_start_free_gib"]),
        )
    elif args.stage == "public-development":
        result = _run_public_development(args)
    elif args.stage == "cz-development":
        if args.raw_cz is None or not args.raw_cz.is_file():
            raise RuntimeError("--raw-cz must identify the frozen CZ workbook")
        result = run_all_cz_development(
            project=args.project,
            run_root=args.run_root,
            raw_path=args.raw_cz.resolve(),
        )
    elif args.stage == "freeze":
        result = create_global_selection_freeze(args.project, args.run_root)
    elif args.stage == "checkpoints":
        result = fit_and_seal_formal_checkpoints(args.project, args.run_root)
    elif args.stage == "test":
        if args.raw_cz is None or not args.raw_cz.is_file():
            raise RuntimeError("--raw-cz must identify the frozen CZ workbook")
        result = run_formal_test_inference(
            project=args.project,
            run_root=args.run_root,
            raw_cz=args.raw_cz.resolve(),
        )
    else:
        raise AssertionError(args.stage)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
