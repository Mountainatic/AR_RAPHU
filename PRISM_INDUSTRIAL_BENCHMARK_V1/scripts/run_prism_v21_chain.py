from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


STAGES = ("e0", "e1", "e2k", "e2c", "e3", "e4", "e5", "e6", "e7", "e8")


def _marker(output: Path, stage: str) -> Path:
    return {
        "e0": output / "FREEZE" / "E0_INHERITANCE_AUDIT.json",
        "e1": output / "FREEZE" / "E1_REGRESSION_TESTS.json",
        "e2k": output / "DEVELOPMENT" / "K" / "SUMMARY.json",
        "e2c": output / "DEVELOPMENT" / "C" / "SUMMARY.json",
        "e3": output / "DEVELOPMENT" / "W" / "SUMMARY.json",
        "e4": output / "DEVELOPMENT" / "A" / "SUMMARY.json",
        "e5": output / "DEVELOPMENT" / "JOINT" / "SUMMARY.json",
        "e6": output / "FREEZE" / "V21_SRU_FINAL_FREEZE_MANIFEST.json",
        "e7": output / "FINAL" / "TEST_ACCESS_AUDIT.json",
        "e8": output / "REPORTS" / "E8_SUMMARY.json",
    }[stage]


def _complete(marker: Path, stage: str) -> bool:
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "e6": {"ASSEMBLY_FROZEN"},
        "e8": {"COMPLETED"},
    }.get(stage, {"PASS"})
    return value.get("status") in expected


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete frozen PRISM v2.1 SRU E0-E8 chain",
    )
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--through", choices=STAGES, default="e8")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Fail when a completed stage marker exists instead of resuming.",
    )
    args = parser.parse_args()
    project = args.project.resolve()
    shared = args.shared.resolve()
    baseline_root = args.baseline_root.resolve()
    output = (args.output or project / "results_prism_v2_1_sru").resolve()
    if output.name != "results_prism_v2_1_sru":
        raise SystemExit("output directory must be named results_prism_v2_1_sru")
    if not shared.is_dir():
        raise SystemExit(f"shared data root does not exist: {shared}")
    if not baseline_root.is_dir():
        raise SystemExit(f"baseline root does not exist: {baseline_root}")
    stop = STAGES.index(args.through)
    selected = STAGES[: stop + 1]
    log = output / "RUN_LOG" / "V21_CHAIN_EVENTS.jsonl"
    stage_script = project / "scripts" / "run_prism_v21_stage.py"
    for stage in selected:
        marker = _marker(output, stage)
        if _complete(marker, stage):
            if args.no_resume:
                raise SystemExit(f"completed stage already exists: {stage}: {marker}")
            _append_event(
                log,
                {
                    "event": "SKIP_COMPLETED",
                    "stage": stage,
                    "marker": str(marker),
                    "time_unix": time.time(),
                },
            )
            continue
        if marker.exists():
            raise SystemExit(f"non-PASS stage marker requires audit: {stage}: {marker}")
        if stage == "e7" and not _complete(_marker(output, "e6"), "e6"):
            raise SystemExit("E7 test access refused because E6 freeze is absent")
        command = [
            sys.executable,
            str(stage_script),
            stage,
            "--project",
            str(project),
            "--shared",
            str(shared),
            "--baseline-root",
            str(baseline_root),
            "--output",
            str(output),
        ]
        started = time.time()
        _append_event(
            log,
            {
                "event": "START",
                "stage": stage,
                "command": command,
                "time_unix": started,
            },
        )
        completed = subprocess.run(command, cwd=project, text=True)
        _append_event(
            log,
            {
                "event": "FINISH",
                "stage": stage,
                "returncode": completed.returncode,
                "elapsed_seconds": time.time() - started,
                "time_unix": time.time(),
            },
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)
        if not _complete(marker, stage):
            raise SystemExit(f"stage returned success without a valid marker: {stage}")
    summary = {
        "status": "PASS",
        "stages": list(selected),
        "through": args.through,
        "output": str(output),
        "test_accessed": STAGES.index(args.through) >= STAGES.index("e7"),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
