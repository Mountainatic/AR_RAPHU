from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] RUN {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")


def _stage(state: dict[str, Any], state_path: Path, name: str, command: list[str], log: Path) -> None:
    state["current_stage"] = name
    state["stages"].setdefault(name, {})["started_utc"] = datetime.now(timezone.utc).isoformat()
    state["stages"][name]["status"] = "RUNNING"
    _write_status(state_path, state)
    current_command = list(command)
    while True:
        try:
            _run(current_command, log)
            break
        except RuntimeError:
            if "--n-jobs" not in current_command:
                raise
            position = current_command.index("--n-jobs") + 1
            workers = int(current_command[position])
            if workers <= 1:
                raise
            current_command[position] = str(workers - 1)
            state["stages"][name].setdefault("parallel_fallbacks", []).append(
                {"failed_workers": workers, "retry_workers": workers - 1, "utc": datetime.now(timezone.utc).isoformat()}
            )
            _write_status(state_path, state)
    state["stages"][name]["status"] = "PASS"
    state["stages"][name]["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_status(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable strict PRISM CPU stage chain")
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--return-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--c2-jobs", type=int, default=8)
    parser.add_argument("--c3-jobs", type=int, default=6)
    parser.add_argument("--c4-jobs", type=int, default=8)
    parser.add_argument("--c5-jobs", type=int, default=2)
    parser.add_argument("--c5-inner-jobs", type=int, default=1)
    parser.add_argument("--c6-jobs", type=int, default=1)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--release-tag", default="prism-industrial-cpu-v1-20260802")
    arguments = parser.parse_args()

    arguments.results.mkdir(parents=True, exist_ok=True)
    state_path = arguments.results / "CHAIN_STATUS.json"
    if state_path.is_file() and json.loads(state_path.read_text(encoding="utf-8")).get("schema_version") == 2:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["resumed_utc"] = datetime.now(timezone.utc).isoformat()
        state.pop("error", None)
        state.pop("failed_utc", None)
        state["status"] = "RUNNING"
    else:
        state = {
            "schema_version": 2,
            "pipeline": "PRISM_CPU_STRICT_CHAIN_V1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "test_accessed": False,
            "stages": {},
        }
    _write_status(state_path, state)
    python = sys.executable
    project = arguments.project.resolve()
    results = arguments.results.resolve()

    commands = [
        ("C1_VALIDATE", [python, str(project / "scripts/validate_shared_data.py"), "--shared", str(arguments.shared), "--output", str(arguments.shared / "C1_VALIDATION.json")]),
        ("C2_SIMPLE", [python, str(project / "scripts/run_c2_simple_baselines.py"), "--shared", str(arguments.shared), "--output", str(results / "C2")]),
        ("C2_CLASSICAL", [python, str(project / "scripts/run_c2_classical.py"), "--shared", str(arguments.shared), "--project", str(project), "--output", str(results / "C2_CLASSICAL"), "--n-jobs", str(arguments.c2_jobs)]),
        ("C3_SYSTEM_IDENTIFICATION", [python, str(project / "scripts/run_c3_system_identification.py"), "--shared", str(arguments.shared), "--project", str(project), "--output", str(results / "C3_SYSTEM_IDENTIFICATION"), "--n-jobs", str(arguments.c3_jobs)]),
        ("C4_PRISM", [python, str(project / "scripts/run_c4_prism_profile_audit.py"), "--shared", str(arguments.shared), "--project", str(project), "--output", str(results / "C4_PRISM"), "--n-jobs", str(arguments.c4_jobs)]),
        ("C5_PRISM_DYNAMIC", [python, str(project / "scripts/run_c5_prism_dynamic.py"), "--shared", str(arguments.shared), "--project", str(project), "--c3-output", str(results / "C3_SYSTEM_IDENTIFICATION"), "--c4-output", str(results / "C4_PRISM"), "--output", str(results / "C5_PRISM_DYNAMIC"), "--n-jobs", str(arguments.c5_jobs), "--inner-jobs", str(arguments.c5_inner_jobs)]),
        ("C6_FINAL", [python, str(project / "scripts/run_c6_final.py"), "--shared", str(arguments.shared), "--project", str(project), "--c2-output", str(results / "C2_CLASSICAL"), "--c3-output", str(results / "C3_SYSTEM_IDENTIFICATION"), "--c4-output", str(results / "C4_PRISM"), "--c5-output", str(results / "C5_PRISM_DYNAMIC"), "--output", str(results / "C6_FINAL"), "--n-jobs", str(arguments.c6_jobs)]),
        ("PACKAGE", [python, str(project / "scripts/build_cpu_bundle.py"), "--project", str(project), "--results", str(results), "--shared", str(arguments.shared), "--output-root", str(arguments.return_root)]),
    ]

    try:
        for name, command in commands:
            previous = state.get("stages", {}).get(name, {})
            if isinstance(previous, dict) and previous.get("status") == "PASS" and name not in {"C1_VALIDATE", "PACKAGE"}:
                continue
            _stage(state, state_path, name, command, results / "LOGS" / f"{name}.log")
            if name == "C6_FINAL":
                state["test_accessed"] = True
        if arguments.publish:
            package_manifest = json.loads((arguments.return_root / "PACKAGE_OUTPUT.json").read_text(encoding="utf-8"))
            command = [
                python,
                str(project / "scripts/publish_cpu_results.py"),
                "--project",
                str(project),
                "--results",
                str(results),
                "--bundle",
                package_manifest["zip"],
                "--sha256-file",
                package_manifest["sha256_file"],
                "--release-tag",
                arguments.release_tag,
            ]
            _stage(state, state_path, "PUBLISH", command, results / "LOGS/PUBLISH.log")
        state["status"] = "COMPLETED"
        state["current_stage"] = None
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_status(state_path, state)
    except Exception as error:
        state["status"] = "FAILED"
        state["error"] = str(error)
        state["failed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_status(state_path, state)
        raise


if __name__ == "__main__":
    main()
