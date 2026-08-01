from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable strict PRISM CPU stage chain")
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    state_path = args.results / "CHAIN_STATUS.json"
    state: dict[str, Any] = {
        "pipeline": "PRISM_CPU_STRICT_CHAIN_V1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "stages": {},
    }
    _write_status(state_path, state)
    python = sys.executable
    env_prefix = [python]

    try:
        _run(
            env_prefix
            + [
                str(args.project / "scripts/validate_shared_data.py"),
                "--shared",
                str(args.shared),
                "--output",
                str(args.shared / "C1_VALIDATION.json"),
            ],
            args.results / "LOGS/C1_VALIDATE.log",
        )
        state["stages"]["C1_VALIDATE"] = "PASS"
        _write_status(state_path, state)

        _run(
            env_prefix
            + [
                str(args.project / "scripts/run_c2_simple_baselines.py"),
                "--shared",
                str(args.shared),
                "--output",
                str(args.results / "C2"),
            ],
            args.results / "LOGS/C2_SIMPLE.log",
        )
        state["stages"]["C2_SIMPLE"] = "PASS"
        state["status"] = "BLOCKED_BY_MISSING_SPECIFICATION"
        state["next_stage"] = "C2_CLASSICAL_SOFT_SENSOR"
        state["blockers"] = [
            "C2 exact Ridge alpha endpoints",
            "C2 DPLS lag/profile grid",
            "C2 SVR and XGBoost candidate grids",
            "C3 exact model grids and solver contracts",
            "C4/C5 model gates listed in docs/IMPLEMENTATION_FREEZE_GATE.md",
        ]
        _write_status(state_path, state)
    except Exception as error:
        state["status"] = "FAILED"
        state["error"] = str(error)
        _write_status(state_path, state)
        raise


if __name__ == "__main__":
    main()
