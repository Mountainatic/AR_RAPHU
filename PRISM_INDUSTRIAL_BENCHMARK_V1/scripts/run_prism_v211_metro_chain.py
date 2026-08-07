from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DEFAULT = Path(__file__).resolve().parents[1]
STAGES = ("m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8")
OUTPUT_DIRECTORY = "results_prism_v2_1_1_metro_p60_w_audit"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen Metro-P60 audit in strict M0-to-M8 order."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=Path(os.environ.get("PRISM_SHARED_ROOT", "/root/autodl-tmp/PRISM_SHARED_DATA_C1")),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--start-stage", choices=STAGES, default="m0")
    parser.add_argument("--stop-after", choices=STAGES, default="m8")
    args = parser.parse_args()
    start = STAGES.index(args.start_stage)
    stop = STAGES.index(args.stop_after)
    if start > stop:
        raise SystemExit("--start-stage must not follow --stop-after")
    project = args.project_root.resolve()
    shared = args.shared_root.resolve()
    output = (args.output_root or project / OUTPUT_DIRECTORY).resolve()
    run_log = output / "RUN_LOG"
    run_log.mkdir(parents=True, exist_ok=True)
    runtime_path = run_log / "STAGE_RUNTIME.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(project / "src")
        if not env.get("PYTHONPATH")
        else str(project / "src") + os.pathsep + env["PYTHONPATH"]
    )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "MALLOC_ARENA_MAX": "2",
            "PRISM_V211_WORKERS": "2",
            "PRISM_V211_METRO_WORKERS": env.get(
                "PRISM_V211_METRO_WORKERS", "27"
            ),
            "PRISM_V211_K_MEMORY_GIB_PER_WORKER": env.get(
                "PRISM_V211_K_MEMORY_GIB_PER_WORKER", "1.75"
            ),
            "PRISM_V211_MEMORY_GIB_PER_WORKER": env.get(
                "PRISM_V211_MEMORY_GIB_PER_WORKER", "20"
            ),
        }
    )
    stage_script = project / "scripts" / "run_prism_v211_metro_stage.py"
    for stage in STAGES[start : stop + 1]:
        command = [
            sys.executable,
            str(stage_script),
            stage,
            "--project-root",
            str(project),
            "--shared-root",
            str(shared),
            "--output-root",
            str(output),
        ]
        resource_path = run_log / f"{stage.upper()}.resource.txt"
        if Path("/usr/bin/time").is_file():
            command = [
                "/usr/bin/time",
                "-v",
                "-o",
                str(resource_path),
                *command,
            ]
        stdout_path = run_log / f"{stage.upper()}.stdout.log"
        stderr_path = run_log / f"{stage.upper()}.stderr.log"
        started = time.time()
        print(f"[{stage.upper()}] starting", flush=True)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=project,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        record = {
            "stage": stage,
            "started_unix": started,
            "finished_unix": time.time(),
            "wall_seconds": time.time() - started,
            "returncode": completed.returncode,
            "stdout": str(stdout_path.relative_to(output)),
            "stderr": str(stderr_path.relative_to(output)),
            "resource_audit": (
                str(resource_path.relative_to(output))
                if resource_path.is_file()
                else None
            ),
        }
        with runtime_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"[{stage.upper()}] returncode={completed.returncode} "
            f"wall_seconds={record['wall_seconds']:.1f}",
            flush=True,
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
