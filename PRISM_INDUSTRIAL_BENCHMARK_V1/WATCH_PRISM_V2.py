from __future__ import annotations

import json
import os
from pathlib import Path


OUTPUT = Path("/root/autodl-tmp/PRISM_V2_MODULAR_CPU_RESULTS")
stages = [
    ("V0", OUTPUT / "FREEZE/DATA_INHERITANCE_AUDIT.json", 1),
    ("V1", OUTPUT / "DEVELOPMENT/STATE_ONLY", 23),
    ("V2", OUTPUT / "DEVELOPMENT/CHANNEL_AUDIT", 322),
    ("V3", OUTPUT / "DEVELOPMENT/JOINT_BASIS", 23),
    ("V4", OUTPUT / "DEVELOPMENT/WIENER", 23),
    ("V5", OUTPUT / "DEVELOPMENT/RESIDUAL_STATE", 25),
    ("V6", OUTPUT / "ASSEMBLY_CARDS", 48),
    ("V7", OUTPUT / "DEVELOPMENT/JOINT_PREDICTIVE", 25),
    ("BDEV", OUTPUT / "BASELINE_DEVELOPMENT/SUMMARY.json", 1),
    ("G3", OUTPUT / "FREEZE/V2_FINAL_FREEZE_MANIFEST.json", 1),
    ("V8C", OUTPUT / "PREDICTIONS/LEVEL_C_CONFIRMATION/SUMMARY.json", 1),
    ("V8B", OUTPUT / "PREDICTIONS/LEVEL_B_PRIMARY_EXPLORATORY/SUMMARY.json", 1),
    ("REPORT", OUTPUT / "PRISM_V2_FINAL_DECISION.json", 1),
    ("PACKAGE", Path("/root/autodl-tmp/PRISM_V2_MODULAR_CPU_RELEASE/PRISM_V2_RELEASE_PARTS.json"), 1),
]
for name, path, expected in stages:
    if path.is_file():
        value = json.loads(path.read_text())
        print(f"{name}: {value.get('status', 'PRESENT')}")
        continue
    if path.is_dir():
        count = len(list(path.rglob("RESULT.json"))) if name not in {"V6"} else len(list(path.rglob("ASSEMBLY_CARD.json")))
        suffix = "?" if expected is None else str(expected)
        print(f"{name}: {count}/{suffix}")
    else:
        print(f"{name}: NOT_STARTED")


def _integer(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


limit = _integer(Path("/sys/fs/cgroup/memory.max"))
current = _integer(Path("/sys/fs/cgroup/memory.current"))
if limit and current is not None:
    print(f"MEMORY_CGROUP: {current / 1024**3:.2f}/{limit / 1024**3:.2f} GiB ({100 * current / limit:.1f}%)")

processes = []
for item in Path("/proc").iterdir():
    if not item.name.isdigit():
        continue
    try:
        command = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        if "run_prism_v2_stage.py" not in command:
            continue
        stat = (item / "stat").read_text().split()
        ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        cpu_seconds = (int(stat[13]) + int(stat[14])) / ticks
        pss_kib = 0
        for line in (item / "smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                pss_kib = int(line.split()[1])
                break
        processes.append((int(item.name), cpu_seconds, pss_kib))
    except (OSError, ValueError, IndexError):
        continue
if processes:
    print(
        f"ACTIVE_STAGE_PROCESSES: {len(processes)}; "
        f"aggregate_PSS={sum(item[2] for item in processes) / 1024**2:.2f} GiB; "
        f"aggregate_CPU_time={sum(item[1] for item in processes):.1f}s"
    )

for log in (OUTPUT / "logs/CHAIN_NOHUP.log", OUTPUT / "logs/CHAIN.log"):
    if not log.is_file():
        continue
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    events = [line for line in lines if "PRISM_PROCESS_POOL_START" in line]
    if events:
        print(f"LATEST_POOL: {events[-1]}")
        break
