from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("/root/autodl-tmp/PRISM_V2_MODULAR_CPU_RESULTS")
stages = [
    ("V0", OUTPUT / "FREEZE/DATA_INHERITANCE_AUDIT.json", 1),
    ("V1", OUTPUT / "DEVELOPMENT/STATE_ONLY", 23),
    ("V2", OUTPUT / "DEVELOPMENT/CHANNEL_AUDIT", None),
    ("V3", OUTPUT / "DEVELOPMENT/JOINT_BASIS", 23),
    ("V4", OUTPUT / "DEVELOPMENT/WIENER", 23),
    ("V5", OUTPUT / "DEVELOPMENT/RESIDUAL_STATE", 25),
    ("V6", OUTPUT / "ASSEMBLY_CARDS", 48),
    ("V7", OUTPUT / "DEVELOPMENT/JOINT_PREDICTIVE", 25),
    ("G3", OUTPUT / "FREEZE/V2_FINAL_FREEZE_MANIFEST.json", 1),
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
