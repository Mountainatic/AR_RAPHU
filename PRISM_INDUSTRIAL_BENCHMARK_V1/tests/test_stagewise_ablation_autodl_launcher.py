from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "launch_stagewise_ablation_hybrid_autodl.py"
SPEC = importlib.util.spec_from_file_location("stagewise_autodl_launcher", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def test_unit_groups_keep_tep_dependency_order_and_expose_other_units() -> None:
    units = [
        {"label": "TEP H0"},
        {"label": "TEP H0 input-only"},
        {"label": "TEP H0 maturity-5"},
        {"label": "Debutanizer"},
        {"label": "SRU H2S"},
    ]
    groups = LAUNCHER._unit_groups(units)
    assert [[unit["label"] for unit in group] for group in groups] == [
        ["TEP H0", "TEP H0 input-only", "TEP H0 maturity-5"],
        ["Debutanizer"],
        ["SRU H2S"],
    ]


def test_development_worker_budget_is_explicit() -> None:
    unit = {
        "shared": Path("shared"),
        "selection_run_root": Path("selection"),
        "head_id": "HEAD",
        "information_set": "dynamic",
        "availability_scenario": "record_time",
        "proxy_policy": "primary",
    }
    command = LAUNCHER._runner_command(
        unit, "development", Path("run"), development_workers=2
    )
    workers_index = command.index("--workers")
    assert command[workers_index + 1] == "2"

