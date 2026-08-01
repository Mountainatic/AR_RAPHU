from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    dataset: str
    target: str
    horizon_seconds: float
    target_window_seconds: float
    dynamic: bool
    input_policy: str
    split_policy: str = "chronological_60_20_20"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


TASKS = (
    TaskSpec("TEP_G12", "TEP", "XMEAS(40)", 12 * 60, 6 * 60, True, "tep_proxy_excluded", "run_or_time"),
    TaskSpec("DEB_C4", "Debutanizer", "y", 30 * 60, 0, True, "all_online_inputs"),
    TaskSpec("SRU_H2S", "SRU", "H2S", 5 * 60, 60, True, "sru_process_only"),
    TaskSpec("SRU_SO2", "SRU", "SO2", 5 * 60, 60, True, "sru_process_only"),
    TaskSpec("PMSM_PM5", "PMSM", "pm", 5 * 60, 30, True, "pmsm_external_only", "profile_id"),
    TaskSpec("METRO_P60", "MetroPT-3", "Reservoirs", 60, 10, True, "metro_tp3_excluded", "month"),
    TaskSpec("METRO_OIL20", "MetroPT-3", "Oil_temperature", 20 * 60, 2 * 60, True, "metro_all_except_target", "month"),
)


def task_registry() -> list[dict[str, Any]]:
    return [t.as_dict() for t in TASKS]


def find_task(task_id: str) -> TaskSpec:
    for task in TASKS:
        if task.task_id == task_id:
            return task
    raise KeyError(task_id)


def dump_protocol(path: Path, *, sample_cap: int | None = None) -> None:
    import json
    payload = {
        "protocol": "PRISM_INDUSTRIAL_BENCHMARK_V1",
        "version": "1.0",
        "status": "PRE-REGISTERED",
        "tasks": task_registry(),
        "sample_cap": sample_cap,
        "raw_data_excluded_from_return_bundle": True,
        "cpu_dtype": "float64",
        "split_rule": "target_index_chronological_no_future_features",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

