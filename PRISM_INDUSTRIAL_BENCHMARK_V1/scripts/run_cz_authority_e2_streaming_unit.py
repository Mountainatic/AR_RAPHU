"""Run one isolated authority E2 unit under bounded-retention execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_streaming_gate(parent_run: Path) -> dict[str, Any]:
    path = parent_run / "STREAMING_STORAGE_GATE.json"
    if not path.is_file():
        raise RuntimeError("STOP_E2_STREAMING_GATE_MISSING")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError("STOP_E2_STREAMING_GATE_NOT_PASS")
    return value


def materialize(
    template_shared: Path,
    checkpoint: Path,
    unit_root: Path,
    *,
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    from prism_benchmark.cz_authority_semisynthetic import materialize_s1_k_pilot

    return materialize_s1_k_pilot(
        template_shared,
        checkpoint,
        unit_root / "shared",
        seed=seed,
        sample_size=sample_size,
    )


def _classify_authority_records(records: list[dict[str, Any]]) -> str:
    statuses = [str(item.get("status")) for item in records]
    strict_pass = {
        "PASS",
        "NOT_RUN_PROTOCOL_INCOMPATIBLE",
        "JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT",
    }
    retained = {
        "SOLVER_FAILED_RETAINED",
        "COMPLETED_WITH_RETAINED_FAILURES",
    }
    if all(status in strict_pass for status in statuses):
        return "PASS"
    if all(status in strict_pass | retained for status in statuses):
        return "COMPLETED_WITH_RETAINED_FAILURES"
    return "FAILED"


def _compact_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": item.get("stage"),
        "status": item.get("status"),
        "channel": item.get("channel"),
        "selected_family": item.get("selected_family"),
        "selection_status": item.get("selection_status"),
        "error_type": item.get("error_type"),
        "error": item.get("error"),
        "test_accessed": item.get("test_accessed", False),
        "ood_accessed": item.get("ood_accessed", False),
    }


def summarize_existing(unit_root: Path, *, inner_workers: int) -> dict[str, Any]:
    paths = sorted((unit_root / "results" / "DEVELOPMENT").rglob("RESULT.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(records) != 8:
        raise RuntimeError(f"STOP_E2_EXISTING_UNIT_STAGE_COUNT:{len(records)}")
    compact = [_compact_record(item) for item in records]
    result = {
        "status": _classify_authority_records(records),
        "role": "P1_TINY_PILOT_NON_SELECTION_AUTHORITY",
        "authority_modules": "K_C_W_A_JOINT_UNMODIFIED",
        "inner_workers": int(inner_workers),
        "records": compact,
        "retained_failure_count": sum(
            item["status"] == "SOLVER_FAILED_RETAINED" for item in compact
        ),
        "formal_target_or_ood_accessed": any(
            bool(item["test_accessed"] or item["ood_accessed"])
            for item in compact
        ),
        "completed_utc": _utc(),
    }
    _write_json(unit_root / "AUTHORITY_REFIT_STATUS.json", result)
    if result["status"] == "FAILED" or result["formal_target_or_ood_accessed"]:
        raise RuntimeError("STOP_E2_EXISTING_UNIT_INVALID")
    return result


def fit_authority(
    project: Path, unit_root: Path, *, inner_workers: int = 1
) -> dict[str, Any]:
    from prism_benchmark.cz_k_support import run_cz_k_channel
    from prism_benchmark.cz_l256_nowcast import INPUT_COLUMNS, view
    from prism_benchmark.v211_a import run_a_view
    from prism_benchmark.v211_c import run_c_view
    from prism_benchmark.v211_config import REPRESENTATIVE_STAGE1_PROTOCOL
    from prism_benchmark.v211_joint_stability import run_joint_stability_view
    from prism_benchmark.v211_w import run_w_view

    if inner_workers < 1:
        raise ValueError("inner_workers must be positive")
    for name in (
        "PRISM_V211_K_INNER_WORKERS",
        "PRISM_V211_C_INNER_WORKERS",
        "PRISM_V211_W_INNER_WORKERS",
        "PRISM_V211_A_INNER_WORKERS",
        "PRISM_V211_JOINT_INNER_WORKERS",
    ):
        os.environ[name] = str(int(inner_workers))
    shared = unit_root / "shared"
    output = unit_root / "results"
    output.mkdir(parents=True, exist_ok=True)
    registry = json.loads((shared / "TASK_REGISTRY.json").read_text(encoding="utf-8"))
    heads = registry.get("heads", [])
    if len(heads) != 1 or int(heads[0].get("h_steps", -1)) != 4:
        raise RuntimeError("STOP_E2_UNIT_TASK_REGISTRY_IS_NOT_SINGLE_H4")
    task_id = str(heads[0]["head_id"])
    input_view = view("input_only", h_steps=4, task_id=task_id)
    dynamic_view = view("dynamic", h_steps=4, task_id=task_id)
    records: list[dict[str, Any]] = []
    for channel in INPUT_COLUMNS:
        records.append(
            run_cz_k_channel(
                shared,
                project,
                output,
                input_view,
                channel,
                REPRESENTATIVE_STAGE1_PROTOCOL,
            )
        )
    records.append(
        run_c_view(
            shared, project, output, input_view, REPRESENTATIVE_STAGE1_PROTOCOL
        )
    )
    records.append(
        run_w_view(
            shared, project, output, input_view, REPRESENTATIVE_STAGE1_PROTOCOL
        )
    )
    records.append(
        run_a_view(
            shared, project, output, dynamic_view, REPRESENTATIVE_STAGE1_PROTOCOL
        )
    )
    records.append(
        run_joint_stability_view(
            shared,
            project,
            output,
            None,
            dynamic_view,
            REPRESENTATIVE_STAGE1_PROTOCOL,
        )
    )
    compact = [_compact_record(item) for item in records]
    result = {
        "status": _classify_authority_records(records),
        "role": "P1_TINY_PILOT_NON_SELECTION_AUTHORITY",
        "authority_modules": "K_C_W_A_JOINT_UNMODIFIED",
        "inner_workers": int(inner_workers),
        "records": compact,
        "retained_failure_count": sum(
            item["status"] == "SOLVER_FAILED_RETAINED" for item in compact
        ),
        "formal_target_or_ood_accessed": False,
        "completed_utc": _utc(),
    }
    _write_json(unit_root / "AUTHORITY_REFIT_STATUS.json", result)
    if result["status"] == "FAILED":
        raise RuntimeError("STOP_E2_TINY_PILOT_AUTHORITY_REFIT_FAILED")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("materialize", "fit", "summarize", "all"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--template-shared", type=Path, required=True)
    parser.add_argument("--authority-checkpoint", type=Path, required=True)
    parser.add_argument("--unit-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--inner-workers", type=int, default=1)
    args = parser.parse_args()
    project = args.project.resolve()
    unit_root = args.unit_root.resolve()
    sys.path.insert(0, str(project / "src"))
    gate = _require_streaming_gate(args.parent_run_root.resolve())
    unit_root.mkdir(parents=True, exist_ok=True)
    if args.stage != "summarize":
        _write_json(
            unit_root / "UNIT_START.json",
            {
                "status": "RUNNING",
                "role": "P1_TINY_PILOT_NON_SELECTION_AUTHORITY",
                "regime": "S1_K",
                "seed": int(args.seed),
                "sample_size": int(args.sample_size),
                "inner_workers": int(args.inner_workers),
                "streaming_amendment_sha256": gate["amendment_sha256"],
                "started_utc": _utc(),
            },
        )
    result: dict[str, Any] = {}
    if args.stage in {"materialize", "all"}:
        result["materialize"] = materialize(
            args.template_shared.resolve(),
            args.authority_checkpoint.resolve(),
            unit_root,
            seed=int(args.seed),
            sample_size=int(args.sample_size),
        )
    if args.stage in {"fit", "all"}:
        result["fit"] = fit_authority(
            project, unit_root, inner_workers=int(args.inner_workers)
        )
    if args.stage == "summarize":
        result["summarize"] = summarize_existing(
            unit_root, inner_workers=int(args.inner_workers)
        )
    print(json.dumps({"stage": args.stage, "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
