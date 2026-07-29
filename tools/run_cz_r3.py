#!/usr/bin/env python3
"""Run the frozen CZ R3 stages with one active ORSS CUDA task."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch

from ar_raphu.cz_real.orss_r3 import run_development_task
from ar_raphu.cz_real.protocol import (
    DIRECT_HORIZONS,
    LX_GRID,
    LY_GRID,
    load_furnace_a,
)
from ar_raphu.cz_real.r3_history import history_complexity_key
from ar_raphu.orss.diagnostics import write_json


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _furnace_a(raw_dir: Path, filename: str) -> Path:
    requested = raw_dir / filename
    if requested.exists():
        return requested
    candidates = [path for path in raw_dir.iterdir() if path.name.endswith("1.xlsx")]
    if len(candidates) != 1:
        raise RuntimeError("Furnace-A path is not uniquely identifiable.")
    return candidates[0]


def _history_specs(config: dict[str, object]):
    return [
        (horizon, L_x, L_y)
        for horizon in config["horizons"]
        for L_x in config["history"]["Lx"]
        for L_y in config["history"]["Ly"]
    ]


def _task_path(output: Path, horizon: int, L_x: int, L_y: int) -> Path:
    return (
        output
        / "R3A"
        / "candidates"
        / f"h_{horizon:03d}"
        / f"Lx_{L_x:03d}_Ly_{L_y:03d}"
        / "task_result.json"
    )


def _aggregate_history(output: Path, config: dict[str, object]) -> dict[str, object]:
    selections: dict[str, object] = {}
    flattened: list[dict[str, object]] = []
    one_se_payload: dict[str, object] = {}
    for horizon in config["horizons"]:
        rows: list[dict[str, object]] = []
        for L_x in config["history"]["Lx"]:
            for L_y in config["history"]["Ly"]:
                path = _task_path(output, int(horizon), int(L_x), int(L_y))
                row = json.loads(path.read_text(encoding="utf-8"))
                if row["status"] != "COMPLETED":
                    raise RuntimeError(f"R3 history task failed: {path}")
                rows.append(row)
                flattened.append(
                    {
                        "horizon": int(horizon),
                        "L_x": int(L_x),
                        "L_y": int(L_y),
                        "validation_MSE_mean": float(row["validation_loss"]),
                        "validation_MSE_SE": float(row["validation_se"]),
                        "elapsed_seconds": float(row["elapsed_seconds"]),
                    }
                )
        minimum = min(
            rows,
            key=lambda row: (
                float(row["validation_loss"]),
                history_complexity_key(
                    int(row["history"]["L_x"]), int(row["history"]["L_y"])
                ),
            ),
        )
        threshold = float(minimum["validation_loss"]) + float(
            minimum["validation_se"]
        )
        eligible = [
            row for row in rows if float(row["validation_loss"]) <= threshold
        ]
        selected = min(
            eligible,
            key=lambda row: history_complexity_key(
                int(row["history"]["L_x"]), int(row["history"]["L_y"])
            ),
        )
        selections[str(horizon)] = {
            "minimum_mean_configuration": minimum["history"],
            "one_se_threshold": threshold,
            "selected_history": selected["history"],
            "selected_validation_MSE_mean": selected["validation_loss"],
            "selected_validation_MSE_SE": selected["validation_se"],
            "selected_penalty": selected["penalty"]["selection"]["selected"][
                "normalized_weights"
            ],
            "history_selected_at_grid_edge": (
                int(selected["history"]["L_x"])
                in {min(LX_GRID), max(LX_GRID)}
                or int(selected["history"]["L_y"])
                in {min(LY_GRID), max(LY_GRID)}
            ),
        }
        one_se_payload[str(horizon)] = {
            "threshold": threshold,
            "eligible_histories": [row["history"] for row in eligible],
        }
    result = {
        "schema": "CZ_R3A_NATIVE_HISTORY_SELECTION_ORSS_V1",
        "status": "COMPLETED",
        "selection_unit": "PER_DIRECT_HORIZON",
        "one_standard_error_rule": True,
        "complexity_key": ["Lx+Ly", "Lx*Ly", "max(Lx,Ly)", "Lx", "Ly"],
        "anchor": {
            "M_tau": int(config["history"]["anchor_Mtau"]),
            "M_x": int(config["history"]["anchor_Mx"]),
            "CONTINUATION_SCALE_COEFFICIENT": float(
                config["history"]["continuation_scale_coefficient"]
            ),
        },
        "selections": selections,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_access_count": 0,
        "next_stage": "R3B_RESOLUTION_AND_PENALTY",
    }
    write_json(output / "R3A" / "history_selection.json", result)
    write_json(
        output / "R3A" / "history_one_se_set.json", one_se_payload
    )
    try:
        import pandas as pd

        frame = pd.DataFrame(flattened)
        frame.to_parquet(output / "R3A" / "history_candidates.parquet", index=False)
    except Exception as error:
        write_json(
            output / "R3A" / "history_candidates.json",
            {"rows": flattened, "parquet_error": repr(error)},
        )
    return result


def _run_history(
    *,
    config: dict[str, object],
    config_path: Path,
    data,
    output: Path,
    device: torch.device,
    resume: bool,
) -> None:
    specifications = _history_specs(config)
    started = time.perf_counter()
    source_commit = _source_commit()
    config_hash = _hash_file(config_path)
    for index, (horizon, L_x, L_y) in enumerate(specifications, start=1):
        destination = _task_path(output, horizon, L_x, L_y)
        if resume and destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            identity = existing.get("checkpoint_identity", {})
            if (
                existing.get("status") == "COMPLETED"
                and identity.get("config_hash") == config_hash
                and identity.get("data_hash") == data.source_sha256
            ):
                print(
                    f"R3A {index}/{len(specifications)} resumed "
                    f"h={horizon} Lx={L_x} Ly={L_y}",
                    flush=True,
                )
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = run_development_task(
            data.inputs,
            data.target,
            horizon=horizon,
            L_x=L_x,
            L_y=L_y,
            M_tau=int(config["history"]["anchor_Mtau"]),
            M_x=int(config["history"]["anchor_Mx"]),
            c_rho=float(config["history"]["continuation_scale_coefficient"]),
            device=device,
            primary_dtype=torch.float32,
            chunk_time=int(config["cuda"]["operator_chunk_time"]),
            positive_grid_points=int(config["penalty"]["positive_grid_points"]),
            maximum_edge_expansions=int(
                config["penalty"]["maximum_edge_expansions"]
            ),
            rb_tolerance=float(
                config["reduced_basis"]["residual_tolerance_development"]
            ),
            rb_maximum_dimension=int(
                config["reduced_basis"]["maximum_dimension"]
            ),
            krylov_tolerance=float(
                config["krylov"]["relative_tolerance_development"]
            ),
            krylov_maximum_iterations=int(
                config["krylov"]["maximum_iterations"]
            ),
        )
        payload["checkpoint_identity"] = {
            "dataset": "CZ_REAL_V1",
            "furnace": "A",
            "fold": "ALL_DEVELOPMENT",
            "horizon": horizon,
            "Lx": L_x,
            "Ly": L_y,
            "Mtau": int(config["history"]["anchor_Mtau"]),
            "Mx": int(config["history"]["anchor_Mx"]),
            "solver": "ORSS",
            "source_commit": source_commit,
            "config_hash": config_hash,
            "data_hash": data.source_sha256,
            "solver_version": "ORSS_V1",
        }
        write_json(destination, payload)
        marker = destination.parent / (
            "DONE" if payload["status"] == "COMPLETED" else "FAILED"
        )
        marker.write_text(payload["status"] + "\n", encoding="utf-8")
        print(
            f"R3A {index}/{len(specifications)} h={horizon} "
            f"Lx={L_x} Ly={L_y} status={payload['status']} "
            f"loss={float(payload['validation_loss']):.8g} "
            f"elapsed={float(payload['elapsed_seconds']):.1f}s",
            flush=True,
        )
        if payload["status"] != "COMPLETED":
            raise RuntimeError(f"R3A_TASK_FAILED:{destination}")
    selection = _aggregate_history(output, config)
    write_json(
        output / "R3A" / "history_runtime_profile.json",
        {
            "status": "COMPLETED",
            "one_active_gpu_task": True,
            "completed_tasks": len(specifications),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    write_json(
        output / "R3A" / "R3A_STATUS.json",
        {
            "stage": "R3A_NATIVE_HISTORY_SELECTION",
            "status": "COMPLETED",
            "H2_NATIVE_COMPLETE": True,
            "next_stage": selection["next_stage"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("history", "resolution-penalty", "continuation", "rank"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--solver", choices=("orss",), default="orss")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every-task", action="store_true")
    parser.add_argument("--furnace-a-only", action="store_true")
    parser.add_argument(
        "--raw-dir",
        default="/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw",
    )
    parser.add_argument(
        "--output", default="results/cz_real_data/complete_5090"
    )
    args = parser.parse_args()
    if not args.furnace_a_only:
        raise RuntimeError("FURNACE_B_ACCESSED_BEFORE_FREEZE")
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("REAL_CUDA_DISPATCH_REQUIRED")
    torch.backends.cuda.matmul.allow_tf32 = bool(
        config["cuda"]["allow_tf32_operator"]
    )
    raw_dir = Path(args.raw_dir)
    data = load_furnace_a(
        _furnace_a(raw_dir, str(config["data"]["furnace_a_filename"]))
    )
    output = Path(args.output)
    if args.stage == "history":
        _run_history(
            config=config,
            config_path=config_path,
            data=data,
            output=output,
            device=device,
            resume=args.resume,
        )
    else:
        raise RuntimeError(f"STAGE_IMPLEMENTATION_PENDING:{args.stage}")


if __name__ == "__main__":
    main()
