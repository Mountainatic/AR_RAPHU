#!/usr/bin/env python3
"""Run and aggregate the frozen CZ R3-A native-history grid."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from ar_raphu.cz_real.protocol import DIRECT_HORIZONS, LX_GRID, LY_GRID, load_furnace_a
from ar_raphu.cz_real.r3_history import history_complexity_key, run_history_candidate


WORKER_X: np.ndarray | None = None
WORKER_Y: np.ndarray | None = None


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _initialize(raw_dir: str) -> None:
    global WORKER_X, WORKER_Y
    os.environ["OMP_NUM_THREADS"] = "3"
    os.environ["OPENBLAS_NUM_THREADS"] = "3"
    os.environ["MKL_NUM_THREADS"] = "3"
    data = load_furnace_a(Path(raw_dir) / "实验数据1.xlsx")
    WORKER_X = data.inputs
    WORKER_Y = data.target


def _task(specification: tuple[int, int, int, str]) -> tuple[str, dict[str, object]]:
    L_x, L_y, horizon, output_text = specification
    if WORKER_X is None or WORKER_Y is None:
        raise RuntimeError("R3-A worker data was not initialized.")
    output = Path(output_text)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            return output_text, existing
    payload = run_history_candidate(
        WORKER_X,
        WORKER_Y,
        L_x=L_x,
        L_y=L_y,
        horizon=horizon,
        M_tau=16,
        M_x=32,
        continuation_scale_coefficient=1.0,
        positive_grid_points=7,
        maximum_edge_expansions=2,
    )
    _write(output, payload)
    return output_text, payload


def _aggregate(output: Path) -> dict[str, object]:
    selections: dict[str, object] = {}
    for horizon in DIRECT_HORIZONS:
        rows = []
        for L_x in LX_GRID:
            for L_y in LY_GRID:
                path = (
                    output
                    / "candidates"
                    / f"h_{horizon:03d}"
                    / f"Lx_{L_x:03d}_Ly_{L_y:03d}.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["status"] != "COMPLETED":
                    raise RuntimeError(f"R3-A candidate failed: {path}")
                rows.append(payload)
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
            "selected_penalty": selected["penalty"]["selected"][
                "scientific_penalty_weight"
            ],
            "history_selected_at_grid_edge": (
                int(selected["history"]["L_x"]) in {min(LX_GRID), max(LX_GRID)}
                or int(selected["history"]["L_y"]) in {min(LY_GRID), max(LY_GRID)}
            ),
        }
    return {
        "schema": "CZ_R3A_NATIVE_HISTORY_SELECTION_V1",
        "status": "COMPLETED",
        "selection_unit": "PER_DIRECT_HORIZON",
        "one_standard_error_rule": True,
        "complexity_key": ["Lx+Ly", "Lx*Ly", "max(Lx,Ly)", "Lx", "Ly"],
        "anchor": {
            "M_tau": 16,
            "M_x": 32,
            "CONTINUATION_SCALE_COEFFICIENT": 1.0,
        },
        "selections": selections,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_accessed": False,
        "next_stage": "R3B_RESOLUTION_AND_PENALTY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output).resolve() / "R3A"
    specifications = [
        (
            L_x,
            L_y,
            horizon,
            str(
                output
                / "candidates"
                / f"h_{horizon:03d}"
                / f"Lx_{L_x:03d}_Ly_{L_y:03d}.json"
            ),
        )
        for horizon in DIRECT_HORIZONS
        for L_x in LX_GRID
        for L_y in LY_GRID
    ]
    started = time.time()
    completed = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize,
        initargs=(args.raw_dir,),
    ) as executor:
        futures = {executor.submit(_task, item): item for item in specifications}
        for future in as_completed(futures):
            item = futures[future]
            path, payload = future.result()
            completed += 1
            print(
                f"R3A {completed}/{len(specifications)} "
                f"h={item[2]} Lx={item[0]} Ly={item[1]} "
                f"status={payload['status']} "
                f"loss={float(payload['validation_loss']):.6g} "
                f"elapsed={float(payload['elapsed_seconds']):.1f}s "
                f"path={path}",
                flush=True,
            )
    result = _aggregate(output)
    result["elapsed_seconds"] = time.time() - started
    _write(output / "R3A_HISTORY_SELECTION.json", result)
    _write(
        output / "R3A_STATUS.json",
        {
            "stage": "R3A_NATIVE_HISTORY_SELECTION",
            "status": "COMPLETED",
            "completed_candidates": len(specifications),
            "elapsed_seconds": result["elapsed_seconds"],
            "next_stage": result["next_stage"],
        },
    )


if __name__ == "__main__":
    main()
