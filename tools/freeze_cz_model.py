#!/usr/bin/env python3
"""Refit and hash the development-frozen CZ ORSS model family."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch

from ar_raphu.cz_real.frozen import (
    fit_frozen_horizon,
    save_frozen_horizon,
)
from ar_raphu.cz_real.protocol import load_furnace_a
from ar_raphu.orss.diagnostics import write_json
from ar_raphu.orss.penalties import PenaltyWeights


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _furnace_a(raw_dir: Path) -> Path:
    candidates = [path for path in raw_dir.iterdir() if path.name.endswith("1.xlsx")]
    if len(candidates) != 1:
        raise RuntimeError("Furnace-A path is not uniquely identifiable.")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results/cz_real_data/complete_5090",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--raw-dir",
        default="/root/OPS_UOI_WORKSPACE/data/private/cz_real_v1/raw",
    )
    parser.add_argument(
        "--config", default="configs/cz_real_data/orss_5090.yaml"
    )
    args = parser.parse_args()
    results = Path(args.results_root)
    continuation = json.loads(
        (results / "R3C" / "continuation_selection.json").read_text(
            encoding="utf-8"
        )
    )
    rank = json.loads(
        (results / "R3D" / "rank_selection.json").read_text(encoding="utf-8")
    )
    if continuation["status"] != "COMPLETED" or rank["status"] != "COMPLETED":
        raise RuntimeError("R3_NOT_COMPLETE")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(config["runtime"]["device"])
    data = load_furnace_a(_furnace_a(Path(args.raw_dir)))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows: dict[str, object] = {}
    for horizon_text, frozen in continuation["selections"].items():
        horizon = int(horizon_text)
        payload = fit_frozen_horizon(
            data.inputs,
            data.target,
            horizon=horizon,
            L_x=int(frozen["history"]["L_x"]),
            L_y=int(frozen["history"]["L_y"]),
            M_tau=int(frozen["resolution"]["M_tau"]),
            M_x=int(frozen["resolution"]["M_x"]),
            normalized_weights=PenaltyWeights(**frozen["penalty"]),
            c_rho=float(
                frozen["selected_CONTINUATION_SCALE_COEFFICIENT"]
            ),
            predictive_rank=rank["profiles"][horizon_text][
                "predictive_rank_by_budget"
            ],
            device=device,
            chunk_time=int(config["cuda"]["operator_chunk_time"]),
            maximum_iterations=int(config["krylov"]["maximum_iterations"]),
        )
        destination = output / f"h_{horizon:03d}.pt"
        save_frozen_horizon(destination, payload)
        rows[horizon_text] = {
            "path": destination.name,
            "sha256": _digest(destination),
            "relative_kkt_residual": payload["relative_kkt_residual"],
            "history": {
                "L_x": payload["L_x"],
                "L_y": payload["L_y"],
            },
            "resolution": {
                "M_tau": payload["M_tau"],
                "M_x": payload["M_x"],
            },
            "penalty": payload["normalized_weights"],
            "CONTINUATION_SCALE_COEFFICIENT": payload[
                "CONTINUATION_SCALE_COEFFICIENT"
            ],
        }
        print(
            f"FROZEN h={horizon} sha256={rows[horizon_text]['sha256']}",
            flush=True,
        )
        torch.cuda.empty_cache()
    manifest = {
        "schema": "CZ_FROZEN_MODEL_MANIFEST_V1",
        "status": "COMPLETED",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "furnace_A_sha256": data.source_sha256,
        "models": rows,
        "development_only_selection": True,
        "furnace_A_confirmation_accessed": False,
        "furnace_B_access_count": 0,
    }
    write_json(output / "FROZEN_MODEL_MANIFEST.json", manifest)
    write_json(
        output / "FROZEN_MODEL_HASH_WRITTEN.json",
        {
            "FROZEN_MODEL_HASH_WRITTEN": True,
            "models": {
                horizon: row["sha256"] for horizon, row in rows.items()
            },
        },
    )


if __name__ == "__main__":
    main()
