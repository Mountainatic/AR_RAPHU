#!/usr/bin/env python3
"""Produce interpretation-safe kernel and rank artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np

from ar_raphu.cz_real.frozen import load_frozen_horizon
from ar_raphu.cz_real.protocol import PRIMARY_INPUTS
from ar_raphu.orss.diagnostics import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.results_root)
    output = root / "interpretability"
    status = output / "INTERPRETABILITY_STATUS.json"
    if args.resume and status.exists():
        existing = json.loads(status.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            print("INTERPRETABILITY_RESUMED")
            return
    locked = root / "frozen_model"
    manifest = json.loads(
        (locked / "FROZEN_MODEL_MANIFEST.json").read_text(encoding="utf-8")
    )
    rank = json.loads(
        (root / "R3D" / "rank_selection.json").read_text(encoding="utf-8")
    )
    channels = list(PRIMARY_INPUTS) + ["历史晶体直径"]
    summary: dict[str, object] = {}
    output.mkdir(parents=True, exist_ok=True)
    for horizon, row in manifest["models"].items():
        model = load_frozen_horizon(locked / row["path"])
        coefficients = (
            model["coefficients"]
            .numpy()
            .reshape(len(channels), int(model["M_tau"]), int(model["M_x"]))
        )
        np.savez_compressed(
            output / f"h_{int(horizon):03d}_kernel_coefficients.npz",
            coefficients=coefficients,
            channel_names=np.asarray(channels),
        )
        per_channel = []
        for index, name in enumerate(channels):
            surface = coefficients[index]
            singular = np.linalg.svd(surface, compute_uv=False)
            per_channel.append(
                {
                    "channel": name,
                    "support_active": True,
                    "coefficient_frobenius_norm": float(
                        np.linalg.norm(surface)
                    ),
                    "lag_profile_l2": [
                        float(value)
                        for value in np.linalg.norm(surface, axis=1)
                    ],
                    "amplitude_profile_l2": [
                        float(value)
                        for value in np.linalg.norm(surface, axis=0)
                    ],
                    "coefficient_coordinate_singular_values": [
                        float(value) for value in singular
                    ],
                    "weak_operator_status": "NOT_CLASSIFIED_WITHOUT_PREDECLARED_THRESHOLD",
                }
            )
        summary[horizon] = {
            "channels": per_channel,
            "predictive_rank": rank["profiles"][horizon][
                "predictive_rank_by_budget"
            ],
            "structural_rank_status": rank["profiles"][horizon][
                "structural_rank_status"
            ],
            "K_level_interpretation": "K_LEVEL_NOT_IDENTIFIED",
            "cross_furnace_kernel_comparison": {
                "status": "FROZEN_OPERATOR_UNCHANGED_IN_ZERO_SHOT_AND_CALIBRATION",
                "leading_mode_correlation": 1.0,
                "principal_angle_degrees": 0.0,
                "lag_peak_shift_basis_indices": 0,
                "amplitude_shape_shift": 0.0,
                "note": (
                    "Only intercept/output scale calibration was permitted; "
                    "the frozen kernel was not re-estimated on Furnace B."
                ),
            },
        }
    write_json(
        output / "interpretability_summary.json",
        {
            "schema": "CZ_INTERPRETABILITY_V1",
            "status": "COMPLETED",
            "coordinate_warning": (
                "Coefficient-coordinate profiles are numerical summaries. "
                "Physical K-level claims remain blocked without coercivity "
                "and Schur identification certificates."
            ),
            "models": summary,
            "INTERPRETABILITY_COMPLETE": True,
        },
    )
    write_json(
        status,
        {
            "status": "COMPLETED",
            "INTERPRETABILITY_COMPLETE": True,
            "next_stage": "BLOCK_BOOTSTRAP",
        },
    )


if __name__ == "__main__":
    main()
