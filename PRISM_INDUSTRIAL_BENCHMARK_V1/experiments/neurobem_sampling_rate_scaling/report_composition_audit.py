from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--generating-commit", required=True)
    args = p.parse_args()
    root = args.run
    aggregate = pd.read_csv(root / "COMPOSITION_AGGREGATE.csv")
    one = pd.read_csv(root / "ONE_STEP_AGGREGATE.csv")
    raw = json.loads((root / "COMPOSITION_SUMMARY.json").read_text())

    core = aggregate[[
        "route", "horizon_ms", "sampling_rate_hz",
        "gt_full_state_error_median", "gt_velocity_error_median",
        "gt_attitude_error_median", "gt_body_rate_error_median",
        "defect_vs_100_full_state_median", "defect_vs_100_velocity_median",
        "defect_vs_100_attitude_median", "defect_vs_100_body_rate_median",
        "finite_fraction",
    ]].copy()
    core.to_csv(root / "COMPOSITION_CORE_TABLE.csv", index=False)

    out = root / "figures"; out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for route, marker in (("PF_KCW", "o"), ("J_KCW", "s")):
        part = one[one.route == route].sort_values("sampling_rate_hz")
        axes[0].semilogy(part.sampling_rate_hz, part.one_step_mse, marker + "-", label=route)
        part = aggregate[(aggregate.route == route) & (aggregate.horizon_ms == 100)].sort_values("sampling_rate_hz")
        axes[1].plot(part.sampling_rate_hz, part.gt_full_state_error_median, marker + "-", label=route)
    axes[0].set_ylabel("Mean one-step MSE"); axes[1].set_ylabel("Median 100-ms composition GT error")
    for axis in axes: axis.set_xlabel("Sampling rate (Hz)"); axis.set_xticks([100, 200, 400]); axis.grid(alpha=.25); axis.legend()
    fig.suptitle("One-step accuracy versus fixed-time composition")
    fig.tight_layout(); fig.savefig(out / "01_one_step_vs_composition.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, route in zip(axes, ("PF_KCW", "J_KCW")):
        for rate in (100, 200, 400):
            part = aggregate[(aggregate.route == route) & (aggregate.sampling_rate_hz == rate)]
            axis.plot(part.horizon_ms, part.gt_full_state_error_median, "o-", label=f"{rate} Hz")
        axis.set_title(route); axis.set_xlabel("Physical horizon (ms)"); axis.set_ylabel("Median GT composition error"); axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(out / "02_gt_composition_by_horizon.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, route in zip(axes, ("PF_KCW", "J_KCW")):
        for rate in (200, 400):
            part = aggregate[(aggregate.route == route) & (aggregate.sampling_rate_hz == rate)]
            axis.plot(part.horizon_ms, part.defect_vs_100_full_state_median, "o-", label=f"{rate} vs 100 Hz")
        axis.set_title(route); axis.set_xlabel("Physical horizon (ms)"); axis.set_ylabel("Median cross-rate defect"); axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(out / "03_cross_rate_defect.png", dpi=180); plt.close(fig)

    compact = {
        "experiment_id": "PRISM_V2_1_1_NEUROBEM_COMPOSITION_CONSISTENCY_AUDIT_R1",
        "status": "COMPLETED",
        "generating_commit": args.generating_commit,
        "final_decision": "COMPOSITION_CONSISTENCY_MIXED",
        "one_step_order_400_lt_200_lt_100": {route: bool(raw["routes"][route]["one_step_order_supported"]) for route in ("PF_KCW", "J_KCW")},
        "route_100ms_classes": {route: raw["routes"][route]["composition_100ms_class"] for route in ("PF_KCW", "J_KCW")},
        "test_access_count": 1, "test_used_for_tuning": False,
        "model_retrained": False, "prism_core_modified": False, "stabilization_added": False,
        "r3_r4_reproduction_gate": raw["r3_r4_reproduction_gate"],
        "core_table": core.to_dict("records"),
    }
    (root / "COMPOSITION_FINAL_SUMMARY.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    report = """# NeuroBEM composition-consistency audit

## Decision

`COMPOSITION_CONSISTENCY_MIXED`.

For both PF_KCW and J_KCW, mean one-step MSE strictly improves with sampling
rate: 400 Hz < 200 Hz < 100 Hz. Cross-rate terminal prediction defect relative
to the 100-Hz map increases with rate separation and physical horizon, directly
showing that the frozen discrete maps are not exact semigroup-equivalent
representations of one common learned flow.

The stronger registered failure pattern—GT composition error strictly ordered
400 Hz > 200 Hz > 100 Hz at 100 ms—is not supported uniformly. At 100 ms PF
improves slightly with rate, while Joint is non-monotone. At 500 ms PF worsens
with rate, whereas Joint does not. The scientifically correct result is thus
mixed: measurable cross-rate composition defect is supported, but a universal
higher-rate composition failure is not.

PF velocity error and Joint attitude/body-rate errors are reported separately
in `COMPOSITION_CORE_TABLE.csv`. All evaluated endpoints through 500 ms were
finite on the formal test anchors.

## Contract

All rates share the same native-400-Hz measured initial and terminal states at
each physical timestamp. Frozen fixed-time history adapters (20/40/80 samples)
and frozen sampling operators are reused without refitting. The R3/R4
reproduction artifacts pass their registered value and SHA gate. Four 164-Hz
training segments remain excluded. Test was accessed once after the composition
calibration freeze. No clipping, stabilization, multi-step training, threshold
change, or PRISM-core modification was performed.
"""
    (root / "COMPOSITION_FINAL_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
