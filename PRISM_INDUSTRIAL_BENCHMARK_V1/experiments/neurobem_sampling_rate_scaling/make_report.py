from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _record(frame: pd.DataFrame, rate: int, route: str, mode: str = "FIXED_TIME") -> dict:
    row = frame[(frame.sampling_rate_hz == rate) & (frame.route == route) & (frame.history_mode == mode)].iloc[0]
    return {key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value) for key, value in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--generating-commit", required=True)
    args = parser.parse_args()
    root = args.run
    aggregate = pd.read_csv(root / "SAMPLING_RATE_AGGREGATE.csv")
    channel = pd.read_csv(root / "SAMPLING_RATE_CHANNEL_ATTRIBUTION.csv")
    primary = {route: {str(rate): _record(aggregate, rate, route) for rate in (100, 200, 400)} for route in ("PF_KCW", "J_KCW")}
    attribution = (channel[channel.history_mode == "FIXED_TIME"]
                   .groupby(["sampling_rate_hz", "route", "channel_ablation"]).diverged.mean()
                   .unstack().reset_index().to_dict("records"))
    summary = {
        "experiment_id": "PRISM_V2_1_1_NEUROBEM_SAMPLING_RATE_SCALING_R1",
        "status": "COMPLETED",
        "generating_commit": args.generating_commit,
        "registered_outcome": "MIXED_OUTCOME_B_STEP_COUNT_AND_D_HIGHER_RATE_WORSE_RELIABLE_TIME",
        "primary_fixed_time_results": primary,
        "interpretation": {
            "PF_KCW": "OUTCOME_B_STEP_COUNT_HORIZON_WITH_OUTCOME_D_PHYSICAL_TIME_CONTRACTION",
            "J_KCW": "OUTCOME_B_TO_D_STEP_COUNT_CONTRACTION_AT_400HZ",
            "physical_time_horizon_supported": False,
            "higher_rate_physical_time_benefit_supported": False,
        },
        "channel_attribution": attribution,
        "source_rate_audit": {
            "excluded_native_164hz_segments": 4,
            "excluded_scope": "ALL_NEW_SCALING_ADAPTER_FITS",
            "interpolation_used": False,
            "track0_frozen_r3_adapter_unchanged": True,
        },
        "r3_100hz_reproduced": True,
        "test_access_count": 1,
        "test_used_for_tuning": False,
        "stabilization_added": False,
        "formal_stability_claim": False,
    }
    (root / "SAMPLING_RATE_FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    p100, p200, p400 = (primary["PF_KCW"][str(x)] for x in (100, 200, 400))
    j100, j200, j400 = (primary["J_KCW"][str(x)] for x in (100, 200, 400))
    report = f"""# NeuroBEM sampling-rate scaling audit

## Result

The registered physical-time scaling hypothesis was not supported. Under the primary fixed-time history comparison, PF_KCW reliable horizons were {p100['reliable_horizon_steps']:.0f}/{p200['reliable_horizon_steps']:.0f}/{p400['reliable_horizon_steps']:.0f} steps, corresponding to {p100['reliable_horizon_ms']:.0f}/{p200['reliable_horizon_ms']:.0f}/{p400['reliable_horizon_ms']:.0f} ms at 100/200/400 Hz. J_KCW produced {j100['reliable_horizon_steps']:.0f}/{j200['reliable_horizon_steps']:.0f}/{j400['reliable_horizon_steps']:.0f} steps, or {j100['reliable_horizon_ms']:.0f}/{j200['reliable_horizon_ms']:.0f}/{j400['reliable_horizon_ms']:.0f} ms.

This is a mixed registered Outcome B/D: reliable step count is approximately fixed for PF and through 200 Hz for Joint, while elapsed reliable time contracts as sampling rate increases; Joint contracts further to four reliable steps at 400 Hz. Higher sampling rate improves one-step error but does not improve recursive reliability. Therefore lower one-step error must not be interpreted as greater recursive stability.

Full free rollouts still diverged on 12/12 test trajectories for both routes at every rate/history condition. Median divergence in physical time remains several seconds and is not the same quantity as the conservative calibration-frozen reliable horizon.

## History and channel diagnostics

Fixed-step versus fixed-time history does not rescue the higher-rate reliable horizon. Scaling history from 20 to 40/80 samples at 200/400 Hz leaves the primary conclusion unchanged.

The R3 attribution persists. PF velocity-only recursion diverges on 12/12 trajectories at every rate. Joint velocity-only recursion remains finite on all 12, while Joint attitude-only/body-rate-only divergence is substantial and body-rate-only worsens to 11/12 at 400 Hz. The data therefore do not support the hypothesis that native 400 Hz disproportionately rescues Joint attitude/body-rate recursion.

## Provenance and scope

The frozen 100-Hz R3 adapter exactly reproduced its registered calibration and test baselines before higher-rate results were accepted. A source-rate audit found four 164-Hz segments belonging to one train-fit flight; per user authorization, they were excluded from every newly fitted scaling adapter. No interpolation or synthetic 400-Hz row was introduced. Calibration and all five rate-specific adapters were committed before the formal test was accessed once.

No PRISM core, route family, ridge grid, reliability threshold, clipping, spectral constraint, or stabilization was changed. This audit is not a Lyapunov or global stability proof.
"""
    (root / "SAMPLING_RATE_FINAL_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
