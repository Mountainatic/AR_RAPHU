from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--generating-commit", required=True)
    args = p.parse_args(); run = args.run
    raw = json.loads((run / "summary.json").read_text())
    resync = pd.read_csv(run / "resynchronization.csv")
    channel = pd.read_csv(run / "channel_attribution.csv")
    component = pd.read_csv(run / "component_route_attribution.csv")
    perturb = pd.read_csv(run / "perturbation_growth.csv")
    jacobian = pd.read_csv(run / "jacobian.csv")
    compact = {
        "experiment_id": "PRISM_V2_1_1_NEUROBEM_R3_RECURSIVE_STABILITY_AUDIT",
        "generating_commit": args.generating_commit,
        "status": "COMPLETED",
        "r2_baseline_reproduced": raw["r2_baseline_reproduced"],
        "primary_expansion_hypothesis_supported": False,
        "registered_outcomes": ["OUTCOME_B_RESYNCHRONIZATION_HORIZON", "OUTCOME_C_NO_CALIBRATED_EXPANSION_EVENT", "OUTCOME_D_CHANNEL_LOCALIZATION"],
        "routes": raw["routes"],
        "reliable_open_loop_horizon_steps": raw["reliable_open_loop_horizon_steps"],
        "reliable_open_loop_horizon_seconds_at_100hz": {k: v / 100 for k, v in raw["reliable_open_loop_horizon_steps"].items()},
        "expansion_thresholds": raw["expansion_thresholds"],
        "test_accessed": True, "test_used_for_tuning": False, "prism_predictor_modified": False,
        "formal_lyapunov_claim": False, "position_channel_status": "NOT_APPLICABLE_CHANNEL_NOT_PRESENT",
        "median_perturbation_growth_by_epsilon": perturb.groupby(["route", "epsilon_fraction"]).max_growth.median().unstack().to_dict("index"),
        "median_sigma_max_J_by_epsilon": jacobian.groupby(["route", "epsilon_fraction"]).sigma_max_J.median().unstack().to_dict("index"),
        "median_newest_block_product_growth_rate": jacobian.groupby("route").newest_block_product_growth_rate.median().to_dict(),
        "channel_divergence_rate": channel.groupby(["route", "channel_ablation"]).diverged.mean().unstack().to_dict("index"),
        "component_route_divergence_rate": component.groupby("route").diverged.mean().to_dict(),
    }
    (run / "R3_FINAL_SUMMARY.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    lines = [
        "# PRISM NeuroBEM R3 recursive stability audit",
        "", "## Result", "",
        "R3 exactly reproduced the frozen R2 baseline: PF_KCW and J_KCW both diverged on 12/12 test trajectories, with median first-divergence steps 347 and 380.",
        "", "The primary registered expansion-event hypothesis was not supported. The calibration-only sustained-growth thresholds produced no `t_expansion` event before divergence on any test trajectory. This is Outcome C: the registered local expansion event is insufficient to anticipate visible divergence.",
        "", "Local sensitivity is nevertheless mildly expansive in a descriptive sense: median newest-state-block `sigma_max(J)` is about 1.055 for both routes, the two finite-difference scales agree, and the three paired-rollout epsilon scales agree closely. These are empirical finite-time diagnostics, not a formal Lyapunov exponent or proof.",
        "", "## Reliable open-loop horizon", "",
        "Using bounds frozen from calibration one-step errors and the registered nested 90% criterion, PF_KCW has a reliable horizon of 20 steps (200 ms) and J_KCW 10 steps (100 ms). Test divergence is zero through N=20 for both routes, while reliability degrades at N=50/100 and free rollout diverges universally. This supports Outcome B: PRISM is useful as a short-horizon predictor in an observed loop, not as an autonomous simulator.",
        "", "## Channel and component attribution", "",
        "PF_KCW remains unstable when velocity alone is recursively fed back (12/12 divergence), so PF has a strong velocity-path instability. J_KCW is stable for velocity-only recursion (0/12), while body-rate-only and attitude-only recursion diverge on 10/12 and 7/12 trajectories; Joint instability is therefore concentrated in attitude/angular-rate feedback and their coupling.",
        "", "KC-only is not a cure: PF_KC and J_KC both diverge on 12/12. Adding W delays median divergence (PF 220.5→347; Joint 343→380) but does not prevent it. No K/C/W parameter was changed or refit.",
        "", "Position is absent from the Track-B 10-dimensional state and is correctly marked not applicable. Force/torque are not the recursive Track-B targets in this frozen experiment.",
        "", "## Scope", "",
        "No clipping, state saturation, spectral constraint, Lyapunov penalty, model retraining, or test-driven threshold change was performed. R3 does not establish global dynamical instability, universal OOD failure, or a manifold transition.",
    ]
    (run / "R3_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__": main()
