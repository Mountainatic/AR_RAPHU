from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _r2_static_logs(root: Path) -> pd.DataFrame:
    frames = []
    for route in ("PF_KCW", "J_KCW"):
        for path in sorted((root / "trajectories").glob(f"*__{route}__static.csv")):
            value = pd.read_csv(path)
            value["route"] = route
            value["trajectory_id"] = path.name.split(".csv__", 1)[0] + ".csv"
            frames.append(value)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r3-run", type=Path, required=True)
    parser.add_argument("--r2-run", type=Path, required=True)
    args = parser.parse_args()
    out = args.r3_run / "figures"; out.mkdir(exist_ok=True)
    logs = _r2_static_logs(args.r2_run)
    perturb = pd.read_csv(args.r3_run / "perturbation_growth.csv")
    jacobian = pd.read_csv(args.r3_run / "jacobian.csv")
    resync = pd.read_csv(args.r3_run / "resynchronization.csv")
    channel = pd.read_csv(args.r3_run / "channel_attribution.csv")
    summary = json.loads((args.r3_run / "summary.json").read_text())

    representative = "ellipse.csv"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, route in zip(axes, ("PF_KCW", "J_KCW")):
        log = logs[(logs.route == route) & (logs.trajectory_id == representative)]
        growth = perturb[(perturb.route == route) & (perturb.trajectory_id == representative)].groupby("anchor_step").max_growth.max()
        axis.semilogy(log.step, log.velocity_error.clip(lower=1e-12), label="velocity rollout error")
        twin = axis.twinx(); twin.plot(growth.index, growth.values, "C1o-", label="max 20-step perturbation growth")
        twin.axhline(summary["expansion_thresholds"][route], color="C1", ls="--", label="calibration threshold")
        div = int(log.loc[log.velocity_error.notna(), "step"].iloc[-1]) if log.empty else summary["routes"][route]["median_t_diverge"]
        actual = pd.read_csv(args.r3_run / "baseline.csv")
        actual = actual[(actual.route == route) & (actual.trajectory_id == representative)].baseline_t_diverge.iloc[0]
        axis.axvline(actual, color="red", ls=":", label="baseline divergence")
        axis.set_ylabel(f"{route} error"); twin.set_ylabel("growth")
        lines = axis.lines + twin.lines; axis.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=8)
    axes[-1].set_xlabel("100-Hz step")
    fig.tight_layout(); fig.savefig(out / "A_expansion_vs_divergence.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, route in zip(axes, ("PF_KCW", "J_KCW")):
        j = jacobian[(jacobian.route == route) & (jacobian.trajectory_id == representative) & (jacobian.epsilon_fraction == jacobian.epsilon_fraction.min())]
        log = logs[(logs.route == route) & (logs.trajectory_id == representative)]
        axis.plot(j.anchor_step, j.sigma_max_J, label="newest-state block sigma_max")
        axis.plot(j.anchor_step, j.newest_block_product_amplification, label="5-step newest-block product")
        twin = axis.twinx(); twin.semilogy(log.step, log.velocity_error.clip(lower=1e-12), color="C2", alpha=.7, label="velocity error")
        axis.set_ylabel(route); twin.set_ylabel("error")
        lines = axis.lines + twin.lines; axis.legend(lines, [line.get_label() for line in lines], fontsize=8)
    axes[-1].set_xlabel("100-Hz step")
    fig.tight_layout(); fig.savefig(out / "B_jacobian_vs_rollout.png", dpi=150); plt.close(fig)

    order = [1, 5, 10, 20, 50, 100, np.inf]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for route, part in resync.groupby("route"):
        aggregate = part.groupby("resync_interval").agg(divergence=("diverged", "mean"), velocity=("velocity_error", "median"), attitude=("attitude_error", "median"))
        x = np.arange(len(order)); aggregate = aggregate.reindex(order)
        axes[0].plot(x, aggregate.divergence, "o-", label=route)
        axes[1].semilogy(x, aggregate.velocity, "o-", label=route)
        axes[2].plot(x, aggregate.attitude, "o-", label=route)
    for axis, title in zip(axes, ("Divergence rate", "Median velocity error", "Median attitude error")):
        axis.set_title(title); axis.set_xticks(range(len(order)), ["1", "5", "10", "20", "50", "100", "inf"]); axis.set_xlabel("resync interval N"); axis.grid(alpha=.25)
    axes[0].legend(); fig.tight_layout(); fig.savefig(out / "C_resynchronization_sweep.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, route in zip(axes, ("PF_KCW", "J_KCW")):
        part = logs[logs.route == route]
        aggregate = part.groupby("step").velocity_error.agg(median="median", q10=lambda x: x.quantile(.1), q90=lambda x: x.quantile(.9))
        x = aggregate.index.to_numpy(dtype=float)
        median = aggregate["median"].clip(lower=1e-12).to_numpy(dtype=float)
        q10 = aggregate.q10.clip(lower=1e-12).to_numpy(dtype=float)
        q90 = aggregate.q90.to_numpy(dtype=float)
        axis.semilogy(x, median, label="median")
        axis.fill_between(x, q10, q90, alpha=.25, label="10-90%")
        axis.axvline(summary["reliable_open_loop_horizon_steps"][route], color="green", ls="--", label="frozen reliable horizon")
        axis.set_ylabel(route); axis.legend()
    axes[-1].set_xlabel("open-loop step"); fig.tight_layout(); fig.savefig(out / "D_error_horizon.png", dpi=150); plt.close(fig)

    aggregate = channel.groupby(["route", "channel_ablation"], as_index=False).diverged.mean()
    pivot = aggregate.pivot(index="channel_ablation", columns="route", values="diverged")
    pivot.plot.bar(figsize=(10, 4)); plt.ylabel("divergence rate"); plt.xticks(rotation=35, ha="right"); plt.grid(axis="y", alpha=.25)
    plt.tight_layout(); plt.savefig(out / "E_channel_attribution.png", dpi=150); plt.close()


if __name__ == "__main__": main()
