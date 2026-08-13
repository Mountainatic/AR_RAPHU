from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    table = pd.read_csv(args.run / "per_trajectory.csv")
    out = args.run / "figures"
    out.mkdir(exist_ok=True)

    summary = table.groupby(["route", "ablation"], as_index=False).agg(
        divergence_rate=("diverged", "mean"), alarm_rate=("t_alarm", lambda x: x.notna().mean())
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for route, part in summary.groupby("route"):
        axes[0].plot(part.ablation, part.divergence_rate, "o-", label=route)
        axes[1].plot(part.ablation, part.alarm_rate, "o-", label=route)
    for axis, title in zip(axes, ("Catastrophic divergence rate", "Causal alarm rate")):
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=60)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "aggregate_ablation_rates.png", dpi=150)
    plt.close(fig)

    static = table[table.ablation.eq("static")].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    for route, part in static.groupby("route"):
        ax.plot(part.trajectory, part.t_diverge, "o-", label=route)
    ax.set_ylabel("First divergence step (100 Hz)")
    ax.tick_params(axis="x", rotation=70)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "static_divergence_times.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
