from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROUTES = ("PF_KCW", "J_KCW")


def _primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.history_mode == "FIXED_TIME"].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    out = root / "figures"
    out.mkdir(exist_ok=True)
    aggregate = pd.read_csv(root / "SAMPLING_RATE_AGGREGATE.csv")
    resync = pd.read_csv(root / "SAMPLING_RATE_RESYNCHRONIZATION.csv")
    channel = pd.read_csv(root / "SAMPLING_RATE_CHANNEL_ATTRIBUTION.csv")
    primary = _primary(aggregate)

    for column, filename, ylabel in (
        ("reliable_horizon_steps", "01_reliable_horizon_steps.png", "Reliable horizon (steps)"),
        ("reliable_horizon_ms", "02_reliable_horizon_ms.png", "Reliable horizon (ms)"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for route in ROUTES:
            part = primary[primary.route == route].sort_values("sampling_rate_hz")
            axis.plot(part.sampling_rate_hz, part[column], "o-", label=route)
        axis.set(xlabel="Sampling rate (Hz)", ylabel=ylabel, xticks=[100, 200, 400])
        axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for route in ROUTES:
        part = primary[primary.route == route].sort_values("sampling_rate_hz")
        axes[0].plot(part.sampling_rate_hz, part.median_t_diverge_steps, "o-", label=route)
        axes[1].plot(part.sampling_rate_hz, part.median_t_diverge_ms, "o-", label=route)
    axes[0].set_ylabel("Median divergence (steps)"); axes[1].set_ylabel("Median divergence (ms)")
    for axis in axes: axis.set_xlabel("Sampling rate (Hz)"); axis.set_xticks([100, 200, 400]); axis.grid(alpha=.25)
    axes[0].legend(); fig.tight_layout(); fig.savefig(out / "03_divergence_time.png", dpi=160); plt.close(fig)

    finite = resync[np.isfinite(pd.to_numeric(resync.resync_interval_ms, errors="coerce"))].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for axis, rate in zip(axes, (100, 200, 400)):
        for route in ROUTES:
            part = finite[(finite.history_mode == "FIXED_TIME") & (finite.sampling_rate_hz == rate) & (finite.route == route)]
            curve = part.groupby("resync_interval_ms").diverged.mean().sort_index()
            axis.plot(curve.index, curve.values, "o-", label=route)
        axis.set_title(f"{rate} Hz"); axis.set_xlabel("Resync interval (ms)"); axis.grid(alpha=.25)
    axes[0].set_ylabel("Divergence rate"); axes[0].legend(); fig.tight_layout(); fig.savefig(out / "04_resynchronization_curves.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for mode, marker in (("FIXED_STEP", "o"), ("FIXED_TIME", "s")):
        for route in ROUTES:
            part = aggregate[(aggregate.history_mode == mode) & (aggregate.route == route)].sort_values("sampling_rate_hz")
            axes[0].plot(part.sampling_rate_hz, part.reliable_horizon_steps, marker + "-", label=f"{route} {mode}")
            axes[1].plot(part.sampling_rate_hz, part.reliable_horizon_ms, marker + "-", label=f"{route} {mode}")
    axes[0].set_ylabel("Reliable steps"); axes[1].set_ylabel("Reliable ms")
    for axis in axes: axis.set_xlabel("Sampling rate (Hz)"); axis.set_xticks([100, 200, 400]); axis.grid(alpha=.25)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "05_history_mode_comparison.png", dpi=160); plt.close(fig)

    focus = channel[(channel.history_mode == "FIXED_TIME") & channel.channel_ablation.isin(("velocity_only", "attitude_only", "body_rate_only"))]
    values = focus.groupby(["sampling_rate_hz", "route", "channel_ablation"]).diverged.mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, route in zip(axes, ROUTES):
        for name, part in values[values.route == route].groupby("channel_ablation"):
            axis.plot(part.sampling_rate_hz, part.diverged, "o-", label=name)
        axis.set_title(route); axis.set_xlabel("Sampling rate (Hz)"); axis.set_xticks([100, 200, 400]); axis.grid(alpha=.25); axis.legend(fontsize=8)
    axes[0].set_ylabel("Divergence rate"); fig.tight_layout(); fig.savefig(out / "06_channel_attribution.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
