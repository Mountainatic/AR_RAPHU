#!/usr/bin/env python3
"""Build CPU-style GPU plots and a protocol-matched CPU/GPU comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


CPU_COLOR = "#4472C4"
CPU_EDGE = "#24364B"
GPU_COLOR = "#D9A441"
GPU_EDGE = "#72551E"
DYNAMIC_COLOR = "#70AD47"
RESIDUAL_COLOR = "#C55A11"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def pretty_gpu_name(model_id: str) -> str:
    name = model_id.removeprefix("final__")
    history = ""
    if name.endswith("_uxy"):
        name = name[:-4]
        history = " (U+Y)"
    elif name.endswith("_u"):
        name = name[:-2]
        history = " (U)"
    replacements = {
        "k_residual_tcn": "K-residual TCN",
        "k_residual_gru": "K-residual GRU",
        "k_residual_transformer": "K-residual Transformer",
        "k_residual_mlp": "K-residual MLP",
        "temporal_autoencoder": "Temporal Autoencoder",
        "gru_vae": "GRU-VAE",
        "pyramid_vae": "Pyramid VAE",
        "dmvaer": "DMVAER",
        "static_gat": "Static GAT",
        "static_gcn": "Static GCN",
        "temporal_gat": "Temporal GAT",
        "temporal_gcn": "Temporal GCN",
        "adaptive_graph_kan": "Adaptive Graph KAN",
        "adaptive_graph_mlp": "Adaptive Graph MLP",
        "fixed_graph_kan": "Fixed Graph KAN",
        "no_graph_kan": "No-Graph KAN",
        "akgnn_window": "AKGNN Window",
        "t_akgnn": "T-AKGNN",
        "informer_lite": "Informer-lite",
        "autoformer_lite": "Autoformer-lite",
        "patchtst": "PatchTST",
        "timesnet": "TimesNet",
        "nlinear": "NLinear",
        "dlinear": "DLinear",
        "lstm_sa": "LSTM-SA",
        "lstm": "LSTM",
        "gru": "GRU",
        "tcn": "TCN",
        "mlp": "MLP",
        "transformer": "Transformer",
        "s4d": "S4D",
        "dgdl": "DGDL",
        "narx_mlp": "NARX MLP",
    }
    return replacements.get(name, name.replace("_", " ").title()) + history


def validate_shared_test_identity(cpu_root: Path, gpu_root: Path) -> None:
    cpu_rows = read_csv(cpu_root / "PHYSICS_FIRST_MODELS.csv")
    gpu_rows = read_csv(gpu_root / "GPU_ALL_RUNS.csv")
    for direction in ("sheet1_to_sheet2", "sheet2_to_sheet1"):
        cpu_hashes = {
            row["sample_id_sha256"]
            for row in cpu_rows
            if row["direction"] == direction and row.get("sample_id_sha256")
        }
        gpu_hashes = {
            row["test_sample_id_sha256"]
            for row in gpu_rows
            if row["direction"] == direction and row.get("test_sample_id_sha256")
        }
        cpu_counts = {
            int(row["test_rows"])
            for row in cpu_rows
            if row["direction"] == direction and row.get("test_rows")
        }
        gpu_counts = {
            int(row["evaluation_rows"])
            for row in gpu_rows
            if row["direction"] == direction and row.get("evaluation_rows")
        }
        if len(cpu_hashes) != 1 or cpu_hashes != gpu_hashes:
            raise RuntimeError(
                f"CPU_GPU_SAMPLE_ID_MISMATCH:{direction}:{cpu_hashes}:{gpu_hashes}"
            )
        if len(cpu_counts) != 1 or cpu_counts != gpu_counts:
            raise RuntimeError(
                f"CPU_GPU_SAMPLE_COUNT_MISMATCH:{direction}:{cpu_counts}:{gpu_counts}"
            )


def gpu_screening_rows(
    gpu_summary: list[dict[str, str]], modes: set[str]
) -> list[dict[str, object]]:
    rows = []
    for row in gpu_summary:
        if row["stage"] not in {"core", "frontier"}:
            continue
        if row["mode"] not in modes:
            continue
        if row["implementation_label"].startswith("NONCAUSAL_CONTROL"):
            continue
        if int(row["complete_two_direction_seeds"]) < 5:
            continue
        rows.append(
            {
                "name": pretty_gpu_name(row["model_id"]),
                "model_id": row["model_id"],
                "mode": row["mode"],
                "rmse": float(row["pooled_RMSE_seed_median"]),
            }
        )
    return sorted(rows, key=lambda item: float(item["rmse"]))


def plot_leaderboard(
    rows: list[dict[str, object]],
    *,
    title: str,
    subtitle: str,
    output: Path,
) -> None:
    selected = rows[:10][::-1]
    figure, axis = plt.subplots(figsize=(10, 6.5))
    values = [float(row["rmse"]) for row in selected]
    labels = [str(row["name"]) for row in selected]
    colors = [
        RESIDUAL_COLOR if row.get("mode") == "residual" else CPU_COLOR
        for row in selected
    ]
    edges = [
        GPU_EDGE if row.get("mode") == "residual" else CPU_EDGE
        for row in selected
    ]
    bars = axis.barh(labels, values, color=colors, edgecolor=edges)
    axis.set_xlabel("Pooled RMSE median across seeds (lower is better)")
    axis.set_title(title, loc="left", fontsize=15, fontweight="bold")
    figure.text(0.125, 0.91, subtitle, fontsize=9, color="#555555")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.4f}",
            va="center",
            fontsize=8,
        )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_bootstrap(
    bootstrap_rows: list[dict[str, str]],
    finalist_modes: dict[str, str],
    output: Path,
) -> None:
    rows = [
        row
        for row in bootstrap_rows
        if row["kind"] == "MODEL_VS_PERSISTENCE"
    ]
    rows.sort(key=lambda row: float(row["median"]))
    labels = [pretty_gpu_name(row["model_a"]) for row in rows]
    medians = np.asarray([float(row["median"]) * 100.0 for row in rows])
    lowers = np.asarray([float(row["lower_95"]) * 100.0 for row in rows])
    uppers = np.asarray([float(row["upper_95"]) * 100.0 for row in rows])
    modes = [finalist_modes[row["model_a"]] for row in rows]
    palette = {
        "input": CPU_COLOR,
        "dynamic": DYNAMIC_COLOR,
        "residual": RESIDUAL_COLOR,
    }
    y = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(10, 6.2))
    axis.errorbar(
        medians,
        y,
        xerr=np.vstack([medians - lowers, uppers - medians]),
        fmt="none",
        ecolor="#555555",
        elinewidth=1.4,
        capsize=4,
        zorder=1,
    )
    axis.scatter(
        medians,
        y,
        s=70,
        c=[palette[mode] for mode in modes],
        edgecolor="#24364B",
        linewidth=0.8,
        zorder=2,
    )
    axis.axvline(0.0, color="#333333", linewidth=0.9)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Relative MSE improvement vs persistence (%)")
    axis.set_title(
        "GPU finalist bootstrap against persistence",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.125,
        0.91,
        "Median and 95% block-bootstrap interval; positive is better",
        fontsize=9,
        color="#555555",
    )
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Patch(facecolor=palette["input"], edgecolor=CPU_EDGE, label="Input-only"),
            Patch(facecolor=palette["dynamic"], edgecolor=CPU_EDGE, label="Dynamic"),
            Patch(facecolor=palette["residual"], edgecolor=GPU_EDGE, label="Residual"),
        ],
        frameon=False,
        loc="lower right",
    )
    for index, value in enumerate(medians):
        axis.text(
            value,
            index,
            f"  {value:+.1f}%",
            va="center",
            ha="left",
            fontsize=8,
        )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def combined_rows(
    cpu_rows: list[dict[str, str]],
    gpu_rows: list[dict[str, str]],
    *,
    panel: str,
    cpu_limit: int,
) -> list[dict[str, object]]:
    selected_cpu = sorted(
        [
            row
            for row in cpu_rows
            if row["status"] == "COMPLETED" and row["direction"] == "POOLED"
        ],
        key=lambda row: float(row["RMSE"]),
    )[:cpu_limit]
    output: list[dict[str, object]] = [
        {
            "panel": panel,
            "platform": "CPU",
            "model": row["name"],
            "pooled_RMSE": float(row["RMSE"]),
            "metric": "deterministic pooled RMSE",
        }
        for row in selected_cpu
    ]
    output.extend(
        {
            "panel": panel,
            "platform": "GPU",
            "model": pretty_gpu_name(row["model_id"]),
            "pooled_RMSE": float(row["pooled_RMSE_seed_median"]),
            "metric": "median pooled RMSE across 10 seeds",
        }
        for row in gpu_rows
    )
    return sorted(output, key=lambda row: float(row["pooled_RMSE"]))


def plot_combined(rows: list[dict[str, object]], output: Path) -> None:
    panels = [
        ("Input-only", "No historical diameter"),
        ("Dynamic / residual", "Causal output history or frozen K residual"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(15, 7.4), sharex=True)
    for axis, (panel, subtitle) in zip(axes, panels):
        selected = [row for row in rows if row["panel"] == panel][::-1]
        values = [float(row["pooled_RMSE"]) for row in selected]
        labels = [str(row["model"]) for row in selected]
        colors = [
            CPU_COLOR if row["platform"] == "CPU" else GPU_COLOR
            for row in selected
        ]
        edges = [
            CPU_EDGE if row["platform"] == "CPU" else GPU_EDGE
            for row in selected
        ]
        bars = axis.barh(labels, values, color=colors, edgecolor=edges)
        axis.set_title(f"{panel}\n{subtitle}", loc="left", fontsize=12)
        axis.set_xlabel("Pooled RMSE (lower is better)")
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:.4f}",
                va="center",
                fontsize=8,
            )
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "CPU and GPU models on the same bidirectional L6 test targets",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.925,
        "CPU values are deterministic; GPU values are medians across 10 confirmed seeds",
        fontsize=9,
        color="#555555",
    )
    figure.legend(
        handles=[
            Patch(facecolor=CPU_COLOR, edgecolor=CPU_EDGE, label="CPU model"),
            Patch(facecolor=GPU_COLOR, edgecolor=GPU_EDGE, label="GPU finalist"),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.97),
    )
    figure.tight_layout(rect=(0.04, 0, 1, 0.9), w_pad=5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_report(
    gpu_root: Path,
    combined: list[dict[str, object]],
    output: Path,
) -> None:
    finalists = read_csv(gpu_root / "GPU_FINALISTS.csv")
    best_input = min(
        (row for row in finalists if row["mode"] == "input"),
        key=lambda row: float(row["pooled_RMSE_seed_median"]),
    )
    best_dynamic = min(
        (row for row in finalists if row["mode"] == "dynamic"),
        key=lambda row: float(row["pooled_RMSE_seed_median"]),
    )
    lines = [
        "# GPU and CPU/GPU Comparison Figures",
        "",
        "All comparison panels use the same two cross-rod L6 test target sets. "
        "The plotting script verifies their sample-ID hashes and row counts "
        "before combining results.",
        "",
        "## GPU input-only screening",
        "",
        "![GPU input-only leaderboard](plots/gpu_input_driven_leaderboard.png)",
        "",
        "## GPU dynamic and residual screening",
        "",
        "![GPU dynamic leaderboard](plots/gpu_dynamic_identification_leaderboard.png)",
        "",
        "## Confirmed finalist bootstrap",
        "",
        "![GPU finalist bootstrap](plots/gpu_finalist_bootstrap.png)",
        "",
        "## CPU and GPU comparison",
        "",
        "![CPU and GPU combined leaderboard](plots/cpu_gpu_combined_leaderboard.png)",
        "",
        "CPU bars are deterministic FP64 benchmark results. GPU bars in the "
        "combined figure are the medians of the 10-seed FP32 finalist "
        "confirmation runs. Models are compared only within the input-only "
        "or dynamic/residual panel.",
        "",
        f"Best confirmed GPU input-only model: "
        f"`{pretty_gpu_name(best_input['model_id'])}` "
        f"(pooled RMSE `{float(best_input['pooled_RMSE_seed_median']):.4f}`).",
        "",
        f"Best confirmed GPU dynamic model: "
        f"`{pretty_gpu_name(best_dynamic['model_id'])}` "
        f"(pooled RMSE `{float(best_dynamic['pooled_RMSE_seed_median']):.4f}`).",
        "",
        "The full comparison data are in `CPU_GPU_COMBINED_LEADERBOARD.csv`.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-results", required=True)
    parser.add_argument("--gpu-results", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cpu_root = Path(args.cpu_results).resolve()
    gpu_root = Path(args.gpu_results).resolve()
    plots = gpu_root / "plots"
    validate_shared_test_identity(cpu_root, gpu_root)

    gpu_summary = read_csv(gpu_root / "GPU_MODEL_SUMMARY.csv")
    input_screening = gpu_screening_rows(gpu_summary, {"input"})
    dynamic_screening = gpu_screening_rows(gpu_summary, {"dynamic", "residual"})
    plot_leaderboard(
        input_screening,
        title="GPU input-driven soft-sensor benchmark",
        subtitle="No historical diameter; median over 5 seeds; pooled bidirectional L6 evaluation",
        output=plots / "gpu_input_driven_leaderboard.png",
    )
    plot_leaderboard(
        dynamic_screening,
        title="GPU dynamic system-identification benchmark",
        subtitle="Causal output history or K residual; median over 5 seeds; pooled L6 evaluation",
        output=plots / "gpu_dynamic_identification_leaderboard.png",
    )

    finalists = read_csv(gpu_root / "GPU_FINALISTS.csv")
    finalist_modes = {row["model_id"]: row["mode"] for row in finalists}
    plot_bootstrap(
        read_csv(gpu_root / "ABLATIONS" / "FINALIST_BOOTSTRAP.csv"),
        finalist_modes,
        plots / "gpu_finalist_bootstrap.png",
    )

    input_finalists = [row for row in finalists if row["mode"] == "input"]
    dynamic_finalists = [
        row for row in finalists if row["mode"] in {"dynamic", "residual"}
    ]
    combined = combined_rows(
        read_csv(cpu_root / "INPUT_DRIVEN_LEADERBOARD.csv"),
        input_finalists,
        panel="Input-only",
        cpu_limit=5,
    )
    combined.extend(
        combined_rows(
            read_csv(cpu_root / "DYNAMIC_IDENTIFICATION_LEADERBOARD.csv"),
            dynamic_finalists,
            panel="Dynamic / residual",
            cpu_limit=5,
        )
    )
    plot_combined(combined, plots / "cpu_gpu_combined_leaderboard.png")
    write_csv(gpu_root / "CPU_GPU_COMBINED_LEADERBOARD.csv", combined)
    build_report(
        gpu_root,
        combined,
        gpu_root / "GPU_CPU_COMPARISON_REPORT.md",
    )
    manifest = {
        "status": "PASS",
        "sample_identity_validation": "PASS",
        "plots": [
            "plots/gpu_input_driven_leaderboard.png",
            "plots/gpu_dynamic_identification_leaderboard.png",
            "plots/gpu_finalist_bootstrap.png",
            "plots/cpu_gpu_combined_leaderboard.png",
        ],
        "comparison_rows": len(combined),
    }
    (gpu_root / "PLOT_BUILD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("GPU_CPU_PLOTS=" + json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
