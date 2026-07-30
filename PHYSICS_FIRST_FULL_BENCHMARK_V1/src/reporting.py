"""Technical report and static evidence figures for the CPU benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _plot_leaderboard(
    rows: list[dict[str, str]],
    *,
    title: str,
    subtitle: str,
    output: Path,
) -> None:
    pooled = [
        row
        for row in rows
        if row.get("direction") == "POOLED"
        and row.get("status") == "COMPLETED"
        and row.get("RMSE")
    ]
    pooled.sort(key=lambda row: float(row["RMSE"]))
    pooled = pooled[:10][::-1]
    figure, axis = plt.subplots(figsize=(10, 6.5))
    values = [float(row["RMSE"]) for row in pooled]
    labels = [row["name"] for row in pooled]
    bars = axis.barh(labels, values, color="#4472C4", edgecolor="#24364B")
    axis.set_xlabel("Pooled RMSE (lower is better)")
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


def _plot_physics(audit: dict[str, Any], output: Path) -> None:
    labels = list(audit["directions"]) + ["POOLED"]
    g_k = [
        audit["directions"][name]["G_K"] for name in audit["directions"]
    ] + [audit["pooled"]["G_K"]]
    g_ar = [
        audit["directions"][name]["G_AR_given_K"]
        for name in audit["directions"]
    ] + [audit["pooled"]["G_AR_given_K"]]
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(9, 5.4))
    bars_k = axis.bar(
        x - width / 2,
        np.asarray(g_k) * 100.0,
        width,
        label="Physical K gain",
        color="#4472C4",
        edgecolor="#24364B",
    )
    bars_ar = axis.bar(
        x + width / 2,
        np.asarray(g_ar) * 100.0,
        width,
        label="Residual AR gain given K",
        color="#D9A441",
        edgecolor="#72551E",
    )
    axis.axhline(0.0, color="#333333", linewidth=0.9)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Relative MSE improvement (%)")
    axis.set_title(
        "Physics-first gain decomposition",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.125,
        0.91,
        "L6, two cross-rod directions and pooled evaluation",
        fontsize=9,
        color="#555555",
    )
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axis.set_axisbelow(True)
    for bars in (bars_k, bars_ar):
        for bar in bars:
            value = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:+.1f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_report(results_root: Path) -> Path:
    decision = json.loads(
        (results_root / "CPU_FINAL_DECISION.json").read_text(encoding="utf-8")
    )
    audit = decision["formal_primary_model"]["physics_audit"]
    input_rows = _read_csv(results_root / "INPUT_DRIVEN_LEADERBOARD.csv")
    dynamic_rows = _read_csv(
        results_root / "DYNAMIC_IDENTIFICATION_LEADERBOARD.csv"
    )
    plots = results_root / "plots"
    _plot_leaderboard(
        input_rows,
        title="Input-driven soft-sensor benchmark",
        subtitle="No historical diameter; pooled bidirectional L6 evaluation",
        output=plots / "input_driven_leaderboard.png",
    )
    _plot_leaderboard(
        dynamic_rows,
        title="Dynamic system-identification benchmark",
        subtitle="Causal historical output is allowed; pooled bidirectional L6 evaluation",
        output=plots / "dynamic_identification_leaderboard.png",
    )
    _plot_physics(audit, plots / "physics_gain_decomposition.png")
    primary = decision["formal_primary_model"]
    pooled = audit["pooled"]
    input_winner = decision["input_driven_winner"]
    dynamic_winner = decision["dynamic_identification_winner"]
    lines = [
        "# Physics-First K → Residual-AR CPU Benchmark",
        "",
        "## Technical summary",
        "",
        f"The frozen L6 benchmark completed with status "
        f"`{decision['status']}`. The formal physics-first model achieved pooled "
        f"RMSE `{primary['pooled_RMSE']:.6f}` and pooled R² "
        f"`{primary['pooled_R2']:.4f}`. Physical K reduced pooled persistence "
        f"MSE by `{pooled['G_K']:.2%}`; the strictly matured residual AR changed "
        f"the remaining K error by `{pooled['G_AR_given_K']:.2%}`. Total pooled "
        f"gain was `{pooled['G_total']:.2%}`.",
        "",
        f"The input-only winner was `{input_winner['name']}` "
        f"(MSE `{input_winner['MSE']:.6g}`), while the dynamic leaderboard winner "
        f"was `{dynamic_winner['name']}` (MSE `{dynamic_winner['MSE']:.6g}`). "
        "These are separate rankings because only the dynamic leaderboard may "
        "read historical diameter.",
        "",
        "## The physical layer and residual layer make distinct contributions",
        "",
        "The figure separates the improvement produced by the frozen joint-lift "
        "kernel from the incremental improvement of residual AR. Residual AR only "
        "reads errors whose 20-minute horizon and 2-minute target window have "
        "fully matured before the current prediction origin.",
        "",
        "![Physics-first gain decomposition](plots/physics_gain_decomposition.png)",
        "",
        f"The pooled physics attribution ratio is `{pooled['rho_phys']}`. "
        f"The nonlinear K block remained `{decision['nonlinear_K']}`; this run "
        "therefore supports a linear amplitude subspace rather than a forced "
        "two-dimensional Urysohn surface.",
        "",
        "## Input-only models are ranked without historical diameter",
        "",
        "This leaderboard uses only the four registered controls represented by "
        "the same causal multiresolution blocks. It therefore measures soft-sensor "
        "performance rather than output persistence.",
        "",
        "![Input-driven leaderboard](plots/input_driven_leaderboard.png)",
        "",
        "## Dynamic models are evaluated on the identical final sample mask",
        "",
        "AR, ARX, adapted classical identification models, NARX controls, and the "
        "physics-first structures use historical output only up to the current "
        "origin. Every reported pooled metric uses the shared evaluation mask "
        "that leaves enough time for residual maturity and the maximum 40-minute "
        "residual history.",
        "",
        "![Dynamic identification leaderboard](plots/dynamic_identification_leaderboard.png)",
        "",
        "## Scope, data and metric definitions",
        "",
        "- Data: the two registered workbook sheets, each analyzed only after its "
        "last frozen diameter breakpoint.",
        "- Target: future 2-minute mean diameter at +20 minutes minus the current "
        "2-minute mean.",
        "- Cadence/history: 10 seconds and 40 minutes.",
        "- Outer validation: Sheet1→Sheet2 and Sheet2→Sheet1.",
        "- Inner selection: four expanding-window folds with at least 22 minutes "
        "of purge; test rods never select hyperparameters.",
        "- `G_K` compares K-only with zero-change persistence; `G_AR|K` compares "
        "K→Residual-AR with the frozen K-only prediction.",
        "",
        "## Model specification and validation",
        "",
        "K is the train-fitted joint-lift PC1 multiresolution linear Urysohn "
        "subspace. K is fitted first and frozen. Rolling cross-fit predictions "
        "create OOF physical residuals. Residual AR is then selected with an exact "
        "zero candidate and cannot backpropagate into or refit K. CPU FP64 is used "
        "for K, residual AR, Gram systems, KKT, predictions, metrics and bootstrap.",
        "",
        f"Maximum certified KKT residual was "
        f"`{decision['FP64_certification']['KKT_max']:.3e}` and maximum recorded "
        f"condition number was "
        f"`{decision['FP64_certification']['condition_number_max']:.3e}`. "
        f"FP64 certification status: "
        f"`{decision['FP64_certification']['status']}`.",
        "",
        "Methods without a reliable paper-faithful Python implementation are "
        "explicitly labeled `ADAPTED_IMPLEMENTATION` in the result registry. "
        "They remain comparison controls and are not presented as full original-"
        "paper reproductions.",
        "",
        "## Limitations, uncertainty and robustness",
        "",
        f"The K bootstrap 95% interval is "
        f"`[{pooled['bootstrap_K_vs_persistence']['lower_95']:.2%}, "
        f"{pooled['bootstrap_K_vs_persistence']['upper_95']:.2%}]`; the residual "
        f"AR-given-K interval is "
        f"`[{pooled['bootstrap_AR_given_K']['lower_95']:.2%}, "
        f"{pooled['bootstrap_AR_given_K']['upper_95']:.2%}]`. Direction-specific, "
        "first/second-half, common-support, OOD, kernel, and time-shift placebo "
        "results are retained in `BOOTSTRAP/physics_first.json`.",
        "",
        "The evidence is limited to two rods and the frozen L6 task. It does not "
        "establish cross-furnace, cross-stage, or unrestricted industrial "
        "generalization. Adapted classical models should be interpreted as "
        "equation-level controlled baselines.",
        "",
        "## Recommended next steps",
        "",
        "1. Transfer the immutable shared dataset bundle to the GPU batch; do not "
        "regenerate PCA, targets, masks, or splits.",
        "2. Merge CPU and GPU results only after all protocol, target, split and "
        "sample-ID hashes match.",
        "3. Preserve K-only as the physical reference and enable residual dynamics "
        "only when the registered bidirectional bootstrap evidence is positive.",
        "",
        "## Further questions",
        "",
        "- Does the same frozen lift kernel remain stable on additional rods and "
        "furnace campaigns?",
        "- Do GPU sequence models improve the dynamic leaderboard without reducing "
        "the stable physical K contribution?",
        "- Can process metadata distinguish true transport delay from controller "
        "and measurement delay?",
        "",
    ]
    output = results_root / "CPU_FINAL_REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
