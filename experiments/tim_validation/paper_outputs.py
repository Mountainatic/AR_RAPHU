"""Create audit-friendly paper tables and static figures from E3--E6 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


CANONICAL_TASK = {
    "TEP_G12": "TEP_H0",
    "CZ_R1_TO_R2": "CZ_H4",
    "CZ_R2_TO_R1": "CZ_H4",
}
UNIVERSES = ("coarse", "standard", "expanded")


def _task(value: Any) -> str:
    return CANONICAL_TASK.get(str(value), str(value))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NOT_RUN"}])


def _manifest_status(directory: Path, names: tuple[str, ...]) -> str:
    for name in names:
        matches = sorted(directory.rglob(name)) if directory.is_dir() else []
        for path in matches:
            value = json.loads(path.read_text(encoding="utf-8"))
            status = str(value.get("status", ""))
            if status in {"COMPLETED", "PARTIAL", "NOT_RUN", "PROTOCOL_BLOCKED", "INVALID"}:
                return status
    return "NOT_RUN"


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    return float(value)


def _average(values: Iterable[Any]) -> float | None:
    numeric = [value for item in values if (value := _number(item)) is not None]
    return mean(numeric) if numeric else None


def _e3_table(root: Path) -> list[dict[str, Any]]:
    aggregate = _read_csv(root / "E3_MULTISCALE_BUDGET" / "aggregate.csv")
    efficiency = _read_csv(root / "E3_MULTISCALE_BUDGET" / "efficiency.csv")
    metrics = {(_task(row["task"]), row["arm"]): row for row in aggregate}
    budget = {(_task(row["task"]), row["arm"]): row for row in efficiency}
    rows = []
    for task in sorted({key[0] for key in metrics}):
        uniform = metrics.get((task, "uniform"))
        scale = metrics.get((task, "multiscale"))
        if not uniform or not scale:
            continue
        uniform_rmse = float(uniform["rmse"])
        scale_rmse = float(scale["rmse"])
        rows.append(
            {
                "status": "COMPLETED",
                "task": task,
                "uniform_RMSE": uniform_rmse,
                "scale_aware_RMSE": scale_rmse,
                "relative_gain": (uniform_rmse - scale_rmse) / uniform_rmse,
                "uniform_candidate_fits": budget.get((task, "uniform"), {}).get(
                    "candidate_fit_attempts"
                ),
                "scale_aware_candidate_fits": budget.get(
                    (task, "multiscale"), {}
                ).get("candidate_fit_attempts"),
                "budget_status": budget.get((task, "multiscale"), {}).get(
                    "budget_status"
                ),
            }
        )
    return rows


def _e4_table(root: Path) -> list[dict[str, Any]]:
    base = root / "E4_CANDIDATE_SENSITIVITY"
    aggregate = _read_csv(base / "aggregate.csv")
    channels = _read_csv(base / "channel_agreement.csv")
    scales = _read_csv(base / "scale_agreement.csv")
    stages = _read_csv(base / "stage_agreement.csv")
    metrics: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in aggregate:
        metrics[(_task(row["task"]), row["universe"])].append(row)
    rows = []
    for task in sorted({key[0] for key in metrics}):
        row: dict[str, Any] = {"status": "COMPLETED", "task": task}
        for universe in UNIVERSES:
            selected = metrics.get((task, universe), [])
            row[f"{universe}_RMSE_mean_across_rods"] = _average(
                item.get("RMSE") for item in selected
            )
            row[f"{universe}_R2_mean_across_rods"] = _average(
                item.get("R2") for item in selected
            )
        for other in ("coarse", "expanded"):
            comparison = f"standard_vs_{other}"
            row[f"channel_jaccard_{comparison}"] = _average(
                value.get("jaccard")
                for value in channels
                if (_task(value["task"]), value["comparison"]) == (task, comparison)
            )
            row[f"scale_agreement_{comparison}"] = _average(
                value.get("scale_agreement")
                for value in scales
                if (_task(value["task"]), value["comparison"]) == (task, comparison)
            )
            row[f"stage_agreement_{comparison}"] = _average(
                value.get("stage_agreement")
                for value in stages
                if (_task(value["task"]), value["comparison"]) == (task, comparison)
            )
        rows.append(row)
    return rows


def _e5_table(root: Path) -> list[dict[str, Any]]:
    base = root / "E5_STRUCTURAL_STABILITY"
    admission = _read_csv(base / "channel_admission.csv")
    jaccard = _read_csv(base / "pairwise_jaccard.csv")
    scales = _read_csv(base / "scale_agreement.csv")
    stages = _read_csv(base / "stage_agreement.csv")
    entropy = _read_csv(base / "selection_entropy.csv")
    rashomon = _read_csv(base / "rashomon_models.csv")
    tasks = sorted({_task(row["task"]) for row in admission + jaccard + stages})
    rows = []
    for task in tasks:
        task_admission = [row for row in admission if _task(row["task"]) == task]
        probabilities = [_number(row.get("admission_probability")) for row in task_admission]
        probabilities = [value for value in probabilities if value is not None]
        summaries = [
            row for row in jaccard
            if _task(row["task"]) == task and row.get("agreement_type") == "PAIRWISE_SUMMARY"
        ]
        task_rashomon = [row for row in rashomon if _task(row["task"]) == task]
        near = [row for row in task_rashomon if row.get("within_1_percent_of_best") == "True"]
        stage_vectors = {row.get("stage_vector") for row in near}
        rows.append(
            {
                "status": "COMPLETED",
                "task": task,
                "mean_channel_admission_probability": _average(probabilities),
                "stable_channel_fraction_p_le_0.1_or_ge_0.9": (
                    sum(value <= 0.1 or value >= 0.9 for value in probabilities)
                    / len(probabilities)
                    if probabilities else None
                ),
                "pairwise_jaccard_mean": _average(row.get("jaccard") for row in summaries),
                "conditional_scale_agreement_mean": _average(
                    row.get("conditional_scale_agreement")
                    for row in scales if _task(row["task"]) == task
                ),
                "stage_agreement_mean": _average(
                    row.get("stage_agreement")
                    for row in stages if _task(row["task"]) == task
                ),
                "normalized_selection_entropy_mean": _average(
                    row.get("normalized_entropy")
                    for row in entropy if _task(row["task"]) == task
                ),
                "rashomon_status": (
                    "PRESENT" if len(near) >= 2 and len(stage_vectors) >= 2 else "NOT_DETECTED"
                ),
                "near_optimal_run_count": len(near),
            }
        )
    return rows


def _e6_table(root: Path) -> list[dict[str, Any]]:
    base = root / "E6_MEASUREMENT_ROBUSTNESS"
    aggregate = _read_csv(base / "N1" / "aggregate.csv") + _read_csv(
        base / "N2" / "aggregate.csv"
    )
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in aggregate:
        key = tuple(
            str(row.get(name, ""))
            for name in (
                "task", "rod", "mode", "perturbation", "measurement_scope",
                "information_set", "availability_scenario", "alpha", "direction",
            )
        )
        grouped.setdefault(key, {})[str(row.get("metric"))] = str(row.get("mean"))
    scale_rows = _read_csv(base / "N2" / "scale_stability.csv")
    scale_groups: dict[tuple[str, ...], list[Any]] = defaultdict(list)
    for row in scale_rows:
        key = tuple(
            str(row.get(name, ""))
            for name in (
                "task", "rod", "perturbation", "measurement_scope",
                "information_set", "availability_scenario", "alpha",
            )
        )
        scale_groups[key].append(row.get("scale_agreement"))
    rows = []
    for key, metrics in sorted(grouped.items()):
        task, rod, mode, perturbation, scope, information, availability, alpha, direction = key
        scale_key = (task, rod, perturbation, scope, information, availability, alpha)
        if mode == "N2" and scale_key in scale_groups:
            metrics["scale_agreement"] = _average(scale_groups[scale_key])
        rows.append(
            {
                "status": "COMPLETED",
                "task": _task(task),
                "rod": rod,
                "mode": mode,
                "perturbation": perturbation,
                "measurement_scope": scope,
                "information_set": information,
                "availability_scenario": availability,
                "level": alpha,
                "direction": direction,
                "prediction_relative_RMSE_degradation": metrics.get(
                    "relative_RMSE_degradation"
                ),
                "R2_change": metrics.get("R2_change"),
                "channel_jaccard": metrics.get("channel_jaccard"),
                "scale_agreement": metrics.get("scale_agreement"),
                "stage_agreement": metrics.get("stage_agreement"),
            }
        )
    return rows


def _save_figure(fig: Any, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=220, bbox_inches="tight")


def _figures(root: Path, tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "figures_for_paper"
    created: list[str] = []
    colors = {"uniform": "#6B7280", "scale-aware": "#2563EB"}

    e3 = tables["E3"]
    if e3:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for row in e3:
            for arm, fit_key, rmse_key in (
                ("uniform", "uniform_candidate_fits", "uniform_RMSE"),
                ("scale-aware", "scale_aware_candidate_fits", "scale_aware_RMSE"),
            ):
                ax.scatter(float(row[fit_key]), float(row[rmse_key]), color=colors[arm], s=55)
                ax.annotate(str(row["task"]), (float(row[fit_key]), float(row[rmse_key])), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(title="E3 performance versus candidate-fit budget", xlabel="Actual candidate fits", ylabel="Test RMSE")
        ax.grid(axis="both", color="#E5E7EB", linewidth=0.8)
        _save_figure(fig, output, "figure_e3_budget_vs_performance")
        plt.close(fig)
        created.append("figure_e3_budget_vs_performance")

    e4 = tables["E4"]
    if e4:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
        x = range(len(e4))
        for universe, marker, color in (
            ("coarse", "o", "#D97706"), ("standard", "s", "#2563EB"),
            ("expanded", "^", "#6B7280"),
        ):
            axes[0].plot(x, [row[f"{universe}_RMSE_mean_across_rods"] for row in e4], marker=marker, color=color, label=universe)
        axes[0].set(title="E4 prediction by candidate universe", ylabel="Test RMSE", xticks=list(x), xticklabels=[row["task"] for row in e4])
        for metric, marker, color in (
            ("channel_jaccard_standard_vs_expanded", "o", "#2563EB"),
            ("stage_agreement_standard_vs_expanded", "s", "#D97706"),
        ):
            axes[1].plot(x, [row.get(metric) for row in e4], marker=marker, color=color, label=metric.split("_standard")[0])
        axes[1].set(title="E4 structure agreement", ylabel="Agreement", ylim=(-0.05, 1.05), xticks=list(x), xticklabels=[row["task"] for row in e4])
        for ax in axes:
            ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
            ax.legend(frameon=False)
        _save_figure(fig, output, "figure_e4_candidate_sensitivity")
        plt.close(fig)
        created.append("figure_e4_candidate_sensitivity")

    e5_admission = _read_csv(root / "E5_STRUCTURAL_STABILITY" / "channel_admission.csv")
    if e5_admission:
        tasks = sorted({_task(row["task"]) for row in e5_admission})
        channels = sorted({row["channel"] for row in e5_admission})
        values = []
        for task in tasks:
            line = []
            for channel in channels:
                selected = [
                    row.get("admission_probability") for row in e5_admission
                    if _task(row["task"]) == task and row["channel"] == channel
                ]
                line.append(_average(selected) if selected else float("nan"))
            values.append(line)
        fig, ax = plt.subplots(figsize=(max(7.5, len(channels) * 0.35), max(3.0, len(tasks) * 0.45)))
        image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        ax.set(title="E5 channel admission probability", xticks=range(len(channels)), xticklabels=channels, yticks=range(len(tasks)), yticklabels=tasks)
        ax.tick_params(axis="x", labelrotation=75, labelsize=7)
        fig.colorbar(image, ax=ax, label="Admission probability")
        _save_figure(fig, output, "figure_e5_channel_stability")
        plt.close(fig)
        created.append("figure_e5_channel_stability")

    e6 = tables["E6"]
    prediction = [row for row in e6 if _number(row.get("prediction_relative_RMSE_degradation")) is not None]
    if prediction:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        series: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in prediction:
            series[(row["task"], row["mode"], row["measurement_scope"], row["information_set"], row["rod"])].append(row)
        for key, values in sorted(series.items()):
            values.sort(key=lambda row: float(row["level"]))
            label = "/".join(part for part in key if part and part != "NA")
            ax.plot([float(row["level"]) for row in values], [100 * float(row["prediction_relative_RMSE_degradation"]) for row in values], marker="o", label=label)
        ax.axhline(0.0, color="#374151", linewidth=0.9)
        ax.set(title="E6 perturbation strength versus prediction degradation", xlabel="Normalized perturbation level alpha", ylabel="Relative RMSE degradation (%)")
        ax.grid(axis="both", color="#E5E7EB", linewidth=0.8)
        ax.legend(frameon=False, fontsize=7)
        _save_figure(fig, output, "figure_e6_prediction_robustness")
        plt.close(fig)
        created.append("figure_e6_prediction_robustness")
    structure = [
        row for row in e6
        if row["mode"] == "N2" and _number(row.get("channel_jaccard")) is not None
    ]
    if structure:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in structure:
            series[(row["task"], row["rod"])].append(row)
        for key, values in sorted(series.items()):
            values.sort(key=lambda row: float(row["level"]))
            label = "/".join(part for part in key if part)
            ax.plot([float(row["level"]) for row in values], [float(row["channel_jaccard"]) for row in values], marker="o", label=f"{label} channel")
            ax.plot([float(row["level"]) for row in values], [float(row["stage_agreement"]) for row in values], marker="s", linestyle="--", label=f"{label} stage")
        ax.set(title="E6 N2 structural agreement", xlabel="Normalized perturbation level alpha", ylabel="Agreement versus alpha=0", ylim=(-0.05, 1.05))
        ax.grid(axis="both", color="#E5E7EB", linewidth=0.8)
        ax.legend(frameon=False, fontsize=7)
        _save_figure(fig, output, "figure_e6_n2_structure_robustness")
        plt.close(fig)
        created.append("figure_e6_n2_structure_robustness")
    return created


def build(root: Path) -> dict[str, Any]:
    root = root.resolve()
    tables = {
        "E3": _e3_table(root),
        "E4": _e4_table(root),
        "E5": _e5_table(root),
        "E6": _e6_table(root),
    }
    output = root / "tables_for_paper"
    for name, rows in tables.items():
        _write_csv(output / f"table_{name.lower()}.csv", rows)
    figures = _figures(root, tables)
    component_status = {
        "E3": _manifest_status(
            root / "E3_MULTISCALE_BUDGET", ("MANIFEST.json",)
        ),
        "E4": _manifest_status(
            root / "E4_CANDIDATE_SENSITIVITY", ("E4_REPORT_MANIFEST.json",)
        ),
        "E5": _manifest_status(
            root / "E5_STRUCTURAL_STABILITY", ("E5_REPORT_MANIFEST.json",)
        ),
        "E6": _manifest_status(
            root / "E6_MEASUREMENT_ROBUSTNESS", ("E6_REPORT_MANIFEST.json",)
        ),
    }
    statuses = set(component_status.values())
    overall = (
        "COMPLETED" if statuses == {"COMPLETED"}
        else "NOT_RUN" if statuses == {"NOT_RUN"}
        else "INVALID" if "INVALID" in statuses
        else "PARTIAL"
    )
    manifest = {
        "status": overall,
        "component_status": component_status,
        "table_rows": {name: len(rows) for name, rows in tables.items()},
        "figures": figures,
        "source_root": str(root),
    }
    (root / "PAPER_OUTPUTS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
