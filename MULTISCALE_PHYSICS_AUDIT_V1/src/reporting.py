"""Final Markdown report assembled strictly from machine-readable results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_final_report(root: Path) -> Path:
    results = root / "results"
    stage1 = _read_csv(results / "STAGE1_SCALE_SCAN.csv")
    stage2 = _read_csv(results / "STAGE2_CONFIRMATION.csv")
    decision_path = results / "FINAL_DECISION.json"
    decision = (
        json.loads(decision_path.read_text(encoding="utf-8"))
        if decision_path.is_file()
        else {"status": "NOT_YET_RUN"}
    )
    lines = [
        "# MULTISCALE-PHYSICS-AUDIT V1 Final Report",
        "",
        "## Answer first",
        "",
        f"- Pipeline status: `{decision.get('status', 'NOT_YET_RUN')}`.",
        f"- Stage 1 completed profile-variants: `{len(stage1)}`.",
        f"- Stage 2 confirmed linear structure: "
        f"`{len(decision.get('confirmatory_linear_structure_tasks', []))}`.",
        f"- AR-conditional gains: "
        f"`{len(decision.get('AR_conditional_gain_tasks', []))}`.",
        f"- Stable nonlinear K gains: "
        f"`{len(decision.get('nonlinear_K_gain_tasks', []))}`.",
        "",
        "The raw workbook is excluded from source control and the return bundle. "
        "All claims below concern two provided rods and the registered stable "
        "segments only.",
        "",
        "## Stage 1 scale candidates",
        "",
        "| Task | Channel | Horizon min | Window min | History min | Pooled Q gain | S1 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in stage1:
        lines.append(
            f"| {row['task_id']} | {row['channel']} | {row['horizon_min']} | "
            f"{row['target_window_min']} | {row['history_min']} | "
            f"{float(row['pooled_q_improvement']):.3%} | {row['S1_status']} |"
        )
    lines += [
        "",
        "## Stage 2 structural and conditional evidence",
        "",
        "| Task | Q gain | Bootstrap P(>0) | Kernel corr | S2 | AR-conditional |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in stage2:
        if not row.get("task_id"):
            continue
        lines.append(
            f"| {row['task_id']} | {float(row['pooled_q_improvement']):.3%} | "
            f"{float(row['bootstrap_positive_probability']):.3f} | "
            f"{float(row['kernel_correlation']):.3f} | "
            f"{row['S2_stable_structure']} | {row['C_AR_conditional_gain']} |"
        )
    lines += [
        "",
        "## Decision boundaries",
        "",
        "- `structure evidence` means bidirectional cross-rod improvement plus "
        "the predeclared bootstrap, kernel-correlation, support-overlap, and "
        "common-support gates.",
        "- `AR-conditional gain` compares frozen scale-matched AR plus Q against "
        "the same frozen AR.",
        "- `nonlinear K gain` is only evaluated after Stage 2 and retains an "
        "exact zero nonlinear block.",
        "- The 180-minute heater profile is exploratory and cannot become a "
        "confirmatory result.",
        f"- Combined model: `{decision.get('combined_model_status', 'NOT_YET_RUN')}`. "
        f"{decision.get('combined_model_note', '')}",
        "",
        "## Reproducibility",
        "",
        f"- Config SHA256: `{decision.get('config_sha256', 'NOT_AVAILABLE')}`",
        f"- Data SHA256: `{decision.get('data_sha256', 'NOT_AVAILABLE')}`",
        "- Linear algebra certification path: CPU FP64.",
        "- Profile failures are isolated and recorded rather than aborting the run.",
        "",
    ]
    output = results / "FINAL_REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
