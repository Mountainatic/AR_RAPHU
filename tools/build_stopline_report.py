#!/usr/bin/env python3
"""Build the reproducible stop-line evidence summary and Markdown report."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


SCENARIO_ROOTS = {
    "AR-S0": "results/phase1/E1_AR-S0_G2/Track-XAR",
    "AR-S1": "results/phase1/E2_AR-S1_G2/Track-XAR",
    "AR-S2": "results/phase1/E3_AR-S2_G2/Track-XAR",
    "AR-S3": "results/phase1/E4_AR-S3_G2/Track-XAR",
    **{
        scenario: (
            f"results/phase1/SUPPORT_{scenario}_{scenario}_G2/Track-XAR"
        )
        for scenario in ("AR-S4", "AR-S5", "AR-S6", "AR-S7")
    },
}


def read_json(relative: str | Path) -> dict[str, Any]:
    path = PROJECT_ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"Required result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def support_metrics(
    rows: list[dict[str, Any]], truth: set[int], *, variable_count: int = 10
) -> dict[str, float | int]:
    true_hits = false_hits = true_total = false_total = 0
    for row in rows:
        selected = set(int(value) for value in row["terminal_support"])
        true_hits += len(selected & truth)
        false_hits += len(selected - truth)
        true_total += len(truth)
        false_total += variable_count - len(truth)
    return {
        "true_hits": true_hits,
        "true_total": true_total,
        "support_recall": true_hits / true_total if true_total else 1.0,
        "false_hits": false_hits,
        "false_total": false_total,
        "support_false_positive_rate": (
            false_hits / false_total if false_total else 0.0
        ),
    }


def model_row(label: str, relative: str) -> dict[str, Any]:
    payload = read_json(relative)
    rows = payload["per_seed"]
    return {
        "model": label,
        "mean_rmse": mean([float(row["rmse"]) for row in rows]),
        "mean_r2": mean([float(row["r2"]) for row in rows]),
        "seed_count": len(rows),
        "source": relative,
    }


def main() -> int:
    output_root = (
        PROJECT_ROOT / "generated" / "AR_RAPHU_STOPLINE_20260725"
    )
    stop_record = read_json(
        "results/runtime/STOPPED_BY_USER_20260725.json"
    )
    scenario_rows = []
    for scenario, root in SCENARIO_ROOTS.items():
        metrics = read_json(f"{root}/test_metrics.json")
        selection = read_json(f"{root}/validation_selection.json")
        per_seed = metrics["per_seed"]
        truth = set(
            int(value)
            for value in metrics["truth_audit_after_selection"][
                "true_support_by_scenario"
            ]
        )
        support = support_metrics(per_seed, truth)
        critical_path = (
            PROJECT_ROOT
            / root
            / "critical30_validation"
            / "validation_selection.json"
        )
        critical = (
            json.loads(critical_path.read_text(encoding="utf-8"))
            if critical_path.is_file()
            else None
        )
        scenario_rows.append(
            {
                "scenario": scenario,
                "screening_status": metrics["status"],
                "screening_seed_count": len(per_seed),
                "screening_selected_scale": float(metrics["selected_scale"]),
                "critical30_status": (
                    "COMPLETED"
                    if critical is not None
                    else (
                        "NOT_APPLICABLE"
                        if scenario in {"AR-S4", "AR-S5", "AR-S6", "AR-S7"}
                        else "NOT_YET_RUN"
                    )
                ),
                "critical30_selected_scale": (
                    float(critical["config_id"].split("=")[1])
                    if critical is not None
                    else None
                ),
                "mean_test_rmse": mean(
                    [float(row["rmse"]) for row in per_seed]
                ),
                "mean_test_r2": mean(
                    [float(row["r2"]) for row in per_seed]
                ),
                "true_support": sorted(truth),
                **support,
                "source": f"{root}/test_metrics.json",
            }
        )

    model_rows = [
        model_row(
            "E2 M5 Scheme A",
            "results/phase1/E2_AR-S1_G2/Track-XAR/test_metrics.json",
        ),
        model_row(
            "E2 M7 convex spline",
            "results/phase1/E2_AR-S1_G2/M7/test_metrics.json",
        ),
        model_row(
            "E2 M8 orthogonal surface",
            "results/phase1/E2_AR-S1_G2/M8/test_metrics.json",
        ),
        model_row(
            "E3 M5 Scheme A",
            "results/phase1/E3_AR-S2_G2/Track-XAR/test_metrics.json",
        ),
        model_row(
            "E3 M6 free rank-1 kernel",
            "results/phase1/E3_AR-S2_G2/M6/test_metrics.json",
        ),
    ]
    model_lookup = {row["model"]: row for row in model_rows}
    comparisons = [
        {
            "comparison": "E2 M7 vs M5",
            "relative_rmse_gain": (
                model_lookup["E2 M5 Scheme A"]["mean_rmse"]
                - model_lookup["E2 M7 convex spline"]["mean_rmse"]
            )
            / model_lookup["E2 M5 Scheme A"]["mean_rmse"],
        },
        {
            "comparison": "E2 M8 vs M7",
            "relative_rmse_gain": (
                model_lookup["E2 M7 convex spline"]["mean_rmse"]
                - model_lookup["E2 M8 orthogonal surface"]["mean_rmse"]
            )
            / model_lookup["E2 M7 convex spline"]["mean_rmse"],
        },
        {
            "comparison": "E3 M6 vs M5",
            "relative_rmse_gain": (
                model_lookup["E3 M5 Scheme A"]["mean_rmse"]
                - model_lookup["E3 M6 free rank-1 kernel"]["mean_rmse"]
            )
            / model_lookup["E3 M5 Scheme A"]["mean_rmse"],
        },
    ]
    bootstrap_path = (
        PROJECT_ROOT
        / "results/phase1/E2_AR-S1_G2/M8/bootstrap_rank_audit.json"
    )
    bootstrap = (
        json.loads(bootstrap_path.read_text(encoding="utf-8"))
        if bootstrap_path.is_file()
        else {
            "status": "NOT_YET_RUN",
            "replicates_per_seed": 0,
            "seed_count": 0,
            "global_rejection_count": 0,
            "global_false_positive_rate": None,
            "variable_rejection_count": 0,
            "variable_false_positive_rate": None,
            "rank1_false_positive_gate_passed": False,
            "test_partition_accessed": False,
        }
    )
    config = read_json("configs/protocol_v2.yaml")
    fp_limit = float(config["statistics"]["support_false_positive_threshold"])
    s1 = next(row for row in scenario_rows if row["scenario"] == "AR-S1")
    s3 = next(row for row in scenario_rows if row["scenario"] == "AR-S3")
    gates = [
        {
            "gate": "AR-S1 support false-positive rate",
            "observed": s1["support_false_positive_rate"],
            "threshold": fp_limit,
            "passed": s1["support_false_positive_rate"] <= fp_limit,
        },
        {
            "gate": "AR-S3 support recall",
            "observed": s3["support_recall"],
            "threshold": 0.80,
            "passed": s3["support_recall"] >= 0.80,
        },
    ]
    phase1_gate_passed = (
        all(bool(row["passed"]) for row in gates)
        and bootstrap["status"] == "COMPLETED"
        and bool(bootstrap["rank1_false_positive_gate_passed"])
    )
    phases = [
        {
            "phase": "Phase 0 protocol and implementation",
            "status": "COMPLETED",
            "reason": "Frozen v2 semantics, tests, leakage guards, and runtime.",
        },
        {
            "phase": "Phase 1 synthetic",
            "status": "FAILED",
            "reason": (
                "AR-S0 through AR-S7 screening completed, but support gates "
                "failed; critical-30 and bootstrap were stopped by the user."
            ),
        },
        {
            "phase": "M9 rank-2 and M10 full surface upgrade",
            "status": "NOT_APPLICABLE",
            "reason": "No external variable passed all frozen upgrade gates.",
        },
        {
            "phase": "Phase 2 TEP",
            "status": "NOT_YET_RUN",
            "reason": "Stopped before Phase 2 because the Phase 1 gate did not pass.",
        },
        {
            "phase": "Phase 3 Debutanizer and Gas Turbine",
            "status": "NOT_YET_RUN",
            "reason": "Phase 2 has not been completed.",
        },
        {
            "phase": "Phase 4 private CZ",
            "status": "NOT_APPLICABLE",
            "reason": "Excluded by user on 2026-07-25; private data was not read.",
        },
        {
            "phase": "Phase 5 multi-rod CZ",
            "status": "NOT_APPLICABLE",
            "reason": "Excluded by user and no additional rods are in scope.",
        },
    ]
    summary = {
        "title": "AR-RAPHU v2 stop-line validation report",
        "generated_date": "2026-07-25",
        "generator_version": 2,
        "privacy": {
            "private_CZ_status": "NOT_APPLICABLE",
            "private_CZ_accessed": False,
        },
        "scenario_results": scenario_rows,
        "model_results": model_rows,
        "model_comparisons": comparisons,
        "rank_bootstrap": {
            key: bootstrap[key]
            for key in (
                "status",
                "replicates_per_seed",
                "seed_count",
                "global_rejection_count",
                "global_false_positive_rate",
                "variable_rejection_count",
                "variable_false_positive_rate",
            "rank1_false_positive_gate_passed",
            "test_partition_accessed",
            )
        },
        "early_stop": {
            "status": "COMPLETED",
            "requested_by_user": True,
            "AR-S0_critical_added_warmup_completed": 20,
            "AR-S0_critical_added_forks_completed": 55,
            "AR-S0_critical_added_forks_planned": 180,
            "partial_critical_results_used_for_conclusions": False,
            "bootstrap_status": "NOT_YET_RUN",
            "actual_logged_optimizer_epochs": stop_record[
                "actual_logged_optimizer_epochs"
            ],
            "epochs_by_stage": stop_record["epochs_by_stage"],
        },
        "gates": gates,
        "phase1_gate_passed": phase1_gate_passed,
        "phases": phases,
        "conclusion_boundary": (
            "Current evidence supports stable rank-1 prediction components only; "
            "it does not support a rank-2/full-Urysohn upgrade or Phase-2 start."
        ),
    }
    atomic_json(output_root / "evidence_summary.json", summary)

    csv_path = output_root / "scenario_evidence.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "screening_status",
                "screening_seed_count",
                "screening_selected_scale",
                "critical30_status",
                "critical30_selected_scale",
                "mean_test_rmse",
                "mean_test_r2",
                "support_recall",
                "support_false_positive_rate",
                "source",
            ],
        )
        writer.writeheader()
        for row in scenario_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    gate_lines = "\n".join(
        f"| {row['gate']} | {row['observed']:.4f} | "
        f"{row['threshold']:.4f} | "
        f"{'PASS' if row['passed'] else 'FAIL'} |"
        for row in gates
    )
    scenario_lines = "\n".join(
        f"| {row['scenario']} | {row['mean_test_rmse']:.4f} | "
        f"{row['mean_test_r2']:.4f} | {row['support_recall']:.3f} | "
        f"{row['support_false_positive_rate']:.3f} | "
        f"{row['screening_selected_scale']:.3g} | "
        f"{row['critical30_selected_scale'] if row['critical30_selected_scale'] is not None else 'N/A'} |"
        for row in scenario_rows
    )
    comparison_lines = "\n".join(
        f"| {row['comparison']} | {row['relative_rmse_gain'] * 100:.2f}% |"
        for row in comparisons
    )
    phase_lines = "\n".join(
        f"| {row['phase']} | {row['status']} | {row['reason']} |"
        for row in phases
    )
    report = f"""# AR-RAPHU v2 停止线验证报告

## 技术摘要

按用户指令提前停止后，Phase 0 与 AR-S0–AR-S7 Scheme A screening 已完成。累计记录 {stop_record['actual_logged_optimizer_epochs']:,} 个实际优化 epoch。AR-S0 critical-30 仅完成新增 20 个 warmup 和 55/180 个分叉，未形成验证选择，且不参与结论；E2/M8 bootstrap 为 `NOT_YET_RUN`。Phase 1 科学结论为 `FAILED`。

当前证据只支持保守的 rank-1 预测组件。由于至少一个预注册支持/rank 门槛失败，M9 rank-2 与 M10 完整 Urysohn 升级为 `NOT_APPLICABLE`，Phase 2 为 `NOT_YET_RUN`。私有 CZ 轨道保持排除，未读取私有工作簿。

## 关键结果

| 场景 | 测试 RMSE | 测试 R² | 支持召回率 | 支持假阳性率 | 10 种子 penalty | 30 种子验证 penalty |
|---|---:|---:|---:|---:|---:|---:|
{scenario_lines}

### 预注册门槛

| 门槛 | 观测值 | 阈值 | 结果 |
|---|---:|---:|---|
{gate_lines}

E2/M8 的 Gram 白化 SVD 已有描述性结果，但 formal bootstrap 按用户指令停止，状态为 `NOT_YET_RUN`，因此不作 rank 显著性结论。

### 模型增益

正值表示候选模型 RMSE 更低。

| 比较 | 相对 RMSE 增益 |
|---|---:|
{comparison_lines}

M7/M8/M6 的超参数均先由 validation-only one-SE 冻结；SVD/bootstrap/rank 量未参与超参数选择。

## 范围、数据与指标

- 数据范围：合成生成器 G2 的 AR-S0–AR-S7；screening test 每场景 10 个独立种子。
- critical-30：AR-S0 只有部分计算产物，没有形成 validation selection；E1–E4 critical-30 均不作为报告证据。
- 指标：RMSE、R²、支持召回率、支持假阳性率、Gram 白化非可分离统计量、全局 bootstrap p 值和 BH-FDR。
- 不在本报告证据范围：TEP、Debutanizer、Gas Turbine、私有 CZ、多晶棒。

## 方法

Scheme A 使用共享 warmup、独立剪枝分叉与跨种子 validation-only one-SE。M7 联合搜索幅值 grid 与平滑权重，one-SE 内先选小 grid、再选强平滑。M8 固定 Scheme A/M7 后顺序选择 lag grid 与平滑。formal bootstrap 未运行，因此报告不进行显著性 rank 判定。

## 限制与稳健性

- 10 种子 test 已在先前筛选流程中打开，因此 critical-30 只能提供验证稳定性证据，不能恢复新的 test 锁箱。
- 高 R² 可由 AR 持续性承担，不能单独证明外生过程支持恢复或机制识别成功。
- AR-S3 的 Scheme A 支持失败会阻止对 rank-2 真值的有效 M8/M9 power 结论；不能用“未检出”替代“证明 rank-1”。
- 结果只覆盖已冻结的合成核心配置；尚无公开长期数据外推证据。

## 阶段状态

| 阶段 | 状态 | 原因 |
|---|---|---|
{phase_lines}

## 建议的下一步

1. 不启动 Phase 2，也不启用 M9/M10；先由用户决定是否允许修改 Phase 1 支持恢复方案或重新预注册筛选策略。
2. 若允许新一轮预注册，应在新 namespace 中校准支持门槛并补齐 AR-S3 rank-2 power，旧结果保持不可变。
3. 若不修改科学方案，则当前版本应以“高预测拟合、有限外生支持证据、无 rank-2 升级依据”封存。

## 进一步问题

- 是否允许新版本协议改变 Scheme A 的支持筛选预算或 penalty 网格？
- 是否仍要在 Phase 1 门槛失败的情况下，单独开展不承载 rank 结论的 TEP 预测基线？
- 是否需要将这次停止线包作为 GitHub Release 之外的长期对象存储副本？
"""
    atomic_text(output_root / "REPORT.md", report)

    checksums = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums[str(path.relative_to(output_root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    atomic_text(
        output_root / "SHA256SUMS.txt",
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
