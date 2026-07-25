#!/usr/bin/env python3
"""Convert the stop-line evidence summary into the canonical report artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "generated" / "AR_RAPHU_STOPLINE_20260725"


def source(
    identifier: str,
    label: str,
    sql: str,
    description: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "label": label,
        "path": "report/scenario_evidence.csv",
        "query": {
            "sql": sql,
            "description": description,
            "engine": "DuckDB",
            "language": "SQL",
            "tables_used": ["report/scenario_evidence.csv"],
            "filters": [
                "Generator version 2",
                "Screening seeds 0 through 9",
                "Private CZ excluded",
            ],
        },
    }


def main() -> int:
    summary_path = REPORT_ROOT / "evidence_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scenario_rows = summary["scenario_results"]
    scenario_dataset = [
        {
            "scenario": row["scenario"],
            "mean_test_rmse": row["mean_test_rmse"],
            "mean_test_r2": row["mean_test_r2"],
            "support_recall": row["support_recall"],
            "support_false_positive_rate": row[
                "support_false_positive_rate"
            ],
            "screening_selected_scale": row["screening_selected_scale"],
            "critical30_selected_scale": row[
                "critical30_selected_scale"
            ],
        }
        for row in scenario_rows
    ]
    support_dataset = [
        {
            "scenario": row["scenario"],
            "metric": metric,
            "rate": row[field],
        }
        for row in scenario_rows
        for metric, field in (
            ("support recall", "support_recall"),
            ("support false-positive rate", "support_false_positive_rate"),
        )
    ]
    phase_dataset = [
        {
            "phase": row["phase"],
            "status": row["status"],
            "reason": row["reason"],
        }
        for row in summary["phases"]
    ]
    rmse_source = source(
        "scenario-evidence",
        "Scenario evidence summary",
        (
            "SELECT scenario, mean_test_rmse, mean_test_r2, "
            "support_recall, support_false_positive_rate "
            "FROM read_csv_auto('report/scenario_evidence.csv') "
            "ORDER BY scenario"
        ),
        "Read the reviewed scenario-level synthetic validation evidence.",
    )
    support_source = source(
        "support-evidence",
        "Support recovery summary",
        (
            "SELECT scenario, metric, rate FROM "
            "(SELECT scenario, support_recall, "
            "support_false_positive_rate FROM "
            "read_csv_auto('report/scenario_evidence.csv')) "
            "UNPIVOT(rate FOR metric IN "
            "(support_recall, support_false_positive_rate)) "
            "ORDER BY scenario, metric"
        ),
        "Reshape the reviewed support recall and false-positive rates.",
    )
    gate_failures = [
        row["gate"] for row in summary["gates"] if not row["passed"]
    ]
    bootstrap = summary["rank_bootstrap"]
    bootstrap_text = (
        f"E2/M8 每个种子执行 {bootstrap['replicates_per_seed']} 次"
        "残差移动块 bootstrap；"
        f"全局假阳性率为 {bootstrap['global_false_positive_rate']:.3f}。"
        if bootstrap["status"] == "COMPLETED"
        else "E2/M8 formal bootstrap 按用户指令停止，状态为 `NOT_YET_RUN`；"
        "不作 rank 显著性结论。"
    )
    comparison_text = "\n".join(
        f"- {row['comparison']}: {row['relative_rmse_gain'] * 100:.2f}%"
        for row in summary["model_comparisons"]
    )
    phase_text = "\n".join(
        f"- `{row['status']}` — {row['phase']}: {row['reason']}"
        for row in summary["phases"]
    )
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "title": "AR-RAPHU v2 停止线验证报告",
            "description": (
                "合成层停止线前的证据、失败门槛、结论边界与后续条件。"
            ),
            "surface": "report",
            "generatedAt": "2026-07-25T00:00:00+08:00",
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": (
                        "# AR-RAPHU v2 停止线验证报告\n\n"
                        "停止线前的计算已完成；科学门槛与计算完成状态分开报告。"
                    ),
                },
                {
                    "id": "technical-summary",
                    "type": "markdown",
                    "body": (
                        "## 技术摘要\n\n"
                        f"Phase 1 总体状态为 "
                        f"`{'COMPLETED' if summary['phase1_gate_passed'] else 'FAILED'}`。"
                        "当前证据只支持保守的 rank-1 预测组件；"
                        "M9/M10 未获得升级资格，Phase 2 未启动。"
                        f"失败门槛：{'；'.join(gate_failures) if gate_failures else '无'}。"
                    ),
                    "sourceId": "scenario-evidence",
                },
                {
                    "id": "rmse-header",
                    "type": "markdown",
                    "body": (
                        "## 场景级预测表现\n\n"
                        "RMSE 与 R² 是 10 个 screening test 种子的均值；"
                        "高 R² 不能代替外生支持恢复证据。"
                    ),
                    "sourceId": "scenario-evidence",
                },
                {
                    "id": "rmse-chart-block",
                    "type": "chart",
                    "chartId": "scenario-rmse",
                },
                {
                    "id": "support-header",
                    "type": "markdown",
                    "body": (
                        "## 支持恢复\n\n"
                        "支持召回率与假阳性率按每个种子、每个候选变量汇总。"
                    ),
                    "sourceId": "support-evidence",
                },
                {
                    "id": "support-chart-block",
                    "type": "chart",
                    "chartId": "support-rates",
                },
                {
                    "id": "rank-audit",
                    "type": "markdown",
                    "body": (
                        "## M8 rank 审计\n\n"
                        f"{bootstrap_text}"
                        "超参数在 SVD/bootstrap 前由 validation-only one-SE 冻结，"
                        "已有的部分 critical-30 结果不参与结论。"
                    ),
                },
                {
                    "id": "model-comparison",
                    "type": "markdown",
                    "body": (
                        "## 模型比较\n\n"
                        "相对 RMSE 增益为正才表示候选模型优于参照：\n\n"
                        f"{comparison_text}"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 限制与结论边界\n\n"
                        "- 10 种子 test 已在先前筛选中打开；critical-30 "
                        "被提前停止且不作为证据。\n"
                        "- 高预测拟合可由 AR 持续性承担，不能证明机制识别。\n"
                        "- AR-S3 支持失败阻止有效的 rank-2 power 结论。\n"
                        "- TEP、Debutanizer、Gas Turbine、私有 CZ 和多晶棒"
                        "不在本次证据范围。"
                    ),
                },
                {
                    "id": "phase-status",
                    "type": "markdown",
                    "body": f"## 阶段状态\n\n{phase_text}",
                },
                {
                    "id": "next-steps",
                    "type": "markdown",
                    "body": (
                        "## 建议的下一步\n\n"
                        "1. 不启动 Phase 2，也不启用 M9/M10。\n"
                        "2. 如需继续，先由用户批准新的 Phase 1 预注册版本，"
                        "在新 namespace 中修正支持筛选并补齐 AR-S3 power。\n"
                        "3. 若不修改方案，则按“高预测拟合、有限外生支持证据、"
                        "无 rank-2 升级依据”封存。"
                    ),
                },
            ],
            "charts": [
                {
                    "id": "scenario-rmse",
                    "title": "AR-S0–AR-S7 测试 RMSE",
                    "type": "bar",
                    "dataset": "scenario_metrics",
                    "encodings": {
                        "x": {"field": "scenario", "type": "nominal"},
                        "y": {
                            "field": "mean_test_rmse",
                            "type": "quantitative",
                        },
                    },
                    "source": rmse_source,
                },
                {
                    "id": "support-rates",
                    "title": "AR-S0–AR-S7 支持恢复率",
                    "type": "bar",
                    "dataset": "support_rates",
                    "encodings": {
                        "x": {"field": "scenario", "type": "nominal"},
                        "y": {"field": "rate", "type": "quantitative"},
                        "color": {"field": "metric", "type": "nominal"},
                    },
                    "source": support_source,
                },
            ],
            "sources": [rmse_source, support_source],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": "2026-07-25T00:00:00+08:00",
            "datasets": {
                "scenario_metrics": scenario_dataset,
                "support_rates": support_dataset,
                "phase_status": phase_dataset,
            },
        },
        "sources": [rmse_source, support_source],
        "package_info": {
            "title": "AR-RAPHU v2 stop-line portable report",
            "offline": True,
            "snapshot": True,
        },
    }
    output = REPORT_ROOT / "artifact.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
