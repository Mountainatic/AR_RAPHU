#!/usr/bin/env python3
"""Build the final answer-first CZ ORSS experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    return f"{float(value):.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_root)
    required = {
        "ORSS equivalence": root / "orss_equivalence.json",
        "R2.1": root / "R2_1" / "R2_1_STATUS.json",
        "History": root / "R3A" / "R3A_STATUS.json",
        "Resolution": root / "R3B" / "R3B_STATUS.json",
        "Continuation": root / "R3C" / "R3C_STATUS.json",
        "Rank": root / "R3D" / "R3D_STATUS.json",
        "Baselines": root / "R4" / "R4_STATUS.json",
        "Furnace A confirmation": root / "R5" / "R5_STATUS.json",
        "Furnace B zero-shot": root / "R6" / "R6_STATUS.json",
        "Furnace B calibration": root / "R7" / "R7_STATUS.json",
        "Interpretability": root
        / "interpretability"
        / "INTERPRETABILITY_STATUS.json",
        "Bootstrap": root / "bootstrap" / "BOOTSTRAP_STATUS.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing final stages: {missing}")
    status_rows = {name: _read(path) for name, path in required.items()}
    incomplete = [
        name
        for name, row in status_rows.items()
        if row.get("status") != "COMPLETED"
    ]
    if incomplete:
        raise RuntimeError(f"Incomplete final stages: {incomplete}")

    history = _read(root / "R3A" / "history_selection.json")
    resolution = _read(root / "R3B" / "resolution_selection.json")
    continuation = _read(root / "R3C" / "continuation_selection.json")
    rank = _read(root / "R3D" / "rank_selection.json")
    baselines = _read(root / "R4" / "baseline_matrix.json")
    confirmation = _read(root / "R5" / "furnace_a_confirmation.json")
    outer = _read(root / "R6" / "outer_zero_shot_metrics.json")
    calibration = _read(root / "R7" / "furnace_b_calibration.json")
    equivalence = _read(root / "orss_equivalence.json")
    bootstrap = _read(root / "bootstrap" / "bootstrap_summary.json")

    lines = [
        "# CZ 5090 ORSS 完整实验报告",
        "",
        "## 结论摘要",
        "",
        "本轮在 RTX 5090 上完成了矩阵自由 ORSS 数值重构、开发选择、"
        "第一炉锁箱确认、第二炉 zero-shot 与轻量校准。所有超参数均在"
        " Furnace A development 上冻结；Furnace A 最后 20% 和 Furnace B "
        "未参与选择。",
        "",
        f"- ORSS/dense 等价门禁：`{equivalence['status']}`。",
        "- 结构性 K-level 结论仍受 coercivity/Schur 前置条件限制；"
        "报告的是 predictive rank，不把 continuation 外推解释为真实 K。",
        "",
        "## 冻结配置",
        "",
        "| h | Lx | Ly | Mtau | Mx | c_rho | predictive rank (5%) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in sorted(history["selections"], key=int):
        h = history["selections"][horizon]["selected_history"]
        r = resolution["selections"][horizon]["selected_resolution"]
        c = continuation["selections"][horizon][
            "selected_CONTINUATION_SCALE_COEFFICIENT"
        ]
        pr = rank["profiles"][horizon]["predictive_rank_by_budget"]["0.05"]
        lines.append(
            f"| {horizon} | {h['L_x']} | {h['L_y']} | "
            f"{r['M_tau']} | {r['M_x']} | {_fmt(c)} | {pr} |"
        )
    lines.extend(
        [
            "",
            "## Direct prediction",
            "",
            "| h | Dev AR RMSE | Dev ARX RMSE | Furnace A confirm RMSE | "
            "Furnace B zero-shot RMSE |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in sorted(history["selections"], key=int):
        aggregate = baselines["results"][horizon]["aggregate"]
        a = confirmation["models"][horizon]["direct_metrics"]
        b = outer["models"][horizon]
        lines.append(
            f"| {horizon} | "
            f"{_fmt(aggregate['AR']['RMSE_mm']['mean_across_folds'])} | "
            f"{_fmt(aggregate['ARX']['RMSE_mm']['mean_across_folds'])} | "
            f"{_fmt(a['RMSE_mm'])} | {_fmt(b['RMSE_mm'])} |"
        )
    lines.extend(
        [
            "",
            "## Furnace B 轻量校准",
            "",
            "| h | 5% RMSE | 10% RMSE |",
            "|---:|---:|---:|",
        ]
    )
    for horizon in sorted(calibration["models"], key=int):
        values = calibration["models"][horizon]
        lines.append(
            f"| {horizon} | {_fmt(values['0.05']['metrics']['RMSE_mm'])} | "
            f"{_fmt(values['0.10']['metrics']['RMSE_mm'])} |"
        )
    lines.extend(
        [
            "",
            "## 不确定性与边界",
            "",
            f"- Bootstrap 状态：`{bootstrap['status']}`；时间块抽样，"
            "未把逐点样本视为独立。",
            "- 当前证据只覆盖两炉等径阶段；不能外推为所有晶棒、所有阶段"
            "或 causal plant 的普适证明。",
            "- 单位和采样周期未补齐的字段继续按采样步报告，不作热传播时间"
            "解释。",
            "",
            "## 完整性",
            "",
            "源码、配置、逐任务检查点、日志、模型哈希、解释性产物、"
            "bootstrap 及 repository bundle 均进入最终结果包。",
            "",
        ]
    )
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    Path("CZ_COMPLETE_5090_STATUS.json").write_text(
        json.dumps(
            {
                "schema": "CZ_COMPLETE_5090_STATUS_V1",
                "status": "COMPLETED",
                "CZ_COMPLETE_PIPELINE_FINISHED": True,
                "stages": {
                    name: row.get("status")
                    for name, row in status_rows.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
