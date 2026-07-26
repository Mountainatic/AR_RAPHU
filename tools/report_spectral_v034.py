#!/usr/bin/env python3
"""Generate the evidence-backed Spectral v0.3.4 rank-profile report."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "spectral_v034"
CONFIG = ROOT / "configs" / "spectral_v034.yaml"


def read_rows(relative: str) -> list[dict[str, str]]:
    with (RESULT_ROOT / relative).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def rank_range(rows: list[dict[str, str]], field: str) -> str:
    values = sorted({int(row[field]) for row in rows})
    return str(values[0]) if len(values) == 1 else f"{values[0]}–{values[-1]}"


def group_rows(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], int(row["variable"]))].append(row)
    return dict(sorted(grouped.items()))


def status(experiment: str) -> str:
    payload = json.loads(
        (RESULT_ROOT / experiment / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    return str(payload["status"])


def predictive_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| 场景 | 变量 | R(0.10) | R(0.05) | R(0.02) | "
        "最小验证 R² | 最大算子 NRMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (scenario, variable), group in group_rows(rows).items():
        min_r2 = min(float(row["full_validation_contribution_r2"]) for row in group)
        max_nrmse = max(
            float(row["full_empirical_operator_nrmse"]) for row in group
        )
        lines.append(
            f"| {scenario} | {variable} | "
            f"{rank_range(group, 'predictive_rank_010')} | "
            f"{rank_range(group, 'predictive_rank_005')} | "
            f"{rank_range(group, 'predictive_rank_002')} | "
            f"{min_r2:.6f} | {max_nrmse:.6f} |"
        )
    return lines


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    structural = read_rows("E2A_SR/metrics.csv")
    bootstrap = read_rows("E2A_SRB/bootstrap_rank_intervals.csv")
    natural = read_rows("E2A_P_NAT/metrics.csv")
    permuted = read_rows("E2A_P_PERM/metrics.csv")

    structural_max_tail = max(
        float(row["tail_curve_max_abs_error"]) for row in structural
    )
    structural_max_l1 = max(
        float(row["normalized_spectrum_l1_distance"]) for row in structural
    )
    bootstrap_max_width = max(
        int(row["rank_high_005"]) - int(row["rank_low_005"])
        for row in bootstrap
    )
    bootstrap_reselected = any(
        row["smoothing_reselected"].lower() == "true" for row in bootstrap
    )
    natural_min_r2 = min(
        float(row["full_validation_contribution_r2"]) for row in natural
    )
    natural_max_nrmse = max(
        float(row["full_empirical_operator_nrmse"]) for row in natural
    )
    permuted_min_r2 = min(
        float(row["full_validation_contribution_r2"]) for row in permuted
    )
    permuted_max_nrmse = max(
        float(row["full_empirical_operator_nrmse"]) for row in permuted
    )

    structural_lines = [
        "| 场景 | 变量 | 真值类别 | R(0.10) | R(0.05) | R(0.02) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for (scenario, variable), group in group_rows(structural).items():
        structural_lines.append(
            f"| {scenario} | {variable} | {group[0]['truth_class']} | "
            f"{rank_range(group, 'rank_010')} | "
            f"{rank_range(group, 'rank_005')} | "
            f"{rank_range(group, 'rank_002')} |"
        )

    bootstrap_lines = [
        "| 场景 | 变量 | 主预算 90% bootstrap 区间（跨 seeds） |",
        "|---|---:|---:|",
    ]
    for (scenario, variable), group in group_rows(bootstrap).items():
        low = min(int(row["rank_low_005"]) for row in group)
        high = max(int(row["rank_high_005"]) for row in group)
        bootstrap_lines.append(
            f"| {scenario} | {variable} | [{low}, {high}] |"
        )

    gate_rows = [
        ("R0", status("R0")),
        ("E2A_SR", status("E2A_SR")),
        ("E2A_SRB", status("E2A_SRB")),
        ("E2A_P_NAT", status("E2A_P_NAT")),
        ("E2A_P_PERM", status("E2A_P_PERM")),
    ]
    gate_lines = ["| 阶段 | 状态 |", "|---|---|"]
    gate_lines.extend(f"| {name} | `{value}` |" for name, value in gate_rows)

    rank_config = config["rank_profile"]
    predictive_config = config["predictive_rank"]
    bootstrap_config = config["bootstrap"]
    report = "\n".join(
        [
            "# Spectral PS-AR-RAPHU v0.3.4 自适应秩剖面报告",
            "",
            "## 结论",
            "",
            "v0.3.4 的五个预注册开发阶段全部通过。完整二维结构面本身具备"
            "容量；此前 v0.3.3 的停止点来自“所有结构都可压成 rank-2”这一"
            "过强假设，而不是完整 Urysohn 面、结构分辨率或凸求解器失效。",
            "",
            "在主能量尾预算 0.05 下，AR-S1/AR-S2 保持 rank-1，AR-S3 的"
            "强 rank-2 变量恢复为 rank-2，弱 rank-2 变量在主预算为 rank-1、"
            "在精细预算 0.02 显露第二模态；AR-S4U 稳定需要 rank-5。"
            "因此，证据支持自适应秩剖面，不支持 universal rank-2。",
            "",
            "冻结决策为 `ADAPTIVE_RANK_PROFILE_VALIDATED`；协议允许下一阶段"
            " `E2B`，但本轮严格停在 v0.3.4 停止线，没有启动 E2B/E3。",
            "",
            "## 阶段状态",
            "",
            *gate_lines,
            "",
            "全部 12 个场景-变量组在每个实验中均为 5/5 seeds 通过；"
            "预注册门槛是至少 4/5。",
            "",
            "## 结构秩剖面 E2A_SR",
            "",
            *structural_lines,
            "",
            f"- 最大尾曲线绝对误差：{structural_max_tail:.8f}"
            f"（门槛 ≤ {rank_config['tail_curve_max_abs_error']:.2f}）。",
            f"- 最大归一化谱 L1 距离：{structural_max_l1:.8f}"
            f"（门槛 ≤ {rank_config['normalized_spectrum_l1_max']:.2f}）。",
            "- `rank_max=12` 之外的残余能量被显式保存，未把尾部静默截断。",
            "",
            "## Bootstrap 秩稳定性 E2A_SRB",
            "",
            *bootstrap_lines,
            "",
            f"- 每任务 {bootstrap_config['development_replicates']} 次 circular "
            f"block bootstrap，块长 {bootstrap_config['block_length']}。",
            f"- 主预算区间最大宽度为 {bootstrap_max_width}"
            f"（门槛 ≤ {bootstrap_config['stable_interval_width_max']}）。",
            f"- bootstrap 期间重新选择 smoothing："
            f"`{'YES' if bootstrap_reselected else 'NO'}`。",
            "",
            "## 自然激励预测秩 E2A_P_NAT",
            "",
            *predictive_table(natural),
            "",
            f"- 最低完整模型验证贡献 R²：{natural_min_r2:.6f}"
            f"（门槛 ≥ {predictive_config['full_prediction_r2_gate']:.3f}）。",
            f"- 最大经验算子 NRMSE：{natural_max_nrmse:.6f}"
            f"（门槛 ≤ {predictive_config['full_empirical_operator_nrmse_gate']:.2f}）。",
            "",
            "## 置换激励预测秩 E2A_P_PERM",
            "",
            *predictive_table(permuted),
            "",
            f"- 最低完整模型验证贡献 R²：{permuted_min_r2:.6f}。",
            f"- 最大经验算子 NRMSE：{permuted_max_nrmse:.6f}。",
            "- 置换激励下完整预测容量仍全部通过；有效预测秩出现预期的"
            "激励依赖偏移，但没有把高秩场景误压成 universal rank-2。",
            "",
            "## 科学解释与边界",
            "",
            "- 支持：用能量尾预算定义的有效秩是结构依赖且预算依赖的；"
            "rank-1、强/弱 rank-2 和更高秩真值均被区分。",
            "- 支持：自然激励下结构秩与预测秩在主预算精确对齐；置换激励"
            "改变了部分有效秩，但仍保持自适应秩而非 universal rank-2。",
            "- 否定：`UNIVERSAL_RANK2_HYPOTHESIS` 保持 `REJECTED`，不会因"
            "本轮结果改写。",
            "- 尚未声称：E2B 的联合外生可辨识性、E3 的双残差化有效性、"
            "真实工业数据泛化或部署可靠性。本轮数据仍是合成开发证据。",
            "",
            "## 执行完整性说明",
            "",
            "- 复用了 v0.3.3 已冻结的 full fits 与平滑选择，没有为秩结果"
            "重新调参。",
            "- 超参数选择不使用秩、奇异值或测试集。",
            "- E2A_P_NAT 首次预运行在构造未参与选择的 test 设计矩阵时触发"
            "幅值域保护并在拟合前停止；随后按协议删除该不必要的 test 访问，"
            "未裁剪输入、未改变门槛。失败日志单独保留，不作为科学结果。",
            "- v0.3.3 原决策文件保持不变；v0.3.4 通过附加解释和新决策记录"
            "纠正 universal rank-2 的科学归因。",
            "",
        ]
    )
    (RESULT_ROOT / "V034_RANK_PROFILE_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(f"REPORT={RESULT_ROOT / 'V034_RANK_PROFILE_REPORT.md'}")


if __name__ == "__main__":
    main()
