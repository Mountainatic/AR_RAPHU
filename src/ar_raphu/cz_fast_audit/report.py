"""Reader-facing Markdown report for the CZ FAST audit."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import numpy as np


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def build_markdown_report(
    *,
    status: dict[str, object],
    conditional_energy_rows: list[dict[str, object]],
    lag_rows: list[dict[str, object]],
    gram_summary: dict[str, object],
    linear_rows: list[dict[str, object]],
    coarse_rows: list[dict[str, object]],
    q_rows: list[dict[str, object]],
    k_rows: list[dict[str, object]],
    runtime_profile: dict[str, object],
    source_sha256: str,
) -> str:
    """Create a technical, answer-first report without causal overclaiming."""

    coarse_by_task: dict[str, list[float]] = defaultdict(list)
    linear_by_task: dict[str, list[float]] = defaultdict(list)
    for row in coarse_rows:
        coarse_by_task[str(row["task"])].append(
            float(row["delta_X_given_AR_coarse"])
        )
    for row in linear_rows:
        linear_by_task[str(row["task"])].append(
            float(row["delta_X_given_AR"])
        )

    energy_best: dict[str, tuple[str, float]] = {}
    for row in conditional_energy_rows:
        task = str(row["scale"])
        candidate = (str(row["input"]), float(row["conditional_energy_ratio"]))
        if task not in energy_best or candidate[1] > energy_best[task][1]:
            energy_best[task] = candidate

    lag_best: dict[str, tuple[str, float, int]] = {}
    for row in lag_rows:
        if not bool(row.get("is_fold_maximum", False)):
            continue
        task = str(row["task"])
        candidate = (
            str(row["input"]),
            float(row["maximum_absolute_correlation"]),
            int(row["maximum_correlation_lag"]),
        )
        if task not in lag_best or candidate[1] > lag_best[task][1]:
            lag_best[task] = candidate

    stable_k = [
        row for row in k_rows if row.get("status") == "K_LOW_ORDER_STABLE"
    ]
    k_not_tested = any(
        row.get("status") == "K_NOT_TESTED_DUE_TO_NO_Q_GAIN"
        for row in k_rows
    )
    generated = datetime.now(timezone.utc).isoformat()
    decision = str(status["status"])
    next_stage = str(status["next_allowed_stage"])

    lines = [
        "# CZ 快速 GO / NO-GO 可辨识性审计报告",
        "",
        "## 技术摘要",
        "",
        f"**自动判定：`{decision}`。** 本判定只适用于 Furnace A 前 80% "
        "development 轨迹上的快速筛查，不是最终模型选择、完整 K 面恢复或"
        "跨晶棒物理认证。",
        "",
        f"- 允许的下一阶段：`{next_stage}`。",
        f"- 外生增量为正的固定 horizon："
        f"`{status['positive_increment_horizons']}`。",
        f"- Furnace B 访问次数：`{status['furnace_b_access_count']}`。",
        f"- 总墙钟时间：`{_fmt(status['runtime_seconds'])}` 秒。",
        "",
        "## 三个固定任务给出的预测证据",
        "",
        "| 尺度 | horizon | 线性 Δ(X|AR) 两折均值 | 粗非线性 Δ(X|AR) 两折均值 | 两折方向一致 |",
        "|---|---:|---:|---:|---|",
    ]
    if decision == "AUDIT_INCOMPLETE":
        lines[10:10] = [
            "**未闭合原因：** 三个固定任务的线性和粗非线性 XAR 增量均为负，"
            "但条件输入能量与条件 Gram 又不弱；现有快速证据既不满足继续完整"
            " K 的 GO 条件，也不满足“激励和条件谱均弱”的 NO-GO 条件。",
            "",
        ]
    horizons = {"short": 1, "medium": 15, "long": 60}
    for task in ("short", "medium", "long"):
        coarse_values = coarse_by_task.get(task, [])
        direction = bool(coarse_values) and all(value > 0.0 for value in coarse_values)
        lines.append(
            f"| {task} | {horizons[task]} | "
            f"{_fmt(np.mean(linear_by_task.get(task, [np.nan])))} | "
            f"{_fmt(np.mean(coarse_values or [np.nan]))} | "
            f"{'是' if direction else '否'} |"
        )
    lines += [
        "",
        "这里的 Δ 定义为 `(MSE_AR - MSE_XAR) / MSE_AR`；正值表示加入五个"
        "外生过程量后验证误差下降。该量是预测增量证据，不等同于物理因果效应。",
        "",
        "## AR 条件化后仍剩多少独立输入信息",
        "",
        "| 尺度 | 条件能量最高的输入 | 条件能量比 | 最大 AR 残差相关输入 | |corr| | lag（采样步） |",
        "|---|---|---:|---|---:|---:|",
    ]
    for task in ("short", "medium", "long"):
        energy = energy_best.get(task, ("NOT_AVAILABLE", float("nan")))
        lag = lag_best.get(task, ("NOT_AVAILABLE", float("nan"), -1))
        lines.append(
            f"| {task} | {energy[0]} | {_fmt(energy[1])} | "
            f"{lag[0]} | {_fmt(lag[1])} | {lag[2]} |"
        )
    lines += [
        "",
        "条件能量使用 train-only ridge 将输入历史对严格滞后的直径历史做"
        "残差化；FAST-B 再检查 AR 残差与输入滞后的相关。相关性仍是诊断量，"
        "不能单独证明工艺机制。",
        "",
        "## 条件 Gram 与低阶 K 稳定性",
        "",
        f"- 联合条件 Gram 的折间中位 effective rank："
        f"`{_fmt(status['conditional_gram_summary'].get('joint_median_effective_rank', 0.0))}`。",
        f"- `1e-3` 相对谱阈值下的中位 coercive dimension："
        f"`{_fmt(status['conditional_gram_summary'].get('joint_median_coercive_dimension_1e-3', 0.0))}`。",
        (
            "- 低阶 K 稳定性：`K_NOT_TESTED_DUE_TO_NO_Q_GAIN`；FAST-E "
            "没有正的 Q 增量，因此合同禁止将其解释为 K 不稳定。"
            if k_not_tested
            else f"- 跨折稳定的低阶 K mode 数量：`{len(stable_k)}`。"
        ),
    ]
    if stable_k:
        lines += [
            "",
            "| 尺度 | 输入 | leading surface mode corr | principal angle (deg) |",
            "|---|---|---:|---:|",
        ]
        for row in stable_k:
            lines.append(
                f"| {row['task']} | {row['input']} | "
                f"{_fmt(row['leading_surface_mode_correlation'])} | "
                f"{_fmt(row['principal_angle_degrees'])} |"
            )
    lines += [
        "",
        "低阶 K 只表示粗网格模型中的 leading lag/amplitude mode 在两折间"
        "是否稳定；报告不会将其称为完整物理核、因果对象或已恢复工厂机理。",
        "",
        "## 数据范围、定义与执行边界",
        "",
        "- 数据：Furnace A / Sheet1，仅前 80% development 区间。",
        "- 输入：主加热功率、晶升速度、晶转速度、埚升速度、埚转速度。",
        "- 目标：晶体直径；history 只使用预测原点及之前的数据。",
        "- 固定任务：`(Lx,Ly,h)=(64,16,1),(256,32,15),(512,64,60)`。",
        "- 固定两折：0–50%/50–60% 与 0–70%/70–80%，并应用"
        "`max(Lx-1,Ly-1)+h` purge。",
        "- Furnace B、完整 ORSS、R3 全搜索、confirmation 均未执行。",
        "",
        "## 方法与数值检查",
        "",
        "- FAST-A：train-only 多目标 ridge 条件能量，moving-block bootstrap。",
        "- FAST-B：线性 AR 残差的滞后相关、block bootstrap 与 block permutation。",
        "- FAST-C：`Mtau=Mx=8` 粗特征经 AR 条件化后的 Gram/Schur 谱。",
        "- FAST-D：Persistence、AR、ridge-ARX 的固定任务比较。",
        "- FAST-E：`Mtau=Mx=8`、共享 penalty path 的粗非线性 XAR。",
        "- FAST-F：跨折 Q contribution 与加权 SVD leading K mode 稳定性。",
        "",
        "## 局限性与稳健性边界",
        "",
        "- 当前只有单根晶棒的一段闭环运行轨迹；不支持跨晶棒、跨炉次或"
        "跨阶段泛化结论。",
        "- 未知采样周期和设备端滤波/延迟，因此 lag 只以采样步报告，不能直接"
        "解释为热传播时间。",
        "- 本轮 coarse resolution 和固定 penalty path 仅用于路线判定，不能"
        "代替完整超参数冻结和锁箱评估。",
        "- 任一 `AUDIT_INCOMPLETE` 都意味着证据未形成闭合链，而不是阴性结论。",
        "",
        "## 建议的下一步",
        "",
        f"严格只进入状态文件允许的阶段：`{next_stage}`。若该值为 `NONE_*`，"
        "应先解决运行门禁或证据歧义，不得自动恢复完整 K 搜索。",
        "",
        "## 仍需回答的问题",
        "",
        "- 增加晶棒和工况后，Q 增量与低阶 K mode 是否保持跨轨迹稳定？",
        "- 取得采样周期、测量滤波和控制回路元数据后，lag 能否获得受限的"
        "物理解释？",
        "- 若继续完整实验，冻结模型能否在未参与选择的 confirmation 区间"
        "保持同方向收益？",
        "",
        f"生成时间（UTC）：`{generated}`  ",
        f"Furnace A SHA256：`{source_sha256}`  ",
        f"执行器：`{runtime_profile.get('solver', 'dense_batched')}`",
        "",
    ]
    return "\n".join(lines)
