# PRISM v2.1.1 Metro-P60 W 退化/触发审计包

## 目的

本包只回答一个问题：

> 在旧 CPU 基准中已由无 W 的 v1.3 Physical-First 明确建模成功的 `METRO_P60__H6__W1` 上，PRISM v2.1.1 是否能够让 W 正常参赛，并在不需要 W 时退化为 identity，在需要 W 时给出可迁移的增量？

这不是一次新的“未见 test”确认实验。Metro-P60 的旧 v1.3 test/OOD 聚合结果已经历史可见，因此本轮必须标记为：

```text
RETROSPECTIVE_TRANSFER_AND_DEGRADATION_AUDIT
```

它可以检验实现合同、选择器行为和跨数据集迁移，但不能单独承担全新确认性 benchmark 的证据等级。

## 冻结对象

- 唯一数据集/头：`METRO_P60__H6__W1`
- 算法：PRISM v2.1.1，继承 SRU 修复后的实现与阈值
- 不修改：K/C/W/A/J 理论、选择阈值、row caps、共享 C1 数据、旧结果
- 新输出：`results_prism_v2_1_1_metro_p60_w_audit/`
- 建议分支：`prism-v2-1-1-metro-p60-w-audit`

## 包内容

- `PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_PLAN.md`：正式实验方案
- `PRISM_V2_1_1_METRO_P60_CONFIG_FROZEN_PROPOSED.json`：机器可读冻结配置
- `PRISM_V2_1_1_METRO_P60_REQUIRED_OUTPUT_CONTRACT.md`：结果文件与字段合同
- `PRISM_V2_1_1_METRO_P60_ACCEPTANCE_MATRIX.md`：结果定性矩阵
- `PRISM_V2_1_1_METRO_P60_CODEX_EXECUTION_PROMPT.md`：可直接交给 Codex 的执行提示词
- `scripts/preflight_env.sh`：资源安全预设
- `scripts/package_results.sh`：最终清理、manifest、hash、校验与压缩脚本
- `reference/`：v2.1.1 理论与旧 Metro-P60 聚合参考

## 最重要的判据

成功不等于“W 必须关闭”，也不等于“W 必须开启”。成功意味着：

1. identity、monotone、cubic 候选都按合同实际参赛；
2. identity 与跳过 W 的预测逐样本等价；
3. PF 与 Joint 使用同一个输入路径 gate；
4. 选择发生在 development，test/OOD 只在冻结后访问；
5. W 的启用/关闭能够被清楚归类，而不是被数值门或上游坍缩偷偷决定。
