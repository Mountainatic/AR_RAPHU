# CZ raw-2s E1–E6 claim impact assessment

## 结论

本次 raw-2s/L256/H4 运行在计算完整性层面通过全部验收，但严格 Phase-A 科学门控判定为 **STOP**。E3–E6 已经计算并保留，状态只能是“探索性/诊断性”，不得作为门控通过后的确认性证据。

## 触发原因

E2 semisynthetic stage recovery 中：

- `FULL` regime 的真实 C 阶段恢复率为 `0.0`；
- `KCW` regime 的真实 C 阶段恢复率为 `0.0`；
- `FULL` regime 的真实 K、W 阶段恢复率各为 `0.5`。

严格规则要求每个应激活的真实阶段恢复率 `> 0.5`。上述结果触发 STOP。总体恢复率 `0.646875` 与总体错误纳入率 `0.10` 不能替代逐真实阶段门槛。

## 可保留的结论

- E1 的双方向冻结路线、开发验证和 formal target-rod 指标可以作为本次专用 adapter 的描述性结果。
- E2 可以用于说明当前结构选择器在该 CZ semisynthetic 设计下的识别局限。
- E3–E6 可用于探索性敏感性分析和后续修订设计，不可写成已通过选择有效性门控的确认性结论。
- 独立的 raw-2s h={1,2,4,8,16} PRISM 主预测扫描不依赖本 E2 门控；它应与本专用 E1–E6 adapter 分开报告。

## 进一步修复方向

- 重新设计 E2 生成器，使 C 的结构真值可识别，并增加 formal seeds 与 null sample-size 曲线；
- 在 runner 中把 Phase-A gate 前置，STOP 时禁止确认性 E3–E6；
- 若仍需计算 STOP 后阶段，输出路径和报告必须显式标为 exploratory。
