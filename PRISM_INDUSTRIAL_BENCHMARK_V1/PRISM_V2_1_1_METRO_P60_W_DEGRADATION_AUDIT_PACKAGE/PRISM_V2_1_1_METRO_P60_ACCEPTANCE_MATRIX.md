# PRISM v2.1.1 Metro-P60 结果接受矩阵

成功的含义是“选择与实现合同正常”，不是强迫 W 开或关。

| Development 正式选择 | Test/OOD 中 W 边际 | 正式标签 | 含义 |
|---|---|---|---|
| PF identity；Joint `J_K/J_KA` | W-on 无稳定正增量 | `NORMAL_NO_W_DEGRADATION` | 正常退化为 v1.3-compatible no-W 子模型 |
| PF W-on 和/或 Joint W-on | test 与 OOD 均正向且统计支持 | `W_TRIGGER_TRANSFER_SUPPORTED` | W 正常触发并具有迁移增量 |
| PF W-on | test/OOD 中性或负向 | `PF_W_DEVELOPMENT_ONLY_TRANSFER_UNSTABLE` | W 实现正常，但 PF 局部曲率未迁移 |
| PF identity | Joint `J_KW/J_KWA` 正向 | `W_INTERACTION_DEPENDENT_JOINT_ONLY` | W 只有联合重调时有效 |
| PF W-on | Joint 无 W | `PF_STATIC_CURVATURE_ONLY` | W 只在逐级残差读出中被支持 |
| PF/Joint W-on，但 OOD 显著恶化 | OOD 风险 | `W_ID_GAIN_OOD_RISK` | 曲率依赖操作域 |
| W 候选因硬门未参赛 | — | `IMPLEMENTATION_CONTRACT_FAILURE` | 停止，不能作科学结论 |
| identity 与跳过 W 不等价 | — | `IMPLEMENTATION_CONTRACT_FAILURE` | 停止 |
| C 抹除 K 或 gate 不一致 | — | `IMPLEMENTATION_CONTRACT_FAILURE` | 停止 |
| test 后未选 W 消融更优 | 任意 | `POST_FREEZE_ABLATION_SIGNAL_ONLY` | 仅作下一版本动机 |

注意：`mu>0 vs mu=0` 不是 `W vs no-W`；`KCW vs baseline` 也不是 W 独立贡献。PF 必须看 `KCW vs KC`，Joint 必须看 `J_KW vs J_K` 或 `J_KWA vs J_KA`。
