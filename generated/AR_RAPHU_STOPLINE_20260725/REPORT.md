# AR-RAPHU v2 停止线验证报告

## 技术摘要

按用户指令提前停止后，Phase 0 与 AR-S0–AR-S7 Scheme A screening 已完成。累计记录 2,476,482 个实际优化 epoch。AR-S0 critical-30 仅完成新增 20 个 warmup 和 55/180 个分叉，未形成验证选择，且不参与结论；E2/M8 bootstrap 为 `NOT_YET_RUN`。Phase 1 科学结论为 `FAILED`。

当前证据只支持保守的 rank-1 预测组件。由于至少一个预注册支持/rank 门槛失败，M9 rank-2 与 M10 完整 Urysohn 升级为 `NOT_APPLICABLE`，Phase 2 为 `NOT_YET_RUN`。私有 CZ 轨道保持排除，未读取私有工作簿。

## 关键结果

| 场景 | 测试 RMSE | 测试 R² | 支持召回率 | 支持假阳性率 | 10 种子 penalty | 30 种子验证 penalty |
|---|---:|---:|---:|---:|---:|---:|
| AR-S0 | 0.2034 | 0.6571 | 1.000 | 0.000 | 0.012 | N/A |
| AR-S1 | 0.3825 | 0.9765 | 0.867 | 0.157 | 0.003 | N/A |
| AR-S2 | 0.4539 | 0.9612 | 0.700 | 0.057 | 0.005 | N/A |
| AR-S3 | 0.4733 | 0.9586 | 0.000 | 0.000 | 0.012 | N/A |
| AR-S4 | 0.6386 | 0.9454 | 0.700 | 0.043 | 0.005 | N/A |
| AR-S5 | 0.3101 | 0.9847 | 0.833 | 0.157 | 0.003 | N/A |
| AR-S6 | 0.3436 | 0.9811 | 1.000 | 0.171 | 0.003 | N/A |
| AR-S7 | 0.5694 | 0.9639 | 0.367 | 0.043 | 0.006 | N/A |

### 预注册门槛

| 门槛 | 观测值 | 阈值 | 结果 |
|---|---:|---:|---|
| AR-S1 support false-positive rate | 0.1571 | 0.1000 | FAIL |
| AR-S3 support recall | 0.0000 | 0.8000 | FAIL |

E2/M8 的 Gram 白化 SVD 已有描述性结果，但 formal bootstrap 按用户指令停止，状态为 `NOT_YET_RUN`，因此不作 rank 显著性结论。

### 模型增益

正值表示候选模型 RMSE 更低。

| 比较 | 相对 RMSE 增益 |
|---|---:|
| E2 M7 vs M5 | -3.33% |
| E2 M8 vs M7 | -0.43% |
| E3 M6 vs M5 | -0.12% |

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
| Phase 0 protocol and implementation | COMPLETED | Frozen v2 semantics, tests, leakage guards, and runtime. |
| Phase 1 synthetic | FAILED | AR-S0 through AR-S7 screening completed, but support gates failed; critical-30 and bootstrap were stopped by the user. |
| M9 rank-2 and M10 full surface upgrade | NOT_APPLICABLE | No external variable passed all frozen upgrade gates. |
| Phase 2 TEP | NOT_YET_RUN | Stopped before Phase 2 because the Phase 1 gate did not pass. |
| Phase 3 Debutanizer and Gas Turbine | NOT_YET_RUN | Phase 2 has not been completed. |
| Phase 4 private CZ | NOT_APPLICABLE | Excluded by user on 2026-07-25; private data was not read. |
| Phase 5 multi-rod CZ | NOT_APPLICABLE | Excluded by user and no additional rods are in scope. |

## 建议的下一步

1. 不启动 Phase 2，也不启用 M9/M10；先由用户决定是否允许修改 Phase 1 支持恢复方案或重新预注册筛选策略。
2. 若允许新一轮预注册，应在新 namespace 中校准支持门槛并补齐 AR-S3 rank-2 power，旧结果保持不可变。
3. 若不修改科学方案，则当前版本应以“高预测拟合、有限外生支持证据、无 rank-2 升级依据”封存。

## 进一步问题

- 是否允许新版本协议改变 Scheme A 的支持筛选预算或 penalty 网格？
- 是否仍要在 Phase 1 门槛失败的情况下，单独开展不承载 rank 结论的 TEP 预测基线？
- 是否需要将这次停止线包作为 GitHub Release 之外的长期对象存储副本？
