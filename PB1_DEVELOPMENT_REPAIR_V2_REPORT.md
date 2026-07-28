# PB1 Development Repair V2 报告

## 结论

本轮 repair-v2 的可执行 development 工作已汇总，但整体状态仍为 `BLOCKED_BY_MISSING_METADATA`，因此不能生成 `PB1_PROTOCOL_FREEZE_V2.json`，也不能进入 confirmation。所有已检查产物的 `official_test_access_count=0`。

当前完成 H3 direct 12 项、每-horizon bootstrap 12 项。Tanks 的 spectral 轨道没有被静默裁剪或用 validation 扩张样条域。

## H3 Shared-History Direct Forecast

| 数据集 | h | 状态 | validation MSE | RMSE | EDF | KKT | free-run |
|---|---:|---|---:|---:|---:|---:|---|
| pwh | 1 | COMPLETED | 6.13223e-06 | 0.00247633 | 387.353 | 6.66952e-16 | FAILED_REPRESENTATION_COVERAGE |
| pwh | 5 | COMPLETED | 0.0001347 | 0.011606 | 385.44 | 1.76421e-15 | NOT_APPLICABLE |
| pwh | 10 | COMPLETED | 0.00135406 | 0.0367975 | 245.536 | 8.0144e-15 | NOT_APPLICABLE |
| pwh | 20 | COMPLETED | 0.043752 | 0.20917 | 245.534 | 9.06512e-15 | NOT_APPLICABLE |
| whpn | 1 | COMPLETED | 4.23788e-06 | 0.00205861 | 402.634 | 1.84953e-15 | FAILED_REPRESENTATION_COVERAGE |
| whpn | 5 | COMPLETED | 0.00155798 | 0.0394712 | 31.0865 | 9.29427e-10 | NOT_APPLICABLE |
| whpn | 10 | COMPLETED | 0.0254279 | 0.159461 | 30.7622 | 7.10684e-09 | NOT_APPLICABLE |
| whpn | 20 | COMPLETED | 0.316351 | 0.562451 | 30.7621 | 7.18481e-09 | NOT_APPLICABLE |
| cascaded_tanks | 1 | NOT_YET_RUN | — | — | — | — | NOT_APPLICABLE |
| cascaded_tanks | 5 | NOT_YET_RUN | — | — | — | — | NOT_APPLICABLE |
| cascaded_tanks | 10 | NOT_YET_RUN | — | — | — | — | NOT_APPLICABLE |
| cascaded_tanks | 20 | NOT_YET_RUN | — | — | — | — | NOT_APPLICABLE |
| silverbox | 1 | COMPLETED | 1.82502e-05 | 0.00427203 | 243.501 | 4.03709e-16 | COMPLETED |
| silverbox | 5 | COMPLETED | 0.24378 | 0.493741 | 243.502 | 2.14421e-14 | NOT_APPLICABLE |
| silverbox | 10 | COMPLETED | 0.453158 | 0.67317 | 8.04641 | 2.5285e-10 | NOT_APPLICABLE |
| silverbox | 20 | COMPLETED | 0.705567 | 0.83998 | 11.3988 | 4.00324e-10 | NOT_APPLICABLE |

## 固定模型的每-horizon Rank Bootstrap

| 数据集 | h | 状态 | B | 自动块长 | 最大 KKT | rank 频数（尾能量预算） |
|---|---:|---|---:|---:|---:|---|
| pwh | 1 | COMPLETED | 250 | — | 1.02733e-15 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |
| pwh | 5 | COMPLETED | 250 | — | 2.89219e-15 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |
| pwh | 10 | COMPLETED | 250 | — | 1.25636e-14 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |
| pwh | 20 | COMPLETED | 250 | — | 2.7675e-14 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |
| whpn | 1 | COMPLETED | 250 | — | 2.84019e-15 | 0.1:r1=250; 0.05:r1=246,r2=4; 0.02:r1=171,r2=79 |
| whpn | 5 | COMPLETED | 250 | — | 6.35117e-09 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |
| whpn | 10 | COMPLETED | 250 | — | 9.57522e-09 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |
| whpn | 20 | COMPLETED | 250 | — | 9.96559e-09 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |
| cascaded_tanks | 1 | NOT_YET_RUN | — | — | — | — |
| cascaded_tanks | 5 | NOT_YET_RUN | — | — | — | — |
| cascaded_tanks | 10 | NOT_YET_RUN | — | — | — | — |
| cascaded_tanks | 20 | NOT_YET_RUN | — | — | — | — |
| silverbox | 1 | COMPLETED | 250 | 66 | 6.4173e-16 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |
| silverbox | 5 | COMPLETED | 250 | 66 | 6.04421e-14 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |
| silverbox | 10 | COMPLETED | 250 | 66 | 2.91929e-10 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |
| silverbox | 20 | COMPLETED | 250 | 66 | 7.06241e-10 | 0.1:r1=250; 0.05:r1=250; 0.02:r1=250 |

## 文献基线与覆盖审计

| 数据集 | ARX history | ARX AIC | pNARX order | pNARX AIC | 状态 |
|---|---|---:|---:|---:|---|
| cascaded_tanks | nx=9, ny=8 | -582.312 | 3 | -511.283 | COMPLETED |
| silverbox | nx=10, ny=10 | -125218 | 5 | -294315 | COMPLETED |

| 数据集 | H3 spectral | 覆盖审计说明 |
|---|---|---|
| pwh | 已运行 | train-fitted 幅值域覆盖 validation；H1 free-run 递推后越界，按协议失败。 |
| whpn | 已运行 | train-fitted 幅值域覆盖 validation；H1 free-run 递推后越界，按协议失败。 |
| cascaded_tanks | BLOCKED | X validation 中 13/324 点超出 train-fitted 域；未裁剪、未使用 validation 扩域。 |
| silverbox | 已运行 | 幅值域覆盖；H1 free-run COMPLETED。 |

## WHPN AR-only 依赖轨道复核

| h | 状态 | validation MSE | EDF | KKT |
|---:|---|---:|---:|---:|
| 10 | COMPLETED | 0.0401728 | 105.222 | 1.27173e-13 |
| 20 | COMPLETED | 0.462305 | 18.8667 | 3.58631e-11 |

修复后的 h20 AR-only 候选通过原坐标 KKT 门槛；该轨道仅用于依赖完整性复核，不改变已经冻结的 XAR penalty 或 rank bootstrap。

## 停止线与尚缺的冻结项

1. H2 的 representation-coverage gate 和 Lepski stability 只有开关，没有预注册数值阈值；这些阈值会实质改变 history/resolution 选择。
2. Tanks 需要一条只由 train 决定的外推/域策略。当前协议明确禁止静默裁剪，也禁止根据 validation 扩张样条域。
3. 在上述两项冻结前，H2、Tanks spectral、统一 protocol freeze 和 official confirmation 均保持未运行。

## 结论边界

- 当前数值只属于 development validation，不是官方 test 结果。
- rank 是冻结表示与惩罚下的预测压缩审计，不是结构 rank 或因果发现。
- 未读取 official test、未运行 confirmation、未涉及 PB2 或私有 CZ。

## 复核信息

- 汇总源码提交：`7890e493058e30d8bc9e957ea4af32adbec4f0f6`
- 生成时间（UTC）：`2026-07-28T02:05:21.724110+00:00`
- 主项目回归：213 passed，16 skipped。
- V20 回归：118 passed。
- 机器可读状态：`PB1_DEVELOPMENT_REPAIR_V2_STATUS.json`
