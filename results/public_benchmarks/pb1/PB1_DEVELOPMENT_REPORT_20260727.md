# OPS-UOI PB1 开发阶段技术报告

## 技术摘要

**结论：当前工作到达开发阶段停止线，但 PB1 整体尚未完成；正式状态为 `PB1_CONFIRMATION=NOT_YET_RUN`、`WHPN_DEVELOPMENT=FAILED`。** PWH 的已注册开发轨道全部完成；WHPN 的 XAR 轨道在四个直接预测 horizon 上均完成，但 AR-only 在 h=10 的正则区间认证和 h=20 的 KKT 门槛失败，因此这两个 horizon 的正式 `Delta_X_given_AR` 不报告。官方 test 从未读取，因而本文不是 confirmation 结果，也不构成最终公开基准结论。

在开发验证上，PWH 的过程输入相对 AR-only 在 h=1/5/10/20 分别降低 71.49%/88.80%/88.18%/69.22% 的 MSE，说明外生输入含有稳定的增量预测信息。WHPN 在 h=1/5 的对应降低为 26.27%/41.03%；h=10/20 虽然 XAR 点估计更低，但因依赖门槛失败不能形成冻结结论。所有 rank 结果只解释为当前基函数、惩罚和开发分布下的预测有效秩，不允许解释为系统结构真秩。

## 一步预测中，Spectral XAR 与 pNARX 接近

下表使用相同的无未来 X 信息集、相同 train-only 目标 z-score 和相同开发记录，单位为 pooled RMSE。表格优先于趋势图，因为每个数据集只有一个一步预测比较点，绘图不会增加信息。

| 数据集 | Persistence | ARX | pNARX | MLP-NARX | Spectral XAR |
|---|---:|---:|---:|---:|---:|
| PWH | 0.060082 | 0.002784 | 0.002485 | 0.009214 | 0.002476 |
| WHPN | 0.078105 | 0.002063 | 0.002061 | 0.008838 | 0.002059 |

PWH 上 Spectral XAR 的 RMSE 为 0.002476，较 pNARX 低约 0.33%，较 ARX 低约 11.0%；WHPN 上相对 pNARX 的开发优势约 0.11%，幅度很小。MLP-NARX 在两组数据上均明显较差，不能据此宣称神经基线具有优势。

## PWH 的外生增量在全部开发 horizon 上成立

| h | AR MSE | AR 状态 | XAR MSE | XAR 状态 | Delta(X|AR) | 相对 AR 降低 | 预测秩(10%/5%/2%) |
|---:|---:|---|---:|---|---:|---:|---|
| 1 | 2.150600e-05 | COMPLETED | 6.132235e-06 | COMPLETED | 1.537377e-05 | 71.49% | 1/1/1 |
| 5 | 0.001202 | COMPLETED | 1.346999e-04 | COMPLETED | 0.001068 | 88.80% | 1/1/1 |
| 10 | 0.011452 | COMPLETED | 0.001354 | COMPLETED | 0.010098 | 88.18% | 1/1/1 |
| 20 | 0.142127 | COMPLETED | 0.043752 | COMPLETED | 0.098375 | 69.22% | 1/1/2 |

PWH 四个 horizon 的 AR 与 XAR 都通过惩罚区间和数值门槛。过程变量的边际价值在中等 horizon 最大；h=20 在严格 2% 尾能量预算下点估计升为 rank-2，但这仍是预测压缩需求，不是结构 rank 发现。

## WHPN 在 h=10 和 h=20 只能保留点估计

| h | AR MSE | AR 状态 | XAR MSE | XAR 状态 | Delta(X|AR) | 相对 AR 降低 | 预测秩(10%/5%/2%) |
|---:|---:|---|---:|---|---:|---:|---|
| 1 | 5.747472e-06 | COMPLETED | 4.237888e-06 | COMPLETED | 1.509584e-06 | 26.27% | 1/1/1 |
| 5 | 0.002637 | COMPLETED | 0.001555 | COMPLETED | 0.001082 | 41.03% | 1/1/1 |
| 10 | 0.040173 | FAILED | 0.025624 | COMPLETED | 未报告 | 未报告（依赖门槛失败） | 1/1/1 |
| 20 | 0.462997 | FAILED | 0.316966 | COMPLETED | 未报告 | 未报告（依赖门槛失败） | 1/1/1 |

WHPN h=10 的 AR penalty interval 在允许的两次边界扩展后仍未认证；h=20 的 AR relative KKT residual 为 1.718e-8，高于冻结的 1e-8。因此两个 horizon 的 XAR 数值可作为开发诊断保留，但不得计算或引用正式增量收益。

## Bootstrap 支持低预测秩，但不支持结构秩声明

- **PWH**：10% 尾能量预算下 rank-1: 250/250; 5% 尾能量预算下 rank-1: 250/250; 2% 尾能量预算下 rank-1: 250/250。
- **WHPN**：10% 尾能量预算下 rank-1: 250/250; 5% 尾能量预算下 rank-1: 246/250, rank-2: 4/250; 2% 尾能量预算下 rank-1: 171/250, rank-2: 79/250。

bootstrap 固定已选惩罚和分辨率，不在重采样中重新调参。PWH 在全部 250 次重采样和三个尾能量预算下均为 rank-1；WHPN 在 2% 严格预算下有 79/250 次需要 rank-2，说明细尾部存在不确定性。由于公开数据没有 K 层真值证书，`structural_rank_claim_allowed=false`。

## 范围、数据和指标定义

- 范围仅包括 PWH 与 WHPN 的 development split；Cascaded Tanks 仍为 `PENDING_SPLIT_ADEQUACY_AUDIT`，Silverbox 为 `BLOCKED_BY_MISSING_METADATA`。
- 主预测协议是 direct forecast：X 与 y 只使用到时刻 t，预测 y[t+h]；不使用未来 X，也不使用中间真实 y。
- `Delta_X_given_AR = MSE_AR - MSE_XAR`；只有 AR 与 XAR 两个依赖轨道同时通过冻结门槛时才报告。
- 所有 scaler、历史和惩罚选择只使用 train/development validation；官方 test 访问计数为 0。

## 模型与验证方法

一步比较包含 persistence、线性 ARX、2024 文献配置的 pNARX、MLP-NARX 与 FP64 Spectral XAR。Spectral 轨道使用共享 H1 历史 (PWH Lx=16, Ly=20；WHPN Lx=18, Ly=15)、固定首个预注册分辨率、归一化三惩罚和 grouped validation one-SE。rank 在模型与惩罚冻结后才计算，并用 250 次按 phase/realization 聚类的 bootstrap 检查稳定性。

## 限制、失败项与鲁棒性边界

- PB1 confirmation、官方 test 和 OOD test 均为 `NOT_YET_RUN`。
- H2 native-history 与完整 basis-resolution 选择尚未冻结；不能用当前 H3 pilot 冒充最终配置。
- WHPN h=10/h=20 的 AR 依赖门槛失败，禁止补写正式增量结论。
- Tanks 的 overflow 样本级定义缺失；Silverbox 许可证元数据未解决。
- WHPN 的过程噪声专用 GRU/状态基线尚未实现。
- 结果只支持预测层面的开发证据，不支持因果、结构真秩或官方 benchmark 优胜声明。

## 下一步必须先解决冻结前置条件

1. 预注册 H2 history 与 basis-resolution 的嵌套选择顺序及候选空间。
2. 决定 WHPN h=10 penalty 边界失败与 h=20 KKT 失败的预注册处理方式，不得事后按结果扩网格或放宽阈值。
3. 补齐 Tanks overflow 元数据/替代门槛、Silverbox 许可证和 WHPN 过程噪声对照。
4. 上述条件冻结后生成 `PB1_PROTOCOL_FREEZE.json`，再一次性运行 confirmation；此前继续保持官方 test 锁箱。

## 可复核性

- 打包前源码提交：`20469ab0eeaf00c3e65e7ebb880042a976235b2e`。
- 结果状态文件：`PB1_DEVELOPMENT_STATUS_20260727.json`。
- 本报告及所有逐模型 JSON/NPZ、配置、相关源码、工具和测试均收入开发结果包；压缩包内 `PACKAGE_MANIFEST.json` 给出逐文件 SHA256。
