# CZ H1 L128：v0.6 离散观测兼容、逐层冻结 E1 与白化诊断

实验完成：2026-10-12；归档发布：2026-10-16。

本目录归档实际执行的代码、汇总指标、频谱/相关性统计、图表和审计回执。原始工作簿、逐样本目标/预测/残差、样本成员、训练权重和候选缓存不在公开范围内。

**当前结论：v0.6 修改已实施并用于诊断，但没有增加针对量化机制的预测修正。约 7 秒谱峰仍存在，不能宣称已消峰、已全域白化或已确定物理来源。**

## 结果入口

- [跨棒 E1 与基线结果](results/crossrod/RESULTS_CN.md)、[164 行汇总指标](results/crossrod/CROSSROD_E1_BASELINE_METRICS.csv)、[对照图](results/crossrod/E1_TEST_COMPARISON.png)。
- [使用最新完整 E1 的同棒白化报告](results/whitening/REPORT_CN.md)、[逐阶段汇总](results/whitening/WHITENING_METRICS.csv)、[A 相对 W 的变化](results/whitening/A_VS_W_COMPARISON.csv)。
- [E2–E6 原登记范围结果](results/experiments)、[跨棒分段 PSD/ACF/PACF](results/crossrod/frozen_v06_diagnostics)、[同棒分段诊断](results/whitening/v06_diagnostics)。
- [v0.6 合成反事实](results/v06_synthetic)、[v0.6 实施与回归证据](audit/v06)、[原实施文档](docs/PRISM_v0_6_Discrete_Domain_Minimal_Implementation_Agent_Prompt.md)。
- [最终完成回执](audit/main/RUN_COMPLETION_RECEIPT.json)、[源码与资产来源](SOURCE_PROVENANCE.json)、[复现范围](REPRODUCIBILITY.md)。

### 跨棒 E1

| dynamic 模型 | Rod1→Rod2 差分 RMSE | R² | Rod2→Rod1 差分 RMSE | R² |
|---|---:|---:|---:|---:|
| PRISM A | 0.006301 | 0.2261 | 0.007274 | 0.0871 |
| AR/ARX | 0.006759 | 0.1095 | 0.007080 | 0.1350 |
| PRISM K | 0.006950 | 0.0584 | 0.008007 | −0.1064 |
| Persistence | 0.007164 | −0.0003 | 0.007613 | −0.0002 |

Rod1→Rod2 PRISM A 更好；Rod2→Rod1 AR/ARX 更好。深度基线保留每任务 10 种子，其表格是指标均值，不是预测 ensemble。

### 同棒描述性尾段：A 相对 W

| arm | RMSE 下降 | E160 下降 | E900 下降 | 0.125–0.155 Hz 能量下降 | A 最大峰周期 |
|---|---:|---:|---:|---:|---:|
| Rod1_Common4 | 9.95% | 67.80% | 66.59% | 37.20% | 约 7.237 秒 |
| Rod2_Common4 | 8.26% | 75.21% | 72.50% | 55.05% | 约 7.237 秒 |

这些是 A 相对 W 的变化，**不是 v0.6 开/关的抑峰效果**。E160/E900 是 ACF 平方和，不是误差能量；lag7 是 14 秒。Rod1_AllLegal 未纳入新版四通道模型，未执行。

## 实际代码与版本

执行版为私有运行目录 `CZ_CROSSROD_L128_STAGEWISE_V3_20261012` 的精确源码快照。主 release SHA256：

```text
a1109dc5d5e3b6bc0f00ed018b4399ad87b2bf3691f6790c5d881e1b48b3f507
```

基础 CZ 诊断分支是 `cz-h1-light-diagnostics-20261007`，本次 Git 归档基于提交 `2d4f8f7ccc925b7a26627ba97e97c8f76cdc9917`。逐层冻结方式参考 `prism-unified-hw-r2-protocols-20260921` 的 `6ba6f1b510cc3476f9dbd1c64e86b07710b239da`，适配到既有 CZ canonical 数值实现；不是直接运行该分支原样代码。

- [StagewisePipeline](runtime/code/canonical/cz_crossrod_l128_20261008/stagewise_engine.py)、[逐层调参](runtime/code/canonical/cz_crossrod_l128_20261008/stagewise_tuning.py)。
- [离散观测合同](runtime/code/canonical/cz_crossrod_l128_20261008/discrete_observation.py)、[时间索引审计](runtime/code/canonical/cz_crossrod_l128_20261008/discrete_audit.py)、[分段诊断](runtime/code/canonical/cz_crossrod_l128_20261008/discrete_diagnostics.py)。
- [推理加速](runtime/fast_inference.py)、[同棒重跑入口](whitening/rerun_new_e1.py)、[白化统计](whitening/whitening_statistics.py)。
- [E2–E6 适配器](runtime/research_extension/authority_extension.py)、[基线适配器](runtime/baseline_adapter)。

`runtime/` 是封存源快照。源棒训练使用 42 份精确兼容 K 复用，C/W/A 重新拟合；基线 28 个 CPU 部署与 120 个深度种子 checkpoint 校验后复用，目标预测重新生成。源训练约 266.55/279.77 秒，以已有 K 为前提，不是从零耗时。同棒白化主模型 fit=0、固定诊断 probe fit=8。

## 实验范围与限制

1. 两棒输入历史均 128 点；H1 为 `D[t]-D[t-1]`，输入最后可见点 t−1；2 秒为注册约定。A 历史源棒开发选择后固定为 Rod1=96、Rod2=32 点。
2. 完整 E1 两方向、CPU/深度基线及分阶段预测已完成。E2/E4 是合成实验；E3 是线性多尺度诊断；E5 汇总结构稳定性；E6 是紧凑噪声代理。**E3/E6 不等于完整主模型重训或噪声鲁棒性试验。**
3. 新版 C/W/A 是外层 fit 本地配方选择、固定配方内层参数 CF；**不是 fully nested hyperparameter CF，也不与旧递归协议完全等价**。
4. v0.6 保持模型数值行为，不根据格点/白化统计改 route，不强制取整，不添加量化似然或隐状态模型。0.01 只是观测格点候选，未经仪器认证。
5. 历史目标信息/尾段曾被查看，属于回顾性协议开发与描述性评价，不是未触碰 lockbox 的确认性试验。
6. 同棒 Welch 频带 0.125–0.155 Hz 与跨棒诊断 0.133–0.153 Hz 不同，统计方法亦不同，不能直接横比绝对能量和频带占比。

## 公开归档校验

只使用 Python 标准库，不加载私有数据或训练模型：

```bash
python verify_publication.py
```

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) 逐文件登记字节数和 SHA256；[PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json) 记录本次公开包与轻量回归检查。原始 release 和审计回执中保留的绝对路径、哈希、运行 PID 是历史证据，**不是公开文件链接，也不代表相关私有资产已上传**。

2026-10-16 新服务器主目录仍匹配上述封存。迁移后的白化目录为空，因此本次白化公开部分取自 2026-10-12 已下载到本机、按原清单全部验证的 215 文件归档；公开时仍只选取汇总、谱统计、图和回执。没有重新训练或重新计算结果。
