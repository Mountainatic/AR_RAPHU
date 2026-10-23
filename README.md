# SRU H2S 条件残差反馈 A：2026-10-23 开发归档

最新入口：[代码、逐样本预测、三块消融及复核报告](sru_a_strengthening_20261023/README.md)。完整冻结模型与候选训练账本见 [Release](https://github.com/Mountainatic/AR_RAPHU/releases/tag/sru-conditional-feedback-a-20261023)。当前属于开发证据，未开启本轮保留尾段。

---

# CZ L128 v0.6 与逐层冻结实验归档

本分支新增 [cz_l128_v06_stagewise_20261012](cz_l128_v06_stagewise_20261012)：2026-10-12 完成的实验代码、汇总指标、图表和审计证据，2026-10-16 发布。

| 入口 | 内容 |
|---|---|
| [归档说明与范围](cz_l128_v06_stagewise_20261012/README.md) | 最新版本、核心结果、协议差异和限制 |
| [实际运行的实验源码](cz_l128_v06_stagewise_20261012/runtime) | canonical K/C/W/A、逐层训练、基线及 E2–E6 适配器 |
| [跨棒 E1 与基线](cz_l128_v06_stagewise_20261012/results/crossrod/RESULTS_CN.md) | 两方向差分预测指标与对照图 |
| [新版 E1 同棒白化](cz_l128_v06_stagewise_20261012/results/whitening/REPORT_CN.md) | 两棒 Common4，逐阶段残差诊断 |
| [复现说明](cz_l128_v06_stagewise_20261012/REPRODUCIBILITY.md) | 公开包检查、轻量回归、仍需私有资产的正式流程 |

v0.6 已实施离散观测合同、时间索引和分段诊断，保持模型数值行为。**它没有加入量化感知预测，约 7 秒峰仍存在。** E3 为线性多尺度诊断，E6 为紧凑噪声代理，不能代表完整 E1 主模型的对应重训。

公开数据限定为汇总指标、谱/相关性统计、图表和回执；不包含原始测量、逐样本预测/残差、样本成员、权重或密钥。

原 `cz_h1_light_diagnostics_20261006` 目录保留历史内容，不能混同为本次完整 E1。仓库根部旧 AR-RAPHU v2 文档保留历史背景；本次具体 CZ H1/L128 协议、目标与证据以新目录的 release 和范围声明为准。
