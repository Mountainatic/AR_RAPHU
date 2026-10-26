# SRU / CZ E128：2026-10-26 实验整理

最新入口：[代码、结果和协议](prism_paper_sru_cz_e128_20261026/README.md)。包含 SRU 消融、两根棒的全模型比较、E128 历史控制、Rod1 模块诊断、输入贡献及复杂度。Rod2 模块消融按本次指定暂缓；不代表该组通过。公开包不含私有 CZ 数据、逐点轨迹或模型权重。

[可下载归档](https://github.com/Mountainatic/AR_RAPHU/releases/tag/prism-paper-sru-cz-e128-20261026)；现有证据属于重复开发结果，不作为独立测试或正式 SOTA 证明。

---

# SRU H2S 全模型同窗口基线：2026-10-25

最新入口：[SRU完整报告、指标和逐点预测](sru_matched_window_baselines_20261025/README.md)。全部基线采用PRISM预先选定的直接历史：标签延迟4/8/16步分别用8/16/16步窗口，完整PRISM在线A保留。当前表为新RTX4090实例的完整重训，首次同窗口实验也一并保留。

两个完整模型与候选账本实验包见[开发版Release](https://github.com/Mountainatic/AR_RAPHU/releases/tag/sru-matched-window-baselines-20261025)。数据仍为已使用开发段，独立尾段未打开。

---

# SRU H2S 基线比较与PRISM自选窗口：2026-10-24

最新入口：[完整报告、基线指标和逐点预测](sru_prism_window_selection_20261024/README.md)。基线固定128步；PRISM按训练内验证从8/16/32/64/128选窗，标签延迟4/8/16分别选择8/16/16步，主表保留完整在线A。共27组指标、54,432行逐点记录。

完整模型、全部候选账本及原基线冻结包见[开发版Release](https://github.com/Mountainatic/AR_RAPHU/releases/tag/sru-baselines-prism-window-selection-20261024)。这是已使用开发段的比较，独立保留尾段未打开；历史结果和标签保持原状。

---

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

