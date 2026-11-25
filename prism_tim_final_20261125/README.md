# PRISM IEEE TIM 实验 E1–E5

本轮以 **CZ Rod1** 和 **SRU H₂S** 为主实验，以 **CZ Rod2** 为多步长漂移专项。主协议分别为 `(H,D,L)=(16,0,128)` 和 `(0,8,64)`。分数属于同实体时间尾段复核。报告披露 SRU 的历史评分接触和两根晶棒的历史重叠，并提供 CZ 剔除已知重叠目标后的结果。

[完整报告](REPORT_CN.md) · [下载 ZIP 与校验文件](https://github.com/Mountainatic/AR_RAPHU/releases/tag/prism-tim-final-20261125)

| 实验 | 结果 |
| --- | --- |
| E1 主基线 | [六指标与重复统计](E1/METRICS.csv) · [DM 与 Holm 校正](E1/DM.csv) |
| E2 核心消融 | [两套主协议的六组消融](E2/ABLATION.csv) |
| E3 预测步长与漂移 | [共同目标行结果](E3/HORIZON.csv) · [步长图](figures/E3_common_target_horizons.svg) · [Rod2 回退开关对照](E3/ROD2_FALLBACK.csv) · [阶跃轨迹](figures/E3_Rod2_step_drift.svg) |
| E4 FP64 计时与长时验证 | [同核延迟](E4/NATIVE_LATENCY.csv) · [精度–延迟图](figures/E4_native_precision_latency.svg) · [长窗口曲线](figures/E4_window_scaling.svg) · [千万步稳定性](E4/LONG_STABILITY.csv) · [前缀和探针](E4/PREFIX_PROBE.csv) |
| E5 机制与敏感性 | [C/W 对齐](E5/ALIGNMENT.csv) · [恒定偏置](E5/BIAS_PRESSURE.csv) · [加法归因](E5/ATTRIBUTION.csv) · [因果和 Python/C 一致性](E5/CAUSAL.csv) · [阈值敏感性](figures/E5_fallback_sensitivity.svg) |

## 代码与复核

[原生 C 内核](stream.c) 每 65,536 步使用合法的过程历史重锚前缀和。[共享入口](core.py)、[基线评分](e1.py)、[Transformer 重扫](tf_sweep.py)、[消融与机制实验](evidence.py)、[原生计时](e4.py)、[数值探针](prefix_probe.py) 和 [报告生成](report.py) 保留原实验入口。逐项复跑需在授权环境中恢复清单绑定的历史引擎、模型和数据快照。公开目录提供代码、选型回执、汇总表和图件，原始观测、权重与逐点预测保存在实验服务器。

- [代码清单](FINAL_CODE_LOCK.json) SHA-256 `52ec8b0dfe871a7266588d30a7e44e56e9d0d121b13aa3d2c53126323aa43797`
- [结果清单](RESULTS_LOCK.json) SHA-256 `a4f520e13d1647b97b9433f0f87560e55fe79d44b114a7c33444701fd2e08e93`
