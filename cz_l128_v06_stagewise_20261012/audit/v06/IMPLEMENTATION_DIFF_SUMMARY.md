# 最小实现改动及重跑边界

已有 runtime 文件与 baseline_runtime 全部字节一致；authority数值代码未修改。新增三个默认未导入的只读模块、离线CLI、tests及七份交付报告。选择规则、K候选/编码、C装配、W坐标、成熟A残差、H1/H4数值和训练/评分行均未改变。元数据独立于K的Q_j；没有 PRISM-Q 模型。

用户要求改完后重做实验：新版须以单独 release、空正式输出/缓存、L128两方向跨棒和既有源开发/目标屏障运行；不续接旧未完成K搜索。数据盘还在复制；本次验收执行只读诊断，没有启动正式重训。上一轮可删除的范围明确为旧 formal_actual 两方向候选/中间缓存；代码、raw、C1、冻结基线和冻结逐样本资产保留。清理需复制完成及无相关活动进程后执行；新产物路径不属于清理目标。

重跑范围：E1主方法两方向、CPU/深度基线及逐样本；E2–E6按原登记协议分别执行，不能用E1/基线完成代替。RERUN_PLAN.json 为准备状态，不是运行回执。本离散兼容验收不触发大规模计算。回退：使用 baseline_runtime 和旧release；三个新模块不在旧数值调用链。

## 强制六问

1. 已在离散index建模吗？是。_sample_rows依赖origin和target_index；BaseAccessor.block_means以排他origin取原始离散历史；Urysohn theta为有限lag×amplitude；a_state检查成熟target_end。缺少的是显式观测元数据和审计解释，不是离散化从零实现。
2. H1真实公式？D[t]-D[t-1]，历史到t-1，母版W=W0=1,h=0特例。项目H4为D[t+3]-D[t-1]，母版h=3。保留真实构造器。
3. .01是什么？只读观测格点候选；比例来自数据而不是仪表，instrument_verified=False。不能称ADC精度，单位也未认证。
4. 无真实7秒正弦能复现什么？斜坡量化可产生规则步长和q点周期/谐波；q=7是14秒，可能有7秒谐波；混叠可落在约.142857Hz。不同斜率/噪声/N会改变峰，不能据此复现实测完整动力学或识别真实根因。
5. 真实7秒峰在哪/来源确定吗？源棒recorded_level与H1/H4 registered_target的各段PSD具体数值在REAL_DATA_FROZEN_DIAGNOSTICS；raw与target应分开解释。最终跨棒样本外post-A创新不可用；附近频带最大值不是显著峰证据。现有结果不足以确定量化、真实振荡或alias来源。
6. 改善什么/哪些属于以后？新增可回退观测合同、原始索引/成熟时间解释、非法支持警告、严格分段频谱与合成反事实、数值不变回归。未增加隐含连续状态识别、量化似然、Bayesian filter、去量化或新route门禁；它们属于后续PRISM-Q独立研究。

