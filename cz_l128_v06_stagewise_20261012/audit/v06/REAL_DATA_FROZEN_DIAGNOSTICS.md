# Phase 4 — 冻结真实资产只读诊断

没有重训、测试集拟合或白化选参。C1 source base/metadata 与 CPU frozen_validation 使用既有 SHA；目标测试未打开。原始源棒记录允许只读审计，结果不反馈训练或跨方向选择。谱图按 train/validation、entity、连续 origin 分开；缺口和非有限值再分段。

下表“附近peak”仅是 .133–.153Hz 区间最大值，任何连续区间有最大值，不能把它自动当显著峰。PSD/ACF/PACF、频率分辨率和区间能量详见各方向 FROZEN_SPECTRAL_REPORT.json 与频谱CSV。已画每个split最长段，所有段均保留表。

post-K/C/W/A innovation：NOT_AVAILABLE_CURRENT_L128_DEPLOYMENT_NOT_FROZEN。上一轮因低磁盘停止在K候选搜索，不能把内层候选或旧L256/同棒Ridge资产改名为最终跨棒输出。最终样本外post-A创新的白化结论 NOT_AVAILABLE；不写“白化已改善/峰被消除”。目标test仍受两方向主模型和全部基线冻结屏障保护。

## Rod_1_to_Rod_2 — SOURCE_READ_ONLY_DIAGNOSTICS_COMPLETED
源棒 Rod_1；已验 hash 的冻结 CPU 源验证预测 14 份，单独标 baseline_frozen_residual，不能冒充 PRISM post-A。
|head/split|rows|legal|warning|
|---|---:|---:|---:|
|H1/train|15431|15431|0|
|H4/train|15431|15425|6|
|H1/validation|3601|3601|0|
|H4/validation|3601|3598|3|
|head/split/quantity/segment|N|全局peak Hz|附近peak Hz|附近能量比例|分辨率 Hz|
|---|---:|---:|---:|---:|---:|
|H1/validation/recorded_level/Rod_1_segment_3|3601|0.00027770064|0.13926687|3.36093e-05|0.00013885032|
|H1/validation/registered_target/Rod_1_segment_3|3601|0.00027770064|0.13926687|0.21297|0.00013885032|
|H4/validation/recorded_level/Rod_1_segment_3|3598|0.00027793218|0.13979989|3.25704e-05|0.00013896609|
|H4/validation/registered_target/Rod_1_segment_3|3598|0.00027793218|0.13979989|0.00959378|0.00013896609|
## Rod_2_to_Rod_1 — SOURCE_READ_ONLY_DIAGNOSTICS_COMPLETED
源棒 Rod_2；已验 hash 的冻结 CPU 源验证预测 14 份，单独标 baseline_frozen_residual，不能冒充 PRISM post-A。
|head/split|rows|legal|warning|
|---|---:|---:|---:|
|H1/train|16092|16092|0|
|H4/train|16092|16089|3|
|H1/validation|3766|3766|0|
|H4/validation|3766|3763|3|
|head/split/quantity/segment|N|全局peak Hz|附近peak Hz|附近能量比例|分辨率 Hz|
|---|---:|---:|---:|---:|---:|
|H1/validation/recorded_level/Rod_2_segment_1|3766|0.00013276686|0.13794477|0.000397967|0.00013276686|
|H1/validation/registered_target/Rod_2_segment_1|3766|0.076739246|0.13794477|0.192039|0.00013276686|
|H4/validation/recorded_level/Rod_2_segment_1|3763|0.00013287271|0.13792187|0.000394591|0.00013287271|
|H4/validation/registered_target/Rod_2_segment_1|3763|0.076800425|0.13792187|0.00811519|0.00013287271|

记录小数格点只能作为候选（原始比例/步长分类见OBSERVATION_AUDIT.json），没有ADC说明。真实时间戳缺失，不能确认设备采样抖动或隐含高频；2秒为原协议约定。粗略iid ACF参考带±1.96/sqrt(N)仅描述，时间依赖/参数估计/多重lag影响解释，不用于模型接纳。

