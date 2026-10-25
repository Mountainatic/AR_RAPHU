# SRU H2S：所有模型采用PRISM选定的相同直接历史窗口

本轮按用户要求，AR、ARX、LSTM、Transformer、iTransformer、TimeXer都使用完整PRISM事先选定的窗口：**标签延迟4步用8步历史，延迟8和16步用16步历史**。完整PRISM在线A保留，另列相同窗口的no_A和Persistence。这与上一轮“基线固定128、PRISM自选窗口”的表不同。

在新RTX4090实例上重新进行了全部训练内选择和基线拟合，没有复用既有基线权重。四类神经模型各训练种子0/1/2，主表为逐点均值；AR/ARX为确定性ridge。模型候选、训练目标日期、前向折、保守隔离128+delay、早停、原随机种子和严格FP32设置均与首次同窗口实验保持一致。完整PRISM及同窗口no_A重放已有冻结权重，没有重新选窗或重新挑A。

|模型|延迟4 / 窗口8 RMSE|延迟8 / 窗口16 RMSE|延迟16 / 窗口16 RMSE|
|---|---:|---:|---:|
|AR|0.03862277|0.04781683|0.05097042|
|ARX|0.02895716|0.03222121|0.03284921|
|LSTM|0.06525508|0.06712233|0.03615414|
|Transformer|0.02681927|0.02490628|0.02620405|
|iTransformer|0.03875445|0.04809020|0.05195036|
|TimeXer|0.03876828|0.05186322|0.05375727|
|PRISM|0.02439936|0.02444773|0.02631989|

完整R²、RMSE、MAE、MAPE、sMAPE、WAPE和bias见[metrics.csv](results/metrics.csv)。每模型同样2016个开发目标，共27组指标、54,432行逐点预测。全部结果原样保留，包括负R²和退化模型，不按开发段结果重新挑配置或删种子。

## 时间协议和结论边界

任务是同期软测量y(t)，X至t可见，y至t-delay可见。直接输入为x(t-L+1)..x(t)和y(t-delay-L+1)..y(t-delay)。AR仅用L个成熟输出，ARX用L个输出及5L个过程值，神经模型用完整六通道L步序列。训练[0,6048)，开发目标[6048,8064)，保留尾段[8064,10081)未读取。4/8/16是实验性标签成熟延迟，不是预测步长或已确认的设备延迟。

统一的是直接历史跨度和标签可用性；PRISM保留分阶段训练、已有窗口搜索及在线A，成熟残差/发行响应/RLS可间接使用更早历史，不能称相同总记忆或相同在线更新预算。开发段此前已多轮使用，当前不是独立最终测试。使用相同种子进行计算复跑，也不是新的独立随机种子样本。

首次同窗口实验在RTX5090上完成，本次在用户提供的RTX4090实例重新拟合；两次结果都保留。[复跑对照](results/comparison_with_first_matched.csv)用于描述硬件/训练轨迹差异，不代表独立数据验证。两次均沿用已审计的严格FP32后端并保留原数值容限；[本次拟合前CPU FP64参考核对](MIGRATION_CHECK.json)和[首次实验登记](previous/first_matched_registration.json)提供精确设置。

LSTM在两次相同窗口复跑间差异较大，例如延迟8步RMSE由首次0.03060984变为本次0.06712233；不能把单轮神经结果当作稳定排名。两次使用相同种子但硬件/训练轨迹不同，这些数据不能单独确定差异原因。PRISM指标相同来自重放同一冻结模型，不是重新训练稳定性证明。两轮全部保留，未按开发成绩选择更有利版本。

## 文件入口

- [完整报告](REPORT_CN.md)、[全部指标](results/metrics.csv)、[逐种子指标](results/metrics_by_seed.csv)、[种子均值/标准差](results/metrics_seed_mean_std.csv)。
- [逐点预测CSV长表](results/point_predictions_long.csv)、[Parquet长表](results/point_predictions_long.parquet)。
- [延迟4逐点](results/point_predictions_D4.csv)、[延迟8逐点](results/point_predictions_D8.csv)、[延迟16逐点](results/point_predictions_D16.csv)。
- [基线训练/评估代码](experiment.py)、[各模型适配器](baseline_models.py)、[官方iTransformer/TimeXer源码](vendor/TimeXer/models)、[PRISM重放](prism_replay.py)、[实际PRISM源码](prism)。源码保持原字节。
- [冻结门槛](ALL_FITS_FROZEN.json)、[独立复核](results/INDEPENDENT_VERIFICATION.json)、[神经因果核查](results/CAUSAL_AUDIT.json)、[PRISM因果核查](audits)。
- [与旧固定128步结果比较](results/comparison_with_fixed128.csv)、[首次同窗口比较数据](previous/first_matched_metrics.csv)、[当前完整包成员清单](FULL_EXPERIMENT_MANIFEST.json)、[发布范围](PUBLICATION_SCOPE.json)。

## 完整权重与恢复

[本轮开发版Release](https://github.com/Mountainatic/AR_RAPHU/releases/tag/sru-matched-window-baselines-20261025)同时提供首次同窗口实验和本次新服务器复跑的完整包。当前主表对应`SRU_BASELINES_MATCHED_WINDOWS_REPEAT_20261025.tar.gz`，首次结果对应`SRU_BASELINES_MATCHED_WINDOWS_20261025.tar.gz`。每包均包含36个新神经检查点、6个新线性模型、6个既有PRISM检查点、完整训练候选、数据前缀缓存和逐种子预测。

Git保存可浏览源码、指标、逐点预测和冻结收据；完整模型及训练账本在Release。GitHub自动生成的Source code压缩包不能替代实验资产。先按Release的SHA256SUMS.txt核验，再依[恢复说明](RELEASE_RESTORE.md)操作；原[实验复现说明](REPRODUCIBILITY.md)也保留。

```bash
python sru_matched_window_baselines_20261025/tools/verify_snapshot.py
```

这个标准库入口只检查Git快照、源码/冻结哈希、训练内选择、全指标和逐点对齐，不重训或读取原始保留尾段。iTransformer与TimeXer均为2024年模型，固定作者仓库提交76011909357972bd55a27adba2e1be994d81b327，不声称涵盖截至2026年全部最新SOTA。原第三方SRU来源见data/REFERENCE_EXPORT.json，发布不另行授予第三方数据许可；不含私有CZ数据或凭据。

![指标](results/metrics_comparison.png)
