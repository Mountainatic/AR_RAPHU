# SRU H2S：固定128步基线与自选窗口 PRISM

本目录归档2026-10-24完成的基线比较。AR、ARX、LSTM、Transformer、iTransformer、TimeXer固定128步直接输入；PRISM按模型设计从8、16、32、64、128步中按训练内验证误差选窗，主表保留完整在线A。另列Persistence和独立选窗的PRISM no_A。

|标签延迟|PRISM窗口|R²|RMSE|MAE|MAPE (%)|
|---:|---:|---:|---:|---:|---:|
|4|8|0.773695|0.02439936|0.01242093|16.1008|
|8|16|0.772797|0.02444773|0.01419648|19.7768|
|16|16|0.736667|0.02631989|0.01459136|20.1881|

以上是已多轮使用的开发段结果。每个延迟下，全部模型评估相同的2016个目标；共27组指标和54,432行逐点记录。基线神经预测为种子0/1/2均值。窗口选择仅使用训练段内部评分，全部15组候选、30套模型冻结后才计算本轮开发结果；既有基线逐点预测保持原值。

任务是同期软测量y(t)，当前X(t)可见，y最多到t-delay；4/8/16是实验性的标签成熟延迟，不是预测步长，也不是已核实的设备标签延迟。训练前缀[0,6048)，开发目标[6048,8064)，保留尾段[8064,10081)未读取。

完整PRISM A使用发行预测、成熟残差和RLS递归状态，间接历史可能超过所选直接窗口。此表的窗口搜索预算及在线更新预算有差异，不代表相同总历史信息约束下的比较，也不能作为独立最终测试或普遍优越性证明。iTransformer和TimeXer均为2024年模型，本归档不声称涵盖截至2026年的全部最新SOTA。

## 结果与代码

- [完整报告、全部模型结果与图表](REPORT_CN.md)。
- [R²/RMSE/MAE/MAPE等完整指标](results/metrics.csv)。
- [逐点预测CSV长表](results/point_predictions_long.csv)、[Parquet长表](results/point_predictions_long.parquet)。
- [延迟4逐点](results/point_predictions_D4.csv)、[延迟8逐点](results/point_predictions_D8.csv)、[延迟16逐点](results/point_predictions_D16.csv)。
- [训练内候选评分](WINDOW_CANDIDATES.csv)、[最终选窗与结构](results/selected_windows.csv)、[全部选择冻结门槛](ALL_SELECTIONS_FROZEN.json)。
- [PRISM选窗/训练/重放入口](prism_search.py)、[调度](launch.py)、[聚合与绘图](report.py)、[实际PRISM模型](prism/model.py)、[在线A](prism/mature_a.py)。实验源码逐字节保留。
- [与固定128步PRISM的描述性比较](results/prism_vs_fixed128.csv)，完整128步对照源代码、权重、候选及36个神经检查点在Release内嵌套的原基线包中。
- [独立指标复核](results/INDEPENDENT_VERIFICATION.json)、[未来X/未成熟y扰动重放](audits)、[发布范围](PUBLICATION_SCOPE.json)。

## 下载、恢复与验证

完整权重和候选账本使用[本轮Release](https://github.com/Mountainatic/AR_RAPHU/releases/tag/sru-baselines-prism-window-selection-20261024)下载，归档为41,850,934字节，SHA256为`caec24da4431f15568d400a93287c29ff53d752846138d4bc68164dcf1f02a08`。

Git只保存可直接浏览的代码/结果/轻量冻结收据，完整模型、公共数据前缀缓存及嵌套基线包在Release中；GitHub自动生成的Source code压缩包不能替代该实验资产。原始[REPORT_CN.md](REPORT_CN.md)中提及的runs、data、baseline_reference全部文件，是指恢复后的完整实验目录。

```bash
# 只使用Python标准库检查Git快照、指标、逐点对齐和训练内选窗
python sru_prism_window_selection_20261024/tools/verify_snapshot.py

# 下载Release实验资产后，校验全部147个原始成员，无需载入模型或读取尾段
python sru_prism_window_selection_20261024/tools/verify_bundle.py /path/to/SRU_PRISM_WINDOW_SELECTION_20261024.tar.gz
```

恢复和重放的完整步骤见[RELEASE_RESTORE.md](RELEASE_RESTORE.md)。PRISM训练和重放使用登记的Python3.10科学计算环境；Torch基线使用原包的Python3.12环境。无需SSH访问原服务器。原SRU第三方来源和哈希见[data/REFERENCE_EXPORT.json](data/REFERENCE_EXPORT.json)及原始预登记；本发布不另行授予第三方数据许可。新增内容不含私有CZ原始数据、保留尾段或凭据。

![训练内选窗](results/window_selection.png)

![完整指标比较](results/metrics_comparison.png)
