# SRU H2S：条件残差反馈 A 的代码和开发结果

此目录归档2026-10-23完成的3个结构分支、9个设置、45条最终配置和186次候选A搜索，包含全部阴性结果。原始实验代码逐字节保留；`tools/`只提供发布校验和不重新拟合的A重放入口。

任务是同期估计y(t)：当前过程传感器x(t)可见，当前y(t)不可见，目标经过假定4/8/16个采样步后成熟。该步数是标签延迟，不是预测未来步长，也不是经核实的分析仪延迟。

|V9标签延迟|完整RMSE|去KC增加%|去W增加%|去A增加%|比保留最强基线降低%|
|---|---:|---:|---:|---:|---:|
|4|0.023721836|62.185|32.924|17.266|13.235|
|16|0.021331180|177.453|135.958|55.682|5.570|

三块删除均独立重训和重新选择，全段及固定启动描述性区间均正。4步保留有限Urysohn KC；16步选择线性过程KC。V9的A用当前已发行响应、历史差分与一条成熟残差的交互项改变条件反馈系数，保留报告的标准化ridge100/stored-phi RLS核心；这是结构改进，不能称未修改v0.6或阻尼正余弦+AR版。

![开发候选](review/SRU_A_strengthening.png)

当前开发段已反复用于研究，区间未校正搜索次数；完整对最强保留基线的配对区间仍跨零，所以只能称开发均值领先。旧SRU来源历史测试暴露没有完全查清，不能称新独立数据。本轮保留尾段8064–10080没有打开。冻结更新控制及阶段训练预算限制见完整报告。

## 文件入口

- [完整报告](review/REPORT_CN.md)、[全分支数值与区间](review/COMBINED_REVIEW.json)。
- [V9实际模型代码](code/conditional_feedback/model.py)、[V9 A实现](code/conditional_feedback/mature_a.py)、[原RLS核心](code/conditional_feedback/rls_core.py)。V8/V10及各自检查也完整保留。
- `results/`含每设置的预登记、发展段逐样本预测、原结果JSON、冻结模型哈希及已选A参数的轻量投影。
- `previous_frozen_references/V6/`保留此前强基线预测；没有通过换掉旧强基线制造领先。
- `review/*CAUSAL_REPLAY.json`记录54条冻结主/消融/ADAPTIVE链的未成熟标签和未来X遮蔽重放。
- [公开范围](PUBLICATION_SCOPE.json)、[原代码哈希](CODE_SOURCE_MANIFEST.json)、[逐文件清单](PUBLICATION_MANIFEST.json)。

## 直接检查和重放

以下命令只读取已保存的发展段数据和已冻结参数，不重训、不重新选参、不读取原始保留段。依赖列于`requirements-replay.txt`；若需创建环境，优先使用项目已配置环境。

```bash
python sru_a_strengthening_20261023/tools/verify_snapshot.py
python sru_a_strengthening_20261023/tools/replay_a.py --branch conditional_feedback --case SRU_H2S_D4
python sru_a_strengthening_20261023/tools/replay_a.py --branch conditional_feedback --case SRU_H2S_D16
```

重放使用当初保存的冻结上游发行响应及严格延迟交付的标签，验证5个主/消融A输出、所有登记基线RMSE和9个前缀的未成熟残差扰动。它不代替原始过程输入到KC/W的完整重放；后者的服务器核查收据和完整冻结模型在归档中。

原`code/*/run.py`、数据视图和复核脚本保留实际运行时的服务器目录；完整重训需要按预登记还原数据目录、源元数据及相同运行环境，不能直接把归档路径当作通用命令。发布过程没有改写这些已哈希绑定的实验脚本。

## 完整模型与候选账本

[Release下载](https://github.com/Mountainatic/AR_RAPHU/releases/tag/sru-conditional-feedback-a-20261023)提供151,919,245字节的`SRU_A_STRENGTHENING_20261023.tar.gz`，含244个经逐项核验的成员：三个分支的冻结模型、完整内层候选账本、逐样本预测、公开SRU训练前缀缓存、运行日志及复核代码。大文件没有放入Git历史。使用Release的`SHA256SUMS.txt`和`RELEASE_RESTORE.md`核验恢复。

第三方SRU数据原来源及哈希保留于预登记/视图文件；本发布不额外授予或假设第三方数据许可证。未包含私有CZ原始数据或本轮保留尾段。父分支原有CZ历史材料单独保留，不属于本轮新增SRU数据。

## 发布前实际复核

直接运行的A重放入口已在原服务器Python 3.10环境通过全部9个设置：45条主/消融输出及每设置17项登记基线的RMSE重算一致；每个有A的变体在9个前缀下遮蔽未成熟残差后仍一致。没有重新拟合或读取原始最终段。精确依赖版本列于`requirements-replay.txt`，完整机器可读收据见[PORTABLE_REPLAY_VERIFIED.json](review/PORTABLE_REPLAY_VERIFIED.json)。
