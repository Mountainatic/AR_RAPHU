# CZ H1 轻量白化与 7 秒短周期诊断（2026-10-06）

此目录公开三轮已完成实验的代码、报告、汇总统计和图表。**不包含 CZ 原始数据、逐行目标/预测/残差、划分 membership、模型参数或原始交付 ZIP。** 发布日期为 2026-10-07。

| 实验 | 报告 | 主要结果 |
|---|---|---|
| 轻量 H1 残差白化 | [交付报告](analysis_outputs/cz_h1_light_whitening_20261006/DELIVERY_REPORT_CN.md) | 成熟过去残差的 Ridge correction 改善三个 arm；Rod2 最稳定。 |
| 约 0.14 Hz / 7 秒峰来源诊断 | [诊断报告](analysis_outputs/cz_seven_second_peak_20261006/SEVEN_SECOND_PEAK_DIAGNOSTIC_REPORT_CN.md) | `UNRESOLVED_RESIDUAL_STATE`；旋转单位未确认，phase tests 未执行。 |
| 短周期增强 A | [实验报告](analysis_outputs/cz_h1_short_cycle_a_20261006/SHORT_CYCLE_A_REPORT_CN.md) | `NO_ADDITIONAL_SHORT_CYCLE_GAIN`；小幅收益未达预注册门槛，保留 A0。 |

轻量实验 descriptive test 的 E160 / E900 降幅：Rod1_Common4 约 51.1% / 50.9%，Rod1_AllLegal 约 65.4% / 63.0%，Rod2_Common4 约 79.7% / 75.8%。短周期 A1 相对 A0 的 Rod2 holdout band power 降幅约 3.70%，RMSE 降幅约 0.317%；未达到本轮预注册的 10% band-power 门槛，未复制到 Rod1。A2 无额外收益。

这些是事后诊断及同棒时间尾段的描述性结果，不能作为跨晶棒泛化、残差已为白噪声或唯一物理机制的确认性证据。采样周期沿用既有代码的 2 s 约定；原工作簿没有独立时间戳校验，旋转通道工程单位仍未确认。

## 文件与复核

- `analysis_outputs/` 保留原三轮目录名。Python、报告、汇总 CSV、PNG、预注册、冻结配置和既有验证结果均按原字节发布。
- `PUBLICATION_MANIFEST.json` 是本次公开文件的 SHA-256 清单；原交付 manifest/ZIP 未上传，因为它们列出或包含私有产物。
- `PUBLICATION_AUDIT.json` 记录本次发布检查，包括每个汇总 CSV 的列名与行数。
- `verify_publication.py` 仅检查公开文件，不打开私有工作簿，也不训练模型。

在此目录可运行仓库约定的检查命令：

```powershell
conda run -n Env_pytorch --no-capture-output python verify_publication.py
```

## 完整数值复现的私有前提

此包是代码和汇总结果归档；单独克隆公开仓库不能完整重跑数值实验。未公开输入包括：

1. 经授权的 `实验数据1-张.xlsx`，应置于此目录根部；仅允许在私有环境中只读使用。
2. 旧 `cz_h1_a_history_20261001` 下的冻结 `fit_support.parquet` 和 `common_support.parquet`；轻量实验用于核对原 source-train/source-validation membership。
3. 轻量实验的冻结 membership、OOF/validation/test 逐行产物、PCA 状态及模型文件；诊断与短周期实验复用它们。完整列表和哈希保留在各轮冻结记录中，但文件本体不公开。
4. 原任务附件路径仅用于运行时间预算/交付时间记录，Python 归档保留这些本地路径；在其他私有环境复现前须另行配置计时来源。原 source helper 仅通过 AST 提取指定数据函数，未导入完整 PRISM runner；helper 的其他模块不是本包执行依赖。

历史环境版本记录见轻量实验 `RUN_SUMMARY.json`。本次上传没有重训、改选模型或重新打开 holdout。报告中提到的私有 parquet/NPZ/ZIP 是原本地交付内容，不代表它们已在 GitHub 发布。请勿运行归档的打包脚本后将新生成的完整交付 ZIP 上传。

隐私范围详见 [PUBLICATION_SCOPE.md](PUBLICATION_SCOPE.md)。
