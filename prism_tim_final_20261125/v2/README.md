# PRISM v2.0 TIM E1–E3

本次发布以 TIM 提交 `f111106` 为复跑基准。E1 和 E2 使用 CZ Rod1 `(H,D,L)=(16,0,128)` 与 SRU H₂S `(0,8,64)`。E3 使用 CZ Rod1 和 Rod2 的 `H=8,16,32,48,64` 共同物理目标行，以及 Rod2 的偏置与振荡压力测试。E4 和 E5 暂缓。

| E1 时间尾段 | 目标行数 | 发布版 RMSE | v2.0 RMSE |
| --- | ---: | ---: | ---: |
| CZ Rod1 非重叠目标 | 2,820 | 0.037580 | 0.037736 |
| SRU H₂S | 2,017 | 0.030122 | 0.032212 |

SRU 的相同尾段曾参与历史评分。CZ 和 SRU 均采用同实体时间复核。与发布版比较的 Holm 校正后 p 值分别为 0.858 和 0.940。

## 结果与报告

| 实验 | 公开证据 |
| --- | --- |
| E1 主基准 | [六项指标](evidence/E1/METRICS.csv) · [配对 DM 和 Holm 校正](evidence/E1/DM.csv) |
| E2 消融 | [正则、支持域与 A0](evidence/E2/ABLATION.csv) |
| E3 步长与压力 | [共同目标行](evidence/E3/HORIZON.csv) · [Rod2 常量偏置与正负脉冲](evidence/E3/ROD2_PRESSURE.csv) |
| 数值核对 | [18 组 Python/C 回放](evidence/NATIVE_PARITY.csv) · [冻结回执](evidence/FROZEN.json) · [结果校验](evidence/RESULTS.json) |

[完整实验报告](evidence/PRISM_v2_E1_E3_Report.docx) · [下载发布包](https://github.com/Mountainatic/AR_RAPHU/releases/tag/prism-tim-v2.0-20261126)

Rod1 消融以 A0 提供主要增益。SRU 的完整模型接纳 C、W 和 A。Rod2 振荡压力下，矩包络比纯均值支持域产生更强的可信度衰减，两组关闭 C 后的部署误差相同。

## Rod1 内生先行诊断

在冻结 A0 上用前缀交叉拟合残差训练高通岭回归。H=16、48、64 的开发段 MSE 改善分别为 0.214%、0.746% 与 0.199%，均低于这次筛查的 1% 实质改善门槛。诊断支持当前 Rod1 协议保留纯内生 A0。 [诊断汇总](diagnostic/DIAGNOSTIC.json) · [诊断脚本](diagnostic/cz_endo_diagnostic.py)

## 模型与复跑

K 的样条外结点覆盖训练幅值范围。CZ 的联合升降坐标在首个 K 内层验证块之前冻结。K 候选使用三段因果前缀评分。均值包络采用 1%–99% 分位数，箱标准差阈值采用 99% 分位数，衰减尺度为该阈值与中位数之差，并以下限截断。

各水平或差分视图在 C 中占一个二维组。组范数用于加权可信度。W 使用带离散曲率惩罚的单调校准头。A 使用前缀交叉拟合产生的部署残差，A0 使用原始成熟目标增量。EMA、近端滞后和混合结构在内层竞争，接纳使用逐点配对增益的 Bartlett Newey–West 统计量及跨块正增益多数。

复跑依赖 TIM 发布件绑定的历史引擎、基线、权重和授权数据快照。使用 Linux 上的 Python、NumPy、SciPy、scikit-learn、LightGBM 及支持 AVX2/FMA 的 GCC 环境。

```bash
export PRISM_TIM_ARCHIVE=/path/to/PRISM_TIM_FINAL_20261125
export PRISM_TIM_PYTHON=/path/to/environment/bin/python
bash run.sh /path/to/new-output
"$PRISM_TIM_PYTHON" -B diagnostic/cz_endo_diagnostic.py \
  --v2 "$PWD" --frozen /path/to/new-output --out /path/to/diagnostic-output.json
```

模型写入 `frozen/`。全部模型冻结后，评估才读取尾段。原生内核通过生成的 C 头文件加载模型。真实数据回放的 Python/C 最大绝对预测差为 `5.68e-14`。
