# PRISM v0.8

2026-11-25 冻结版。包含 \(H=0,D=4,L=128\) 下的 SRU H₂S、CZ Rod1、CZ Rod2 时间尾段复核，以及固定单核端到端计时。时刻 \(t\) 估计 \(y_t\)，可用过程量至 \(t\)，可用标签至 \(t-4\)。

## 阅读与复用

- [协议与时间切分](PROTOCOL.json) · [发布范围与冻结摘要](PUBLICATION_SCOPE.json)
- [开发段报告](results/DEVELOPMENT_REPORT_CN.md) · [尾段与工程报告](results/FINAL_REPORT_CN.md) · [尾段六指标表](results/TAIL_TABLES_CN.md)
- [汇总 CSV](results/TAIL_SUMMARY.csv) · [逐种子 CSV](results/TAIL_SEEDS.csv) · [种子样本标准差](results/TAIL_SEED_SD.csv)
- [精度–延迟帕累托图](results/PARETO.svg) · [时延表](results/PARETO_TABLE.csv) · [长窗口曲线](results/LONG_WINDOW.svg) · [长窗口时延表](results/LONG_WINDOW_TABLE.csv)

| 完整时间尾段 | PRISM v0.8 RMSE | Elastic Net 均值 | 扩展 Transformer 五种子均值 |
| --- | ---: | ---: | ---: |
| SRU H₂S | 0.029745 | **0.029407** | 0.030209 |
| CZ Rod1 | **0.011279** | 0.016060 | 0.013678 |
| CZ Rod2 | **0.012036** | 0.012412 | 0.012523 |

SRU 的尾段目标行曾以 \(D=8\) 评分。两根晶棒另报告剔除已知重叠起点的子集，状态沿实际时间连续更新。脱丁烷塔已用于历史实验，本发布无零接触外部测试。[历史记录](results/EXTERNAL_PROVENANCE.json)

## 模型与代码

1. **K** 对水平和 \(D\) 步差分视图分别构造滞后箱与自然样条。外侧节点为训练样本的 5% 和 95% 分位点，近端逐拍候选为 4 和 8。标准收缩与保真收缩档位共用下游校准流程。
2. **C** 用 Elastic Net 融合视图，以训练支持范围衰减超界响应，并用活跃系数汇总可信度。
3. **W** 对上游坐标做条件后置非线性修正，通过时间块门控才接入。
4. **A** 用已成熟的上游残差做固定参数校正。EMA、近端 AR 和混合状态按家族独立跨块评估，通过门控后择优。

| 入口 | 用途 |
| --- | --- |
| [v08.py](workflow/PRISM_V08_20261125/v08.py) | 统一选型、训练、冻结和预测 |
| [preflight.py](workflow/PRISM_V08_20261125/preflight.py) | 绑定模型、基线和尾段来源 |
| [tail_replay.py](workflow/PRISM_V08_FINAL_EVIDENCE_20261125/tail_replay.py) | 冻结尾段复核、加法归因、因果与原生一致性检查 |
| [benchmark.py](workflow/PRISM_V08_FINAL_EVIDENCE_20261125/benchmark.py) | 固定单核计时和等编译选项 SIMD 窗口扫描 |
| [stream_fast.c](workflow/PRISM_V071_REVIEW_20261125/stream_fast.c) · [arx_mirror.c](workflow/PRISM_V08_20261125/arx_mirror.c) | PRISM 和镜像环 ARX 的 FP64 原生热路径 |

## 校验公开包

在本目录运行 Node.js 18 或更新版本。校验器验证发布文件摘要、尾段范围、种子统计、审计上界与计时表。

~~~sh
node verify_release.mjs
~~~

如需重画图表，在 Linux 或 macOS 上用带 Matplotlib 的 Python 运行下列命令，输出写入新建的目录。

~~~sh
python3 render_results.py ./rendered
~~~

归档源码保留原始执行布局。精确复跑需在授权环境恢复私有输入、已冻结的模型与基线权重、完整锁定与数据授权清单，以及本仓库 [v0.6 发布](https://github.com/Mountainatic/AR_RAPHU/releases/tag/prism-v06-h0-d4-l128-20261124) 留存的早期运行依赖。原清单 SHA-256 为 `4403fe0c314b0d5f4fe773c2f4a34ef5d17914db030f09efb657d86eafbc1a53`。完整复跑入口提供 `verify`、`ingest`、`evaluate` 与 `aggregate` 四个动作。公开包提供代码、协议、汇总审计和图表。
