# PRISM v0.6 KCWA 与 QC32-RLS 同协议比较

统一时间协议为 \(H=0,D=4,L=128\)。分别对 SRU H₂S、CZ Rod1 和 CZ Rod2 建模，在各自的完整开发段比较 10 月原版 KCWA、10 月 KCW 配 11 月 22 日 QC32-RLS，以及 14 个外部基线。发布包含五次运行的 RMSE、MAE、MSE 和水平 \(R^2\)，以及原版的消融、解释度和推理计时。

- [时间协议与实现结构](IMPLEMENTATION_CN.md)
- [四指标完整报告](results/BASELINES_CN.md) · [均值与标准差](results/MEAN_METRICS.csv) · [285 次逐次得分](results/RUN_METRICS.csv)
- [消融、解释度与时间复杂度](results/DIAGNOSTICS_CN.md) · [消融表](results/ABLATION_SCORES.csv)
- [机器可读协议](PROTOCOL.json) · [复跑入口](study.py) · [公开包检查](check_release.py)

## 复跑

科学计算环境使用 Python 3.10 与 [CPU 依赖](requirements-cpu.txt)。神经基线另需支持 CUDA 的 PyTorch 环境，历史运行使用 PyTorch 2.12.1 与 CUDA 13.0。C 计时需要 Linux 与 GCC，历史测量使用 GCC 11.4 和单个固定 CPU 核。使用单独的 CPU 与 GPU 环境时，可分别传给 `--cpu-python` 与 `--gpu-python`。

Windows 上使用 Git 时，可在短路径下用 `git -c core.longpaths=true clone` 检出冻结内核的历史文件。

用户提供按原始时间顺序切好的 NPZ 文件。每个文件含一个名为 `values` 的有限浮点二维数组，首列为目标，后五列为过程量。每套数据的列顺序、行数和原始导出文件的 SHA-256 均见协议。使用原始导出文件时传入 `--verify-reference-files`。另行导出的数组可走同协议复跑，历史比较的准确输入仍由协议中的哈希标识。

```text
inputs/
  SRU_H2S/train.npz
  SRU_H2S/development.npz
  CZ_Rod1/train.npz
  CZ_Rod1/development.npz
  CZ_Rod2/train.npz
  CZ_Rod2/development.npz
```

在本目录执行以下命令。工作目录放在 Git checkout 之外。`prepare` 写入路径及校验记录，训练脚本只读取训练文件。各阶段冻结后才读取对应开发段的数值做评分。神经基线复用仓库已有的 TimeXer 与 iTransformer 实现目录，完整 checkout 可直接定位。单独复制本实验目录时，可通过 `--baseline-source` 指向包含 `baseline_models.py` 和 `vendor/TimeXer` 的既有发布目录。

```bash
python check_release.py
python study.py prepare --inputs /path/to/inputs --workspace /path/to/work --verify-reference-files
python study.py run all --workspace /path/to/work \
  --cpu-python /path/to/cpu/bin/python --gpu-python /path/to/gpu/bin/python
```

分阶段复跑时按 `hybrid`、`baselines`、`legacy`、`diagnostics`、`repeats` 的顺序运行。`diagnostics` 包括原生计时。加上 `--reference-timing` 会额外运行较慢的原版 Python 逐点计时。`--jobs` 控制 PRISM 的任务并行数。

发布包包含模型与实验源码、配置、汇总指标和图。晶棒原始记录、拟合权重、逐行预测及残差、保留尾段数据仍保存在授权环境。公开检查通过合成序列验证窗口和成熟度，并重算五次指标汇总。发布版路径适配还在授权环境用既有冻结模型核对了 10,162 个开发点上两种 PRISM 的逐点预测，并运行了 SRU 的 AR 和 ARX 拟合。 [验证范围](PUBLICATION_SCOPE.json)
