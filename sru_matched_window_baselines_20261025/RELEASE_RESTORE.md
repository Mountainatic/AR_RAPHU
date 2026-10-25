# SRU相同直接窗口基线：两个完整实验包

当前主表对应SRU_BASELINES_MATCHED_WINDOWS_REPEAT_20261025.tar.gz（RTX4090新实例重训）；首次同窗口实验为SRU_BASELINES_MATCHED_WINDOWS_20261025.tar.gz（RTX5090）。两次均为延迟4/8/16对应历史8/16/16，6类基线独立重训，完整PRISM保持原冻结模型和在线A。

下载全部资产后先`sha256sum -c SHA256SUMS.txt`。两个tar包分别解压到新目录，不要混合覆盖。各包内部SHA256SUMS.json是全部成员清单；外部FULL_EXPERIMENT_MANIFEST.json对应当前repeat包，FIRST_RUN_PACKAGE_RECEIPT.json对应首次包。

已有冻结权重的完整重放：在恢复后的实验根目录，使用登记的Python3.10环境依次执行`python prism_replay.py 4`、`python prism_replay.py 8`、`python prism_replay.py 16`，再用登记的Torch/Python3.12环境执行`python experiment.py evaluate`和`python report.py`。这不重新训练或重新选参，不读取保留尾段。精确依赖和严格FP32设置在每包REGISTRATION.json中。

重新训练在单独空目录进行，不能覆盖已有登记/冻结模型。先创建新登记，执行smoke、CPU FP64数值核对，再运行launch.py；将launch.py的LEGACY解释器映射到自己的Python3.10环境。全部拟合冻结后才运行新模型开发评估。更多说明见各包REPRODUCIBILITY.md。

首次包的audit/保留在新拟合前实施严格FP32修正的完整历史；本次复跑直接沿用该后端，没有重新放宽容限。原L128检查点的数值迁移证明可通过上一轮2026-10-24 Release内的嵌套L128包重放。

Git是浏览和轻量验证快照；Source code自动压缩包不含所有模型文件。当前仍属多轮使用开发段的证据，不是独立最终测试；相同种子的复跑不是新的独立统计样本。两包均不包含独立保留尾段或私有CZ原始数据。
