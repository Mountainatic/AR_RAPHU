# 恢复与重放

此包是相同直接窗口的SRU基线实验：标签延迟4/8/16分别使用8/16/16步历史。完整权重、候选分数和三种子逐点结果均包含。模型与源码保持冻结字节，没有完整Python环境。

核验包外SHA256并解压后，按包内SHA256SUMS.json逐文件核验。新模型的Torch/Python3.12环境版本见REGISTRATION.json；PRISM使用Python3.10科学计算环境。所有数据仅为[0,8064)的原公开SRU训练/开发前缀，不含原保留尾段或私有CZ数据。

已有权重重放：Python3.10依次运行`python prism_replay.py 4`、`python prism_replay.py 8`、`python prism_replay.py 16`，然后Torch环境运行`python experiment.py evaluate`。report.py使用同一Torch科学环境独立复核并生成图表；没有重新训练或选参。以上不需要旧服务器路径。

重新训练须在新目录保留源码、data、previous和prism_reference，创建新登记并先做smoke。launch.py中的LEGACY解释器路径需映射到自己的Python3.10环境。旧L128四个seed0冻结模型用于数值迁移证明，可从上一轮GitHub Release下载完整归档并解开其嵌套原基线包，再运行`python migration_check.py /path/to/sru_baselines_L128_20261023`；这不会参与超参数选择。启动时配置独立MPS目录，原服务器使用两GPU任务、每进程3个CPU线程。

旧基线完整包SHA256：ca4d91e9fbcee61311ab6719ef82675ecf18efc367876aaca6f22198252f2fda。上一轮Release：https://github.com/Mountainatic/AR_RAPHU/releases/tag/sru-baselines-prism-window-selection-20261024 。

本轮在新RTX4090实例重新拟合全部基线，沿用前一轮严格FP32路径、相同配置候选和随机种子；没有复用既有基线权重。MIGRATION_CHECK.json记录本轮拟合前CPU FP64参考核对，原容限保持不变。前一轮RTX5090的同窗口指标和逐点参考在previous/中，复跑对照在results/comparison_with_first_matched.csv。使用相同种子不构成新的独立随机种子样本。完整PRISM旧预测保持一致，主表仍有额外在线A记忆。当前是多轮使用开发段的比较，独立尾段未打开。
