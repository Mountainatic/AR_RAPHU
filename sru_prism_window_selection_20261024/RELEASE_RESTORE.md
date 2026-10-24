# 恢复SRU固定128步基线与自选窗口PRISM实验

从本Release下载SRU_PRISM_WINDOW_SELECTION_20261024.tar.gz、ARCHIVE_MEMBER_MANIFEST.json、PACKAGE_RECEIPT.json、LOCAL_VERIFIED.json、SHA256SUMS.txt和本说明。

归档41,850,934字节，SHA256：caec24da4431f15568d400a93287c29ff53d752846138d4bc68164dcf1f02a08。

```bash
sha256sum -c SHA256SUMS.txt
python /path/to/repository/sru_prism_window_selection_20261024/tools/verify_bundle.py SRU_PRISM_WINDOW_SELECTION_20261024.tar.gz
mkdir restored_sru
tar -xzf SRU_PRISM_WINDOW_SELECTION_20261024.tar.gz -C restored_sru
cd restored_sru/sru_prism_window_selection_20261024
```

147个原始成员另加包内SHA256SUMS.json。全部30套PRISM候选模型及内层账本在runs/；训练/开发原字节前缀在data/；原128步基线包在baseline_reference/SRU_BASELINES_L128_20261023.tar.gz，其SHA256为ca4d91e9fbcee61311ab6719ef82675ecf18efc367876aaca6f22198252f2fda。嵌套基线包内有全部模型代码、权重、36个神经检查点、候选分数和种子结果。

重放PRISM：使用REGISTRATION.json登记的Python3.10、NumPy/SciPy/pandas/scikit-learn/pyarrow版本，分别执行`python prism_search.py evaluate 4`、`... evaluate 8`、`... evaluate 16`。这只使用已有冻结模型，不重新选窗或重新拟合；在线A仅在标签成熟后递归更新。未打开保留尾段。pickle需要相容环境。

重新训练：在单独空目录保留prism_search.py、launch.py、report.py、prism/、data/和baseline_reference/，先`python prism_search.py prepare`、`python prism_search.py smoke`，再`python launch.py`。不能在已存在REGISTRATION.json/FROZEN.json的原归档目录上直接启动重训。report.py使用REPORT_ENVIRONMENT.json的聚合环境，只复核和绘图。

原基线重放：在另一目录解压嵌套的SRU_BASELINES_L128_20261023.tar.gz，依其REPORT_CN.md、REGISTRATION.json及ARCHIVE_SCOPE_NOTES.md恢复原环境与入口。包内.venv只含两个原冻结清单绑定的uv脚本快照，不是可运行环境；不可把它当作完整解释器/依赖安装。

Git目录是浏览/轻量验证快照，自动生成的Source code资产不是完整实验包。所有源码、模型、预测、原始实验报告和登记均保留原字节。本Release属于开发归档，既有开发段曾反复用于研究，不是独立最终测试。
