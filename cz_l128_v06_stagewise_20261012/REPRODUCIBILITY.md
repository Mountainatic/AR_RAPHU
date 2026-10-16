# 复现范围与依赖

## 可在公开包独立完成

- `python verify_publication.py`：文件哈希、完整性、CSV/JSON/PNG/Python 格式与公开范围检查；不访问数据或训练。
- `v06_compatibility/tests/test_discrete_v06.py`：历史 v0.6 的数学、目标索引、区段和模型数值不变回归。该命名空间同时保留新增前后源码，用合成夹具运行，不是正式主方法重训。
- `runtime/code/tests/test_v2_urysohn.py`：保留的 Urysohn 数值单元测试。
- 从公开汇总表、PSD/ACF/PACF 表复核报告。频域/lag 表是聚合统计，不含原始时序。

依赖 numpy、scipy、pandas、pyarrow、pytest；画图另需 matplotlib。正式模型的环境指纹在 `runtime/release/ENVIRONMENT_FINGERPRINT.json`：Python 3.12.3、numpy 2.4.6、scipy 1.15.3、pandas 2.2.3、pyarrow 24.0.0、threadpoolctl 3.6.0，FP64，BLAS/OMP 单线程。本机轻量回归环境可能不同，不能据此声明跨环境正式训练位级等价。

POSIX 环境的轻量测试示例（在本目录执行）：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CZ_AUTHORITY="$PWD/runtime/code"
export PYTHONPATH="$PWD/v06_compatibility/runtime:$PWD/runtime/code/src"
python -m pytest -q v06_compatibility/tests/test_discrete_v06.py

export PYTHONPATH="$PWD/runtime/code:$PWD/runtime/code/src"
python -m pytest -q runtime/code/tests/test_v2_urysohn.py
```

## 正式实验仍需私有资产

公开代码并不等于公开完整私有运行环境。下列内容没有上传：C1 原始/派生样本、划分成员、42 个 K 训练产物、CPU/神经 checkpoint、部署权重、成熟残差回放与逐样本预测。

`StagewisePipeline.tune_k` 在正式模式下强制校验精确 K 复用。缺少匹配工件会拒绝执行，不会偷偷启动一次新的全量 K 搜索。`FINAL_PROTOCOL.json` 中 `stagewise_migration` 绑定旧目录、K 复用回执及哈希；原 K 复用回执以历史证据保存在 `audit/main/K_REUSE_COMPATIBILITY.json`。

源脚本仍保留当时绝对路径和防覆盖/封存检查；改路径、改源码、改协议或换环境都可能使旧 seal 失效。拥有私有资产者应恢复完整运行身份，或显式建立新的注册/release/输出目录并说明差异，不能编辑旧 seal 来伪装同一运行。

### 历史入口（说明用途，非公开包的一键运行命令）

| 入口 | 作用与先决条件 |
|---|---|
| `runtime/prepare_stagewise.py` | 建立新 stagewise 注册，要求旧源资产和兼容性证据 |
| `runtime/stagewise_entry.py --case-id ... --output-root ... --resume` | 已注册、环境/数据/K 复用核验通过后的源棒开发 |
| `runtime/accept_stagewise.py` | 使用真实源棒/K 工件的开发验收，不是纯单元测试 |
| `runtime/finalize_crossrod_fast.py` | 两方向及基线全局冻结后目标预测 |
| `runtime/run_synthetic_extensions.py` / `run_dependent_extensions.py` | 原 E2–E6 各自登记范围；后者需要私有源棒数据 |
| `runtime/run_frozen_diagnostics_fixed.py` | 最终实际采用的分阶段只读诊断适配器 |
| `whitening/rerun_new_e1.py` | 最新完整 E1 的独立同棒重预测；需要原模型、精确成员和 release；拒绝覆盖现有实验 |

`auto_start_*`、`supervise_*`、`launch_*` 是当时调度源码，不应作为状态查询或安装步骤执行。

## 结果身份与解释

主 release：`a1109dc5d5e3b6bc0f00ed018b4399ad87b2bf3691f6790c5d881e1b48b3f507`。这是 release seal 文件 SHA256，不是 Git commit。`SOURCE_PROVENANCE.json` 记录精确服务器导出文件和本地归档文件的哈希；`PUBLICATION_MANIFEST.json` 检查公开文件本身。

早期 v0.6 报告中的“E1 尚未完成”等是 2026-10-10 实施时状态；最新跨棒结果以 `audit/main/RUN_COMPLETION_RECEIPT.json` 为准。白化结果以 `audit/whitening/RUN_STATUS.json` 为准。公开包不把早期 55% 递归进度继续当作现行任务，也不把线性/噪声代理结果扩称为完整 E1。

本次仅公开已完成产物，未借上传之机重训、修改实验协议或追加新试验。
