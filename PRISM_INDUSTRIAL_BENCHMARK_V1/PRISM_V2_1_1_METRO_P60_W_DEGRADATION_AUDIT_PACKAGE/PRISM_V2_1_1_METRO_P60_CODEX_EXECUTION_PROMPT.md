# 给 Codex 的完整执行提示词

在现有 PRISM 仓库中执行 **PRISM v2.1.1 Metro-P60 W 退化/触发审计**。先完整阅读本包全部文件和 reference 理论文档。

## 任务边界

只运行 `METRO_P60__H6__W1`；输出到 `results_prism_v2_1_1_metro_p60_w_audit/`；建议分支 `prism-v2-1-1-metro-p60-w-audit`。

禁止修改/重建共享 C1、重切 split、运行其他数据集、覆盖旧结果、根据历史 test/OOD 改算法、改变候选/阈值、freeze 前读取 v2.1.1 test/OOD、test 后改正式模型。

本轮是 retrospective transfer audit，不是新未见 test 确认实验。

## 先审计

1. 记录 HEAD、branch、git status、完整 diff。
2. 查明 SRU v2.1.1 correction 的实际 commit/实现文件，不猜路径。
3. 确认 Metro-P60 C1、split、IDs、availability/purge。
4. 运行前后对共享数据做全文件 SHA256。
5. 给 test/OOD 加冻结锁。
6. 只在 view/runner 层过滤 active head。

若必须改变算法语义才能运行，立即停止并报告。

## 必须复用的合同

- K profile one-SE + 2% regret guard；
- fixed-support smallest stable ridge；
- C 不得抹除 K；
- W identity/monotone/cubic 全部实际参赛；
- 单折不适用不关闭全部 W；
- identity W 与跳过 W 逐样本等价；
- A 成熟条件含 D 和 latest index；
- Joint 只有 `J_K/J_KW/J_KA/J_KWA`；
- W block 与 K/A 同时求解；
- PF/Joint 共用 gate；
- loss/prediction/contract/candidate-id 一致。

## 资源

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

默认 workers=2、prediction_chunk_rows=50000、float64。继承 row caps；嵌套候选共享 train row IDs；test/OOD 全行流式预测。

## 阶段

```text
M0 inheritance/data audit
M1 regression tests
M2 development K/C
M3 development W
M4 development A
M5 development Joint
M6 development freeze
M7 test and OOD materialization
M8 bootstrap/report/package
```

M6 前必须保持 `test_accessed=false`、`ood_accessed=false`。

## 必须物化

PF：`KC, KCW, KCA, KCWA, PF_SELECTED`

Joint：`J_K, J_KW, J_KA, J_KWA, J_SELECTED`

所有模型保存逐样本预测；未被选中的只标记为 pre-registered ablation。

## 统计

重点比较 `KCW vs KC`、`KCWA vs KCA`、`J_KW vs J_K`、`J_KWA vs J_KA`；500 次 moving-block bootstrap + Holm。旧 baseline 无逐样本预测时只展示历史聚合参考，不能伪造配对统计。

## 停止条件

```text
STOP_DATA_BASE_MUTATED
STOP_TEST_OR_OOD_EARLY_ACCESS
STOP_V211_REGRESSION_TEST_FAILED
STOP_KC_INPUT_PATH_COLLAPSED
STOP_PF_JOINT_INPUT_GATE_INCONSISTENT
STOP_W_CANDIDATES_NOT_ACTUALLY_COMPARED
STOP_IDENTITY_W_NOT_EQUIVALENT
STOP_CANDIDATE_ID_MISMATCH
STOP_JOINT_W_NOT_JOINTLY_FIT
```

## 最终打包

先清理旧 bundle/staging，再收集代码、配置、tests、development/freeze/final 逐样本预测、metrics/bootstrap/report、logs、hash、git diff/status、runtime/peak memory。使用 `scripts/package_results.sh` 或完全等价命令，必须生成 manifest/hash、`unzip -t` 校验，并返回：

```text
PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_RESULTS_bundle.zip
PRISM_V2_1_1_METRO_P60_W_DEGRADATION_AUDIT_RESULTS_bundle.zip.sha256
```

不要只返回摘要；必须返还完整可复核产物。
