# PRISM v2.1.1 Metro-P60 W 退化/触发审计方案

## 0. 实验定位

### 0.1 正式名称

```text
PRISM v2.1.1 Metro-P60 Wiener Degradation/Activation Transfer Audit
```

### 0.2 研究问题

旧 CPU 版本中，`METRO_P60__H6__W1` 的 v1.3 Physical-First 在没有 W 模块时已取得强输入路径结果。现在需要检验：

1. v2.1.1 的 W 是否能在该头上真正参加候选比较；
2. 当 W 不提供稳定增量时，PF 是否能严格退化为 `K+C` 或 `K+C+A`；
3. 当 W 仅能与 K/A 联合发挥作用时，Joint 是否能选择 `J_KW` 或 `J_KWA`；
4. SRU 修复后的 K/C/W/A/J 合同能否迁移到更大规模的 CPU 数据头；
5. W 是否会放大 Metro-P60 已知的时间域/OOD 迁移问题。

### 0.3 证据等级

Metro-P60 的旧 test/OOD 聚合结果已经历史可见，因此本轮是：

```text
RETROSPECTIVE_TRANSFER_AND_DEGRADATION_AUDIT
```

允许的结论：

- 实现合同是否成立；
- W 是否能正常关闭/触发；
- 新模型相对自身嵌套候选的 test/OOD 边际；
- v2.1.1 相对旧聚合结果的复现性描述。

禁止的结论：

- 把 Metro-P60 称为全新未见确认集；
- 根据历史 test/OOD 调整阈值、候选或算法；
- test 后把未选候选升级为正式模型。

---

## 1. 为什么选择 Metro-P60

历史 CPU 聚合参考显示：

- v1.3 Physical-First test MSE 约 `0.052632`，`R²≈0.6446`；
- v1.3 K-Joint-AR test MSE 约 `0.058783`；
- ARX test MSE 约 `0.080622`；
- Linear NARX test MSE 约 `0.110988`；
- test 行数约 `640,307`；
- OOD 行数约 `29,498`。

它适合本审计，因为：

1. 无 W 的旧 K 路线已知有效，便于检查新 W 是否无端激活；
2. 样本量足够大，不像小数据头那样容易被单折偶然性支配；
3. 有独立 OOD 段，可检查 W 是否放大操作域漂移；
4. 规模明显大于 SRU，但远小于 TEP 全量审计的资源压力；
5. 历史上 PF 的 ID 与 OOD 表现反差较大，正适合审计静态曲率的迁移稳定性。

历史结果只作背景，不参与新 development 选择。

---

## 2. 不可更改项

### 2.1 数据与视图

只启用：

```text
METRO_P60__H6__W1
```

必须继承现有 C1：

- 原始 base；
- 行顺序与 sample/base-origin ID；
- train/validation/test/OOD 边界；
- 可用性与 purge 合同；
- `latest_available_target_index`；
- 目标定义、horizon 与 target window；
- 缺失值与标准化合同。

不得：

- 重新切分；
- 删除或覆盖其他 dataset base；
- 根据旧 test/OOD 选择新窗口；
- 修改目标头；
- 重新生成共享 C1。

运行前后都要生成共享数据全文件 hash 审计。任何变化立即停止：

```text
STOP_DATA_BASE_MUTATED
```

### 2.2 算法

完整继承 PRISM v2.1.1：

- profile one-SE + 2% regret guard；
- 每通道最多保留两个 profile；
- fixed-support 最小稳定 ridge；
- C 输入路径保存门；
- W 尺度无关可用性门；
- W identity/monotone/cubic 候选；
- A 成熟条件与 fold-local centering；
- Joint 候选 `J_K/J_KW/J_KA/J_KWA`；
- 禁止 AR-only 与 K-zero Joint；
- PF/Joint 共用输入路径 gate；
- 选择、loss、prediction、contract 的 candidate-id 一致性。

本轮只允许：

- 数据头过滤；
- 路径适配；
- chunked prediction；
- 日志与审计增强；
- 修复会阻止既有 v2.1.1 合同执行的通用工程错误。

任何会改变模型候选、阈值或选择语义的代码修改都必须停止本轮，并另开版本，不得边看结果边修。

---

## 3. 冻结数值与资源合同

### 3.1 Row caps

继承 CPU 冻结合同：

```text
single_channel_k_fit          = 100000
validation_selection_per_fold = 50000
joint_physical_fit            = 250000
wiener_fit                    = 250000
state_fit                     = 250000
joint_predictive_fit          = 250000
final_prediction              = ALL_IMMUTABLE_ROWS
```

所有 cap 必须：

- 只在 train/inner-train 上应用；
- 使用确定性 sample/base-origin ID hash 排序；
- 嵌套候选使用相同的采样行；
- 不得对 test/OOD 采样计分，最终预测覆盖全部合法行。

### 3.2 资源配置

推荐：

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
workers=2
prediction_chunk_rows=50000
float64
```

禁止同时缓存全部：

```text
profile × family × rank × penalty × fold
```

每个 fold/candidate 结束后释放稠密设计矩阵；最终预测必须流式写出。

---

## 4. 分阶段执行

## M0：继承与污染审计

输出：

- 当前分支与 commit；
- git clean 状态；
- Python/NumPy/SciPy/sklearn/BLAS 版本；
- C1 数据路径、文件大小、SHA256；
- Metro-P60 split 行数与 sample-id hash；
- 历史结果只读路径；
- `test_accessed=false`、`ood_accessed=false`。

M0 失败不得运行训练。

## M1：回归测试

必须运行 v2.1.1 已有测试，并新增 Metro 规模测试：

1. identity W 与跳过 W 逐样本等价；
2. identity 与非线性 W 使用相同 fold mask 和 row cap；
3. chunked 与非 chunked 小样本预测一致；
4. capped train rows 完全由 inner-train 决定；
5. test/OOD 在 freeze 前无法读取；
6. PF/Joint 对同一输入路径状态一致；
7. C 不得在 active K 后输出近常数；
8. active W 的 materialized prediction 与选择 loss 属于同一 candidate-id；
9. Joint 不注册 AR-only；
10. 数据 hash 前后不变。

M1 全部通过后才能进入 M2。

## M2：Development K/C

只读取 train/validation。

对每个过程变量执行：

- profile 双门；
- K family/rank 选择；
- fixed-support refit；
- 通道激活审计。

C 只在：

- `ADDITIVE_COMPRESSED`；
- `ADDITIVE_JOINT_BASIS`

之间选择，并使用最小稳定 ridge。

必须输出：

- 每通道所有 profile 风险与 retained 原因；
- K exact-zero/active 状态；
- C 前后预测方差比；
- C 相对 best-active-K 的 MSE 比；
- input-path gate；
- fallback 是否触发。

若输入路径坍缩，停止：

```text
METRO_P60_INPUT_PATH_COLLAPSED
TEST_NOT_ACCESSED
OOD_NOT_ACCESSED
```

## M3：Development W 退化/触发审计

W 候选必须包括：

```text
IDENTITY_CORRECTION
MONOTONE_I_SPLINE_CORRECTION
NATURAL_CUBIC_CORRECTION
```

并扫描已冻结 knots、smoothness、soft-overlap mu。

每折输出：

- latent 有限值/不同值数量；
- `[1,q]` 数值秩；
- relative scale check；
- candidate applicability；
- identity 与非线性候选使用的 row IDs/hash；
- fold MSE 与相对改善；
- support exceedance；
- derivative/Lipschitz 数值审计；
- candidate-id。

选择后必须运行两个恒等测试：

\[
\max_t|\hat y_{KC+I_W,t}-\hat y_{KC,t}|\le\varepsilon,
\]

\[
\max_t|r^{(2)}_{I_W,t}-r^{(1)}_t|\le\varepsilon.
\]

正式 PF W 可能为 identity，也可能为非线性；两者都是合法结果。

## M4：Development A

A 只读取成熟的 `r2` 历史：

```text
s + h + target_window + D <= t
```

并使用：

- `latest_available_target_index`；
- fold-local residual centering；
- exact-zero 候选；
- practical gate 后 candidate-id 重绑定。

新增 active-near-zero 检查：若 A 的有效预测方差、非截距系数和增益均低于门槛，必须物化为 exact-zero，不能保留 active 标签。

## M5：Development Joint

运行：

```text
J_K
J_KW
J_KA
J_KWA
```

要求：

- 使用 M2 冻结 K 支持；
- W basis 系数与 K/A 一次联合求解；
- 不允许 `kw_scalar` 固定列替代联合 W block；
- 不允许 K-zero、AR-only、both-zero；
- PF/Joint 输入状态一致。

必须保存所有预注册 Joint 消融的 development 预测，但只有开发选择器选中的候选可成为正式模型。

## M6：Development 冻结

生成不可变冻结清单：

- 选中的 K 通道/profile/family/rank；
- C family 与 ridge；
- PF 的 W/A 状态；
- Joint 正式候选；
- 所有阈值；
- 代码/配置/data/sample-id hash；
- `test_accessed=false`；
- `ood_accessed=false`；
- 待物化候选 IDs。

继续门：

1. K/C 输入路径非坍缩；
2. PF/Joint gate 一致；
3. W 非线性候选在数值可用时实际参赛；
4. identity 等价测试通过；
5. 所有最终候选 loss/prediction/contract 一致；
6. 无 test/OOD 访问；
7. 数据未变。

不要求 W 必须激活，也不要求开发阶段击败旧历史 baseline。

## M7：冻结后 Test 与 OOD

冻结后一次性物化全部预注册嵌套候选：

PF：

```text
KC
KCW
KCA
KCWA
PF_SELECTED
```

Joint：

```text
J_K
J_KW
J_KA
J_KWA
J_SELECTED
```

即使 development 没选 W，也保存预注册 W 消融用于诊断；但不得事后升级为正式模型。

Test/OOD 都覆盖全部合法行，采用流式预测。

## M8：统计与报告

主要配对比较：

PF：

```text
KCW vs KC
KCWA vs KCA
PF_SELECTED vs KC
```

Joint：

```text
J_KW vs J_K
J_KWA vs J_KA
J_SELECTED vs J_K
```

迁移比较：

```text
同一 W 增量在 validation / test / OOD 的符号与幅度
```

使用移动块 bootstrap，500 次；至少报告：

- MSE 差；
- 相对 MSE 改善；
- 95% CI；
- 候选更优概率；
- Holm 修正；
- 最长块结论。

旧 baseline 若只有聚合指标，不得伪造配对统计。它们只放入历史对照表。确需配对比较时，冻结后从旧冻结代码重建对应逐样本预测，不允许重调。

---

## 5. 主要终点

### 5.1 实现终点

以下全部通过：

- W 三类候选实际参赛；
- identity 等价；
- PF/Joint gate 一致；
- C 未抹除 K；
- Joint W block 真联合；
- test/OOD 冻结后访问；
- 数据不变。

### 5.2 选择终点

根据 development 选择和冻结后的迁移表现，按配套接受矩阵定性。

### 5.3 不是终点的事项

- W 是否一定激活；
- v2.1.1 是否必须超过旧 v1.3；
- test 上某个未选消融是否更好；
- 单次 OOD 是否夺冠。

---

## 6. 预注册停止条件

立即停止且不得访问 test/OOD：

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

资源不足不允许改变候选语义。只能降低并行、使用 chunk、释放缓存；仍失败则记录：

```text
RESOURCE_LIMITED_RETAINED
```

---

## 7. 最终结果的合规表述

本轮报告必须同时给出：

1. development 正式选择；
2. test/OOD 正式模型；
3. 所有预注册嵌套消融；
4. W 在 PF 与 Joint 中的边际；
5. 选择是否与迁移表现一致；
6. 历史结果已知导致的证据等级限制。

不得用 test/OOD 重新命名正式模型。
