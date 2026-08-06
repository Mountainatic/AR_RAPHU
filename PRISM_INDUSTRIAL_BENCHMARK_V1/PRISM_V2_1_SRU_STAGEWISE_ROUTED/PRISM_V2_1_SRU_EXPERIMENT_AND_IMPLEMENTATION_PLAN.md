# PRISM v2.1：SRU 单数据集实验与现有代码修正扩展方案

> **方案版本**：2026-08-06  
> **理论依据**：`PRISM_Theory_v2_1_Stagewise_Routed_Modular_Assembly_Theory_Only.md`  
> **代码母本**：现有 `PRISM_INDUSTRIAL_BENCHMARK_V1` 与 `PRISM_V2_MODULAR_NUMERICALLY_FROZEN` 代码树  
> **执行原则**：不继续旧 V2 全数据集重跑；只对 SRU 启用 v2.1；保留并校验全部既有 C1 数据 base；不删除、覆盖或重建其他数据集。  
> **实验性质**：v2.1 机制确认与实现修复，不宣称完成七数据集普适 benchmark。

---

# 0. 本轮目标

本轮只回答四个问题：

1. 在不使用全局 A-only 优先选择的情况下，SRU 的 K 输入支路能否被正确保留或正确判为不支持；
2. K→ΔW→A 的逐级 OOF 目标路由，能否比旧 v2.0 的硬投影与全局装配更稳定；
3. 将 v1.3 K-Joint AR 扩展为 K/W/AR 固定基联合优化后，是否提高 SRU 预测；
4. 修复旧 V2 实现错误后，结果是否仍支持停止旧 V2、转向 v2.1。

本轮不回答：

- v2.1 是否在全部工业数据集上普适；
- v2.1 是否改善 OOD；
- Joint 内部 K/W/AR 是否可作物理归因；
- SRU 结果是否可直接外推到 Debutanizer、TEP、PMSM 或 MetroPT。

---

# 1. 数据范围与不可变数据 base

## 1.1 只启用 SRU 数据集

“只跑一个 SRU”冻结为：

```text
active_datasets = ["sru"]
```

SRU 保留两个既有主目标头：

```text
SRU_H2S__H5__W1
SRU_SO2__H5__W1
```

不新增 horizon，不修改目标窗口，不修改历史可用性语义。

## 1.2 其他数据集 base 必须保留

既有 C1 数据包中的所有对象继续保留，包括但不限于：

- `base_data/`
- `sample_ids/`
- `dataset_views/VIEW_REGISTRY.json`
- `TASK_REGISTRY.json`
- `PROTOCOL.json`
- 数据继承 manifest；
- 每个数据集的 train/validation/test/ood 分区；
- 既有 sample ID、entity ID、origin 和 primitive support。

代码只在 view 枚举时过滤 SRU，不得：

- 删除非 SRU parquet；
- 重写非 SRU registry；
- 重新切分其他数据集；
- 为节省空间清理非 SRU base；
- 覆盖旧 V2 输出目录。

## 1.3 数据不变性审计

运行前生成：

```text
V21_DATA_BASE_PRE_AUDIT.json
```

至少记录：

- 所有 registry 文件 SHA256；
- 每个 `base_data/<dataset>/*.parquet` 的路径、字节数、SHA256；
- 每个 `sample_ids/<head>/.../*.parquet` 的路径、字节数、SHA256；
- 总文件数与总字节数；
- SRU 与非 SRU 分开汇总。

运行后生成：

```text
V21_DATA_BASE_POST_AUDIT.json
```

并强制：

```text
pre_hashes == post_hashes
```

任何不一致均：

```text
STOP_DATA_BASE_MUTATED
```

## 1.4 split 与 test 访问

沿用既有 C1 冻结 split、sample IDs、proxy policy 和 availability scenario。

开发阶段只能读取：

- train；
- validation；
- 开发 fold。

只有在 v2.1 final freeze manifest 生成后，才能读取 SRU test。

SRU 没有既有注册 OOD 时，不构造人工 OOD，也不把随机切分称为 OOD。

---

# 2. 新分支、目录与输出隔离

## 2.1 Git 分支

不得修改旧冻结分支。新建：

```text
prism-v2-1-sru-stagewise-routed
```

分支创建前记录：

```bash
git rev-parse HEAD
git status --short
```

未提交变更必须：

- 明确收入新提交；
- 或单独 stash/patch 保存；
- 不得静默丢弃。

## 2.2 方案目录

建议新增：

```text
PRISM_INDUSTRIAL_BENCHMARK_V1/
  PRISM_V2_1_SRU_STAGEWISE_ROUTED/
    PRISM_Theory_v2_1_Stagewise_Routed_Modular_Assembly_Theory_Only.md
    PRISM_V2_1_SRU_EXPERIMENT_AND_IMPLEMENTATION_PLAN.md
    PRISM_V2_1_CHANGELOG_FROM_V2_0_AND_CODE_AUDIT.md
    PRISM_V2_1_SRU_CONFIG_FROZEN.json
    PRISM_V2_1_SRU_NUMERICAL_FREEZE.md
```

## 2.3 输出目录

```text
results_prism_v2_1_sru/
```

不得复用：

```text
results_prism_v2/
ASSEMBLY_CARDS/
DEVELOPMENT/
```

的旧路径。

旧 V2 停止线包和输出只读保留。

---

# 3. 比较模型

## 3.1 v2.1 PRISM 模型

### 输入-only

1. `PRISM_V2_1_K`
2. `PRISM_V2_1_K_W`

### Dynamic Physical-First

3. `PRISM_V2_1_PF_K`
4. `PRISM_V2_1_PF_K_W`
5. `PRISM_V2_1_PF_K_A`
6. `PRISM_V2_1_PF_K_W_A`

最终 PF 不是在六个模型之间全局选择，而是由 K、W、A 各自局部选择自然确定。

### Dynamic Joint

7. `PRISM_V2_1_J_K`
8. `PRISM_V2_1_J_KW`
9. `PRISM_V2_1_J_KA`
10. `PRISM_V2_1_J_KWA`

Joint 候选中不包含：

```text
A_ONLY
EXACT_K_ZERO
EXACT_BOTH_ZERO
```

## 3.2 v1.3 PRISM 对照

保留原正式路线：

- `PRISM_CHANNEL_SPECIFIC`
- `PRISM_PHYSICS_FIRST`
- `PRISM_K_JOINT_AR`

这三条路线的合同和超参数不因 v2.1 结果重新选择。

## 3.3 既有 input-only 基线

与之前 C6 相同：

- `PERSISTENCE`
- `MEAN`
- `RIDGE`
- `PLS`
- `DPLS`
- `RBF_SVR`
- `XGBOOST`
- `PARALLEL_HAMMERSTEIN`
- `HAMMERSTEIN_WIENER`

N4SID 不属于 input-only。

## 3.4 既有 dynamic 基线

与之前 C6 相同：

- `PERSISTENCE`
- `MEAN`
- `LOCAL_LINEAR_TREND`
- `AR`
- `ARX`
- `LINEAR_NARX`
- `N4SID`（仅在既有开发合同可用时；失败则 `FAILED_RETAINED`）

A-only/AR 只作为预测基线，不参与 v2.1 PF 或 Joint 的结构选择。

## 3.5 基线复用原则

优先顺序：

1. 若服务器仍有与当前 C1 sample IDs 完全一致的逐样本 SRU 基线预测，校验 hash 后复用；
2. 若只有冻结模型合同，没有预测文件，按原合同重新 final-fit 和预测 SRU；
3. 若只有聚合指标，没有逐样本预测，不得用聚合指标代替配对统计，应重新生成 SRU 逐样本预测；
4. 不重新调参，不因 v2.1 表现修改基线。

---

# 4. v2.1 数值冻结

以下参数在读取 SRU test 前冻结。

## 4.1 fold 与选择

```text
inner_folds = 4
minimum_usable_folds = 3
one_se_ddof = 1
practical_improvement_min = 0.01
positive_fold_fraction_min = 0.75
```

所有 v2.1 模块使用局部 guarded one-SE。

外部基线保持其原有冻结选择规则，不改写为 v2.1 规则。

## 4.2 K/C

K 的以下网格继承现有 V2 冻结配置：

- channel-specific profiles；
- `m_tau={4,8,12}`；
- `m_x={4,6,8}`；
- rank `{0,1,2,3,FULL}`；
- ridge/smoothness 网格；
- active channel 上限；
- FP64 solver 证书。

首轮 SRU：

```text
C candidate:
- ADDITIVE_COMPRESSED
- ADDITIVE_JOINT_BASIS
```

暂不激活：

```text
SPARSE_PAIRWISE_ANOVA_MAX3
```

理由：本轮先检验选择与目标路由，不让 pairwise interaction 再引入硬残差化和额外搜索方差。

## 4.3 K 固定支持重拟合

选择完成后，为 OOF 残差生成使用：

```text
fixed_support_refit = true
```

候选：

- 选择 penalty；
- 更弱但通过数值证书的 penalty；
- 可选 debiased ridge。

最终取冻结规则允许的最弱数值稳定解，不在 test 上选择。

## 4.4 W

候选：

```text
IDENTITY_CORRECTION
MONOTONE_I_SPLINE_CORRECTION
NATURAL_CUBIC_CORRECTION
```

knots：

```text
MONOTONE: {4,6}
NATURAL_CUBIC: {4,6,8}
```

smoothness 沿用旧 V2 网格。

不执行 W 特征硬 residualization。

软惩罚：

```text
mu_W = {0, 0.03, 0.3, 3, 30}
```

\(\mu=0\) 必须存在。

W 的 selection loss 必须针对：

```text
r1 = y - pK_oof
```

并且最终记录：

```text
final_selected_candidate
final_selected_fold_losses
final_selected_prediction_path
final_selected_contract
```

## 4.5 A

首轮 PF 候选：

```text
EXACT_ZERO
MATURE_RESIDUAL_AR
```

不再注册：

```text
ORTHOGONAL_MATURE_RESIDUAL_AR
```

作为主候选。

A profile 与 ridge 网格沿用现有尺度匹配状态 profile。

软惩罚：

```text
mu_A = {0, 0.03, 0.3, 3, 30}
```

A 目标：

```text
r2 = y - pK_oof - deltaW_oof
```

成熟条件：

```text
s + h + W + D <= t
```

实现优先读取：

```text
latest_available_target_index
```

## 4.6 Joint-KWA

Joint 使用：

- PF 注册的 K profile 和 K basis family；
- fold-local K seed latent；
- 未硬投影 W spline basis；
- 合法目标历史 AR features；
- FP64 block ridge。

候选：

```text
J_K
J_KW
J_KA
J_KWA
```

不允许 AR-only。

块 penalty ratio 初始沿用旧 V2：

```text
K_over_A = {0.25, 1, 4}
```

增加 W block ratio 时，使用对称小网格：

```text
W_over_A = {0.25, 1, 4}
```

总组合数过大时，先固定 K/W/A 的主 alpha，再扫描两个 ratio，不做全笛卡尔积无限扩张。

Joint 不使用 soft overlap penalty。

Joint 必须通过输入支路非坍缩门：

```text
relative_gain_over_AR_only_diagnostic >= 0.01
positive_fold_fraction >= 0.75
input_prediction_variance > numerical_floor
```

失败输出：

```text
JOINT_INPUT_PATH_COLLAPSED
```

不回退 AR-only。

---

# 5. 实验阶段

## E0：继承与停止线审计

检查：

- C1 package identity；
- registry/sample IDs；
- 旧 V2 停止线只读；
- 当前 Git HEAD；
- 工作树；
- SRU 两个目标头存在；
- 非 SRU base 存在。

输出：

```text
E0_INHERITANCE_AUDIT.json
V21_DATA_BASE_PRE_AUDIT.json
```

## E1：原实现问题回归测试

先写测试，再改代码。至少覆盖：

1. A-only 不在 v2.1 PF assembly 候选；
2. Joint 不存在 K-zero/AR-only 候选；
3. practical gate 后的最终 candidate、loss、prediction path 一致；
4. \(D>0\) 时 mature residual 不读取未成熟标签；
5. `latest_available_target_index` 生效；
6. 每折 residual mean 只由 fit fold 估计；
7. W monotone 和 cubic 均不使用硬 projection；
8. \(\mu=0\) 精确退化为无 soft penalty；
9. K/W/A 选择作用域隔离；
10. 非 SRU base 前后 hash 不变。

E1 不通过不得运行训练。

## E2：SRU K/C

对两个目标头分别：

1. K channel profile；
2. K rank ladder；
3. numerical certificate；
4. local guarded one-SE；
5. fixed-support OOF refit；
6. C compressed vs joint basis；
7. 输出通道贡献和 \(p_K^{OOF}\)。

不访问 test。

## E3：SRU ΔW

对 \(r^{(1)}\)：

1. identity；
2. monotone；
3. cubic；
4. smoothness；
5. \(\mu_W\)；
6. local guarded one-SE；
7. OOF ΔW；
8. validation input-only prediction。

输出 W overlap、effective df 和 support。

## E4：SRU A

对 \(r^{(2)}\)：

1. exact-zero；
2. mature residual AR；
3. profile；
4. alpha；
5. \(\mu_A\)；
6. maturity coverage；
7. local guarded one-SE。

输出 PF validation prediction和装配卡。

## E5：SRU Joint-KWA

构造：

- \(\Phi_K\)；
- \(B_W(q^{seed})\)；
- \(H_Y\)。

运行 `J_K/J_KW/J_KA/J_KWA` route-local one-SE。

输出：

- total validation MSE；
- input path gain over AR-only diagnostic；
- block coefficient norms；
- block prediction variance；
- K/W/AR prediction correlations；
- `JOINT_INPUT_PATH_VALIDATED` 或 `COLLAPSED`。

## E6：最终冻结

生成：

```text
V21_SRU_FINAL_FREEZE_MANIFEST.json
```

至少包括：

- theory SHA256；
- plan SHA256；
- config SHA256；
- code commit；
- dirty status；
- data pre-audit SHA256；
- selected K/C/W/A；
- selected Joint candidate；
- all validation files and hashes；
- test_accessed=false。

## E7：SRU test

冻结后只运行两个目标头的 test。

禁止：

- test 后重新选择 K/W/A/J；
- test 后更改 \(\mu\)；
- test 后更改 baseline inclusion；
- test 后增加 horizon；
- test 后删除失败模型。

## E8：统计、报告与打包

生成逐样本预测、指标、配对统计、模型审计、资源统计、最终报告和 zip。

---

# 6. 原 V2 实现问题与本轮必须修复的代码点

## 6.1 全局 assembly 将 A-only 固定排在 K 前

旧 `v2_assembly.py` 使用总顺序：

```text
A_ONLY
K_COMPRESSED
K_JOINT_BASIS
K_JOINT_BASIS_W
K_JOINT_BASIS_A
K_JOINT_BASIS_W_A
...
```

并用一次全局 one-SE 选择。

问题：

- A-only 与 K-only属于不同信息集；
- 该顺序不是模块嵌套必然推出；
- 强输出自相关会使 A-only 通过捷径撤销 K。

v2.1 修复：

- 删除该全局选择器；
- PF 只由局部 K→W→A 结果生成；
- A-only 仅作为 baseline。

## 6.2 one-SE 选择结果与 practical gate 后物化模型不一致

旧 W/A 代码：

1. `one_se_select()` 得到非中性 candidate；
2. practical gate 失败后将 `selected` 改为 neutral；
3. prediction file 使用 neutral；
4. assembly loss 仍可能读取 `one_se["selected"]` 的非中性 fold loss。

问题：

```text
loss belongs to model A
prediction belongs to model B
```

v2.1 修复：

统一写入和读取：

```text
final_selected_candidate
final_selected_fold_losses
final_selected_prediction_path
final_selected_contract
```

并增加 hash/alignment 测试。

## 6.3 A 成熟条件漏掉 D

旧 A 特征构造使用：

```text
latest = origin - h_steps - w_steps
```

没有纳入：

```text
D_m
latest_available_target_index
```

v2.1 修复：

- 所有 mature query 从样本级 latest index 出发；
- 对 \(D>0\) 构造专门单元测试；
- 输出明确 maturity rule。

## 6.4 A 的 residual mean 在 inner fold 外全局计算

旧代码先对全部 OOF residual 计算 mean，再用于各 inner fold。

问题：

- evaluation fold 参与 fit fold 特征中心；
- 形成轻微 fold leakage。

v2.1 修复：

- 每折 fit residual mean 独立计算；
- evaluation 只使用 fit mean；
- final refit 的 mean 只用开发 fit 部分。

## 6.5 W 的 MONOTONE 与 NATURAL_CUBIC 实现合同不一致

旧代码中：

- natural cubic 被硬 residualize 到 \([1,q]\) 外；
- monotone I-spline 未执行同样 residualization。

问题：

- 同一理论 W 候选遵循不同归因合同；
- 结构选择不在同一候选语义下。

v2.1 修复：

- 两类 W 都使用未硬投影原始合法基；
- 归因统一由目标路由和 soft overlap audit 完成。

## 6.6 Joint 允许 K-zero 和 AR-only

旧 `v2_j.py` 注册：

```text
EXACT_BOTH_ZERO
EXACT_K_ZERO
EXACT_STATE_ZERO
JOINT_...
```

并可最终选择 `EXACT_K_ZERO`。

这与本轮要求“AR 不能独当一面”不一致。

v2.1 修复：

- Joint candidate 仅 `J_K/J_KW/J_KA/J_KWA`；
- AR-only仅为 diagnostic baseline；
- 输入支路坍缩则报告失败状态，不回退 AR-only。

## 6.7 旧 Joint 的 W 不是完整联合 W block

旧 Joint 主要把一个先拟合的 `kw_scalar` 附加到 K 特征后再联合 ridge。

问题：

- W 基本身没有作为完整可调 block 参与联合；
- K/W/AR 的相互侵占能力有限；
- 不能完整对应“联合 K/W/AR”。

v2.1 修复：

- 把 fold-local W spline basis columns 直接加入 Joint design；
- K/W/AR blocks 同时拟合；
- 不对 blocks 做硬投影；
- 只解释总预测。

## 6.8 逐级目标路由已有部分实现，不应重复发明

旧 W 已使用：

```text
target - latent
```

旧 A 已使用：

```text
target - physical_w
```

因此 v2.1 不是从零改写目标，而是：

- 保留已有目标递减；
- 修复选择作用域；
- 修复成熟性；
- 修复 fold 预处理；
- 取消硬 projection；
- 扩展 Joint；
- 保证输出合同一致。

---

# 7. 建议代码文件

为了不破坏旧 V2，新增而不是覆盖：

```text
src/prism_benchmark/
  v21_views.py
  v21_selection.py
  v21_k.py
  v21_c.py
  v21_w.py
  v21_a.py
  v21_joint.py
  v21_assembly.py
  v21_final.py
  v21_runner.py
  v21_audit.py
```

可以复用：

```text
cpu_data.py
v2_basis.py
v2_numerics.py
v2_config.py（仅复用 loader 形式，不复用旧 protocol_id）
```

建议新增测试：

```text
tests/test_v21_selection.py
tests/test_v21_maturity.py
tests/test_v21_w.py
tests/test_v21_a.py
tests/test_v21_joint.py
tests/test_v21_assembly.py
tests/test_v21_data_immutability.py
tests/test_v21_prediction_contract.py
```

---

# 8. 主要比较与统计

## 8.1 每个目标头的主要比较

### Input-only

1. `V2.1 K+W` vs `V1.3 PRISM_CHANNEL_SPECIFIC`
2. `V2.1 K+W` vs `HAMMERSTEIN_WIENER`
3. `V2.1 K+W` vs `PARALLEL_HAMMERSTEIN`
4. `V2.1 K+W` vs best frozen input-only baseline

### Dynamic

5. `V2.1 PF` vs `V1.3 PRISM_PHYSICS_FIRST`
6. `V2.1 Joint-KWA` vs `V1.3 PRISM_K_JOINT_AR`
7. `V2.1 PF` vs `ARX`
8. `V2.1 PF` vs `LINEAR_NARX`
9. `V2.1 Joint-KWA` vs best frozen dynamic baseline

## 8.2 机制比较

10. K vs K+W
11. K vs K+A
12. K+W vs K+W+A
13. Joint-KA vs Joint-KWA
14. soft \(\mu=0\) vs selected \(\mu\)
15. old hard projection replay vs v2.1 no-hard projection，仅作为开发消融，不参与最终 selector

## 8.3 指标

- MSE
- RMSE
- MAE
- \(R^2\)
- NRMSE
- relative Persistence skill
- dynamic relative AR skill
- parameter count
- effective degrees of freedom
- fit time
- prediction time
- peak RSS
- module activation
- soft overlap
- numerical condition/rank/KKT

## 8.4 配对统计

- paired moving-block bootstrap；
- 500 replicates；
- 冻结 seed；
- block lengths 沿用以 \(h+W\) 和历史覆盖构造的规则；
- 每个 head、split、information set、comparison family 内 Holm correction；
- 同时报告 CI、positive probability、raw p、Holm p。

只跑 SRU 时不报告七任务 cross-task rank。可以报告两个 SRU 目标头的简单平均，但必须标记：

```text
SRU_WITHIN_DATASET_SUMMARY
```

不能称为 benchmark-wide mean rank。

---

# 9. 成功、部分支持与停止条件

## 9.1 实现通过

必须满足：

- 所有 v2.1 单元测试通过；
- 数据 base hash 不变；
- no test leakage；
- final candidate/loss/path/contract 一致；
- maturity D 测试通过；
- Joint 无 AR-only candidate；
- A-only 不进入 PF assembly。

## 9.2 机制支持

至少满足其中之一：

- PF 相对 v1.3 PF 在一个或两个 SRU heads 上取得稳定改善；
- Joint-KWA 相对 v1.3 K-Joint AR 取得稳定改善；
- v2.1 正确保留非零输入支路，而旧 V2 错误退回 A-only；
- v2.1 正确关闭 W/A，且没有虚假下游激活；
- 修复后结果解释了旧 V2 SRU 退化的主要来源。

## 9.3 部分支持

例如：

- Joint 改善而 PF 不改善；
- H2S 改善而 SO2 不改善；
- W 激活但 A exact-zero；
- 预测改善但 K 结构证书不足。

应输出：

```text
PARTIAL_SUPPORT
```

而不是强行 PASS。

## 9.4 停止条件

- 数据 base 被修改；
- sample ID 对不齐；
- maturity 不可实现；
- 选择损失和预测无法一致；
- Joint 输入支路坍缩；
- 数值 hard fail；
- test 被提前访问；
- 基线合同无法重现且只有聚合指标。

失败必须保留，不伪造预测。

---

# 10. 输出目录与文件

```text
results_prism_v2_1_sru/
  FREEZE/
  DATA_AUDIT/
  DEVELOPMENT/
    K/
    C/
    W/
    A/
    JOINT/
  ASSEMBLY_CARDS/
  BASELINES/
  FINAL/
    test_predictions/
    metrics/
    bootstrap/
    audits/
  REPORTS/
```

必须生成：

- `V21_DATA_BASE_PRE_AUDIT.json`
- `V21_DATA_BASE_POST_AUDIT.json`
- `V21_SRU_CONFIG_FROZEN.json`
- `V21_SRU_FINAL_FREEZE_MANIFEST.json`
- `V21_SRU_FINAL_METRICS.csv`
- `V21_SRU_ENTITY_METRICS.csv`
- `V21_SRU_BOOTSTRAP.csv`
- `V21_SRU_MODEL_AUDIT.jsonl`
- `V21_SRU_ASSEMBLY_CARDS.jsonl`
- `V21_SRU_FINAL_REPORT.md`
- 每个模型逐样本 parquet；
- `MANIFEST.json`
- `SHA256SUMS.txt`

---

# 11. 打包与返还要求

最终必须清理旧临时输出，但不得删除：

- C1 数据 base；
- 旧 V2 停止线包；
- 旧正式 C6 release；
- Git 源代码。

建议打包命令语义：

```bash
set -euo pipefail

PACKAGE="PRISM_V2_1_SRU_STAGEWISE_ROUTED_RESULTS_bundle"
STAGE="${PACKAGE}_stage"

rm -rf "$STAGE" "${PACKAGE}.zip" "${PACKAGE}.zip.sha256"
mkdir -p "$STAGE"

cp -a \
  PRISM_INDUSTRIAL_BENCHMARK_V1/PRISM_V2_1_SRU_STAGEWISE_ROUTED \
  "$STAGE/theory_and_plan"

cp -a \
  results_prism_v2_1_sru/FREEZE \
  results_prism_v2_1_sru/DATA_AUDIT \
  results_prism_v2_1_sru/ASSEMBLY_CARDS \
  results_prism_v2_1_sru/FINAL \
  results_prism_v2_1_sru/REPORTS \
  "$STAGE/results"

git rev-parse HEAD > "$STAGE/GIT_HEAD.txt"
git status --short > "$STAGE/GIT_STATUS.txt"

python - <<'PY'
from pathlib import Path
import hashlib, json

root = Path("PRISM_V2_1_SRU_STAGEWISE_ROUTED_RESULTS_bundle_stage")
records = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    records.append({
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    })

(root / "MANIFEST.json").write_text(
    json.dumps({"files": records}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

(root / "SHA256SUMS.txt").write_text(
    "".join(f'{r["sha256"]}  {r["path"]}\n' for r in records),
    encoding="utf-8",
)
PY

zip -r -9 "${PACKAGE}.zip" "$STAGE"
sha256sum "${PACKAGE}.zip" > "${PACKAGE}.zip.sha256"

python - <<'PY'
from zipfile import ZipFile
p = "PRISM_V2_1_SRU_STAGEWISE_ROUTED_RESULTS_bundle.zip"
with ZipFile(p) as z:
    bad = z.testzip()
    if bad is not None:
        raise SystemExit(f"ZIP_INTEGRITY_FAILED: {bad}")
    names = set(z.namelist())
    required = [
        "MANIFEST.json",
        "SHA256SUMS.txt",
        "V21_SRU_FINAL_REPORT.md",
        "V21_SRU_FINAL_METRICS.csv",
        "V21_SRU_FINAL_FREEZE_MANIFEST.json",
    ]
    for token in required:
        if not any(name.endswith(token) for name in names):
            raise SystemExit(f"MISSING_REQUIRED_ARTIFACT: {token}")
print("ZIP_INTEGRITY_PASS")
PY
```

返还：

```text
PRISM_V2_1_SRU_STAGEWISE_ROUTED_RESULTS_bundle.zip
PRISM_V2_1_SRU_STAGEWISE_ROUTED_RESULTS_bundle.zip.sha256
```

---

# 12. 最终执行顺序

\[
\boxed{
E0\ \text{数据继承}
\rightarrow
E1\ \text{回归测试}
\rightarrow
E2\ K/C
\rightarrow
E3\ \Delta W
\rightarrow
E4\ A
\rightarrow
E5\ Joint\text{-}KWA
\rightarrow
E6\ \text{冻结}
\rightarrow
E7\ \text{SRU test}
\rightarrow
E8\ \text{统计与打包}
}
\]

本轮最重要的不是“让 v2.1 必须赢”，而是确保：

\[
\boxed{
\text{同一份 SRU 数据、同一 split、同一基线下，}
\text{理论选择、代码执行、物化预测和最终声明完全一致。}
}
