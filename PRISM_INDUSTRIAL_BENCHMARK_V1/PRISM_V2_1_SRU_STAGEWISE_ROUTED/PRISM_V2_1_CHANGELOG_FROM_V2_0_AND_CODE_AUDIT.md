# PRISM v2.1：从 v2.0 的变更说明与原代码实现审计

> **日期**：2026-08-06  
> **适用范围**：`PRISM_INDUSTRIAL_BENCHMARK_V1` 中现有 Modular V2 实现向 v2.1 SRU Stagewise-Routed 路线迁移  
> **结论**：旧 V2 的时间因果、模块接口、OOF 目标递减和数值求解可以继承；全局 assembly、硬特征投影、成熟索引、候选—损失—预测一致性及 Joint 候选集合必须修正。  
> **注意**：本审计区分“理论设计不鲁棒”“实现与理论不一致”“明确代码正确性错误”三类问题。

---

# 1. 总体判断

旧 V2 不是完全错误，也不是需要全部重写。

可直接继承的主体：

- C1 数据接口；
- sample ID 和 split 合同；
- E/K/C/W/A/J 模块命名；
- K/C 先于 W/A；
- W 拟合 `target - K latent` 的一级残差目标；
- A 拟合 `target - physical_w` 的二级残差目标；
- rolling OOF 框架；
- FP64 Cholesky→QR→SVD 数值链；
- exact-zero/identity 候选；
- baseline 与 test freeze 框架。

必须停止或替换的主体：

- A-only 与 K/PF 的全局 one-SE；
- `A_ONLY` 固定排在所有 K 装配前；
- W/A 硬特征 residualization；
- Joint 中 K-zero/AR-only 退化候选；
- practical gate 后损失与预测文件不一致；
- A 成熟索引遗漏 \(D\)；
- A fold 中心泄漏；
- W 候选归因合同不一致。

---

# 2. 设计层问题

## D1：全局 assembly one-SE 混合了不同科学问题

### 旧实现

`v2_assembly.py::_view_card()` 汇总：

- A-only；
- K compressed；
- K joint basis；
- K+W；
- K+A；
- K+W+A。

然后做一次 one-SE。

### 问题性质

**理论设计不鲁棒。**

它把以下问题混在同一个风险选择器中：

1. 输入 K 是否存在；
2. W 是否解释 K 后静态误差；
3. A 是否解释成熟残差；
4. A-only 是否能利用输出惯性预测。

强自相关时间序列中，A-only 可以通过历史输出间接重构输入影响。A-only 的 MSE 低不能否定 K。

### v2.1 修复

- K、W、A 分别局部选择；
- A-only移出 PRISM PF assembly；
- A-only仅为 baseline；
- 下游不能撤销上游。

## D2：硬特征投影在闭环共线数据中高方差

### 旧理论/实现

- W natural cubic basis residualize 到 \([1,q]\) 之外；
- A mature residual history residualize 到 physical prediction 之外。

### 问题性质

**数学上合法，但统计上不鲁棒。**

可能导致：

- 特征能量接近零；
- effective rank 降低；
- 系数方差放大；
- neutral 过度选择；
- OOD 更敏感。

### v2.1 修复

- 使用 OOF 目标递减定义归属；
- W/A 使用未硬投影合法特征；
- 可选软输出重叠惩罚；
- \(\mu=0\) 必须保留。

## D3：A-only 与 K-only不构成单一物理复杂度偏序

### 旧实现

```python
order = [
    "A_ONLY",
    "K_COMPRESSED",
    ...
]
```

### 问题性质

**理论偏序被错误简化为人为总顺序。**

A-only 与 K-only读取不同信息集，通常不可比较。不能因为 A-only 参数少或被写在前面，就称其为 K 的自然中性退化。

### v2.1 修复

- PF 候选必须包含输入支路；
- state-only模型单独报告；
- Joint 也必须包含输入支路。

---

# 3. 明确代码正确性问题

## C1：practical gate 后 candidate 与 fold loss 不一致

### 旧路径

W/A 的典型流程：

```python
selection = one_se_select(...)
selected = selection.selected

if selected != neutral:
    activation = practical_activation(...)
    if not activation["pass"]:
        selected = neutral
```

预测文件根据修改后的 `selected` 生成。

但 assembly 读取：

```python
result["one_se"]["selected"]
```

对应的 fold losses。

### 结果

可能发生：

```text
fold loss: nonlinear W
prediction file: identity W
```

或：

```text
fold loss: active A
prediction file: exact-zero A
```

### 严重度

**P0：会直接改变最终 assembly 选择。**

### v2.1 修复

结果文件只允许通过：

```text
final_selected_candidate
final_selected_fold_losses
final_selected_prediction_path
final_selected_contract
```

读取最终模型。

必须测试：

```text
recomputed_loss(final_prediction) == stored_final_loss
```

在允许的浮点误差内成立。

## C2：A 成熟条件遗漏 \(D_m\)

### 旧实现

`mature_features()` 使用：

```python
latest = origins - h_steps - w_steps
```

### 理论要求

\[
s+h+W+D\le t.
\]

### 结果

在标签延迟或 analyzer maturity 场景中，A 可能读取尚未可用的残差。

### 严重度

**P0：潜在时间泄漏。**

### v2.1 修复

- 从样本级 `latest_available_target_index` 查询；
- profile offsets 从该 index 向后；
- 每个 entity 独立；
- 增加 \(D>0\) 测试；
- 输出 `maturity_rule` 和最晚残差 origin 审计。

## C3：A residual mean 在 fold 外估计

### 旧实现

```python
residual_mean = float(work["residual"].mean())
```

随后所有 inner folds 共用。

### 结果

evaluation fold 残差参与 fit fold 特征中心。

### 严重度

**P1：轻度但真实的 fold leakage。**

### v2.1 修复

对每个 fold：

```python
residual_mean_fold = fit["residual"].mean()
```

fit/evaluation 均使用该 fit mean。

最终 refit 只用开发 fit 部分估计 mean。

## C4：W 两类候选不遵循同一特征合同

### 旧实现

- `NATURAL_CUBIC`：硬 residualize；
- `MONOTONE`：直接使用 I-spline raw basis。

### 结果

同一个 W selector 比较的候选不具有相同归因语义。

### 严重度

**P1：实现—理论不一致。**

### v2.1 修复

两类候选都：

- 不硬 residualize；
- 使用同一 OOF 一级残差；
- 使用同一 soft overlap audit；
- 使用相同 final selection contract。

## C5：旧 Joint 允许 AR-only 成为最终 PRISM 输出

### 旧候选

```text
EXACT_BOTH_ZERO
EXACT_K_ZERO
EXACT_STATE_ZERO
JOINT_K_STATE_LINEAR
JOINT_KW_STATE_LINEAR
```

`EXACT_K_ZERO` 实际为 state/AR-only。

### 结果

Joint 可能失去输入支路，违背本轮要求。

### 严重度

**P0（相对 v2.1 新合同）。**

### v2.1 修复

候选只保留：

```text
J_K
J_KW
J_KA
J_KWA
```

AR-only 只做 diagnostic。输入支路失败输出 collapse status。

## C6：旧 Joint 的 W 只有预拟合标量，不能完全联合调节

### 旧实现

1. 先拟合 K scalar；
2. 用旧 W contract 生成 `kw_scalar`；
3. 把 `kw_scalar` 作为一列加入联合 ridge。

### 结果

- W spline basis 系数没有与 K/AR 同时优化；
- W 的自由度被压缩成单列；
- K/W/AR 的共享空间侵占不完整。

### 严重度

**P1：能力不足，不是泄漏。**

### v2.1 修复

Joint design 直接包括：

```text
K basis columns
W spline basis columns
AR history columns
```

使用 block ridge 联合拟合。

---

# 4. 已经正确实现、应当保留的部分

## K1：W 已经拟合一级残差

旧 `fit_w_candidate()` 使用：

```python
residual_target = target - train_latent
```

这与 v2.1 的：

\[
r^{(1)}=z-p_K^{OOF}
\]

方向一致。

## K2：A 已经拟合物理输出后的残差

旧 A 路径构造：

```text
residual = y_true - physical_w
```

这与 v2.1 二级目标方向一致。

需要修正的是：

- maturity；
- fold-local centering；
- hard projection；
- selection contract；
- global assembly。

## K3：K/C/W/A 已经分阶段运行

旧代码已经没有把 PF 路线完全端到端联合回调。v2.1 应继续禁止 PF callback。

## K4：数值求解链可复用

旧：

- certified Cholesky；
- QR；
- SVD rescue；
- FP64；
- KKT/condition/effective rank。

可直接复用，但每个 v2.1 contract 需记录 solver provenance。

---

# 5. 选择机制迁移

## 5.1 旧 one-SE

旧 `one_se_select()`：

1. 计算 best mean 和 best SE；
2. 形成 acceptable set；
3. neutral 在 acceptable set 时立即选择 neutral。

该逻辑适合强保守 shrinkage，但在全局装配中会过度 exact-off。

## 5.2 v2.1 guarded local one-SE

新逻辑：

1. 只在模块局部候选中形成 one-SE set；
2. 对 acceptable non-neutral 候选计算 practical gain；
3. 要求 paired positive fold fraction；
4. 有通过候选时选择其中最简单者；
5. 无通过候选时选择 neutral；
6. W/A 不能回调 K。

建议新增：

```python
guarded_local_one_se_select(
    fold_losses,
    complexity_key,
    neutral,
    minimum_relative_improvement,
    minimum_positive_fraction,
)
```

输出：

```text
best_candidate
acceptable_candidates
passing_active_candidates
final_selected_candidate
final_selected_fold_losses
activation_audit
```

---

# 6. 文件级迁移建议

| 旧文件 | v2.1 处理 |
|---|---|
| `v2_selection.py` | 保留旧文件；新增 `v21_selection.py` |
| `v2_k.py` | 复用候选构造；新增 fixed-support weak-ridge OOF refit |
| `v2_c.py` | 复用 compressed/joint basis；SRU 首轮关闭 pairwise |
| `v2_w.py` | 新建 `v21_w.py`，取消 hard projection，统一 final contract |
| `v2_a.py` | 新建 `v21_a.py`，修复 maturity、fold mean、取消 hard projection |
| `v2_j.py` | 新建 `v21_joint.py`，删除 K-zero 候选，加入 W basis block |
| `v2_assembly.py` | 不复用全局 selector；新建 `v21_assembly.py` |
| `v2_final.py` | 新建版本化 final runner，读取 v2.1 manifest |
| `v2_views.py` | 新建 `v21_views.py`，只过滤执行，不修改数据 base |
| `v2_state.py` | 仅作为外部 state baseline/AR feature helper |

---

# 7. 必须新增的单元测试

## 7.1 Selection

- neutral acceptable但 active practical/stable 时 active 可被保留；
- active 不达门槛时 neutral；
- W/A selection 不更改 K；
- final loss 与 final candidate 一致。

## 7.2 Maturity

- \(D=0\)；
- \(D>0\)；
- entity boundary；
- missing residual；
- latest available index；
- 不读取目标窗口；
- 不跨 run。

## 7.3 W

- identity correction 精确为零；
- monotone 和 cubic 均无 hard projection metadata；
- \(\mu=0\) 与普通 ridge 一致；
- soft penalty 增大时 overlap 不增；
- train-only knots。

## 7.4 A

- exact-zero；
- fold-local residual mean；
- no-U/X；
- mature coverage；
- \(\mu=0\)；
- final contract round trip。

## 7.5 Joint

- candidate set 不含 K-zero；
- `J_K` 为最简单候选；
- W basis block 实际存在；
- input path collapse 被识别；
- AR-only diagnostic 不被物化为 Joint prediction。

## 7.6 Data immutability

- only SRU views returned；
- 非 SRU registries/file hashes 不变；
- sample ID alignment；
- output 不写入 shared package。

---

# 8. 结果解释边界

## 可声明

- v2.1 在 SRU 上修复了旧实现错误；
- v2.1 PF 是否保留输入支路；
- W/A 是否在局部残差上激活；
- Joint-KWA 是否提供额外预测增量；
- 软惩罚是否被选择；
- 与 v1.3 和既有基线的 SRU 配对结果。

## 不可声明

- v2.1 已通过全工业 benchmark；
- 两个 SRU heads 足以证明普适性；
- Joint block 系数具有机理含义；
- SRU test 改善等于 OOD 改善；
- K 被选择即开放环因果；
- A exact-zero 即不存在内部状态。

---

# 9. 最终迁移判定

v2.1 不是：

```text
对旧 V2 原地打补丁后继续全量重跑
```

而是：

```text
保留旧 V2 数据与模块基础
+
建立独立版本化代码路径
+
只运行 SRU
+
用局部选择与逐级目标路由重做 PF
+
用 mandatory-input Joint-KWA 扩展 v1.3 Joint
```

最终结构：

\[
\boxed{
\text{旧 V2：停止线、只读归档}
}
\]

\[
\boxed{
\text{v2.1：新分支、新协议、新输出、SRU 机制确认}
}
\]
