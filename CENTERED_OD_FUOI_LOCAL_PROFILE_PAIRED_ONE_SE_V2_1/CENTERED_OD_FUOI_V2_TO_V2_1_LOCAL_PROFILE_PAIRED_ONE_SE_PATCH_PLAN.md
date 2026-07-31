# Centered OD-FUOI V2 → V2.1 修正方案
## 局部风险剖面解析 + 配对时间块 1-SE，不改模型结构

> **修订名称**：`CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1`  
> **源版本**：`CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2`  
> **源提交**：`99bf45fee25a97db0206413f02b2341e78377112`  
> **V2 结果包 SHA256**：`fc75a208c87855eed83de6ae886694b2d0b9813fa14a1cf60938e776a523037d`  
> **修订性质**：只修正 E2B/E2C 的复杂度选择与状态判定，不修改 centered-increment Full Urysohn、Sobolev 几何、basis、输入变量、预测目标或 residual 结构。  
>
> 本修订不是 V3，也不是新模型。它只修正 V2 的两个统计实现问题：
>
> 1. 用“整条 EDF 风险曲线达到 \(10^{-4}\) 插值误差”判断 resolved，导致远离最优点的高损失区域耗尽 80 次预算；
> 2. 用四个非同分布 expanding folds 的绝对 MSE 离散度计算 one-SE，导致 Sheet1→Sheet2 被过度平滑到 \(d=1.0674\)。

---

# 0. V2 已经确认的事实

V2 不需要推倒重来。

已经确认：

\[
\mathrm{RMSE}_{\mathrm{V1\ Full}}=173.856761
\]

降为：

\[
\mathrm{RMSE}_{\mathrm{V2\ Centered\ Full}}=0.382494.
\]

相对 old K-only：

\[
0.412383\rightarrow0.382494,
\]

40 min paired bootstrap 正改善概率：

\[
0.995.
\]

因此：

\[
\boxed{
\text{centered increment 与 blocked predictive selection 已经消除了数值爆炸。}
}
\]

当前剩余问题不是模型重新设计，而是：

\[
\boxed{
\text{V2 对“风险最低点是否解析”和“1-SE 应选多平滑”的判定过严且不合适。}
}
\]

---

# 1. 本轮严格不改什么

以下内容全部冻结：

1. 目标 \(z_t\)；
2. L6：40 min history、20 min horizon、2 min output window、10 s cadence；
3. 两个 outer directions；
4. 四折 expanding-window 和 22 min purge；
5. 四个输入和联合升速 PCA；
6. centered increment：
   \[
   \delta_{j,t}(\ell)=x_{j,t-\ell}-x_{j,t};
   \]
7. 20 个 lag basis；
8. centered-amplitude quantile basis；
9. 单一 Sobolev penalty；
10. fold-specific \(\lambda_f(d)\)；
11. Full Urysohn 面；
12. 线性/纯非线性正交投影；
13. derived Rank-1 只作审计；
14. \(C^1\) 延拓；
15. frozen K 后 matured residual PS/AR；
16. old K、NLinear、Joint-K+AR 等冻结基线；
17. outer test 只评估一次；
18. 不增加任何第二正则系数；
19. 不引入 shared/private；
20. 不引入通道训练权重；
21. 不使用人工 \(\lambda_{\min}\) 或 df=30 锁。

因此正式模型仍然是：

\[
\widehat z_t^K
=
b+
\sum_{j=1}^4
\sum_{\ell=0}^{239}
K_j\!\left(\ell\Delta t,\delta_{j,t}(\ell)\right)\Delta t.
\tag{1.1}
\]

---

# 2. V2 的两个具体问题

## 2.1 全局插值误差被错误地当成模型选择证书

V2 `ContinuousProfile.resolve()` 对每个区间使用三点二次插值，并要求：

```text
relative interpolation error <= 1e-4
```

否则继续细分。

实际最优点已经定位到：

### Sheet1→Sheet2

\[
d_{\min}=7.4232,
\qquad
[7.3839,7.4430].
\]

### Sheet2→Sheet1

\[
d_{\min}=26.3907,
\qquad
[26.3471,26.4176].
\]

但是搜索仍试图解析 EDF 高达数百、MSE 已达到几十甚至上万的远场区间，最终耗尽预算并登记：

```text
CONTINUOUS_EDF_PROFILE_UNRESOLVED
```

这混淆了两个不同命题：

1. 最低风险盆地是否已经解析；
2. 整条风险曲线是否被高精度插值。

正式选择只需要第一个命题以及 1-SE 左边界，不需要把远场高损失曲线拟合到 \(10^{-4}\)。

## 2.2 绝对 fold MSE 的 one-SE 过度平滑

V2 使用：

\[
T_{\mathrm{1SE}}
=
\overline L(d_{\min})+
\operatorname{SE}_{\mathrm{fold}}(d_{\min}).
\]

Sheet1→Sheet2 四折 MSE 约为：

\[
0.1484,\ 0.1451,\ 0.2142,\ 0.6629.
\]

最后一折明显更难，使：

\[
\operatorname{SE}_{\mathrm{fold}}=0.1244.
\]

于是：

\[
d_{\min}=7.4232
\]

被向左推到：

\[
d_{\mathrm{1SE}}=1.0674.
\]

expanding folds 对应不同时间阶段，不是同分布重复实验。绝对 MSE 的阶段差异不应全部解释成“复杂度选择的不确定性”。

---

# 3. 修订后的总体流程

V2.1 的 E2 选择变为：

\[
\boxed{
\text{自动全域盆地发现}
\rightarrow
\text{最优盆地局部精修}
\rightarrow
\text{配对时间块 1-SE 左边界}
}
\]

分为：

1. **E2B-1：全域盆地发现**  
   自动寻找可能包含最低风险的盆地，不要求远场高损失区域高精度插值。

2. **E2B-2：局部最低点精修**  
   在最低盆地内使用 bounded Brent，直到最低点和括区间满足局部容差。

3. **E2C：配对时间块 1-SE**  
   比较每个更平滑 \(d\) 与 \(d_{\min}\) 在同一验证样本上的平方误差差，消除 fold 难度公共项。

4. **E2D：outer full-training refit**  
   仅当新 \(d_{\mathrm{1SE}}\) 确定后重拟合最终模型。

---

# 4. E2B-1：自动全域盆地发现

## 4.1 搜索坐标

为避免 \([1,500]\) 的高 EDF 远场占据绝大多数线性区间，在：

\[
s=\log(d-d_0+\epsilon)
\tag{4.1}
\]

上进行自动递归搜索。

其中 \(d_0\) 来自共同 EDF 区间下界，\(\epsilon\) 只用于数值定义：

\[
\epsilon=10^{-6}\max(1,d_{\mathrm{upper}}-d_0).
\]

这不改变模型，只改变一维搜索坐标。

## 4.2 自动初始化

不使用手工九锚点。

算法自动评估：

- 共同区间左右端；
- \(s\) 坐标中点；
- 递归二分产生的点。

所有点均由区间几何自动生成。

## 4.3 区间分类

对每个区间 \(I=[d_L,d_R]\)，维护：

- 左、中、右三点真实 BCV；
- 当前全局最好值 \(L_\star\)；
- 是否存在内部下降趋势；
- 是否可能进入最低盆地；
- 是否可能进入 1-SE 选择区域。

区间分为：

```text
CANDIDATE_BASIN
SELECTION_RELEVANT
FAR_FIELD_PRUNED
NUMERICALLY_UNRESOLVED
```

## 4.4 远场剪枝

不再依据全局二次插值误差单独继续细分。

若区间满足以下条件：

1. 左、中、右三个真实风险都显著高于当前最低值；
2. 三点中不存在向区间内部持续下降的趋势；
3. 区间不与当前最低盆地相邻；
4. 区间内最低观测值已高于当前“选择相关上界”；

则登记：

```text
FAR_FIELD_PRUNED
```

不再要求其插值误差达到 \(10^{-4}\)。

选择相关上界在最低点尚未稳定前使用保守值：

\[
L_{\mathrm{rel}}
=
L_\star+
\max\left(
\operatorname{SE}_{\mathrm{paired,pre}},
0.1L_\star
\right).
\tag{4.2}
\]

其中 `paired,pre` 来自当前最好点与相邻已评估点的配对块误差。式 (4.2) 只用于搜索队列优先级和远场剪枝，不用于最终 one-SE 决策。

## 4.5 不宣称数学上的无条件全局最优

有限次函数评价不可能在完全无形状假设下证明连续函数的绝对全局最优。

V2.1 的正式状态改为：

```text
GLOBAL_BASIN_DISCOVERY_PASS
```

它表示：

- 全域经过确定性多尺度扫描；
- 所有观测到的局部最低盆地已建立括区间；
- 最低盆地经过独立精修；
- 其余已发现盆地的风险显著更高；
- 未解析远场不可能进入已观察的 1-SE 相关区域。

不再使用“整条曲线全局插值完成”作为条件。

---

# 5. E2B-2：局部最低点解析

## 5.1 最低盆地

从全域搜索得到全部候选盆地：

\[
\mathcal B_1,\ldots,\mathcal B_m.
\]

每个盆地分别使用 bounded Brent：

\[
d_k^\star
=
\arg\min_{d\in\mathcal B_k}\overline L(d).
\tag{5.1}
\]

取实际风险最低者：

\[
d_{\min}
=
\arg\min_k\overline L(d_k^\star).
\tag{5.2}
\]

## 5.2 局部解析条件

最低盆地满足以下条件即登记：

```text
LOCAL_MINIMUM_RESOLVED
```

条件：

1. 最终括区间宽度：
   \[
   d_R-d_L\le0.05;
   \]
2. 三次独立启动的 bounded Brent 结果差异：
   \[
   \max d^\star-\min d^\star\le0.05;
   \]
3. 最低点不撞共同 EDF 上界；
4. 括区间两端风险均不低于中心；
5. 相邻候选盆地最低风险大于当前最低风险；
6. 所有预测 finite；
7. fold-specific EDF inversion 误差通过。

原来的：

```text
max_quadratic_interpolation_error <= 1e-4 over whole domain
```

从正式门禁中删除，仅保留为诊断。

## 5.3 新状态拆分

输出：

```json
{
  "global_basin_discovery": "PASS|UNRESOLVED",
  "local_minimum_resolved": true,
  "far_field_interpolation_resolved": false,
  "far_field_required_for_selection": false
}
```

`far_field_interpolation_resolved=false` 不再自动导致 estimator unresolved。

---

# 6. E2C：配对时间块 1-SE

## 6.1 配对误差差

对任意更平滑候选：

\[
d\le d_{\min},
\]

在同一个 validation sample 上定义：

\[
\Delta_{f,t}(d)
=
e_{f,t}^2(d)
-
e_{f,t}^2(d_{\min}).
\tag{6.1}
\]

其中：

\[
e_{f,t}(d)
=
z_{f,t}-\widehat z_{f,t}(d).
\]

这样：

- fold 本身难或容易的公共部分被相减；
- 比较直接回答“更平滑模型相对最低点损失了多少”；
- 不再用四个绝对 MSE 的阶段差异作为 SE。

## 6.2 时间块重采样

正式主块长：

\[
40\ \mathrm{min}=240\ \mathrm{samples}.
\]

原因：等于输入历史窗口，保守覆盖相邻预测的主要重叠尺度。

重采样约束：

- 每个 validation fold 内独立移动块重采样；
- block 不跨 fold；
- block 不跨断点；
- 所有 \(d\) 使用同一组预生成重采样索引；
- 固定种子；
- common random numbers 保证 \(d\mapsto\operatorname{SE}_\Delta(d)\) 足够平滑。

正式选择使用：

```text
replicates = 500
primary block = 40 min
seed = 20260731
```

敏感性审计：

```text
22 min
60 min
```

这些是统计计算精度和冻结时间尺度，不是模型超参数。

## 6.3 配对 one-SE 规则

汇总配对差：

\[
\overline\Delta(d)
=
\frac{
\sum_f\sum_{t\in V_f}\Delta_{f,t}(d)
}{
\sum_f|V_f|
}.
\tag{6.2}
\]

由固定移动块 bootstrap 得到：

\[
\operatorname{SE}_\Delta(d).
\tag{6.3}
\]

定义：

\[
g(d)
=
\overline\Delta(d)
-
\operatorname{SE}_\Delta(d).
\tag{6.4}
\]

在 \(d=d_{\min}\)：

\[
\overline\Delta(d_{\min})=0.
\]

从 \(d_{\min}\) 向左进入更平滑区域，选择第一次满足：

\[
g(d)=0
\tag{6.5}
\]

的交点。

即：

\[
\boxed{
d_{\mathrm{P1SE}}
=
\inf
\left\{
\text{包含 }d_{\min}\text{ 的连通区域}:
\overline\Delta(d)\le
\operatorname{SE}_\Delta(d)
\right\}.
}
\tag{6.6}
\]

这里 `P1SE` 表示 `Paired One Standard Error`。

## 6.4 连通分支

若 \(g(d)\) 因有限样本出现多个交点：

- 只取包含 \(d_{\min}\) 的可接受连通分支；
- 从 \(d_{\min}\) 向左的第一个交点为正式边界；
- 不跳到远端偶然低谷。

## 6.5 根搜索

1. 先在最低盆地左侧和已评估 profile 点中寻找符号变化；
2. 不足时自适应补点评估；
3. 用 `brentq` 求交点；
4. 终止容差：
   \[
   |d_R-d_L|\le0.05.
   \]

若一直到共同下界仍满足配对 one-SE：

```text
PAIRED_ONE_SE_HITS_LOWER_BOUND
```

若没有连续括区间：

```text
PAIRED_ONE_SE_BOUNDARY_UNRESOLVED
```

---

# 7. 为什么不直接换成 d_min

本修订不应简单地把最终复杂度改为：

\[
d=d_{\min}.
\]

原因：

- \(d_{\min}\) 仍可能对训练棒时间块轻微过拟合；
- one-SE 的简约偏好仍然合理；
- 真正的问题是 V2 的 SE 估计方式，而不是 one-SE 原则本身。

因此保留：

\[
\boxed{\text{最低风险 + 更平滑等价解}}
\]

但把“等价”改为同一样本上的配对时间块证据。

---

# 8. 代码修改清单

## 8.1 `src/edf.py`

### 修改 `ContinuousProfile.evaluate`

除现有汇总外，缓存每个 fold 的：

- validation prediction；
- squared error；
- sample IDs 或严格顺序 hash；
- fold boundaries。

建议内存结构：

```python
self.prediction_cache[d] = [
    {
        "prediction": pred_f,
        "squared_error": (target_f - pred_f) ** 2,
    }
    for f in folds
]
```

结果包不必保存每个 \(d\) 的完整预测；正式返回包保存：

- 最终最低点；
- paired-one-SE 边界；
- 根括区间；
- 用于复核的少量邻近点。

完整 profile prediction cache 可在运行结束后清理。

### 替换 `resolve()`

拆成：

```python
discover_basins()
refine_local_minima()
resolve_selection_region()
```

删除：

```python
resolved = not unresolved and not upper_hit
```

改为独立字段：

```python
global_basin_discovery_pass
local_minimum_resolved
upper_bound_hit
far_field_interpolation_resolved
far_field_required_for_selection
```

### 删除正式全域门禁

下面变量只保留诊断：

```python
max_quadratic_interpolation_error
```

不能再单独决定 estimator 状态。

### 新增

```python
paired_block_difference(
    candidate_d,
    reference_d,
    block_samples,
    bootstrap_indices,
)
```

```python
paired_one_se(
    d_min,
    primary_block_samples=240,
    sensitivity_blocks=(132, 360),
)
```

## 8.2 `src/v2_runner.py`

原逻辑：

```python
estimator_stable = all(item["selection_resolved"] ...)
```

改为：

```python
estimator_stable = all(
    item["global_basin_discovery_pass"]
    and item["local_minimum_resolved"]
    and item["paired_one_se_boundary_resolved"]
    and not item["upper_bound_hit"]
    for item in directions
)
```

保存：

```text
d_min
d_paired_1se
lambda_full
paired_delta_at_selection
paired_se_at_selection
paired_block_sensitivity
```

## 8.3 `configs/frozen_protocol.yaml`

删除正式作用：

```yaml
profile_interpolation_relative_tolerance: 1e-4
```

可保留为：

```yaml
diagnostic_profile_interpolation_relative_tolerance: 1e-4
```

新增：

```yaml
continuous_edf_v2_1:
  search_coordinate: log_excess_df
  local_d_tolerance: 0.05
  max_actual_profile_evaluations: 120
  independent_local_refinements: 3
  paired_one_se:
    primary_block_min: 40
    sensitivity_block_min: [22, 60]
    bootstrap_replicates: 500
    seed: 20260731
    common_resamples_across_d: true
```

`max_actual_profile_evaluations=120` 是求解器预算，不是模型超参数。由于远场剪枝，实际通常不会用满。

## 8.4 新增脚本

```text
scripts/continue_local_profile_v2_1.py
scripts/select_paired_one_se_v2_1.py
scripts/refit_from_v2_1_selection.py
scripts/regenerate_v2_1_e3_e9.py
```

---

# 9. 哪些阶段需要重跑

## 9.1 可直接复用

- E0 baseline 导入；
- shared dataset；
- centered coordinate 定义；
- basis 规格；
- fold splits；
- purge；
- `FOLD_EDF_MAPS.csv` 的数值审计；
- V2 已评估的 69/76 个 profile 点；
- 旧模型逐样本预测；
- V1/V2 否定历史。

## 9.2 需要继续或重算

### E2B

加载 V2 已有 `CONTINUOUS_EDF_PROFILE.csv`，作为搜索初始缓存。

只补充：

- 最低盆地精修点；
- paired-one-SE 左边界所需点；
- 必要的全域盆地发现点。

不从零丢弃已有 profile。

### E2C

必须重算为 paired one-SE。

### E2D

若：

\[
|d_{\mathrm{P1SE}}-d_{\mathrm{V2,1SE}}|>0.05,
\]

重新 outer full-training refit。

### E3–E9

只要任一方向 \(d\) 变化，就全部重新生成：

- full surfaces；
- linear/nonlinear projections；
- derived Rank-1；
- C1/OOD；
- full-K OOF residual；
- residual PS/AR；
- leaderboards；
- bootstrap；
- final report。

旧 V2 最终模型不能与新选择混用。

---

# 10. 新结果状态

## 10.1 选择状态

### `SELECTION_RESOLVED_V2_1`

两个方向均满足：

- global basin discovery pass；
- local minimum resolved；
- paired one-SE boundary resolved；
- no upper-bound hit；
- fold EDF inversion pass。

### `LOCAL_MINIMUM_RESOLVED_PAIRED_ONE_SE_UNRESOLVED`

最低点确定，但 paired-one-SE 左边界不确定。

### `BASIN_DISCOVERY_UNRESOLVED`

仍无法排除另一个未解析低风险盆地。

## 10.2 估计器状态

### `ESTIMATOR_STABLE_V2_1`

选择 resolved，最终 refit、KKT、mesh 和 finite prediction 通过。

### `ESTIMATOR_SELECTION_UNRESOLVED`

模型可以运行，但平滑选择证据仍不足。

## 10.3 模型结论

保持原等级：

```text
CENTERED_FULL_URYSOHN_CONFIRMED
CENTERED_URYSOHN_PARETO
CENTERED_URYSOHN_IMPROVES_OLD_K_ONLY
CENTERED_FULL_URYSOHN_REJECTED
```

必须先判断 estimator 状态，再判断模型等级。

---

# 11. 三种可能结果及含义

## 情况 A：Sheet1→Sheet2 的 paired-one-SE 明显大于 1.0674

例如选择回到 \(d\approx4\sim7\)。

这表明：

> V2 第一方向的差性能主要来自绝对 fold-SE 导致的过度平滑。

随后看 outer RMSE 是否改善。不能预先保证。

## 情况 B：paired-one-SE 仍接近 1

这表明：

> 即使用配对时间块比较，第一方向仍支持极强平滑；方向不对称更可能是真实的数据迁移问题，而不是 one-SE 算法错误。

## 情况 C：两个方向的 paired-one-SE 均稳定，但最佳复杂度差异很大

例如仍近似：

\[
d_1\approx7,\qquad d_2\approx20.
\]

这意味着：

> 两个训练棒支持的可辨识曲面复杂度不同。当前两棒不足以冻结一个跨棒统一复杂度，但每个 outer predictor 可分别稳定选择。

不能把两个 \(d\) 的差异解释成两个不同物理系统自由度。

---

# 12. 新增测试

至少新增：

1. V2 profile cache 正确加载；
2. log-excess-EDF 映射单调；
3. 远场高损失区可被剪枝；
4. 远场插值不通过不自动否定局部最低点；
5. 合成双谷全部被发现；
6. 最低盆地三次独立精修一致；
7. 每个 \(d\) 保存 fold-level paired errors；
8. block 不跨 fold；
9. block 不跨断点；
10. common resamples across \(d\)；
11. \(\Delta(d_{\min})=0\)；
12. paired SE finite；
13. paired-one-SE 首个左交点；
14. 多交点时选择包含 \(d_{\min}\) 的连通分支；
15. lower-bound hit；
16. upper-bound hit；
17. d 改变后强制重跑 E2D–E9；
18. 旧 V2 surfaces 不被误用；
19. estimator 状态不再依赖 global interpolation error；
20. package roundtrip。

---

# 13. 输出文件

新增：

```text
results/
├── V2_PROFILE_CACHE_IMPORT.json
├── V2_1_BASIN_DISCOVERY.csv
├── V2_1_LOCAL_MINIMA.json
├── V2_1_PAIRED_ONE_SE_PROFILE.csv
├── V2_1_PAIRED_ONE_SE_SELECTION.json
├── V2_1_BLOCK_SENSITIVITY.csv
├── V2_1_SELECTION_DECISION.json
├── V2_1_FINAL_REPORT.md
└── diagnostics/
    ├── far_field_pruning.csv
    └── global_interpolation_diagnostic.csv
```

正式选择 JSON 示例：

```json
{
  "sheet1_to_sheet2": {
    "global_basin_discovery": "PASS",
    "local_minimum_resolved": true,
    "d_min": 7.4232,
    "paired_one_se_boundary_resolved": true,
    "d_paired_1se": null,
    "paired_delta": null,
    "paired_se": null,
    "upper_bound_hit": false,
    "far_field_interpolation_resolved": false,
    "far_field_required_for_selection": false
  }
}
```

---

# 14. 运行建议

从 V2 分支新建：

```bash
git checkout od-fuoi-nlinear-projection-cpu-confirm-v1
git checkout centered-od-fuoi-continuous-edf-cpu-confirm-v2
git checkout -b centered-od-fuoi-local-profile-paired-one-se-v2-1
```

推荐直接基于 V2 代码和结果缓存继续：

```bash
uv run python scripts/continue_local_profile_v2_1.py \
  --v2-results /path/to/CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS_bundle.zip \
  --output results_v2_1
```

然后：

```bash
uv run python scripts/select_paired_one_se_v2_1.py \
  --results results_v2_1
```

如果选择发生变化：

```bash
uv run python scripts/refit_from_v2_1_selection.py \
  --results results_v2_1
```

最终：

```bash
uv run python scripts/regenerate_v2_1_e3_e9.py \
  --results results_v2_1
```

这不是完整重跑 E0–E9。主要成本是补充少量 profile 点、配对块计算和可能的 E2D–E9 refit。

---

# 15. 打包合同

清理旧输出：

```bash
rm -rf return/CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS
rm -f return/CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS_bundle.zip
rm -f return/CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS_bundle.zip.sha256
```

收集：

```bash
uv run python scripts/build_return_bundle.py \
  --source-root . \
  --results results_v2_1 \
  --output return/CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS
```

Manifest：

```bash
uv run python scripts/build_manifest.py \
  --root return/CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS \
  --output return/CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS/MANIFEST.json
```

隐私校验：

```bash
uv run python scripts/validate_package.py \
  --package-dir return/CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS \
  --forbid "*.xlsx,.git,__pycache__,cache,*.tmp,raw_data"
```

压缩与 SHA256：

```bash
cd return

zip -r CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS_bundle.zip \
  CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS

sha256sum CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS_bundle.zip \
  > CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS_bundle.zip.sha256
```

回环校验：

```bash
uv run python ../scripts/validate_zip_roundtrip.py \
  --zip CENTERED_OD_FUOI_LOCAL_PROFILE_PAIRED_ONE_SE_V2_1_RESULTS_bundle.zip
```

最终必须打印：

```text
FINAL_ZIP=<absolute path>
FINAL_SHA256=<hash>
ZIP_SIZE=<bytes>
MANIFEST_FILE_COUNT=<count>
V2_RESULTS_SHA256=fc75a208c87855eed83de6ae886694b2d0b9813fa14a1cf60938e776a523037d
PROTOCOL_SHA256=<hash>
SELECTION_STATUS=<status>
ESTIMATOR_STATUS=<status>
VALIDATION_STATUS=PASS
```

---

# 16. 本轮唯一主结论格式

> V2.1 未修改 centered Full-Urysohn 模型，仅将 V2 的整域二次插值门禁改为全域盆地发现与局部最低点解析，并将基于非同分布 fold 绝对 MSE 的 one-SE 改为同一样本平方误差差的配对时间块 one-SE。两个方向的 \(d_{\min}\)、\(d_{\mathrm{P1SE}}\)、选择边界和块长敏感性分别为……。相对 V2，第一方向是否解除过度平滑为……。最终 centered Full-Urysohn 的双向与 pooled RMSE 为……。因此，选择状态登记为……，估计器状态登记为……，模型状态登记为……。

禁止把小数 EDF 解释为物理自由度，也禁止因为远场高损失区域未达到全局插值误差阈值而自动否定已经局部解析的最低盆地。
