# Excitation-Aware Spectral Predictive-State AR-RAPHU v0.3.2
## 从 v0.3.1 停止线继续的完整实验计划与启动合同

> **文档性质**：冻结执行合同。  
> **起点**：当前完整代码仓以及已冻结的
>
> ```text
> SPECTRAL_PS_AR_RAPHU_V031_CORE_RESULTS.zip
> ```
>
> **旧状态保留**：
>
> ```text
> E0_COMPONENT_IDENTITY_PASS
> E1R_REPRESENTATION_CERTIFIED_32x16
> E2A_ESTIMATOR_OR_DATA_EXCITATION_FAIL
> NEXT_ALLOWED_STAGE: STOP_SINGLE_KERNEL_CAPACITY
> ```
>
> **v0.3.2 目的**：不覆盖旧停止线；修复幅值域、S4 模型类和自然输入可辨识性混淆；重新验证二维 Urysohn 核容量。  
> **本轮第一停止点**：完成 R1、E1A、E2A0、E2A-NAT、E2A-PERM、E2A-SPACE 后生成 `V032_CAPACITY_DECISION.md` 并暂停。  
> **禁止**：未经容量决策直接启动 E2B/E3；恢复 group-prox；把旧 S4C 纳入二维模型通过门槛；让 Codex 自行改变阈值、basis 或输入激励。

---

# 0. 版本与目录

创建新分支：

```bash
git switch -c ps-ar-raphu-v032-domain-identifiability
```

若不用 Git，复制当前完整工程为：

```text
PS_AR_RAPHU_SPECTRAL_V032/
```

旧目录只读保留：

```text
results/spectral_v03/
results/spectral_v031/
configs/spectral_v03.yaml
configs/spectral_v031.yaml
```

新增：

```text
PROTOCOL_REVISION_V032.md
configs/spectral_v032.yaml

src/ar_raphu/spectral/
├── amplitude_domain.py
├── scenario_registry.py
├── excitation.py
├── operator_metrics.py
└── capacity_diagnostics.py

tests/
├── test_amplitude_domain.py
├── test_scenario_registry.py
├── test_s4u_generator.py
├── test_operator_closure.py
└── test_excitation_design.py

results/spectral_v032/
```

修改：

```text
src/ar_raphu/spectral/spline_basis.py
src/ar_raphu/spectral/synthetic_components.py
src/ar_raphu/spectral/design.py
tools/run_spectral_suite.py
tools/summarize_spectral_suite.py
```

不得修改旧结果文件。

---

# 1. 协议修订文件

`PROTOCOL_REVISION_V032.md` 固定写入：

```text
Revision:
Spectral PS-AR-RAPHU v0.3.1 -> v0.3.2

Frozen old result:
E2A stopped with 15/60 passing rows.
KKT residuals were approximately 1e-16.

Newly identified protocol problems:
1. The amplitude spline silently clipped values outside train Q01-Q99.
2. A majority of 64-step windows contained at least one clipped value.
3. Frozen AR-S4 is a conditional kernel K(tau,u;c), not the tested 2D K(tau,u).
4. Natural-input contribution prediction and full rectangular-surface recovery
   were incorrectly combined into one capacity gate.

Interpretation:
The old stop action remains valid under v0.3.1.
The old scientific label is not treated as a final rejection of the full-kernel method.
```

---

# 2. 固定配置

创建 `configs/spectral_v032.yaml`，内容必须等价于：

```yaml
schema_version: 3
status: DOMAIN_MODEL_CLASS_AND_EXCITATION_REPAIR

provenance:
  preserve_v031_results: results/spectral_v031
  preserve_v031_decision: results/spectral_v031/V031_CORE_DECISION.md

common:
  development_seeds: [0, 1, 2, 3, 4]
  confirmation_seeds:
    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
     110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
  n_samples_natural: 10000
  n_samples_excitation: 20000
  external_variables: 10
  L_x: 64
  L_y: 32
  primary_horizon: 1
  dtype_solver: float64
  numerical_jitter_relative: 1.0e-10

scenario_sets:
  core_2d_urysohn: [AR-S1, AR-S2, AR-S3, AR-S4U]
  conditional_challenge: [AR-S4]
  closed_loop_later: [AR-S7]

amplitude_domain:
  fit_rule: train_minmax_padding
  padding_fraction: 0.10
  core_quantiles: [0.01, 0.99]
  silent_clipping: forbidden
  maximum_validation_point_ood_rate: 0.0
  maximum_test_point_ood_rate: 0.0
  old_audit_quantiles: [0.01, 0.99]

basis_recertification:
  lag_basis_count: 32
  upper_lag_neighbor: 40
  identity_lag_reference: 64
  amplitude_basis_candidates: [16, 20, 24]
  degree: 3
  evaluation_grid_points: 401
  core_surface_max_nrmse: 0.05
  fit_surface_max_nrmse: 0.08
  lag_reference_ratio_max: 2.0
  selection_rule: smallest_amplitude_basis_that_passes

regularization:
  lag_smoothness_candidates: [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
  amplitude_smoothness_candidates: [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
  ridge_weight: 1.0e-8
  selection_metric: validation_contribution_mse
  selection_rule: minimum_validation_mse
  one_se_used_for_capacity: false

e2a0:
  forward_operator_tolerance: 1.0e-10
  target_replay_tolerance: 1.0e-12
  random_solver_relative_error: 1.0e-8
  direct_vs_matrix_free_relative_error: 1.0e-7

e2a_natural:
  validation_contribution_r2: 0.995
  empirical_operator_nrmse: 0.08
  kkt_relative_residual: 1.0e-8
  required_seed_fraction: 0.80
  full_surface_is_gate: false

excitation:
  modes: [permuted_marginal, space_filling_core]
  split_fractions: [0.70, 0.15, 0.15]
  space_filling_engine: scrambled_sobol
  scramble_seed_offset: 10000
  permuted_seed_offset: 20000
  burn_in: 64

e2a_excitation:
  validation_contribution_r2: 0.995
  surface_nrmse_max: 0.05
  projection_error_multiplier: 2.0
  kkt_relative_residual: 1.0e-8
  required_seed_fraction: 0.80

solver:
  direct_dimension_limit: 2000
  pcg_relative_residual: 1.0e-8
  pcg_max_iterations: 2000
  block_jacobi_preconditioner: true

later_support:
  null_quantile: 0.95
  positive_ablation_fold_fraction: 0.80
  recall_gate: 0.80
  fpr_gate: 0.10

later_rank:
  block_length: 64
  bootstrap_development: 100
  bootstrap_confirmation: 500
  bh_fdr: 0.10
  mode_stability: 0.70
```

Codex 不得改动配置值。

---

# 3. R1：旧协议域与模型类审计

## 3.1 目的

只生成证据，不训练模型：

1. 旧 1%–99% basis 的 point clipping rate；
2. 64 步窗口至少含一个裁剪值的比例；
3. 旧 AR-S4 的二维 Urysohn 模型类审计。

## 3.2 裁剪统计

对场景：

```text
AR-S1, AR-S2, AR-S3, AR-S4
```

和 seeds 0–4、变量 0–2，使用旧训练分位边界计算：

\[
r_{\mathrm{point}}
=
\frac{
\#\{x<a\text{ or }x>b\}
}{
N
},
\]

\[
r_{\mathrm{window}}
=
\frac{
\#\{\text{64-step window contains any OOD value}\}
}{
N_{\mathrm{windows}}
}.
\]

分别对 train/validation/test 报告。

输出：

```text
results/spectral_v032/R1/old_clipping_audit.csv
```

不得把这一统计用于选择新 padding。

## 3.3 S4 模型类审计

在 `scenario_registry.py` 注册：

```python
ScenarioSpec(
    name="AR-S4",
    canonical_name="AR-S4C",
    model_class="conditional_urysohn_3d",
    eligible_for_2d_capacity=False,
)
```

并用数值混合差分验证存在跨时滞交互：

\[
\Delta_{01}g
=
g(u_0+\delta,u_1+\delta)
-g(u_0+\delta,u_1)
-g(u_0,u_1+\delta)
+g(u_0,u_1).
\]

对普通二维加性 Urysohn，\(\Delta_{01}g=0\)；对 S4C 应显著非零。

输出：

```text
results/spectral_v032/R1/scenario_classification.json
```

R1 通过条件：

```text
AR-S4 eligible_for_2d_capacity == false
AR-S4U eligible_for_2d_capacity == true
```

---

# 4. 修改幅值基：禁止裁剪

## 4.1 新数据结构

在 `amplitude_domain.py` 实现：

```python
@dataclass(frozen=True)
class AmplitudeDomain:
    fit_lower: float
    fit_upper: float
    core_lower: float
    core_upper: float
    padding_fraction: float

    def in_domain_mask(self, values: np.ndarray) -> np.ndarray:
        ...
```

训练段：

```python
range_ = train_max - train_min
fit_lower = train_min - 0.10 * range_
fit_upper = train_max + 0.10 * range_
core_lower, core_upper = np.quantile(train_values, [0.01, 0.99])
```

## 4.2 `CenteredSplineBasis`

改为：

```python
CenteredSplineBasis.fit(
    train_values,
    n_basis=...,
    degree=3,
    domain=AmplitudeDomain(...),
)
```

`transform()`：

1. 检查域；
2. `strict=True` 时发现域外值直接抛出 `AmplitudeOutOfDomainError`；
3. 不使用 `np.clip`；
4. 不允许 `extrapolate=True`；
5. 可用 `transform_with_mask()` 返回 `(basis, mask)`，但结构实验必须要求 mask 全真。

旧 quantile API 只保留在 `legacy_transform_for_audit()`，不得被新 E1A/E2A 调用。

---

# 5. 新增 AR-S4U

## 5.1 不修改旧 AR-S4 语义

旧 AR-S4 保持原样，以保证历史结果可重放。

## 5.2 新场景

在合成生成器增加：

```text
AR-S4U
```

对 active variable \(j\)：

```python
for lag in range(L_x):
    u = x[time - 1 - lag, j]
    center = 8.0 + 12.0 / (1.0 + np.exp(-2.0 * u))
    q_value = normalized_gaussian_value(
        lag=lag,
        L_x=L_x,
        center=center,
        sigma=2.0,
    )
    contribution += q_value * truth_response(j, u)
```

注意：每个 lag 的 Gaussian 必须按该幅值对应的完整离散 lag 轴归一化后取该 lag 分量，不能只计算未归一化点值。

二维真核：

\[
K_j(\tau,u)
=
q_j(\tau;u)f_j(u).
\]

`true_kernel_surface()`：

- 支持 AR-S1/S2/S3/S4U；
- 遇到 AR-S4 必须抛出 `ModelClassMismatchError`；
- 不得再为 AR-S4 返回伪二维表面。

## 5.3 恒等重放

新增 E0U：

\[
\max_t
|
y_t^{latent}
-g_t^{AR}
-g_t^X
-\xi_t
|
\le10^{-12}.
\]

只有 S4U 恒等式通过，才可进入 E1A。

---

# 6. E1A：新域上的表示再认证

## 6.1 科学问题

> 在无裁剪 fit 域和核心报告域上，固定 lag basis 32、候选幅值 basis 16/20/24 是否能表示 S1/S2/S3/S4U 的真核？

## 6.2 场景

```text
AR-S1, AR-S2, AR-S3, AR-S4U
```

seeds 0–4，真实 active variables 0–2。

## 6.3 候选

主 lag：

```text
32
```

幅值：

```text
16, 20, 24
```

参考：

```text
identity lag 64 with same amplitude count
upper lag 40 with selected amplitude count
```

## 6.4 两个评价域

### Core surface

\[
u\in\mathcal I_j^{core}.
\]

### Fit surface

\[
u\in\mathcal I_j^{fit}.
\]

各 401 点。

## 6.5 选择规则

对每个幅值 basis count \(M_x\)：

```text
worst core NRMSE <= 0.05
worst fit NRMSE <= 0.08
each scenario's 32-lag error
    <= 2 * identity-lag error with same M_x
```

选择最小通过的 \(M_x\)。

如果 16、20、24 均失败：

```text
E1A_AMPLITUDE_REPRESENTATION_FAIL
```

停止，不允许自动增加到 28/32。

E1A 输出：

```text
results/spectral_v032/E1A/projection_core.csv
results/spectral_v032/E1A/projection_fit.csv
results/spectral_v032/E1A/representation_certificate.json
```

后续 basis 完全冻结。

---

# 7. E2A0：实现与算子闭环一致性

E2A0 不承担科学容量结论，只排除索引、目标和线性代数实现错误。

## 7.1 A0-Target

对所有 core scenarios、seed、active variable：

```python
replayed = replay_synthetic_components(...).x_contribution_by_variable[:, j]
direct = direct_apply_truth_kernel(...)
```

要求：

\[
\max|replayed-direct|
\le10^{-12}.
\]

## 7.2 A0-Design

取 E1A 投影系数 \(\theta^{proj}\)。

计算：

```python
via_matrix = design.matrix @ theta_proj
via_direct = direct_apply_projected_surface_to_windows(...)
```

要求：

\[
\max|via_{\mathrm{matrix}}-via_{\mathrm{direct}}|
\le10^{-10}.
\]

这一步检查 lag 方向、target-minus-one、flatten order 和中心化。

## 7.3 A0-Solver

在固定随机满秩问题上比较：

- direct Cholesky；
- matrix-free PCG；
- reference `numpy.linalg.solve`。

要求：

```text
relative coefficient error <= 1e-8
prediction relative error <= 1e-9
```

## 7.4 A0-OOD

验证：

- 训练值全部 in-domain；
- 域外值在 strict transform 中抛错；
- 没有 `np.clip`。

任一失败：

```text
E2A0_IMPLEMENTATION_CONSISTENCY_FAIL
```

停止。

---

# 8. E2A-NAT：自然输入贡献容量

## 8.1 目的

只回答：

> 在自然 AR 输入轨迹上，单变量外生贡献能否被准确预测？

不把完整矩形 surface NRMSE 作为门槛。

## 8.2 场景

```text
AR-S1, AR-S2, AR-S3, AR-S4U
```

seeds 0–4，variables 0–2。

模型只含一个 oracle variable，不含 AR、不含其他变量。

目标：

\[
y_{j,t}^{NAT}
=
g_{j,t}^{X,\star}.
\]

## 8.3 域检查

在构建设计前：

```text
validation point OOD rate == 0
test point OOD rate == 0
```

否则该 seed：

```text
AMPLITUDE_DOMAIN_COVERAGE_FAIL
```

不允许扩大域或裁剪。

## 8.4 平滑选择

候选由配置给定。

只按 validation contribution MSE 选择最小值；容量实验不使用 one-SE，因为 one-SE 会主动过度平滑并混淆容量。

不使用：

- surface truth；
- rank；
- test；
- support。

## 8.5 指标

主指标：

\[
R^2_{\mathrm{contrib,val}},
\]

经验算子 NRMSE：

\[
E_{\mathbb P}^{val}
=
\frac{
\|\widehat g-g^\star\|_2
}{
\|g^\star-\bar g^\star\|_2
}.
\]

另报告：

- train/test contribution metrics；
- core/fit surface NRMSE；
- Gram condition number；
- effective rank；
- KKT。

surface 只作诊断，不参与 NAT 通过判定。

## 8.6 通过规则

每个场景、每个变量至少 4/5 seeds：

```text
validation contribution R2 >= 0.995
empirical operator NRMSE <= 0.08
relative KKT residual <= 1e-8
```

场景通过要求三个 active variables 均通过。

---

# 9. E2A-PERM：保持边际、打破串行相关

## 9.1 构造

对每个 seed/variable：

1. 取该 seed 自然训练段的幅值样本；
2. 使用 `seed + 20000` 做随机排列；
3. 循环排列至长度 20000；
4. 丢弃前 64 点；
5. 用同一真核生成无噪声单变量贡献。

保持经验边际近似不变，但打破 AR 串行相关。

## 9.2 模型和选择

使用 E1A 冻结 basis。

平滑仍只按 validation contribution MSE 选择。

## 9.3 指标

与 NAT 相同，并增加：

- design condition number；
- effective rank improvement relative to NAT；
- core surface NRMSE。

PERM 不单独决定 full surface 通过，但用于原因分类。

---

# 10. E2A-SPACE：空间填充结构容量

## 10.1 构造

对每个 seed/variable，在

\[
\mathcal I_j^{core}
\]

上生成长度 20000 的 scrambled Sobol 1D 序列：

```python
sampler = scipy.stats.qmc.Sobol(
    d=1,
    scramble=True,
    seed=seed + 10000,
)
```

映射到 core domain。

按 70%/15%/15% 时间切分。

使用同一真核生成无噪声贡献。

## 10.2 目的

回答：

> 在充分覆盖、低串行相关的输入下，完整二维核面能否被恢复？

## 10.3 指标

- contribution R²；
- core surface NRMSE；
- projection oracle NRMSE；
- excess surface error；
- condition number；
- effective rank；
- rank-1/rank-2/full 截断误差；
- KKT。

## 10.4 通过规则

每个场景、变量至少 4/5 seeds：

```text
validation contribution R2 >= 0.995

core surface NRMSE
  <= max(
       0.05,
       2 * E1A core projection NRMSE
     )

relative KKT residual <= 1e-8
```

S1/S2 的 rank-1 截断：

```text
rank1 validation MSE <= 1.05 * full validation MSE
```

S3 的 rank-2 截断：

```text
rank2 validation MSE <= 1.05 * full validation MSE
rank2 captures >= 0.80 of rank1-to-full reducible MSE
```

S4U 不预注册整数 rank，只报告谱尾。

---

# 11. v0.3.2 容量决策矩阵

生成：

```text
results/spectral_v032/V032_CAPACITY_DECISION.md
```

状态必须按下表自动产生。

## 11.1 停止映射

```text
R1 fail
-> STOP_SCENARIO_OR_DOMAIN_AUDIT

E1A fail
-> STOP_REPRESENTATION

E2A0 fail
-> STOP_IMPLEMENTATION_CONSISTENCY
```

## 11.2 科学映射

### NAT、PERM、SPACE 都通过

```text
CAPACITY_STATUS:
FULL_2D_URYSOHN_CAPACITY_VALID
NEXT_ALLOWED_STAGE:
ALLOW_E2B
```

### NAT 贡献通过，SPACE surface 通过，但 NAT surface 较差

```text
CAPACITY_STATUS:
NATURAL_PREDICTIVE_EQUIVALENCE_VALID
FULL_SURFACE_REQUIRES_DESIGNED_EXCITATION
NEXT_ALLOWED_STAGE:
ALLOW_E2B_WITH_IDENTIFIABILITY_QUALIFIER
```

### NAT 失败、PERM/SPACE 通过

```text
CAPACITY_STATUS:
NATURAL_LAG_CORRELATION_LIMIT
NEXT_ALLOWED_STAGE:
ALLOW_E2B_PREDICTION_ONLY
```

### NAT/PERM 失败、SPACE 通过

```text
CAPACITY_STATUS:
NATURAL_AMPLITUDE_COVERAGE_LIMIT
NEXT_ALLOWED_STAGE:
ALLOW_E2B_PREDICTION_ONLY
```

### SPACE 失败

```text
CAPACITY_STATUS:
FULL_KERNEL_ESTIMATOR_OR_BASIS_FAIL
NEXT_ALLOWED_STAGE:
STOP_CAPACITY
```

旧 S4C 的结果永远不进入上述决策。

---

# 12. 第一批重新启动顺序

当前只执行：

```text
Step 1  创建 v0.3.2 分支和协议文件
Step 2  R1 旧裁剪/模型类审计
Step 3  实现无裁剪 amplitude domain
Step 4  新增 AR-S4U 和 E0U 恒等测试
Step 5  E1A 新域表示认证
Step 6  E2A0 实现一致性
Step 7  E2A-NAT
Step 8  E2A-PERM
Step 9  E2A-SPACE
Step 10  生成 V032_CAPACITY_DECISION.md
Step 11  暂停并打包
```

不得自动运行 E2B/E3。

---

# 13. 容量通过后的 E2B 设计

只有 `NEXT_ALLOWED_STAGE` 允许时才实施。

## 13.1 E2B-NAT

目标：

\[
g_t^{X,\star}
=
\sum_{j\in S^\star}g_{j,t}^{X,\star}.
\]

模型只含 oracle active variables，不含 AR。

主门槛：

```text
validation total contribution R2 >= 0.95
```

不再以每个变量完整 surface NRMSE 作为自然输入主门槛。

报告：

- 总贡献；
- 每变量经验算子误差；
- joint Gram condition number；
- variable contribution cross-correlation。

## 13.2 E2B-SPACE

为每个 active variable生成独立 scrambled Sobol 输入，联合生成总贡献。

用于验证多变量完整结构可辨识。

门槛：

```text
total contribution R2 >= 0.995
median active core surface NRMSE <= 0.08
```

---

# 14. E3 双残差设计修订

E3 只在 core M2 场景运行：

```text
AR-S1, AR-S3, AR-S4U
```

AR-S7 只有在 `scenario_registry` 明确标记属于 M2 后才加入。

比较：

```text
O  oracle AR subtraction
Y  only-y residualization
D  double residualization
J  joint convex AR+X
```

主评价分成两类：

## 14.1 自然预测结构

- total contribution RMSE；
- empirical operator NRMSE；
- conditional predictive support；
- validation prediction RMSE。

## 14.2 完整核结构

只在 excitation certificate 通过的合成对照上评价 surface/rank。

E3 不再要求自然闭环数据恢复完整矩形面。

---

# 15. E4/E5 的最终分拆

## E4P：预测 support

自然工况，依据：

- 样本外块消融；
- contribution energy；
- D5 路径稳定性。

## E4S：结构 support

SPACE 激励合成场景，依据：

- 核范数 null distribution；
- surface recovery；
- true support recall/FPR。

## E5P：预测 rank

自然工况下，通过 rank 截断的样本外增益选择。

## E5S：结构 rank

SPACE 激励下，通过：

- rank-1 null bootstrap；
- 谱间隔；
- 第二模态稳定性；
- 32/40 basis 一致性。

真实 CZ 数据默认只进入 E4P/E5P。

---

# 16. 目标测试集

正式启动前只运行以下针对性测试：

```bash
python -m pytest \
  tests/test_spectral_contracts.py \
  tests/test_synthetic_components.py \
  tests/test_spectral_design.py \
  tests/test_spectral_projection_repair.py \
  tests/test_spectral_solver.py \
  tests/test_gram_svd.py \
  tests/test_amplitude_domain.py \
  tests/test_scenario_registry.py \
  tests/test_s4u_generator.py \
  tests/test_operator_closure.py \
  tests/test_excitation_design.py \
  -q
```

必须覆盖：

1. 新 transform 不含 `np.clip`；
2. train fit-domain 覆盖率 100%；
3. 域外 strict transform 抛错；
4. S4C 被拒绝为二维 surface；
5. S4U 恒等重放误差 \(\le10^{-12}\)；
6. design matrix 与直接 projected-kernel application 一致；
7. Sobol/permute 输入可复现；
8. direct 与 PCG 解一致。

不运行全仓库测试、旧 M7/M8、SHA、manifest 或 HTML。

---

# 17. CLI

扩展 `run_spectral_suite.py`：

```text
--experiment {
  R1,
  E1A,
  E2A0,
  E2A_NAT,
  E2A_PERM,
  E2A_SPACE
}
--stage {development,confirmation}
--device {cpu,cuda}
--config configs/spectral_v032.yaml
--force
```

不允许 CLI 覆盖：

- padding；
- basis；
- scenario set；
- threshold；
- excitation length；
- seed；
- smoothing grid。

---

# 18. 启动命令

## 18.1 R1

```bash
python tools/run_spectral_suite.py \
  --config configs/spectral_v032.yaml \
  --experiment R1 \
  --stage development \
  --device cpu
```

## 18.2 E1A

```bash
python tools/run_spectral_suite.py \
  --config configs/spectral_v032.yaml \
  --experiment E1A \
  --stage development \
  --device cpu
```

## 18.3 E2A0

```bash
python tools/run_spectral_suite.py \
  --config configs/spectral_v032.yaml \
  --experiment E2A0 \
  --stage development \
  --device cpu
```

## 18.4 三类容量

```bash
python tools/run_spectral_suite.py \
  --config configs/spectral_v032.yaml \
  --experiment E2A_NAT \
  --stage development \
  --device cpu

python tools/run_spectral_suite.py \
  --config configs/spectral_v032.yaml \
  --experiment E2A_PERM \
  --stage development \
  --device cpu

python tools/run_spectral_suite.py \
  --config configs/spectral_v032.yaml \
  --experiment E2A_SPACE \
  --stage development \
  --device cpu
```

这些实验以 FP64 线性代数为主，GPU 空闲正常。不得为提高 GPU 利用率改用 Adam。

---

# 19. 输出结构

```text
results/spectral_v032/
├── R1/
├── E1A/
├── E2A0/
├── E2A_NAT/
├── E2A_PERM/
├── E2A_SPACE/
├── V032_CAPACITY_DECISION.md
└── spectral_v032_capacity_summary.csv
```

每个实验只保存：

```text
contract.json
config.json
summary.json
metrics.csv
fit.npz
```

R1 增加：

```text
old_clipping_audit.csv
scenario_classification.json
```

E1A 增加：

```text
projection_core.csv
projection_fit.csv
representation_certificate.json
```

---

# 20. 自动决策文件

`V032_CAPACITY_DECISION.md` 只能包含：

```text
V031_FROZEN_STATUS:
R1_STATUS:
E0U_STATUS:
E1A_STATUS:
E2A0_STATUS:
E2A_NAT_STATUS:
E2A_PERM_STATUS:
E2A_SPACE_STATUS:

AMPLITUDE_DOMAIN_VALID:
MODEL_CLASS_REGISTRY_VALID:
REPRESENTATION_VALID:
IMPLEMENTATION_CONSISTENCY_VALID:
NATURAL_PREDICTIVE_CAPACITY:
DECORRELATED_CAPACITY:
SPACE_FILLING_SURFACE_CAPACITY:
PRIMARY_LIMITATION:
NEXT_ALLOWED_STAGE:
```

不得由 Codex自由解释。

---

# 21. 最终打包

本轮只打包一次，不生成 SHA/manifest：

```bash
rm -f SPECTRAL_PS_AR_RAPHU_V032_CAPACITY_RESULTS.zip

zip -r SPECTRAL_PS_AR_RAPHU_V032_CAPACITY_RESULTS.zip \
  PROTOCOL_REVISION_V032.md \
  configs/spectral_v032.yaml \
  src/ar_raphu/spectral \
  tools/run_spectral_suite.py \
  tools/summarize_spectral_suite.py \
  tests/test_spectral_*.py \
  tests/test_amplitude_domain.py \
  tests/test_scenario_registry.py \
  tests/test_s4u_generator.py \
  tests/test_operator_closure.py \
  tests/test_excitation_design.py \
  results/spectral_v032

test -f results/spectral_v032/V032_CAPACITY_DECISION.md
test -f results/spectral_v032/spectral_v032_capacity_summary.csv
unzip -t SPECTRAL_PS_AR_RAPHU_V032_CAPACITY_RESULTS.zip
```

终端打印：

```text
FINAL_PACKAGE=<absolute path>
NEXT_ALLOWED_STAGE=<value from V032_CAPACITY_DECISION.md>
```

完成后暂停，不实现 E2B/E3。
