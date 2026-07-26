# Spectral Predictive-State AR-RAPHU v0.3  
## 从当前 D1–D6 结果继续的完整实验与实现合同

> **文档性质**：执行合同。Codex 不得自行补充模型、阈值、数据目标或停止条件。  
> **起点**：`PS_AR_RAPHU_V3_DIAGNOSTICS_RESULTS.zip` 中的源码，加上原停止线 `source/`。  
> **终点**：验证“交叉拟合双残差 + 全变量平滑 Urysohn + Gram 谱分解”的可行性，并决定 uniform spectral 或 D5-adaptive spectral 是否进入最终 v3。  
> **本轮不做**：公开数据、真实 CZ、MPC、PLC、最终论文图、全仓库 SHA/manifest。  
> **审计策略**：只保留目标语义测试、时间泄漏测试、凸求解测试和最终单次打包。

---

# 0. 防止再次出现设计错误的总规则

每个实验在代码中必须声明以下字段：

```json
{
  "scientific_question": "...",
  "target_contains_ar": true,
  "model_contains_ar": true,
  "target_contains_x": true,
  "model_contains_x": true,
  "truth_used_for_training": false,
  "truth_used_for_evaluation": true,
  "support_used_for_training": "all|oracle",
  "hyperparameter_selection_metric": "validation_prediction_loss_only",
  "rank_inputs_used_for_selection": false,
  "test_used_for_selection": false
}
```

运行器在启动时自动验证：

1. 若 `target_contains_ar=true` 且 `model_contains_ar=false`，只能标记为 `ORACLE_COMPONENT_DIAGNOSTIC`，不得使用完整 \(y\) 作为容量验收目标；
2. 若模型用于正式结构恢复，必须对 \(y\) 和外生设计矩阵同时残差化；
3. truth 只能用于合成评价和 oracle capacity，不得用于正式超参数选择；
4. support、奇异值、rank、test loss 不得进入 grid/smoothing 选择；
5. 每项失败必须按预注册映射解释，不得自动换网络或增加 epoch。

---

# 1. 源代码策略

## 1.1 分支

```bash
git switch -c ps-ar-raphu-v4-spectral
```

不覆盖 v2/v3 diagnostics。

## 1.2 复用模块

从当前源码复用：

```text
src/ar_raphu/synthetic.py
src/ar_raphu/sequence_data.py
src/ar_raphu/data_protocol.py
src/ar_raphu/preprocessing.py
src/ar_raphu/model.py
src/ar_raphu/training.py
src/ar_raphu/phase1_evidence.py
src/ar_raphu/statistics.py
src/ar_raphu/diagnostics/gate_fista.py
STAGE1_DUAL_SOLVER_V20_bundle/stage1/variational_spline.py
```

不得复用：

```text
training.prune_external_path
A-support-only M8 dispatch
原 group-prox 支持选择
```

## 1.3 新增目录

```text
src/ar_raphu/spectral/
├── __init__.py
├── contracts.py
├── synthetic_components.py
├── spline_basis.py
├── design.py
├── nuisance.py
├── crossfit.py
├── penalties.py
├── solver.py
├── gram_svd.py
├── support_evidence.py
├── rank_inference.py
├── adaptive_weights.py
├── recombine.py
└── metrics.py

tools/
├── run_spectral_job.py
├── run_spectral_suite.py
└── summarize_spectral_suite.py

configs/
└── spectral_v03.yaml

tests/
├── test_spectral_contracts.py
├── test_synthetic_components.py
├── test_spectral_design.py
├── test_double_residualization.py
├── test_spectral_solver.py
├── test_gram_svd.py
└── test_rank_bootstrap.py
```

---

# 2. 精简审计

正式运行前只执行：

```bash
python -m pytest \
  tests/test_ar_raphu_model.py \
  tests/test_sequence_training.py \
  tests/test_spectral_contracts.py \
  tests/test_synthetic_components.py \
  tests/test_spectral_design.py \
  tests/test_double_residualization.py \
  tests/test_spectral_solver.py \
  tests/test_gram_svd.py \
  tests/test_rank_bootstrap.py \
  -q
```

不执行：

- 全仓库测试；
- 每文件 SHA；
- manifest；
- HTML；
- Git tag 检查；
- 每阶段压缩包；
- 旧 M7/M8 审计；
- checkpoint 重放。

最终只生成一个 zip。

---

# 3. 固定配置

创建 `configs/spectral_v03.yaml`：

```yaml
schema_version: 1
status: PROPOSED_CORE_VALIDATION

common:
  development_seeds: [0, 1, 2, 3, 4]
  confirmation_seeds:
    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
     110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
  null_seeds:
    [200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
     210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
     220, 221, 222, 223, 224, 225, 226, 227, 228, 229,
     230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
     240, 241, 242, 243, 244, 245, 246, 247, 248, 249]
  n_samples: 10000
  external_variables: 10
  L_x: 64
  L_y: 32
  horizons: [1, 5, 10, 30, 60]
  primary_horizon: 1
  dtype_solver: float64
  block_length: 64
  bootstrap_replicates_development: 100
  bootstrap_replicates_confirmation: 500
  numerical_jitter_relative: 1.0e-10

external_basis:
  degree: 3
  lag_basis_candidates: [5, 8]
  amplitude_basis_candidates: [8, 12]
  fallback_lag_basis: 12
  fallback_amplitude_basis: 16
  amplitude_quantiles: [0.01, 0.99]

ar_nuisance_basis:
  degree: 3
  lag_basis_count: 6
  amplitude_basis_count: 8
  ridge_candidates: [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]

spectral_regularization:
  lag_smoothness_candidates: [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
  amplitude_smoothness_candidates: [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
  ridge_weight: 1.0e-6
  selection_rule: joint_one_se
  complexity_order:
    - total_basis_coefficients
    - stronger_total_smoothing
    - smaller_lag_basis
    - smaller_amplitude_basis

crossfit:
  folds: 4
  initial_nuisance_prefix_targets: 2000
  purge_gap: 65
  nuisance_selection_tail_fraction: 0.20

support:
  null_quantile: 0.95
  required_positive_ablation_fold_fraction: 0.80
  synthetic_recall_gate: 0.80
  synthetic_fpr_gate: 0.10

rank:
  bootstrap_alpha: 0.05
  bh_fdr: 0.10
  required_positive_gain_fold_fraction: 0.80
  mode_stability_threshold: 0.70
  neighbor_configuration_agreement: 2

adaptive:
  enabled_as_comparison_only: true
  fista_lambda_ratios: [0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0]
  score_epsilon: 0.05
  weight_min: 0.25
  weight_max: 4.0
```

Codex 不得修改这些值。

---

# 4. 基础实现

## 4.1 合成分量重放

在 `synthetic_components.py` 实现：

```python
@dataclass(frozen=True)
class SyntheticComponents:
    ar_contribution: np.ndarray
    x_contribution_by_variable: np.ndarray
    x_total_contribution: np.ndarray
    process_innovation: np.ndarray
    measurement_noise: np.ndarray
```

函数：

```python
def replay_synthetic_components(
    sequence: SyntheticSequence,
) -> SyntheticComponents:
    ...
```

必须使用：

- `sequence.x`
- `sequence.y_latent`
- `sequence.y_measurement_clean`
- `sequence.y_observed`
- `sequence.truth["q_y"]`
- `q_primary`
- `q_secondary`
- 与原生成器完全相同的 `_truth_response`、`_second_truth_response`、`_ar_response`

逐时重放：

\[
g_t^{AR}
=
q_y^\top f_y(y^{latent}_{t-1:t-L_y}),
\]

\[
g_{j,t}^{X}
=
\sum_\tau K_j^\star(\tau,x_{j,t-1-\tau}),
\]

\[
\xi_t
=
y_t^{latent}
-
g_t^{AR}
-
\sum_jg_{j,t}^{X}.
\]

测量噪声：

\[
\eta_t
=
y_t^{observed}
-
y_t^{measurement\_clean}.
\]

测试必须验证：

```text
max_abs(latent - ar - x_total - process_innovation) <= 1e-10
max_abs(observed - measurement_clean - measurement_noise) <= 1e-10
```

这一步修复 D1/D2 的目标语义错误。

## 4.2 外生设计

`design.py` 实现：

```python
@dataclass
class SpectralDesign:
    matrix: np.ndarray
    variable_slices: dict[int, slice]
    lag_basis: np.ndarray
    amplitude_bases: list[np.ndarray]
    lag_gram: np.ndarray
    amplitude_grams: list[np.ndarray]
    target_indices: np.ndarray
```

单样本设计：

\[
\Phi_{j,t}^{a,b}
=
\sum_{\tau=0}^{L_x-1}
c_a(\tau)b_{j,b}(x_{j,t-\tau}).
\]

要求：

- lag index 0 对应 origin time；
- 与现有 `PreparedDirectForecastData` 一致；
- 幅值基只用训练段拟合；
- 幅值基按训练经验均值中心化；
- 不读取未来 X。

## 4.3 AR nuisance 设计

定义

\[
\Psi_t^{a,b}
=
\sum_{\ell=1}^{L_y}
c^y_a(\ell)b^y_b(y_{t-\ell+1}).
\]

使用同一套三次 B 样条构建。

## 4.4 nuisance ridge

在每个前向交叉拟合折中：

\[
\widehat\mu(Z)
=
\Psi a,
\]

\[
\widehat\pi(Z)
=
\Psi M.
\]

分别求解：

\[
a
=
(\Psi^\top\Psi+\lambda_yI)^{-1}\Psi^\top y,
\]

\[
M
=
(\Psi^\top\Psi+\lambda_\phi I)^{-1}\Psi^\top\Phi.
\]

\(\lambda_y\) 和 \(\lambda_\phi\) 分别在训练前缀尾部验证选择。

## 4.5 双残差输出

保存：

```text
y_residual.npy
phi_residual.npy
nuisance_summary.json
```

`nuisance_summary.json` 必须包含：

- out-of-fold y RMSE/R²；
- feature residual Frobenius R²；
- residual design condition number；
- residual feature与 nuisance basis 最大绝对相关；
- 每折训练/评价时间边界；
- purge gap。

---

# 5. 凸谱求解器

## 5.1 惩罚归一化

二阶差分矩阵构成：

\[
R_\tau=D_\tau^\top D_\tau,
\qquad
R_x=D_x^\top D_x.
\]

为了使不同 grid 的 \(\lambda\) 可比较，归一化：

\[
\bar R
=
\frac{d}{\operatorname{tr}(R)}R,
\]

其中 \(d\) 是矩阵维数。

每个变量惩罚块：

\[
P_j
=
\lambda_\tau
(\bar R_\tau\otimes I_x)
+
\lambda_x
(I_\tau\otimes\bar R_x)
+
\lambda_0
(G_{x,j}\otimes G_\tau).
\]

## 5.2 直接求解

维数小于 2000 时固定使用 FP64 Cholesky：

```python
system = X.T @ X / n + penalty
rhs = X.T @ y / n
theta = scipy.linalg.cho_solve(
    scipy.linalg.cho_factor(system + jitter * I),
    rhs,
)
```

不得使用 Adam 或 FISTA 求解主 full-kernel ridge。

## 5.3 KKT

保存：

\[
r_{\mathrm{KKT}}
=
\|
(X^\top X/n+P)\widehat\theta-X^\top y/n
\|_2.
\]

要求：

```text
relative KKT residual <= 1e-8
```

---

# 6. 超参数选择

每个候选由

```text
(M_tau, M_x, lambda_tau, lambda_x)
```

组成。

只使用 validation prediction MSE。

联合 one-SE：

1. 找到平均 validation MSE 最小候选；
2. 用折间标准误构造 one-SE 集；
3. 在 one-SE 集内按以下顺序选择：
   - 最少总核系数；
   - 最大 \(\lambda_\tau\lambda_x\)；
   - 更小 \(M_\tau\)；
   - 更小 \(M_x\)；
   - 固定声明顺序。

禁止使用：

- support recall/FPR；
- kernel truth；
- singular values；
- rank p-value；
- test MSE；
- 图形美观。

---

# 7. 实验 E0：目标与 oracle ceiling

## 7.1 目的

在任何模型训练前回答：

- 完整目标由哪些分量组成？
- X-only、AR-only 理论上各能解释多少？
- 当前实验的模型与目标是否匹配？

## 7.2 场景

```text
AR-S0
AR-S1
AR-S2
AR-S3
AR-S4
AR-S7
```

seeds `[0,1,2,3,4]`。

## 7.3 计算

对 train/validation/test 分别计算：

- \(R^2(y,g_X^\star)\)；
- \(R^2(y,g_{AR}^\star)\)；
- \(R^2(y,g_X^\star+g_{AR}^\star)\)；
- \(R^2(g_X^\star,g_X^\star)\)；
- process/noise variance ratios；
- active variable contribution variances。

## 7.4 停止条件

任何分量重放恒等式失败，整个 suite 停止。

输出：

```text
results/spectral_v03/E0/oracle_ceiling.csv
results/spectral_v03/E0/component_identity.json
```

---

# 8. 实验 E1：样条投影 oracle

## 8.1 目的

先判断基函数空间是否有能力表示真核，避免把 basis approximation failure 误判为估计器失败。

## 8.2 场景

```text
AR-S1：Gamma rank-1
AR-S2：非 Gamma rank-1
AR-S3：rank-2
AR-S4：幅值相关动态时滞
```

## 8.3 方法

对每个真实 active variable：

1. 在 lag \(0,\ldots,63\)；
2. 在训练分位幅值区间的 401 点；
3. 构造真实核矩阵；
4. 用 Gram 加权最小二乘投影到每个候选样条空间；
5. 计算 projection surface NRMSE。

## 8.4 固定 fallback

主候选中若某场景任一 active variable 的最佳 projection NRMSE > 0.10，则该场景容量实验使用固定 fallback：

```text
M_tau = 12
M_x = 16
```

不得继续扩大。

输出：

```text
projection_oracle.csv
selected_capacity_basis.json
```

真核只用于 E1/E2 评价，不用于正式 E3 之后的超参数选择。

---

# 9. 实验 E2：修正后的核容量实验

## 9.1 修复的问题

不再让 X-only 模型预测含强 AR 的完整 \(y\)。

目标固定为：

\[
y_t^{capacity}
=
g_{X,t}^\star.
\]

模型：

- X-only；
- oracle active support；
- full tensor Urysohn；
- 无 AR；
- 无 support penalty。

## 9.2 模型

对 full fit 做 Gram SVD，得到：

- full；
- rank-1 truncation；
- rank-2 truncation。

不单独训练神经 rank-1/rank-2。

## 9.3 指标

- contribution RMSE/R²；
- kernel surface NRMSE；
- full 相对 projection oracle 的 excess error；
- rank-1、rank-2 截断误差；
- \(\eta(1),\eta(2)\)。

定义：

\[
E_{\mathrm{proj}}
=
\|K^\star-K^{proj}\|_G,
\]

\[
E_{\mathrm{fit}}
=
\|K^\star-\widehat K\|_G.
\]

容量比：

\[
R_{\mathrm{capacity}}
=
\frac{E_{\mathrm{fit}}}
{E_{\mathrm{proj}}+10^{-8}}.
\]

## 9.4 通过规则

### full capacity

至少 4/5 development seeds：

```text
contribution validation R2 >= 0.95
R_capacity <= 1.50
```

若 projection error 接近零，则改用：

```text
surface NRMSE <= 0.15
```

### rank-1 truth

AR-S1、AR-S2：

```text
rank1 validation MSE <= 1.05 * full validation MSE
eta(1) <= 0.10
```

至少 4/5 seeds。

### rank-2 truth

AR-S3：

```text
rank2 validation MSE <= 1.05 * full validation MSE
rank2 captures >= 0.80 of (rank1 - full) reducible MSE
```

至少 4/5 seeds。

若 E2 失败，停止，不进入正式 support/rank 实验。

---

# 10. 实验 E3：双残差化有效性

## 10.1 目的

验证正式结构路径不能只残差化 \(y\)，而应同时残差化外生设计。

## 10.2 场景

```text
AR-S1
AR-S3
AR-S7
```

## 10.3 四种方法

### O：oracle AR subtraction

\[
y^{O}=y-g_{AR}^\star.
\]

只作合成参考。

### Y：只残差化 y

\[
\widetilde y=y-\widehat\mu(Z),
\qquad
\Phi\text{ 不变}.
\]

对应旧 D3 思路。

### D：双残差化

\[
\widetilde y=y-\widehat\mu(Z),
\]

\[
\widetilde\Phi=\Phi-\widehat\pi(Z).
\]

这是主方法。

### J：联合凸 AR+X

同时放入 AR spline design 和 X full-kernel design，做统一 ridge；作为不正交化对照。

## 10.4 所有方法

- 全部十个变量；
- 相同外生 basis；
- 相同 validation-only 超参数选择；
- 不使用 support penalty；
- 不读取 test 选择。

## 10.5 nuisance 适用检查

若任一 seed：

```text
condition_number(phi_residual_gram) > 1e8
```

则该 seed 标记：

```text
CONDITIONAL_IDENTIFIABILITY_WEAK
```

不计入机制恢复通过率。

若：

```text
max_abs_corr(phi_residual, nuisance_basis) > 0.10
```

则标记：

```text
NUISANCE_RESIDUALIZATION_INADEQUATE
```

不允许继续解释 support/rank。

## 10.6 主比较

对每个 seed 计算估计核到 oracle 方法 O 的距离：

\[
d_M
=
\|\widehat K^M-\widehat K^O\|_G,
\quad
M\in\{Y,D,J\}.
\]

双残差化通过条件：

```text
D 的 d_M < Y 的 d_M，在至少 4/5 seeds
median(d_D / d_Y) <= 0.85
```

同时 validation RMSE 不得比 O 恶化超过 5%。

若失败：

```text
E3_NUISANCE_MODEL_INADEQUATE
```

停止 E4/E5；不得自动换 MLP 或增加网络。

---

# 11. 实验 E4：全变量 support 证据

## 11.1 场景

```text
AR-S0
AR-S1
AR-S2
AR-S3
AR-S4
AR-S7
```

## 11.2 主估计器

使用 E3 通过的双残差化和 uniform spectral ridge。

所有变量始终进入模型。

## 11.3 null cutoff

使用 50 个 `null_seeds` 的 AR-S0：

1. 拟合全变量 uniform spectral；
2. 对每个变量计算
   \[
   T_j=\|\widetilde\Theta_j\|_F;
   \]
3. 每个 seed 取
   \[
   T_{\max}=\max_jT_j;
   \]
4. cutoff 为 \(T_{\max}\) 的 95% 分位。

这只用于合成 support 评价。

## 11.4 消融

对每个变量：

1. 固定已选 hyperparameters；
2. 去掉该变量；
3. 重新求解凸问题；
4. 在每个外层 validation block 计算
   \[
   \Delta_{j,k}^{abl}.
   \]

要求正消融收益的折比例：

\[
\frac{
\#\{k:\Delta_{j,k}^{abl}>0\}
}{K}
\ge0.80.
\]

## 11.5 合成 support 标签

仅在合成评价中：

\[
j\in\widehat S
\]

当且仅当：

```text
kernel_norm > AR-S0 max-null cutoff
AND
positive ablation fold fraction >= 0.80
```

## 11.6 通过规则

在 20 个 confirmation seeds：

- AR-S0：FPR \(\le0.10\)；
- AR-S1：recall \(\ge0.80\)，FPR \(\le0.10\)；
- AR-S2：recall \(\ge0.80\)，FPR \(\le0.10\)；
- AR-S3：recall \(\ge0.80\)，FPR \(\le0.10\)。

AR-S4 报告连续结果，不作为第一轮硬门槛。

AR-S7 只解释为 conditional predictive support，不写成因果支持。

---

# 12. 实验 E5：rank 统计验证

## 12.1 场景

```text
AR-S1
AR-S2
AR-S3
AR-S4
```

## 12.2 固定前提

- 使用 E4 已冻结的 grid/smoothing；
- rank 统计不得反向改变超参数；
- 只对 E4 有支持证据的变量进行 rank 推断；
- 对未支持变量仍保存谱，但不声明 rank。

## 12.3 rank-1 null block bootstrap

对变量 \(j\)：

1. 取 full kernel 的最佳 rank-1 投影；
2. 形成 rank-1 拟合值；
3. 对拟合残差执行 circular moving-block bootstrap；
4. block length = 64；
5. 生成伪目标；
6. 用相同 grid/smoothing 重新拟合 full kernel；
7. 记录
   \[
   \eta_j^\star(1).
   \]

观察统计量：

\[
\eta_j^{obs}(1).
\]

p-value：

\[
p_j
=
\frac{
1+\#\{\eta_j^\star\ge\eta_j^{obs}\}
}{
B+1
}.
\]

active variables 内进行 BH-FDR \(q=0.10\)。

## 12.4 样本外 rank gain

在每个外层 validation block：

- full fit 后分别取 rank-1 和 rank-2 truncation；
- 只在训练内重拟合全局 intercept 和 modal scalar gains；
- 计算
  \[
  \Delta_{2|1,k}
  =
  MSE_{R1,k}-MSE_{R2,k}.
  \]

正增益折比例要求至少 0.80。

## 12.5 第二模态稳定性

对 bootstrap fit 的第二模态，用白化空间外积

\[
P_{j,2}
=
(u_{j,2}v_{j,2}^\top)
\]

计算与原 fit 的绝对 Frobenius correlation：

\[
\mathrm{stab}_{j,2}
=
\operatorname{median}_b
\frac{
|\langle P_{j,2}^{(b)},P_{j,2}\rangle_F|
}{
\|P_{j,2}^{(b)}\|_F\|P_{j,2}\|_F
}.
\]

要求：

```text
stability >= 0.70
```

## 12.6 rank-2 声明

三项同时满足：

```text
BH-adjusted p <= 0.10
positive rank2 gain fold fraction >= 0.80
second-mode stability >= 0.70
```

## 12.7 邻近配置检查

在已选配置的两个预声明邻居上重复一次谱和 validation gain，不重选配置。

至少两个邻居中结论一致。

## 12.8 通过规则

confirmation seeds：

- AR-S1/AR-S2 错误升级率 \(\le0.10\)；
- AR-S3 rank-2 检出率 \(\ge0.80\)；
- AR-S4 至少稳定拒绝 rank-1，具体 rank 只报告谱能量。

---

# 13. 实验 E6：D5-adaptive spectral 对照

## 13.1 目的

判断 D5 是否只应作为证据，还是可以作为有限自适应 Tikhonov 权重。

## 13.2 权重生成

在每个外层训练折：

1. 训练 dense simultaneous XAR，不使用 proximal；
2. 在该折的独立 validation block 提取变量贡献；
3. 标准化贡献；
4. 运行固定 FISTA 路径；
5. 对变量定义进入强度
   \[
   s_j
   =
   \max\{
   \alpha:
   g_j(\alpha\lambda_{\max})\ne0
   \}.
   \]
   从未进入则 \(s_j=0\)；
6. 权重
   \[
   \omega_j
   =
   \operatorname{clip}
   \left(
   \frac{
   \operatorname{median}_k(s_k+0.05)
   }{
   s_j+0.05
   },
   0.25,
   4.0
   \right).
   \]

权重只作用于下一内折，不得同折生成并使用。

## 13.3 比较

```text
U：uniform spectral
A：adaptive spectral
```

比较：

- validation/test RMSE；
- support recall/FPR；
- active kernel NRMSE；
- inactive kernel norm；
- rank 检出率；
- condition number。

## 13.4 升级条件

A 只有同时满足以下条件才成为最终默认：

```text
support recall 不低于 U
FPR 至少降低 20% 或 active kernel NRMSE 至少降低 10%
rank2 detection 不低于 U
validation RMSE 不恶化超过 2%
至少 4/5 development seeds 同方向
20 confirmation seeds 结论一致
```

否则最终保持 uniform spectral，D5 只作为辅助证据。

---

# 14. 实验 E7：谱结构重新并入预测模型

## 14.1 输入

使用 E4/E5 产生的：

- 所有变量核；
- 白化 SVD；
- 每变量候选 rank；
- 时滞模态；
- 幅值模态。

## 14.2 模型

### B0：AR-only

现有 `ARRAPHURank1(track="AR")`。

### B1：dense XAR no-prox

现有 `ARRAPHURank1(track="XAR")`，取消 pruning。

### B2：spectral fixed

使用 SVD 模态，固定 \(q,f\)，只拟合：

- bias；
- AR branch；
- modal scalar gains；
- horizon-specific readout。

### B3：spectral anchored refit

从 B2 开始：

- rank 固定；
- 全变量保留；
- 小学习率；
- 核锚定；
- 不使用 group-prox。

## 14.3 多视野

```text
h = 1, 5, 10, 30, 60
```

不允许未来 X。

## 14.4 指标

- RMSE、MAE、R²；
- \(\Delta_{X|AR}(h)\)；
- support/rank 稳定性；
- 模态增益漂移；
- 原窗口前向与递推实现误差；
- 每周期状态维数和运算量。

## 14.5 通过规则

B3：

1. 在每个视野的 validation/test RMSE 位于 B1 的 one-SE 范围；
2. 在至少三个视野上优于 AR-only；
3. 不改变 E5 冻结 rank；
4. kernel anchor 偏离不超过训练核范数的 20%；
5. 无变量在 refit 中被删除。

---

# 15. 实验 E8：稳定递推压缩

只在 E7 通过后运行。

每个谱时滞模态依次尝试：

```text
1. single Gamma/Erlang cascade
2. multi-exponential stable basis
3. Laguerre basis
4. generic stable state-space fit
```

选择满足

\[
\frac{\|q-\widehat q\|_2}{\|q\|_2}
\le0.05
\]

的最小状态阶数。

总部署误差要求：

\[
RMSE_{\mathrm{recursive}}
-
RMSE_{\mathrm{window}}
\le
0.01\times
\operatorname{Std}(y).
\]

状态矩阵必须满足：

\[
\rho(A)\le0.995.
\]

---

# 16. 实验顺序和停止线

```text
E0 目标分量恒等式
 └─失败：停止，修复 generator replay

E1 样条投影能力
 └─失败：使用唯一预声明 fallback；fallback 仍失败则停止

E2 正确目标上的 full/rank 容量
 └─失败：停止，不讨论 support/rank

E3 双残差化有效性
 └─失败：标记 nuisance inadequate，停止，不换网络

E4 support 证据
 └─失败：保留连续证据，不进入“可靠支持恢复”主张

E5 rank 统计
 └─失败：保留预测器，不声称 rank 自适应

E6 D5-adaptive 对照
 └─未通过：uniform 保持默认

E7 预测重合并
 └─失败：结构估计器和预测器保持双模型，不强行统一

E8 递推压缩
 └─失败：保留窗口实现，不声称 PLC 低阶实现
```

---

# 17. 运行命令

## 17.1 单实验

```bash
python tools/run_spectral_suite.py \
  --experiment E0 \
  --stage development \
  --device cuda
```

允许：

```text
--experiment {E0,E1,E2,E3,E4,E5,E6,E7,E8}
--stage {development,confirmation}
--device {cpu,cuda}
--force
```

禁止 CLI 覆盖：

- grid；
- smoothing；
- thresholds；
- seed；
- target semantics；
- bootstrap repetitions。

## 17.2 依赖顺序

```bash
python tools/run_spectral_suite.py --experiment E0 --stage development
python tools/run_spectral_suite.py --experiment E1 --stage development
python tools/run_spectral_suite.py --experiment E2 --stage development
python tools/run_spectral_suite.py --experiment E3 --stage development
python tools/run_spectral_suite.py --experiment E4 --stage development
python tools/run_spectral_suite.py --experiment E5 --stage development
python tools/run_spectral_suite.py --experiment E6 --stage development
python tools/run_spectral_suite.py --experiment E7 --stage development
```

只有 development 全部达到对应进入条件，才运行 confirmation。

---

# 18. GPU/CPU 实现

## 18.1 设计矩阵

- PyTorch/CUDA 分块构建设计；
- 立即转 CPU FP64；
- 单个 \(10000\times864\) 矩阵约 69 MB；
- 可直接保留。

## 18.2 线性求解

- CPU FP64 Cholesky；
- 设置
  `OMP_NUM_THREADS=1`；
- 多 seed 进程并发 4–8 个；
- 不使用 DDP。

## 18.3 bootstrap

- 固定 hyperparameters；
- 复用设计矩阵和 penalty；
- 每个 bootstrap 只更新 target 和右端；
- rank-1 null 可批量并发；
- 不重复构建 spline basis。

---

# 19. 输出

```text
results/spectral_v03/
├── E0/
├── E1/
├── E2/
├── E3/
├── E4/
├── E5/
├── E6/
├── E7/
├── E8/
├── DEVELOPMENT_DECISION.md
├── CONFIRMATION_DECISION.md
└── spectral_summary.csv
```

每个 job 只保存：

```text
contract.json
config.json
summary.json
metrics.csv
fit.npz
```

需要 bootstrap 时增加：

```text
bootstrap_statistics.npz
```

不保存所有中间 checkpoint。

---

# 20. 自动决策文件

`DEVELOPMENT_DECISION.md` 必须只使用预注册映射：

```text
E0_STATUS:
E1_STATUS:
E2_STATUS:
E3_STATUS:
E4_STATUS:
E5_STATUS:
E6_STATUS:
E7_STATUS:
E8_STATUS:

FULL_KERNEL_CAPACITY:
DOUBLE_RESIDUALIZATION_VALID:
SUPPORT_RECOVERY_VALID:
RANK_ADAPTATION_VALID:
ADAPTIVE_WEIGHTING_ADOPTED:
PREDICTION_RECOMBINATION_VALID:
RECURSIVE_DEPLOYMENT_VALID:
NEXT_ALLOWED_STAGE:
```

不得生成自由发挥的结论。

---

# 21. 最终打包

只在本轮完成后执行一次：

```bash
zip -r SPECTRAL_PS_AR_RAPHU_V03_RESULTS.zip \
  src/ar_raphu/spectral \
  tools/run_spectral_job.py \
  tools/run_spectral_suite.py \
  tools/summarize_spectral_suite.py \
  configs/spectral_v03.yaml \
  tests/test_spectral_*.py \
  results/spectral_v03
```

只检查：

```bash
test -f results/spectral_v03/DEVELOPMENT_DECISION.md
test -f results/spectral_v03/spectral_summary.csv
unzip -t SPECTRAL_PS_AR_RAPHU_V03_RESULTS.zip
```

不生成 SHA 和 manifest。

---

# 22. 从当前实验立即开始的第一批任务

当前不应直接重跑百万 epoch。第一批只实现并运行：

```text
Task 1：synthetic component replay + E0
Task 2：tensor spline design + projection oracle E1
Task 3：FP64 full-kernel ridge + Gram SVD
Task 4：正确外生目标上的 E2
Task 5：前向 cross-fit nuisance + 双残差单元测试
Task 6：E3 development 5 seeds
```

这六项通过后，才开始 support/rank 的 E4/E5。

预计第一批主要是凸线性代数，计算量远低于上一轮神经稀疏实验。即使使用 5 个 seeds 和全部候选，单张 4070/5080 或普通多核 CPU 都足够，不需要先租高端服务器。
