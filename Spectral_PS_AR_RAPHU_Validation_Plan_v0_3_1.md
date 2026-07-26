# Spectral Predictive-State AR-RAPHU v0.3.1
## 表示修复后的完整验证方案与重新启动合同

> **文档性质**：冻结执行合同。  
> **当前状态**：
>
> ```text
> E0_COMPONENT_IDENTITY_PASS
> E1_COMPRESSED_LAG_BASIS_UNDERSPECIFIED
> E2–E8_NOT_STARTED
> ```
>
> **本次目标**：保留 v0.3 已通过的 E0 和已失败的 E1；新增 E1R 修复表示层；E1R 通过后按 E2→E3→E4→E5→E6→E7→E8 继续。  
> **禁止**：覆盖 v0.3 结果、把 E1 改写成通过、恢复 group-prox、让 Codex 自行增加 basis、启动未被上一阶段允许的实验。  
> **精简审计**：只保留语义、时间索引、投影、双残差和求解残差测试；不生成逐文件 SHA、复杂 manifest 或 HTML。

---

# 0. 协议修订记录

创建：

```text
PROTOCOL_REVISION_V031.md
```

固定写入：

```text
Revision:
Spectral PS-AR-RAPHU v0.3 -> v0.3.1

Reason:
E0 passed.
E1 and its only preregistered fallback failed because the compressed lag basis
could not represent narrow, bimodal, and amplitude-dependent lag kernels.

Interpretation:
Representation subspace undercapacity.
Not a solver failure.
Not a full Urysohn model failure.
Not evidence against spectral rank recovery.

Frozen old outputs:
results/spectral_v03/E0/
results/spectral_v03/E1/
results/spectral_v03/DEVELOPMENT_DECISION.md
```

不得修改旧输出。

---

# 1. 新工作目录与分支

从当前已经包含 `src/ar_raphu/spectral/` 的代码仓启动：

```bash
git switch -c ps-ar-raphu-v031-representation-repair
```

若不使用 Git，复制整个当前工程目录为：

```text
PS_AR_RAPHU_SPECTRAL_V031/
```

保留：

```text
results/spectral_v03/
configs/spectral_v03.yaml
```

新增：

```text
configs/spectral_v031.yaml
results/spectral_v031/
src/ar_raphu/spectral/projection.py
tests/test_spectral_projection_repair.py
PROTOCOL_REVISION_V031.md
```

现有 `run_spectral_suite.py` 应扩展，不另起完全独立框架。

---

# 2. 科学合同

每个实验必须保存：

```json
{
  "scientific_question": "...",
  "target_semantics": "...",
  "target_contains_ar": true,
  "model_contains_ar": true,
  "target_contains_x": true,
  "model_contains_x": true,
  "truth_used_for_training": false,
  "truth_used_for_evaluation": true,
  "support_used_for_training": "all|oracle|single_oracle_variable",
  "basis_selection_uses_truth": false,
  "smoothing_selection_metric": "validation_prediction_mse",
  "rank_inputs_used_for_selection": false,
  "test_used_for_selection": false,
  "allowed_next_experiment": "..."
}
```

运行前强制验证：

1. 容量实验若模型不含 AR，目标不得含 AR；
2. E2A 单变量容量目标必须是该变量的真实外生贡献；
3. E2B 多变量容量目标必须是总真实外生贡献；
4. 正式 E3–E7 结构估计必须同时残差化 \(y\) 和 \(\Phi\)；
5. basis 在 E1R 后冻结；
6. smoothing 只由 validation prediction MSE 选择；
7. support truth、surface truth、奇异值、rank 和 test 不参与超参数选择。

---

# 3. v0.3.1 固定配置

创建 `configs/spectral_v031.yaml`：

```yaml
schema_version: 2
status: REPRESENTATION_REPAIR_AND_CORE_VALIDATION

provenance:
  reuse_e0_from: results/spectral_v03/E0
  preserve_failed_e1_from: results/spectral_v03/E1
  previous_decision: results/spectral_v03/DEVELOPMENT_DECISION.md

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
  numerical_jitter_relative: 1.0e-10

amplitude_basis:
  degree: 3
  basis_count: 16
  quantiles: [0.01, 0.99]
  evaluation_grid_points: 401

lag_representation_repair:
  compressed_candidates: [24, 28, 32, 40]
  identity_reference: 64
  prediction_grade_max_nrmse: 0.10
  structural_grade_max_nrmse: 0.05
  structural_grade_max_reference_ratio: 2.0
  frozen_structural_basis: 32
  lower_sensitivity_basis: 28
  upper_sensitivity_basis: 40
  regression_tolerance: 5.0e-6
  expected_worst_nrmse:
    AR-S1:
      24: 0.0197655724
      28: 0.0191848190
      32: 0.0191374543
      40: 0.0191279716
      64: 0.0191218783
    AR-S2:
      24: 0.0824809263
      28: 0.0385967218
      32: 0.0239039326
      40: 0.0195050475
      64: 0.0191218783
    AR-S3:
      24: 0.0813967136
      28: 0.0398569367
      32: 0.0229130697
      40: 0.0165423674
      64: 0.0161209014
    AR-S4:
      24: 0.0873408383
      28: 0.0437828786
      32: 0.0245928326
      40: 0.0187414198
      64: 0.0182692059

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
    - stronger_total_smoothing
    - fixed_configuration_order

crossfit:
  folds: 4
  initial_nuisance_prefix_targets: 2000
  purge_gap: 65
  nuisance_selection_tail_fraction: 0.20

solver:
  direct_dimension_limit: 2000
  pcg_relative_residual: 1.0e-8
  pcg_max_iterations: 2000
  block_jacobi_preconditioner: true
  warm_start_smoothing_path: true

support:
  null_quantile: 0.95
  required_positive_ablation_fold_fraction: 0.80
  synthetic_recall_gate: 0.80
  synthetic_fpr_gate: 0.10

rank:
  block_length: 64
  bootstrap_replicates_development: 100
  bootstrap_replicates_confirmation: 500
  bootstrap_alpha: 0.05
  bh_fdr: 0.10
  required_positive_gain_fold_fraction: 0.80
  mode_stability_threshold: 0.70
  main_basis: 32
  required_upper_neighbor_agreement: 40
  lower_stress_basis: 28

adaptive:
  enabled_as_comparison_only: true
  fista_lambda_ratios: [0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0]
  score_epsilon: 0.05
  weight_min: 0.25
  weight_max: 4.0
```

Codex 不得改动上述数值。

---

# 4. E0 的处理

E0 不重新运行，除非以下任一文件发生改变：

```text
src/ar_raphu/synthetic.py
src/ar_raphu/spectral/synthetic_components.py
```

在未改变时，新决策文件写：

```text
E0_STATUS: REUSED_E0_COMPONENT_IDENTITY_PASS_FROM_V03
```

同时记录旧最大误差：

```text
1.7763568394002505e-15
```

不得复制并伪装成新运行。

---

# 5. 新增高效投影模块

创建：

```text
src/ar_raphu/spectral/projection.py
```

实现：

```python
@dataclass(frozen=True)
class SurfaceProjectionResult:
    coefficients: np.ndarray
    estimate: np.ndarray
    nrmse: float

def project_tensor_surface(
    truth_surface: np.ndarray,
    lag_basis: np.ndarray,
    amplitude_basis: np.ndarray,
) -> SurfaceProjectionResult:
    ...
```

不得构建：

\[
(L_xN_u)\times(M_\tau M_x)
\]

的巨大 Kronecker design。

使用两次最小二乘：

\[
X=L^\dagger K,
\]

\[
\Theta^\top=A^\dagger X^\top,
\]

\[
\widehat K=L\Theta A^\top.
\]

代码：

```python
intermediate = np.linalg.lstsq(lag_basis, truth_surface, rcond=None)[0]
theta_t = np.linalg.lstsq(amplitude_basis, intermediate.T, rcond=None)[0]
theta = theta_t.T
estimate = lag_basis @ theta @ amplitude_basis.T
```

测试要求：在 \(8\times8\) 小型随机问题上，与显式 Kronecker least squares 的估计差：

```text
max_abs_difference <= 1e-10
```

---

# 6. E1R：Representation Repair

## 6.1 科学问题

> 在不改变真核、幅值基和评价定义的情况下，哪个压缩时滞空间首次达到结构谱级容量？

## 6.2 场景和样本

```text
AR-S1
AR-S2
AR-S3
AR-S4
```

seeds：

```text
0,1,2,3,4
```

只评价真实 active variables。

## 6.3 表示

幅值固定：

```text
M_x = 16
degree = 3
quantile range = [0.01,0.99]
grid points = 401
```

时滞：

```text
24, 28, 32, 40
identity reference 64
```

identity 64 使用：

```python
lag_basis = np.eye(64, dtype=np.float64)
```

不得把 64 个均匀三次 B 样条称为 identity。两者应分别记录：

```text
basis_type = cubic_bspline
basis_type = discrete_identity
```

## 6.4 中心化

必须与旧 E1 完全一致：

\[
K_j^{centered}(\tau,u)
=
K_j(\tau,u)
-
\frac1{n_{\mathrm{train}}}
\sum_{t\in\mathrm{train}}
K_j(\tau,x_{j,t}).
\]

幅值基也使用训练经验均值中心化。

## 6.5 输出

```text
results/spectral_v031/E1R/projection_repair.csv
results/spectral_v031/E1R/representation_certificate.json
results/spectral_v031/E1R/summary.json
```

每行：

```text
scenario
seed
variable
lag_basis_type
lag_basis_count
amplitude_basis_count
projection_surface_nrmse
identity_reference_nrmse
reference_ratio
prediction_grade_pass
structural_grade_pass
```

## 6.6 决策

对每个 \(M\) 取所有场景、seed、active variable 的最坏误差。

预测级：

```text
worst_nrmse <= 0.10
```

结构级：

```text
worst_nrmse <= 0.05
AND
for every scenario:
  scenario_worst_nrmse <= 2.0 * identity_scenario_worst_nrmse
```

选择最小结构级 \(M\)。

必须得到：

```text
selected_structural_lag_basis = 32
```

并且每个场景最坏值与配置中的 regression target 差：

```text
<= 5e-6
```

否则标记：

```text
E1R_IMPLEMENTATION_MISMATCH
```

并停止。

E1R 通过标签：

```text
E1R_REPRESENTATION_CERTIFIED_32x16
```

---

# 7. E2：正确目标上的容量验证

E2 分成 E2A 和 E2B，防止再次混淆“单核容量”和“多变量联合可辨识”。

---

## 7.1 E2A：单变量单核容量

### 科学问题

> 给定正确的单变量真实贡献目标和正确的变量身份，\(32\times16\) full kernel 是否能恢复该变量核？

### 数据

场景：

```text
AR-S1, AR-S2, AR-S3, AR-S4
```

对每个 seed、每个 active variable \(j\)，目标为：

\[
y_{j,t}^{E2A}
=
g_{j,t}^{X,\star}.
\]

模型只含该变量：

\[
\widehat y_{j,t}
=
\langle\Theta_j,\Phi_{j,t}\rangle_F+b.
\]

不含 AR，不含其他变量，不含 support penalty。

合同必须声明：

```text
target_contains_ar = false
model_contains_ar = false
support_used_for_training = single_oracle_variable
truth_used_for_training = target_component_only
```

这里使用真分量作为 capacity target 是允许的，但 smoothing 仍只由 validation contribution MSE 选择。

### 求解

维数：

\[
32\times16=512.
\]

使用 FP64 direct Cholesky。

### 指标

- train/validation/test contribution RMSE、R²；
- surface NRMSE；
- E1R projection NRMSE；
- excess surface error；
- KKT residual；
- full kernel SVD；
- rank-1/rank-2 truncation contribution MSE。

### 通过条件

至少 4/5 seeds，所有 active variables 的中位结果满足：

```text
validation contribution R2 >= 0.995
surface NRMSE <= max(0.03, 1.5 * E1R projection NRMSE)
relative KKT residual <= 1e-8
```

若 E2A 失败：

```text
E2A_ESTIMATOR_OR_DATA_EXCITATION_FAIL
```

停止，不进入 E2B/E3。

---

## 7.2 E2B：oracle-support 联合容量

### 科学问题

> 在多个真实变量联合存在时，正确 support 下是否仍可恢复总外生贡献和主要核结构？

目标：

\[
y_t^{E2B}
=
\sum_{j\in S^\star}
g_{j,t}^{X,\star}.
\]

模型：

\[
\widehat y_t
=
b+
\sum_{j\in S^\star}
\langle\Theta_j,\Phi_{j,t}\rangle_F.
\]

不含 AR，不含 inactive variables。

维数：

\[
3\times32\times16=1536.
\]

使用 FP64 direct Cholesky。

### 通过条件

至少 4/5 seeds：

```text
validation total-contribution R2 >= 0.95
median active surface NRMSE <= 0.15
relative KKT residual <= 1e-8
```

同时报告变量间残差设计条件数。若条件数大于 \(10^8\)，标记：

```text
E2B_JOINT_IDENTIFIABILITY_WEAK
```

但不把它错误解释为单核容量失败。

---

## 7.3 E2 rank 截断

对 E2A 的 full fit 直接 SVD 截断，不单独训练 rank-1/rank-2 神经模型。

### AR-S1、AR-S2

至少 4/5 seeds：

```text
rank1 validation MSE <= 1.05 * full validation MSE
spectral tail eta(1) <= 0.10
```

### AR-S3

至少 4/5 seeds：

```text
rank2 validation MSE <= 1.05 * full validation MSE
rank2 captures >= 0.80 of the reducible rank1-to-full MSE gap
```

### AR-S4

不预注册整数 rank，只报告：

```text
eta(1), eta(2), eta(3)
```

因为移动时滞峰可能需要多个谱模态。

---

# 8. E3：双残差化验证

E3 只有 E1R、E2A、E2B 全部通过后允许启动。

## 8.1 场景

```text
AR-S1
AR-S3
AR-S7
```

主视野：

```text
h = 1
```

development seeds：

```text
0–4
```

全部十个外生变量进入模型。

结构基固定：

```text
32 x 16
```

只选择平滑权重。

## 8.2 四条方法

### O：oracle AR residual

\[
r_t^O
=
y_t^{observed}
-
g_t^{AR,\star}.
\]

外生设计不残差化。只作合成 oracle benchmark。

### Y：只残差化输出

\[
\widetilde y_t
=
y_t-\widehat\mu(Z_t),
\qquad
\Phi_t\text{ 不变}.
\]

### D：双残差

\[
\widetilde y_t
=
y_t-\widehat\mu(Z_t),
\]

\[
\widetilde\Phi_t
=
\Phi_t-\widehat\pi(Z_t).
\]

这是主方法。

### J：联合凸 AR+X

同时使用 AR spline design 和 full X design，统一 ridge，不做残差化。

## 8.3 前向 cross-fitting 边界

训练目标索引按时间排序。

设训练目标总数为 \(N\)，前 2000 个作为 initial nuisance prefix。剩余目标均分为 4 个连续 evaluation blocks。

对第 \(k\) 个 evaluation block：

1. nuisance train 只用该 block 开始之前的目标；
2. 删除紧邻 evaluation block 的最后 65 个目标作为 purge；
3. nuisance hyperparameter selection 使用剩余 nuisance train 的最后 20%；
4. 选择后，在 nuisance train 全部非 purge 目标上重拟合；
5. 只预测当前 evaluation block；
6. 不使用未来 block。

## 8.4 nuisance 模型

AR spline design：

```text
lag basis = 6
amplitude basis = 8
degree = 3
```

\(\mu\)：

\[
\widehat\mu(Z)=\Psi a.
\]

\(\pi\)：

\[
\widehat\pi(Z)=\Psi M.
\]

两者分别使用 ridge。

## 8.5 nuisance 诊断

保存：

- OOF y RMSE/R²；
- \(\Phi\) residual Frobenius R²；
- residual design condition number；
- residual design 与 AR nuisance basis 的最大绝对相关；
- 每折边界和 purge。

若：

```text
max_abs_corr(residual_design, nuisance_basis) > 0.10
```

标记：

```text
E3_NUISANCE_RESIDUALIZATION_INADEQUATE
```

若：

```text
condition_number(residual_design_gram) > 1e8
```

标记：

```text
E3_CONDITIONAL_IDENTIFIABILITY_WEAK
```

这些 seed 不允许用于强结构结论。

## 8.6 主评价

由于合成 truth 可用，直接评价：

- active kernel surface NRMSE；
- inactive kernel norm；
- total external contribution RMSE；
- validation prediction RMSE；
- kernel distance to oracle method O；
- support evidence separation。

D 相对 Y 的通过条件：

```text
至少 4/5 eligible seeds:
  D active-kernel NRMSE < Y active-kernel NRMSE

median(D NRMSE / Y NRMSE) <= 0.85

D validation prediction RMSE
  <= 1.05 * O validation prediction RMSE
```

若 D 未通过，不自动改 nuisance 网络，不启动 E4。

---

# 9. E4–E8 的基修订

后续实验沿用 v0.3，但统一做以下修订。

## E4 support

主基固定 \(32\times16\)。所有变量进入。support label 只由：

```text
AR-S0 max-null kernel norm cutoff
AND
positive ablation fold fraction >= 0.80
```

产生。

不得由 basis 数量、SVD 或 D5 gate 决定资格。

## E5 rank

主基：

```text
32 x 16
```

上邻居：

```text
40 x 16
```

二者 rank 结论必须一致。

下邻居：

```text
28 x 16
```

只作压力测试。下邻居反转必须报告，但由于它未通过 reference-ratio 认证，不单独否决主结论。

identity 64 仅在 development 的 selected variables 上作参考，不参与超参数选择。

## E6 adaptive

D5 权重只作用于 Gram 白化核 ridge，且由独立内折产生。uniform 始终为主线，除非 E6 完整通过。

## E7 recombination

从 \(32\times16\) full-kernel SVD 初始化固定 rank XAR。无 group-prox，无变量删除。

## E8 deployment

对谱时滞模态做稳定状态空间逼近。Gamma/Erlang 只是首选，不是硬假设。

---

# 10. 求解器修订

## 10.1 E1R

使用两侧最小二乘，不使用巨大 Kronecker design。

## 10.2 E2A/E2B

维数不超过 1536，使用 CPU FP64 Cholesky。

KKT：

\[
\frac{
\|A\widehat\theta-b\|_2
}{
\|b\|_2+10^{-12}
}
\le10^{-8}.
\]

## 10.3 E3 以后

十变量维数：

\[
10\times32\times16=5120.
\]

默认使用 matrix-free FP64 PCG：

\[
v
\mapsto
\frac1nX^\top(Xv)+Pv.
\]

预条件器为每变量 penalty+local Gram 的块对角 Cholesky。

固定：

```text
relative residual <= 1e-8
max iterations = 2000
```

相邻 smoothing 配置 warm-start。

若 PCG 未收敛：

```text
SOLVER_NOT_CONVERGED
```

不得将结果用于科学结论，也不得自动降低容差。

---

# 11. 重新启动的准确顺序

当前只启动第一批：

```text
Step 1  保留 v0.3 E0/E1 结果
Step 2  实现 projection.py 和 E1R
Step 3  运行 E1R development
Step 4  E1R 通过后运行 E2A development
Step 5  E2A 通过后运行 E2B development
Step 6  E2B 通过后运行 E3 development
Step 7  生成 V031_CORE_DECISION.md
Step 8  暂停并返还结果
```

本轮不自动启动 E4–E8。

---

# 12. 测试

只运行：

```bash
python -m pytest \
  tests/test_ar_raphu_model.py \
  tests/test_sequence_training.py \
  tests/test_spectral_contracts.py \
  tests/test_synthetic_components.py \
  tests/test_spectral_design.py \
  tests/test_spectral_projection_repair.py \
  tests/test_double_residualization.py \
  tests/test_spectral_solver.py \
  tests/test_gram_svd.py \
  -q
```

新增测试必须包括：

1. identity lag basis 为 \(64\times64\) 单位阵；
2. 两侧投影与显式 Kronecker 投影等价；
3. E1R regression table 在容差内；
4. E2A target 等于单变量真贡献，不含 AR；
5. E2B target 等于 active X 总贡献，不含 AR；
6. E3 D 方法同时残差化 y 和 Phi；
7. target/origin index 不读取未来；
8. PCG/direct 在 200 维随机强凸问题上解相对差 \(\le10^{-7}\)。

不运行全仓库测试和旧 M7/M8 审计。

---

# 13. 运行命令

## 13.1 E1R

```bash
python tools/run_spectral_suite.py \
  --config configs/spectral_v031.yaml \
  --experiment E1R \
  --stage development \
  --device cpu
```

## 13.2 E2

```bash
python tools/run_spectral_suite.py \
  --config configs/spectral_v031.yaml \
  --experiment E2A \
  --stage development \
  --device cpu

python tools/run_spectral_suite.py \
  --config configs/spectral_v031.yaml \
  --experiment E2B \
  --stage development \
  --device cpu
```

## 13.3 E3

设计矩阵可在 GPU 分块构建，但线性求解使用 CPU FP64：

```bash
python tools/run_spectral_suite.py \
  --config configs/spectral_v031.yaml \
  --experiment E3 \
  --stage development \
  --device cuda
```

如果 CUDA 不可用，允许：

```bash
--device cpu
```

算法和阈值不变。

---

# 14. 自动决策文件

生成：

```text
results/spectral_v031/V031_CORE_DECISION.md
```

只包含：

```text
E0_STATUS:
OLD_E1_STATUS:
E1R_STATUS:
E2A_STATUS:
E2B_STATUS:
E3_STATUS:

REPRESENTATION_CERTIFIED:
SINGLE_KERNEL_CAPACITY_VALID:
JOINT_EXTERNAL_CAPACITY_VALID:
DOUBLE_RESIDUALIZATION_VALID:
NEXT_ALLOWED_STAGE:
```

映射：

```text
E1R fail -> STOP_REPRESENTATION
E2A fail -> STOP_SINGLE_KERNEL_CAPACITY
E2B fail -> STOP_JOINT_IDENTIFIABILITY
E3 fail -> STOP_NUISANCE_ORTHOGONALIZATION
all pass -> ALLOW_E4_SUPPORT_VALIDATION
```

不得自由发挥修改标签。

---

# 15. 输出

```text
results/spectral_v031/
├── E1R/
├── E2A/
├── E2B/
├── E3/
├── V031_CORE_DECISION.md
└── spectral_v031_core_summary.csv
```

每个作业只保存：

```text
contract.json
config.json
summary.json
metrics.csv
fit.npz
```

E1R 增加：

```text
projection_repair.csv
representation_certificate.json
```

E3 增加：

```text
nuisance_summary.json
```

不保存重复设计矩阵和所有中间模型。

---

# 16. 本轮打包

只打包一次：

```bash
zip -r SPECTRAL_PS_AR_RAPHU_V031_CORE_RESULTS.zip \
  PROTOCOL_REVISION_V031.md \
  configs/spectral_v031.yaml \
  src/ar_raphu/spectral \
  tools/run_spectral_job.py \
  tools/run_spectral_suite.py \
  tools/summarize_spectral_suite.py \
  tests/test_spectral_*.py \
  results/spectral_v031
```

只执行：

```bash
test -f results/spectral_v031/V031_CORE_DECISION.md
test -f results/spectral_v031/spectral_v031_core_summary.csv
unzip -t SPECTRAL_PS_AR_RAPHU_V031_CORE_RESULTS.zip
```

不生成逐文件 SHA、manifest 或 HTML。

---

# 17. 计算资源说明

E1R 和 E2 主要是小规模 CPU FP64 线性代数，GPU 空闲是正常现象。

E3 的 GPU 只负责：

- 构建设计矩阵；
- 分块样条评价；
- 批量贡献计算。

PCG/Cholesky 使用 CPU FP64。不得为了提升 GPU 利用率改成 Adam 或降低数值精度。

---

# 18. 本轮完成后的解释边界

## E1R 通过

只能说明：

> \(32\times16\) 对当前冻结合成核族具有结构级表示容量。

## E2A 通过

说明：

> 在单变量正确目标上，full kernel 估计器具有容量。

## E2B 通过

说明：

> 在 oracle support 下，多变量联合外生贡献可以被恢复到预注册水平。

## E3 通过

说明：

> 双残差化相较只残差化输出，更适合强 AR/闭环条件下的外生结构估计。

只有这四项全部通过，才允许进入 E4 的正式支持恢复验证。
