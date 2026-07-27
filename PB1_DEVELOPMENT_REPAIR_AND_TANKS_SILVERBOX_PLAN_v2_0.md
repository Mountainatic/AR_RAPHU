# OPS-UOI / Predictive-State Spectral AR-RAPHU
## PB1 Development 修复、Tanks 与 Silverbox 正式开发方案 v2.0

> **日期**：2026-07-27  
> **输入结果包**：`OPS_UOI_PB1_DEVELOPMENT_RESULTS_20260727.zip`  
> **输入报告**：`PB1_DEVELOPMENT_REPORT_20260727.md`  
> **结果源码提交**：`20469ab0eeaf00c3e65e7ebb880042a976235b2e`  
> **目标分支**：`public-benchmark-pb1`  
> **本轮性质**：development repair；禁止读取 official test  
> **下一允许状态**：
>
> ```text
> PB1_DEVELOPMENT_REPAIR_V2
> ```
>
> **本轮禁止状态**：
>
> ```text
> PB1_CONFIRMATION
> PB2
> CZ_CONFIRMATION
> ```

---

# 0. 本轮决定

本轮不再增加新的公开数据集，也不处理数据许可证元数据。PB1 仍由四套数据组成：

1. Parallel Wiener–Hammerstein，PWH；
2. Wiener–Hammerstein Process Noise，WHPN；
3. Cascaded Tanks with Overflow；
4. Silverbox。

正式路线为：

\[
\boxed{
\text{修复 PWH/WHPN 开发协议}
\rightarrow
\text{运行 Tanks/Silverbox development}
\rightarrow
\text{冻结 PB1 Protocol V2}
\rightarrow
\text{一次性 confirmation}
}
\]

其中：

- PWH/WHPN 不重新定义科学目标，只修复参数空间与数值实现；
- Tanks 采用 Champneys 等 2024 权威 baseline 论文及其配套代码协议；
- Silverbox 采用同一论文的公开参数和配套代码的 development split；
- 许可证不再构成训练、评价或协议冻结的阻塞项；
- 原始公开数据是否重新打包不属于本轮科学实验范围。

---

# 1. 当前开发结果的保留结论

## 1.1 PWH

当前 H3/shared-history development 结果显示：

\[
\Delta_{X\mid AR}(h)>0,
\qquad
h\in\{1,5,10,20\},
\]

相对 AR-only 的 MSE 降低分别为：

\[
71.49\%,\quad88.80\%,\quad88.18\%,\quad69.22\%.
\]

该结论保留为：

```text
PWH_H3_DEVELOPMENT_EVIDENCE
```

但不升级为 confirmation。

## 1.2 WHPN

WHPN 在 \(h=1,5\) 上通过 AR/XAR 依赖门槛；\(h=10,20\) 只保留点估计：

```text
WHPN_H10_AR_PENALTY_INTERVAL_FAILED
WHPN_H20_AR_KKT_FAILED
```

这两个失败目前解释为：

- \(h=10\)：非负 penalty 参数空间缺少精确零端点；
- \(h=20\)：线性系统尺度和求解精度问题。

在修复前，不能把它们解释为 WHPN 模型能力失败。

## 1.3 rank

当前 rank 只允许解释为：

\[
R_{P,\mathrm{svd}}^\star
\]

或冻结 full kernel 的预测压缩需求，不允许解释为物理结构真秩。

当前 bootstrap 仅覆盖 H1/H3 的 \(h=1\)，因此必须明确标记：

```text
BOOTSTRAP_SCOPE = H1_OR_H3_H1_ONLY
```

不能泛化到全部 horizon。

---

# 2. 修复原则

所有修复必须满足：

## 2.1 不使用 official test

```text
official_test_access_count == 0
```

必须在每个 runner 启动和结束时检查。

## 2.2 不按结果放宽门槛

不得：

- 把 KKT 门槛从 \(10^{-8}\) 放宽；
- 只为 WHPN 增加更小正 ridge；
- 删除失败 horizon；
- 看过 validation 结果后改变 history 候选；
- 事后改变 rank budget。

## 2.3 修复必须是全局规则

任何修复必须统一适用于：

- PWH；
- WHPN；
- Tanks；
- Silverbox；
- AR；
- X；
- XAR。

不能只对失败任务加例外。

## 2.4 baseline 忠实与主方法选择分开

Tanks/Silverbox 的 baseline 参数直接采用权威论文。

但 PS-AR-RAPHU 是新方法，权威 baseline 论文不存在其 spectral penalty 和 basis 参数。因此：

\[
\boxed{
\text{baseline 直接复现论文参数；}
}
\]

\[
\boxed{
\text{PS-AR-RAPHU 使用原已注册候选空间和机器化选择规则。}
}
\]

---

# 3. Repair R1：非负 penalty 加入精确零端点

## 3.1 问题

当前正数 log interval 无法表达：

\[
\lambda^\star=0.
\]

于是当最优解趋近零时，搜索会不断扩展到：

\[
10^{-8},\quad10^{-24},\quad10^{-55},
\]

最后错误触发：

```text
PENALTY_INTERVAL_NOT_CERTIFIED
```

## 3.2 新参数空间

对每个非负 penalty：

\[
\lambda_q\in
\{0\}
\cup
[\lambda_{q,\min}^{+},\lambda_{q,\max}^{+}],
\qquad
q\in\{\tau,x,0\}.
\]

其中：

- \(q=\tau\)：lag smoothness；
- \(q=x\)：amplitude smoothness；
- \(q=0\)：isotropic scientific ridge。

## 3.3 零端点规则

若选择：

\[
\widehat\lambda_q=0,
\]

输出：

```text
ZERO_PENALTY_ENDPOINT_CERTIFIED
```

并满足：

- 不触发下边界扩展；
- 不把零替换为 machine epsilon；
- numerical jitter 仍单独记录；
- 其余 penalty 可继续为正；
- 在报告中明确该方向未发现需要平滑/岭正则的开发证据。

## 3.4 正区间边界规则

只有当最优点是**最小正候选**且零明显更差时，才允许扩展正区间下界。

上边界最多扩展两次。

两次扩展后仍位于上边界：

```text
PENALTY_UPPER_INTERVAL_NOT_CERTIFIED
```

## 3.5 单元测试

新增：

```text
test_zero_penalty_candidate_is_exact
test_zero_penalty_does_not_expand_lower_boundary
test_positive_lower_boundary_can_expand
test_upper_boundary_expands_at_most_twice
test_numerical_jitter_is_not_scientific_ridge
```

---

# 4. Repair R2：FP64 solver rescue ladder

## 4.1 保持的门槛

\[
r_{\mathrm{KKT}}
=
\frac{\|A\widehat\theta-b\|_2}
{\|b\|_2+\epsilon}
\le10^{-8}.
\]

该门槛不变。

## 4.2 一级求解

对称化：

\[
A_s=\frac{A+A^\top}{2}.
\]

采用 FP64 Cholesky 或现有主求解器。

## 4.3 二级：对角均衡

定义：

\[
D_{ii}
=
\frac{1}
{\sqrt{\max(A_{s,ii},\epsilon_D)}}.
\]

求解：

\[
(D A_s D)z=Db,
\qquad
\theta=Dz.
\]

要求记录：

```text
diag_equilibration_used
condition_estimate_before
condition_estimate_after
```

## 4.4 三级：iterative refinement

在原始坐标中计算残差：

\[
r=b-A_s\theta.
\]

最多进行 5 次 FP64 refinement：

\[
A_s\Delta\theta=r,
\qquad
\theta\leftarrow\theta+\Delta\theta.
\]

停止条件：

\[
r_{\mathrm{KKT}}\le10^{-8}
\]

或连续两次改善小于预注册比例。

## 4.5 四级：SVD/minimum-norm fallback

若：

- 精确零 penalty 导致半正定；
- Cholesky 失败；
- refinement 后 KKT 仍失败；

采用 SVD 或 pivoted least squares，求最小范数解。

必须报告：

```text
effective_rank
smallest_retained_singular_value
svd_rcond
solution_is_minimum_norm
```

SVD fallback 不能静默改变 objective。

## 4.6 失败状态

全部救援后仍有：

\[
r_{\mathrm{KKT}}>10^{-8},
\]

保持：

```text
FAILED_NUMERICAL_KKT
```

不得放宽门槛。

## 4.7 单元测试

```text
test_equilibrated_solver_matches_direct_well_conditioned
test_iterative_refinement_reduces_original_coordinate_kkt
test_svd_fallback_handles_zero_penalty_semidefinite_case
test_solver_does_not_change_selected_penalties
test_kkt_is_recomputed_in_original_coordinates
```

---

# 5. Repair R3：H2 history 与 resolution 的冻结顺序

当前 H3/shared-history 只是公平性 pilot。正式模型必须增加 H2/native-history。

## 5.1 候选空间保持原配置

不得根据当前结果增删：

```text
lx_grid
ly_grid
lag_basis_candidates
amplitude_count_grid
```

四套数据都使用各自当前 schema-6 配置中的候选空间。

## 5.2 H2.1：history screening

对每个：

\[
(L_x,L_y)\in\mathcal L_x\times\mathcal L_y
\]

使用固定 anchor mother representation：

- lag basis：`discrete_identity`；
- amplitude basis：该数据集预注册的最大 \(M_x\)；
- penalty：R1 修复后的自动选择；
- dtype：FP64。

对 validation 计算主 direct forecast loss。

使用 one-standard-error rule：

\[
L_{val}(L_x,L_y)
\le
L_{val}^{\min}
+
\widehat{SE}_{\min}.
\]

在满足条件的候选中按以下复杂度键选最小：

\[
C_H=
(
L_x+L_y,\,
L_xL_y,\,
\max(L_x,L_y),\,
L_x,\,
L_y
).
\]

输出：

```text
H2_HISTORY_FROZEN
```

## 5.3 H2.2：resolution selection

冻结 \((L_x,L_y)\) 后，对原配置中的：

\[
M_\tau\times M_x
\]

候选运行。

优先顺序：

1. representation/coverage gate；
2. grouped/blocked validation loss；
3. Lepski stability；
4. one-SE 最小复杂度。

复杂度键：

\[
C_R=
(
M_\tau M_x,\,
M_\tau,\,
M_x
).
\]

## 5.4 H2.3：final penalty refit

在选定 history 与 resolution 后，重新运行一次 R1 penalty selector。

输出：

```text
H2_PENALTY_FROZEN
```

## 5.5 H3 保留

H3 使用权威 ARX/NARX 的 history，并用于公平性消融。

最终必须同时报告：

```text
H2_NATIVE_HISTORY
H3_SHARED_HISTORY
```

不能只挑更好的一条。

---

# 6. Repair R4：rank 报告增加 relative-loss inflation

主 rank 预算保持：

\[
\epsilon\in\{0.10,0.05,0.02\}.
\]

继续报告：

\[
\Delta_R^{pred}
=
\sqrt{
\frac{
\max(MSE_R-MSE_{\mathrm{full}},0)
}{
\operatorname{Var}(y)+\epsilon_0
}
}.
\]

新增次级指标：

\[
I_R^{rel}
=
\frac{
MSE_R-MSE_{\mathrm{full}}
}{
MSE_{\mathrm{full}}+\epsilon_0
}.
\]

每个 horizon 必须输出：

```text
full_mse
rank1_mse
rank2_mse
adaptive_mse
output_standardized_predictive_loss
relative_loss_inflation
selected_rank_10pct
selected_rank_5pct
selected_rank_2pct
```

这不改变原 rank 定义，只防止低总体方差归一化掩盖显著的相对 MSE 损失。

---

# 7. Repair R5：bootstrap 扩展到每个正式 horizon

## 7.1 Development

每个数据集、每个正式 horizon：

\[
B_{\mathrm{dev}}=250.
\]

## 7.2 Confirmation

\[
B_{\mathrm{conf}}=1000.
\]

## 7.3 重采样单元

### PWH

完整 phase cluster：

- 五档 amplitude 不拆；
- 双 period 不拆。

### WHPN

完整 realization。

### Tanks/Silverbox

单记录时间序列采用自动 block-length selector，不能固定 64。

## 7.4 冻结模型

bootstrap 固定：

- history；
- resolution；
- penalty；
- scaler；
- basis domain。

主 bootstrap 不在每次重采样中重新调参。

另可在 appendix 做 selection-aware bootstrap，但不阻塞 PB1。

---

# 8. Repair R6：增加 spectral free-run simulation

公开 nonlinear system identification benchmark 的主比较通常是 free-running simulation。

## 8.1 Direct 与 free-run 分开

### Direct

\[
\widehat y_{t+h\mid t}
=
F_h(X_{\le t},Y_{\le t}).
\]

### Free-run

初始化窗口后：

\[
\widehat y_t
=
F(X_{\le t},\widehat Y_{<t}),
\]

不再读取中间真实 \(y_t\)。

## 8.2 Development free-run

只在 development validation 上运行。

初始化长度：

\[
N_{\mathrm{init}}
=
\max(
L_y,\,
N_{\mathrm{official-init}}
).
\]

## 8.3 Confirmation free-run

official test 只允许：

- 使用 test input；
- 使用官方允许的前 \(N_{\mathrm{init}}\) 个真实输出初始化；
- 后续完全递归；
- 一次性计算 test RMSE。

## 8.4 输出分栏

```text
DIRECT_H1
DIRECT_H5
DIRECT_H10
DIRECT_H20
FREE_RUN_SIMULATION
```

direct 结果不能用于宣称官方 leaderboard 性能。

---

# 9. Tanks 正式开发协议

## 9.1 权威来源

采用：

- Champneys et al., 2024，*Baseline Results for Selected Nonlinear System Identification Benchmarks*；
- DOI：`10.1016/j.ifacol.2024.08.574`；
- 配套代码：`MDCHAMP/nonlinear_baselines`；
- official loader：`nonlinear_benchmarks.Cascaded_Tanks()`。

## 9.2 数据划分

按配套代码冻结：

```text
estimation[0:700]   -> train
estimation[700:end] -> validation
official test       -> locked confirmation
```

不再使用 80/20 split。

输出：

```text
TANKS_SPLIT = CHAMPNEYS_ASSOCIATED_CODE_700_REST
```

## 9.3 official test 初始化

\[
N_{\mathrm{init}}=50.
\]

confirmation free-run 的前 50 点用于状态初始化，不计正式 RMSE。

## 9.4 baseline 参数

### LTI ARX

论文参数：

\[
n_x=9,\qquad n_y=8.
\]

求解：

```text
pivoted QR / SVD
ridge = 0
```

### pNARX

- history：
  \[
  n_x=9,\quad n_y=8;
  \]
- Legendre 单变量 polynomial basis；
- 不加入 multinomial cross terms；
- order：
  \[
  p\in\{2,3,4,5,6,7\};
  \]
- validation AIC 选阶；
- QR/SVD 最小二乘。

### MLP-NARX

权威论文 profile：

```text
hidden_layers = 1
activation = tanh
width_candidates = [2, 5, 7, 10]
optimizer = Adam
iterations = 20000
initializations = 5
early_stopping = false
history = (9, 8)
```

学习率采用锁定配套实现：

\[
10^{-2}.
\]

### 可选 LTI state-space

论文使用：

\[
n_{\mathrm{state}}=2.
\]

若当前仓库没有稳定实现，不阻塞本轮；应标记：

```text
OPTIONAL_LITERATURE_BASELINE
```

## 9.5 文献 sanity reference

论文报告 free-run test RMSE：

| 模型 | RMSE / V |
|---|---:|
| LTI ARX | 0.685 |
| pNARX | 0.413 |
| GP-NARX | 0.622 |
| MLP-NARX | 2.04 |
| GRU | 0.396 |
| LSTM | 0.490 |

这些数值用于复现审计，不作为硬门槛。任何差异必须解释：

- 代码版本；
- scaler；
- initialization；
- optimizer；
- split；
- simulation recursion。

## 9.6 PS-AR-RAPHU

### H3

使用：

\[
L_x=9,\qquad L_y=8.
\]

### H2

使用 Tanks 当前预注册候选：

```text
Lx: [8, 16, 32, 64]
Ly: [1, 4, 8, 16, 32]
lag basis:
  identity
  cubic B-spline 16
  cubic B-spline 32
amplitude basis:
  [12, 16, 20, 24, 28]
```

按 R3 选择。

### 指标

- direct \(h=1,5,10,20\)；
- free-run validation；
- full/rank-1/rank-2/adaptive；
- relative-loss inflation；
- bootstrap；
- solver diagnostics。

## 9.7 overflow

overflow 阈值不再阻塞 formal development。

主结果只报告官方整体 RMSE。

若后续得到可靠阈值，再增加：

```text
normal
near_overflow
overflow
```

作为 appendix，不得事后改变主模型。

---

# 10. Silverbox 正式开发协议

## 10.1 权威来源

采用：

- Champneys et al., 2024；
- Silverbox official loader；
- Champneys 配套代码的 development split；
- official multisine/arrow tests。

许可证元数据不参与本轮 preflight。

## 10.2 数据划分

配套代码采用：

```text
cut = len(multisine_estimation) // 2
train      = estimation[:cut]
validation = estimation[cut:]
```

冻结为：

```text
SILVERBOX_SPLIT = CHAMPNEYS_ASSOCIATED_CODE_HALF_HALF
```

## 10.3 official test

三条 test 分开：

1. `multisine`；
2. `arrow_full`；
3. `arrow_no_extrapolation`。

其中：

```text
arrow_no_extrapolation
```

是 `arrow_full` 的子集，必须单独报告，不能平均。

初始化：

\[
N_{\mathrm{init}}=50.
\]

单位：

```text
mV
```

## 10.4 baseline 参数

### LTI ARX

论文参数：

\[
n_x=10,\qquad n_y=10.
\]

### pNARX

- history：
  \[
  n_x=n_y=10;
  \]
- Legendre 单变量 polynomial basis；
- order：
  \[
  p\in\{2,\ldots,7\};
  \]
- validation AIC 选阶。

### MLP-NARX

```text
hidden_layers = 1
activation = tanh
width_candidates = [2, 5, 7, 10]
optimizer = Adam
learning_rate = 1e-2
iterations = 20000
initializations = 5
early_stopping = false
history = (10, 10)
```

本轮以同行评议论文为主，不采用配套仓库中 10,000 iterations 的旧代码 profile。

## 10.5 文献 sanity reference

论文报告 free-run RMSE：

| 模型 | multisine / mV | arrow full / mV | arrow no-extrap / mV |
|---|---:|---:|---:|
| LTI ARX | 6.95 | 14.2 | 6.59 |
| pNARX | 0.640 | 2.25 | 0.571 |
| MLP-NARX | 2.80 | 5.18 | 2.40 |
| GRU | 1.49 | 2.80 | 0.947 |
| LSTM | 1.50 | 2.68 | 0.960 |

只作复现审计，不是硬 gate。

## 10.6 PS-AR-RAPHU

### H3

\[
L_x=L_y=10.
\]

### H2

使用当前预注册候选：

```text
Lx: [16, 32, 64, 128]
Ly: [1, 4, 8, 16, 32, 64]
lag basis:
  identity
  cubic B-spline 32
  cubic B-spline 48
amplitude basis:
  [16, 20, 24, 28, 32, 40]
```

按 R3 选择。

### 评价

Development：

- direct \(h=1,5,10,20\)；
- validation free-run；
- full/rank-1/rank-2/adaptive；
- multisine estimation 后半段 validation。

Confirmation：

- multisine test；
- arrow full；
- arrow no-extrap；
- official initialization 50；
- direct 与 free-run 分栏。

---

# 11. 四数据集统一实验矩阵

| 数据 | H3 文献/shared history | H2 native | Direct | Free-run | Bootstrap |
|---|---|---:|---:|---:|---:|
| PWH | 已有 H1/H3 | 必做 | 1/5/10/20 | 必做 | 各 horizon 250 |
| WHPN | 已有 H1/H3 | 必做 | 1/5/10/20 | 必做 | 各 horizon 250 |
| Tanks | 9/8 | 必做 | 1/5/10/20 | 必做 | 各 horizon 250 |
| Silverbox | 10/10 | 必做 | 1/5/10/20 | 必做 | 各 horizon 250 |

每套数据运行：

```text
Persistence
AR
X
ARX/XAR
pNARX
MLP-NARX
rank-1 AR-RAPHU
fixed rank-2 AR-RAPHU
full spectral AR-RAPHU
adaptive spectral AR-RAPHU
```

---

# 12. 开发执行顺序

## Stage A：实现修复

1. exact zero penalty；
2. solver rescue ladder；
3. H2 selector；
4. relative-loss inflation；
5. horizon-aware bootstrap；
6. spectral free-run。

## Stage B：回归测试

必须保持：

```text
ALL_EXISTING_TESTS_PASS
V20_TESTS_PASS
LEGACY_V034_REGRESSION_PASS
```

## Stage C：重跑 PWH/WHPN development

只重跑受修复影响的任务：

- 所有 H2；
- WHPN AR \(h=10,20\)；
- 所有 final resolution/penalty；
- all-horizon bootstrap；
- free-run。

保留旧 H3 结果作为历史对照，不覆盖。

## Stage D：Tanks smoke

```text
ARX paper profile
pNARX paper profile
Spectral H3 h=1
free-run short smoke
```

通过后启动完整 development。

## Stage E：Silverbox smoke

同样先：

```text
ARX paper profile
pNARX paper profile
Spectral H3 h=1
validation free-run
```

通过后启动完整 development。

## Stage F：统一报告

输出：

```text
PB1_DEVELOPMENT_REPAIR_V2_REPORT.md
PB1_DEVELOPMENT_REPAIR_V2_STATUS.json
```

## Stage G：是否允许 freeze

满足全部 gate 后，生成：

```text
PB1_PROTOCOL_FREEZE_V2.json
```

official test 仍保持 0 次访问。

---

# 13. 机器 gate

## 13.1 实现 gate

```text
ZERO_PENALTY_IMPLEMENTED
SOLVER_RESCUE_LADDER_IMPLEMENTED
H2_SELECTION_IMPLEMENTED
RELATIVE_LOSS_INFLATION_IMPLEMENTED
ALL_HORIZON_BOOTSTRAP_IMPLEMENTED
SPECTRAL_FREE_RUN_IMPLEMENTED
```

## 13.2 数据 gate

```text
PWH_SPLIT_UNCHANGED
WHPN_SPLIT_UNCHANGED
TANKS_SPLIT_700_REST
SILVERBOX_SPLIT_HALF_HALF
OFFICIAL_TEST_ACCESS_COUNT_ZERO
```

## 13.3 模型 gate

```text
PWH_H2_COMPLETE
WHPN_H2_COMPLETE
TANKS_H2_COMPLETE
SILVERBOX_H2_COMPLETE
ALL_PRIMARY_KKT_PASS
ALL_PENALTY_INTERVALS_CERTIFIED
```

允许预注册 partial 状态，但必须在 freeze 前明确：

```text
DATASET_HORIZON_PARTIAL_ALLOWED
```

不得静默忽略。

## 13.4 baseline gate

```text
TANKS_ARX_9_8
TANKS_PNARX_ORDER_2_7
TANKS_MLPNARX_PAPER_PROFILE
SILVERBOX_ARX_10_10
SILVERBOX_PNARX_ORDER_2_7
SILVERBOX_MLPNARX_PAPER_PROFILE
```

## 13.5 打包 gate

```text
PACKAGE_SHA256_VALID
ZIP_CRC_VALID
SOURCE_COMMIT_RECORDED
REPOSITORY_BUNDLE_INCLUDED
SELF_CONTAINED_PB1_TESTS_PASS
```

---

# 14. 失败与停止规则

## F1：零端点仍产生 interval failure

```text
STOP_PENALTY_IMPLEMENTATION
```

## F2：solver rescue 后 KKT 仍失败

保留失败结果：

```text
FAILED_NUMERICAL_KKT
```

不得放宽门槛。

## F3：H2 最优在 history 最大边界

不自动扩 history。

输出：

```text
HISTORY_RANGE_EDGE_SELECTED
```

并在 confirmation freeze 前决定：

- 接受候选范围内最优；
- 或整体重新预注册更长历史并重跑四套数据。

不得只扩单个数据集。

## F4：basis 最优位于最大分辨率

同理：

```text
RESOLUTION_RANGE_EDGE_SELECTED
```

不得事后只扩失败数据。

## F5：Tanks/Silverbox baseline 无法复现文献量级

不停止主方法开发，但输出：

```text
LITERATURE_REPLICATION_AUDIT_FAILED
```

先查：

- split；
- recursion；
- initialization；
- scaling；
- polynomial basis；
- units。

## F6：Spectral direct 好但 free-run 失稳

不能进入官方 benchmark 优胜声明。

输出：

```text
DIRECT_ONLY_MODEL
```

并保留为软测量/direct prediction 结果。

---

# 15. 配置变更要求

## Tanks

将旧状态：

```text
BLOCKED_PENDING_USER_FREEZE
PENDING_SPLIT_ADEQUACY_AUDIT
```

替换为：

```text
FROZEN_LITERATURE_SPLIT
```

配置：

```yaml
development_split:
  source: Champneys2024_associated_code
  train_rows: [0, 700]
  validation_rows: [700, end]

literature_baselines:
  arx:
    nx: 9
    ny: 8
    solver: qr_svd
  pnarx:
    nx: 9
    ny: 8
    polynomial_orders: [2, 3, 4, 5, 6, 7]
    basis: legendre_univariate
    selector: validation_aic
  mlp_narx:
    nx: 9
    ny: 8
    hidden_layers: 1
    widths: [2, 5, 7, 10]
    activation: tanh
    learning_rate: 1.0e-2
    iterations: 20000
    initializations: 5
    early_stopping: false
```

## Silverbox

删除 formal-experiment blocker：

```text
BLOCKED_BY_MISSING_METADATA
```

替换为：

```text
FROZEN_LITERATURE_SPLIT
```

配置：

```yaml
development_split:
  source: Champneys2024_associated_code
  train_fraction: [0.0, 0.5]
  validation_fraction: [0.5, 1.0]

literature_baselines:
  arx:
    nx: 10
    ny: 10
    solver: qr_svd
  pnarx:
    nx: 10
    ny: 10
    polynomial_orders: [2, 3, 4, 5, 6, 7]
    basis: legendre_univariate
    selector: validation_aic
  mlp_narx:
    nx: 10
    ny: 10
    hidden_layers: 1
    widths: [2, 5, 7, 10]
    activation: tanh
    learning_rate: 1.0e-2
    iterations: 20000
    initializations: 5
    early_stopping: false
```

---

# 16. 推荐命令

## 16.1 preflight

```bash
python tools/preflight_pb1_repair_v2.py \
  --config-dir configs/public_benchmarks \
  --require-test-access-zero
```

## 16.2 tests

```bash
python -m pytest -q
python -m pytest -q tests/test_pb1_repair_v2.py
```

## 16.3 PWH/WHPN repair

```bash
python tools/run_pb1_repair_suite.py \
  --datasets pwh whpn \
  --stage development-repair \
  --device cpu \
  --dtype float64
```

## 16.4 Tanks

```bash
python tools/run_pb1_repair_suite.py \
  --datasets cascaded_tanks \
  --stage smoke \
  --profile literature

python tools/run_pb1_repair_suite.py \
  --datasets cascaded_tanks \
  --stage development \
  --profile literature-and-spectral
```

## 16.5 Silverbox

```bash
python tools/run_pb1_repair_suite.py \
  --datasets silverbox \
  --stage smoke \
  --profile literature

python tools/run_pb1_repair_suite.py \
  --datasets silverbox \
  --stage development \
  --profile literature-and-spectral
```

## 16.6 report

```bash
python tools/summarize_pb1_repair_v2.py \
  --results-root results/public_benchmarks/pb1_repair_v2 \
  --output PB1_DEVELOPMENT_REPAIR_V2_REPORT.md
```

---

# 17. 返回结果包

名称：

```text
OPS_UOI_PB1_DEVELOPMENT_REPAIR_V2_RESULTS_bundle.zip
```

必须包含：

```text
README.md
CHANGELOG.md
PB1_DEVELOPMENT_REPAIR_V2_REPORT.md
PB1_DEVELOPMENT_REPAIR_V2_STATUS.json
PB1_REPAIR_PREFLIGHT_V2.yaml
configs/
src/
tools/
tests/
results/
tables/
figures/
logs/
environment/
repository.bundle
SOURCE_COMMIT.txt
PACKAGE_MANIFEST.json
SHA256SUMS.txt
```

不得只返还汇总表。

每个模型必须包含：

```text
config_resolved.yaml
data_lineage.json
split_manifest.csv
metrics_direct.json
metrics_free_run.json
solver_diagnostics.json
penalty_search.json
history_selection.json
resolution_selection.json
rank_profile.csv
rank_bootstrap.json
decision.json
```

---

# 18. 打包命令

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
NAME="OPS_UOI_PB1_DEVELOPMENT_REPAIR_V2_RESULTS_bundle"
OUT="$ROOT/return/$NAME"
ZIP="$ROOT/return/$NAME.zip"

rm -rf "$OUT" "$ZIP" "$ZIP.sha256"
mkdir -p "$OUT"

cp -a README.md CHANGELOG.md "$OUT/" 2>/dev/null || true
cp -a src tools tests configs "$OUT/"
cp -a results/public_benchmarks/pb1_repair_v2 "$OUT/results"
cp -a PB1_DEVELOPMENT_REPAIR_V2_REPORT.md "$OUT/"
cp -a PB1_DEVELOPMENT_REPAIR_V2_STATUS.json "$OUT/"
cp -a PB1_REPAIR_PREFLIGHT_V2.yaml "$OUT/"
cp -a environment "$OUT/" 2>/dev/null || true

git rev-parse HEAD > "$OUT/SOURCE_COMMIT.txt"
git bundle create "$OUT/repository.bundle" --all

find "$OUT" -type d \
  \( -name '__pycache__' -o -name '.pytest_cache' \) \
  -prune -exec rm -rf {} +

find "$OUT" -type f \
  \( -name '*.pyc' -o -name '*.tmp' \) \
  -delete

python tools/build_manifest.py \
  --root "$OUT" \
  --output "$OUT/PACKAGE_MANIFEST.json"

(
  cd "$OUT"
  find . -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS.txt
)

python - <<'PY'
from pathlib import Path
import json

root = Path("return/OPS_UOI_PB1_DEVELOPMENT_REPAIR_V2_RESULTS_bundle")
required = [
    "src",
    "tools",
    "tests",
    "configs",
    "results",
    "PB1_DEVELOPMENT_REPAIR_V2_REPORT.md",
    "PB1_DEVELOPMENT_REPAIR_V2_STATUS.json",
    "PB1_REPAIR_PREFLIGHT_V2.yaml",
    "SOURCE_COMMIT.txt",
    "repository.bundle",
    "PACKAGE_MANIFEST.json",
    "SHA256SUMS.txt",
]
missing = [x for x in required if not (root / x).exists()]
if missing:
    raise SystemExit(f"Missing required package entries: {missing}")

json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
print("PACKAGE_STRUCTURE_OK")
PY

(
  cd "$ROOT/return"
  zip -qr "$NAME.zip" "$NAME"
  unzip -t "$NAME.zip"
  sha256sum "$NAME.zip" > "$NAME.zip.sha256"
)

echo "FINAL_PACKAGE=$ZIP"
echo "FINAL_SHA256=$ZIP.sha256"
```

---

# 19. PB1 Protocol Freeze V2 的准入条件

只有当以下全部完成：

```text
PWH_DEVELOPMENT_REPAIR_PASS
WHPN_DEVELOPMENT_REPAIR_PASS_OR_EXPLICIT_PARTIAL_FREEZE
TANKS_DEVELOPMENT_PASS
SILVERBOX_DEVELOPMENT_PASS
H2_HISTORY_FROZEN_ALL_DATASETS
RESOLUTION_FROZEN_ALL_DATASETS
PENALTY_INTERVAL_CERTIFIED_ALL_PRIMARY_MODELS
KKT_PASS_ALL_PRIMARY_MODELS
SPECTRAL_FREE_RUN_AVAILABLE_ALL_DATASETS
ALL_HORIZON_BOOTSTRAP_AVAILABLE
OFFICIAL_TEST_ACCESS_COUNT_ZERO
SELF_CONTAINED_PACKAGE_PASS
```

才生成：

```text
PB1_PROTOCOL_FREEZE_V2.json
```

随后才能一次性运行 official test confirmation。

---

# 20. 本轮结束后的科学状态

完成本方案后，PB1 能回答：

1. Spectral XAR 在四套公开物理系统上的 direct prediction 表现；
2. 它是否能在 free-run simulation 中保持稳定；
3. full kernel 相对 rank-1/rank-2 的真实预测价值；
4. adaptive rank 是否在全部 horizon 稳定；
5. 过程噪声、硬非线性和外推对模型的影响；
6. PS-AR-RAPHU 的 native history 是否优于 shared literature history；
7. 现有 PWH/WHPN 失败究竟属于 penalty/solver 还是模型本身。

仍然不能回答：

- closed-loop quotient/K 层；
- 多变量 Schur 可辨识；
- 因果 plant kernel；
- CZ 跨晶棒泛化。

这些属于 PB2、PB4 和 CZ 阶段。

---

# 21. 参考协议

1. Champneys, M. D., Beintema, G. I., Tóth, R., Schoukens, M., Rogers, T. J.  
   *Baseline Results for Selected Nonlinear System Identification Benchmarks*.  
   IFAC-PapersOnLine 58(15), 2024, 474–479.  
   DOI: `10.1016/j.ifacol.2024.08.574`.

2. Associated implementation:  
   `MDCHAMP/nonlinear_baselines`.

3. Official benchmark loader:  
   `GerbenBeintema/nonlinear_benchmarks`.

权威论文明确采用：

- free-run simulation；
- train-only normalization；
- ARX lag 最大 20 并用 validation AIC 选择；
- Tanks ARX/NARX history \(9/8\)；
- Silverbox history \(10/10\)；
- pNARX Legendre 单变量多项式，最高 7 阶；
- MLP-NARX 单隐藏层 tanh，宽度 2/5/7/10，20,000 次 Adam，5 次初始化。

