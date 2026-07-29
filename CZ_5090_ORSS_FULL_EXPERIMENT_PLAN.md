# CZ 真实数据完整实验推进方案
## 基于 RTX 5090 的 Operator-Reduced Spectral Solver（ORSS）重构与全实验闭环

> 项目：OPS-UOI / Spectral PS-AR-RAPHU  
> 目标设备：RTX 5090 32GB × 1  
> 目标：不再使用 A800。在不改变科学模型、数据划分和统计协议的前提下，把当前稠密正规方程求解器重构为矩阵自由、可预条件、可降阶、可混合精度认证的算子求解器，并一直推进到 CZ 真实数据实验全部完成。  
> 主数据：
>
> ```text
> Furnace A: 实验数据1(3).xlsx / Sheet1
> Furnace B: 实验数据1-张.xlsx / Sheet2
> ```
>
> 主输入：
>
> ```text
> 主加热功率
> 晶升速度
> 晶转速度
> 埚升速度
> 埚转速度
> ```
>
> 目标：晶体直径 / mm  
> 主实验范围：等径阶段、固定阶段、闭环弱激励条件下的可解释非线性系统辨识与跨炉软测量。

---

# 0. 冻结原则

本方案不改变：

1. OPS-UOI 的 operator-first 识别对象；
2. PS-AR-RAPHU 的 X / AR / XAR 结构；
3. full-first-then-compress；
4. H2 原生 history 为主结果；
5. H3 shared-history 仅作公平消融；
6. exact-zero penalty endpoint；
7. 原坐标 KKT 门槛：
   \[
   r_{\mathrm{KKT}}\le10^{-8};
   \]
8. v4.1 三域关系：
   \[
   \mathcal S^{\mathrm{cert}}
   \subseteq
   \mathcal S^{\mathrm{train}}
   \subseteq
   \mathcal D^{\mathrm{model}};
   \]
9. bounded \(C^1\) continuation；
10. 第一炉用于 development 与 internal confirmation；
11. 第二炉在模型冻结前保持不可访问；
12. direct prediction 与 free-run simulation 分开；
13. 域外 continuation 不升级为 K 层解释；
14. 所有主指标以 mm 报告；
15. 所有参数选择只使用 development 数据。

本轮只改变：

\[
\boxed{\text{求解器与计算实现}}
\]

即把：

\[
\text{显式 Gram}
+
\text{重复稠密 Cholesky}
\]

改造成：

\[
\boxed{
\text{矩阵自由增广算子}
+
\text{谱坐标}
+
\text{Kronecker/Schur 预条件}
+
\text{参数化 reduced basis}
+
\text{混合精度 refinement}
}
\]

dense solver 保留为：

- 小规模参考解；
- 最终 KKT 复核；
- 数值回归；
- 救援路径。

---

# 1. 当前复杂度

当前离散问题：

\[
\min_\theta
\frac1n\|X\theta-y\|_2^2
+
\lambda_\tau \theta^\top P_\tau\theta
+
\lambda_x \theta^\top P_x\theta
+
\lambda_0\|\theta\|_2^2.
\]

正规方程：

\[
H_\lambda\theta=b,
\]

\[
H_\lambda
=
G+
\lambda_\tau P_\tau+
\lambda_xP_x+
\lambda_0I,
\]

\[
G=\frac{X^\top X}{n},
\qquad
b=\frac{X^\top y}{n}.
\]

若参数维度为 \(p\)，penalty candidate 数为 \(C\)，当前近似复杂度为：

\[
O(np^2)+O(Cp^3).
\]

而且：

\[
\kappa(X^\top X)=\kappa(X)^2.
\]

这导致：

- 高分辨率下 Cholesky 成本高；
- penalty 网格反复分解；
- effective df 更昂贵；
- FP64 需求被放大；
- RTX 5090 的 FP32/TF32 优势没有发挥。

目标是把主要代价改写为：

\[
O(mkT_{\mathcal A})
+
O(Cr^3)
+
O(skT_{\mathcal A}),
\]

其中：

- \(m\)：完整 anchor solve 数；
- \(k\)：Krylov 迭代数；
- \(T_{\mathcal A}\)：一次算子作用成本；
- \(r\)：reduced basis 维数；
- \(s\)：最终 full-space 复核候选数；
- \(m,s\ll C\)；
- \(r\ll p\)。

---

# 2. 函数空间重构

对第 \(j\) 个输入：

\[
(\mathcal A_jK_j)_t
=
\sum_{\tau=0}^{L_x-1}
K_j(\tau,x_{j,t-\tau}).
\]

AR 分支：

\[
(\mathcal A_y f_y)_t
=
\sum_{\ell=1}^{L_y}
q_\ell^y f_y(y_{t-\ell}).
\]

总体：

\[
\mathcal A
=
[
\mathcal A_1,\ldots,\mathcal A_p,\mathcal A_y
].
\]

核空间：

\[
\mathcal H
=
\bigoplus_{j=1}^p
\left(
\mathbb R^{L_x}
\otimes
L^2(\mathcal I_j,\nu_j)
\right)
\oplus
\mathcal H_y.
\]

目标：

\[
\min_{K\in\mathcal H}
\|\mathcal AK-y\|_{\mathcal Y}^2
+
\lambda_\tau
\|\mathcal L_\tau^{1/2}K\|_{\mathcal H}^2
+
\lambda_x
\|\mathcal L_x^{1/2}K\|_{\mathcal H}^2
+
\lambda_0\|K\|_{\mathcal H}^2.
\]

Euler 方程：

\[
\left(
\mathcal A^*\mathcal A
+
\lambda_\tau\mathcal L_\tau
+
\lambda_x\mathcal L_x
+
\lambda_0I
\right)K
=
\mathcal A^*y.
\]

定义参数化算子族：

\[
\mathcal H_\lambda
=
\mathcal H_0
+
\lambda_\tau\mathcal L_\tau
+
\lambda_x\mathcal L_x
+
\lambda_0I.
\]

penalty 搜索因此不是 512 个无关问题，而是同一个仿射算子族在不同 \(\lambda\) 上的解。

---

# 3. ORSS 总体架构

ORSS：

```text
Operator-Reduced Spectral Solver
```

完整流程：

```text
数据窗口
→ 矩阵自由 A / A*
→ 增广最小二乘
→ Demmler–Reinsch 谱坐标
→ Kronecker / Schur 预条件器
→ 参数化 reduced basis
→ mixed precision refinement
→ FP64 residual / KKT certification
```

建议代码结构：

```text
src/ar_raphu/orss/
  __init__.py
  operator.py
  adjoint.py
  augmented.py
  dr_basis.py
  penalties.py
  preconditioner.py
  krylov.py
  reduced_basis.py
  residual_estimator.py
  mixed_precision.py
  zero_endpoint.py
  effective_df.py
  diagnostics.py
  checkpoint.py
  cuda_backend.py
  cpu_reference.py
```

工具：

```text
tools/
  run_orss_smoke.py
  run_orss_equivalence.py
  run_orss_profile.py
  run_cz_r2_2_orss.py
  run_cz_r2_1_audit.py
  run_cz_r3.py
  run_cz_r4_baselines.py
  run_cz_r5_confirmation.py
  run_cz_r6_outer.py
  run_cz_r7_calibration.py
  run_cz_interpretability.py
  run_cz_bootstrap.py
  summarize_cz_complete.py
  build_manifest.py
  package_cz_complete.sh
```

---

# 4. 矩阵自由 \(\mathcal A\) 与 \(\mathcal A^*\)

核展开：

\[
K_j(\tau,u)
=
\sum_{a=1}^{M_\tau}
\sum_{b=1}^{M_x}
\theta_{j,ab}\phi_a(\tau)\psi_b(u).
\]

于是：

\[
(\mathcal A_jK_j)_t
=
\sum_{\tau,a,b}
\theta_{j,ab}
\phi_a(\tau)
\psi_b(x_{j,t-\tau}).
\]

定义：

\[
Z_{j,t,ab}
=
\sum_\tau
\phi_a(\tau)
\psi_b(x_{j,t-\tau}),
\]

则：

\[
(\mathcal A_jK_j)_t
=
\langle Z_{j,t},\Theta_j\rangle_F.
\]

GPU 主路径只实现：

```python
y_hat = A(theta)
grad = AT(residual)
```

不构造完整 design 或 Gram 作为主求解路径。

B-spline 局部支撑下，若每个幅值点激活 \(s_x\) 个 basis，每个 lag 点激活 \(s_\tau\) 个 basis，一次算子作用更接近：

\[
O(nL_xs_\tau s_x).
\]

必须验证 adjoint identity：

\[
\langle \mathcal A\theta,r\rangle_{\mathcal Y}
=
\langle \theta,\mathcal A^*r\rangle_{\mathcal H}.
\]

FP64 参考门槛：

\[
\frac{
|\langle \mathcal A\theta,r\rangle
-
\langle \theta,\mathcal A^*r\rangle|
}{
|\langle \mathcal A\theta,r\rangle|+
|\langle \theta,\mathcal A^*r\rangle|+
\epsilon
}
\le10^{-10}.
\]

---

# 5. 增广最小二乘

定义：

\[
\mathscr B_\lambda K
=
\begin{bmatrix}
\mathcal AK\\
\sqrt{\lambda_\tau}\mathcal L_\tau^{1/2}K\\
\sqrt{\lambda_x}\mathcal L_x^{1/2}K\\
\sqrt{\lambda_0}K
\end{bmatrix}.
\]

原问题等价为：

\[
\min_K
\left\|
\mathscr B_\lambda K
-
\begin{bmatrix}
y\\0\\0\\0
\end{bmatrix}
\right\|^2.
\]

首选：

```text
LSMR
```

备选：

```text
LSQR
augmented MINRES
```

目的：

- 不显式形成 \(A^*A\)；
- 条件数不平方；
- 支持矩阵自由；
- 支持 warm-start；
- 支持 mixed precision refinement。

---

# 6. Demmler–Reinsch 谱坐标

解：

\[
P_\tau v_r
=
\mu_r M_\tau v_r,
\]

\[
P_x w_s
=
\nu_s M_x w_s.
\]

使用张量基：

\[
e_{rs}=v_r\otimes w_s.
\]

正则项变为：

\[
d_{rs}(\lambda)
=
\lambda_\tau\mu_r
+
\lambda_x\nu_s
+
\lambda_0.
\]

效果：

1. 正则算子变成逐元素缩放；
2. 谱坐标天然白化；
3. 降低 Krylov 迭代数；
4. penalty 变化只更新对角项；
5. reduced operator 更容易投影；
6. 更适合混合精度。

实现要求：

```text
DR transform: FP64
operator application: FP32/TF32
projected matrices: FP64
```

---

# 7. Kronecker / Schur 预条件器

真实 Gram 不严格 Kronecker，因此禁止把近似直接当最终系统。

构造：

\[
\mathcal P_\lambda
=
\widetilde{\mathcal G}
+
\lambda_\tau\mathcal L_\tau
+
\lambda_x\mathcal L_x
+
\lambda_0I.
\]

层级：

```text
P0 diagonal
P1 channel block
P2 lag-amplitude Kronecker
P3 Kronecker + AR/X Schur
```

主候选：

```text
P2 + AR/X block Schur
```

预条件器只改变迭代速度，不改变最终目标。

---

# 8. 参数化 reduced basis

penalty 参数：

\[
\lambda
=
(\lambda_\tau,\lambda_x,\lambda_0).
\]

anchor：

```text
exact zero
center
three axis extremes
box corners
previously selected neighbors
```

对 anchor 做 full-space ORSS 求解：

\[
K^{(i)}=K(\lambda^{(i)}).
\]

构造：

\[
V_r
=
\operatorname{orth}
\left[
K^{(1)},\ldots,K^{(m)}
\right].
\]

近似：

\[
K(\lambda)\approx V_rc(\lambda).
\]

投影：

\[
H_r(\lambda)c(\lambda)=b_r,
\]

\[
H_r(\lambda)
=
H_{r,0}
+
\lambda_\tau L_{r,\tau}
+
\lambda_xL_{r,x}
+
\lambda_0I_r.
\]

全部 penalty candidate 在 reduced space 中批量求解。

## 8.1 Greedy enrichment

残差：

\[
r_\lambda
=
b-\mathcal H_\lambda V_rc(\lambda).
\]

估计：

\[
\eta_r(\lambda)
=
\|r_\lambda\|_{\mathcal P_\lambda^{-1}}.
\]

循环：

```text
scan all candidates in reduced space
→ find largest residual candidate
→ full ORSS solve
→ add basis vector
→ orthogonalize
→ repeat
```

development：

```text
epsilon_RB = 1e-5
```

confirmation：

```text
epsilon_RB = 1e-7
```

建议：

```text
r_initial = 8
r_max = 128
```

若超过 128 仍不通过：

```text
REDUCED_BASIS_NOT_EFFECTIVE
```

该 task 回退 full ORSS，不改变科学结果。

---

# 9. Exact-zero endpoint

当三个 penalty 全为零时，系统可能半正定。

路径：

```text
ZERO_ENDPOINT_LSMR_MINIMUM_NORM
```

要求：

\[
\|\mathcal A^*(\mathcal AK-y)\|
\le10^{-8}.
\]

零点解：

- 作为 reduced basis anchor；
- 不使用 SPD Cholesky；
- 不使用依赖一致 coercivity 的误差界；
- 用 normal residual 和 minimum-norm 条件认证。

---

# 10. Mixed precision

RTX 5090 使用：

\[
\boxed{
\text{低精度大算子}
+
\text{FP64 小矩阵和最终认证}
}
\]

FP32/TF32：

- basis evaluation；
- \(\mathcal A\)、\(\mathcal A^*\)；
- Krylov 大向量运算；
- reduced candidate screening；
- validation prediction；
- free-run；
- rank 初筛。

FP64：

- DR 特征分解；
- reduced basis 正交化；
- projected matrices；
- residual estimator；
- final refinement；
- exact-zero residual；
- KKT；
- final metrics；
- bootstrap 汇总。

Iterative refinement：

\[
r_k=b-\mathcal H_\lambda K_k,
\]

\[
\mathcal H_\lambda\Delta K_k=r_k,
\]

\[
K_{k+1}=K_k+\Delta K_k.
\]

停止：

\[
r_{\mathrm{KKT}}\le10^{-8}.
\]

5 次 refinement 仍失败：

```text
MIXED_PRECISION_REFINEMENT_FAILED
→ FULL_FP64_FINAL_SOLVE
```

只对最终候选回退。

---

# 11. Effective degrees of freedom

不再对所有 candidate 计算：

\[
df(\lambda)
=
\operatorname{tr}
\left[
H_\lambda^{-1}G
\right].
\]

流程：

```text
all candidates:
  validation loss
  residual certificate
  KKT screening

then:
  one-SE eligible set

only eligible set:
  effective df
```

先用 reduced-space：

\[
df_r(\lambda)
=
\operatorname{tr}
\left[
H_r(\lambda)^{-1}G_r
\right].
\]

最终需要时使用 Hutchinson：

```text
development probes = 16
confirmation probes = 64
```

---

# 12. RTX 5090 配置

`configs/cz_real_data/orss_5090.yaml`：

```yaml
runtime:
  backend: torch
  device: cuda:0
  primary_dtype: float32
  certification_dtype: float64
  real_cuda_dispatch_required: true
  silent_cpu_fallback_forbidden: true

cuda:
  allow_tf32_operator: true
  allow_tf32_projected: false
  usable_vram_fraction: 0.78
  one_active_task: true
  operator_chunk_time: 2048
  operator_chunk_lag: 64
  reduced_candidate_batch: 512
  cache_basis_on_device: true
  cache_windows_on_device: true
  cache_dr_transform_on_device: true

krylov:
  method: lsmr
  max_iterations: 1000
  relative_tolerance_development: 1.0e-5
  relative_tolerance_confirmation: 1.0e-8
  warm_start: true

preconditioner:
  type: kronecker_schur
  rebuild_per_resolution: true
  update_diagonal_per_penalty: true

reduced_basis:
  enabled: true
  initial_anchors: 8
  max_dimension: 128
  residual_tolerance_development: 1.0e-5
  residual_tolerance_confirmation: 1.0e-7
  greedy_enrichment: true
  full_solve_fallback: true

mixed_precision:
  enabled: true
  operator_dtype: float32
  residual_dtype: float64
  projected_dtype: float64
  maximum_refinement_steps: 5
  final_kkt_threshold: 1.0e-8

checkpoint:
  every_task: true
  every_anchor: true
  every_greedy_iteration: true
  resume: true
  verify_config_hash: true
  verify_source_commit: true
```

---

# 13. 全流程阶段

```text
S0  冻结 dense reference
S1  矩阵自由 A / A*
S2  增广 LSMR
S3  Demmler–Reinsch
S4  Kronecker / Schur 预条件器
S5  参数化 reduced basis
S6  mixed precision refinement
S7  dense / ORSS 等价
S8  R2.1 correctness audit
S9  R3-A native history
S10 R3-B resolution / penalty
S11 R3-C continuation
S12 R3-D rank
S13 完整 baselines
S14 Furnace A internal confirmation
S15 Furnace B zero-shot
S16 Furnace B 5% / 10% calibration
S17 解释性审计
S18 bootstrap
S19 最终报告与打包
```

---

# 14. S0：冻结 dense reference

```bash
git status
git add -A
git commit -m "Freeze dense spectral solver before ORSS"

mkdir -p results/orss_reference

git rev-parse HEAD \
  > results/orss_reference/DENSE_REFERENCE_COMMIT.txt

python -m pytest -q \
  | tee results/orss_reference/pytest_dense_reference.txt
```

reference tasks：

```text
T1: Lx=32,  Ly=8,   Mtau=16, Mx=16, h=1
T2: Lx=128, Ly=32,  Mtau=16, Mx=32, h=15
T3: Lx=256, Ly=64,  Mtau=32, Mx=28, h=30
```

保存：

```text
design hash
Gram hash
selected penalty
theta
validation prediction
validation loss
KKT
effective df
```

大任务：

```text
Lx=512, Ly=128, Mtau=64, Mx=32
```

只做 profiler，不要求 dense 完整求解。


---

# 15. S1：矩阵自由算子

实现：

```python
class UrysohnLinearOperator:
    def forward(self, theta): ...
    def adjoint(self, residual): ...
    def normal(self, theta): ...
```

要求：

- 不构造完整 design；
- 支持多输入；
- 支持 AR；
- 支持 train/validation；
- 支持 FP32/FP64；
- 支持 chunk；
- 支持 deterministic seed。

测试：

```text
test_operator_forward_matches_dense_design
test_operator_adjoint_identity
test_operator_normal_matches_gram
test_operator_multichannel
test_operator_ar_branch
test_operator_chunking_equivalence
```

门槛：

\[
\frac{
\|\mathcal A_{\mathrm{ORSS}}\theta-X\theta\|
}{
\|X\theta\|+\epsilon
}
\le10^{-10}
\]

FP64 下通过。

---

# 16. S2：增广 LSMR

实现：

```python
class AugmentedRegularizedOperator:
    def forward(self, theta): ...
    def adjoint(self, residual_blocks): ...
```

比较：

```text
dense Cholesky
dense SVD
ORSS LSMR
```

门槛：

\[
\frac{\|\theta_{\mathrm{ORSS}}-\theta_{\mathrm{dense}}\|}
{\|\theta_{\mathrm{dense}}\|+\epsilon}
\le10^{-7},
\]

\[
\frac{|L_{\mathrm{ORSS}}-L_{\mathrm{dense}}|}
{|L_{\mathrm{dense}}|+\epsilon}
\le10^{-9}.
\]

---

# 17. S3：DR 谱基

实现：

```text
lag generalized eigendecomposition
amplitude generalized eigendecomposition
tensor transform
inverse transform
penalty diagonal application
```

测试：

```text
test_dr_transform_roundtrip
test_dr_mass_orthogonality
test_dr_penalty_diagonalization
test_dr_solution_equivalence
```

---

# 18. S4：预条件器

比较：

```text
NONE
P0 diagonal
P1 channel block
P2 kronecker
P3 kronecker + AR/X Schur
```

记录：

```text
iterations
wall_time
operator_calls
peak_vram
final_residual
```

主选择：

```text
fastest preconditioner
subject to
same final solution and KKT
```

---

# 19. S5：Reduced basis

实现：

```python
class ParametricReducedBasis:
    def add_anchor(self, lambda_value, theta_full): ...
    def project_operators(self): ...
    def solve_candidates(self, lambda_grid): ...
    def estimate_residuals(self): ...
    def enrich_greedy(self): ...
```

测试：

```text
test_rb_projected_solution
test_rb_residual_estimator
test_rb_greedy_enrichment
test_rb_candidate_selection_matches_full
test_rb_zero_endpoint_handling
test_rb_fallback_when_dimension_large
```

---

# 20. S6：Mixed precision

测试：

```text
FP64 full
FP32 full
FP32 + FP64 residual
FP32 + iterative refinement
```

必须证明：

```text
same selected penalty
same one-SE set
same final validation ranking
final KKT <= 1e-8
```

不能满足时自动回退：

```text
FULL_FP64_FINAL_SOLVE
```

但只回退最终候选，不回退整个 penalty sweep。

---

# 21. S7：ORSS 等价审计

运行：

```bash
python tools/run_orss_equivalence.py \
  --config configs/cz_real_data/orss_5090.yaml \
  --tasks T1,T2,T3 \
  --strict
```

硬门槛：

```text
ADJOINT_IDENTITY_PASS
DENSE_ORSS_THETA_EQUIVALENCE_PASS
DENSE_ORSS_VALIDATION_LOSS_PASS
PENALTY_SELECTION_EQUIVALENCE_PASS
ONE_SE_SET_EQUIVALENCE_PASS
FINAL_KKT_PASS
```

性能目标：

```text
T2 speedup >= 3x
T3 speedup >= 5x
peak_vram <= 26GB
```

性能目标不通过不会否定科学正确性，但不能直接进入完整 R3，需先 profiler。

---

# 22. S8：R2.1 correctness audit

```bash
python tools/run_cz_r2_1_audit.py \
  --config configs/cz_real_data/orss_5090.yaml \
  --solver orss \
  --device cuda \
  --furnace-a-only \
  --strict \
  --resume
```

必须通过：

```text
XAR_TRAINING_NESTEDNESS_PASS
TARGET_ALIGNMENT_PASS
NO_FUTURE_X_PASS
PURGE_PASS
CONTINUATION_TRAIN_IDENTITY_PASS
FURNACE_B_ACCESS_COUNT_ZERO
FINAL_KKT_PASS
```

---

# 23. S9：R3-A 原生 history

候选：

\[
L_x\in\{32,64,128,256,512\},
\]

\[
L_y\in\{8,16,32,64,128\}.
\]

anchor：

```text
Mtau = 16
Mx = 32
c_rho = 1.0
```

horizon：

\[
h\in\{1,5,15,30,60\}.
\]

fold：

```text
F1 0–40% → 40–50%
F2 0–50% → 50–60%
F3 0–60% → 60–70%
F4 0–70% → 70–80%
```

运行：

```bash
python tools/run_cz_r3.py \
  --stage history \
  --config configs/cz_real_data/r3_history.yaml \
  --solver orss \
  --device cuda \
  --resume \
  --checkpoint-every-task \
  --furnace-a-only
```

one-SE 复杂度键：

\[
(
L_x+L_y,
L_xL_y,
\max(L_x,L_y),
L_x,
L_y
).
\]

输出：

```text
history_candidates.parquet
history_selection.json
history_one_se_set.json
history_runtime_profile.json
```

---

# 24. S10：R3-B resolution / penalty

冻结 history 后：

\[
M_\tau\in\{16,32,48,64\},
\]

\[
M_x\in\{16,20,24,28,32\}.
\]

penalty：

```text
exact zero
7 positive values per axis
maximum 2 edge expansions
```

运行：

```bash
python tools/run_cz_r3.py \
  --stage resolution-penalty \
  --config configs/cz_real_data/r3_resolution.yaml \
  --solver orss \
  --device cuda \
  --resume \
  --checkpoint-every-task \
  --furnace-a-only
```

门禁：

```text
REPRESENTATION_GATE_PASS
LEPSKI_PASS
PENALTY_INTERVAL_CERTIFIED
RB_RESIDUAL_CERTIFIED
FINAL_KKT_PASS
```

---

# 25. S11：R3-C continuation

候选：

\[
c_\rho\in\{0.5,1,2,4\}.
\]

运行：

```bash
python tools/run_cz_r3.py \
  --stage continuation \
  --config configs/cz_real_data/r3_continuation.yaml \
  --solver orss \
  --device cuda \
  --resume \
  --checkpoint-every-task \
  --furnace-a-only
```

选择顺序：

```text
finite complete free-run
lowest validation free-run loss
lower normalized extrapolation distance
lower continuation usage
smaller c_rho
```

输出：

```text
continuation_selection.json
first_exit_audit.json
continuation_usage.csv
free_run_metrics.json
```

---

# 26. S12：R3-D rank

rank budget：

\[
\epsilon\in\{0.10,0.05,0.02\}.
\]

输出：

```text
full
rank1
rank2
adaptive rank
predictive SVD rank
relative-to-full inflation
structural rank only when K-level conditions pass
```

运行：

```bash
python tools/run_cz_r3.py \
  --stage rank \
  --config configs/cz_real_data/r3_rank.yaml \
  --solver orss \
  --device cuda \
  --resume \
  --checkpoint-every-task \
  --furnace-a-only
```

---

# 27. S13：完整 baselines

最低基线：

```text
Mean
Persistence
AR
X-only/FIR
ARX
```

非线性基线：

```text
pNARX
MLP-NARX
GRU or LSTM
AKGNN reference
Stage1TargetDelayKAN if reproducible
```

统一主输入：

```text
主加热功率
晶升速度
晶转速度
埚升速度
埚转速度
```

温度只作 sensitivity branch。

运行：

```bash
python tools/run_cz_r4_baselines.py \
  --config configs/cz_real_data/r4_baselines.yaml \
  --furnace-a-only \
  --resume
```

---

# 28. S14：第一炉 internal confirmation

模型、history、resolution、penalty、continuation、rank 全部冻结后，访问第一炉最后 20%。

运行：

```bash
python tools/run_cz_r5_confirmation.py \
  --config configs/cz_real_data/r5_confirmation.yaml \
  --locked-model results/cz_real_data/frozen_model \
  --resume
```

禁止：

- 重新调参；
- 扩展 scaler；
- 扩展 basis domain；
- 改 continuation；
- 改 rank；
- 删除不利 horizon。

---

# 29. S15：第二炉 zero-shot outer

只有以下全部通过后才允许打开第二炉：

```text
H2_NATIVE_COMPLETE
R3_RESOLUTION_COMPLETE
R3_CONTINUATION_COMPLETE
R3_RANK_COMPLETE
BASELINE_MATRIX_COMPLETE
FURNACE_A_CONFIRMATION_COMPLETE
FROZEN_MODEL_HASH_WRITTEN
FURNACE_B_ACCESS_COUNT_ZERO
```

运行：

```bash
python tools/run_cz_r6_outer.py \
  --config configs/cz_real_data/r6_outer.yaml \
  --frozen-model results/cz_real_data/frozen_model \
  --outer-workbook data/raw/实验数据1-张.xlsx \
  --sheet Sheet2 \
  --zero-shot \
  --resume
```

输出：

```text
outer_zero_shot_metrics.json
outer_support_shift.json
outer_domain_usage.csv
outer_free_run.json
outer_incremental_value.json
```

---

# 30. S16：第二炉轻量校准

zero-shot 完成后：

```text
5% calibration
10% calibration
```

只允许：

```text
intercept
output scale
pre-registered low-dimensional adapter
```

禁止重搜：

```text
history
resolution
penalty
rank
kernel basis
```

运行：

```bash
python tools/run_cz_r7_calibration.py \
  --config configs/cz_real_data/r7_calibration.yaml \
  --fractions 0.05 0.10 \
  --resume
```

---

# 31. S17：解释性与机制审计

每个输入输出：

```text
lag profile
amplitude response
full kernel surface
singular spectrum
rank1/rank2 reconstruction
bootstrap band
support mask
weak-operator/inactive status
```

跨炉比较：

```text
leading singular value ratio
leading mode correlation
principal angle
lag peak shift
amplitude shape shift
continuation usage shift
```

K 层解释只在：

\[
\mathcal S^{\mathrm{cert}}
\]

和 finite-sieve coercivity / Schur 条件通过时允许。

---

# 32. S18：Bootstrap

development：

```text
B = 250
```

confirmation：

```text
B = 1000
```

单位为连续时间块，禁止把约 20,000 个高度相关样本视为独立样本。

报告：

\[
\Delta RMSE
=
RMSE_{\mathrm{baseline}}
-
RMSE_{\mathrm{ours}},
\]

\[
\Delta_{U|AR}(h)
=
\frac{MSE_{AR}(h)-MSE_{U+AR}(h)}
{MSE_{AR}(h)}.
\]

---

# 33. 最终论文结果矩阵

## 表 1：数据与变量

```text
samples
duration
sampling period
units
input roles
constant variables
excluded variables
```

## 表 2：Direct prediction

```text
h = 1,5,15,30,60
RMSE / MAE / R2
AR incremental value
```

## 表 3：Free-run

```text
full-trajectory RMSE
drift slope
maximum bias
continuation usage
```

## 表 4：Rank

```text
full
rank1
rank2
adaptive
predictive rank
```

## 表 5：Cross-furnace

```text
zero-shot
5% calibration
10% calibration
```

## 表 6：Computation

```text
dense time
full ORSS time
RB-ORSS time
speedup
peak RAM
peak VRAM
Krylov iterations
RB dimension
full anchor solves
```

---

# 34. 一键启动脚本

建立：

```text
tools/launch_cz_5090_complete.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

ROOT="results/cz_real_data/complete_5090"
LOGS="$ROOT/logs"

mkdir -p "$LOGS"

python tools/run_orss_smoke.py \
  --config configs/cz_real_data/orss_5090.yaml \
  --strict \
  2>&1 | tee "$LOGS/00_orss_smoke.log"

python tools/run_orss_equivalence.py \
  --config configs/cz_real_data/orss_5090.yaml \
  --tasks T1,T2,T3 \
  --strict \
  2>&1 | tee "$LOGS/01_orss_equivalence.log"

python tools/run_cz_r2_1_audit.py \
  --config configs/cz_real_data/orss_5090.yaml \
  --solver orss \
  --device cuda \
  --furnace-a-only \
  --strict \
  --resume \
  2>&1 | tee "$LOGS/02_r2_1.log"

python tools/run_cz_r3.py \
  --stage history \
  --config configs/cz_real_data/r3_history.yaml \
  --solver orss \
  --device cuda \
  --furnace-a-only \
  --checkpoint-every-task \
  --resume \
  2>&1 | tee "$LOGS/03_history.log"

python tools/run_cz_r3.py \
  --stage resolution-penalty \
  --config configs/cz_real_data/r3_resolution.yaml \
  --solver orss \
  --device cuda \
  --furnace-a-only \
  --checkpoint-every-task \
  --resume \
  2>&1 | tee "$LOGS/04_resolution_penalty.log"

python tools/run_cz_r3.py \
  --stage continuation \
  --config configs/cz_real_data/r3_continuation.yaml \
  --solver orss \
  --device cuda \
  --furnace-a-only \
  --checkpoint-every-task \
  --resume \
  2>&1 | tee "$LOGS/05_continuation.log"

python tools/run_cz_r3.py \
  --stage rank \
  --config configs/cz_real_data/r3_rank.yaml \
  --solver orss \
  --device cuda \
  --furnace-a-only \
  --checkpoint-every-task \
  --resume \
  2>&1 | tee "$LOGS/06_rank.log"

python tools/run_cz_r4_baselines.py \
  --config configs/cz_real_data/r4_baselines.yaml \
  --furnace-a-only \
  --resume \
  2>&1 | tee "$LOGS/07_baselines.log"

python tools/run_cz_r5_confirmation.py \
  --config configs/cz_real_data/r5_confirmation.yaml \
  --resume \
  2>&1 | tee "$LOGS/08_furnace_a_confirmation.log"

python tools/freeze_cz_model.py \
  --results-root "$ROOT" \
  --output "$ROOT/frozen_model" \
  2>&1 | tee "$LOGS/09_freeze_model.log"

python tools/run_cz_r6_outer.py \
  --config configs/cz_real_data/r6_outer.yaml \
  --frozen-model "$ROOT/frozen_model" \
  --zero-shot \
  --resume \
  2>&1 | tee "$LOGS/10_furnace_b_zero_shot.log"

python tools/run_cz_r7_calibration.py \
  --config configs/cz_real_data/r7_calibration.yaml \
  --fractions 0.05 0.10 \
  --resume \
  2>&1 | tee "$LOGS/11_furnace_b_calibration.log"

python tools/run_cz_interpretability.py \
  --config configs/cz_real_data/cz_real_data_v1.yaml \
  --results-root "$ROOT" \
  --resume \
  2>&1 | tee "$LOGS/12_interpretability.log"

python tools/run_cz_bootstrap.py \
  --config configs/cz_real_data/cz_real_data_v1.yaml \
  --development-replicates 250 \
  --confirmation-replicates 1000 \
  --resume \
  2>&1 | tee "$LOGS/13_bootstrap.log"

python tools/summarize_cz_complete.py \
  --results-root "$ROOT" \
  --output CZ_COMPLETE_5090_REPORT.md \
  2>&1 | tee "$LOGS/14_summary.log"

echo "CZ_5090_COMPLETE_PIPELINE_FINISHED"
```

---

# 35. Checkpoint 唯一键

```text
dataset
furnace
fold
horizon
Lx
Ly
Mtau
Mx
penalty_round
lambda_tau
lambda_x
lambda_0
continuation_c_rho
rank_budget
solver
preconditioner
RB_dimension
source_commit
config_hash
data_hash
```

完成文件：

```text
task_result.json
model.pt
solver_diagnostics.json
rb_state.pt
DONE
```

失败文件：

```text
FAILED
exception.txt
solver_state.pt
```

恢复必须验证：

```text
source_commit
config_hash
data_hash
solver_version
```

---

# 36. 状态码

## 允许继续

```text
ORSS_OPERATOR_PASS
ADJOINT_IDENTITY_PASS
DR_BASIS_PASS
PRECONDITIONER_PASS
RB_RESIDUAL_CERTIFIED
MIXED_PRECISION_REFINED
FINAL_KKT_PASS
R2_1_PASS
R3_HISTORY_COMPLETE
R3_RESOLUTION_COMPLETE
R3_CONTINUATION_COMPLETE
R3_RANK_COMPLETE
FURNACE_A_CONFIRMATION_COMPLETE
FURNACE_B_ZERO_SHOT_COMPLETE
CZ_COMPLETE_PIPELINE_FINISHED
```

## 自动回退

```text
RB_DIMENSION_EXCEEDED
RB_RESIDUAL_NOT_CERTIFIED
MIXED_PRECISION_REFINEMENT_FAILED
PRECONDITIONER_INEFFECTIVE
```

回退目标：

```text
FULL_ORSS
FINAL_FP64_SOLVE
```

## 硬停止

```text
ADJOINT_IDENTITY_FAILED
DENSE_ORSS_EQUIVALENCE_FAILED
TARGET_ALIGNMENT_FAILED
PURGE_FAILED
FURNACE_B_ACCESSED_BEFORE_FREEZE
CONTINUATION_CHANGED_IN_SUPPORT_MODEL
FINAL_KKT_FAILED
RESULT_HASH_MISMATCH
```

## 只记录

```text
WEAK_EXTERNAL_INCREMENT
AR_DOMINATES_EXTERNAL_INPUT
HISTORY_SELECTED_AT_GRID_EDGE
HIGH_CONTINUATION_USAGE
GLOBAL_INCREMENTAL_STABILITY_NOT_CERTIFIED
DIRECT_GOOD_FREE_RUN_POOR
CROSS_FURNACE_DOMAIN_SHIFT
K_LEVEL_NOT_IDENTIFIED
```

---

# 37. 测试清单

```text
test_operator_forward_matches_dense
test_adjoint_identity
test_normal_operator_matches_gram
test_augmented_operator
test_lsmr_matches_dense
test_dr_roundtrip
test_dr_penalty_diagonalization
test_preconditioner_does_not_change_solution
test_rb_projection
test_rb_residual_estimator
test_rb_greedy_enrichment
test_rb_zero_endpoint
test_mixed_precision_refinement
test_final_kkt
test_one_se_set_equivalence
test_effective_df_eligible_only
test_continuation_train_identity
test_furnace_b_lock
test_checkpoint_resume
test_result_hash
```

运行：

```bash
python -m pytest -q
python -m pytest -q tests/test_orss.py
python -m pytest -q tests/test_cz_complete.py
```

---

# 38. Profiler 输出

每个代表任务保存：

```text
window_seconds
basis_seconds
operator_forward_seconds
operator_adjoint_seconds
krylov_iterations
preconditioner_build_seconds
anchor_full_solve_seconds
rb_projection_seconds
rb_scan_seconds
greedy_iterations
refinement_seconds
final_kkt_seconds
peak_vram
peak_ram
```

必须对比：

```text
dense solver
full ORSS
RB-ORSS
mixed precision RB-ORSS
```

---

# 39. 最终结果包

名称：

```text
CZ_5090_ORSS_COMPLETE_EXPERIMENT_RESULTS_bundle.zip
```

必须包含：

```text
README.md
CHANGELOG.md
CZ_COMPLETE_5090_REPORT.md
CZ_COMPLETE_5090_STATUS.json

src/
tools/
tests/
configs/

results/
logs/
environment/

orss_equivalence.json
orss_profile.json
operator_adjoint_audit.json
rb_residual_audit.json
mixed_precision_audit.json

history_selection.json
resolution_selection.json
penalty_search.json
continuation_selection.json
rank_profile.csv

furnace_a_confirmation.json
furnace_b_zero_shot.json
furnace_b_calibration.json

interpretability/
bootstrap/

SOURCE_COMMIT.txt
FINAL_COMMIT.txt
repository.bundle
PACKAGE_MANIFEST.json
SHA256SUMS.txt
```

---

# 40. 清理、manifest、hash、校验与打包

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
NAME="CZ_5090_ORSS_COMPLETE_EXPERIMENT_RESULTS_bundle"
OUT="$ROOT/return/$NAME"
ZIP="$ROOT/return/$NAME.zip"

rm -rf "$OUT" "$ZIP" "$ZIP.sha256"
mkdir -p "$OUT"

cp -a README.md CHANGELOG.md "$OUT/" 2>/dev/null || true
cp -a src tools tests configs "$OUT/"
cp -a results/cz_real_data/complete_5090 "$OUT/results"
cp -a environment "$OUT/" 2>/dev/null || true

cp -a \
  CZ_COMPLETE_5090_REPORT.md \
  CZ_COMPLETE_5090_STATUS.json \
  "$OUT/"

git rev-parse HEAD \
  > "$OUT/FINAL_COMMIT.txt"

cp -a \
  results/orss_reference/DENSE_REFERENCE_COMMIT.txt \
  "$OUT/SOURCE_COMMIT.txt"

git bundle create \
  "$OUT/repository.bundle" \
  --all

find "$OUT" -type d \
  \( -name "__pycache__" -o -name ".pytest_cache" \) \
  -prune -exec rm -rf {} +

find "$OUT" -type f \
  \( -name "*.pyc" -o -name "*.tmp" \) \
  -delete

python tools/build_manifest.py \
  --root "$OUT" \
  --output "$OUT/PACKAGE_MANIFEST.json"

(
  cd "$OUT"

  find . -type f \
    ! -name SHA256SUMS.txt \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS.txt
)

python - <<'PY'
from pathlib import Path
import json

root = Path(
    "return/CZ_5090_ORSS_COMPLETE_EXPERIMENT_RESULTS_bundle"
)

required = [
    "src",
    "tools",
    "tests",
    "configs",
    "results",
    "environment",
    "CZ_COMPLETE_5090_REPORT.md",
    "CZ_COMPLETE_5090_STATUS.json",
    "FINAL_COMMIT.txt",
    "SOURCE_COMMIT.txt",
    "repository.bundle",
    "PACKAGE_MANIFEST.json",
    "SHA256SUMS.txt",
]

missing = [
    x for x in required
    if not (root / x).exists()
]

if missing:
    raise SystemExit(
        f"Missing required package entries: {missing}"
    )

json.loads(
    (root / "PACKAGE_MANIFEST.json")
    .read_text(encoding="utf-8")
)

print("PACKAGE_STRUCTURE_OK")
PY

(
  cd "$ROOT/return"

  zip -qr \
    "$NAME.zip" \
    "$NAME"

  unzip -t \
    "$NAME.zip"

  sha256sum \
    "$NAME.zip" \
    > "$NAME.zip.sha256"
)

echo "FINAL_PACKAGE=$ZIP"
echo "FINAL_SHA256=$ZIP.sha256"
```

---

# 41. 完成条件

整个实验完成必须同时满足：

```text
ORSS_EQUIVALENCE_PASS
ORSS_PROFILE_COMPLETE
R2_1_PASS
H2_NATIVE_COMPLETE
RESOLUTION_COMPLETE
PENALTY_COMPLETE
CONTINUATION_COMPLETE
RANK_COMPLETE
BASELINE_MATRIX_COMPLETE
FURNACE_A_CONFIRMATION_COMPLETE
FURNACE_B_ZERO_SHOT_COMPLETE
FURNACE_B_CALIBRATION_COMPLETE
INTERPRETABILITY_COMPLETE
BOOTSTRAP_COMPLETE
PACKAGE_INTEGRITY_PASS
```

---

# 42. 结论边界

完成后可以主张：

- 提出 operator-first、矩阵自由、可降阶、可混合精度认证的 Spectral PS-AR-RAPHU 求解框架；
- 稠密 \(O(Cp^3)\) penalty sweep 被 reduced operator sweep 替代；
- ORSS 与 dense reference 在小规模任务上数值等价；
- RTX 5090 完成完整实验；
- 第一炉完成 development 和 confirmation；
- 第二炉完成 zero-shot 与轻量校准；
- prediction、free-run、rank 和解释性具有完整证据链。

仍不能主张：

- continuation 学到了训练支撑外真实 K；
- 固定阶段模型自动覆盖全部拉晶阶段；
- 当前闭环预测算子等于 causal plant；
- 两炉数据足以证明所有晶棒普适性；
- reduced basis 在所有任务上必然高效。

---

# 43. 最终执行原则

\[
\boxed{\text{模型不变，求解器重构；}}
\]

\[
\boxed{\text{大算子低精度，小投影高精度；}}
\]

\[
\boxed{
\text{全部候选先 reduced sweep，}
\text{最终候选 full-space 认证；}
}
\]

\[
\boxed{\text{RTX 5090 完成整个实验，无需 A800。}}
\]
