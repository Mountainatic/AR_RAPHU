# Centered OD-FUOI-PSAR：连续 EDF 时间块选择 CPU 确认实验方案 V2

> **实验名称**：`CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2`  
> **源方案**：`OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_EXPERIMENT_PLAN.md`  
> **目标硬件**：32 vCPU，AMD EPYC 9654 96-Core Processor  
> **数值精度**：CPU FP64  
> **环境**：`uv`  
> **正式修改**：
>
> 1. absolute amplitude 改为 centered increment；
> 2. pointwise GCV 改为 continuous effective-df blocked CV + continuous 1-SE；
> 3. GCV/REML/L-curve 降级为诊断；
> 4. 其余 full Urysohn、派生投影、C1 和 residual PS/AR 保持。
>
> **禁止引入**：公共/私有字典、外挂通道权重、Rank 搜索、Rank-2 救援、两个手工 penalty、人工 \(\lambda_{\min}\)、人工 df=30 锁。

---

# 0. 科学目标

V1 已经证明：

```text
absolute-amplitude + pointwise GCV + full Urysohn
```

在当前两棒闭环数据上发生严重下平滑、跨棒 OOD 和线性—非线性抵消。

V2 只检验一个干净修复：

\[
\boxed{
\text{Centered-increment Full Urysohn}
+
\text{Continuous-EDF blocked predictive selection}.
}
\]

要回答：

1. centered increment 是否显著减少跨棒幅值 OOD？
2. continuous EDF blocked-CV 是否避免 \(d_{\mathrm{eff}}\approx700\) 的插值解？
3. full centered Urysohn 是否至少恢复到有限、稳定、可迁移的预测量级？
4. 派生 Rank-1 linear 是否与冻结 NLinear-U 出现一致结构？
5. 纯非线性面是否带来稳定增量？
6. frozen centered K 后 residual PS/AR 是否有增益？

---

# 1. 冻结协议

保持 V1：

- shared dataset SHA256：
  ```text
  fb2e3a3957c2bc4b4dcb30f0c5d425fd6e75068e945aace835e1a3698a6206db
  ```
- cadence：10 s；
- history：40 min；
- horizon：20 min；
- output mean：2 min；
- four inputs；
- two outer transfers；
- 4-fold expanding-window；
- purge ≥22 min；
- sample IDs、target、PCA、scaler、breaks 全部冻结；
- outer test 只评估一次。

复用旧逐样本预测：

- Persistence；
- old K-only；
- Dynamic-PLS；
- NLinear-U seed-median；
- Temporal AE；
- Joint-K+AR。

---

# 2. Centered-increment Full Urysohn

## 2.1 坐标

共享 bundle 中输入记为 \(x_{j,t}\)。

定义：

\[
\delta_{j,t}(\ell)
=
x_{j,t-\ell}-x_{j,t},
\qquad
\ell=0,\ldots,239.
\tag{2.1}
\]

必有：

\[
\delta_{j,t}(0)=0.
\tag{2.2}
\]

正式模型：

\[
\boxed{
\widehat z_t^K
=
b+
\sum_{j=1}^{4}
\sum_{\ell=0}^{239}
K_j\!\left(
\ell\Delta t,
\delta_{j,t}(\ell)
\right)\Delta t.
}
\tag{2.3}
\]

## 2.2 与 NLinear 的嵌套

若：

\[
K_j(\tau,\delta)
=
\beta_j(\tau)\delta
\tag{2.4}
\]

且：

\[
\operatorname{rank}
[\beta_1,\ldots,\beta_4]=1,
\tag{2.5}
\]

则模型退化为 NLinear 型共享时滞 Rank-1。

不直接训练 Rank-1 因子；只在 full fit 后派生。

---

# 3. 固定 basis

## 3.1 Lag basis

保持 V1：

\[
s=\sqrt{\tau/40\mathrm{min}}.
\]

```text
degree = 3
number_of_basis = 20
```

## 3.2 Centered-amplitude basis

对每个 inner fold 和 outer full-training rod，分别从训练部分的：

\[
\{\delta_{j,t}(\ell)\}_{t,\ell}
\]

生成分位数 knots：

```text
0%, 5%, 15%, 30%, 50%, 70%, 85%, 95%, 100%
degree = 3
```

validation/test 不能参与。

重复 knots 合并；支持退化时登记：

```text
CENTERED_AMPLITUDE_SUPPORT_DEGENERATE
```

## 3.3 Mesh refinement

仅作审计：

```text
lag basis = 28
centered-amplitude quantiles =
0, 2, 10, 20, 35, 50, 65, 80, 90, 98, 100%
```

不得用 refinement 选择主模型。

---

# 4. 单一 Sobolev 几何

保持一个全局尺度：

\[
\min_{b,K_1,\ldots,K_4}
\frac1n
\sum_t
\left[
z_t-b-\sum_j\mathcal K_j[\delta_j](t)
\right]^2
+
\lambda
\sum_j
\|K_j\|_{\mathcal H_j^\star}^2.
\tag{4.1}
\]

归一化 lag/amplitude 坐标中的 penalty：

\[
\|K_j\|_{\mathcal H_j^\star}^2
=
\|K_j\|_{L_2}^2
+
\int
(K_{ss}^2+2K_{sv}^2+K_{vv}^2).
\tag{4.2}
\]

不存在第二个 penalty。

---

# 5. Continuous EDF blocked predictive selection

## 5.1 Fold smoothing map

对 inner fold \(f\)：

\[
S_{f,\lambda}
=
\Phi_f
(\Phi_f^\top\Phi_f+n_f\lambda P_f)^{-1}
\Phi_f^\top.
\tag{5.1}
\]

\[
d_f(\lambda)=\operatorname{tr}S_{f,\lambda}.
\tag{5.2}
\]

将 penalty null-space 分离后，计算 generalized eigenspectrum：

\[
d_f(\lambda)
=
d_{0,f}
+
\sum_i
\frac{\gamma_{f,i}}
{\gamma_{f,i}+n_f\lambda}.
\tag{5.3}
\]

## 5.2 Common stable EDF interval

每个 fold 通过 continuation 找到：

\[
\mathcal D_f=(d_{0,f},d_{\mathrm{stable},f}].
\]

数值稳定要求：

\[
\kappa(H_{f,\lambda})\epsilon_{\mathrm{FP64}}
\le10^{-6},
\tag{5.4}
\]

并要求：

- Cholesky/QR 成功；
- KKT finite；
- prediction finite；
- leverage finite。

共同区间：

\[
\mathcal D=\bigcap_f\mathcal D_f.
\tag{5.5}
\]

若为空，登记：

```text
NO_COMMON_STABLE_EDF_INTERVAL
```

并停止。

## 5.3 Fold-specific inversion

给定 \(d\in\mathcal D\)，每个 fold 单独用 `brentq` 求：

\[
d_f(\lambda_f(d))=d.
\tag{5.6}
\]

求根相对误差：

```text
1e-8
```

## 5.4 Blocked validation profile

\[
L_f(d)
=
\frac1{|V_f|}
\sum_{t\in V_f}
(z_t-\widehat z_{t,f,d})^2.
\tag{5.7}
\]

\[
\overline L(d)
=
\frac{\sum_f|V_f|L_f(d)}
{\sum_f|V_f|}.
\tag{5.8}
\]

\[
\operatorname{SE}(d)
=
\sqrt{
\frac{
\sum_f|V_f|
(L_f(d)-\overline L(d))^2
}{
(F-1)\sum_f|V_f|
}
}.
\tag{5.9}
\]

## 5.5 Adaptive continuous global profile

不预设九个候选 anchors，不假设单峰。

算法：

```text
1. Evaluate common interval endpoints and midpoint.
2. Recursively subdivide unresolved intervals.
3. Save every actual evaluation.
4. Detect all local-minimum brackets.
5. Apply bounded Brent in every bracket.
6. Choose the lowest refined minimum.
```

终止条件：

```text
d absolute tolerance = 0.05
relative profile interpolation tolerance = 1e-4
max actual profile evaluations = 80
```

若未解析：

```text
CONTINUOUS_EDF_PROFILE_UNRESOLVED
```

## 5.6 Continuous 1-SE

\[
d_{\min}
=
\arg\min_{d\in\mathcal D}\overline L(d).
\tag{5.10}
\]

\[
T_{\mathrm{1SE}}
=
\overline L(d_{\min})
+
\operatorname{SE}(d_{\min}).
\tag{5.11}
\]

取 1-SE 可接受集合中包含 \(d_{\min}\) 的连通分支，选择其左端：

\[
\boxed{
d_{\mathrm{1SE}}
=
\inf\mathcal C_{\min}.
}
\tag{5.12}
\]

从 \(d_{\min}\) 向左，用 `brentq` 求：

\[
\overline L(d)=T_{\mathrm{1SE}}.
\tag{5.13}
\]

若撞下界：

```text
ONE_SE_HITS_LOWER_COMPLEXITY_BOUND
```

若 \(d_{\min}\) 或 \(d_{\mathrm{1SE}}\) 撞上界：

```text
BCV_OPTIMUM_HITS_EDF_UPPER_BOUND
SMOOTHING_SELECTION_UNRESOLVED
```

## 5.7 Outer full refit

在完整 outer training rod 上求：

\[
d_{\mathrm{full}}(\lambda_{\mathrm{full}})
=
d_{\mathrm{1SE}}.
\tag{5.14}
\]

保存：

```text
selected_edf
derived_lambda_full
```

GCV/REML/L-curve只作诊断。

---

# 6. 幅值投影

对每个 lag 的 centered increment empirical measure：

\[
\phi(\delta)=
\begin{bmatrix}
1\\
\delta
\end{bmatrix}.
\]

\[
G_{j,\tau}
=
\int\phi\phi^\top\,d\nu_{j,\tau}.
\tag{6.1}
\]

\[
c_{j,\tau}
=
G_{j,\tau}^{-1}
\int K_j(\tau,\delta)\phi(\delta)\,d\nu_{j,\tau}.
\tag{6.2}
\]

得到：

\[
K_j(\tau,\delta)
=
m_j(\tau)
+
\beta_j(\tau)\delta
+
N_j(\tau,\delta).
\tag{6.3}
\]

要求：

\[
\langle N_j,1\rangle=0,
\qquad
\langle N_j,\delta\rangle=0.
\tag{6.4}
\]

重构和正交误差均小于：

```text
1e-10
```

---

# 7. 派生模型

## M0

Persistence。

## M1

冻结 GPU NLinear-U baseline。

## M2

Centered derived Rank-1 linear：

- 从最终 full K 的 \(\beta_j\) 构造矩阵；
- Gram-SVD；
- 只保留最佳 Rank-1；
- 不重新训练。

## M3

Centered general linear Urysohn：

\[
K_j^{\mathrm{LIN}}=\beta_j(\tau)\delta.
\]

## M4

Centered full Urysohn：

\[
K_j^{\mathrm{FULL}}
=
\beta_j(\tau)\delta+N_j(\tau,\delta).
\]

## M5

Centered full Urysohn + matured residual PS/AR。

---

# 8. E0–E9

## E0：Preflight

- 复用 shared/baseline；
- 校验 0.365854 NLinear；
- 登记 V1：
  ```text
  ABSOLUTE_AMPLITUDE_POINTWISE_GCV_REJECTED
  ```

## E1：Centered coordinate audit

输出：

- absolute vs centered OOD；
- per-channel centered support；
- validation/test extension ratio；
- saturated ratio；
- \(\delta(0)=0\)；
- no-future-input。

## E2A：Fold EDF maps

输出每 fold：

- generalized eigenvalues；
- null-space df；
- stable upper df；
- \(d_f(\lambda)\) audit。

## E2B：Continuous profile

执行 adaptive BCV profile。

## E2C：Continuous 1-SE

选择：

```text
d_min
d_1se
```

## E2D：Full-training refit

求：

```text
lambda_full(d_1se)
```

输出 centered full surfaces。

## E3：Projection

输出：

- \(m_j\)；
- \(\beta_j\)；
- \(N_j\)；
- reconstruction/orthogonality。

## E4：NLinear audit

输出：

- Rank-1 energy；
- time-shape correlation；
- channel coordinates；
- M2 RMSE；
- vs GPU NLinear paired errors。

## E5：Linear vs Full

比较 M3 与 M4。

## E6：Channel audit

保持 V1，但在 centered coordinates 下重算。

## E7：C1/OOD

延拓作用于 centered increment 幅值轴。

## E8：Residual PS/AR

使用新 full-K rolling OOF residual，不能复用 V1。

## E9：Leaderboard and bootstrap

生成 input-only 和 dynamic 两榜。

---

# 9. 主要状态

## 9.1 估计器状态

### `CENTERED_ESTIMATOR_STABLE`

要求：

- common EDF interval 非空；
- profile resolved；
- optimum 不撞上界；
- outer predictions finite；
- condition/KKT 通过；
- mesh refinement 不爆炸；
- RMSE 不出现数量级爆炸。

### `SMOOTHING_SELECTION_UNRESOLVED`

任一：

- profile unresolved；
- optimum 撞上界；
- fold-specific minima 完全冲突；
- final refit 无法匹配 \(d_{\mathrm{1SE}}\)。

## 9.2 模型状态

### `CENTERED_FULL_URYSOHN_CONFIRMED`

- pooled RMSE < 0.365854；
- 两方向优于 Persistence；
- vs NLinear pooled bootstrap positive probability ≥90%；
- nonlinear/full 证书通过。

### `CENTERED_URYSOHN_PARETO`

- 与 NLinear RMSE 差不超过 1%；
- 两方向均为正；
- 结构/数值证书通过。

### `CENTERED_URYSOHN_IMPROVES_OLD_K_ONLY`

- 优于 0.412383；
- 但未达到 NLinear。

### `CENTERED_FULL_URYSOHN_REJECTED`

- 稳定估计器下仍不优于旧 K；
- 或方向性严重失败；
- 或 OOD/mesh/placebo 不通过。

必须区分：

```text
ESTIMATOR_UNRESOLVED
```

和：

```text
MODEL_REJECTED_UNDER_STABLE_ESTIMATOR
```

---

# 10. 结果文件

```text
results/
├── PRECHECK_REPORT.md
├── CENTERED_COORDINATE_AUDIT.json
├── FOLD_EDF_MAPS.csv
├── COMMON_EDF_INTERVAL.json
├── CONTINUOUS_EDF_PROFILE.csv
├── CONTINUOUS_EDF_MINIMA.json
├── CONTINUOUS_ONE_SE_SELECTION.json
├── DIAGNOSTIC_GCV_REML_LCURVE.csv
├── CENTERED_FULL_URYSOHN_METRICS.csv
├── CENTERED_PROJECTION_METRICS.csv
├── CENTERED_NLINEAR_AUDIT.csv
├── CENTERED_NONLINEAR_INCREMENT.csv
├── CENTERED_CHANNEL_AUDIT.csv
├── CENTERED_C1_OOD_AUDIT.csv
├── CENTERED_RESIDUAL_PSAR.csv
├── FINAL_INPUT_LEADERBOARD.csv
├── FINAL_DYNAMIC_LEADERBOARD.csv
├── PAIRWISE_BOOTSTRAP.csv
├── FINAL_DECISION.json
├── FINAL_REPORT.md
├── predictions/
├── surfaces/
├── projections/
├── edf_profiles/
├── ood/
├── diagnostics/
└── plots/
```

---

# 11. 必须测试

1. shared hash；
2. baseline sample IDs；
3. centered delta formula；
4. \(\delta(0)=0\)；
5. no future input；
6. fold-training-only supports；
7. tensor basis；
8. penalty SPD on penalized subspace；
9. generalized eigenspectrum；
10. \(d_f(\lambda)\) monotonic；
11. brentq inversion；
12. common EDF interval；
13. equal target df across folds；
14. synthetic unimodal profile；
15. synthetic bimodal profile；
16. all minima bracketed；
17. continuous one-SE connected component；
18. upper-bound unresolved；
19. final full refit df match；
20. GCV not used for selection；
21. projection reconstruction；
22. nonlinear orthogonality；
23. NLinear exact subset；
24. C1 continuity；
25. matured residual causality；
26. paired bootstrap；
27. mesh refinement；
28. FP64 KKT；
29. package privacy；
30. manifest/hash/ZIP roundtrip。

---

# 12. 运行和资源

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

建议：

```text
fold/profile workers = 8
bootstrap workers = 16
BLAS threads = 1
dtype = float64
```

预计：

```text
12–36 h
```

continuous profile 比 V1 更贵，但 generalized eigenspectrum 可缓存。

每次 profile evaluation 必须 checkpoint。

---

# 13. 代码目录

```text
CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2/
├── configs/
│   └── frozen_protocol.yaml
├── src/
│   ├── shared_data/
│   ├── centered_increment/
│   ├── tensor_spline/
│   ├── sobolev_geometry/
│   ├── edf_geometry/
│   ├── blocked_cv/
│   ├── continuous_profile/
│   ├── full_urysohn/
│   ├── amplitude_projection/
│   ├── nlinear_audit/
│   ├── c1_extension/
│   ├── matured_residual/
│   ├── diagnostics/
│   ├── evaluation/
│   └── packaging/
├── scripts/
│   ├── preflight.py
│   ├── build_centered_design.py
│   ├── build_fold_edf_maps.py
│   ├── run_continuous_edf_profile.py
│   ├── select_continuous_one_se.py
│   ├── refit_centered_full_urysohn.py
│   ├── derive_centered_projection.py
│   ├── run_centered_nlinear_audit.py
│   ├── run_centered_nonlinear_audit.py
│   ├── run_c1_ood_audit.py
│   ├── run_residual_psar.py
│   ├── run_final_compare.py
│   └── build_return_bundle.py
├── tests/
├── results/
├── logs/
├── return/
├── RUN_CPU_CONFIRM.sh
└── RESUME_CPU_CONFIRM.sh
```

---

# 14. 启动合同

```bash
cd CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2

bash RUN_CPU_CONFIRM.sh \
  --shared /path/to/SHARED_BENCHMARK_DATASET_bundle.zip \
  --cpu-baselines /path/to/PHYSICS_FIRST_CPU_RESULTS_bundle.zip \
  --gpu-baselines /path/to/PHYSICS_FIRST_GPU_RESULTS_bundle.zip \
  --v1-results /path/to/OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS_bundle.zip \
  --protocol configs/frozen_protocol.yaml \
  --profile-workers 8 \
  --bootstrap-workers 16 \
  2>&1 | tee logs/full_run.log
```

---

# 15. 打包合同

```bash
rm -rf return/CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS
rm -f return/CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS_bundle.zip
rm -f return/CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS_bundle.zip.sha256

uv run python scripts/build_return_bundle.py \
  --source-root . \
  --results results \
  --output return/CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS

uv run python scripts/validate_package.py \
  --package-dir return/CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS \
  --forbid "*.xlsx,.git,__pycache__,cache,*.tmp,raw_data"

uv run python scripts/build_manifest.py \
  --root return/CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS \
  --output return/CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS/MANIFEST.json

cd return
zip -r CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS_bundle.zip \
  CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS

sha256sum CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS_bundle.zip \
  > CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS_bundle.zip.sha256

uv run python ../scripts/validate_zip_roundtrip.py \
  --zip CENTERED_OD_FUOI_CONTINUOUS_EDF_CPU_CONFIRM_V2_RESULTS_bundle.zip
```

终端必须打印：

```text
FINAL_ZIP=<absolute path>
FINAL_SHA256=<hash>
ZIP_SIZE=<bytes>
MANIFEST_FILE_COUNT=<count>
PROTOCOL_SHA256=<hash>
SHARED_DATASET_SHA256=<hash>
V1_RESULTS_SHA256=<hash>
CPU_BASELINE_BUNDLE_SHA256=<hash>
GPU_BASELINE_BUNDLE_SHA256=<hash>
VALIDATION_STATUS=PASS
```

---

# 16. 最终报告固定格式

> V2 将 V1 的 absolute-amplitude Urysohn 坐标替换为 centered increment，并将 pointwise GCV 替换为 fold-specific \(\lambda_f(d)\) 映射上的 continuous-EDF blocked-CV。共同稳定 EDF 区间为……；连续风险最低点为 \(d_{\min}=\cdots\)，连续 1-SE 选择为 \(d_{\mathrm{1SE}}=\cdots\)，outer full-training 对应 \(\lambda_{\mathrm{full}}=\cdots\)。Centered coordinate 相对 V1 的 OOD 变化为……。Derived Rank-1、general linear、full Urysohn 和 full Urysohn + PS/AR 的双向及 pooled RMSE 分别为……。因此，估计器状态为 `CENTERED_ESTIMATOR_STABLE / SMOOTHING_SELECTION_UNRESOLVED`，模型状态为 `CENTERED_FULL_URYSOHN_CONFIRMED / CENTERED_URYSOHN_PARETO / CENTERED_URYSOHN_IMPROVES_OLD_K_ONLY / CENTERED_FULL_URYSOHN_REJECTED`。

禁止将连续 EDF 的小数精度解释为物理自由度精确到小数点后两位；它只是平滑算子的有效复杂度坐标。
