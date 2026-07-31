# OD-FUOI-PSAR：完整 Urysohn 面与 NLinear 派生投影 CPU 确认实验方案 V1

> **实验名称**：`OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1`  
> **理论文档**：`OPS_UOI_PS_AR_RAPHU_CUDA_PLC_Three_Layer_Complete_System_v3_2.md`  
> **目标硬件**：32 vCPU，AMD EPYC 9654 96-Core Processor  
> **数值精度**：CPU FP64  
> **环境管理**：`uv`  
> **实验范围**：只测试 v3.2 的完整 Urysohn 面、其唯一线性/非线性投影，以及冻结 K 后的成熟残差 PS/AR  
> **禁止重跑**：LSTM、GRU、TCN、Transformer、AKGNN、shared/private 字典实验  
> **比较方式**：导入已发布旧模型的冻结逐样本预测，在完全一致的 sample IDs 上比较 RMSE/MSE 和 paired block bootstrap

---

# 0. 实验要回答的问题

本实验只回答六个问题：

1. 保留完整二维 Urysohn 面后，输入层 RMSE 能否优于旧 K-only？
2. 完整 Urysohn 面能否达到或超过现有 NLinear-U 逐样本中位集成？
3. full 面中的线性幅值投影是否接近 Rank 1？
4. 纯非线性曲面 \(N_j(\tau,u)\) 是否带来跨棒稳定的增量预测信息？
5. 若 full 面优于 linear 面，增益是否来自同一通道、同一幅值区间和同一时滞结构？
6. 冻结 full K 后，成熟残差 PS/AR 是否仍有稳定增益？

本实验不再测试：

- 公共/私有字典；
- shared/private Rank；
- Rank-2 救援；
- 外挂通道权重；
- 两个手工正则超参数；
- 直接训练 NLinear Rank-1 因子；
- 通过测试集选择 spline 网格。

核心比较链：

\[
\boxed{
\text{derived Rank-1 linear}
\longrightarrow
\text{general linear Urysohn}
\longrightarrow
\text{full Urysohn}
\longrightarrow
\text{full Urysohn + matured residual PS/AR}.
}
\]

---

# 1. 冻结数据与协议

## 1.1 共享数据

必须直接读取：

```text
SHARED_BENCHMARK_DATASET_bundle.zip
SHA256:
fb2e3a3957c2bc4b4dcb30f0c5d425fd6e75068e945aace835e1a3698a6206db
```

禁止：

- 重新读取原始 Excel；
- 重新检测断点；
- 重新生成 target；
- 重新计算 PCA；
- 修改 scaler；
- 修改 sample mask；
- 修改方向；
- 修改 history/horizon/window；
- 依据新模型重选 L6。

## 1.2 冻结 L6

- cadence：10 s；
- history：40 min；
- horizon：20 min；
- future output mean：2 min；
- input shape：240 × 4；
- target：
  \[
  z_t=
  \overline y_{t+20\mathrm{min}}^{(2\mathrm{min})}
  -
  \overline y_t^{(2\mathrm{min})}.
  \]

## 1.3 输入

固定顺序：

1. 联合升速；
2. 主加热功率；
3. 晶转速度；
4. 埚转速度。

联合升速必须直接读取共享 bundle 中冻结后的值，不重新 PCA。

## 1.4 Outer transfer

固定：

\[
\mathrm{Sheet1}\rightarrow\mathrm{Sheet2},
\qquad
\mathrm{Sheet2}\rightarrow\mathrm{Sheet1}.
\]

共同评估样本：

- Sheet1→Sheet2：3233；
- Sheet2→Sheet1：3130；
- pooled：6363。

## 1.5 Inner protocol

每个 outer training rod 内：

- 4 折 expanding-window；
- purge 至少 22 min；
- GCV、basis statistics、幅值支持和 AR 选择只使用训练棒；
- outer test 只评估一次；
- 不允许 random split；
- 不允许测试后修改 basis 或模型结构。

---

# 2. 冻结基线

本轮不重训旧模型，导入其逐样本预测。

## 2.1 输入型基线

| 模型 | pooled RMSE | 说明 |
|---|---:|---|
| Persistence | 0.513061 | 零输入变化 |
| 旧 K-only | 0.412383 | 旧联合升速物理 K |
| Dynamic-PLS | 0.399628 | 传统输入模型 |
| NLinear-U，逐样本种子中位集成 | 0.365854 | 当前最佳输入型参照 |

## 2.2 动态基线

| 模型 | pooled RMSE | 说明 |
|---|---:|---|
| Temporal Autoencoder | 0.384272 | 最佳稳定 GPU 深度动态模型 |
| Joint-K+AR | 0.351642 | 当前总体预测冠军 |

注意：

- full Urysohn 输入层只与输入型榜正式比较；
- full K + residual PS/AR 才进入动态榜；
- NLinear-U 的 0.365854 是已发布逐样本 seed-median ensemble；
- 本轮不重新训练 NLinear。

---

# 3. 待测模型

## 3.1 Full Urysohn

\[
\widehat z_t^{\mathrm{FULL}}
=
b+
\sum_{j=1}^{4}
\sum_{\ell=0}^{239}
K_j(\ell\Delta t,u_{j,t-\ell})\Delta t.
\tag{3.1}
\]

## 3.2 唯一正交分解

训练棒内定义：

\[
\xi_j(u)=\frac{u-\mu_j}{s_j}.
\]

拟合完成后派生：

\[
K_j(\tau,u)
=
\beta_j(\tau)\xi_j(u)
+
N_j(\tau,u),
\tag{3.2}
\]

其中：

\[
\langle N_j(\tau,\cdot),1\rangle_{\nu_j}=0,
\]

\[
\langle N_j(\tau,\cdot),\xi_j\rangle_{\nu_j}=0.
\]

## 3.3 派生模型，不重新训练

### M0：Zero

\[
\widehat z_t^{0}=0.
\]

### M1：Derived Rank-1 Linear Projection

对：

\[
B=[\beta_1,\beta_2,\beta_3,\beta_4]
\]

做 Gram-SVD，保留最佳 Rank 1：

\[
B_1.
\]

由 \(B_1\) 构造预测。不得重新拟合 \(q\) 或通道权重。

### M2：General Linear Urysohn Projection

\[
K_j^{\mathrm{LIN}}(\tau,u)
=
\beta_j(\tau)\xi_j(u).
\]

### M3：Full Urysohn

\[
K_j^{\mathrm{FULL}}
=
K_j^{\mathrm{LIN}}+N_j.
\]

### M4：Full Urysohn + matured residual PS/AR

\[
\widehat z_t^{\mathrm{FINAL}}
=
\widehat z_t^{\mathrm{FULL}}
+
\widehat{\mathcal A}(R_{t,\mathrm{mature}}^-).
\]

M1、M2、M3 共用同一个 full Urysohn 拟合结果；M1 和 M2 不重新训练。

---

# 4. 固定数值坐标

## 4.1 时滞 basis

固定 \(T_{\max}=40\) min。

使用 cubic B-spline，先将时滞映射为：

\[
s=\sqrt{\tau/T_{\max}}\in[0,1].
\]

在 \(s\) 上使用开区间均匀 cubic B-spline：

```text
number_of_basis = 20
degree = 3
boundary = [0, 1]
```

该 square-root warp 只用于在短时滞区分配更高数值分辨率，不作为待调参数。

## 4.2 幅值 basis

对每个通道，使用 outer training rod 的分位数 knots：

```text
0%, 5%, 15%, 30%, 50%, 70%, 85%, 95%, 100%
```

使用 cubic B-spline：

```text
degree = 3
```

重复分位点必须合并；若唯一 knot 数不足，退化为自然 cubic spline 或线性子空间，并记录：

```text
AMPLITUDE_SUPPORT_DEGENERATE
```

## 4.3 Basis 不搜索

本轮只有这一套正式 basis。

额外 mesh-refinement 只允许作为数值审计：

```text
lag basis = 28
amplitude quantiles = 0, 2, 10, 20, 35, 50, 65, 80, 90, 98, 100%
```

refinement 结果不能用于选择主模型，只用于判断：

```text
BASIS_RESOLUTION_STABLE
or
BASIS_RESOLUTION_INSUFFICIENT
```

正式 RMSE 来自主 basis。

---

# 5. 单一自动平滑估计

## 5.1 目标

在归一化坐标中：

\[
\min_{b,K_1,\ldots,K_4}
\frac1n
\sum_t
\left[
z_t-b-\sum_j\mathcal K_j[u_j](t)
\right]^2
+
\lambda
\sum_j
\|K_j\|_{\mathcal H_j^\star}^2.
\tag{5.1}
\]

其中：

\[
\|K_j\|_{\mathcal H_j^\star}^2
=
\|K_j\|_{L_2}^2
+
\int
\left[
K_{ss}^2+2K_{sv}^2+K_{vv}^2
\right].
\tag{5.2}
\]

## 5.2 只有一个自动尺度

\[
\widehat\lambda
=
\arg\min_{\lambda>0}\operatorname{GCV}(\lambda).
\]

实现：

- 在 \(\log_{10}\lambda\in[-12,12]\) 上先做 25 点 deterministic bracket；
- 再用 bounded Brent 搜索；
- 不把网格点作为候选模型；
- 不使用 test；
- 不使用 random seed；
- 不使用通道特异 penalty；
- 不使用 lag/amplitude 两个权重。

## 5.3 规范约束

常数幅值截面吸收进总截距。

每个 \(N_j\) 强制：

\[
N_j\perp1,
\qquad
N_j\perp\xi_j
\]

于训练经验幅值测度下。

## 5.4 数值输出

每个 outer direction 必须保存：

- \(\widehat\lambda\)；
- GCV curve；
- effective degrees of freedom；
- KKT residual；
- condition number；
- coefficient hash；
- prediction hash；
- basis Gram eigenvalues；
- surface files。

---

# 6. 完整执行阶段

## Stage E0：预检和旧基线导入

必须验证：

1. shared bundle SHA256；
2. CPU/GPU baseline bundle SHA256；
3. sample IDs；
4. target hash；
5. direction masks；
6. old metrics 重算误差小于 \(10^{-10}\)；
7. NLinear seed-median prediction 已冻结；
8. 原始 Excel 不进入结果目录。

失败即停止。

## Stage E1：主 basis 与设计矩阵

生成：

- lag basis；
- amplitude basis；
- tensor design；
- Sobolev penalty；
- orthogonality constraints；
- common-support mask；
- OOD mask。

检查：

- basis partition-of-unity；
- Gram 正定性；
- constraint residual；
- no-future-input；
- no-break crossing。

## Stage E2：Full Urysohn FP64 fit

对两个 outer directions：

1. 在训练棒内 GCV 选 \(\lambda\)；
2. FP64 求唯一正则解；
3. 输出 OOF 预测；
4. 在 outer test 一次评估；
5. 保存所有 \(K_j\) 面。

## Stage E3：幅值正交分解

从同一个 full fit 计算：

\[
\beta_j,\qquad N_j.
\]

必须通过：

\[
\max_{\tau}
|\langle N_j,1\rangle|
<10^{-10},
\]

\[
\max_{\tau}
|\langle N_j,\xi_j\rangle|
<10^{-10}.
\]

重构误差：

\[
\|K_j-\beta_j\xi_j-N_j\|/\|K_j\|
<10^{-10}.
\]

## Stage E4：NLinear 派生 Rank-1 审计

计算：

- \(B\)；
- Gram-SVD；
- singular values；
- \(\rho_{\mathrm{NLinear}}\)；
- derived rank-1 prediction；
- rank-1 vs general-linear paired errors；
- two-direction time-shape correlation；
- channel-coordinate sign agreement；
- spectral gap。

不允许：

- 再训练 \(q,a\)；
- 多随机种子；
- Rank 2 救援；
- test-based Rank selection。

## Stage E5：Linear vs Full Urysohn

比较 M2 和 M3：

\[
\Delta_t
=
e_{\mathrm{LIN},t}^2
-
e_{\mathrm{FULL},t}^2.
\]

正式 nonlinear increment 证据要求：

1. 两方向 MSE 都不恶化；
2. pooled 40-min block bootstrap 正改善概率不低于 95%；
3. 22/40/60 min block sensitivity 结论一致；
4. 非线性面 common-support correlation 不低于 0.6；
5. time-shift placebo 破坏增益；
6. OOD 区不是主要增益来源。

若不满足，只允许声明：

```text
FULL_SURFACE_PREDICTIVE_GAIN_NOT_CERTIFIED
```

不能声明真实过程线性或非线性不存在。

## Stage E6：通道贡献审计

不做 support 搜索，不训练通道权重。

对每个通道报告：

1. \(\|\beta_j\|^2\)；
2. \(\|N_j\|^2\)；
3. leave-one-channel-out prediction delta；
4. 两方向 \(\beta_j\) correlation；
5. 两方向 \(N_j\) common-support correlation；
6. sign/peak/centroid；
7. 40-min paired bootstrap；
8. shift placebo；
9. OOD 占比。

通道结论只分为：

```text
CHANNEL_SURFACE_CERTIFIED
PREDICTIVE_ONLY
UNRESOLVED
```

不因一个阈值直接修改正式模型系数为零。

## Stage E7：C1 延拓

对 \(N_j\) 实现有限带 cubic Hermite 延拓。

测试：

- 函数值连续；
- 一阶导连续；
- extension band；
- saturation；
- total K 的 C1；
- extension contribution；
- OOD fraction。

主 RMSE 同时报告：

- all registered samples；
- fit/common-support only；
- extension-band only；
- saturated only。

## Stage E8：Matured residual PS/AR

使用 E2 的 rolling OOF full-K residual。

候选：

```text
A0: exact zero
A1: ridge AR, 2 min
A2: ridge AR, 5 min
A3: ridge AR, 10 min
A4: ridge AR, 20 min
A5: ridge AR, 40 min
A6: stable state-space, dimension 2
A7: stable state-space, dimension 4
```

所有残差必须满足：

\[
s+h+W\le t.
\]

选择：

- 训练棒内 OOF；
- one-SE 优先 A0；
- test 不参与；
- 不再训练深度 residual 模型。

## Stage E9：统一比较

生成两个榜：

### Input-only

- Persistence；
- old K-only；
- Dynamic-PLS；
- NLinear-U seed-median；
- M1 derived rank-1 linear；
- M2 general linear Urysohn；
- M3 full Urysohn。

### Dynamic

- Temporal AE；
- Joint-K+AR；
- M4 full Urysohn + residual PS/AR。

---

# 7. 评价指标

## 7.1 主指标

必须报告：

- Sheet1→Sheet2 MSE/RMSE/MAE/\(R^2\)；
- Sheet2→Sheet1；
- pooled；
- relative Persistence；
- relative old K；
- relative NLinear；
- relative Joint-K+AR。

Pooled RMSE：

\[
\mathrm{RMSE}_{\mathrm{pooled}}
=
\sqrt{
\frac{
\sum_d\sum_{t\in d}(y_t-\widehat y_t)^2
}{
\sum_dn_d
}}.
\]

## 7.2 Paired moving-block bootstrap

对同一 sample ID：

\[
\Delta_t=e_{A,t}^2-e_{B,t}^2.
\]

```text
replicates = 1000
block lengths = 10, 22, 40, 60 min
primary block = 40 min
seed = 20260731
```

输出：

- median relative MSE improvement；
- 95% interval；
- positive probability；
- direction-specific；
- pooled；
- block sensitivity。

## 7.3 核结构指标

### 线性投影

- \(\rho_{\mathrm{NLinear}}\)；
- singular values；
- spectral gap；
- time-shape correlation；
- channel-coordinate sign agreement；
- \(\beta_j\) peak/centroid/support。

### 非线性面

- \(\|N_j\|^2/\|K_j\|^2\)；
- common-support surface correlation；
- first nonlinear singular spectrum；
- amplitude-response sections；
- lag-response sections；
- OOD dependence。

### 数值

- GCV \(\lambda\)；
- effective df；
- KKT；
- condition number；
- mesh-refinement delta；
- coefficient/prediction hashes。

---

# 8. 预注册判断

## 8.1 NLinear 结构状态

### `NLINEAR_PROJECTION_SUPPORTED`

同时满足：

1. 两方向 \(\rho_{\mathrm{NLinear}}\ge0.90\)；
2. 两方向 Rank-1 time shape 绝对相关不低于 0.7；
3. channel-coordinate 符号一致率不低于 75%；
4. derived Rank-1 projection 与 general linear Urysohn pooled MSE 差异不超过 2%；
5. 两方向 derived Rank-1 均优于 Persistence。

允许声明：

> full Urysohn 面中的线性幅值投影近似具有共享 Rank-1 时滞结构。

禁止声明 plant kernel 已证明。

### `NLINEAR_PREDICTIVE_ONLY`

现有 GPU NLinear 仍然优秀，但 full 面派生线性投影未通过上述结构门禁。

## 8.2 Full Urysohn 成功等级

### Level A：`FULL_URYSOHN_CONFIRMED`

同时满足：

1. M3 pooled RMSE 小于 0.365854；
2. 两方向均优于 Persistence；
3. M3 vs old K 的 40-min bootstrap 正改善概率不低于 95%；
4. M3 vs NLinear 的 pooled bootstrap 正改善概率不低于 90%；
5. mesh refinement 结论稳定；
6. OOD 不主导；
7. KKT/约束/C1 全部通过。

### Level B：`URYSOHN_PARETO_EQUIVALENT`

1. M3 与 NLinear RMSE 相差不超过 1%；
2. 两方向均为正；
3. full 面分解稳定；
4. 可给出 linear/nonlinear 结构解释；
5. 数值和 OOD 通过。

### Level C：`URYSOHN_IMPROVES_OLD_K_ONLY`

1. M3 优于旧 K-only；
2. 但未达到 NLinear；
3. 或 full/nonlinear 结构证书不完整。

### Level D：`FULL_URYSOHN_REJECTED_ON_CURRENT_DATA`

任一：

- 不优于旧 K-only；
- 一个方向严重恶化；
- GCV 解不稳定；
- mesh refinement 结论翻转；
- common-support 面完全不一致；
- placebo 不破坏；
- OOD 主导增益。

## 8.3 非线性状态

### `NONLINEAR_INCREMENT_CERTIFIED`

M3 相对 M2 满足 E5 全部门禁。

### `NONLINEAR_INCREMENT_NOT_CERTIFIED`

full 面被拟合，但 M3 相对 M2 没有稳定增益。

注意：

> `NOT_CERTIFIED` 不等于真实系统没有非线性。

## 8.4 Residual 状态

### `RESIDUAL_EXACT_ZERO`

两方向 one-SE 都选择 A0，或 pooled 增益不稳定。

### `MATURED_RESIDUAL_PREDICTIVE_GAIN`

至少一个非零残差模型在两方向均不恶化，并通过 paired bootstrap。

---

# 9. 主要输出表

## 9.1 模型表

| ID | 来源 | 重新训练 | 说明 |
|---|---|---:|---|
| PERSIST | baseline | 否 | zero-change |
| OLD-K | baseline | 否 | old K-only |
| DPLS | baseline | 否 | Dynamic-PLS |
| NLINEAR-GPU | baseline | 否 | seed-median ensemble |
| R1-LIN-DERIVED | full K 派生 | 否 | \(\widehat B\) 最佳 Rank-1 |
| LIN-UOI | full K 派生 | 否 | \(\widehat\beta_j\xi_j\) |
| FULL-UOI | 主模型 | 是 | \(\widehat\beta_j\xi_j+\widehat N_j\) |
| FULL-UOI-PSAR | 主模型 + residual | 是 | final dynamic |

## 9.2 必须生成的图

1. 两方向和 pooled RMSE；
2. full vs linear vs rank1 error deltas；
3. GCV curve；
4. 每通道完整 Urysohn surface；
5. 每通道 \(\beta_j(\tau)\)；
6. 每通道 \(N_j(\tau,u)\)；
7. \(B\) singular spectrum；
8. 两方向 Rank-1 time shapes；
9. channel coordinates；
10. nonlinear energy ratio；
11. common-support surface comparison；
12. C1 extension sections；
13. OOD contribution；
14. bootstrap intervals；
15. residual AR ablation；
16. mesh-refinement sensitivity。

---

# 10. CPU 并行与资源

## 10.1 环境

```bash
uv sync --dev
```

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## 10.2 并行

建议：

```text
outer/fold jobs = 12
bootstrap jobs = 16
BLAS threads per process = 1
dtype = float64
```

GCV 的单方向求解可复用：

- eigendecomposition；
- Cholesky factor；
- trace identities。

## 10.3 预计资源

- 32 vCPU 足够；
- 建议内存 64 GB 以上；
- 不需要 GPU；
- 预计 6–18 h；
- mesh refinement 可单独追加；
- 每个 stage 必须 checkpoint；
- 失败 stage 不得伪造后续结果。

---

# 11. 推荐目录

```text
OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1/
├── configs/
│   └── frozen_protocol.yaml
├── src/
│   ├── shared_data/
│   ├── tensor_spline/
│   ├── sobolev_geometry/
│   ├── gcv/
│   ├── full_urysohn/
│   ├── amplitude_projection/
│   ├── nlinear_audit/
│   ├── c1_extension/
│   ├── matured_residual/
│   ├── evaluation/
│   └── packaging/
├── scripts/
│   ├── preflight.py
│   ├── import_frozen_baselines.py
│   ├── build_design.py
│   ├── run_full_urysohn.py
│   ├── derive_amplitude_projection.py
│   ├── run_nlinear_audit.py
│   ├── run_nonlinear_increment_audit.py
│   ├── run_c1_audit.py
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

# 12. 必须通过的测试

1. shared bundle SHA256；
2. baseline sample-ID 完全一致；
3. target hash；
4. no future input；
5. no break crossing；
6. PCA/scaler 隔离；
7. basis partition of unity；
8. lag warp 单调；
9. tensor design shape；
10. Sobolev penalty 对称正定；
11. GCV deterministic；
12. full fit KKT；
13. amplitude projection reconstruction；
14. \(N_j\perp1\)；
15. \(N_j\perp\xi_j\)；
16. Gram-SVD basis invariance；
17. derived Rank-1 不重新训练；
18. C1 value continuity；
19. C1 derivative continuity；
20. matured residual causality；
21. paired bootstrap alignment；
22. mesh-refinement comparison；
23. FP64 prediction；
24. package privacy；
25. manifest/hash/ZIP roundtrip。

任何数据因果、投影正交、KKT 或 sample-ID 测试失败，正式结果无效。

---

# 13. 启动命令

```bash
cd OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1

bash RUN_CPU_CONFIRM.sh \
  --shared /path/to/SHARED_BENCHMARK_DATASET_bundle.zip \
  --cpu-baselines /path/to/PHYSICS_FIRST_CPU_RESULTS_bundle.zip \
  --gpu-baselines /path/to/PHYSICS_FIRST_GPU_RESULTS_bundle.zip \
  --protocol configs/frozen_protocol.yaml \
  --n-jobs 12 \
  --bootstrap-jobs 16 \
  2>&1 | tee logs/full_run.log
```

断点恢复：

```bash
bash RESUME_CPU_CONFIRM.sh \
  --checkpoint results/checkpoints/latest.json
```

---

# 14. 结果文件

```text
results/
├── PRECHECK_REPORT.md
├── DESIGN_AUDIT.json
├── GCV_RESULTS.csv
├── FULL_URYSOHN_METRICS.csv
├── AMPLITUDE_PROJECTION_METRICS.csv
├── NLINEAR_DERIVED_AUDIT.csv
├── NONLINEAR_INCREMENT_AUDIT.csv
├── CHANNEL_SURFACE_AUDIT.csv
├── C1_EXTENSION_AUDIT.csv
├── RESIDUAL_PSAR_RESULTS.csv
├── FINAL_INPUT_LEADERBOARD.csv
├── FINAL_DYNAMIC_LEADERBOARD.csv
├── PAIRWISE_BOOTSTRAP.csv
├── FINAL_DECISION.json
├── FINAL_REPORT.md
├── predictions/
├── surfaces/
├── projections/
├── spectra/
├── ood/
├── diagnostics/
└── plots/
```

`FINAL_DECISION.json` 至少包含：

```json
{
  "registration": "FULL_URYSOHN_CONFIRMED | URYSOHN_PARETO_EQUIVALENT | URYSOHN_IMPROVES_OLD_K_ONLY | FULL_URYSOHN_REJECTED_ON_CURRENT_DATA",
  "nlinear_projection_status": "NLINEAR_PROJECTION_SUPPORTED | NLINEAR_PREDICTIVE_ONLY",
  "nonlinear_increment_status": "NONLINEAR_INCREMENT_CERTIFIED | NONLINEAR_INCREMENT_NOT_CERTIFIED",
  "residual_status": "RESIDUAL_EXACT_ZERO | MATURED_RESIDUAL_PREDICTIVE_GAIN",
  "gcv_lambda_by_direction": {},
  "effective_df_by_direction": {},
  "rank1_energy_ratio_by_direction": {},
  "direction_metrics": {},
  "pooled_metrics": {},
  "comparison_vs_old_k": {},
  "comparison_vs_nlinear": {},
  "comparison_vs_joint_k_ar": {},
  "ood_summary": {},
  "numerical_certification": {},
  "scientific_claims_allowed": [],
  "scientific_claims_forbidden": []
}
```

---

# 15. 最终打包合同

清理旧输出：

```bash
rm -rf return/OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS
rm -f return/OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS_bundle.zip
rm -f return/OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS_bundle.zip.sha256
```

构建：

```bash
uv run python scripts/build_return_bundle.py \
  --source-root . \
  --results results \
  --output return/OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS
```

隐私与内容校验：

```bash
uv run python scripts/validate_package.py \
  --package-dir return/OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS \
  --forbid "*.xlsx,.git,__pycache__,cache,*.tmp,raw_data"
```

生成 manifest/hash：

```bash
uv run python scripts/build_manifest.py \
  --root return/OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS \
  --output return/OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS/MANIFEST.json
```

压缩：

```bash
cd return
zip -r OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS_bundle.zip \
  OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS
```

SHA256：

```bash
sha256sum OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS_bundle.zip \
  > OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS_bundle.zip.sha256
```

回环校验：

```bash
uv run python ../scripts/validate_zip_roundtrip.py \
  --zip OD_FUOI_NLINEAR_PROJECTION_CPU_CONFIRM_V1_RESULTS_bundle.zip
```

最终必须打印：

```text
FINAL_ZIP=<absolute path>
FINAL_SHA256=<hash>
ZIP_SIZE=<bytes>
MANIFEST_FILE_COUNT=<count>
PROTOCOL_SHA256=<hash>
SHARED_DATASET_SHA256=<hash>
CPU_BASELINE_BUNDLE_SHA256=<hash>
GPU_BASELINE_BUNDLE_SHA256=<hash>
VALIDATION_STATUS=PASS
```

---

# 16. 最终报告固定结论格式

> 在冻结 L6、双向跨晶棒协议下，本轮直接拟合了完整多通道 Urysohn 面，并在训练幅值测度下唯一分解为线性幅值投影 \(\beta_j(\tau)\xi_j(u)\) 与纯非线性曲面 \(N_j(\tau,u)\)。由线性投影矩阵派生的 Rank-1 能量比例分别为……，其结构状态为……。Derived Rank-1、general linear Urysohn、full Urysohn 和 full Urysohn + matured residual PS/AR 的双向与 pooled RMSE 分别为……。相对 old K-only、NLinear-U 和 Joint-K+AR 的 paired block-bootstrap 结果为……。非线性增量状态为……，通道曲面状态为……，C1/OOD 状态为……。因此，本轮登记为 `FULL_URYSOHN_CONFIRMED / URYSOHN_PARETO_EQUIVALENT / URYSOHN_IMPROVES_OLD_K_ONLY / FULL_URYSOHN_REJECTED_ON_CURRENT_DATA`。

禁止仅凭 pooled RMSE、Rank-1 能量或单棒曲面形状宣称开放环物理机制。
