# OPS-UOI Shared–Private Orthogonal Spectral Urysohn K：CPU 单体系确认实验方案 V1

> **实验名称**：`OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1`  
> **目标硬件**：32 vCPU，AMD EPYC 9654 96-Core Processor  
> **运行方式**：CPU FP64，`uv` 管理环境  
> **理论依据**：`OPS_UOI_PS_AR_RAPHU_CUDA_PLC_Three_Layer_Complete_System_v3_1.md`  
> **实验范围**：只测试最新版 Shared–Private Orthogonal Spectral Urysohn K 体系及其内部消融，不重跑完整 CPU/GPU 模型池  
> **比较方式**：复用已经冻结的 CPU/GPU 逐样本预测与指标，仅比较相同 sample IDs 上的 RMSE/MSE 和配对 bootstrap  
> **正式主任务**：冻结 L6，40 min 历史、20 min horizon、2 min 输出窗口、10 s cadence

---

# 0. 实验目的

本实验只回答以下五个问题：

1. NLinear 暴露出的**多通道共享时滞 Rank-1**能否被一个 FP64、凸求解、可认证的 Urysohn K 正式恢复？
2. 共享 Rank 是否需要从 \(R_s=1\) 增加到 \(R_s=2\)？
3. 冻结公共子空间后，是否存在一个跨棒稳定、与公共正交的**通道私有 Rank-1**？
4. 在线性 shared/private 支持上，额外幅值非线性是否仍然选择精确零？
5. 冻结物理 K 后，成熟残差 Predictive-State/AR 是否仍然选择精确零？

本实验不再回答：

- LSTM、Transformer、AKGNN 等模型谁更优；
- 是否需要继续扩大深度模型；
- 当前核是否为开放环 plant kernel；
- 未注册通道是否“没有物理作用”。

核心目标为：

\[
\boxed{
\text{用同一冻结协议，确认 shared/private K 是否优于旧 K-only 和 NLinear。}
}
\]

---

# 1. 冻结数据与目标

## 1.1 必须复用的共享数据

实验必须直接读取已经发布的：

```text
SHARED_BENCHMARK_DATASET_bundle.zip
SHA256:
fb2e3a3957c2bc4b4dcb30f0c5d425fd6e75068e945aace835e1a3698a6206db
```

禁止重新：

- 读取原始 Excel 并生成新样本；
- 检测断点；
- 计算 PCA；
- 计算 scaler；
- 生成目标；
- 修改 sample mask；
- 修改外层方向；
- 修改 cadence、history、horizon 或 output window。

GPU 和 CPU 旧结果只允许在 protocol、split、target 和 sample-ID hash 全部一致时进入比较。

## 1.2 输入与输出

四个冻结控制输入：

1. 联合升速；
2. 主加热功率；
3. 晶转速度；
4. 埚转速度。

输出为晶体直径，预测目标固定为：

\[
z_t=
\overline y_{t+20\mathrm{min}}^{(2\mathrm{min})}
-
\overline y_t^{(2\mathrm{min})}.
\tag{1.1}
\]

恢复未来平均直径：

\[
\widehat{\overline y}_{t+20}^{(2)}
=
\overline y_t^{(2)}+\widehat z_t.
\tag{1.2}
\]

## 1.3 外层验证

固定两个 outer transfer：

\[
\mathrm{Sheet1}\rightarrow\mathrm{Sheet2},
\qquad
\mathrm{Sheet2}\rightarrow\mathrm{Sheet1}.
\tag{1.3}
\]

最终共同评估 mask：

- Sheet1→Sheet2：3233 rows；
- Sheet2→Sheet1：3130 rows；
- pooled：6363 rows。

## 1.4 内层选择

每个 outer training rod 内使用：

- 4 折 expanding-window；
- purge 至少 22 min；
- 所有字典、正则、Rank、支持和 AR 阶数仅由训练棒内 OOF 选择；
- test rod 只用于一次 outer evaluation；
- 不允许随机时间划分。

---

# 2. 只比较哪些旧结果

本轮不重训旧模型，只导入其冻结逐样本预测。

## 2.1 输入型排行榜参照

| 模型 | pooled MSE | pooled RMSE | 本轮用途 |
|---|---:|---:|---|
| Persistence / zero-change | 0.263231 | 0.513061 | 基础参照 |
| 旧单通道 K-only | 0.170060 | 0.412383 | 必须超过的旧物理模型 |
| Dynamic-PLS | 0.159703 | 0.399628 | 最佳传统输入模型 |
| NLinear-U | 0.156264 | 0.395302 | 本轮最关键结构参照 |

## 2.2 动态排行榜参照

| 模型 | pooled MSE | pooled RMSE | 本轮用途 |
|---|---:|---:|---|
| Temporal Autoencoder | 0.147665 | 0.384272 | 最佳 GPU 动态模型 |
| Joint-K+AR | 0.123652 | 0.351642 | 当前总体预测冠军 |

注意：

- `Joint-K+AR` 是预测性能参照，不是物理 K 稳定性证据；
- 输入型 shared/private K 不能与允许历史直径的动态模型混榜；
- 最终 `K→Residual PS/AR` 才进入动态榜。

---

# 3. 最新待测模型

对第 \(j\) 个通道：

\[
K_j(\tau,u)
=
K_j^s(\tau,u)+K_j^p(\tau,u).
\tag{3.1}
\]

公共部分：

\[
K_j^s(\tau,u)
=
\sum_{r=1}^{R_s}
\sigma_r q_r(\tau)f_{j,r}(u).
\tag{3.2}
\]

私有部分：

\[
K_j^p(\tau,u)
=
\sum_{\ell=1}^{R_{p,j}}
\sigma_{j,\ell}^p
p_{j,\ell}(\tau)g_{j,\ell}(u),
\tag{3.3}
\]

并强制：

\[
\langle q_r,p_{j,\ell}\rangle_{G_\tau}=0.
\tag{3.4}
\]

当前实验只允许：

\[
R_s\in\{0,1,2\},
\qquad
R_{p,j}\in\{0,1\},
\qquad
\sum_jR_{p,j}\le1.
\tag{3.5}
\]

即：

- 公共 Rank 最多 2；
- 每个通道私有 Rank 最多 1；
- 整个最终模型最多开放一个私有通道；
- 不允许多个私有分支组合搜索；
- 不允许私有 Rank-1 失败后使用 Rank-2 救援。

正式预测：

\[
\widehat z_t
=
\sum_j
\left(
\widehat{\mathcal K}_j^s+
\widehat{\mathcal K}_j^p
\right)[u_j^-]
+
\widehat{\mathcal A}
\left(R_{t,\mathrm{mature}}^-\right).
\tag{3.6}
\]

Residual PS/AR 包含 exact-zero 候选。

---

# 4. 冻结母空间

## 4.1 时滞范围

固定：

\[
\tau\in[0,40\ \mathrm{min}].
\tag{4.1}
\]

本轮不能声明发现 40 min 之外的功率或热过程。

若 40 min 边界附近仍有显著核能量，只登记：

```text
MOTHER_SUPPORT_OR_TMAX_INSUFFICIENT
```

不得通过增加 Rank 掩盖时滞窗不足。

## 4.2 两个嵌套 cubic B-spline 母空间

只比较两个预注册母空间，不做任意 knot 搜索。

### \(V_0\)：基础空间

三次 B-spline，边界为 0 和 40 min，内部 knots：

```text
0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30 min
```

约 14 个时滞基函数。

### \(V_1\)：局部加密空间

在 \(V_0\) 上增加：

```text
0.25, 0.75, 1.5, 3, 5, 7, 9, 12, 18, 25, 35 min
```

约 25 个时滞基函数。

约束：

\[
V_0\subset V_1.
\tag{4.2}
\]

## 4.3 母空间选择

只看训练棒内 OOF：

1. 若 \(V_1\) 相对 \(V_0\) 的 OOF MSE 改善不足 1%，one-SE 选择 \(V_0\)；
2. 若改善大于等于 1%，且新增基函数在折间稳定，选择 \(V_1\)；
3. 若 \(V_1\) 仍在 40 min 尾部有集中能量，登记支持不足，不扩大 \(T_{\max}\)；
4. outer test 不能决定母空间。

---

# 5. 正则化与 full convex K

## 5.1 线性优先

第一阶段固定：

\[
f_{j,r}(u)=u,
\qquad
g_{j,\ell}(u)=u.
\tag{5.1}
\]

先拟合多通道 full linear K：

\[
\widehat z_t
=
\sum_{j=1}^{4}
\sum_{a=1}^{M_\tau}
\theta_{j,a}C_a(\tau)u_{j,t-\tau}.
\tag{5.2}
\]

目标函数：

\[
\min_{\Theta}
\frac{1}{n}\|z-\Phi\Theta\|_2^2
+
\lambda_0\|\Theta\|_2^2
+
\lambda_2\sum_j\|D_\tau^2\theta_j\|_2^2.
\tag{5.3}
\]

所有求解使用 CPU FP64。

## 5.2 exact-zero 通道支持

由于只有 4 个输入，直接枚举 16 个通道支持子集：

\[
S\subseteq
\{\mathrm{lift,power,crystal\ rot,crucible\ rot}\}.
\tag{5.4}
\]

包括空集 \(S=\varnothing\)。

这样可获得严格的通道 exact-zero，而不依赖不稳定的 group-lasso 阈值。

## 5.3 正则网格

基础 ridge：

```text
lambda_0 ∈ 10^{[-6,-4,-2,0,2,4,6]}
```

二阶时滞平滑：

```text
lambda_2 ∈ {0, 1e-4, 1e-2, 1, 1e2, 1e4}
```

选择顺序：

1. 最低 OOF MSE；
2. one-SE 内优先更少通道；
3. 同支持下优先更小母空间；
4. 再优先更强平滑；
5. 最后才优先更弱 ridge。

---

# 6. 公共 Rank 实验

## 6.1 Gram 白化

对 full K 的时滞系数矩阵做：

\[
\widetilde\Theta
=
G_\tau^{1/2}\Theta.
\tag{6.1}
\]

SVD：

\[
\widetilde\Theta
=
U\Sigma V^\top.
\tag{6.2}
\]

公共时滞基：

\[
q_r(\tau)
=
C(\tau)^\top G_\tau^{-1/2}u_r.
\tag{6.3}
\]

## 6.2 候选

只比较：

```text
S0: Rs = 0
S1: Rs = 1
S2: Rs = 2
```

每个 Rank 下重新做 fixed-subspace convex refit，不直接使用截断系数作为最终模型。

## 6.3 公共参与门禁

第 \(r\) 个模态的通道参与率：

\[
\pi_{j,r}
=
\frac{a_{j,r}^2}{\sum_k a_{k,r}^2},
\qquad
N_{\mathrm{part},r}
=
\frac{1}{\sum_j\pi_{j,r}^2}.
\tag{6.4}
\]

进入公共候选需同时满足：

1. \(N_{\mathrm{part},r}\ge1.8\)；
2. 至少两个通道各自贡献不低于 10%；
3. leave-one-channel-out 后，时间模态与原模态绝对相关不低于 0.6；
4. 在 4 个内层训练折中 principal-angle 中位数不大于 35°；
5. 增加该公共 Rank 的 OOF MSE 改善至少 0.5%；
6. one-SE 不选择更小 Rank。

## 6.4 公共 Rank 的物理认证

两个 outer 方向完成后再检查：

- 两方向是否选择相同 \(R_s\)，或 one-SE Rank 区间是否有交集；
- 两方向公共子空间最大 principal angle；
- 时间模态支持重叠；
- 通道载荷符号与大小；
- time-shift placebo 是否破坏效果。

若预测改善但子空间不一致，只登记：

```text
PREDICTIVE_SHARED_LOW_RANK
NOT_SHARED_K_CERTIFIED
```

---

# 7. 私有 Rank 实验

## 7.1 冻结公共

先冻结选中的公共投影：

\[
P_s=Q_sQ_s^\top G_\tau.
\tag{7.1}
\]

对第 \(j\) 个通道 full K 计算：

\[
E_j=(I-P_s)\widetilde\theta_j.
\tag{7.2}
\]

私有模态只能来自 \(E_j\)，不得重新对原始 K 自由分解。

## 7.2 只测试五个候选

```text
P0: 无私有分支
P1-lift: 联合升速 private Rank-1
P1-power: 功率 private Rank-1
P1-crystal-rot: 晶转 private Rank-1
P1-crucible-rot: 埚转 private Rank-1
```

不测试任意两个私有分支同时开启。

## 7.3 私有进入门禁

某通道私有 Rank-1 必须同时满足：

1. 与公共子空间数值正交误差小于 \(10^{-10}\)；
2. 私有残差能量至少 70% 集中在该通道；
3. 4 个内层折私有模态绝对相关中位数不低于 0.6；
4. 内层 conditional OOF MSE 改善至少 0.5%；
5. one-SE 不选择 private zero；
6. 两个 outer 方向增益均不为负；
7. 两方向私有模态绝对相关不低于 0.6；
8. 40 min block bootstrap 正改善概率不低于 90%；
9. time-shift placebo 后增益消失；
10. 不能通过增加一个公共 Rank 更经济地解释。

任一条件失败：

```text
PRIVATE_EXACT_ZERO
```

若只在一根棒或一个方向出现：

```text
ROD_SPECIFIC_OR_UNRESOLVED
```

不得进入物理 K。

---

# 8. 非线性幅值扩展

## 8.1 触发条件

只有最终线性 shared/private 模型已经满足：

- 两方向相对 Persistence 不为负；
- pooled paired bootstrap 为正；
- 公共/私有子空间数值认证通过；

才进入非线性阶段。

## 8.2 幅值基

每个已激活通道使用训练棒内分位数构造 cubic B-spline：

```text
5%, 20%, 40%, 60%, 80%, 95%
```

线性空间 \(\operatorname{span}\{1,u\}\) 从非线性基中 Gram 正交移除，使：

\[
K^{\mathrm{LIN}}
\subset
K^{\mathrm{LIN+NL}}.
\tag{8.1}
\]

非线性增量具有 exact-zero 候选。

## 8.3 防止容量再次膨胀

本轮最多允许一个通道启用 nonlinear increment。

候选：

```text
NL0: 全部非线性精确零
NL-lift
NL-power
NL-crystal-rot
NL-crucible-rot
```

只在已激活线性支持上测试。

若非线性候选 one-SE 未超过 NL0，登记：

```text
NONLINEAR_EXACT_ZERO
```

## 8.4 \(C^1\) 延拓

测试棒幅值超出训练支持时，必须使用 v3.1 中冻结的有限带三次 Hermite \(C^1\) 延拓。

必须记录：

- 每通道训练支持上下界；
- 测试 OOD 比例；
- 延拓带宽；
- 延拓贡献的预测比例；
- 超出有限带后的安全策略。

不得使用硬 clamp 代替正式结果。

---

# 9. Residual Predictive-State/AR

## 9.1 OOF 残差

使用 rolling cross-fit 的冻结 K 预测：

\[
r_s^{\mathrm{OOF}}
=
z_s-\widehat K^{\mathrm{OOF}}(U_s^-).
\tag{9.1}
\]

预测时刻 \(t\) 只允许使用满足：

\[
s+h+W\le t
\tag{9.2}
\]

的成熟残差。

## 9.2 候选

```text
A0: exact zero
A1: ridge AR, history 2 min
A2: ridge AR, history 5 min
A3: ridge AR, history 10 min
A4: ridge AR, history 20 min
A5: ridge AR, history 40 min
A6: stable linear state-space, state dimension 2
A7: stable linear state-space, state dimension 4
```

只在训练棒内选择。

one-SE 默认优先 A0。

本轮不再训练 GRU、TCN 或 Transformer residual，因为它们已经在 GPU benchmark 中失败。

---

# 10. 完整实验矩阵

## Stage E0：协议与旧结果导入

目标：

- 校验 shared bundle hash；
- 校验 CPU/GPU 旧预测 sample IDs；
- 重算所有旧模型 RMSE；
- 保证与已发布数字误差小于 \(10^{-10}\)。

失败则停止。

## Stage E1：母空间审计

运行：

```text
V0 full linear K
V1 full linear K
```

只在内层决定 \(V_\tau^\star\)。

## Stage E2：full multichannel linear K

在 \(V_\tau^\star\) 上：

- 16 个通道支持子集；
- 7 个 ridge；
- 6 个 smoothness；
- 两个 outer 方向；
- 4 个 inner folds。

该阶段用于得到 full K 和 exact-zero 支持，不作为最终低秩模型。

## Stage E3：公共 Rank

比较：

```text
Rs = 0, 1, 2
```

输出每个 Rank 的：

- OOF MSE；
- outer MSE/RMSE；
- participation；
- singular values；
- principal angles；
- leave-one-channel-out；
- support overlap。

## Stage E4：单私有 Rank-1

在冻结 \(R_s^\star\) 后比较：

```text
P0
P1-lift
P1-power
P1-crystal-rot
P1-crucible-rot
```

最终 \(\sum_jR_{p,j}\le1\)。

## Stage E5：fixed-subspace FP64 refit

冻结：

- 母空间；
- 通道支持；
- \(R_s\)；
- 公共子空间；
- 私有通道与私有子空间；
- 线性幅值形式。

随后重新 FP64 凸 refit，并输出：

- KKT residual；
- Gram condition number；
- 正交误差；
- 系数 hash；
- 两方向逐样本预测。

## Stage E6：单 nonlinear increment

只在 E5 通过后运行：

```text
NL0 + 每个已激活通道单独 NL 候选
```

最多选择一个 nonlinear block。

## Stage E7：matured residual PS/AR

只在最终冻结 K 上比较 A0–A7。

## Stage E8：统一比较与 bootstrap

生成：

1. 输入型榜：最终 K 与 K-only、Dynamic-PLS、NLinear-U；
2. 动态榜：最终 K→PS/AR 与 Temporal AE、Joint-K+AR；
3. 解释性表：公共 Rank、私有 Rank、通道支持、非线性和 AR 是否精确零；
4. 计算成本表。

---

# 11. 主要评价指标

## 11.1 主指标

以 pooled MSE 为基础：

\[
\mathrm{RMSE}_{\mathrm{pooled}}
=
\sqrt{
\frac{
\sum_d\sum_{t\in d}(y_t-\hat y_t)^2
}{
\sum_dn_d
}}.
\tag{11.1}
\]

必须同时报告：

- Sheet1→Sheet2 RMSE；
- Sheet2→Sheet1 RMSE；
- pooled RMSE；
- MAE；
- \(R^2\)；
- relative Persistence；
- relative old K-only；
- relative NLinear；
- relative Joint-K+AR。

## 11.2 配对 bootstrap

对同一 sample ID 的平方误差差：

\[
\Delta_t=e_{A,t}^2-e_{B,t}^2.
\tag{11.2}
\]

正式使用移动块 bootstrap：

```text
block lengths = 10, 22, 40, 60 min
replicates = 500
primary block = 40 min
```

输出：

- median relative MSE improvement；
- 95% interval；
- positive probability；
- 两方向分别的 bootstrap；
- pooled paired bootstrap。

GPU 模型使用已冻结的 seed-median ensemble 预测，不把多个 seed 当作独立观测。

## 11.3 子空间指标

- singular values；
- spectral gap；
- principal angles；
- projection Frobenius distance；
- participation ratio；
- leave-one-channel-out correlation；
- shared/private orthogonality error；
- support overlap；
- peak/centroid/half-life；
- fold/rod stability。

---

# 12. 成功等级

## Level A：shared/private K 正式成功

同时满足：

1. pooled RMSE 小于 NLinear-U 的 0.395302；
2. 两方向相对 Persistence 均为正；
3. 相对旧 K-only 的 40 min paired bootstrap 正改善概率不低于 95%；
4. 公共 Rank/子空间通过稳定门禁；
5. 若私有被启用，私有通过全部门禁；
6. placebo 破坏；
7. FP64/KKT/正交认证通过。

允许声明：

> Shared–Private Orthogonal Spectral Urysohn K 在冻结 L6 任务上形成优于旧单通道 K 和 NLinear 的可认证多通道低秩物理输入模型。

## Level B：解释性 Pareto 成功

满足：

- pooled MSE 与 NLinear-U 差异不超过 1%；
- 两方向均为正；
- 参数量、数值稳定和解释性明显优于黑箱；
- 公共/私有证书通过。

允许声明：

> 在统计性能近似等价时，新 K 提供更明确的公共/私有物理分解与 PLC 编译结构。

## Level C：部分成功

满足：

- 超过旧 K-only；
- 但未超过 NLinear 或公共/私有证书不完整。

登记：

```text
PREDICTIVE_IMPROVEMENT_ONLY
NOT_FULLY_K_CERTIFIED
```

## Level D：否定结果

出现任一情况：

- 不超过旧 K-only；
- 仍有方向为负；
- 公共子空间跨棒不稳定；
- 私有只在一根棒出现；
- 依赖 test 才选择 Rank；
- placebo 不破坏。

结论：

> 当前两棒数据不支持 shared/private 扩展，保留旧 K-only 或 exact-zero 物理层，不再增加 Rank。

## 动态预测的独立判定

- 超过 Temporal AE：动态层有竞争力；
- 超过 Joint-K+AR：成为新的总体预测冠军；
- 未超过 Joint-K+AR 不构成物理 K 失败。

---

# 13. 必须输出的消融表

| 模型 ID | 公共 Rank | 私有 | 非线性 | Residual AR | 用途 |
|---|---:|---|---|---|---|
| Z0 | 0 | 0 | 0 | 0 | Persistence |
| OLD-K | 旧联合升速 K | 0 | 0 | 0 | 冻结旧参照 |
| FULL-LIN | full | 无分解 | 0 | 0 | 母空间上限诊断 |
| S1 | 1 | 0 | 0 | 0 | NLinear 的凸可认证对应物 |
| S2 | 2 | 0 | 0 | 0 | 第二公共模态 |
| S1P1/S2P1 | 1/2 | 单通道 Rank-1 | 0 | 0 | 私有字典验证 |
| SP-NL | selected | selected | 单增量 | 0 | 幅值非线性 |
| SP-AR | selected | selected | selected/0 | selected | 最终三层模型 |

必须画出：

1. RMSE 对模型复杂度；
2. 两方向 RMSE；
3. 公共奇异值谱；
4. 两方向公共时间基；
5. 私有候选时间基；
6. 通道载荷；
7. shared/private gain decomposition；
8. 旧 K、NLinear、新 K 的误差差时间序列；
9. bootstrap 区间；
10. OOD 与 \(C^1\) 延拓贡献。

---

# 14. CPU 并行与数值协议

## 14.1 环境

```bash
uv sync --dev
```

验证：

```bash
uv run python - <<'PY'
import numpy as np
import scipy
print('numpy', np.__version__)
print('scipy', scipy.__version__)
print('float64', np.dtype(np.float64))
PY
```

## 14.2 防止线程过度订阅

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

建议调度：

```text
outer/config jobs: 20
bootstrap jobs: 16
linear algebra dtype: float64
prediction dtype: float64
```

单个任务内部不得再开启多线程 BLAS。

## 14.3 预计资源

- 主要计算为多个小型 FP64 ridge/smoothing 线性系统；
- 32 vCPU 足够；
- 不需要 GPU；
- 建议内存 64 GB 以上；
- 预计总耗时约 4–12 h，取决于 full-support 网格和 bootstrap 实现；
- 每个 stage 必须 checkpoint，可断点恢复。

---

# 15. 推荐目录

```text
OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1/
├── configs/
│   ├── frozen_l6.yaml
│   ├── mother_spaces.yaml
│   ├── regularization.yaml
│   └── decision_gates.yaml
├── src/
│   ├── shared_data/
│   ├── mother_basis/
│   ├── full_convex_k/
│   ├── gram_geometry/
│   ├── shared_subspace/
│   ├── private_subspace/
│   ├── nonlinear_amplitude/
│   ├── c1_extension/
│   ├── matured_residual/
│   ├── evaluation/
│   └── packaging/
├── scripts/
│   ├── preflight.py
│   ├── import_frozen_baselines.py
│   ├── run_e1_mother_audit.py
│   ├── run_e2_full_linear.py
│   ├── run_e3_shared_rank.py
│   ├── run_e4_private_rank.py
│   ├── run_e5_refit.py
│   ├── run_e6_nonlinear.py
│   ├── run_e7_residual_ar.py
│   ├── run_e8_compare.py
│   └── build_return_bundle.py
├── tests/
├── results/
├── return/
├── RUN_CPU_CONFIRM.sh
├── RESUME_CPU_CONFIRM.sh
└── README.md
```

---

# 16. 必须通过的测试

至少包括：

1. shared bundle hash 测试；
2. sample-ID 与旧预测完全一致；
3. 无未来输入；
4. 窗口不跨断点；
5. scaler/PCA 训练棒隔离；
6. expanding-window purge；
7. \(V_0\subset V_1\) 数值测试；
8. Gram 白化基不变性；
9. shared/private 正交误差；
10. private zero 与 rank-1 嵌套；
11. nonlinear zero 与 linear 嵌套；
12. \(C^1\) 延拓函数值和一阶导连续；
13. matured residual 因果索引；
14. fixed-subspace refit 不旋转子空间；
15. FP64 prediction；
16. paired bootstrap 对齐；
17. package privacy；
18. ZIP roundtrip 和 manifest hash。

任何因果、sample-ID、正交或嵌套测试失败，正式结果无效。

---

# 17. 启动命令合同

实现完成后的统一入口：

```bash
cd OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1

bash RUN_CPU_CONFIRM.sh \
  --shared /path/to/shared \
  --cpu-baselines /path/to/PHYSICS_FIRST_CPU_RESULTS_bundle.zip \
  --gpu-baselines /path/to/PHYSICS_FIRST_GPU_RESULTS_bundle.zip \
  --config configs/frozen_l6.yaml \
  --n-jobs 20 \
  --bootstrap-jobs 16 \
  2>&1 | tee logs/cpu_confirm.log
```

断点恢复：

```bash
bash RESUME_CPU_CONFIRM.sh \
  --checkpoint results/checkpoints/latest.json
```

查看：

```bash
tail -f logs/cpu_confirm.log
```

---

# 18. 最终结果文件

```text
results/
├── PRECHECK_REPORT.md
├── MOTHER_SPACE_AUDIT.csv
├── FULL_LINEAR_K.csv
├── SHARED_RANK_RESULTS.csv
├── PRIVATE_RANK_RESULTS.csv
├── NONLINEAR_RESULTS.csv
├── RESIDUAL_AR_RESULTS.csv
├── FINAL_INPUT_LEADERBOARD.csv
├── FINAL_DYNAMIC_LEADERBOARD.csv
├── PAIRWISE_BOOTSTRAP.csv
├── FINAL_DECISION.json
├── FINAL_REPORT.md
├── predictions/
├── kernels/
├── subspaces/
├── c1_extension/
├── diagnostics/
└── plots/
```

`FINAL_DECISION.json` 至少包含：

```json
{
  "mother_space": "V0_or_V1",
  "active_channels": [],
  "shared_rank": 0,
  "private_channel": null,
  "private_rank": 0,
  "nonlinear_channel": null,
  "nonlinear_exact_zero": true,
  "residual_model": "EXACT_ZERO",
  "direction_metrics": {},
  "pooled_metrics": {},
  "shared_certification": "...",
  "private_certification": "...",
  "physical_registration": "...",
  "comparison_vs_old_K": {},
  "comparison_vs_NLinear": {},
  "comparison_vs_Joint_K_AR": {}
}
```

---

# 19. 最终打包合同

不得包含：

- 原始 Excel；
- 解压后的完整旧结果包副本；
- `.git`；
- cache；
- 无用 checkpoint；
- `__pycache__`；
- 临时矩阵文件。

打包：

```bash
rm -rf return/OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS
rm -f return/OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS_bundle.zip
rm -f return/OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS_bundle.zip.sha256

uv run python scripts/build_return_bundle.py \
  --source-root . \
  --results results \
  --output return/OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS

uv run python scripts/validate_package.py \
  --package-dir return/OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS \
  --forbid "*.xlsx,.git,__pycache__,cache,*.tmp"

cd return
zip -r OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS_bundle.zip \
  OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS

sha256sum OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS_bundle.zip \
  > OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS_bundle.zip.sha256

uv run python ../scripts/validate_zip_roundtrip.py \
  --zip OPS_UOI_SHARED_PRIVATE_K_CPU_CONFIRM_V1_RESULTS_bundle.zip
```

最终终端必须打印：

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

# 20. 实验结束后的唯一主结论格式

最终报告不得只报一个 pooled RMSE，必须按以下结构给出：

> 在冻结 L6、双向跨晶棒协议下，最终模型选择母空间 \(V_\tau^\star\)、公共 Rank \(R_s^\star\)、私有 Rank 配置 \(R_{p,j}^\star\)、非线性配置和 Residual PS/AR 配置。其 Sheet1→Sheet2、Sheet2→Sheet1 和 pooled RMSE 分别为……。相对旧 K-only、NLinear-U、Dynamic-PLS、Temporal AE 和 Joint-K+AR 的配对 MSE 改善分别为……。公共子空间状态为……，私有子空间状态为……，非线性状态为……，Residual AR 状态为……。因此，本轮允许登记为 `K_CERTIFIED / PREDICTIVE_ONLY / NOT_BIDIRECTIONALLY_STABLE / REJECTED` 中的一项。

禁止根据 pooled RMSE 单独宣称跨棒物理稳定。
