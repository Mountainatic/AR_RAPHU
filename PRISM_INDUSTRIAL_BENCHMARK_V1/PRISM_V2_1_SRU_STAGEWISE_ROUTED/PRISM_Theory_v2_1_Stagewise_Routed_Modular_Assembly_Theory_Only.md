# PRISM Theory v2.1
## Stagewise-Routed Physics-first Response Identification with Scale-specific Multirate Modular Operators
### 逐级目标路由、物理优先、尺度专属、多速率模块化响应辨识理论体系

> **正式中文名**：PRISM v2.1 逐级目标路由物理优先尺度专属多速率模块化响应辨识理论  
> **正式英文名**：Stagewise-Routed Physics-first Response Identification with Scale-specific Multirate Modular Operators  
> **简称**：PRISM v2.1 / PRISM-SR  
> **版本日期**：2026-08-06  
> **文档性质**：理论对象、模块接口、解释权边界、局部选择规则、联合预测路线、认证条件与部署合同  
> **直接理论母本**：`PRISM_Theory_v2_0_Modular_Assembly_Theory_Only.md`  
> **母本位置**：`PRISM_INDUSTRIAL_BENCHMARK_V1/PRISM_V2_MODULAR_NUMERICALLY_FROZEN/PRISM_Theory_v2_0_Modular_Assembly_Theory_Only.md`  
> **母本 Git blob SHA**：`7156a0c77ff19cc2cef1eec538207bc4b2b5d87b`  
> **继承边界**：保留 v2.0 的时间合同、E/K/C/W/A/J 模块代数、通道专属多速率、Urysohn 阶梯、Wiener 读出、成熟残差、联合预测、商空间/Schur 可辨识性、数值证书和部署合同。  
> **核心修订**：停止全局装配 one-SE、停止 A-only 撤销物理输入支路、停止 W/A 的硬特征投影；改为 K→ΔW→A 的逐级 OOF 目标路由、模块局部 guarded one-SE、可关闭软重叠惩罚，以及 K/W/AR 可相互侵占但输入支路必须存在的联合预测路线。  
> **理论边界**：本文不写入任何数据集胜负、排名或误差数值；实验不能反向定义理论。  
> **版本状态**：v2.0 的冻结实验路线已停止；v2.1 继承其模块化成果，但不继承其全局装配选择与硬归因机制。

---

# 0. v2.1 的核心修订

## 0.1 停止的是 v2.0 的旧选择路线，不是 v2 的模块代数

PRISM v2.1 保留：

\[
\mathfrak P
=
\mathfrak T
\circ
\mathfrak E
\circ
(\mathfrak K,\mathfrak C,\mathfrak W,\mathfrak A,\mathfrak J),
\tag{0.1}
\]

其中：

- \(\mathfrak T\)：不可关闭的因果、可用性与 purge 合同；
- \(\mathfrak E\)：单尺度、固定多分辨率或通道专属编码；
- \(\mathfrak K\)：输入历史响应算子；
- \(\mathfrak C\)：通道融合；
- \(\mathfrak W\)：冻结输入响应后的静态读出修正；
- \(\mathfrak A\)：成熟残差状态；
- \(\mathfrak J\)：联合预测路线。

v2.1 停止：

1. 在 \(A\)、\(K\)、\(K+A\)、\(K+W+A\) 之间做一次全局 winner-takes-all one-SE；
2. 把 A-only 规定为比 K-only 更简单的物理装配；
3. 通过 \((I-P)\) 对 W、A 特征进行硬投影并把该投影当作主要解释权保障；
4. 允许下游 W/A 的不稳定反向撤销已经通过独立门槛的 K；
5. 把 Joint 路线中的 \(K=0\)、AR-only 作为可选最终 PRISM 装配。

因此 v2.1 的结构是：

\[
\boxed{
\text{v1.3 的低方差冻结纪律}
+
\text{v2.0 的模块化接口}
+
\text{局部选择与逐级目标路由}
}
\tag{0.2}
\]

## 0.2 三条正式评价路线

### 路线 U：输入-only 路线

\[
\widehat z_t^{U}
=
p_{K,t}
+
\delta w_t,
\tag{0.3}
\]

其中：

\[
p_{K,t}=C(K(E(U_t^-))),
\tag{0.4}
\]

\[
\delta w_t=\Delta W(q_t).
\tag{0.5}
\]

\(\Delta W=0\) 为 identity 中性状态。

### 路线 PF：Physical-First 逐级路由路线

\[
\widehat z_t^{PF}
=
p_{K,t}
+
\delta w_t
+
a_t.
\tag{0.6}
\]

训练顺序固定为：

\[
K/C
\rightarrow
\text{freeze}
\rightarrow
r^{(1)}
\rightarrow
\Delta W
\rightarrow
\text{freeze}
\rightarrow
r^{(2)}
\rightarrow
A.
\tag{0.7}
\]

其中：

\[
r_t^{(1)}
=
z_t-p_{K,t}^{OOF},
\tag{0.8}
\]

\[
r_t^{(2)}
=
r_t^{(1)}-\delta w_t^{OOF}.
\tag{0.9}
\]

A 只读取严格成熟的 \(r^{(2)}\) 历史。

### 路线 J：K/W/AR 联合预测路线

联合路线使用固定、已注册的 K、W 和 AR 特征族，但允许三个块在同一目标上联合优化、相互侵占和调节：

\[
\widehat z_t^{J}
=
b_0+
\Phi_t^K\beta_K+
B_t^W\gamma_W+
H_t^Y\beta_A.
\tag{0.10}
\]

它不实施硬正交，也不按 PF 的冻结顺序分配解释权。它只用于：

- 工程预测；
- 信息集上限评估；
- 检验 K/W/AR 联合信息是否比 v1.3 K-Joint AR 更充分。

联合路线必须满足：

\[
\Phi^K\beta_K+B^W\gamma_W\not\equiv0.
\tag{0.11}
\]

AR-only 只能作为外部基线或输入支路坍缩诊断，不能成为联合路线的最终 PRISM 装配。

## 0.3 A-only 的新地位

v2.1 将 A-only 移出物理装配格：

\[
A\text{-only}\notin\mathcal G_{PF},
\qquad
A\text{-only}\notin\mathcal G_J.
\tag{0.12}
\]

A-only、AR、NAR、状态空间仍可作为：

- 预测基线；
- 输出惯性强度诊断；
- 条件新颖性诊断；
- Joint 输入支路坍缩对照。

但它们不能：

- 撤销已经通过独立 K 门槛的输入结构；
- 被写成 Physical-First 的中性退化；
- 在 Joint 路线中替代 K/W 输入支路。

## 0.4 从硬特征正交改为目标路由与软重叠控制

v2.0 通过

\[
\widetilde B=(I-P_{\mathcal P})B,
\qquad
\widetilde H=(I-P_{\Phi^{PW}})H
\tag{0.13}
\]

进行硬特征分离。

v2.1 的主要归因方式改为：

\[
z
\rightarrow
r^{(1)}=z-p_K^{OOF}
\rightarrow
r^{(2)}=r^{(1)}-\delta w^{OOF}.
\tag{0.14}
\]

W 和 A 使用未硬投影的原始合法特征。为防止下游模块任意重构上游预测，可加入可关闭的软重叠惩罚：

\[
\Omega_W
=
\left\|
\frac{1}{n}
\widetilde P_K^\top
\widetilde{\delta w}
\right\|_2^2,
\tag{0.15}
\]

\[
\Omega_A
=
\left\|
\frac{1}{n}
\widetilde P_{KW}^\top
\widetilde a
\right\|_2^2,
\qquad
P_{KW}=[p_K,\delta w].
\tag{0.16}
\]

其中 \(\widetilde{\cdot}\) 表示 fold-fit 部分内标准化后的预测列，不表示硬投影。

软惩罚的系数 \(\mu_W,\mu_A\) 必须包含：

\[
\mu=0.
\tag{0.17}
\]

软低相关只是一种经验所有权约束，不证明统计独立、因果独立或机理唯一。

## 0.5 从全局 one-SE 改为模块局部 guarded one-SE

v2.1 不再在所有装配之间做单次全局 one-SE。选择作用域分别为：

\[
\mathcal A_K,\quad
\mathcal A_{C|K},\quad
\mathcal A_{W|K,C},\quad
\mathcal A_{A|K,C,W},\quad
\mathcal A_J.
\tag{0.18}
\]

每个模块只与同层中性元和同层复杂度阶梯比较。W/A 不能反向撤销 K。

## 0.6 v2.1 的正式装配集合

Physical-First：

\[
\mathcal G_{PF}
=
\{
K,\;
K+\Delta W,\;
K+A,\;
K+\Delta W+A
\}.
\tag{0.19}
\]

联合预测：

\[
\mathcal G_J
=
\{
J_K,\;
J_{KW},\;
J_{KA},\;
J_{KWA}
\}.
\tag{0.20}
\]

其中所有 \(J\) 候选均包含非零输入支路。不存在 \(J_A\)。

---

# 1. 不可变时间与信息合同

## 1.1 目标头

对目标头 \(m\)，定义：

\[
\eta_m=(h_m,W_m,W_{0,m},D_m).
\tag{1.1}
\]

其中：

- \(h_m\)：预测提前步；
- \(W_m\)：未来目标平均窗口；
- \(W_{0,m}\)：当前状态锚点窗口；
- \(D_m\)：目标或残差额外成熟延迟。

使用半开区间：

\[
\bar y_t^{(W_0)}
=
\frac{1}{W_0}
\sum_{s=t-W_0}^{t-1}y_s,
\tag{1.2}
\]

\[
\bar y_{t+h}^{(W)}
=
\frac{1}{W}
\sum_{s=t+h}^{t+h+W-1}y_s.
\tag{1.3}
\]

主目标为：

\[
z_t
=
\bar y_{t+h_m}^{(W_m)}
-
\bar y_t^{(W_{0,m})}.
\tag{1.4}
\]

## 1.2 合法信息集

输入信息集：

\[
\mathcal I_t^U
=
\sigma\{u_{j,s},x_{k,s},c_s:s<t\}.
\tag{1.5}
\]

状态信息集：

\[
\mathcal I_t^Y
=
\sigma\{y_s:s\le t-1-D_m\}.
\tag{1.6}
\]

动态信息集：

\[
\mathcal I_t^D
=
\mathcal I_t^U\vee\mathcal I_t^Y.
\tag{1.7}
\]

所有 train-only 标准化、knots、基、penalty、profile、支持和选择必须只读取当前 fold-fit 部分。

## 1.3 成熟残差合同

OOF 预测 \(p_s^{OOF}\) 对应的目标残差只有在：

\[
s+h_m+W_m+D_m\le t
\tag{1.8}
\]

时可进入预测原点 \(t\) 的 A 模块。

实现上应优先使用每个样本已经冻结的：

```text
latest_available_target_index
```

而不是只用 \(t-h-W\) 的简化公式。

## 1.4 primitive support 与 purge

若装配 \(a\) 的最大向后依赖为 \(L_a\)，则：

\[
I_t^{(a)}
=
[t-L_a,\;t+h_m+W_m+D_m).
\tag{1.9}
\]

开发与评估边界使用：

\[
G_a
=
L_a+h_m+W_m+D_m+b.
\tag{1.10}
\]

完整 run/entity 之间禁止输入、目标、残差和状态窗口跨界。

---

# 2. 模块接口与语义

## 2.1 编码与输入响应

\[
E_j:\mathcal I_t^U\to\mathbb R^{d_{E,j}},
\tag{2.1}
\]

\[
K_j:\mathbb R^{d_{E,j}}\to\mathbb R^{d_{K,j}}.
\tag{2.2}
\]

每个通道保留 v2.0 的候选阶梯：

\[
0\subset L\subset R1\subset R2\subset\cdots\subset F.
\tag{2.3}
\]

通道可以 exact-zero，但整个 PF/J 路线只有在至少一个输入通道形成非零输入支路时成立。

## 2.2 通道融合

\[
C:\prod_j\mathbb R^{d_{K,j}}\to\mathbb R^{d_C}.
\tag{2.4}
\]

核心中性融合为加性：

\[
C_{\mathrm{add}}(Q_t)
=
\beta_0+\sum_j\beta_j^\top Q_{j,t}.
\tag{2.5}
\]

通道 joint basis 可以保留。高阶 pairwise interaction 不作为首轮 v2.1 SRU 核心搜索的必要组成，但理论接口保留。

## 2.3 Wiener 读出修正

v2.1 将 W 明确写成 identity 基线加残差修正：

\[
W_{\mathrm{total}}(q)
=
W_0(q)+\Delta W(q).
\tag{2.6}
\]

在标量物理预测下：

\[
W_0(q)=q.
\tag{2.7}
\]

在向量潜变量下：

\[
W_0(q)=\alpha_0+\alpha^\top q.
\tag{2.8}
\]

因此：

\[
\Delta W=0
\tag{2.9}
\]

等价于 `WIENER_IDENTITY`。

## 2.4 成熟残差状态

\[
A:H_{r^{(2)}}^-\to\mathbb R.
\tag{2.10}
\]

A 不读取原始 U/X，只读取符合成熟合同的二级残差历史与允许的固定上下文。

## 2.5 联合预测模块

\[
J:
(\Phi^K,B^W,H^Y)\to\mathbb R.
\tag{2.11}
\]

Joint 可共享 PF 注册的：

- K profile；
- K 基族；
- C 表示；
- W spline 基族；
- AR profile。

但 Joint 不共享 PF 的解释权证书。

---

# 3. K/C 物理输入层

## 3.1 通道专属 profile

对通道 \(j\) 和目标头 \(m\)：

\[
\pi_{j,m,r}
=
(\Delta_{j,r},h_m,W_m,W_{0,m},T_{j,r},\mathcal B_{j,r}).
\tag{3.1}
\]

不同通道可拥有不同：

- 重采样步长；
- 历史覆盖；
- 时滞基；
- 幅值分辨率；
- rank。

同一目标头共享 \((h,W,W_0,D)\)。

## 3.2 K 局部候选族

\[
\mathcal A_K
=
\{
K_0,\;
K_L,\;
K_{R1},\ldots,K_{RR},K_F
\}.
\tag{3.2}
\]

\(K_0\) 是输入增量 exact-zero。实现中应明确区分：

- fold-local intercept/锚点；
- 输入增量为零。

即：

\[
\widehat z_{K_0}=b_{0,f},
\tag{3.3}
\]

而不是把“无输入增量”与“强制总预测为零”混为一谈，除非目标合同明确规定零变化即唯一中性预测。

## 3.3 K 的两类证据

### 独立输入预测证据

\[
\Delta_K^{standalone}
=
\frac{
R(K_0)-R(K)
}{
R(K_0)
}.
\tag{3.4}
\]

### 条件新颖性诊断

使用容量冻结、尺度匹配、完全 OOF 的诊断状态模型 \(A_{\mathrm{diag}}\)：

\[
\Delta_{K|A_{\mathrm{diag}}}
=
R(A_{\mathrm{diag}})
-
R(A_{\mathrm{diag}}+K).
\tag{3.5}
\]

\(A_{\mathrm{diag}}\) 只用于诊断 K 是否提供 AR 之外的新信息，不能回调 K，也不能成为 PF 选择器。

## 3.4 固定支持弱收缩重拟合

K 的 profile、rank、active support 和 penalty 选择冻结后，应在每个 OOF fit fold 内进行固定支持弱收缩重拟合：

\[
\widehat\theta_K^{refit}
=
\arg\min_{\theta\in\mathcal S_K}
\|z-\Phi_K\theta\|_2^2
+
\lambda_{refit}\|\theta\|_2^2,
\tag{3.6}
\]

其中 \(\mathcal S_K\) 是冻结支持，\(\lambda_{refit}\) 取可通过数值证书的最弱冻结值。

目的不是提高自由度，而是避免 W/A 仅仅恢复被 K 强 ridge 压缩掉的幅值。

## 3.5 K 的状态标签

- `K_STRUCTURE_SELECTED`
- `K_PREDICTIVE_VALIDATED`
- `K_WEAK_BUT_CONDITIONALLY_NOVEL`
- `K_PHYSICAL_CERTIFIED`
- `K_EXACT_ZERO`
- `K_UNRESOLVED`

下游 W/A 的失败不能撤销已经成立的 K 标签。

## 3.6 C 的选择作用域

C 只在冻结 K 表示内选择：

\[
\mathcal A_{C|K}
=
\{
C_{\mathrm{compressed}},
C_{\mathrm{joint\ basis}},
C_{\mathrm{pairwise}}
\}.
\tag{3.7}
\]

C 的选择不能改变：

- 单通道 profile；
- 单通道支持；
- K 证书；
- 目标头时间合同。

---

# 4. W：一级残差读出修正

## 4.1 OOF 一级残差

冻结 K/C 后，生成：

\[
p_{K,s}^{OOF}
=
C(K(E(U_s^-))).
\tag{4.1}
\]

定义：

\[
r_s^{(1)}
=
z_s-p_{K,s}^{OOF}.
\tag{4.2}
\]

W 的训练目标不是原始 \(z\)，而是 \(r^{(1)}\)。

## 4.2 W 候选

\[
\Delta W_0=0.
\tag{4.3}
\]

非零候选包括：

- monotone I-spline correction；
- natural cubic spline correction；
- 少量冻结 ANOVA correction。

其预测为：

\[
\delta w_t
=
B_W(q_t)\gamma_W.
\tag{4.4}
\]

总输入预测：

\[
p_{KW,t}
=
p_{K,t}+\delta w_t.
\tag{4.5}
\]

## 4.3 不再硬投影 W 基

v2.1 不再要求：

\[
B_W\leftarrow(I-P_{[1,q]})B_W.
\tag{4.6}
\]

原因是闭环共线和低有效激励下，硬投影可能产生：

- 低能量特征；
- 低有效秩；
- 高方差系数；
- W 错误 exact-zero；
- OOD 不稳定。

归因主要由 OOF 目标路由保障。

## 4.4 W 的软重叠惩罚

W 的目标为：

\[
\min_{\gamma_W}
\|r^{(1)}-B_W\gamma_W\|_2^2
+
\lambda_W\gamma_W^\top R_W\gamma_W
+
\mu_W\Omega_W.
\tag{4.7}
\]

其中：

\[
\Omega_W
=
\left\|
\frac1n
\widetilde P_{K,\mathrm{ch}}^\top
\widetilde{B_W\gamma_W}
\right\|_2^2,
\tag{4.8}
\]

\(P_{K,\mathrm{ch}}\) 至少包含：

- 总 \(p_K\)；
- 可用时的通道级 K 贡献。

这样可避免不同通道贡献相互抵消后，W 看似与总 \(p_K\) 不相关。

## 4.5 W 激活状态

- `WIENER_IDENTITY`
- `W_RESIDUAL_VALIDATED`
- `W_SOFT_OVERLAP_GUARDED`
- `W_PREDICTIVE_ONLY`
- `W_UNRESOLVED`

W 通过不自动证明真实过程具有唯一 Wiener 构成方程。

---

# 5. A：二级成熟残差状态

## 5.1 二级 OOF 残差

冻结 W 后：

\[
\delta w_s^{OOF}
=
\Delta W(q_s)^{OOF},
\tag{5.1}
\]

\[
r_s^{(2)}
=
z_s-p_{K,s}^{OOF}-\delta w_s^{OOF}.
\tag{5.2}
\]

## 5.2 成熟残差历史

对预测原点 \(t\)，A 只能读取：

\[
\mathcal H_t^{r^{(2)}}
=
\{
r_s^{(2)}:
s+h_m+W_m+D_m\le t
\}.
\tag{5.3}
\]

实现必须：

1. 按 entity/run 构造；
2. 不跨 run；
3. 使用 `latest_available_target_index`；
4. fold-fit 内估计中心和尺度；
5. 不用 evaluation fold 的残差均值中心化 fit fold。

## 5.3 A 候选

首选核心候选：

\[
A_0=0,
\tag{5.4}
\]

\[
A_{\mathrm{AR}}(H_t)
=
\beta_0+\sum_{\ell\in\mathcal L_A}\beta_\ell r_{t-\ell}^{(2)}.
\tag{5.5}
\]

可扩展到：

- 稳定状态空间；
- 低自由度目标-only NAR；
- 工况固定的残差状态。

首轮 v2.1 SRU 核心路线以线性 mature residual AR 为主，以减少重新引入高方差状态捷径。

## 5.4 不再硬投影 A 特征

v2.1 不再使用：

\[
H\leftarrow(I-P_{[p_K,\delta w]})H.
\tag{5.6}
\]

A 的上游隔离依靠：

- OOF 二级残差目标；
- 成熟条件；
- no-U/X 合同；
- 局部 exact-zero；
- 软重叠惩罚；
- 无回调冻结顺序。

## 5.5 A 的软重叠惩罚

\[
\min_{\beta_A}
\|r^{(2)}-H\beta_A\|_2^2
+
\lambda_A\|\beta_A\|_2^2
+
\mu_A\Omega_A,
\tag{5.7}
\]

\[
\Omega_A
=
\left\|
\frac1n
\widetilde P_{KW}^\top
\widetilde{H\beta_A}
\right\|_2^2.
\tag{5.8}
\]

其中 \(\mu_A=0\) 必须保留为候选。

## 5.6 A 的状态标签

- `STATE_EXACT_ZERO`
- `A_RESIDUAL_VALIDATED`
- `A_SOFT_OVERLAP_GUARDED`
- `A_PREDICTIVE_ONLY`
- `A_UNRESOLVED`

A 只能解释为物理输入层之外的成熟可预测状态，不得自动命名为具体未测机理。

---

# 6. Physical-First 的严格训练与选择顺序

## 6.1 Stage P0：数据与协议冻结

冻结：

- 数据包与哈希；
- 目标头；
- split；
- sample IDs；
- proxy policy；
- availability scenario；
- target maturity；
- purge；
- candidate grids；
- baseline contracts；
- selection thresholds。

## 6.2 Stage P1：K 通道审计

每个通道：

1. profile；
2. exact-zero；
3. linear/rank/full 阶梯；
4. numerical certificate；
5. Gram/Schur；
6. common support；
7. outer stability；
8. conditional novelty diagnostic。

## 6.3 Stage P2：K/C 固定支持联合重拟合

冻结通道 profile 和基后，在兼容目标头内拟合 C。

输出：

- OOF \(p_K\)；
- validation \(p_K\)；
- 通道贡献；
- K/C contracts；
- local one-SE audit。

## 6.4 Stage P3：W 局部选择

在 \(r^{(1)}\) 上比较：

\[
\Delta W_0
\quad\text{vs}\quad
\Delta W_{\mathrm{spline}}.
\tag{6.1}
\]

W 选择后冻结，不回调 K/C。

## 6.5 Stage P4：A 局部选择

在 \(r^{(2)}\) 上比较：

\[
A_0
\quad\text{vs}\quad
A_{\mathrm{mature\ AR}}.
\tag{6.2}
\]

A 选择后冻结。

## 6.6 Stage P5：最终固定结构重拟合

在 train+validation 上按冻结结构依次重拟合：

1. K/C；
2. W；
3. A。

仍禁止端到端回调。

## 6.7 PF 不存在 A-only 回退

若总输入支路未通过：

\[
p_K+\delta w\equiv0
\tag{6.3}
\]

则输出：

```text
PHYSICS_ROUTE_NOT_SUPPORTED
```

而不是选择 A-only 作为 PF 最终结构。

---

# 7. Joint-KWA 联合预测路线

## 7.1 目标

Joint 路线回答：

> 在相同输入 profile、W 基族和 AR profile 下，允许 K、W、AR 在共享目标空间中相互侵占和调节时，能达到怎样的预测上限？

它不回答：

- K 的唯一物理贡献；
- W 的唯一静态曲率；
- AR 的唯一未测状态；
- 开放环 plant 结构。

## 7.2 固定基联合模型

在每个 inner fold 内：

1. 使用 fit fold 构造 K 设计 \(\Phi^K\)；
2. 用 fit fold 的 K seed latent \(q^{seed}\) 构造未硬投影 W 基 \(B^W(q^{seed})\)；
3. 构造合法目标历史 AR 特征 \(H^Y\)；
4. 对三个块联合拟合。

\[
\widehat z
=
b_0+
\Phi^K\beta_K+
B^W\gamma_W+
H^Y\beta_A.
\tag{7.1}
\]

采用块惩罚：

\[
\min
\|z-\widehat z\|_2^2
+
\lambda_K\|\beta_K\|_2^2
+
\lambda_W\gamma_W^\top R_W\gamma_W
+
\lambda_A\|\beta_A\|_2^2.
\tag{7.2}
\]

Joint 不使用 PF 的软重叠惩罚，因为其目的就是允许块间共享和侵占。

## 7.3 候选阶梯

\[
J_K:
\quad
\gamma_W=0,\ \beta_A=0,
\tag{7.3}
\]

\[
J_{KW}:
\quad
\beta_A=0,
\tag{7.4}
\]

\[
J_{KA}:
\quad
\gamma_W=0,
\tag{7.5}
\]

\[
J_{KWA}:
\quad
\beta_K,\gamma_W,\beta_A\ \text{均可非零}.
\tag{7.6}
\]

不注册：

\[
J_A,\quad J_0.
\tag{7.7}
\]

## 7.4 输入支路非坍缩门

定义 Joint 输入贡献：

\[
g_U
=
\Phi^K\beta_K+B^W\gamma_W.
\tag{7.8}
\]

AR-only 诊断风险为 \(R(A_{\mathrm{diag}})\)。联合输入条件新颖性为：

\[
\Delta_{U|A}
=
\frac{
R(A_{\mathrm{diag}})
-
R(A_{\mathrm{diag}}+g_U)
}{
R(A_{\mathrm{diag}})
}.
\tag{7.9}
\]

Joint 只有在：

1. \(\Delta_{U|A}\) 超过冻结门槛；
2. paired fold 正方向比例达标；
3. \(g_U\) 方差或能量不低于数值阈值；
4. K/W 输入支持合法；

时标记：

```text
JOINT_INPUT_PATH_VALIDATED
```

否则标记：

```text
JOINT_INPUT_PATH_COLLAPSED
```

并且不把 AR-only 作为 Joint 的替代结果。

## 7.5 Joint 的不可归因性

若：

\[
\operatorname{span}(\Phi^K)
\cap
\operatorname{span}(B^W,H^Y)
\ne\{0\},
\tag{7.10}
\]

则存在不同参数块给出相同总预测。块 ridge 只选择数值代表，不制造唯一分解。

因此 Joint 的合法声明是：

\[
\boxed{
\text{联合输入—读出—状态信息提高预测}
}
\tag{7.11}
\]

而不是：

\[
\boxed{
\text{Joint 内部 K/W/AR 系数等于真实机理贡献}
}.
\tag{7.12}
\]

---

# 8. 模块局部 guarded one-SE

## 8.1 局部风险

对模块作用域 \(M\) 的候选 \(m\in\mathcal A_M\)：

\[
\widehat R_M(m)
=
\frac1F
\sum_{f=1}^F L_{M,f}(m).
\tag{8.1}
\]

最小风险候选为：

\[
m_*=\arg\min_m\widehat R_M(m).
\tag{8.2}
\]

其标准误：

\[
SE_M^*
=
\frac{
sd\{L_{M,f}(m_*)\}
}{
\sqrt{F}
}.
\tag{8.3}
\]

局部 one-SE 集合：

\[
\mathcal A_{M,1SE}
=
\{
m:
\widehat R_M(m)
\le
\widehat R_M(m_*)+SE_M^*
\}.
\tag{8.4}
\]

## 8.2 guarded activation

设中性元为 \(m_0\)。对非中性候选：

\[
\Delta_M(m)
=
\frac{
\widehat R_M(m_0)-\widehat R_M(m)
}{
\widehat R_M(m_0)
}.
\tag{8.5}
\]

并定义 paired positive fraction：

\[
\rho_M(m)
=
\frac1F
\sum_f
\mathbf 1
\{
L_{M,f}(m)<L_{M,f}(m_0)
\}.
\tag{8.6}
\]

可激活集合：

\[
\mathcal A_M^+
=
\{
m\in\mathcal A_{M,1SE}\setminus\{m_0\}:
\Delta_M(m)\ge\epsilon_M,\;
\rho_M(m)\ge\rho_M^{min}
\}.
\tag{8.7}
\]

选择规则：

\[
\widehat m_M
=
\begin{cases}
\min_{\preceq_M}\mathcal A_M^+,
&
\mathcal A_M^+\ne\varnothing,
\\
m_0,
&
\mathcal A_M^+=\varnothing.
\end{cases}
\tag{8.8}
\]

这与 v2.0 “只要 neutral 在 one-SE 集合就立即选择 neutral”不同。v2.1 允许一个统计可接受、具有预注册实用增益且折间方向稳定的非中性候选被保留。

## 8.3 选择作用域隔离

\[
\widehat K
=
Select_{1SE}^{local}(\mathcal A_K),
\tag{8.9}
\]

\[
\widehat W
=
Select_{1SE}^{local}(\mathcal A_{W|\widehat K,\widehat C}),
\tag{8.10}
\]

\[
\widehat A
=
Select_{1SE}^{local}(\mathcal A_{A|\widehat K,\widehat C,\widehat W}).
\tag{8.11}
\]

W/A 的选择结果不参与重新选择 K。

## 8.4 Joint 的 route-local one-SE

Joint 候选只在：

\[
\mathcal A_J
=
\{J_K,J_{KW},J_{KA},J_{KWA}\}
\tag{8.12}
\]

内比较。

复杂度偏序：

\[
J_K
\preceq
J_{KW},
\qquad
J_K
\preceq
J_{KA},
\tag{8.13}
\]

\[
J_{KW}
\preceq
J_{KWA},
\qquad
J_{KA}
\preceq
J_{KWA}.
\tag{8.14}
\]

\(J_{KW}\) 与 \(J_{KA}\) 通常不可比较，应通过冻结 tie-break 决定，不得将 AR-only 置于其前。

---

# 9. 可辨识性与解释权

## 9.1 总预测与模块证书分离

v2.1 区分：

1. 总预测；
2. 输入支路预测；
3. K 商空间代表；
4. W 条件读出；
5. A 成熟残差预测；
6. Joint 数值块；
7. 机理解释。

总预测好不等于每个模块均可解释。

## 9.2 K 的商空间

\[
K\sim K'
\iff
\mathcal A(K-K')=0.
\tag{9.1}
\]

可识别对象首先是：

\[
\mathcal H/\ker\mathcal A.
\tag{9.2}
\]

## 9.3 W 的条件语义

W 只在冻结 K/C 潜变量下解释：

\[
r^{(1)}\approx\Delta W(q).
\tag{9.3}
\]

W 的稳定曲率不证明真实传感器或过程存在唯一静态方程。

## 9.4 A 的条件语义

A 只在冻结 K/C/W 后解释：

\[
r^{(2)}\approx A(H_{r^{(2)}}^-).
\tag{9.4}
\]

A 可能包含：

- 未测内部状态；
- 控制器状态；
- 未测扰动；
- 截断误差；
- 可预测测量误差。

## 9.5 软重叠不等于正交证明

\[
\Omega_W\approx0,\quad\Omega_A\approx0
\tag{9.5}
\]

只表示开发 fold 中预测列低线性重叠，不表示：

- 模块独立；
- 因果独立；
- OOD 不变；
- 唯一物理分解。

## 9.6 闭环边界

被动闭环数据默认识别：

\[
\boxed{
\text{当前控制策略、运行域和观测分布下的预测响应算子}
}.
\tag{9.6}
\]

没有外生激励、控制器知识、工具变量或跨策略复现时，不得写成开放环 plant。

---

# 10. 模块级证书与状态标签

## 10.1 K 证书

至少包括：

- profile 合法；
- exact-zero 对照；
- Gram/Schur；
- common support；
- rank margin；
- fold/run stability；
- placebo；
- block bootstrap；
- numerical certificate；
- regularization/resolution sensitivity；
- closed-loop scope。

## 10.2 W 证书

至少包括：

- identity 对照；
- OOF 一级残差；
- train-only knots；
- shape/monotone constraint；
- effective degrees of freedom；
- support/extrapolation；
- soft overlap audit；
- numerical certificate。

## 10.3 A 证书

至少包括：

- rolling OOF；
- \(D_m\) 成熟条件；
- `latest_available_target_index`；
- no-U/X；
- fold-local centering；
- exact-zero；
- soft overlap audit；
- observed mature feature fraction；
- no test/OOD selection。

## 10.4 Joint 证书

Joint 不授予物理证书，只输出：

- `JOINT_INPUT_PATH_VALIDATED`
- `JOINT_INPUT_PATH_COLLAPSED`
- `JOINT_PREDICTIVE_VALIDATED`
- `JOINT_PREDICTIVE_UNSTABLE`
- `JOINT_OOD_UNSTABLE`

## 10.5 装配状态

| 状态 | 含义 |
|---|---|
| `ASSEMBLY_FROZEN` | 开发数据上结构与超参数已冻结 |
| `INPUT_ONLY` | K/C/ΔW 输入支路，无 A |
| `PHYSICS_FIRST_STAGEWISE` | K/C→ΔW→A 逐级冻结 |
| `WIENER_IDENTITY` | ΔW exact-zero |
| `STATE_EXACT_ZERO` | A exact-zero |
| `K_EXACT_ZERO` | 输入支路未成立 |
| `PHYSICS_ROUTE_NOT_SUPPORTED` | 不回退 A-only |
| `PREDICTIVE_JOINT_KWA` | K/W/AR 联合预测 |
| `PHYSICAL_CERTIFIED` | 对应物理模块证书完整通过 |
| `PREDICTIVE_VALIDATED` | 总预测证据通过但结构证据不足 |
| `UNRESOLVED` | 证据不足 |
| `OOD_UNSTABLE` | OOD 方向失败 |

---

# 11. 数值实现合同

## 11.1 FP64 与直接求解

固定基下：

\[
(\Phi^\top\Phi+\Lambda)\theta
=
\Phi^\top y.
\tag{11.1}
\]

求解顺序：

1. Cholesky；
2. pivoted QR；
3. SVD rescue。

必须记录：

- relative KKT；
- condition number；
- effective rank；
- solver path；
- rescue reason。

## 11.2 不显式构造大型投影矩阵

v2.1 的核心 W/A 不再做硬投影。仍禁止构造大型 \(I-P\)。

C 的可选 ANOVA 交互若需要主效应归属，应使用 QR/SVD 或软 ANOVA penalty，不得显式构造 \(I-P\)。

## 11.3 fold-local 预处理

每个 fold 独立估计：

- mean/scale；
- amplitude knots；
- W knots；
- residual mean；
- state normalization；
- soft overlap matrices；
- support bounds。

任何 evaluation/test 行不得影响这些量。

## 11.4 预测文件与选择损失一致性

每个最终候选必须同时记录：

```text
final_selected_candidate
final_selected_fold_losses
final_selected_prediction_path
final_selected_contract
```

四者必须指向同一候选。

禁止使用 one-SE 前候选的损失，却物化 activation gate 后的 neutral 预测。

## 11.5 参数量

\[
N_{\mathrm{total}}
=
N_E+N_K+N_C+N_W+N_A.
\tag{11.2}
\]

Joint 另计：

\[
N_J
=
N_K^{joint}+N_W^{joint}+N_A^{joint}+1.
\tag{11.3}
\]

报告：

- stored parameter count；
- active coefficient count；
- effective degrees of freedom；
- deployment state count。

---

# 12. 部署合同

## 12.1 PF 在线执行

1. 多速率缓存；
2. K 通道响应；
3. C 融合；
4. ΔW 查表/样条；
5. 成熟二级残差更新；
6. A 更新；
7. 总预测和审计标志。

## 12.2 Joint 在线执行

Joint 可以部署为固定基线性读出：

\[
[\Phi^K,B^W,H^Y]\theta_J.
\tag{12.1}
\]

但部署标签必须保留：

```text
PREDICTIVE_JOINT_KWA
```

不得改写为物理分解。

## 12.3 误差预算

\[
\varepsilon_{\mathrm{deploy}}
\le
\varepsilon_E+
\varepsilon_K+
\varepsilon_C+
\varepsilon_W+
\varepsilon_A+
\varepsilon_q.
\tag{12.2}
\]

关闭模块对应误差项为零。

---

# 13. 理论命题

## 命题 1：逐级路由的因果闭包

若 K/C 只读 \(\mathcal I_t^U\)，W 只读冻结 K/C 潜变量，A 只读符合式 (1.8) 的成熟二级残差，则 PF 装配是 \(\mathcal I_t^D\) 上的因果可测函数。

## 命题 2：逐级路由不需要特征正交即可定义模块顺序

OOF 目标：

\[
r^{(1)}=z-p_K^{OOF},
\qquad
r^{(2)}=r^{(1)}-\delta w^{OOF}
\]

足以定义 W 和 A 的训练对象。特征硬正交不是定义该顺序的必要条件。

## 命题 3：目标路由不自动产生完全正交分解

逐级残差拟合不一般推出：

\[
p_K\perp\delta w,\quad
p_K\perp a,\quad
\delta w\perp a.
\]

除非相关投影空间满足额外正交、嵌套或交换条件。因而 v2.1 使用“层级残差归属”，不声称“完备正交分解”。

## 命题 4：局部选择阻止下游撤销上游

若 K 在 \(\mathcal A_K\) 内冻结，W 和 A 的候选集合不含 K 的重新选择，则 W/A 的 exact-zero 或失败不能改变 K 的开发选择结果。

## 命题 5：A-only 不是 PF 的嵌套子模型

A-only 与 K-only使用不同信息集，通常不存在“将 PF 某个后置模块设为中性元即可得到 A-only”的单向物理偏序。因此将 A-only 固定排在 K 前不是由模块嵌套推出的。

## 命题 6：Joint 的总预测可稳定而块分解不唯一

在联合正规方程严格凸时，总预测系数代表可唯一；若 K/W/AR 子空间相交，则块贡献依赖 penalty 和参数化，不能作物理归因。

## 命题 7：输入支路门阻止 Joint 退化为 AR-only

若 Joint 候选集不含 \(J_A\)，且最终结果必须通过式 (7.9) 的输入条件新颖性门，则 Joint 不会以 AR-only 作为正式 PRISM 输出。

## 命题 8：软重叠惩罚是连续归因约束

\(\mu\Omega\) 在 \(\mu\to\infty\) 时趋向强低重叠，在 \(\mu=0\) 时退化为无归因惩罚。它提供连续偏好，不像硬投影那样直接删除特征方向。

## 命题 9：软低重叠不证明 OOD 稳定

即使开发域 \(\Omega_W,\Omega_A\) 很小，分布改变后模块相关结构仍可能变化，因此 OOD 证据必须独立评估。

## 命题 10：代码正确性是理论检验的前置条件

当选择损失、物化预测、成熟索引或 fold 预处理不一致时，实验结果不能被用于判断理论模块是否有效。

---

# 14. 正式研究流程

1. 冻结 C1 数据、目标头、split、sample IDs 和 hashes；
2. 注册 v2.1 候选格；
3. 仅在开发数据运行 K 局部选择；
4. 固定支持弱收缩 OOF 重拟合；
5. 运行 C；
6. 生成一级 OOF 残差；
7. 运行 ΔW 局部选择；
8. 生成二级 OOF 成熟残差；
9. 运行 A 局部选择；
10. 输出 PF 装配卡；
11. 使用相同注册特征运行 Joint-KWA；
12. 检查输入支路非坍缩；
13. 冻结最终 manifest；
14. 访问 test；
15. 与冻结基线配对比较；
16. 输出预测、结构、证书、参数和资源；
17. 打包 manifest/hash。

---

# 15. 经验中立性

## 15.1 本文不声称

- v2.1 一定优于 v1.3；
- W 一定在 SRU 激活；
- A 一定提供增益；
- Joint 一定优于 Linear NARX；
- PF 一定优于 Hammerstein-Wiener；
- 软惩罚一定改善 OOD；
- K/W/AR 联合系数具有唯一机理含义。

## 15.2 本文正式声称

- A-only 不应撤销已独立评估的 K；
- PF 应按 K→ΔW→A 逐级 OOF 目标路由；
- W/A 的归因约束不必依赖硬特征投影；
- 所有 v2.1 PRISM 选择应在对应局部作用域内完成；
- Joint 可以允许 K/W/AR 相互侵占，但必须保留非零输入支路；
- 理论证书与预测基线必须分开报告。

---

# 16. 最终语义链

\[
\boxed{
\text{冻结时间、数据和信息合同}
\rightarrow
\text{K 局部评估}
\rightarrow
\text{固定支持 OOF 重拟合}
\rightarrow
\text{C 融合}
\rightarrow
r^{(1)}
\rightarrow
\Delta W\text{ 局部选择}
\rightarrow
r^{(2)}
\rightarrow
A\text{ 成熟残差选择}
\rightarrow
\text{PF 装配卡}
\rightarrow
\text{独立 Joint-KWA}
\rightarrow
\text{冻结后 test}
}
\tag{16.1}
\]

PRISM v2.1 的核心问题是：

\[
\boxed{
\text{先确定输入支路是否成立，}
\text{再逐级解释输入读出误差与成熟状态误差；}
\text{预测上限允许联合侵占，}
\text{但不把联合分解冒充物理分解。}
}
\tag{16.2}
\]

---

# 附录 A：核心符号

| 符号 | 含义 |
|---|---|
| \(z_t\) | 未来目标窗口与当前锚点之差 |
| \(E_j\) | 通道多尺度编码 |
| \(K_j\) | 通道输入响应 |
| \(Q_j\) | 通道冻结基表示 |
| \(C\) | 通道融合 |
| \(p_K\) | K/C 输入预测 |
| \(\Delta W\) | identity 之后的静态残差读出修正 |
| \(r^{(1)}\) | K/C 后一级残差 |
| \(r^{(2)}\) | K/C/ΔW 后二级残差 |
| \(A\) | 成熟二级残差状态 |
| \(J\) | K/W/AR 联合预测 |
| \(D_m\) | 标签/残差成熟延迟 |
| \(\Omega_W,\Omega_A\) | 软输出重叠惩罚 |
| \(\mathcal A_M\) | 模块 M 的局部候选集合 |

# 附录 B：v2.0 到 v2.1 的映射

| v2.0 | v2.1 |
|---|---|
| E/K/C/W/A/J 模块代数 | 保留 |
| 全局装配 one-SE | 删除 |
| A-only 正式装配 | 降级为外部预测基线 |
| W 正交曲率基 | 改为未硬投影 ΔW + 可关闭软惩罚 |
| A 物理空间残差化特征 | 改为二级目标路由 + 可关闭软惩罚 |
| K/C/W 冻结后 A | 保留并强化 OOF 一致性 |
| J 允许 K=0/AR-only | 删除 |
| Joint 只加入固定 KW 标量 | 扩展为 K/W/AR 固定基联合块拟合 |
| one-SE neutral 绝对优先 | 改为局部 guarded one-SE |
| 选择后损失/预测可能分离 | 强制 final candidate/loss/path/contract 一致 |
| 简化成熟条件 \(s+h+W\le t\) | 恢复 \(s+h+W+D\le t\) 和 latest index |

# 附录 C：v2.1 首轮核心实验暂不激活

- 多工况 mixture-of-experts；
- 高阶全通道交互；
- 深度 W；
- 深度 Joint；
- 在线 K 形状适配；
- 测试时自适应；
- 把不确定性作为预测层；
- 用 AR-only 作为 Joint 退化候选。
