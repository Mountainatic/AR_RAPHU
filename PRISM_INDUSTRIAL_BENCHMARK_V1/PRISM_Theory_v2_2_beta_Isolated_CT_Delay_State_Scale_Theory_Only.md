# PRISM Theory v2.2 (beta)
## Isolated Continuous-Time Delay–State–Scale Extension of PRISM v2.1.1
### PRISM v2.1.1 的独立连续时间“延迟–状态–尺度”扩展理论

> **正式中文名**：PRISM v2.2(beta) 独立连续时间延迟–状态–尺度扩展理论  
> **正式英文名**：PRISM v2.2 (beta): Isolated Continuous-Time Delay–State–Scale Extension of PRISM v2.1.1  
> **版本日期**：2026-08-24  
> **版本状态**：独立实验性小版本 / beta  
> **规范分支**：`prism-v2-2-beta-ct`  
> **稳定母本**：PRISM v2.1.1，分支 `prism-v2-1-1-metro-p60-joint-stability-final`  
> **直接理论母本**：`PRISM_Theory_v2_1_1_Implementation_Safe_Stagewise_Routed_Modular_Assembly_Theory_Only.md`  
> **文档性质**：Theory Only。本文定义 v2.2(beta) 的理论对象、继承边界、新增时间状态表示、装配合同、解释权边界、证书要求与回退规则；不写入任何数据集排名、实验胜负或 test 数值。

> [!IMPORTANT]
> ## 隔离 / 回退声明
>
> 本文只对 `prism-v2-2-beta-ct` 分支成立。**PRISM v2.2(beta) 不是 PRISM v2.1.1 的追认式重解释，也不是新的稳定母本。**
>
> 若未来明确要求“按 v2.1.1 整理 PRISM”“回退到 v2.1.1”“重新说明稳定版 PRISM”，除非同时明确要求纳入 v2.2(beta)，否则：
>
> 1. **不得引用本文来补全、修改或解释 v2.1.1；**
> 2. **不得把本文的 CT 状态、Delay–State–Scale 分解、\(\Gamma_{CT}\) 装配、CT 证书或 beta 假设写回 v2.1.1；**
> 3. **不得通过从 v2.2(beta) 删除新增项的方式“反推” v2.1.1；**
> 4. v2.1.1 的冻结理论、冻结代码和冻结合同始终是回退时唯一权威来源。
>
> 因而版本关系是：
>
> \[
> \boxed{\text{PRISM v2.1.1 stable}}
> \longrightarrow
> \boxed{\text{PRISM v2.2(beta) isolated experimental branch}},
> \]
>
> 而不是“v2.2 覆盖 v2.1.1”。

---

# 0. 版本关系与核心修订

## 0.1 v2.2(beta) 改什么，不改什么

PRISM v2.2(beta) 完整继承 v2.1.1 的以下核心原则：

- 不可关闭的时间、因果、availability 与 purge 合同 \(\mathfrak T\)；
- K/C/\(\Delta W\)/A/J 的模块语义；
- K/C \(\rightarrow\) freeze \(\rightarrow\) 一级 OOF 残差 \(\rightarrow\) \(\Delta W\) \(\rightarrow\) freeze \(\rightarrow\) 二级成熟残差 \(\rightarrow\) A 的逐级目标路由；
- 模块局部 guarded one-SE，而非全局 winner-takes-all；
- 结构选择与数值稳定分离；
- candidate / loss / OOF prediction / materialized prediction 一致；
- PF 与 Joint 的输入支路不得静默坍缩；
- final holdout 不参与 profile、penalty、route、装配权重或证书阈值选择；
- A-only 仍然不是 PF 的嵌套物理子模型；
- Joint 仍然不获得 PF 的解释权证书。

v2.2(beta) 的**唯一核心理论增量**是：

\[
\boxed{
\text{离散历史 profile}
\quad\to\quad
\text{Delay–State–Scale 并行时间表示}
}
\]

即在原输入时间编码层中加入固定稳定连续时间状态基，并用受约束的预测级装配组合不同时间表示。

## 0.2 正式新增对象

v2.2(beta) 新增四个符号：

- \(D\)：Discrete Delay branch，显式离散延迟支路；
- \(M\)：CT-Multires branch，连续时间多分辨率增量支路；
- \(S\)：CT-Absolute branch，连续时间绝对慢状态支路，**可选且必须通过数值准入**；
- \(\Gamma_{CT}\)：时间支路的受约束 late assembly。

必须特别说明：

\[
\boxed{\Gamma_{CT}\neq \mathfrak A}
\]

v2.1.1 中 \(\mathfrak A\) 已经正式表示“二级成熟残差状态模块”。早期开发文档中出现的 “A-level late assembly” 只表示“assembly level”，**不得继续在理论文档中使用 A 作为该装配器名称**。从本文开始，统一记为：

\[
\boxed{\Gamma_{CT}}.
\]

这样 v2.1.1 的 \(\mathfrak A\) 语义保持不变。

## 0.3 v2.2(beta) 的规范 PF 结构

对时间支路集合

\[
\mathcal B_{CT}\subseteq\{D,M,S\},
\]

每个支路先形成独立输入预测：

\[
E_b\rightarrow K_b\rightarrow C_b\rightarrow p_{b,t},
\qquad b\in\mathcal B_{CT}.
\]

随后进行预测级装配：

\[
p_{\Gamma,t}
=
\Gamma_{CT}
\left(
\{p_{b,t}\}_{b\in\mathcal B_{CT}},
 p_{P,t}
\right).
\tag{0.1}
\]

冻结 \(p_\Gamma\) 后，v2.1.1 原有顺序继续：

\[
\boxed{
\{D,M,S\}
\to K/C
\to \Gamma_{CT}
\to \Delta W
\to \mathfrak A
}
\tag{0.2}
\]

因此 v2.2(beta) 不是把 CT-SSM 放在 \(\mathfrak A\) 里，也不是用 CT-SSM 替换 K/C/W/A。

---

# 1. 不可变时间与信息合同

除本节明确增加的 CT 状态携带规则外，本节完整继承 v2.1.1。

## 1.1 目标头

对目标头 \(m\)：

\[
\eta_m=(h_m,W_m,W_{0,m},D_m).
\tag{1.1}
\]

当前状态锚点与未来目标窗口继续使用半开区间：

\[
\bar y_t^{(W_0)}
=
\frac1{W_0}\sum_{s=t-W_0}^{t-1}y_s,
\tag{1.2}
\]

\[
\bar y_{t+h}^{(W)}
=
\frac1W\sum_{s=t+h}^{t+h+W-1}y_s.
\tag{1.3}
\]

主变化目标仍可写为：

\[
z_t=
\bar y_{t+h_m}^{(W_m)}-
\bar y_t^{(W_{0,m})}.
\tag{1.4}
\]

若某任务使用绝对目标，则 persistence anchor 应解释为相应绝对持久预测；若使用式 (1.4) 的变化目标，则 persistence 的增量预测为零。

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

所有 CT 状态必须是这些合法历史的确定性函数。不得用未来样本反向初始化、双向滤波或 evaluation fold 统计量构造 CT state。

## 1.3 CT 的状态携带合同

连续时间滤波器理论上具有无限衰减尾，因此不能简单把它伪装成固定有限 lag。

对同一 entity/run，允许在时间边界携带由**边界之前的合法输入**形成的确定性 CT state：

\[
z(t_b^-)
=F\bigl(\mathcal I_{t_b}^U\bigr).
\tag{1.8}
\]

该状态携带不是泄漏，因为部署时同样拥有过去输入。

但必须满足：

1. 独立 run/entity 之间状态重置；
2. validation/test 的初始 CT state 不得读取 validation/test 的未来；
3. fold-fit 产生的标准化、support reference、penalty 与 selector 不能从 evaluation fold 回流；
4. 若协议要求严格有限 primitive support，则必须显式声明近似截断容差，而不是默认为有限历史。

## 1.4 CT 的 \(\varepsilon\)-有效支持

对一阶稳定模态，距离当前 \(L\) 之前的历史权重至多按

\[
e^{-L/\tau}
\]

衰减。给定预注册容差 \(\varepsilon\in(0,1)\)，可定义：

\[
L_{\tau,\varepsilon}
=
\tau\log\frac1\varepsilon.
\tag{1.9}
\]

全 CT bank 的有效支持：

\[
L_{CT,\varepsilon}
=
\max_r L_{\tau_r,\varepsilon}.
\tag{1.10}
\]

若 sample-level purge 需要有限长度，可以使用式 (1.10)；但必须把 \(\varepsilon\) 写入冻结配置。该近似支持不能被描述为 CT state 的真实严格有限记忆。

---

# 2. 连续时间稳定状态基

## 2.1 固定一阶稳定模态

对输入通道 \(x_j(t)\) 和时间常数 \(\tau_r>0\)：

\[
\dot z_{j,r}(t)
=
-\frac1{\tau_r}z_{j,r}(t)
+
\frac1{\tau_r}x_j(t).
\tag{2.1}
\]

对应极点：

\[
\lambda_r=-\frac1{\tau_r}<0.
\tag{2.2}
\]

因此每个 CT mode 在结构上稳定。

## 2.2 精确 ZOH 离散化

若 \([t_{k-1},t_k)\) 内输入按 zero-order hold 处理，令：

\[
\Delta t_k=t_k-t_{k-1},
\]

则：

\[
a_{k,r}
=
\exp\left(-\frac{\Delta t_k}{\tau_r}\right),
\tag{2.3}
\]

\[
z_{j,r,k}
=
a_{k,r}z_{j,r,k-1}
+
(1-a_{k,r})x_{j,k}.
\tag{2.4}
\]

规则采样只是 \(\Delta t_k\equiv\Delta t\) 的特例。

这意味着 v2.2(beta) 的理论对象按**物理时间常数**定义，而不是按“多少个样本”定义。采样频率改变时，只需重新计算 \(a_{k,r}\)，不应把相同 \(\tau_r\) 重新解释为另一种物理尺度。

## 2.3 稳定性边界

因为 \(\tau_r>0\)，所以：

\[
0<a_{k,r}<1.
\tag{2.5}
\]

于是：

\[
|z_{j,r,k}|
\le
\max\left(
|z_{j,r,0}|,
\sup_{s\le k}|x_{j,s}|
\right).
\tag{2.6}
\]

因此固定 CT bank 是 BIBO stable 的确定性因果编码器。

## 2.4 固定 \(\tau\) 而不是学习 \(\tau\)

v2.2(beta) **不学习 \(\tau_r\)**。

时间尺度集合：

\[
\mathcal T_\tau
=
\{\tau_1<\tau_2<\cdots<\tau_R\}
\tag{2.7}
\]

必须在 final holdout 之前冻结。

当前硅数据 beta 配置中的一组实现冻结值是：

\[
\{10,30,60,120,300,600,1200,2400,4800,7200\}\;\mathrm{s},
\tag{2.8}
\]

但式 (2.8) 是**当前 beta 配置**，不是跨所有工业过程的自然常数。稳定 v2.2 若要跨数据集发布，必须冻结统一的 time-bank 生成规则或按数据集预注册规则，并做 sensitivity audit。

---

# 3. Delay–State–Scale 三类时间表示

## 3.1 D：离散延迟支路

Delay branch 保留 v2.1.1 原来 profile / lag 的解释权：

\[
D_{j,t}
=
\bigl[
 x_{j,t-\ell_1},
 x_{j,t-\ell_2},
\ldots,
 x_{j,t-\ell_q}
\bigr].
\tag{3.1}
\]

D 回答：

\[
\boxed{\text{when did an informative event occur?}}
\]

即“哪个明确延迟上的历史输入有用”。

v2.2(beta) **不允许用 CT state 完全替代 D**。D 和 CT 是互补的表示族。

## 3.2 S：CT-Absolute 慢状态支路

定义：

\[
S_{j,t}
=
[z_{j,1,t},z_{j,2,t},\ldots,z_{j,R,t}].
\tag{3.2}
\]

S 回答：

\[
\boxed{\text{what slow dynamical state is the process in?}}
\]

但该解释只允许写成“slow dynamical state proxy”。

不得写成：

- 熔体真实温度；
- 未测热场的真实状态；
- 某个已被辨识出的唯一物理状态；

除非存在独立物理验证。

S 为**可选分支**。若 standardized design conditioning 不通过，必须拒绝该表示，而不是依靠极强 ridge 把病态基隐藏起来。

## 3.3 M：CT-Multires 增量支路

v2.2(beta) 的规范 CT 表示优先使用 adjacent-scale increments：

\[
m_{j,0,t}=x_{j,t}-z_{j,1,t},
\tag{3.3}
\]

\[
m_{j,r,t}=z_{j,r,t}-z_{j,r+1,t},
\qquad r=1,\ldots,R-1.
\tag{3.4}
\]

所以：

\[
M_{j,t}
=[m_{j,0,t},m_{j,1,t},\ldots,m_{j,R-1,t}].
\tag{3.5}
\]

M 回答：

\[
\boxed{\text{which dynamical timescale band is changing?}}
\]

它不是原始物理变量，而是结构可解释的多尺度动态坐标。

## 3.4 telescoping identity

M 具有：

\[
\sum_{r=0}^{R-1}m_{j,r,t}
=
x_{j,t}-z_{j,R,t}.
\tag{3.6}
\]

因此 M 把“当前值相对最慢状态的总偏离”分解为相邻时间尺度之间的增量。

式 (3.6) 说明 M 不是任意 feature engineering，而是具有确定代数结构的尺度分解。

## 3.5 不声称正交

即使 M 比 S 更容易获得较好的数值条件，也不一般有：

\[
m_{j,r}\perp m_{j,s}.
\]

因此 v2.2(beta) 只声称：

- adjacent-scale decomposition；
- 数值条件可审计；
- 分支语义可区分；

不声称 wavelet 式严格正交，也不声称统计独立。

---

# 4. K/C 与 CT 分支的关系

## 4.1 CT 不是新的 K

\(D,M,S\) 属于扩展的时间表示层。它们本身不等于物理响应算子 K。

正式关系是：

\[
E_D\to K_D\to C_D,
\]

\[
E_M\to K_M\to C_M,
\]

\[
E_S\to K_S\to C_S.
\tag{4.1}
\]

其中 \(E_M,E_S\) 通过式 (2.1)–(3.5) 构造时间状态，K/C 仍负责将这些合法输入表示映射到目标响应。

## 4.2 分支局部选择

每个可用支路应在自己的候选族内完成：

\[
\mathcal A_{K|D},\qquad
\mathcal A_{K|M},\qquad
\mathcal A_{K|S}.
\tag{4.2}
\]

同样遵守 v2.1.1：

- exact-zero 中性候选必须存在；
- one-SE 只能在同层候选比较；
- numerical rescue 不得改变结构标签；
- 选择对象和最终物化对象必须一致。

## 4.3 early fusion 的地位

原始拼接：

\[
[D_t;M_t;S_t]
\tag{4.3}
\]

不是 v2.2(beta) 的优先结构。

原因不是“理论上禁止拼接”，而是 early fusion 会：

- 混合 Delay 与 CT 的不同解释语义；
- 重新引入跨尺度共线性；
- 让一个大回归器通过正负系数抵消分支；
- 降低 branch-level certificate 的可读性。

因此 raw early fusion 在 v2.2(beta) 中只能作为：

```text
ABLATION_ONLY
```

不能绕过 \(\Gamma_{CT}\) 成为默认主路线。

---

# 5. \(\Gamma_{CT}\)：受约束预测级 late assembly

## 5.1 定义

令通过数值准入的动态分支预测为：

\[
p_D,p_M,p_S.
\]

令 persistence anchor 为 \(p_P\)。

一般目标下：

\[
p_{\Gamma,t}
=
w_Dp_{D,t}
+w_Mp_{M,t}
+w_Sp_{S,t}
+w_Pp_{P,t}.
\tag{5.1}
\]

满足：

\[
w_b\ge0,
\tag{5.2}
\]

\[
w_D+w_M+w_S+w_P=1.
\tag{5.3}
\]

若预测目标为未来变化 \(z_t\)，persistence correction 为：

\[
p_{P,t}=0.
\tag{5.4}
\]

## 5.2 beta 目标函数

在仅允许使用的 selection/validation 数据 \(\mathcal V\) 上：

\[
\widehat w
=
\arg\min_{w\in\Delta}
\frac1{|\mathcal V|}
\sum_{t\in\mathcal V}
\left(
 z_t-\sum_b w_bp_{b,t}
\right)^2
+
\lambda_\Gamma\sum_{b\ne P}w_b^2,
\tag{5.5}
\]

其中：

\[
\Delta=
\{w:w_b\ge0,\sum_bw_b=1\}.
\tag{5.6}
\]

\(\lambda_\Gamma\) 属于冻结数值配置，不属于物理结构复杂度阶梯。

## 5.3 为什么有 persistence anchor

显式 \(w_P\) 允许模型在动态支路证据不足时自动收缩：

\[
w_P\to1,
\qquad
w_{D,M,S}\to0.
\tag{5.7}
\]

而无需：

- 负权重抵消；
- 额外黑盒 gate；
- 通过无限增大 ridge 偷偷把所有 branch 缩没。

## 5.4 为什么非负

非负约束保证最终预测位于候选预测的凸包内：

\[
p_\Gamma
\in
\operatorname{conv}\{p_D,p_M,p_S,p_P\}.
\tag{5.8}
\]

这不证明某个权重是“真实物理贡献比例”，但使“模型在当前冻结协议中使用哪个时间表示、使用多少”成为可审计的结构归属。

## 5.5 final holdout 禁止训练权重

\(\Gamma_{CT}\) 的：

- branch eligibility；
- \(\lambda_\Gamma\)；
- simplex weights；
- route decisions；

均不得读取 final holdout target。

若发生：

```text
fit Gamma_CT on test target
```

则整个 v2.2(beta) 结果失效。

---

# 6. 与 v2.1.1 的 \(\Delta W\) 和 \(\mathfrak A\) 对接

## 6.1 一级 OOF 残差重新定义

冻结 branch predictors 与 \(\Gamma_{CT}\) 后，生成严格 OOF：

\[
p_{\Gamma,s}^{OOF}.
\tag{6.1}
\]

v2.1.1 的一级残差在 v2.2(beta) 中自然替换为：

\[
r_s^{(1)}
=
z_s-p_{\Gamma,s}^{OOF}.
\tag{6.2}
\]

之后 W 仍然只拟合冻结输入预测留下的一级残差。

## 6.2 \(\Delta W\) 不变

\[
\delta w_s
=
\Delta W(q_s),
\tag{6.3}
\]

\[
p_{\Gamma W,s}
=
p_{\Gamma,s}+\delta w_s.
\tag{6.4}
\]

\(\Delta W=0\) 仍然是 identity 中性候选。

## 6.3 二级 OOF 残差不变

\[
r_s^{(2)}
=
z_s-p_{\Gamma,s}^{OOF}-\delta w_s^{OOF}.
\tag{6.5}
\]

原 v2.1.1 的成熟状态模块 \(\mathfrak A\) 继续只读取满足 maturity contract 的 \(r^{(2)}\) 历史：

\[
\mathfrak A:
H_{r^{(2)}}^-	o\mathbb R.
\tag{6.6}
\]

因此：

\[
\boxed{
\Gamma_{CT}\text{ 是输入时间表示装配器；}
\quad
\mathfrak A\text{ 是成熟残差状态模块。}
}
\]

二者严格区分。

---

# 7. PF、U 与 Joint 路线

## 7.1 输入-only 路线 U

v2.2(beta) 的 U 路线可写为：

\[
\widehat z_t^U
=
p_{\Gamma,t}+\delta w_t.
\tag{7.1}
\]

## 7.2 Physical-First 路线 PF

\[
\widehat z_t^{PF}
=
p_{\Gamma,t}+\delta w_t+a_t.
\tag{7.2}
\]

训练顺序：

\[
\{D,M,S\}\to K/C
\to\Gamma_{CT}
\to\text{freeze}
\to r^{(1)}
\to\Delta W
\to\text{freeze}
\to r^{(2)}
\to\mathfrak A.
\tag{7.3}
\]

## 7.3 Joint 路线的 beta 边界

v2.2(beta) **不自动重新定义 v2.1.1 的 Joint 路线**。

稳定母本中的：

\[
\widehat z_t^J
=b_0+\Phi_t^K\beta_K+B_t^W\gamma_W+H_t^Y\beta_A
\tag{7.4}
\]

仍然成立。

CT branch 若要正式进入 Joint，必须在未来版本冻结：

- CT block 的联合参数化；
- 与 K/W/AR block 的共同 input-branch gate；
- joint numerical certificate；
- 解释权边界。

在这些合同冻结前，CT+Joint 只能标记为实验性扩展，不能声称是 v2.2(beta) 的 canonical Joint。

这条限制是故意的：v2.2(beta) 只做一个受控的小版本递进，不把所有路线一次性改写。

---

# 8. 数值证书与准入

## 8.1 conditioning certificate

对支路设计矩阵 \(X_b\)，仅用 fit 部分估计均值和尺度：

\[
\widetilde X_b
=(X_b-\mu_b)D_b^{-1}.
\tag{8.1}
\]

通过 SVD：

\[
\widetilde X_b=U\Sigma V^\top
\tag{8.2}
\]

定义：

\[
\kappa_b
=
\frac{\sigma_{\max}}{\sigma_{\min}}.
\tag{8.3}
\]

若数值秩不足，\(\kappa_b=\infty\)。

S branch 尤其必须在 ridge 之前先做此审计，禁止用强正则掩盖一个本身不可辨识的 CT basis。

当前 beta 配置保存一个 provisional hard threshold；该阈值属于冻结实现配置，不是理论自然常数。

## 8.2 M 优先不是先验“必胜”

v2.2(beta) 允许：

- M 通过；
- S 失败；
- D 通过；
- 某个 branch exact-zero。

但这不是先验规定 M 必须被选中。M 只是**规范优先 CT 表示**，仍需独立 predictive validation。

## 8.3 support certificate

CT state 的 support 问题比原始输入复杂。

当前 beta 中，support audit 只记录诊断：

```text
CT_SUPPORT_DIAGNOSTIC_ONLY
```

不能作为 universal hard kill switch。

原因是坐标分布变化不一定等价于动态结构不可迁移。

稳定后继版本需要在以下定义中做选择并冻结：

- raw CT coordinate support；
- normalized state-energy support；
- branch-level support；
- dynamical manifold support。

在此之前，不允许在 test 结果出来后改变 support 定义来“救”某一分支。

## 8.4 其他继承证书

v2.1.1 下列不变量继续适用于 v2.2(beta)：

- numerical KKT / finite coefficient；
- candidate-id consistency；
- materialized prediction consistency；
- input-path non-collapse；
- OOF target routing；
- residual maturity；
- no-cross-run；
- final holdout isolation。

---

# 9. 解释性合同

## 9.1 四层解释

v2.2(beta) 的正式解释层级为：

### D：事件延迟

\[
\boxed{D\Rightarrow\text{什么时候发生的输入变化重要}}
\]

### M：动态尺度带

\[
\boxed{M\Rightarrow\text{哪个时间尺度区间正在变化}}
\]

### S：慢动态状态代理

\[
\boxed{S\Rightarrow\text{系统处于怎样的慢动态状态代理}}
\]

### \(\Gamma_{CT}\)：结构使用比例

\[
\boxed{w_D,w_M,w_S,w_P\Rightarrow\text{冻结模型在预测中采用哪些时间表示}}
\]

## 9.2 不允许的过度解释

以下说法不由 v2.2(beta) 自动支持：

- “\(z_{40min}\) 就是真实熔体温度”；
- “某个 CT mode 被证明对应某个唯一物理机理”；
- “\(w_M=0.4\) 表示真实物理过程 40% 由 M 机制贡献”；
- “不同 CT mode 统计独立”；
- “branch 权重等于因果效应”。

因此正式定位应为：

\[
\boxed{
\text{structurally interpretable}
+
\text{dynamically interpretable}
+
\text{physics-compatible}
}
\tag{9.1}
\]

而不是“每个 latent state 都是已确认物理状态”。

## 9.3 相对 v2.1.1 的解释性变化

v2.1.1 主要提供：

\[
\text{historical profile interpretability}.
\]

v2.2(beta) 增加：

\[
\text{dynamical-state / timescale interpretability}.
\]

因此它牺牲了一点“一个 lag 就能肉眼理解”的简单性，但获得更清晰的动力学尺度解释。

---

# 10. Associative Scan 的理论地位

## 10.1 affine update

式 (2.4) 可写为：

\[
z_k=a_kz_{k-1}+c_k,
\qquad
c_k=(1-a_k)x_k.
\tag{10.1}
\]

定义 affine pair：

\[
g_k=(a_k,c_k).
\]

两个时间步的组合定义为：

\[
(a_2,c_2)\star(a_1,c_1)
=
(a_2a_1,\;c_2+a_2c_1).
\tag{10.2}
\]

则：

\[
(g_3\star g_2)\star g_1
=
g_3\star(g_2\star g_1).
\tag{10.3}
\]

所以该组合满足结合律，可用 parallel prefix / associative scan 计算。

## 10.2 scan 不改变模型语义

Associative Scan 在 v2.2(beta) 中只是：

\[
\boxed{\text{exact implementation acceleration}}
\]

而不是新的可学习模块。

只要数值精度合同满足，串行递推与 scan 必须产生同一 CT state 语义。

因此不能把“用了 scan”写成 PRISM 的物理创新；真正的理论增量是固定稳定 CT state basis 与 Delay–State–Scale 分解。

---

# 11. 逐级训练协议

## Stage B0：冻结母本与分支

必须记录：

- parent = PRISM v2.1.1；
- branch = `prism-v2-2-beta-ct`；
- beta freeze config；
- 本理论文档版本。

## Stage B1：冻结时间合同

冻结：

- target head；
- cadence；
- \(h,W,W_0,D\)；
- split / entity；
- availability；
- purge/state-carry policy。

## Stage B2：冻结 temporal candidate universe

冻结：

- Delay lag/profile candidate universe；
- \(\mathcal T_\tau\)；
- M/S construction rule；
- CT initialization/reset rule；
- conditioning threshold config；
- support diagnostic rule。

final holdout 后不得修改。

## Stage B3：构造 D/M/S

全部只读取合法历史信息。

输出：

- feature audit；
- causality audit；
- rank / condition certificate；
- candidate IDs。

## Stage B4：branch-local K/C

每个 branch 独立完成 K/C 局部选择和 OOF 预测。

禁止使用一个 early-fusion 大模型替代 branch-local selection。

## Stage B5：拟合并冻结 \(\Gamma_{CT}\)

只使用 selection/validation 数据拟合 simplex weights。

输出：

- active branches；
- branch weights；
- persistence weight；
- assembly numerical certificate；
- OOF/materialized prediction identity。

## Stage B6：\(\Delta W\)

使用：

\[
r^{(1)}=z-p_\Gamma^{OOF}.
\]

其余完全遵循 v2.1.1。

## Stage B7：\(\mathfrak A\)

使用：

\[
r^{(2)}=z-p_\Gamma^{OOF}-\delta w^{OOF}.
\]

只读成熟残差历史。

## Stage B8：final holdout

final holdout 只能：

- 物化冻结模型；
- 计算预注册指标；
- 输出 failure labels；
- 做预注册 ablation。

不能重新选择任何 v2.2(beta) 结构。

---

# 12. 理论命题

## 命题 1：CT 编码的因果性

若 \(z_{k}\) 按式 (2.4) 仅由 \(z_{k-1}\) 与当前合法输入更新，且 state initialization 不读取未来，则：

\[
z_{k}\in\mathcal I_{t_k}^U.
\]

故 M/S 均为因果可测函数。

## 命题 2：固定 CT bank 的 BIBO 稳定性

若 \(\tau_r>0\) 且输入有界，则式 (2.6) 成立。因此每个 fixed CT mode 和有限 CT bank 均 BIBO stable。

## 命题 3：M 的 telescoping 分解

按式 (3.3)–(3.4) 定义的多尺度增量满足：

\[
\sum_{r=0}^{R-1}m_r=x-z_R.
\]

因此 M 是当前状态相对最慢注册状态偏离的精确相邻尺度分解。

## 命题 4：物理时间常数对采样率具有表示不变性

对同一连续时间 \(\tau_r\)，若采样间隔变化而仍使用式 (2.3) 的精确 ZOH 系数，则模型的极点仍对应同一物理时间常数。改变采样率不需要重新解释 \(\tau_r\) 的物理尺度。

## 命题 5：凸装配的点态平方误差上界

对任意真实值 \(y\)、候选预测 \(p_b\) 与 simplex 权重 \(w_b\)：

\[
\left(y-\sum_bw_bp_b\right)^2
\le
\sum_bw_b(y-p_b)^2.
\tag{12.1}
\]

该式由平方函数凸性得到。

它说明 convex late assembly 不会依赖无限制的负权重抵消来创造预测，但不保证其必然优于每个单独候选。

## 命题 6：\(\Gamma_{CT}\) 后的逐级目标路由仍保持 v2.1.1 顺序

若 branch predictors 与 \(\Gamma_{CT}\) 均使用严格 OOF 预测并在 W 之前冻结，则：

\[
r^{(1)}=z-p_\Gamma^{OOF}
\]

仍定义一个合法冻结输入层残差目标；进一步冻结 W 后，\(r^{(2)}\) 仍可作为 \(\mathfrak A\) 的成熟目标。

因此加入 \(\Gamma_{CT}\) 不破坏 v2.1.1 的 stagewise residual ownership。

## 命题 7：Associative Scan 与串行递推代数等价

式 (10.2) 的 affine pair 组合满足结合律，因此 scan 只改变求值次序，不改变理想精确算术下的 CT state。

## 命题 8：branch 权重不产生唯一物理分解

即使 \(\Gamma_{CT}\) 的凸优化解唯一，也只说明冻结候选预测空间中的最优装配权重。若 D/M/S 表示空间重叠，则这些权重不自动等于真实机理贡献。

## 命题 9：v2.2(beta) 不改变 v2.1.1 的回退定义

v2.2(beta) 的全部新增对象都位于独立分支，且 v2.1.1 母本无需引用它们即可完整定义。因此丢弃 v2.2(beta) 不需要对 v2.1.1 做逆变换或删减重构。

---

# 13. 失败模式与硬错误

以下状态必须显式暴露，而不是静默继续：

### `CT_CAUSALITY_VIOLATION`

CT state 或初始化使用未来信息。

### `CT_CROSS_ENTITY_STATE_LEAK`

一个 run/entity 的 CT state 被带到另一个独立 run/entity。

### `CT_ABSOLUTE_CONDITIONING_FAIL`

S branch 设计矩阵数值秩不足或条件数超过冻结 hard threshold。

### `CT_EARLY_FUSION_NOT_CANONICAL`

raw D+M+S 拼接被误作 v2.2(beta) 主路线。

### `GAMMA_TEST_TUNING_VIOLATION`

\(\Gamma_{CT}\) 权重、ridge 或 active branch 使用 final holdout target 调整。

### `GAMMA_SIMPLEX_VIOLATION`

出现负权重或权重和不为 1。

### `CT_SUPPORT_OVERCLAIM`

把 beta 的 diagnostic-only CT support 当成已经冻结的 universal physical support certificate。

### `CT_LATENT_PHYSICAL_OVERCLAIM`

未经独立物理验证，把 CT latent state 直接命名为真实未测物理量。

### `A_SYMBOL_COLLISION`

把 \(\Gamma_{CT}\) 写成 v2.1.1 的 \(\mathfrak A\)，从而混淆“时间支路装配”和“成熟残差状态”。

---

# 14. v2.2(beta) 明确排除项

以下内容**不属于**本文定义的 v2.2(beta)：

1. learned / free \(\tau_r\)；
2. input-dependent pole；
3. unrestricted time-varying \(A_t\)；
4. physics-first trunk 中的 Mamba / selective SSM；
5. unconstrained neural router；
6. 用深度网络 end-to-end 替换 K/C/\(\Delta W\)/A；
7. 根据 final holdout 自动选择 time-bank；
8. 把 CT latent state 声称为已测物理量；
9. 在未确认物理身份时把两个数据 Sheet 写成已确认独立晶棒；
10. 把 CT 扩展写回 v2.1.1 稳定版定义。

这些内容若未来进入 PRISM，必须拥有新的版本号和新的 freeze contract。

---

# 15. 从 beta 到稳定 v2.2 的必要条件

PRISM v2.2(beta) 要升级为稳定 v2.2，至少应完成：

- 将 D/M/S 与 \(\Gamma_{CT}\) 接入 canonical C4/K/C/W/A stage contracts；
- 统一 branch candidate ID、OOF materialization 与 final contract；
- 在全部公开数据集上按与 v2.1.1 完全一致的 split 重跑；
- 完成 D-only / M-only / S-only / D+M late assembly / early-fusion ablation / persistence-anchor ablation；
- 完成 \(\tau_{min},\tau_{max}\)、pole density、time-bank spacing 的 sensitivity audit；
- 完成 conditioning threshold sensitivity audit；
- 冻结 CT Native Support 的正式定义；
- 检查 irregular cadence 与 state-carry contract；
- 验证 scan 与 serial implementation 的数值等价；
- 完成 interpretability audit 与 failure-mode audit；
- 决定 Joint 路线是否正式吸收 CT representation；
- 只有在上述合同冻结后，才重新生成正式 benchmark tables。

在此之前：

\[
\boxed{\text{PRISM v2.2(beta) remains experimental.}}
\]

---

# 16. 最终版本边界

PRISM v2.2(beta) 的核心可以压缩为：

\[
\boxed{
\text{PRISM v2.1.1}
+
\text{fixed stable CT state basis}
+
\text{Delay–State–Scale decomposition}
+
\text{certified branch-local prediction}
+
\text{constrained }\Gamma_{CT}\text{ late assembly}
}
\tag{16.1}
\]

其中：

\[
D=\text{when},
\qquad
M=\text{which timescale is changing},
\qquad
S=\text{what slow dynamical state proxy},
\tag{16.2}
\]

而 \(\Gamma_{CT}\) 回答：

\[
\boxed{\text{which admissible temporal explanation should the frozen model trust, and how much?}}
\]

其后仍是 v2.1.1 的：

\[
\Delta W\to\mathfrak A.
\]

因此 v2.2(beta) 的研究主旨不是“把 PRISM 变成 Mamba”，而是：

\[
\boxed{
\text{把 PRISM 的历史 profile 解释扩展为稳定、可审计的动态状态与时间尺度解释。}
}
\]

---

# 17. 回退权威声明

再次冻结：

若未来任务明确要求：

- “整理 PRISM v2.1.1”；
- “按稳定版本说明 PRISM”；
- “回退 v2.2(beta)”；
- “不考虑 CT 分支重新组织理论”；

则本文必须被视为：

```text
OUT_OF_SCOPE_FOR_V2_1_1_RECONSTRUCTION
```

除非用户明确说“同时参考 v2.2(beta)”。

正确的回退关系是：

```text
PRISM v2.2(beta) isolated branch
        |
        | discard / ignore
        v
PRISM v2.1.1 frozen theory + frozen code + frozen contracts
```

而不是：

```text
PRISM v2.2(beta)
        |
        | delete CT paragraphs and guess
        v
supposed v2.1.1
```

**v2.1.1 从始至终独立完整；v2.2(beta) 只是一条可随时丢弃的实验性递进分支。**
