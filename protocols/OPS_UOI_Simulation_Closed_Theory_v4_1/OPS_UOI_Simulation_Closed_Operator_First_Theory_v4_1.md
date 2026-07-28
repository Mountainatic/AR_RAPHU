# Simulation-Closed Orthogonal Predictive-State Urysohn Operator Identification v4.1
## 全局可计算、递推闭合与认证解释分离的 operator-first 理论

> **简称**：OPS-UOI  
> **理论对象**：固定阶段、闭环观测分布下的条件预测 Urysohn 投影算子  
> **不声称**：无条件 plant transfer、跨阶段统一动力学、未测突变的事前预测  
> **v4.1 冻结定位**：PS-AR-RAPHU 与 ARX、pNARX、MLP-NARX、RNN 等处于同一系统辨识层级；它必须作为一个固定辨识模型完成 free-run simulation，而不是依赖额外控制器或在线更新来补救。
>
> **v4.1 核心新增**：严格区分数学定义域、训练支撑域和 K 层认证域；引入有界 \(C^1\) continuation，使模型全局可计算且 free-run 有界；同时建立域外解释防火墙、递推稳定性证书以及 H2 原生模型选择闭环。
>
> **保持不变**：v4.0 的 operator-first 识别对象、Q/K 分层、双残差正交化、有限 sieve coercivity、predictive/structural rank 边界全部保留。

---

# 1. 原始过程、信息集和固定阶段

设原始随机过程为

\[
\mathscr W_t
=
(Y_t,X_{1,t},\ldots,X_{p,t},U_t,\ldots).
\]

全文固定在一个工艺阶段 \(r\) 内。阶段切换允许改变状态空间、预测坐标、算子和参数，不属于单一模型的鲁棒性要求。

预测原点为 \(t\)，视野为 \(h\ge1\)。定义

\[
Z_t=(Y_t,\ldots,Y_{t-L_y+1})
\]

和变量 \(j\) 的输入历史

\[
W_{j,t}=(X_{j,t},\ldots,X_{j,t-L_x+1}).
\]

目标为

\[
Y_t^+=Y_{t+h}.
\]

主轨道不使用 \(X_{t+1:t+h}\)。

---

# 2. 与实现一致的 \(L^2\) 核空间

变量 \(j\) 的幅值参考域为 \(\mathcal I_j\)，参考测度为 \(\nu_j\)。定义

\[
\mathcal H_j
=
\mathbb R^{L_x}\otimes L^2(\mathcal I_j,\nu_j),
\]

\[
\mathcal H
=
\bigoplus_{j=1}^p\mathcal H_j.
\]

范数为

\[
\|K\|_{\mathcal H}^2
=
\sum_{j=1}^p
\sum_{\tau=0}^{L_x-1}
\int_{\mathcal I_j}
K_j(\tau,u)^2\,d\nu_j(u).
\tag{2.1}
\]

该范数就是结构谱使用的 Hilbert–Schmidt 几何。

## A1：有界设计密度

对所有 \(j,\tau\)，\(X_{j,t-\tau}\) 相对 \(\nu_j\) 的边际密度满足

\[
\frac{dP_{j,\tau}}{d\nu_j}(u)
\le \bar p_j
\quad \nu_j\text{-a.e.}
\tag{2.2}
\]

定义 Urysohn 映射

\[
(\mathcal A K)_t
=
\sum_{j=1}^p
\sum_{\tau=0}^{L_x-1}
K_j(\tau,X_{j,t-\tau}).
\tag{2.3}
\]

## 定理 2.1：总体 Urysohn 映射有界

在 A1 下，

\[
\|\mathcal A K\|_{L^2(\mathbb P)}
\le
C_A\|K\|_{\mathcal H},
\tag{2.4}
\]

其中可取

\[
C_A^2
\le
pL_x
\max_j\bar p_j
\]

乘以参考测度权重常数。

### 证明

由

\[
\left|\sum_{q=1}^{pL_x}z_q\right|^2
\le pL_x\sum_q|z_q|^2
\]

及密度上界，

\[
\mathbb E K_j(\tau,X)^2
\le
\bar p_j
\int K_j(\tau,u)^2d\nu_j(u).
\]

求和即得。

> 这里不需要假定无限维 \(L^2\) 中点值泛函有界。样本评价只在有限 spline sieve 内显式计算。

---

# 3. 条件中心化算子和精确识别对象

令

\[
\Pi_Zf=\mathbb E[f\mid Z_t]
\]

为 \(L^2\) 中投影到 \(Z_t\)-可测子空间的正交投影。定义

\[
\mathcal C_Z=I-\Pi_Z,
\]

\[
\mathcal D=\mathcal C_Z\mathcal A.
\tag{3.1}
\]

因此

\[
(\mathcal DK)_t
=
\mathcal A_tK
-
\mathbb E[\mathcal A_tK\mid Z_t].
\tag{3.2}
\]

\(\mathcal D:\mathcal H\to L^2(\mathbb P)\) 有界，且

\[
\Gamma=\mathcal D^\ast\mathcal D
\tag{3.3}
\]

为自伴半正定算子。

定义输出残差

\[
R_t
=
Y_t^+
-
\mu_0(Z_t),
\qquad
\mu_0(Z)=\mathbb E[Y_t^+\mid Z_t=Z].
\tag{3.4}
\]

## 3.1 总是存在的贡献目标

定义

\[
g_0
=
P_{\overline{\operatorname{ran}\mathcal D}}R
\in
\overline{\operatorname{ran}\mathcal D}.
\tag{3.5}
\]

它是给定 predictive state 后，能够由加性 Urysohn 历史解释的最佳 \(L^2\) 贡献。

残差

\[
\varepsilon=R-g_0
\tag{3.6}
\]

满足

\[
\mathbb E[\varepsilon\mid Z_t]=0,
\qquad
\mathcal D^\ast\varepsilon=0.
\tag{3.7}
\]

这一定义即使真实系统含有未建模交互也成立。此时 \(g_0\) 是最佳加性预测投影，而不是完整物理机制。

## 3.2 需要 source condition 才存在的核目标

### A2：核可表示性

假设

\[
g_0\in\operatorname{ran}\mathcal D.
\tag{3.8}
\]

则存在 \(K\in\mathcal H\) 使 \(\mathcal DK=g_0\)。取最小 \(\mathcal H\)-范数代表：

\[
K_0
=
\mathcal D^\dagger g_0
\in
(\ker\mathcal D)^\perp.
\tag{3.9}
\]

若 A2 不成立，理论仍识别 \(g_0\)，但不能声称存在有限范数完整核。

## 定理 3.1：商空间可辨识性

\[
\ker\mathcal D
=
\ker\Gamma.
\tag{3.10}
\]

映射

\[
[K]\mapsto\mathcal DK
\]

将

\[
\mathcal H/\ker\mathcal D
\]

等距映到 \(\operatorname{ran}\mathcal D\)。该 quotient 在

\[
\|[K]\|_\Gamma=\|\mathcal DK\|_{L^2}
\]

下的完备化与

\[
\overline{\operatorname{ran}\mathcal D}
\]

等距同构。

> 严格地说，未完备 quotient 本身只映到 \(\operatorname{ran}\mathcal D\)，不是自动满射到其闭包。

---

# 4. 预测辨识和结构辨识是两个层次

## Q 层：贡献/quotient 辨识

Q 层只要求估计

\[
g_0=\mathcal DK_0
\]

或 quotient class。它允许 \(\ker\mathcal D\ne\{0\}\)。

## K 层：完整核辨识

对结构 sieve \(V_m\subset\mathcal H\)，要求：

### A3：sieve injectivity

\[
\langle K,\Gamma K\rangle
\ge
\kappa_m\|K\|_{\mathcal H}^2,
\qquad
\forall K\in V_m.
\tag{4.1}
\]

即

\[
V_m\cap\ker\mathcal D=\{0\}.
\]

只有在 A3 下，\(V_m\) 内的核坐标、HS 误差和结构 rank 才由数据稳定识别。

ridge 使正规方程可解，但若 A3 失败，ridge 选择的是一个规则化代表，而不是被数据唯一决定的物理核。

---

# 5. 有限 sieve 和样本设计

令

\[
V_1\subset\cdots\subset V_M\subset\mathcal H,
\qquad
\dim V_m=d_m.
\]

选择 \(\mathcal H\)-正交基

\[
\{\phi_{m,1},\ldots,\phi_{m,d_m}\}.
\]

定义原始设计坐标

\[
x_{m,t,k}
=
(\mathcal A\phi_{m,k})_t,
\tag{5.1}
\]

\[
x_{m,t}
=
(x_{m,t,1},\ldots,x_{m,t,d_m})^\top.
\]

定义 nuisance

\[
\pi_{0,m}(Z)
=
\mathbb E[x_{m,t}\mid Z_t=Z],
\tag{5.2}
\]

残差设计

\[
d_{m,t}
=
x_{m,t}-\pi_{0,m}(Z_t).
\tag{5.3}
\]

总体 Gram：

\[
\Gamma_m
=
\mathbb E[d_{m,t}d_{m,t}^\top].
\tag{5.4}
\]

## 5.1 sieve 投影目标

无论 A2 是否成立，定义

\[
\theta_m^\star
=
\arg\min_{\theta\in\mathbb R^{d_m}}
\|g_0-d_m^\top\theta\|_{L^2}^2.
\tag{5.5}
\]

在 A3 下唯一，并满足

\[
\Gamma_m\theta_m^\star
=
\mathbb E[d_{m,t}g_{0,t}].
\tag{5.6}
\]

令

\[
K_m^\star
=
\sum_k\theta_{m,k}^\star\phi_{m,k}.
\]

贡献逼近误差：

\[
a_{m,\Gamma}
=
\|g_0-\mathcal DK_m^\star\|_{L^2}.
\tag{5.7}
\]

若 A2 成立，再定义 HS 逼近误差：

\[
a_{m,HS}
=
\|K_m^\star-K_0\|_{\mathcal H}.
\tag{5.8}
\]

---

# 6. sieve-level 双残差 score

对候选 \((\theta,\mu,\pi)\)，定义

\[
\psi_{m,t}(\theta,\mu,\pi)
=
(x_{m,t}-\pi(Z_t))
\left[
Y_t^+-\mu(Z_t)
-
(x_{m,t}-\pi(Z_t))^\top\theta
\right].
\tag{6.1}
\]

## 定理 6.1：一阶 Neyman 正交性

在

\[
(\theta_m^\star,\mu_0,\pi_{0,m})
\]

处，对任意平方可积方向 \(h_\mu,h_\pi\)，

\[
\partial_\mu
\mathbb E\psi_{m,t}[h_\mu]
=0,
\tag{6.2}
\]

\[
\partial_\pi
\mathbb E\psi_{m,t}[h_\pi]
=0.
\tag{6.3}
\]

该结论只使用：

\[
\mathbb E[d_{m,t}\mid Z_t]=0,
\]

\[
\mathbb E[d_{m,t}e_{m,t}]=0,
\qquad
\mathbb E[e_{m,t}\mid Z_t]=0,
\]

其中

\[
e_{m,t}
=
Y_t^+-\mu_0(Z_t)-d_{m,t}^\top\theta_m^\star.
\]

## 定理 6.2：精确 nuisance 偏差

令

\[
\delta_\mu=\mu-\mu_0,
\qquad
\delta_\pi=\pi-\pi_{0,m}.
\]

则

\[
\mathbb E[
\psi_{m,t}(\theta_m^\star,\mu,\pi)
]
=
\mathbb E[\delta_\pi\delta_\mu]
-
\mathbb E[\delta_\pi\delta_\pi^\top]\theta_m^\star.
\tag{6.4}
\]

故

\[
\left\|
\mathbb E\psi_{m,t}
\right\|_2
\le
r_{\pi,m}r_{\mu,m}
+
R_mr_{\pi,m}^2,
\tag{6.5}
\]

其中

\[
R_m=\|\theta_m^\star\|_2.
\]

该结果不要求把 \(K_0\) 解释为因果 plant。

---

# 7. 前向 cross-fitting 的正确时间间隔

设一个预测原点 \(t\) 使用的 primitive-process 索引支持为

\[
[t-L_\star,\ t+h],
\qquad
L_\star=\max(L_x-1,L_y-1).
\tag{7.1}
\]

第 \(k\) 个评价块原点为

\[
\mathcal I_k=[s_k,e_k].
\]

nuisance 训练原点最多到

\[
s_k-G_{n,k}-1.
\]

定义额外 mixing gap

\[
b_{n,k}
=
G_{n,k}-L_\star-h.
\tag{7.2}
\]

要求

\[
G_{n,k}
=
L_\star+h+b_{n,k},
\qquad
b_{n,k}>0.
\tag{7.3}
\]

Berbee coupling 误差为

\[
\beta(b_{n,k}),
\tag{7.4}
\]

不是 \(\beta(G_{n,k})\)。

几何 mixing 下可取

\[
b_{n,k}\asymp c_b\log N.
\]

---

# 8. 概率假设

## A4：固定阶段平稳与几何 mixing

派生序列在阶段内严格平稳，primitive \(\beta\)-mixing 系数满足

\[
\beta(q)\le C_\beta e^{-c_\beta q}.
\tag{8.1}
\]

## A5：条件包络

条件于每折 nuisance 训练数据，评价块上的：

- \(d_{m,t}\)；
- \(\delta_{\mu,t}\)；
- \(\delta_{\pi,t}\)；
- sieve 残差；
- 乘积 score；

满足统一有界或 Bernstein/Orlicz 矩条件。

区分：

\[
r_{\mu,m},r_{\pi,m}
\]

为 \(L^2\) 误差，

\[
\bar r_{\mu,m},\bar r_{\pi,m}
\]

为条件包络尺度。

## A6：逼近残差矩控制

令

\[
u_{m,t}
=
g_{0,t}-d_{m,t}^\top\theta_m^\star.
\]

要求其 Bernstein 矩尺度与

\[
a_{m,\Gamma}=\|u_m\|_{L^2}
\]

成比例，或显式使用单独包络 \(B_{a,m}\)。

---

# 9. 正交 ridge sieve 估计器

第 \(k\) 折使用过去训练得到

\[
\widehat\mu_k,\qquad \widehat\pi_{k,m}.
\]

对 \(t\in\mathcal I_k\)：

\[
\widehat y_{t}^{\perp}
=
Y_t^+-\widehat\mu_k(Z_t),
\]

\[
\widehat d_{m,t}
=
x_{m,t}-\widehat\pi_{k,m}(Z_t).
\]

拼接评价折，求解

\[
\widehat\theta_m
=
\arg\min_\theta
\frac1{2N}
\sum_{k}
\sum_{t\in\mathcal I_k}
(\widehat y_t^\perp-\widehat d_{m,t}^\top\theta)^2
+
\frac12\theta^\top P_m\theta.
\tag{9.1}
\]

---

# 10. 确定性主不等式

定义

\[
\widehat H_m
=
\frac1N\sum_t
\widehat d_{m,t}\widehat d_{m,t}^\top+P_m,
\]

以及在 \(\theta_m^\star\) 处的 score

\[
\widehat S_m
=
\frac1N
\sum_t
\widehat d_{m,t}
(\widehat y_t^\perp-\widehat d_{m,t}^\top\theta_m^\star).
\]

## 定理 10.1：确定性误差分解

若

\[
\lambda_{\min}(\widehat H_m)\ge\underline\kappa_m>0,
\tag{10.1}
\]

则

\[
\|\widehat\theta_m-\theta_m^\star\|_2
\le
\frac1{\underline\kappa_m}
\left[
\|\widehat S_m-\mathbb E^\star\widehat S_m\|_2
+
\|\mathbb E^\star\widehat S_m\|_2
+
\|P_m\theta_m^\star\|_2
\right],
\tag{10.2}
\]

其中 \(\mathbb E^\star\) 表示 coupling 后、条件于 nuisance 训练数据的评价期望。

该定理把：

- 随机波动；
- nuisance 二阶偏差；
- regularization bias；

彻底分开，不依赖某个特定 mixing inequality 的常数。

---

# 11. 多折 mixing 高概率界

设第 \(k\) 折大小为 \(n_k\)，总评价样本

\[
N=\sum_kn_k.
\]

定义

\[
q_{cf,m}(\delta)
=
\sum_k
\frac{n_k}{N}
q_{n_k,d_m}(\delta/K_f),
\tag{11.1}
\]

其中

\[
q_{n,d}(\alpha)
=
C
\left[
\sqrt{\frac{d+\log(1/\alpha)}n}
+
\frac{(d+\log(1/\alpha))\log^2n}{n}
\right].
\tag{11.2}
\]

## 定理 11.1：固定 sieve 高概率误差

在 A1–A6 和 A3 的 K 层条件下，以至少

\[
1-\delta-\sum_{k=1}^{K_f}\beta(b_{n,k})
\tag{11.3}
\]

的概率，

\[
\begin{aligned}
\|\widehat\theta_m-\theta_m^\star\|_2
\le
\frac{C}{\kappa_m}
\Big[
&
(\sigma_m+a_{m,\Gamma}
+\bar r_{\mu,m}
+R_m\bar r_{\pi,m})
q_{cf,m}
\\
&
+r_{\mu,m}r_{\pi,m}
+R_mr_{\pi,m}^2
+b_{\lambda,m}
\Big],
\end{aligned}
\tag{11.4}
\]

只要 Gram 扰动小于 \(\kappa_m/2\)。

其中

\[
b_{\lambda,m}
=
\|P_m\theta_m^\star\|_2.
\]

于是贡献误差：

\[
\|\mathcal D(\widehat K_m-K_m^\star)\|_{L^2}
\le
\sqrt{\Lambda_m}
\|\widehat\theta_m-\theta_m^\star\|_2.
\tag{11.5}
\]

总贡献误差：

\[
\|\mathcal D\widehat K_m-g_0\|_{L^2}
\le
a_{m,\Gamma}
+
\sqrt{\Lambda_m}
\|\widehat\theta_m-\theta_m^\star\|_2.
\tag{11.6}
\]

若 A2 成立，则 HS 核误差满足：

\[
\|\widehat K_m-K_0\|_{\mathcal H}
\le
a_{m,HS}
+
\|\widehat\theta_m-\theta_m^\star\|_2.
\tag{11.7}
\]

> 式 (11.7) 是 rank 理论真正需要的界；不能用式 (11.6) 自动替代。

---

# 12. 一致性和幅值 sieve 速率

预测贡献一致性需要：

\[
a_{m,\Gamma}\to0,
\quad
q_{cf,m}\to0,
\quad
r_\mu r_\pi+R_mr_\pi^2\to0,
\quad
b_{\lambda,m}\to0,
\]

且 \(\kappa_m\) 不过快退化。

结构核一致性还额外需要：

\[
a_{m,HS}\to0.
\]

若 \(p,L_x,K_f\) 固定，幅值截面具有 \(s\) 阶 \(L^2\)-Sobolev 光滑度，

\[
a_{m,\Gamma}\lesssim M_x^{-s},
\qquad
d_m\asymp M_x,
\]

且 injectivity 常数有正下界，则主随机率仍为

\[
\left(
\frac{\log^2N}{N}
\right)^{s/(2s+1)}
\tag{12.1}
\]

加上 nuisance 二阶项。

若 \(\kappa_m\) 随分辨率下降，该逆问题病态因子必须显式保留。

---

# 13. Lepski 自适应分辨率

对贡献目标使用

\[
\|\mathcal D(\widehat K_m-\widehat K_\ell)\|_{L^2}
\]

或独立 selection block 上的经验对应量。

假设存在同时覆盖半径

\[
\|\mathcal D(\widehat K_m-K_m^\star)\|_{L^2}
\le\rho_m,
\qquad \forall m,
\tag{13.1}
\]

且 \(\rho_m\) 单调不减。

定义

\[
\widehat m
=
\min\left\{
m:
\|\mathcal D(\widehat K_m-\widehat K_\ell)\|_{L^2}
\le4\rho_\ell,\ 
\forall\ell\ge m
\right\}.
\tag{13.2}
\]

crossing index：

\[
m^\circ
=
\min\{m:a_{m,\Gamma}\le\rho_m\}.
\]

则在覆盖事件上：

\[
\widehat m\le m^\circ,
\]

\[
\|\mathcal D\widehat K_{\widehat m}-g_0\|_{L^2}
\le6\rho_{m^\circ}.
\tag{13.3}
\]

只有再加入 crossing regularity：

\[
\rho_{m^\circ}
\le
C_o\inf_m(a_{m,\Gamma}+\rho_m)
\tag{13.4}
\]

时，才能称为 oracle inequality。

结构分辨率若用于 rank，还必须同时控制 \(a_{m,HS}\)，不能只依据自然贡献范数。

---

# 14. 多变量块可辨识性

有限 sieve Gram 按变量分块：

\[
\Gamma_m=
\begin{bmatrix}
\Gamma_{jj}&\Gamma_{j,-j}\\
\Gamma_{-j,j}&\Gamma_{-j,-j}
\end{bmatrix}.
\]

在值域相容条件下，广义 Schur 补

\[
S_{j,m}
=
\Gamma_{jj}
-
\Gamma_{j,-j}
\Gamma_{-j,-j}^{\dagger}
\Gamma_{-j,j}
\tag{14.1}
\]

满足：

\[
\inf_v
\begin{bmatrix}u\\v\end{bmatrix}^{\!\top}
\Gamma_m
\begin{bmatrix}u\\v\end{bmatrix}
=
u^\top S_{j,m}u.
\]

变量 \(j\) 在该 sieve 中可独立归因，当且仅当

\[
S_{j,m}\succ0.
\tag{14.2}
\]

实验必须同时报告：

- \(\lambda_{\min}(\Gamma_m)\)；
- \(\lambda_{\min}(S_{j,m})\)；
- 条件数；
- ridge 前后数值稳定性。

ridge 后条件数好转不等于 Schur 可辨识性改善。

---

# 15. 结构 rank：必须满足 K 层条件

对变量核 \(K_{0j}\)，视为从幅值 \(L^2\) 到有限 lag 空间的 Hilbert–Schmidt 算子。

定义

\[
\tau_R(K)
=
\frac{
\inf_{\operatorname{rank}(A)\le R}
\|K-A\|_{HS}
}{
\|K\|_{HS}
}.
\tag{15.1}
\]

## A7：结构 rank 可报告条件

1. A2 核可表示性；
2. 结构 sieve injectivity；
3. \(a_{m,HS}\to0\)；
4. 信号强度
   \[
   \|K_{0j}\|_{HS}\ge s_0>0;
   \]
5. rank margin。

若任何一项失败，输出：

```text
STRUCTURAL_RANK_NOT_IDENTIFIED
```

而不是强制给 rank。

## 定理 15.1：可观测尾比误差界

若

\[
\|\widehat K-K_0\|_{HS}\le\delta_U
\]

且 \(\|\widehat K\|_{HS}>0\)，则对所有 \(R\)：

\[
|\tau_R(\widehat K)-\tau_R(K_0)|
\le
\eta_U
:=
\min\left\{
1,
\frac{2\delta_U}{\|\widehat K\|_{HS}}
\right\}.
\tag{15.2}
\]

若

\[
\|\widehat K\|_{HS}\le2\delta_U,
\]

则信号与零算子不可稳定区分，应报告：

```text
WEAK_OPERATOR_OR_INACTIVE
```

## 15.2 rank interval

定义

\[
R_L
=
\min\{R:\widehat\tau_R\le\varepsilon+\eta_U\},
\]

\[
R_U
=
\min\{R:\widehat\tau_R\le\varepsilon-\eta_U\}.
\]

若第二集合为空，则上界在当前 \(R_{\max}\) 内不可认证。

在有效误差球内：

\[
R_L
\le
R_S^\star(\varepsilon)
\le
R_U.
\tag{15.3}
\]

有 margin \(\gamma>\eta_U\) 时，单值 rank 恢复。

---

# 16. predictive rank 的两个版本

## 16.1 最优预测 rank

\[
d_{P,R}^{opt}(K_0)
=
\inf_{\operatorname{rank}(A)\le R}
\|\mathcal D(K_0-A)\|_{L^2(\mathbb P)}.
\tag{16.1}
\]

## 16.2 结构 SVD 截断的预测 rank

令 \(K_{0,R}^{HS}\) 为 HS-SVD rank-\(R\) 截断：

\[
d_{P,R}^{svd}(K_0)
=
\|\mathcal D(K_0-K_{0,R}^{HS})\|_{L^2(\mathbb P)}.
\tag{16.2}
\]

显然

\[
d_{P,R}^{opt}
\le
d_{P,R}^{svd}.
\tag{16.3}
\]

当前 NAT/PERM 实验估计的是：

\[
R_{P,\mathrm{svd}}^\star,
\]

不是一般的 \(R_{P,\mathrm{opt}}^\star\)。

结构尾提供：

\[
d_{P,R}^{svd}(K_0)
\le
\|\mathcal D\|_{op}
\,d_R^{HS}(K_0).
\]

---

# 17. 不同分布的正确比较方式

对固定核 \(K\) 和固定映射 \(f(W)\)，若有限窗口分布满足

\[
0<c\le\frac{d\mathbb Q}{d\mathbb P}\le C,
\]

则 \(L^2\) 范数可比较。

但 residualized 算子

\[
\mathcal D_{\mathbb P}
=
(I-\Pi_{Z,\mathbb P})\mathcal A
\]

随分布变化。

因此：

- 冻结 \(\pi^\dagger\) 后可以比较 P/Q；
- 分别重新拟合 \(\pi_{\mathbb P},\pi_{\mathbb Q}\) 时，不能无条件比较 null space；
- PERM/SPACE 结果应解释为不同诊断环境，不是 NAT estimand 的直接一致性证据。

---

# 18. 与传统 closed-loop plant identification 的边界

OPS-UOI 的 \(K_0\) 是闭环观测分布下的最佳条件预测算子。

它不自动等于开放环 plant。要赋予 plant/causal 解释，需要额外条件，例如：

- 已知控制器与噪声结构；
- 合法 instrumental variables；
- sequential exogeneity；
- 专门的 closed-loop experiment design。

本文不以这些条件作为主贡献，因此必须使用：

```text
conditional predictive operator
```

而不是：

```text
causal process transfer kernel
```

---

# 19. v4.0 统一主结论

在固定阶段、A1–A6、有限嵌套 sieve、正确 forward gap、几何 mixing、fold-wise nuisance 率和 crossing regularity 下：

1. 数据总能识别最佳加性 Urysohn 贡献 \(g_0\)；
2. 若 source condition 成立，识别 quotient class \([K_0]\)；
3. 若结构 sieve injectivity 成立，可得到核坐标和 HS 误差界；
4. nuisance 误差以
   \[
   r_\mu r_\pi+Rr_\pi^2
   \]
   二阶形式进入；
5. Lepski 可自适应选择贡献分辨率；
6. 只有在 HS-bias、信号强度和 margin 同时成立时，结构 rank 才可恢复；
7. 否则应报告 predictive rank、rank interval 或不可辨识状态。

这是一条可防守的理论链，而不是把数值唯一解误当成科学可辨识性。

---

# 20. 系统辨识层级与 v4.1 的任务边界

## 20.1 与 baseline 的同级关系

给定训练输入输出序列

\[
\mathcal D_{\mathrm{tr}}
=
\{(X_t,Y_t)\}_{t\in\mathcal T_{\mathrm{tr}}},
\]

ARX、pNARX、MLP-NARX、RNN 和 PS-AR-RAPHU 均执行同一类任务：

\[
\mathcal D_{\mathrm{tr}}
\longmapsto
\widehat{\mathcal M}.
\tag{20.1}
\]

随后固定 \(\widehat{\mathcal M}\)，在 validation/test 上进行：

1. 一步或 direct prediction；
2. free-run simulation。

因此 PS-AR-RAPHU 的主模型不得依赖：

- 额外 MPC；
- Knowledge Planner；
- 在线读取测试输出重新训练；
- 另一套 baseline 提供最终模型；
- 域外失败后临时改变模型定义。

H3 使用 ARX history 仅是公平消融；H2 才是主方法。

## 20.2 prediction 与 simulation

定义 teacher-forced 一步预测：

\[
\widehat Y_{t+1|t}^{\mathrm{TF}}
=
\widehat{\mathcal M}
\left(
X_{\le t},
Y_{\le t}
\right).
\tag{20.2}
\]

定义初始化长度 \(L_y\) 后的 free-run：

\[
\widehat Y_{t+1}^{\mathrm{FR}}
=
\widehat{\mathcal M}
\left(
X_{\le t},
\widehat Y_{\le t}^{\mathrm{FR}}
\right),
\qquad
t\ge t_0+L_y.
\tag{20.3}
\]

其中不读取中间真实输出。两类指标必须分栏：

```text
DIRECT_OR_TEACHER_FORCED
FREE_RUN_SIMULATION
```

direct 好不能替代 simulation 完成；simulation 差也不否定其作为 soft sensor 的 direct 价值。

---

# 21. 三类域：数学定义、训练支撑与认证解释

对每个外生通道 \(j\) 和 AR 输出通道 \(y\)，分别定义：

\[
\mathcal S_{j}^{\mathrm{cert}}
\subseteq
\mathcal S_{j}^{\mathrm{train}}
\subseteq
\mathcal D_{j}^{\mathrm{model}},
\tag{21.1}
\]

\[
\mathcal S_{y}^{\mathrm{cert}}
\subseteq
\mathcal S_{y}^{\mathrm{train}}
\subseteq
\mathcal D_{y}^{\mathrm{model}}.
\tag{21.2}
\]

## 21.1 数学定义域

\[
\mathcal D_j^{\mathrm{model}}
\]

是响应函数或核曲线能够被数学计算的范围。v4.1 主模型通过 continuation 取：

\[
\mathcal D_j^{\mathrm{model}}=\mathbb R,
\qquad
\mathcal D_y^{\mathrm{model}}=\mathbb R.
\tag{21.3}
\]

这只表示模型不会因有限 spline 区间而停止。

## 21.2 训练支撑域

训练支撑由训练数据和预注册的 train-only padding 规则确定：

\[
\mathcal S_j^{\mathrm{train}}
=
[a_j,b_j].
\tag{21.4}
\]

validation/test 不得改变 \(a_j,b_j\)、basis knots、scaler 或 continuation 参数候选集。

## 21.3 认证域

\[
\mathcal S_j^{\mathrm{cert}}
\]

是训练支撑中满足局部样本覆盖、有限 sieve coercivity、置信半径和模型错设门禁的部分。

认证域外的声明规则为：

| 区域 | 数值计算 | Q 层预测声明 | K 层结构声明 |
|---|---:|---:|---:|
| \(\mathcal S^{\mathrm{cert}}\) | 允许 | 允许 | 满足 K 层条件时允许 |
| \(\mathcal S^{\mathrm{train}}\setminus\mathcal S^{\mathrm{cert}}\) | 允许 | 经验预测允许 | 禁止完整 K 解释 |
| \(\mathcal D^{\mathrm{model}}\setminus\mathcal S^{\mathrm{train}}\) | 允许 | 仅 continuation 诊断 | 禁止 Q/K 识别声明 |

域外状态必须标记：

```text
MATHEMATICALLY_DEFINED
OUTSIDE_TRAIN_SUPPORT
Q_IDENTIFICATION_NOT_CLAIMED
K_INTERPRETATION_FORBIDDEN
```

## 21.4 逐通道掩码

定义

\[
m_{j,t}^{\mathrm{train}}
=
\mathbf 1\{X_{j,t}\in\mathcal S_j^{\mathrm{train}}\},
\tag{21.5}
\]

\[
m_{j,t}^{\mathrm{cert}}
=
\mathbf 1\{X_{j,t}\in\mathcal S_j^{\mathrm{cert}}\},
\tag{21.6}
\]

以及 AR 历史掩码

\[
m_{y,t,\ell}^{\mathrm{train}}
=
\mathbf 1\{\widetilde Y_{t-\ell}\in\mathcal S_y^{\mathrm{train}}\}.
\tag{21.7}
\]

free-run 必须区分：

- 原始外生输入 OOD；
- 模型递推输出 OOD；
- 两者同时发生。

---

# 22. 有界 \(C^1\) continuation

## 22.1 标量响应函数

设训练域内响应

\[
f\in C^1([a,b]).
\]

给定左右 continuation 尺度 \(\rho_->0,\rho_+>0\)，定义：

\[
\widetilde f(u)
=
\begin{cases}
f(a)+\rho_- f'(a)
\tanh\!\left(\dfrac{u-a}{\rho_-}\right),
&u<a,\\[3mm]
f(u),
&a\le u\le b,\\[2mm]
f(b)+\rho_+ f'(b)
\tanh\!\left(\dfrac{u-b}{\rho_+}\right),
&u>b.
\end{cases}
\tag{22.1}
\]

它不是域外物理模型，而是固定辨识模型的 simulation closure。

## 22.2 核曲线

对 full Urysohn 核，逐 lag 定义：

\[
\widetilde K_j(\tau,u)
=
\operatorname{Cont}_{a_j,b_j,\rho_{j,-},\rho_{j,+}}
\left[
K_j(\tau,\cdot)
\right](u).
\tag{22.2}
\]

continuation 不参与结构谱的参考域积分。结构 Hilbert–Schmidt 几何仍定义在：

\[
\mathcal I_j
=
\mathcal S_j^{\mathrm{train}}
\quad\text{或预注册结构参考域}.
\tag{22.3}
\]

因此域外 continuation 不制造新的结构奇异值证据。

## 22.3 尺度来源

\(\rho_{j,\pm}\) 不是随 validation 越界距离临时设定。候选由训练 knot geometry 决定：

\[
\rho_{j,\pm}
\in
\mathcal R_j
=
\{c\,h_{j,\pm}:c\in\mathcal C_\rho\},
\tag{22.4}
\]

其中：

- \(h_{j,\pm}\)：训练边界相邻 knot 间距；
- \(\mathcal C_\rho\)：预注册有限集合；
- 选择只使用 development validation；
- official test 前冻结。

若所有候选 simulation 性能相同，选择最小 continuation 范围对应的最小复杂度候选。

## 定理 22.1：\(C^1\) 拼接

若 \(f\in C^1([a,b])\)，则由式 (22.1) 定义的

\[
\widetilde f\in C^1(\mathbb R),
\tag{22.5}
\]

且：

\[
\widetilde f(a^-)=\widetilde f(a^+)=f(a),
\qquad
\widetilde f'(a^-)=\widetilde f'(a^+)=f'(a),
\tag{22.6}
\]

右边界同理。

## 定理 22.2：全局有界性与导数上界

有：

\[
\|\widetilde f\|_\infty
\le
\max\left\{
\|f\|_{\infty,[a,b]},
|f(a)|+\rho_-|f'(a)|,
|f(b)|+\rho_+|f'(b)|
\right\},
\tag{22.7}
\]

以及

\[
\|\widetilde f'\|_\infty
\le
\max\left\{
\|f'\|_{\infty,[a,b]},
|f'(a)|,
|f'(b)|
\right\}.
\tag{22.8}
\]

因此该 continuation 同时避免：

- hard stop；
- boundary clipping 的零导数断裂；
- 无限线性外推的无界增长。

---

# 23. free-run 的全局可计算性和有界性

考虑 simulation-closed 模型：

\[
\widehat y_{t+1}
=
b+
\sum_{j=1}^{p}
\sum_{\tau=0}^{L_x-1}
\widetilde K_j(\tau,X_{j,t-\tau})
+
\sum_{\ell=1}^{L_y}
q_\ell^y
\widetilde f^y(\widehat y_{t+1-\ell}).
\tag{23.1}
\]

定义：

\[
M_{x,j,\tau}
=
\sup_{u\in\mathbb R}
|\widetilde K_j(\tau,u)|,
\qquad
M_y
=
\sup_{u\in\mathbb R}
|\widetilde f^y(u)|.
\tag{23.2}
\]

## 定理 23.1：任意有限时域的存在唯一性

若初始历史有限，且所有 continuation 全局定义，则对任意有限输入序列和任意有限 \(T\)，递推 (23.1) 在 \(t_0,\ldots,T\) 上存在唯一实值解。

这来自模型的显式递推结构，不需要求解隐式方程。

## 定理 23.2：统一输出有界性

在定理 22.2 下，对所有 simulation 时刻：

\[
|\widehat y_{t+1}|
\le
|b|
+
\sum_{j,\tau}M_{x,j,\tau}
+
M_y\sum_{\ell=1}^{L_y}|q_\ell^y|
=:B_{\mathrm{sim}}.
\tag{23.3}
\]

因此：

\[
\sup_{t\ge t_0}
|\widehat y_t|
\le
\max\left\{
\max_{0\le r<L_y}|\widehat y_{t_0-r}|,
B_{\mathrm{sim}}
\right\}.
\tag{23.4}
\]

这给出全局 simulation closure。它只保证轨迹有限，不保证轨迹准确或遗忘初值。

## 23.3 与线性 continuation 的区别

裸线性 continuation 只给出 affine growth：

\[
|\widetilde f(u)|
\le c_0+c_1|u|.
\]

此时还需要 AR 递推增益的额外收缩条件才能排除爆炸。v4.1 因此把有界 \(C^1\) continuation 作为主定义，线性 continuation 只作消融。

---

# 24. 递推稳定性与初值遗忘证书

定义 history state：

\[
\widehat{\mathbf y}_t
=
(
\widehat y_t,
\widehat y_{t-1},
\ldots,
\widehat y_{t-L_y+1}
)^\top.
\tag{24.1}
\]

令：

\[
L_y^f
=
\sup_{u\in\mathbb R}
|\widetilde f^{y\prime}(u)|.
\tag{24.2}
\]

构造非负伴随矩阵：

\[
A_{\mathrm{AR}}
=
\begin{bmatrix}
|q_1^y|L_y^f & |q_2^y|L_y^f & \cdots & |q_{L_y}^y|L_y^f\\
1&0&\cdots&0\\
0&1&\cdots&0\\
\vdots&&\ddots&\vdots\\
0&\cdots&1&0
\end{bmatrix}.
\tag{24.3}
\]

## 定理 24.1：全局增量稳定的充分条件

若：

\[
\rho(A_{\mathrm{AR}})<1,
\tag{24.4}
\]

则对相同外生输入、不同初始 history 的两条 free-run 轨迹，存在 \(C<\infty\)、\(\varrho\in(0,1)\) 使：

\[
\|\widehat{\mathbf y}_t-\widehat{\mathbf y}_t'\|
\le
C\varrho^{t-t_0}
\|\widehat{\mathbf y}_{t_0}-\widehat{\mathbf y}_{t_0}'\|.
\tag{24.5}
\]

因此模型遗忘初始化误差。

## 定理 24.2：外生输入扰动界

若式 (24.4) 成立，且两条轨迹的外生贡献差为 \(\Delta g_t^x\)，则：

\[
\|\Delta\widehat{\mathbf y}_t\|
\le
C\varrho^{t-t_0}\|\Delta\widehat{\mathbf y}_{t_0}\|
+
C\sum_{s=t_0}^{t-1}
\varrho^{t-1-s}
|\Delta g_s^x|.
\tag{24.6}
\]

这是 simulation 映射的离散 ISS 型界。

## 24.3 证书失败时的正确解释

若：

\[
\rho(A_{\mathrm{AR}})\ge1,
\]

不能据此断言 free-run 必然发散，因为式 (24.3) 是绝对值全局上界，可能保守。

正确状态为：

```text
SIMULATION_GLOBALLY_BOUNDED_BY_CONTINUATION
GLOBAL_INCREMENTAL_STABILITY_NOT_CERTIFIED
```

还必须报告沿验证轨迹的局部 Jacobian：

\[
J_t
=
\begin{bmatrix}
q_1^y\widetilde f^{y\prime}(\widehat y_t)
&
\cdots
&
q_{L_y}^y\widetilde f^{y\prime}(\widehat y_{t-L_y+1})
\\
1&\cdots&0\\
&\ddots&\\
0&1&0
\end{bmatrix}.
\tag{24.7}
\]

局部证书只允许描述 observed simulation trajectory，不升级为全局稳定性定理。

---

# 25. continuation 不改变域内识别对象

## 定理 25.1：训练设计不变性

若训练样本全部位于 \([a_j,b_j]\)，则：

\[
\widetilde K_j(\tau,X_{j,t-\tau})
=
K_j(\tau,X_{j,t-\tau})
\tag{25.1}
\]

对所有训练设计行成立。

因此在相同 history、basis、penalty 和数值求解器下：

- 训练设计矩阵不变；
- 训练 objective 不变；
- 域内 fitted values 不变；
- v4.0 的 Q/K 识别定理不变。

continuation 只补充模型在训练支撑外的数学定义。

## 定理 25.2：结构谱不变性

若结构谱在参考域 \(\mathcal I_j\subseteq\mathcal S_j^{\mathrm{train}}\) 上定义，则：

\[
T_j
=
W_{\tau,j}^{1/2}
K_j|_{\mathcal I_j}
W_{x,j}^{1/2}
\tag{25.2}
\]

与 continuation 参数 \(\rho_{j,\pm}\) 无关。

因此域外 extension：

- 不改变 structural singular values；
- 不增加 structural rank 证据；
- 不允许被计入 HS kernel recovery error。

---

# 26. 域外非识别性与解释防火墙

## 定理 26.1：训练支撑外不可由固定训练分布识别

设两个全局函数 \(f_1,f_2\) 满足：

\[
f_1(u)=f_2(u)
\quad
P_X\text{-a.s.},
\tag{26.1}
\]

但在 \(\mathcal D^{\mathrm{model}}\setminus\operatorname{supp}(P_X)\) 上不同。则它们诱导相同的训练风险和训练观测分布。

所以训练数据不能区分其域外值。

由此，continuation 只能解释为：

\[
\boxed{\text{模型定义选择}}
\]

而不是：

\[
\boxed{\text{域外系统机制估计}}
\]

## 26.2 声明矩阵

### 允许

- 模型完成整个 free-run；
- continuation 使用率和最大距离；
- 域内、域外 simulation loss 分开；
- 结构谱仅在参考域报告；
- 域外结果作为 robustness/simulation 诊断。

### 禁止

- 将域外 continuation 曲线称为 learned K；
- 用域外样本表现反推结构 rank；
- 把数学可计算等同于数据支持；
- 把 continuation 成功写成在线辨识或主动探索。

---

# 27. H2 原生模型的完整选择顺序

H2 是主方法，H3 是 shared-history 公平消融。

## 27.1 H2-A：history

在预注册候选：

\[
(L_x,L_y)\in\mathcal L_x\times\mathcal L_y
\]

上，使用固定 anchor representation 和每个候选自己的 penalty 选择，按 grouped/blocked validation risk 与 one-SE rule 冻结 history。

复杂度键：

\[
C_H
=
(
L_x+L_y,
L_xL_y,
\max(L_x,L_y),
L_x,
L_y
).
\tag{27.1}
\]

## 27.2 H2-B：resolution 与 penalty

冻结 history 后选择：

\[
(M_\tau,M_x,\lambda_\tau,\lambda_x,\lambda_0).
\]

顺序是：

1. representation gate；
2. validation contribution risk；
3. Lepski stability；
4. one-SE 最小复杂度；
5. exact-zero endpoint 和 KKT certification。

## 27.3 H2-C：simulation closure

在冻结的 history/resolution/penalty 上，从预注册 continuation family 中选择：

\[
c_\rho\in\mathcal C_\rho.
\]

选择指标按优先级：

1. 全序列无 NaN/Inf、无 hard stop；
2. continuation 使用率与最大距离；
3. validation free-run loss；
4. 初值扰动敏感度；
5. 最小 continuation 复杂度。

不得用 official test 选择 continuation。

## 27.4 H2-D：rank

先估计 full kernel，再报告：

- rank-1；
- rank-2；
- adaptive predictive rank；
- output-standardized loss；
- relative-to-full loss inflation；
- structural rank 仅在 K 层条件成立时报告。

最终主链为：

\[
\boxed{
\text{history}
\rightarrow
\text{resolution}
\rightarrow
\text{penalty}
\rightarrow
\text{simulation closure}
\rightarrow
\text{rank}
}
\tag{27.2}
\]

---

# 28. v4.1 统一主定理

## 定理 28.1：Simulation-Closed OPS-UOI

在 v4.0 的 A1–A7、有限嵌套 sieve、正确 forward gap 和相应统计条件下，再假设：

1. 每个训练域内核曲线与 AR 响应属于 \(C^1\)；
2. 使用式 (22.1) 的有界 \(C^1\) continuation；
3. continuation 参数只由 development 数据冻结；
4. 结构谱只在预注册参考域计算；

则：

### 识别层

v4.0 的贡献目标 \(g_0\)、quotient class \([K_0]\)、K 层 sieve 识别条件、正交误差界和 rank 条件保持不变。

### simulation 层

固定 fitted model 在整个实数幅值域上可计算；对任意有限输入序列，free-run 唯一存在，且输出满足统一界 (23.3)。

### 稳定层

若 \(\rho(A_{\mathrm{AR}})<1\)，则 free-run 对初值增量稳定，并满足外生输入扰动界 (24.6)。

### 解释层

模型在 \(\mathcal D^{\mathrm{model}}\setminus\mathcal S^{\mathrm{train}}\) 上的值仅由 continuation 定义，不具备 Q/K 识别含义；K 层解释只允许在 \(\mathcal S^{\mathrm{cert}}\) 及对应 sieve coercivity 条件下成立。

这使 PS-AR-RAPHU 成为：

\[
\boxed{
\text{可独立完成 prediction 与 free-run simulation 的系统辨识方法}
}
\]

而不是有限 spline 域内的局部预测器。

---

# 29. v4.1 论文声明

## 29.1 可以声称

1. 模型与 ARX、pNARX、MLP-NARX 等处于同一系统辨识层级；
2. bounded \(C^1\) continuation 提供全局数学定义和 free-run 有界性；
3. continuation 不改变训练支撑内的识别对象和结构谱；
4. Q/K 解释被严格限制在数据和 Gram 支持的区域；
5. H2 形成独立模型选择链；
6. direct prediction 与 free-run simulation 被明确区分；
7. 全局增量稳定性可通过伴随矩阵谱半径给出充分证书。

## 29.2 不能声称

1. continuation 恢复了域外真实物理核；
2. 全局有界等价于准确 simulation；
3. 局部 Jacobian 稳定等价于全局稳定；
4. fixed-stage 结果自动覆盖全部工艺阶段；
5. closed-loop predictive operator 自动等于 causal plant；
6. online identification 已经由 v4.1 完成。

---

# 30. v4.1 后的理论完成度与剩余 12%

以当前论文范围：

\[
\text{固定阶段系统辨识}
+
\text{Q/K 认证}
+
\text{完整 simulation}
+
\text{PB1/CZ 验证}
\]

为 100%，v4.1 的理论设计完成度冻结为：

\[
\boxed{88\%}.
\]

已经闭合：

- operator-first estimand；
- Q/K 分层；
- orthogonal score；
- finite-sieve coercivity；
- adaptive resolution/rank；
- 三域定义；
- simulation closure；
- 全局有界性；
- 增量稳定充分条件；
- H2 独立模型选择；
- 解释防火墙。

剩余 12% 是：

1. dependent block/bootstrap 同时半径的最终有限样本定理；
2. Lepski crossing regularity 的更完整验证；
3. multi-variable Schur 条件的公开 benchmark 实证；
4. 四数据集 H2/free-run 全部完成；
5. official test confirmation；
6. 多晶棒 outer validation。

这些工作不再要求增加新架构。
