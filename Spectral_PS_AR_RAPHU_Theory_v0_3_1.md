# Spectral Predictive-State AR-RAPHU v0.3.1
## 离散时滞完备空间、经认证的压缩表示、双残差正交化、全变量 Urysohn 谱分解与稳定递推

> **状态**：`THEORY_V0.3.1 / REPRESENTATION-REPAIRED / PROPOSED_AND_TESTABLE`  
> **继承**：v0.3 的 Predictive-State、双残差化、全变量凸 Urysohn、Gram 白化谱分解和稳定递推。  
> **修订原因**：v0.3 的 E1 证明 \(M_\tau=12,M_x=16\) 的均匀三次 B 样条空间无法表示冻结生成器中的窄峰、双峰和幅值依赖时滞核。  
> **核心修订**：
>
> \[
> \boxed{
> \text{理论母空间使用完整离散时滞基；压缩时滞基只是经独立容量认证后的工程子空间。}
> }
> \]
>
> \[
> \boxed{
> \text{Scheme A 是估计核的第一谱模态，Scheme B 是同一核的谱尾；二者之间不存在门控。}
> }
> \]

---

# 0. v0.3.1 修订的证据边界

冻结的 E0 已通过：

\[
\max_t
\left|
y_t^{\mathrm{latent}}
-
g_t^{AR}
-
g_t^X
-
\xi_t
\right|
=
1.78\times10^{-15}.
\]

因此，生成器重放和目标语义正确。

冻结的 E1 结果为：

| 场景 | \(12\times16\) 最坏投影 NRMSE |
|---|---:|
| AR-S1 | 0.1471 |
| AR-S2 | 0.4475 |
| AR-S3 | 0.4649 |
| AR-S4 | 0.4372 |

它只证明：

\[
\boxed{
V_{\tau,12}\otimes V_{x,16}
\text{ 对当前真核族欠容量。}
}
\]

它不证明：

- 完整离散 Urysohn 模型失败；
- 谱分解失败；
- 双残差化失败；
- 求解器失败；
- rank-2 不可学习。

因此 v0.3.1 不推翻 v0.3 的主理论，只修复“理论母空间”和“工程压缩空间”被混为一谈的问题。

---

# 1. 受迫系统、历史和任务

设

\[
z_{t+1}=F(z_t,X_t,\xi_t),
\qquad
y_t=H(z_t)+\eta_t,
\]

其中 \(X_t=(x_{1,t},\ldots,x_{p,t})^\top\)。

对预测原点 \(t\)，定义

\[
\mathcal X_t^{(L_x)}
=
\{X_t,X_{t-1},\ldots,X_{t-L_x+1}\},
\]

\[
\mathcal Y_t^{(L_y)}
=
\{y_t,y_{t-1},\ldots,y_{t-L_y+1}\}.
\]

直接视野 \(h\) 的预测只允许：

\[
(\mathcal X_t^{(L_x)},\mathcal Y_t^{(L_y)})
\mapsto y_{t+h}.
\]

不读取未来过程变量。

必须区分：

\[
\widehat y_{t+h\mid t}
\approx
\mathbb E[
y_{t+h}
\mid
\mathcal X_t^{(L_x)},\mathcal Y_t^{(L_y)}
]
\]

和条件外生增量：

\[
\mathbb E[
y_{t+h}
\mid
\mathcal X_t^{(L_x)},\mathcal Y_t^{(L_y)}
]
-
\mathbb E[
y_{t+h}
\mid
\mathcal Y_t^{(L_y)}
].
\]

预测充分性不自动等价于生成机制可辨识性。

---

# 2. Predictive State 的定位

定义预测等价：

\[
z\sim_{\mathrm{pred}}z'
\]

当且仅当，对所有 admissible future input sequences，两状态产生相同未来输出分布。

预测商空间：

\[
\mathcal S_{\mathrm{pred}}
=
\mathcal Z/\sim_{\mathrm{pred}}.
\]

本体系从有限历史构造

\[
s_t
=
\Phi(
\mathcal X_t^{(L_x)},
\mathcal Y_t^{(L_y)}
)
\]

以近似参数化预测商状态。它不声称：

- 恢复完整物理状态；
- 满足全局 Takens 嵌入；
- KAN、Koopman lifting 或谱分解能够创造历史中不存在的信息。

---

# 3. 条件加性 Urysohn 模型

令

\[
Z_t=\mathcal Y_t^{(L_y)},
\qquad
W_{j,t}
=
(x_{j,t},x_{j,t-1},\ldots,x_{j,t-L_x+1}).
\]

对每个视野 \(h\)，假设

\[
\boxed{
\mathbb E[
y_{t+h}\mid Z_t,W_{1,t},\ldots,W_{p,t}
]
=
m_h(Z_t)
+
\sum_{j=1}^{p}
\mathcal U_{j,h}(W_{j,t}).
}
\tag{3.1}
\]

其中

\[
\mathcal U_{j,h}(W_{j,t})
=
\sum_{\tau=0}^{L_x-1}
K_{j,h}(\tau,x_{j,t-\tau}).
\tag{3.2}
\]

该模型允许：

- rank-1 Hammerstein 核；
- rank-2 和更高 rank Urysohn 核；
- 窄峰、双峰和一般平滑时滞；
- 幅值依赖动态；
- 变量特异的时滞—幅值面。

当前版本暂不包含变量间显式交互核。

---

# 4. 离散时滞母空间

## 4.1 时滞不是连续物理坐标的必要假设

当前数据中的时滞索引为有限离散集合：

\[
\mathcal T
=
\{0,1,\ldots,L_x-1\}.
\]

因此理论上不需要假设真实时滞核必须属于一个低维连续 B 样条空间。

定义标准基：

\[
e_\tau
=
(0,\ldots,0,1,0,\ldots,0)^\top
\in\mathbb R^{L_x}.
\]

完整离散时滞基为

\[
C_{\mathrm{id}}(\tau)=e_\tau.
\tag{4.1}
\]

对幅值方向选择中心化三次 B 样条基

\[
B_j(u)
=
(b_{j,1}(u),\ldots,b_{j,M_x}(u))^\top.
\]

理论母模型写为

\[
K_{j,h}(\tau,u)
=
e_\tau^\top
\Theta_{j,h}^{\mathrm{id}}
B_j(u),
\tag{4.2}
\]

其中

\[
\Theta_{j,h}^{\mathrm{id}}
\in
\mathbb R^{L_x\times M_x}.
\]

当幅值基固定时，式 (4.2) 对离散时滞方向没有压缩逼近误差。

## 4.2 工程压缩时滞子空间

为降低计算量，可以选择

\[
C_M(\tau)
=
(c_1(\tau),\ldots,c_M(\tau))^\top,
\qquad
M<L_x,
\]

并表示

\[
K_{j,h}^{(M)}(\tau,u)
=
C_M(\tau)^\top
\Theta_{j,h}^{(M)}
B_j(u).
\tag{4.3}
\]

压缩空间

\[
V_{\tau,M}
=
\operatorname{span}\{c_1,\ldots,c_M\}
\subset\mathbb R^{L_x}
\]

只是母空间的子空间，不是理论本体。

## 4.3 投影误差

给定核函数 Hilbert 空间 \(\mathcal H_j\)，定义

\[
\epsilon_{M,M_x}(K_j)
=
\inf_{\Theta}
\frac{
\|
K_j-C_M^\top\Theta B_j
\|_{\mathcal H_j}
}{
\|K_j\|_{\mathcal H_j}
}.
\tag{4.4}
\]

完整离散时滞参考误差为

\[
\epsilon_{\mathrm{id},M_x}(K_j)
=
\inf_{\Theta}
\frac{
\|
K_j-e_\tau^\top\Theta B_j
\|_{\mathcal H_j}
}{
\|K_j\|_{\mathcal H_j}
}.
\tag{4.5}
\]

它主要反映幅值基和中心化带来的误差。

定义压缩比：

\[
\chi_M(K_j)
=
\frac{
\epsilon_{M,M_x}(K_j)
}{
\epsilon_{\mathrm{id},M_x}(K_j)+\varepsilon_0
}.
\tag{4.6}
\]

压缩空间只有在独立于结构结论的容量实验中通过后，才允许进入 support/rank 验证。

---

# 5. v0.3.1 的表示认证规则

表示认证属于实验协议，不是普适定理。

对当前冻结合成核族，设：

\[
\epsilon_{\max}(M)
=
\max_{\substack{
s\in\{\mathrm{S1,S2,S3,S4}\}\\
\text{seed},j\in S^\star
}}
\epsilon_{M,16}(K_{j,s}).
\]

定义：

## 5.1 预测级认证

\[
\epsilon_{\max}(M)\le0.10.
\tag{5.1}
\]

## 5.2 结构谱级认证

必须同时满足：

\[
\epsilon_{\max}(M)\le0.05,
\tag{5.2}
\]

且对每个场景：

\[
\epsilon_{\max,s}(M)
\le
2\,
\epsilon_{\max,s}(\mathrm{id}).
\tag{5.3}
\]

式 (5.3) 防止压缩时滞误差远大于幅值参考误差，并进入伪谱尾。

在冻结源代码和生成器下，独立复算得到：

| \(M_\tau\) | S1 | S2 | S3 | S4 |
|---:|---:|---:|---:|---:|
| 24 | 0.01977 | 0.08248 | 0.08140 | 0.08734 |
| 28 | 0.01918 | 0.03860 | 0.03986 | 0.04378 |
| 32 | 0.01914 | 0.02390 | 0.02291 | 0.02459 |
| 40 | 0.01913 | 0.01951 | 0.01654 | 0.01874 |
| identity 64 | 0.01912 | 0.01912 | 0.01612 | 0.01827 |

因此按预注册规则：

\[
\boxed{
M_\tau^{\mathrm{struct}}=32,
\qquad
M_x^{\mathrm{struct}}=16.
}
\tag{5.4}
\]

\(24\times16\) 仅达到预测级；\(28\times16\) 通过绝对 0.05 门槛，但在 S2–S4 中压缩误差超过 identity 参考的两倍，因此不作为主结构基。

后续主结构估计固定 \(32\times16\)，下邻居为 \(28\times16\)，上邻居为 \(40\times16\)。identity \(64\times16\) 只用于容量参考和少量最终审计。

该选择只对当前 \(L_x=64\) 和当前核族有效，不宣称普适。

---

# 6. 幅值基、中心化与设计矩阵

幅值适用域固定为训练段：

\[
\mathcal I_j
=
[Q_{0.01}(x_j),Q_{0.99}(x_j)].
\]

定义经验中心化基：

\[
\widetilde B_j(u)
=
B_j(u)
-
\frac1{n_{\mathrm{train}}}
\sum_{t\in\mathrm{train}}
B_j(x_{j,t}).
\tag{6.1}
\]

后续默认 \(B_j\) 已中心化。

单样本设计矩阵：

\[
\Phi_{j,t}
=
\sum_{\tau=0}^{L_x-1}
C_M(\tau)
B_j(x_{j,t-\tau})^\top.
\tag{6.2}
\]

于是

\[
\mathcal U_{j,h}(W_{j,t})
=
\langle\Theta_{j,h},\Phi_{j,t}\rangle_F.
\tag{6.3}
\]

向量化全部变量：

\[
y_{t+h}
=
m_h(Z_t)
+
\phi_t^\top\theta_h
+
\varepsilon_{t,h}.
\tag{6.4}
\]

---

# 7. 正式结构估计必须双残差化

只残差化输出：

\[
y-\widehat m(Z)
\]

并不能消除外生设计中可由 AR 历史预测的部分。

定义 nuisance：

\[
\mu_h(Z)
=
\mathbb E[y_{t+h}\mid Z_t=Z],
\tag{7.1}
\]

\[
\pi_h(Z)
=
\mathbb E[\phi_t\mid Z_t=Z].
\tag{7.2}
\]

定义：

\[
\widetilde y_t
=
y_{t+h}-\mu_h(Z_t),
\tag{7.3}
\]

\[
\widetilde\phi_t
=
\phi_t-\pi_h(Z_t).
\tag{7.4}
\]

则条件加性模型化为：

\[
\widetilde y_t
=
\widetilde\phi_t^\top\theta_h
+
\varepsilon_{t,h}.
\tag{7.5}
\]

矩条件：

\[
\psi(
y,\phi,Z;\theta,\mu,\pi
)
=
(\phi-\pi(Z))
[
y-\mu(Z)-(\phi-\pi(Z))^\top\theta
].
\tag{7.6}
\]

在真实参数处：

\[
\mathbb E[\psi]=0.
\]

由于

\[
\mathbb E[\phi-\pi(Z)\mid Z]=0,
\]

该矩条件对 \(\mu,\pi\) 的一阶扰动为零，即具有 Neyman 正交性。

nuisance 必须使用前向连续 cross-fitting 生成样本外预测，不能在同一记录上训练后再产生自身残差。

---

# 8. 全变量平滑 Urysohn 强凸估计

给定交叉拟合残差：

\[
\widehat\theta_h
=
\arg\min_\theta
\frac1{2n}
\|
\widehat{\widetilde y}
-
\widehat{\widetilde\Phi}\theta
\|_2^2
+
\sum_j\mathcal R_j(\Theta_{j,h}).
\tag{8.1}
\]

正则：

\[
\begin{aligned}
\mathcal R_j(\Theta_j)
=&
\lambda_\tau
\|D_\tau\Theta_j\|_F^2
+
\lambda_x
\|\Theta_jD_x^\top\|_F^2
\\
&
+
\lambda_0
\|
G_\tau^{1/2}
\Theta_j
G_{x,j}^{1/2}
\|_F^2.
\end{aligned}
\tag{8.2}
\]

其中 \(\lambda_0>0\)。

若 Gram 矩阵在保留子空间正定，则总目标强凸，具有唯一全局最优解。

本体系不使用：

- 训练期变量删除；
- 对 KAN 参数块的 group-prox；
- A 对 B 的硬门控；
- sigmoid/Concrete rank gate。

---

# 9. Uniform 与 D5-adaptive 估计器

## 9.1 理论主线：Uniform spectral

所有变量使用相同函数空间 ridge 权重：

\[
\omega_j=1.
\]

这是最少附加假设的主估计器。

## 9.2 待验证增强：D5-adaptive spectral

D5 贡献路径在独立内折产生连续分数 \(s_j\)，构造有限权重：

\[
\omega_j
=
\operatorname{clip}
\left(
\frac{
\operatorname{median}_k(s_k+\epsilon)
}{
s_j+\epsilon
},
\omega_{\min},
\omega_{\max}
\right).
\tag{9.1}
\]

加权项：

\[
\lambda_0
\sum_j
\omega_j
\|
G_\tau^{1/2}
\Theta_j
G_{x,j}^{1/2}
\|_F^2.
\tag{9.2}
\]

它作用于函数核范数，不作用于任意神经参数坐标。所有权重有限，因此变量不会失去被完整核估计救回的资格。

在正式 E6 通过前，adaptive 只能作为对照。

---

# 10. Gram 几何和基不变谱

定义：

\[
\widetilde\Theta_j
=
G_\tau^{1/2}
\Theta_j
G_{x,j}^{1/2}.
\tag{10.1}
\]

核 Hilbert–Schmidt 范数：

\[
\|K_j\|_{\mathcal H_j}
=
\|\widetilde\Theta_j\|_F.
\tag{10.2}
\]

对白化矩阵做 SVD：

\[
\widetilde\Theta_j
=
U_j\Sigma_jV_j^\top.
\tag{10.3}
\]

---

# 11. Scheme A 与 Scheme B

## 11.1 Scheme A

\[
\widetilde\Theta_j^A
=
\sigma_{j,1}
u_{j,1}v_{j,1}^\top.
\tag{11.1}
\]

它是估计核在 Gram 度量下的最佳 rank-1 近似。

## 11.2 Scheme B

\[
\widetilde\Theta_j^B
=
\sum_{r\ge2}
\sigma_{j,r}
u_{j,r}v_{j,r}^\top.
\tag{11.2}
\]

因此：

\[
\boxed{
K_j=K_j^A+K_j^B.
}
\tag{11.3}
\]

A 和 B 是同一算子的第一模态与谱尾，不是两个有资格门的训练模块。

---

# 12. 表示误差为何会影响 rank

设真实白化核为 \(T\)，压缩投影和估计得到

\[
\widehat T
=
T+E_{\mathrm{basis}}+E_{\mathrm{est}}.
\]

由奇异值扰动界：

\[
|
\widehat\sigma_r-\sigma_r
|
\le
\|
E_{\mathrm{basis}}+E_{\mathrm{est}}
\|_{\mathrm{op}}
\le
\|
E_{\mathrm{basis}}+E_{\mathrm{est}}
\|_F.
\tag{12.1}
\]

因此，过大的 basis error 可以：

- 制造伪第二奇异值；
- 压低真实第二奇异值；
- 改变第二模态方向；
- 使 rank-1 与 rank-2 比较失真。

若谱间隔为

\[
\delta
=
\min(
\sigma_1-\sigma_2,
\sigma_2-\sigma_3
),
\]

则模态子空间的稳定性还依赖：

\[
\|
E_{\mathrm{basis}}+E_{\mathrm{est}}
\|_{\mathrm{op}}
\ll\delta.
\]

所以 E1R 不是普通工程调参，而是 rank 结论的前置合法性条件。

---

# 13. 支持证据

模型训练时不删除变量。

对每个变量报告：

## 13.1 核范数

\[
S_j^K
=
\|\widetilde\Theta_j\|_F.
\]

## 13.2 样本外贡献能量

\[
S_j^C
=
\sqrt{
\frac1n
\sum_t
(c_{j,t}-\bar c_j)^2
}.
\]

## 13.3 块消融

\[
\Delta_j^{abl}
=
L_{-j}^{val}
-
L_{\mathrm{full}}^{val}.
\]

## 13.4 D5 路径稳定性

记录变量进入标准化 FISTA 路径的强度和跨折频率。

真实数据默认报告连续证据；只有合成评价使用预注册 null cutoff 转成 support label。

---

# 14. rank 证据与部署 rank

谱尾比例：

\[
\eta_j(R)
=
\frac{
\sum_{r>R}\sigma_{j,r}^2
}{
\sum_{r\ge1}\sigma_{j,r}^2
}.
\tag{14.1}
\]

结构 rank-2 结论必须同时满足：

1. rank-1 null block bootstrap 显著；
2. rank-2 样本外增益稳定为正；
3. 第二模态方向稳定；
4. 主基 \(32\times16\) 与上邻居 \(40\times16\) 结论一致；
5. 下邻居 \(28\times16\) 作为压力测试报告；
6. 超参数选择不使用奇异值和 rank 结果。

部署 rank 则按误差预算选取：

\[
R_j^{deploy}
=
\min
\left\{
R:
\eta_j(R)
+
\epsilon_{j,\mathrm{realization}}
\le
\epsilon_{j,\mathrm{budget}}
\right\}.
\]

结构 rank 和部署 rank 不能混为同一结论。

---

# 15. 预测路径与结构路径

## 15.1 预测路径

保留 dense simultaneous XAR：

\[
\widehat y_{t+h\mid t}
=
A_h(\mathcal Y_t)
+
X_h(\mathcal X_t).
\]

它负责 RMSE、R²、多视野和部署性能。

## 15.2 结构路径

使用：

\[
\text{forward cross-fit}
\to
\text{double residualization}
\to
\text{all-variable full Urysohn}
\to
\text{Gram SVD}.
\]

它负责条件外生核、support evidence 和 rank evidence。

## 15.3 重新合并

从谱模态初始化：

\[
\widehat y_{t+h\mid t}
=
A_h(\mathcal Y_t)
+
\sum_j
\sum_{r=1}^{R_j}
\sigma_{j,r,h}
\sum_\tau
q_{j,r,h}(\tau)
f_{j,r,h}(x_{j,t-\tau}).
\]

refit 中 rank 固定，不使用 group-prox，不重新删除变量。

---

# 16. 稳定递推部署

单个谱模态：

\[
v_{j,r,t}
=
f_{j,r}(x_{j,t}),
\]

\[
c_{j,r,t}
=
\sum_\tau
q_{j,r}(\tau)v_{j,r,t-\tau}.
\]

寻找稳定实现：

\[
s_{j,r,t+1}
=
A_{j,r}s_{j,r,t}
+
B_{j,r}v_{j,r,t},
\]

\[
\widehat c_{j,r,t}
=
C_{j,r}s_{j,r,t},
\qquad
\rho(A_{j,r})<1.
\]

Gamma/Erlang 是优先尝试的低阶实现，不是唯一合法时滞核。

部署时没有输入相关变量门，也没有在线 rank 开关。

---

# 17. 完整误差分解

\[
\begin{aligned}
\|\widehat y-y^\star\|
\le&
\epsilon_{\mathrm{state}}
+
\epsilon_{\mathrm{nuisance}}
+
\epsilon_{\mathrm{amp\ basis}}
\\
&
+
\epsilon_{\mathrm{lag\ compression}}
+
\epsilon_{\mathrm{est}}
+
\epsilon_{\mathrm{rank}}
\\
&
+
\epsilon_{\mathrm{realization}}
+
\epsilon_{\mathrm{quant}}
+
\epsilon_{\mathrm{drift}}.
\end{aligned}
\tag{17.1}
\]

v0.3 的 E1 失败具体定位到：

\[
\epsilon_{\mathrm{lag\ compression}}
\text{ 过大}.
\]

v0.3.1 通过 E1R 单独控制该项。

---

# 18. 假设

## A1 阶段局部稳定  
研究数据属于同一正常阶段，或漂移足够慢。

## A2 条件加性  
式 (3.1) 成立或逼近误差可控。

## A3 记忆可截断  
\(L_x,L_y\) 之外的尾部影响可控。

## A4 幅值平滑  
核在幅值方向可由选定样条空间逼近。

## A5 压缩基已认证  
用于结构推断的时滞子空间通过独立 E1R。

## A6 条件外生创新  
\[
\mathbb E[\varepsilon\mid Z,\phi]=0.
\]

## A7 条件激励  
双残差设计协方差在研究子空间非退化。

## A8 nuisance 可估计  
交叉拟合的 \(\mu,\pi\) 误差足够小。

## A9 时间弱依赖  
block bootstrap 的弱依赖条件近似成立。

## A10 选择与 rank 分离  
basis 和平滑参数在 rank 统计前冻结。

---

# 19. 不可能性边界

若

\[
\phi_{j,t}=g(Z_t)
\]

几乎处处成立，则

\[
\widetilde\phi_{j,t}=0,
\]

条件外生结构不可识别。

不存在一种方法能无条件同时保证：

- 精确支持；
- 精确 rank；
- 无需激励；
- 不受闭环共线影响；
- 不需要统计判定；
- 对所有阶段全局有效。

\(32\times16\) 也不宣称适用于所有未来数据。它只在当前冻结合成核族上获得结构级认证。真实数据若在 \(28,32,40\) 三种表示下结论不稳定，则不得声称 rank。

---

# 20. v0.3.1 统一流程

\[
\boxed{
\begin{aligned}
&\text{E0：生成分量语义}
\\
\to\;&
\text{E1R：完整离散时滞参考与压缩基认证}
\\
\to\;&
\text{E2：正确外生目标上的容量和谱截断}
\\
\to\;&
\text{E3：前向双残差化}
\\
\to\;&
\text{E4：全变量支持证据}
\\
\to\;&
\text{E5：basis-stable rank 统计}
\\
\to\;&
\text{E6：D5-adaptive 对照}
\\
\to\;&
\text{E7：谱结构重新并入 dense XAR}
\\
\to\;&
\text{E8：稳定递推压缩}.
\end{aligned}
}
\]

---

# 21. 允许的结论

完成 E1R 只能声明：

> 当前 \(32\times16\) 压缩表示对冻结合成核族具有足够低的投影误差。

完成 E2 才能声明：

> 在正确目标上，完整核与低秩截断具有足够容量。

完成 E3 才能声明：

> 双残差结构路径相较只残差化输出更接近 oracle 条件外生核。

完成 E4/E5 后，才能声明：

> 方法在预注册合成场景中达到支持和 rank 的恢复门槛。

在真实 CZ 单棒数据上，默认只声明：

- 样本外预测能力；
- 条件外生贡献证据；
- 核形状和谱稳定性；
- 当前阶段内的适用性。

不得直接声明物理因果关系。
