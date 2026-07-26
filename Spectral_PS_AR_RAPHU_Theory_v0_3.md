# Spectral Predictive-State AR-RAPHU v0.3  
## 交叉拟合、双残差正交化、全变量 Urysohn 谱分解与稳定递推的完整理论

> **状态**：`THEORY_V0.3 / PROPOSED_AND_TESTABLE`  
> **替代对象**：v0.2 中“Scheme A 先做硬支持筛选，Scheme B 只审计 A 支持”的结构。  
> **保留对象**：Predictive-State 解释、有限记忆 Urysohn 表示、Gram 白化、低秩谱分解、稳定递推部署和多视野预测。  
> **明确删除**：训练期变量硬门控、对 KAN 原始参数块直接 group-prox、以单个激活权重解释 rank、Scheme A 对 Scheme B 的资格控制。  
> **核心结论**：
>
> \[
> \boxed{
> \text{预测路径使用 dense XAR；结构路径使用双残差正交化后的全变量平滑 Urysohn；}
> }
> \]
>
> \[
> \boxed{
> \text{Scheme A 是完整核的第一谱模态，Scheme B 是同一核的谱尾。}
> }
> \]

---

# 0. 为什么需要 v0.3

当前停止线和 D1–D6 诊断揭示了四件事。

1. 在不使用稀疏近端删除时，dense simultaneous XAR 可以形成显著外生贡献，说明“AR 必然让 X 学不会”并不成立。
2. 对标准化贡献进行凸 FISTA 路径时，真实支持在当前 Gamma rank-1 合成场景中可以被稳定分离，说明信息并未消失。
3. 原 group-prox 直接压缩 KAN 参数块时，外生贡献会在已经形成后继续坍缩，说明正则对象选错了。
4. D1、D2 将“去掉 AR 的 X-only 模型”用于预测仍含强 AR 项的完整输出，目标与模型不匹配，因此不能据此否定 rank-2 容量。

因此，新体系必须同时满足：

- 不让预测目标和结构恢复目标互相绑架；
- 不依赖变量硬门控；
- 不用任意激活函数定义支持或 rank；
- 不把参数坐标范数当成函数贡献强度；
- 每一项实验都先声明“模型包含什么、目标包含什么、理论上可达到什么”。

---

# 1. 真实系统、观测历史与预测任务

## 1.1 受迫非线性系统

设真实过程满足

\[
z_{t+1}
=
F(z_t,X_t,\xi_t),
\qquad
y_t
=
H(z_t)+\eta_t,
\]

其中：

- \(z_t\in\mathcal Z\)：不可完全观测的物理状态；
- \(X_t=(x_{1,t},\ldots,x_{p,t})^\top\)：可观测过程变量；
- \(y_t\in\mathbb R\)：目标输出；
- \(\xi_t\)：过程创新；
- \(\eta_t\)：测量噪声。

本项目当前为九个外生过程变量和一个历史输出通道，故 \(p=9\)；合成实验可保留第十个外生干扰通道用于假阳性审计。

## 1.2 可用历史

对预测原点 \(t\)，定义

\[
\mathcal X_t^{(L_x)}
=
\{X_{t-L_x+1},\ldots,X_t\},
\]

\[
\mathcal Y_t^{(L_y)}
=
\{y_{t-L_y+1},\ldots,y_t\}.
\]

对直接预测视野 \(h\ge1\)，只允许

\[
(\mathcal X_t^{(L_x)},\mathcal Y_t^{(L_y)})
\longmapsto y_{t+h}.
\]

不允许读取 \(X_{t+1:t+h}\)。

## 1.3 两类不同目标

必须区分：

### 预测目标

\[
\widehat y_{t+h\mid t}
\approx
\mathbb E[y_{t+h}\mid
\mathcal X_t^{(L_x)},\mathcal Y_t^{(L_y)}].
\]

它允许历史输出吸收大量过程信息。

### 条件外生结构目标

给定历史输出后，研究外生历史仍保留的增量作用：

\[
\mathbb E[y_{t+h}\mid
\mathcal X_t^{(L_x)},\mathcal Y_t^{(L_y)}]
-
\mathbb E[y_{t+h}\mid
\mathcal Y_t^{(L_y)}].
\]

预测最优并不自动意味着外生结构可辨识。

---

# 2. Predictive State 的正确定位

## 2.1 预测等价关系

在固定阶段和 admissible future inputs 下，若两个潜在状态 \(z,z'\) 对所有允许的未来输入序列产生相同未来输出分布，则定义

\[
z\sim_{\mathrm{pred}}z'.
\]

预测商状态空间为

\[
\mathcal S_{\mathrm{pred}}
=
\mathcal Z/\sim_{\mathrm{pred}}.
\]

本项目的目标是从有限历史中构造

\[
s_t
=
\Phi(
\mathcal X_t^{(L_x)},
\mathcal Y_t^{(L_y)}
)
\]

使其近似足以预测未来，而不是恢复完整物理状态 \(z_t\)。

## 2.2 不违反 Takens

本体系不声称：

- 标量输出窗口构成完整状态空间的全局嵌入；
- \(L_y\) 满足某个简单的 \(2d+1\) 规则就必然恢复真实状态；
- KAN 或任何 lifting 可以创造历史中不存在的信息。

它只声称：在给定阶段、给定视野和给定数据分布中，有限历史可近似参数化预测商状态。

---

# 3. 条件加性 Urysohn 假设

对每个视野 \(h\)，令

\[
Z_t
=
\mathcal Y_t^{(L_y)}
\]

表示 AR 历史，令

\[
W_{j,t}
=
\mathcal X_{j,t}^{(L_x)}
=
(x_{j,t-L_x+1},\ldots,x_{j,t})
\]

表示第 \(j\) 个变量的历史。

核心结构假设为：

\[
\boxed{
\mathbb E[y_{t+h}\mid Z_t,W_{1,t},\ldots,W_{p,t}]
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

式 (3.1) 是当前版本的模型类边界。它允许：

- 任意平滑的时滞—幅值耦合；
- rank-1、rank-2 和更高 rank；
- 不同变量不同动态核；
- AR 历史形成独立预测状态分支。

它暂不允许：

- \(x_i\) 与 \(x_j\) 的显式交互核；
- 未观测阶段切换；
- 无穷长且不可稳定逼近的记忆；
- 完全不可辨识的闭环共线结构。

---

# 4. 张量积样条表示

## 4.1 时滞基和幅值基

对时滞方向选取基函数

\[
C(\tau)
=
(c_1(\tau),\ldots,c_{M_\tau}(\tau))^\top.
\]

对第 \(j\) 个变量的幅值方向选取三次 B 样条基

\[
B_j(u)
=
(b_{j,1}(u),\ldots,b_{j,M_x}(u))^\top.
\]

定义

\[
K_{j,h}(\tau,u)
=
C(\tau)^\top
\Theta_{j,h}
B_j(u),
\tag{4.1}
\]

其中

\[
\Theta_{j,h}\in
\mathbb R^{M_\tau\times M_x}.
\]

## 4.2 线性设计

定义单个样本的核设计矩阵

\[
\Phi_{j,t}
=
\sum_{\tau=0}^{L_x-1}
C(\tau)
B_j(x_{j,t-\tau})^\top.
\tag{4.2}
\]

则

\[
\mathcal U_{j,h}(W_{j,t})
=
\langle \Theta_{j,h},\Phi_{j,t}\rangle_F.
\tag{4.3}
\]

向量化后：

\[
\phi_{j,t}
=
\operatorname{vec}(\Phi_{j,t}),
\qquad
\theta_{j,h}
=
\operatorname{vec}(\Theta_{j,h}),
\]

有

\[
\mathcal U_{j,h}
=
\phi_{j,t}^\top\theta_{j,h}.
\]

把全部变量拼接：

\[
\phi_t
=
(\phi_{1,t}^\top,\ldots,\phi_{p,t}^\top)^\top,
\qquad
\theta_h
=
(\theta_{1,h}^\top,\ldots,\theta_{p,h}^\top)^\top.
\]

于是模型变为部分线性形式

\[
y_{t+h}
=
m_h(Z_t)
+
\phi_t^\top\theta_h
+
\varepsilon_{t,h}.
\tag{4.4}
\]

注意：对系数 \(\theta_h\) 是线性的，但对原始输入仍然是非线性动态映射。

---

# 5. 中心化与可辨识规范

## 5.1 截距混淆

若某个 \(K_j(\tau,u)\) 含有常数幅值部分，则它可以与全局偏置互相转移。为消除该自由度，在训练经验分布 \(\nu_j\) 上定义中心化幅值基：

\[
\widetilde B_j(u)
=
B_j(u)
-
\int B_j(v)\,d\nu_j(v).
\tag{5.1}
\]

后续默认 \(B_j\) 已中心化。

## 5.2 适用域

幅值域固定为训练段的分位区间

\[
\mathcal I_j
=
[Q_{0.01}(x_j),Q_{0.99}(x_j)].
\]

超出该区间属于外推，不用于结构结论。

## 5.3 条件激励

定义在给定 AR 历史后仍剩余的外生特征协方差

\[
Q_h
=
\mathbb E[
\widetilde\phi_t\widetilde\phi_t^\top
],
\]

其中 \(\widetilde\phi_t\) 将在第 6 节定义。

只有当 \(Q_h\) 在研究子空间上非退化时，条件外生结构才可辨识。若某些变量在闭环中完全由 \(Y\) 决定，则相应方向无法从观测数据中区分。

---

# 6. 为什么必须“双残差化”

## 6.1 只残差化 \(y\) 不够

若直接先拟合 AR：

\[
\widehat r_t
=
y_{t+h}
-
\widehat m_h(Z_t),
\]

再回归

\[
\widehat r_t
\sim \phi_t,
\]

当 \(\phi_t\) 与 \(Z_t\) 相关时，\(\phi_t\) 仍包含可由 AR 历史预测的部分。此时得到的系数一般不是条件外生效应。

这正是闭环和强 AR 场景下必须防止的设计错误。

## 6.2 两个 nuisance 函数

定义

\[
\mu_h(Z)
=
\mathbb E[y_{t+h}\mid Z_t=Z],
\tag{6.1}
\]

\[
\pi_h(Z)
=
\mathbb E[\phi_t\mid Z_t=Z].
\tag{6.2}
\]

定义双残差：

\[
\widetilde y_t
=
y_{t+h}
-
\mu_h(Z_t),
\tag{6.3}
\]

\[
\widetilde\phi_t
=
\phi_t
-
\pi_h(Z_t).
\tag{6.4}
\]

由式 (4.4) 可得

\[
\widetilde y_t
=
\widetilde\phi_t^\top\theta_h
+
\varepsilon_{t,h}.
\tag{6.5}
\]

## 6.3 正交矩条件

定义

\[
\psi_h(
y,\phi,Z;\theta,\mu,\pi
)
=
(\phi-\pi(Z))
[
y-\mu(Z)
-
(\phi-\pi(Z))^\top\theta
].
\tag{6.6}
\]

在真实参数处：

\[
\mathbb E[\psi_h]=0.
\tag{6.7}
\]

而且该矩条件对 nuisance 的一阶扰动为零，即 Neyman 正交：

\[
D_\mu
\mathbb E[\psi_h]
=
0,
\qquad
D_\pi
\mathbb E[\psi_h]
=
0.
\tag{6.8}
\]

### 证明

对 \(\mu\) 方向扰动 \(a(Z)\)：

\[
D_\mu
\mathbb E[\psi_h][a]
=
-
\mathbb E[
(\phi-\pi(Z))a(Z)
].
\]

因为

\[
\mathbb E[\phi-\pi(Z)\mid Z]=0,
\]

故该式为零。

对 \(\pi\) 方向扰动 \(b(Z)\)，在真实模型下展开一阶项：

\[
-\mathbb E[
b(Z)\varepsilon
]
+
\mathbb E[
(\phi-\pi(Z))b(Z)^\top\theta
].
\]

由条件均值零和
\(\mathbb E[\phi-\pi(Z)\mid Z]=0\)，两项均为零。

证毕。

## 6.4 交叉拟合

不能用同一批样本训练 nuisance 并生成自身残差。将训练时间轴划分为前向连续折 \(\mathcal I_k\)。

对每个折：

1. 只使用该折之前的数据训练 \(\widehat\mu_h^{(-k)}\) 和 \(\widehat\pi_h^{(-k)}\)；
2. 在 \(\mathcal I_k\) 上生成样本外残差；
3. 拼接全部折得到
   \(\widehat{\widetilde y}\) 和
   \(\widehat{\widetilde\Phi}\)。

这一步同时避免：

- nuisance 过拟合制造过小残差；
- 同一噪声被两次利用；
- 未来时间数据泄漏到过去折。

---

# 7. 全变量平滑 Urysohn 估计

## 7.1 主估计器

给定交叉拟合残差，定义

\[
\widehat\theta_h
=
\arg\min_{\theta}
\frac{1}{2n}
\|
\widehat{\widetilde y}
-
\widehat{\widetilde\Phi}\theta
\|_2^2
+
\mathcal R(\theta).
\tag{7.1}
\]

对每个变量：

\[
\mathcal R_j(\Theta_j)
=
\lambda_\tau
\|D_\tau\Theta_j\|_F^2
+
\lambda_x
\|\Theta_jD_x^\top\|_F^2
+
\lambda_0
\|
G_\tau^{1/2}
\Theta_j
G_{x,j}^{1/2}
\|_F^2.
\tag{7.2}
\]

总正则：

\[
\mathcal R(\theta)
=
\sum_j\mathcal R_j(\Theta_j).
\]

其中：

- \(D_\tau\)：时滞方向二阶差分；
- \(D_x\)：幅值方向二阶差分；
- \(G_\tau\)、\(G_{x,j}\)：基函数 Gram 矩阵；
- \(\lambda_0>0\)：稳定 ridge，不承担变量删除。

## 7.2 凸性与唯一性

### 定理 7.1

若：

1. 基函数固定；
2. \(\lambda_\tau,\lambda_x\ge0\)；
3. \(\lambda_0>0\)；
4. Gram 矩阵在保留基子空间上正定；

则式 (7.1) 是强凸二次问题，存在唯一全局最优解。

### 证明

数据项 Hessian 为

\[
\frac1n
\widehat{\widetilde\Phi}^\top
\widehat{\widetilde\Phi}
\succeq0.
\]

平滑惩罚半正定。ridge 项在保留子空间上至少提供

\[
\lambda_0\lambda_{\min}(G)>0
\]

的曲率，故总 Hessian 正定。强凸函数具有唯一全局极小值。

证毕。

## 7.3 闭式正规方程

令 \(R\) 为总惩罚矩阵，则

\[
\widehat\theta_h
=
\left(
\frac1n\widetilde\Phi^\top\widetilde\Phi
+
R
\right)^{-1}
\frac1n\widetilde\Phi^\top\widetilde y.
\tag{7.3}
\]

当前约九个变量、\(M_\tau=8\)、\(M_x=12\) 时，总参数约

\[
9\times8\times12=864,
\]

可直接使用 FP64 Cholesky 求解，无需神经网络长时间迭代。

## 7.4 扰动稳定性

若总 Hessian 的最小特征值为 \(\kappa>0\)，目标右端扰动为 \(\delta b\)，矩阵扰动为 \(\delta A\)，且 \(\|\delta A\|<\kappa\)，则

\[
\|\widehat\theta-\theta^\star\|
\le
\frac{
\|\delta b\|
+
\|\delta A\|
\|\theta^\star\|
}{
\kappa-\|\delta A\|
}.
\tag{7.4}
\]

因此稳定 ridge 不只是数值补丁，也给出了 nuisance 和有限样本误差传播的控制量。

---

# 8. Gram 几何与基不变谱

## 8.1 核函数范数

在离散时滞测度 \(\mu_\tau\) 和幅值测度 \(\nu_j\) 下：

\[
\|K_j\|_{\mathcal H_j}^2
=
\int\sum_\tau
K_j(\tau,u)^2
\,d\nu_j(u).
\]

用非正交基表示时：

\[
\|K_j\|_{\mathcal H_j}^2
=
\operatorname{tr}
(
\Theta_j^\top
G_\tau
\Theta_j
G_{x,j}
).
\tag{8.1}
\]

定义白化矩阵

\[
\widetilde\Theta_j
=
G_\tau^{1/2}
\Theta_j
G_{x,j}^{1/2}.
\tag{8.2}
\]

则

\[
\|K_j\|_{\mathcal H_j}
=
\|\widetilde\Theta_j\|_F.
\tag{8.3}
\]

因此谱分析必须对白化矩阵进行，而不能直接对任意样条系数矩阵做 SVD。

---

# 9. Scheme A 与 Scheme B 的统一定义

对

\[
\widetilde\Theta_j
=
U_j
\Sigma_j
V_j^\top
\]

进行 SVD：

\[
\Sigma_j
=
\operatorname{diag}
(\sigma_{j,1},\sigma_{j,2},\ldots).
\]

## 9.1 Scheme A

定义为最佳 rank-1 投影：

\[
\widetilde\Theta_j^{A}
=
\sigma_{j,1}
u_{j,1}v_{j,1}^\top.
\tag{9.1}
\]

反白化得到

\[
\Theta_j^A
=
G_\tau^{-1/2}
\widetilde\Theta_j^A
G_{x,j}^{-1/2}.
\tag{9.2}
\]

这就是数据度量下的主导 Hammerstein 模态。

## 9.2 Scheme B

定义为谱尾：

\[
\widetilde\Theta_j^{B}
=
\sum_{r\ge2}
\sigma_{j,r}
u_{j,r}v_{j,r}^\top.
\tag{9.3}
\]

因此

\[
\boxed{
K_j=K_j^A+K_j^B.
}
\tag{9.4}
\]

A 和 B 不再是两个互相门控的训练器，而是同一估计算子的主模态与残余谱。

## 9.3 最佳低秩逼近

### 定理 9.1

对任意 \(R\)，令

\[
\widetilde\Theta_{j,R}
=
\sum_{r=1}^{R}
\sigma_{j,r}
u_{j,r}v_{j,r}^\top.
\]

则在 Hilbert–Schmidt/Gram 范数下，它是所有 rank 不超过 \(R\) 的核中距离 \(\widehat K_j\) 最近者，并且

\[
\|
\widetilde\Theta_j
-
\widetilde\Theta_{j,R}
\|_F^2
=
\sum_{r>R}\sigma_{j,r}^2.
\tag{9.5}
\]

这是 Eckart–Young–Mirsky 结论在白化坐标中的直接应用。

---

# 10. 支持不再是模型内部开关

## 10.1 连续核证据

定义

\[
S_j^{K}
=
\|\widetilde\Theta_j\|_F.
\tag{10.1}
\]

它衡量函数空间中的核强度。

## 10.2 实际贡献证据

在评价时间段定义

\[
c_{j,t}
=
\langle
\widehat\Theta_j,
\Phi_{j,t}
\rangle_F,
\]

\[
S_j^{C}
=
\sqrt{
\frac1n
\sum_t
(c_{j,t}-\overline c_j)^2
}.
\tag{10.2}
\]

## 10.3 块消融证据

令完整模型验证损失为 \(L_{\mathrm{full}}\)，去掉第 \(j\) 个变量并重新求解剩余凸问题后的损失为 \(L_{-j}\)。定义

\[
\Delta_j^{\mathrm{abl}}
=
L_{-j}-L_{\mathrm{full}}.
\tag{10.3}
\]

若 \(\Delta_j^{\mathrm{abl}}>0\)，说明该变量具有样本外增量预测价值。

## 10.4 D5 路径证据

dense XAR 的变量贡献标准化后：

\[
Z_{tj}
=
\frac{c_{j,t}-\bar c_j}{s_j}.
\]

离线求解

\[
\min_{b,g}
\frac1{2n}
\|r-b-Zg\|_2^2
+
\lambda\|g\|_1.
\tag{10.4}
\]

这一步允许非光滑，因为它不进入实时计算图，也不决定 B 是否运行。它只产生：

- 路径进入强度；
- 跨折稳定性；
- 辅助自适应权重。

## 10.5 真实数据不强制二值化

在真实 CZ 数据上，默认报告

\[
(S_j^K,S_j^C,\Delta_j^{\mathrm{abl}},\text{路径稳定性})
\]

而不是声称“变量绝对存在/不存在”。

合成实验为了计算 recall/FPR，可以使用预注册的统计阈值，但该阈值只属于评价层。

---

# 11. rank 证据与部署 rank 必须分开

## 11.1 谱尾能量

定义

\[
\eta_j(R)
=
\frac{
\sum_{r>R}\sigma_{j,r}^2
}{
\sum_{r\ge1}\sigma_{j,r}^2
}.
\tag{11.1}
\]

特别地：

\[
\eta_j(1)
=
\frac{
\sum_{r\ge2}\sigma_{j,r}^2
}{
\sum_{r\ge1}\sigma_{j,r}^2
}.
\]

## 11.2 结构 rank 证据

声明 rank-2 不能只看 \(\sigma_2>0\)，因为有限样本噪声下几乎总有非零第二奇异值。

rank-2 证据必须同时来自：

1. rank-1 null block bootstrap；
2. 外层验证中 rank-2 相对 rank-1 的预测增益；
3. 第二模态子空间跨重采样稳定性；
4. 邻近 grid/smoothing 下结论不反转。

## 11.3 部署 rank

部署 rank 由误差预算决定：

\[
R_j^{\mathrm{deploy}}
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
\tag{11.2}
\]

结构上存在弱第二模态，不代表部署时必须保留；部署保留第二模态，也不等于它具有已确认物理含义。

---

# 12. D5 自适应 Tikhonov 的正确位置

D5 可以生成连续证据 \(s_j>0\)，再定义有限权重

\[
\omega_j
=
\operatorname{clip}
\left(
\frac{\operatorname{median}_k(s_k+\epsilon)}
{s_j+\epsilon},
\omega_{\min},
\omega_{\max}
\right).
\tag{12.1}
\]

加权估计为

\[
\widehat\theta^{\mathrm{adaptive}}
=
\arg\min
\frac1{2n}
\|
\widetilde y-\widetilde\Phi\theta
\|^2
+
\mathcal R_{\mathrm{smooth}}
+
\lambda_0
\sum_j
\omega_j
\|\widetilde\Theta_j\|_F^2.
\tag{12.2}
\]

关键点：

- 权重作用于函数空间核范数，不是 KAN 参数范数；
- 所有 \(\omega_j\) 有限；
- 所有变量仍保留救援机会；
- 权重必须由独立内折生成；
- uniform estimator \(\omega_j=1\) 永远是理论主线；
- adaptive estimator 只有在正式实验稳定改善后才能升级为默认版本。

---

# 13. 预测路径与结构路径重新合并

结构路径得到谱模态：

\[
\widehat K_{j,h}(\tau,u)
=
\sum_{r=1}^{R_j}
\sigma_{j,r,h}
q_{j,r,h}(\tau)
f_{j,r,h}(u).
\tag{13.1}
\]

预测模型为

\[
\widehat y_{t+h\mid t}
=
A_h(\mathcal Y_t^{(L_y)})
+
\sum_{j=1}^{p}
\sum_{r=1}^{R_j}
\sigma_{j,r,h}
\sum_{\tau=0}^{L_x-1}
q_{j,r,h}(\tau)
f_{j,r,h}(x_{j,t-\tau}).
\tag{13.2}
\]

训练顺序：

1. 从完整核 SVD 初始化 \(q,f,\sigma\)；
2. 冻结 rank；
3. 先拟合偏置和模态增益；
4. 再小学习率联合微调；
5. 加入核锚定项
   \[
   \lambda_{\mathrm{anchor}}
   \sum_j
   \|K_j-K_j^{\mathrm{SVD}}\|_{\mathcal H_j}^2;
   \]
6. 不再使用 group-prox；
7. 不在 refit 中删除变量。

预测模型可继续使用 KAN/样条实现 \(f_{j,r}\)，但 KAN 只负责函数逼近，不负责支持选择。

---

# 14. 多视野预测

对

\[
\mathcal H
=
\{1,5,10,30,60\}
\]

采用直接多视野头：

\[
\widehat y_{t+h\mid t}
=
b_h+
C_hs_t.
\]

共享状态提取器，但每个 \(h\) 使用独立读出。

外生增量定义：

\[
\Delta_{X\mid AR}(h)
=
L_{AR}(h)-L_{XAR}(h).
\]

一步预测中的 AR 优势不能被外推为所有视野中的 X 无效。

---

# 15. 从谱核到稳定递推状态

## 15.1 模态卷积

单个谱模态：

\[
v_{j,r,t}
=
f_{j,r}(x_{j,t}),
\]

\[
c_{j,r,t}
=
\sum_{\tau=0}^{L_x-1}
q_{j,r}(\tau)
v_{j,r,t-\tau}.
\]

## 15.2 稳定状态空间逼近

寻找

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
\]

满足

\[
\rho(A_{j,r})<1.
\]

Gamma/Erlang 只是优先尝试的低阶实现，不再是唯一允许的动态核。

## 15.3 BIBO 稳定

若 \(f_{j,r}\) 在适用域有界：

\[
|f_{j,r}(x)|\le M_{j,r},
\]

且

\[
\rho(A_{j,r})<1,
\]

则状态和输出有界。

## 15.4 在线连续性

部署时没有输入相关 gate。只要样条 \(f_{j,r}\) 连续，状态递推线性稳定，则

\[
(\mathcal X_t,\mathcal Y_t)
\mapsto
\widehat y_{t+h\mid t}
\]

连续。若使用三次样条，分段内部可二阶连续。

---

# 16. 总误差分解

对最终部署预测器：

\[
\begin{aligned}
\|\widehat y-y^\star\|
\le
&
\epsilon_{\mathrm{state}}
+
\epsilon_{\mathrm{nuisance}}
+
\epsilon_{\mathrm{basis}}
+
\epsilon_{\mathrm{est}}
\\
&
+
\epsilon_{\mathrm{rank}}
+
\epsilon_{\mathrm{realization}}
+
\epsilon_{\mathrm{quant}}
+
\epsilon_{\mathrm{drift}}.
\end{aligned}
\tag{16.1}
\]

分别表示：

- 预测状态历史不足；
- \(\mu,\pi\) nuisance 估计误差；
- 样条空间逼近误差；
- 有限样本估计误差；
- 谱截断误差；
- 稳定递推逼近误差；
- 定点/浮点量化误差；
- 运行阶段漂移误差。

任何单项实验都不能替代整条误差链。

---

# 17. 主要假设

理论结论只在以下条件下成立。

## A1 阶段局部稳定

研究区间属于同一正常工艺阶段，或漂移足够缓慢。

## A2 条件加性

式 (3.1) 对研究视野成立或具有足够小的逼近误差。

## A3 有限或可截断记忆

超出 \(L_x,L_y\) 的尾部影响可控。

## A4 核平滑

\(K_j\) 属于所选 Sobolev/样条逼近空间的闭包。

## A5 条件外生创新

\[
\mathbb E[
\varepsilon_{t,h}
\mid
Z_t,\phi_t
]
=
0.
\]

## A6 条件激励

残差特征协方差在研究子空间上非退化。

## A7 nuisance 可估计

交叉拟合 nuisance 误差足够小；若用于渐近推断，还需满足相应乘积收敛条件。

## A8 时间依赖可控

序列满足适合 block bootstrap 和经验过程控制的 mixing/弱依赖条件。

## A9 基与正则在测试前固定

grid、平滑和 bootstrap 规则不能根据最终 rank 结果事后调整。

---

# 18. 不可能性边界

不存在一种方法能在所有场景中同时保证：

- 精确恢复真实支持；
- 精确恢复真实 rank；
- 不需要激励；
- 不受闭环共线影响；
- 不需要任何统计判定；
- 对所有工艺阶段全局有效。

若

\[
\phi_{j,t}
=
g(Z_t)
\]

几乎处处成立，则

\[
\widetilde\phi_{j,t}=0,
\]

条件外生作用无法从观测数据中识别。此时模型可以预测，但不能声称恢复真实生成机制。

---

# 19. 可允许与禁止的论文表述

## 19.1 可以表述

- 方法估计的是阶段局部的预测状态和条件外生动态核；
- 全变量 Urysohn 估计在固定基下是强凸问题；
- Scheme A/B 是同一估计核的主模态和谱尾；
- 双残差化减少 AR nuisance 估计对外生核估计的一阶影响；
- 部署 rank 根据误差预算选择；
- 当前真实数据只支持预测与稳定贡献证据，不支持无条件因果结论。

## 19.2 禁止表述

- 恢复了完整物理状态；
- 任意短窗口都满足 Takens 嵌入；
- PIP、gate 或奇异值天然等于因果重要性；
- 所有非零第二奇异值都意味着真实 rank-2；
- 单根晶棒足以证明跨阶段机制；
- 闭环观测数据中可无条件恢复真实操纵变量因果效应。

---

# 20. v0.3 的四种形态

## 20.1 工程训练形态

- 时间轴先切分；
- dense simultaneous XAR 预测；
- 前向 blocked cross-fitting；
- FP64 凸 full-kernel 求解；
- Gram 白化 SVD；
- 无训练期剪枝。

## 20.2 结构设计形态

- 所有变量进入结构估计；
- uniform spectral 为主；
- D5-adaptive spectral 为增强对照；
- A=第一模态；
- B=谱尾；
- 支持与 rank 由统计证据报告。

## 20.3 数学底层形态

- predictive quotient state；
- partially linear Urysohn；
- Neyman-orthogonal double residualization；
- 强凸 Tikhonov；
- basis-invariant Hilbert–Schmidt spectrum；
- 最佳低秩逼近。

## 20.4 实时部署形态

- 固定低秩谱模态；
- 稳定状态空间递推；
- 全部分支可保留；
- 无在线开关；
- 误差预算截断；
- 漂移检测与重新辨识。

---

# 21. 最终统一流程

\[
\boxed{
\begin{aligned}
&\text{时间语义与适用域固定}
\\
\to\;&
\text{dense XAR 多视野预测基线}
\\
\to\;&
\text{前向交叉拟合 } \mu_h(Z),\pi_h(Z)
\\
\to\;&
\text{双残差化}
\\
\to\;&
\text{全变量平滑 Urysohn 强凸估计}
\\
\to\;&
\text{Gram 白化 SVD}
\\
\to\;&
\text{A=第一谱模态，B=谱尾}
\\
\to\;&
\text{support/rank block bootstrap 与外层验证}
\\
\to\;&
\text{SVD 初始化的固定结构 XAR refit}
\\
\to\;&
\text{稳定递推与 MCU/PLC 部署}.
\end{aligned}
}
\]

---

# 附录 A　符号表

| 符号 | 含义 |
|---|---|
| \(z_t\) | 隐状态 |
| \(X_t\) | 外生过程变量向量 |
| \(y_t\) | 输出 |
| \(h\) | 预测视野 |
| \(L_x,L_y\) | 外生、输出历史长度 |
| \(Z_t\) | AR 历史 |
| \(K_{j,h}\) | 第 \(j\) 变量、视野 \(h\) 的 Urysohn 核 |
| \(C(\tau)\) | 时滞基 |
| \(B_j(u)\) | 幅值基 |
| \(\Theta_{j,h}\) | 核系数矩阵 |
| \(\Phi_{j,t}\) | 样本核设计矩阵 |
| \(\mu_h\) | \(E[y_{t+h}\mid Z_t]\) |
| \(\pi_h\) | \(E[\phi_t\mid Z_t]\) |
| \(\widetilde y,\widetilde\phi\) | 双残差 |
| \(G_\tau,G_{x,j}\) | Gram 矩阵 |
| \(\widetilde\Theta_j\) | Gram 白化系数 |
| \(\sigma_{j,r}\) | 核奇异值 |
| \(\eta_j(R)\) | rank-\(R\) 谱尾能量比 |
| \(S_j^K,S_j^C\) | 核强度、贡献强度 |
| \(\Delta_j^{abl}\) | 块消融增益 |

---

# 附录 B　当前证据的正确承接

- D1、D2：由于目标仍含强 AR 项，而模型删除 AR，不能用来判定 rank 容量；新实验必须改为外生真值目标或双残差目标。
- D3：说明给定 AR 后，X 在一步预测中仍保留增量信息，并为 rank-2 residual 提供正面证据。
- D4：说明不使用错误稀疏近端时，simultaneous XAR 可以学习外生贡献。
- D5：说明标准化函数贡献比原 KAN 参数块范数更适合作为支持证据。
- D6：说明主要失败发生在 proximal 持续收缩，而非长期 X 梯度饿死。

这些结果支持 v0.3 的设计方向，但尚未完成 v0.3 的正式验证。
