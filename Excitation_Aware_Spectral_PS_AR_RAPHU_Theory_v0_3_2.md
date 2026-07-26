# Excitation-Aware Spectral Predictive-State AR-RAPHU v0.3.2
## 适用域一致性、模型类分层、经验算子可辨识、全变量谱分解与稳定递推

> **状态**：`THEORY_V0.3.2 / DOMAIN-AND-IDENTIFIABILITY-REPAIRED / PROPOSED_AND_TESTABLE`  
> **继承**：v0.3.1 的 Predictive-State、完整离散时滞母空间、双残差正交化、全变量平滑 Urysohn、Gram 白化谱分解和稳定递推。  
> **修订原因**：v0.3.1 的 E2A 虽然 KKT 残差约为 \(10^{-16}\)，但只有 15/60 个单变量任务通过。进一步检查表明：
>
> 1. 旧幅值基在训练段 \(1\%\)–\(99\%\) 分位区间外执行静默裁剪，而约 57%–68% 的 64 步历史窗口至少含一个被裁剪值；
> 2. 旧 AR-S4 的生成式是条件核 \(K(\tau,u;c)\)，一般不属于二维加性 Urysohn 类 \(K(\tau,u)\)；
> 3. 自然 AR 输入下设计矩阵条件数约为 \(10^9\)，高贡献预测精度不等于完整矩形核面可唯一恢复。
>
> **核心修订**：
>
> \[
> \boxed{
> \text{模型适用域必须覆盖训练历史，禁止静默裁剪；}
> }
> \]
>
> \[
> \boxed{
> \text{二维 Urysohn、条件 Urysohn 与模型错设压力测试必须分层；}
> }
> \]
>
> \[
> \boxed{
> \text{自然工况首先识别经验预测算子等价类；完整核面和真实 rank 需要额外激励证书。}
> }
> \]

---

# 0. 当前冻结证据

## 0.1 已通过

E0 的生成分量恒等式通过：

\[
\max_t
\left|
y_t^{\mathrm{latent}}
-g_t^{AR}
-g_t^X
-\xi_t
\right|
=
1.78\times10^{-15}.
\]

E1R 已认证旧幅值域上的 \(32\times16\) 表示。

## 0.2 已停止

E2A 的数值求解器满足：

\[
r_{\mathrm{KKT}}
\approx10^{-16},
\]

因此不是线性求解器未收敛。

但 60 行中只有 15 行满足旧容量门槛。通过项集中在 AR-S1、AR-S2、AR-S3 的变量 0；变量 1/2 的表面恢复较差，旧 AR-S4 的验证 \(R^2\) 约为 0.43–0.61。

## 0.3 v0.3.2 的解释

冻结标签

```text
STOP_SINGLE_KERNEL_CAPACITY
```

作为旧协议执行结果保留，但科学解释改为：

```text
E2A_PROTOCOL_DOMAIN_MISMATCH
AR_S4_OUTSIDE_2D_URYSOHN_CLASS
NATURAL_INPUT_FULL_SURFACE_IDENTIFIABILITY_UNRESOLVED
```

不得把旧 E2A 直接解释为完整 Urysohn 模型或谱理论失败。

---

# 1. 系统、历史与预测状态

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

- \(z_t\)：潜在物理状态；
- \(X_t=(x_{1,t},\ldots,x_{p,t})^\top\)：过程变量；
- \(y_t\)：被预测输出；
- \(\xi_t,\eta_t\)：过程与测量创新。

定义预测原点 \(t\) 的历史：

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

不读取 \(X_{t+1:t+h}\)。

定义预测等价关系：

\[
z\sim_{\mathrm{pred}}z'
\]

当且仅当两状态在所有 admissible future input sequences 下产生相同的未来输出分布。预测商状态空间为

\[
\mathcal S_{\mathrm{pred}}
=
\mathcal Z/\sim_{\mathrm{pred}}.
\]

本体系只追求对该预测商状态的近似参数化，不声称恢复完整物理状态。

---

# 2. 三类动态核必须分开

## 2.1 M2：二维加性 Urysohn

主模型类为

\[
\boxed{
g_{j,t}
=
\sum_{\tau=0}^{L_x-1}
K_j(\tau,u_{j,t,\tau}),
}
\tag{2.1}
\]

其中

\[
u_{j,t,\tau}
=
x_{j,t-\tau}.
\]

每个时滞位置的贡献只依赖该位置自己的幅值。

条件加性预测模型：

\[
\mathbb E[y_{t+h}\mid Z_t,W_{1,t},\ldots,W_{p,t}]
=
m_h(Z_t)
+
\sum_{j=1}^{p}g_{j,t}^{(h)}.
\tag{2.2}
\]

其中 \(Z_t=\mathcal Y_t^{(L_y)}\)。

## 2.2 M3：条件 Urysohn

更一般的条件核为

\[
\boxed{
g_{j,t}
=
\sum_{\tau=0}^{L_x-1}
K_j(\tau,u_{j,t,\tau};c_{j,t}),
}
\tag{2.3}
\]

其中 \(c_{j,t}\) 是整个历史共享的条件变量，例如当前幅值、工艺阶段或负荷。

M3 含有跨时滞交互，通常不能化为 M2。

## 2.3 M1：rank-1 Hammerstein

M2 的 rank-1 子类：

\[
K_j(\tau,u)=q_j(\tau)f_j(u).
\tag{2.4}
\]

Scheme A 对应 M2 核的最佳 rank-1 谱投影，而不是一个先验资格门。

---

# 3. 旧 AR-S4 为什么不属于 M2

旧 AR-S4 生成器使用

\[
g_{j,t}
=
\sum_{\tau=0}^{L_x-1}
q_j(\tau;c_{j,t})
f_j(u_{j,t,\tau}),
\tag{3.1}
\]

其中

\[
c_{j,t}=x_{j,t}
\]

是所有时滞共享的控制幅值。

取两个时滞 \(0,1\)，则包含项

\[
q_{j,1}(u_0)f_j(u_1).
\]

若它可写成 M2 加性形式

\[
G(u_0,u_1)=K_0(u_0)+K_1(u_1),
\]

则在可微区域必须满足

\[
\frac{\partial^2G}
{\partial u_0\partial u_1}
=0.
\]

但旧 S4 的交互项满足

\[
\frac{\partial^2}
{\partial u_0\partial u_1}
[
q_{j,1}(u_0)f_j(u_1)
]
=
q_{j,1}'(u_0)f_j'(u_1),
\]

一般不为零。

因此，除非 \(q_{j,1}\) 或 \(f_j\) 退化为常数，否则旧 S4 不属于二维加性 Urysohn 类。

旧 S4 正式重命名为：

\[
\boxed{\mathrm{AR\!-\!S4C}}
\]

其中 C 表示 conditional。它只用于模型错设压力测试，不进入 M2 的容量、支持或 rank 通过门槛。

---

# 4. 新 AR-S4U：真正的幅值相关二维 Urysohn

新增场景：

\[
\boxed{\mathrm{AR\!-\!S4U}}
\]

定义

\[
g_{j,t}
=
\sum_{\tau=0}^{L_x-1}
q_j(\tau;u_{j,t,\tau})
f_j(u_{j,t,\tau}),
\tag{4.1}
\]

其中

\[
q_j(\tau;u)
=
\frac{
\exp\!\left[
-\frac{(\tau-c(u))^2}{2\sigma_\tau^2}
\right]
}{
\sum_{s=0}^{L_x-1}
\exp\!\left[
-\frac{(s-c(u))^2}{2\sigma_\tau^2}
\right]
},
\]

\[
c(u)
=
8+
\frac{12}{1+\exp(-2u)},
\qquad
\sigma_\tau=2.
\]

于是

\[
K_j(\tau,u)
=
q_j(\tau;u)f_j(u),
\tag{4.2}
\]

严格属于 M2。

S4U 用来验证“幅值改变动态时滞”但不引入共享条件变量的情况；S4C 用来验证模型错设检测能力。

---

# 5. 幅值适用域与禁止静默裁剪

## 5.1 三个不同的域

对第 \(j\) 个变量，定义：

### 训练覆盖域

\[
\mathcal I_j^{\mathrm{fit}}
=
[
m_j-\delta_j,\,
M_j+\delta_j
],
\tag{5.1}
\]

其中

\[
m_j
=
\min_{t\in\mathrm{train}}x_{j,t},
\qquad
M_j
=
\max_{t\in\mathrm{train}}x_{j,t},
\]

\[
\delta_j
=
0.10(M_j-m_j).
\tag{5.2}
\]

### 核心报告域

\[
\mathcal I_j^{\mathrm{core}}
=
[
Q_{0.01}^{\mathrm{train}}(x_j),
Q_{0.99}^{\mathrm{train}}(x_j)
].
\tag{5.3}
\]

### 运行域

运行时实际访问的幅值集合记为

\[
\mathcal I_j^{\mathrm{run}}.
\]

## 5.2 规则

1. 样条结点覆盖 \(\mathcal I_j^{\mathrm{fit}}\)；
2. 训练历史不得出现域外值；
3. `transform()` 不允许 `np.clip`；
4. 验证/测试域外值必须显式返回 OOD mask；
5. 若结构实验的验证或测试存在任何域外历史值，则该 seed 标记为 `AMPLITUDE_DOMAIN_COVERAGE_FAIL`；
6. 核面主要在 \(\mathcal I_j^{\mathrm{core}}\) 报告；
7. \(\mathcal I_j^{\mathrm{fit}}\) 上的误差作为边界稳定性指标；
8. 真实工业部署应优先使用传感器/工艺允许范围，而不是经验分位数。

## 5.3 静默裁剪改变了模型

设裁剪投影为

\[
\Pi_{\mathcal I}(u)
=
\min(\max(u,a),b).
\]

若设计使用

\[
B(\Pi_{\mathcal I}(u)),
\]

实际拟合的是

\[
K^{\mathrm{clip}}(\tau,u)
=
K(\tau,\Pi_{\mathcal I}(u)),
\]

而不是原核 \(K(\tau,u)\)。

所以静默裁剪不是数值细节，而是模型类变更。

---

# 6. 完整离散时滞母空间与压缩空间

时滞索引为

\[
\mathcal T
=
\{0,\ldots,L_x-1\}.
\]

理论母空间使用单位基：

\[
C_{\mathrm{id}}(\tau)=e_\tau.
\]

幅值方向使用中心化三次 B 样条：

\[
B_j(u)
=
(b_{j,1}(u),\ldots,b_{j,M_x}(u))^\top.
\]

完整核：

\[
K_j(\tau,u)
=
e_\tau^\top
\Theta_j^{\mathrm{id}}
B_j(u).
\tag{6.1}
\]

工程压缩使用

\[
C_M(\tau)\in\mathbb R^M,
\qquad
M<L_x,
\]

\[
K_j^{(M)}(\tau,u)
=
C_M(\tau)^\top
\Theta_j^{(M)}
B_j(u).
\tag{6.2}
\]

压缩空间必须在新适用域和 S4U 上重新认证；旧 E1R 的 \(32\times16\) 结论不能自动迁移到新幅值域。

---

# 7. 设计算子和经验等价类

## 7.1 设计算子

定义样本设计矩阵

\[
\Phi_{j,t}
=
\sum_{\tau=0}^{L_x-1}
C_M(\tau)
B_j(u_{j,t,\tau})^\top.
\tag{7.1}
\]

向量化：

\[
\phi_{j,t}
=
\operatorname{vec}(\Phi_{j,t}),
\qquad
\theta_j
=
\operatorname{vec}(\Theta_j).
\]

贡献为

\[
g_{j,t}
=
\phi_{j,t}^\top\theta_j.
\tag{7.2}
\]

定义线性算子

\[
\mathcal A_{\mathbb P_j}:
\theta_j
\mapsto
\phi_{j,t}^\top\theta_j
\quad
\text{under }W_{j,t}\sim\mathbb P_j.
\]

## 7.2 经验算子半范数

定义

\[
\|\theta\|_{\mathbb P_j}^2
=
\mathbb E_{\mathbb P_j}
[
(\phi^\top\theta)^2
]
=
\theta^\top
Q_j
\theta,
\tag{7.3}
\]

其中

\[
Q_j
=
\mathbb E_{\mathbb P_j}
[\phi\phi^\top].
\]

若 \(Q_j\) 奇异，则它只是半范数。

## 7.3 预测等价核

定义

\[
\theta\sim_{\mathbb P_j}\theta'
\]

当且仅当

\[
\|\theta-\theta'\|_{\mathbb P_j}=0.
\]

等价地：

\[
\mathcal A_{\mathbb P_j}\theta
=
\mathcal A_{\mathbb P_j}\theta'
\quad
\mathbb P_j\text{-a.s.}
\]

自然工况数据首先识别的是等价类

\[
[\theta]_{\mathbb P_j},
\]

而不是任意矩形域上的唯一系数矩阵。

---

# 8. 完整核面可辨识条件

## 8.1 必要条件

若存在非零方向 \(v\) 满足

\[
Q_jv=0,
\]

则

\[
\theta_j
\quad\text{和}\quad
\theta_j+v
\]

产生相同的自然工况贡献，因此无法从该轨迹区分。

## 8.2 充分激励

若在中心化保留子空间上存在

\[
\lambda_{\min}(Q_j)\ge\kappa>0,
\tag{8.1}
\]

则系数在该空间可辨识，并有

\[
\|\theta-\theta^\star\|_2^2
\le
\kappa^{-1}
\|\theta-\theta^\star\|_{\mathbb P_j}^2.
\tag{8.2}
\]

当条件数约为 \(10^9\) 时，式 (8.2) 的误差放大非常严重。因此：

- contribution \(R^2\) 可以接近 1；
- 完整 surface NRMSE 仍可能较大；
- 两者不矛盾。

## 8.3 两种结构结论

### 条件预测结构

在自然工况分布 \(\mathbb P_j^{\mathrm{nat}}\) 下可稳定识别的贡献等价类。

### 全域结构

在独立、充分覆盖的激励分布 \(\mathbb P_j^{\mathrm{exc}}\) 下识别的完整核面。

论文必须明确使用哪一种含义。

---

# 9. 自然输入、去相关输入与空间填充输入

定义三种输入分布。

## 9.1 NAT：自然工况

保持原 AR/闭环输入轨迹。

它检验实际预测能力，但可能存在：

- 高串行相关；
- 幅值覆盖不足；
- 变量共线；
- 条件可辨识性不足。

## 9.2 PERM：保边际去相关

对训练幅值样本做独立随机排列，保持经验边际分布但打破时序相关：

\[
x_t^{\mathrm{perm}}
=
x_{\pi(t)}^{\mathrm{train}}.
\]

它用于判断失败是否主要来自 lag-window 共线性。

## 9.3 SPACE：空间填充激励

在 \(\mathcal I_j^{\mathrm{core}}\) 上使用 scrambled Sobol/Latin-hypercube 序列，近似均匀覆盖幅值域，并构造长度足够的独立时序。

它用于验证完整核面和结构 rank 的恢复能力。

## 9.4 诊断映射

| NAT | PERM | SPACE | 解释 |
|---|---|---|---|
| 通过 | 通过 | 通过 | 自然工况已充分 |
| 贡献通过、面失败 | 通过 | 通过 | 自然输入只识别预测等价类 |
| 失败 | 通过 | 通过 | 主要是串行相关/lag 共线 |
| 失败 | 失败 | 通过 | 主要是幅值覆盖不足 |
| 失败 | 失败 | 失败 | 模型、表示或估计器仍有问题 |

---

# 10. 双残差正交化

完整部分线性模型：

\[
y_{t+h}
=
m_h(Z_t)
+
\phi_t^\top\theta_h
+
\varepsilon_{t,h}.
\tag{10.1}
\]

定义 nuisance：

\[
\mu_h(Z)
=
\mathbb E[y_{t+h}\mid Z],
\]

\[
\pi_h(Z)
=
\mathbb E[\phi_t\mid Z].
\]

双残差：

\[
\widetilde y_t
=
y_{t+h}-\mu_h(Z_t),
\]

\[
\widetilde\phi_t
=
\phi_t-\pi_h(Z_t).
\]

正式结构估计：

\[
\widetilde y_t
=
\widetilde\phi_t^\top\theta_h
+
\varepsilon_{t,h}.
\tag{10.2}
\]

矩条件

\[
\psi
=
(\phi-\pi(Z))
[
y-\mu(Z)
-(\phi-\pi(Z))^\top\theta
]
\]

对 nuisance 的一阶误差具有 Neyman 正交性。

只残差化 \(y\) 不能消除 \(\phi\) 中由历史输出解释的部分，因此只可作为对照。

---

# 11. 平滑全核估计

给定固定 basis 和双残差：

\[
\widehat\theta
=
\arg\min_\theta
\frac1{2n}
\|
\widetilde y-\widetilde\Phi\theta
\|_2^2
+
\sum_j\mathcal R_j(\Theta_j).
\tag{11.1}
\]

其中

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
\tag{11.2}
\]

\(\lambda_0>0\) 时目标强凸，具有唯一正则化最优解。

KKT 通过只说明求解器正确解出式 (11.1)，不能替代：

- 模型类检查；
- 域覆盖检查；
- basis 认证；
- 激励检查。

---

# 12. Scheme A/B 的统一谱定义

Gram 白化：

\[
\widetilde\Theta_j
=
G_\tau^{1/2}
\Theta_j
G_{x,j}^{1/2}.
\]

SVD：

\[
\widetilde\Theta_j
=
\sum_{r\ge1}
\sigma_{j,r}u_{j,r}v_{j,r}^\top.
\]

Scheme A：

\[
\widetilde\Theta_j^A
=
\sigma_{j,1}u_{j,1}v_{j,1}^\top.
\]

Scheme B：

\[
\widetilde\Theta_j^B
=
\sum_{r\ge2}
\sigma_{j,r}u_{j,r}v_{j,r}^\top.
\]

因此：

\[
K_j=K_j^A+K_j^B.
\]

不存在 A 对 B 的资格门。

---

# 13. rank 的两种含义

## 13.1 结构 rank

真实核在 Hilbert–Schmidt 空间中的 rank。它需要：

- 模型类正确；
- basis 误差足够小；
- 激励算子近似可注入；
- 谱间隔大于估计扰动。

由 Weyl 型界：

\[
|\widehat\sigma_r-\sigma_r|
\le
\|\widehat T-T\|_{\mathrm{op}}.
\]

若自然输入只能识别等价类，则真实结构 rank 也未必可识别。

## 13.2 预测 rank

在给定数据分布下，使样本外预测误差达到预算所需的最小谱截断阶数：

\[
R_j^{\mathrm{pred}}
=
\min
\left\{
R:
L_j(R)-L_j(\mathrm{full})
\le\epsilon_{\mathrm{pred}}
\right\}.
\]

真实工业数据默认报告预测 rank，而不是无条件声称结构 rank。

## 13.3 结构 rank 的正式验证域

结构 rank 的主要合成验证必须在 SPACE 激励上完成；NAT 只用于预测 rank 和稳定性对照。

---

# 14. support 的两种含义

## 14.1 条件预测 support

给定历史输出和自然工况分布后，变量是否提供样本外增量预测：

\[
\Delta_{j,h}^{\mathrm{pred}}
=
L_{-j,h}^{\mathrm{val}}
-
L_{\mathrm{full},h}^{\mathrm{val}}.
\]

## 14.2 生成结构 support

真实核是否非零：

\[
\|K_j\|_{\mathcal H_j}>0.
\]

生成结构 support 的恢复同样需要充分激励。闭环单轨迹中一般不能把预测 support 直接解释为物理因果 support。

---

# 15. D5-adaptive 的位置

D5 标准化贡献路径只产生辅助权重：

\[
\omega_j
=
\operatorname{clip}
\left(
\frac{\operatorname{median}_k(s_k+\epsilon)}
{s_j+\epsilon},
\omega_{\min},\omega_{\max}
\right).
\]

它作用于函数空间核范数：

\[
\lambda_0
\sum_j
\omega_j
\|\widetilde\Theta_j\|_F^2.
\]

Uniform spectral 始终为主线。D5-adaptive 不决定变量能否进入模型，也不进入 rank 定义。

---

# 16. 预测路径、结构路径和错设路径

## 16.1 P：预测路径

dense simultaneous XAR，负责：

- 多视野 RMSE；
- \(\Delta_{X\mid AR}(h)\)；
- 实时部署；
- 漂移监测。

## 16.2 S：二维结构路径

只处理声明属于 M2 的场景：

\[
\text{S1,S2,S3,S4U}
\]

以及经过语义检查的真实数据假设。

流程：

\[
\text{domain-safe basis}
\to
\text{double residualization}
\to
\text{full Urysohn}
\to
\text{Gram SVD}.
\]

## 16.3 C：条件核扩展路径

旧 S4C 属于 M3：

\[
K(\tau,u;c).
\]

当前 v0.3.2 只把它作为错设压力测试。三维条件核是未来 Scheme C，不在本轮实现。

---

# 17. 稳定递推部署

谱模态：

\[
K_j(\tau,u)
=
\sum_{r=1}^{R_j}
\sigma_{j,r}q_{j,r}(\tau)f_{j,r}(u).
\]

定义

\[
v_{j,r,t}
=
f_{j,r}(x_{j,t}),
\]

并用稳定状态空间逼近：

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

Gamma/Erlang、Laguerre、Kautz 或一般稳定 realization 由误差预算选择。部署时没有在线 support gate 或 rank 开关。

---

# 18. 总误差分解

\[
\begin{aligned}
\|\widehat y-y^\star\|
\le&
\epsilon_{\mathrm{state}}
+
\epsilon_{\mathrm{model\ class}}
+
\epsilon_{\mathrm{domain}}
\\
&
+
\epsilon_{\mathrm{lag\ basis}}
+
\epsilon_{\mathrm{amp\ basis}}
+
\epsilon_{\mathrm{nuisance}}
\\
&
+
\epsilon_{\mathrm{excitation}}
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
\tag{18.1}
\]

旧 E1 对应 \(\epsilon_{\mathrm{lag\ basis}}\) 过大。

旧 E2A 同时混入：

- \(\epsilon_{\mathrm{domain}}\)：静默裁剪；
- \(\epsilon_{\mathrm{model\ class}}\)：S4C；
- \(\epsilon_{\mathrm{excitation}}\)：自然设计病态。

v0.3.2 将三项分开验证。

---

# 19. 理论假设

## A1 阶段局部稳定  
当前研究区间属于同一工艺阶段或缓慢漂移阶段。

## A2 模型类声明正确  
进入 M2 核验证的场景必须满足式 (2.1)。

## A3 幅值域覆盖  
训练历史完全位于 \(\mathcal I^{\mathrm{fit}}\)，无静默裁剪。

## A4 basis 已认证  
在新适用域和 S4U 上完成独立投影认证。

## A5 条件加性  
变量间显式交互可以忽略或被另行建模。

## A6 条件外生创新  
\[
\mathbb E[\varepsilon\mid Z,\phi]=0.
\]

## A7 nuisance 可估计  
前向 cross-fitting 产生有效 \(\widehat\mu,\widehat\pi\)。

## A8 自然预测可辨识  
经验算子在预测相关子空间上具有足够有效秩。

## A9 全域结构可辨识  
只有在 SPACE 激励下验证满足充分覆盖。

## A10 选择分离  
basis、平滑、support 和 rank 的选择信息互不泄漏。

---

# 20. 允许的科学结论

## 自然工况通过 E2A-NAT

可以声明：

> 在自然工况输入分布上，该核的外生贡献可被准确预测。

不能自动声明完整二维核面唯一恢复。

## SPACE 通过 E2A-SPACE

可以声明：

> 在充分覆盖的合成激励下，完整二维核面可恢复到预注册误差。

## NAT 面误差大但 SPACE 通过

应声明：

> 当前自然轨迹识别了预测等价类，但不足以唯一恢复全域核面。

## S4C 失败

应声明：

> 二维 Urysohn 正确拒绝或无法拟合条件交互核；该场景需要条件 Urysohn 扩展。

不能把它解释为主模型容量失败。

---

# 21. v0.3.2 统一理论流程

\[
\boxed{
\begin{aligned}
&\text{R0：冻结旧停止线}
\\
\to\;&
\text{R1：域裁剪与模型类语义审计}
\\
\to\;&
\text{R2：无裁剪幅值基与 S4U}
\\
\to\;&
\text{E1A：新域表示认证}
\\
\to\;&
\text{E2A0：实现与算子闭环一致性}
\\
\to\;&
\text{E2A-NAT：自然输入贡献容量}
\\
\to\;&
\text{E2A-PERM/SPACE：激励与全域结构容量}
\\
\to\;&
\text{E2B：多变量联合容量}
\\
\to\;&
\text{E3：双残差结构估计}
\\
\to\;&
\text{E4P/E4S：预测 support / 结构 support}
\\
\to\;&
\text{E5P/E5S：预测 rank / 结构 rank}
\\
\to\;&
\text{E6：D5-adaptive 对照}
\\
\to\;&
\text{E7：谱结构重并入预测模型}
\\
\to\;&
\text{E8：稳定递推部署}.
\end{aligned}
}
\]
