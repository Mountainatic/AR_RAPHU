# OPS-UOI → Physics-First Orthogonally Decomposed Urysohn Surface → Residual Predictive-State AR → CUDA/PLC
## 三层完整理论、辨识、数值实现与工业部署体系 v3.2

> **文档性质**：项目总理论、模型、辨识、实验合同与部署主文档  
> **版本日期**：2026-07-31  
> **替代版本**：`OPS_UOI_PS_AR_RAPHU_CUDA_PLC_Three_Layer_Complete_System_v3_1.md`  
> **当前应用实例**：CZ 法单晶硅等径阶段的晶体直径增量预测  
> **当前科学状态**：`THEORY_V3.2 / FULL_URYSOHN_RESTORED / NLINEAR_AS_DERIVED_LINEAR_PROJECTION / SHARED_PRIVATE_WITHDRAWN`  
>
> v3.1 中提出的“公共时滞字典 + 通道私有字典”已完成冻结确认实验，但最终得到
>
> ```text
> LEVEL_D_REJECTED
> shared rank = 0
> private = exact zero
> ```
>
> 因而该结构不再作为正式主模型。v3.2 恢复完整的多通道二维 Urysohn 面，不再把它拆成公共/私有字典，也不把 NLinear 改写成额外的共享 FIR 层或外挂通道权重。
>
> 本版的核心是：
>
> \[
> \boxed{
> \text{完整 Urysohn 面保持不变；NLinear 只作为其线性幅值投影的 Rank-1 性质。}
> }
> \]
>
> \[
> \boxed{
> K_j(\tau,u)
> =
> \beta_j(\tau)\,\xi_j(u)
> +
> N_j(\tau,u),
> \qquad
> N_j\perp\operatorname{span}\{1,\xi_j\}.
> }
> \]
>
> 这里 \(\beta_j\) 和 \(N_j\) 都是由同一个已拟合 Urysohn 面经过唯一正交投影得到的对象，不增加新的模型层，不增加外挂训练权重。
>
> 正式预测仍遵循：
>
> \[
> \boxed{
> \text{Physics-first full Urysohn }K
> \longrightarrow
> \text{matured residual Predictive-State/AR，可精确为零}.
> }
> \]

---

# 0. v3.2 的结论性修订

## 0.1 正式保留

1. OPS-UOI 的算子优先思想；
2. 被动闭环自然数据不能无条件等同于开放环 plant 的科学边界；
3. 每个物理输入通道拥有完整二维有限记忆 Urysohn 面；
4. Physics-first \(K\rightarrow\) matured residual PS/AR；
5. 无未来输入、断点隔离、outer cross-rod transfer、purge、placebo 和 block bootstrap；
6. CPU FP64 参考求解、CUDA 批量算子与 PLC/MCU 编译；
7. 非线性幅值域外的有限带 \(C^1\) 延拓；
8. 可辨识性、跨棒稳定性和物理声明边界；
9. NLinear、线性动态核和 full Urysohn 之间的严格嵌套关系；
10. 非线性 Urysohn 面的谱分解只用于描述、诊断和部署压缩，不再作为模型资格门控。

## 0.2 正式撤销

以下内容不再属于主模型：

1. 公共时滞字典；
2. 通道私有时滞字典；
3. shared/private Rank；
4. participation gate；
5. 公共—私有正交残差搜索；
6. “先 full K，再用后验 pooled SVD 生成物理公共字典”的主估计路径；
7. NLinear 作为额外共享 FIR 层；
8. 外挂通道训练权重 \(a_j\)；
9. universal Rank 2；
10. 两个独立手工正则超参数 \((\lambda_0,\lambda_2)\)；
11. 为了降低 RMSE 而增加未认证字典、私有分支或 Rank 救援；
12. 用 Rank 解释所有误差来源；
13. 非线性未解锁时写成“再次证明非线性为零”。

## 0.3 NLinear 的新位置

NLinear 不再是正式模型层。它只对应完整 Urysohn 面内部的一个派生命题：

\[
\boxed{
N_j(\tau,u)=0
\quad\text{且}\quad
\operatorname{rank}B=1,
}
\]

其中 \(B\) 是各通道线性幅值投影 \(\beta_j(\tau)\) 组成的通道—时滞矩阵。

因此：

- 不新增 \(a_j\)；
- 不新增共享 \(q(\tau)\)；
- 不直接训练 Rank-1 因子；
- 不用 NLinear 取代二维 Urysohn 面；
- 只在完整面拟合后检查其线性投影是否接近 Rank 1。

## 0.4 当前主方程

对预测视野 \(h\) 和输出平均窗口 \(W\)，定义：

\[
z_{t,h,W}
=
\overline y_{t+h}^{(W)}
-
\overline y_t^{(W)}.
\tag{0.1}
\]

第 \(j\) 个输入通道的物理响应为：

\[
\mathcal K_{j,h}[u_j](t)
=
\int_0^{T_j}
K_{j,h}\!\left(\tau,u_j(t-\tau)\right)\,d\tau.
\tag{0.2}
\]

离散采样下：

\[
\mathcal K_{j,h}[u_j](t)
=
\sum_{\ell=0}^{L_j-1}
K_{j,h}\!\left(\ell\Delta t,u_{j,t-\ell}\right)\Delta t.
\tag{0.3}
\]

完整预测为：

\[
\boxed{
\widehat{\overline y}_{t+h}^{(W)}
=
\overline y_t^{(W)}
+
b_h
+
\sum_{j=1}^{p}
\widehat{\mathcal K}_{j,h}[u_j](t)
+
\widehat{\mathcal A}_h
\!\left(R_{t,\mathrm{mature}}^-\right).
}
\tag{0.4}
\]

其中：

- \(K_j(\tau,u)\) 是完整 Urysohn 面；
- \(b_h\) 是不可分配到单个通道—时滞常数截面的总截距；
- \(\mathcal A_h\) 只使用成熟残差，且允许精确为零；
- 不存在独立 Q 层；
- 不存在 shared/private 字典层。

---

# 1. 三层体系

## 1.1 第一层：OPS-UOI

OPS-UOI 回答：

1. 哪个输入历史算子在当前闭环观测设计上可稳定估计？
2. 哪些 Urysohn 面在跨棒、时间块、共同支持和 placebo 下可复现？
3. 当前对象只是 closed-loop observational response，还是满足更强条件后可解释为 plant-related kernel？
4. 估计误差来自时滞支持、函数逼近、统计噪声、共线性还是结构错误？
5. 哪些结论可以用于预测，哪些只能用于结构诊断？

## 1.2 第二层：完整 Urysohn 面 + 正交幅值分解 + Residual PS/AR

中层包含：

1. 多通道有限记忆 Urysohn 面；
2. 每个面内部的常数、线性幅值和纯非线性正交分解；
3. NLinear Rank-1 作为线性投影的派生诊断；
4. full Urysohn 与 linear Urysohn 的嵌套比较；
5. 非线性曲面残差的 Hilbert–Schmidt 谱；
6. 冻结 K 后的成熟残差 PS/AR。

## 1.3 第三层：CPU / CUDA / PLC

底层负责：

- CPU FP64 唯一参考实现；
- 稀疏/矩阵自由前向和伴随；
- CUDA 批量张量积与 bootstrap；
- spline/LUT/FIR/低秩非线性曲面压缩；
- PLC/MCU 固定点部署；
- \(C^1\) 延拓和 OOD 安全策略。

计算层不能改变 estimand，不能以 GPU 速度替代跨棒证据。

---

# 2. 闭环数据、变量和预测目标

## 2.1 闭环过程

设过程隐状态为 \(x_t\)，控制输入为 \(u_t\)，未测扰动为 \(w_t\)：

\[
x_{t+1}=F_{\vartheta_t}(x_t,u_t,w_t),
\tag{2.1}
\]

\[
y_t=h(x_t)+\nu_t.
\tag{2.2}
\]

控制器可能满足：

\[
u_t=\pi_t(y_{\le t},r_{\le t},\zeta_t).
\tag{2.3}
\]

因此从自然闭环数据估计的 \(K_j\) 首先是：

\[
\boxed{
\text{registered closed-loop input-history response surface},
}
\]

而不是无条件的开放环 plant impulse response。

## 2.2 当前 CZ 输入

冻结的四个输入为：

1. 主加热功率；
2. 联合升速；
3. 晶转速度；
4. 埚转速度。

联合升速由训练棒内 PCA PC1 得到，符号冻结为正，并原样应用于验证棒和测试棒。

严格表述：

> 晶升与埚升作为联合升速控制变量处理；现有数据不再尝试分离二者独立贡献。

当前排除：

- 晶体长度；
- 加热元件温度；
- 氩气流量；
- 炉压。

排除只表示本版本未把它们注册为输入，不表示这些变量没有物理作用。

## 2.3 冻结 L6 任务

当前主任务：

- cadence：10 s；
- 输入历史：40 min；
- horizon：20 min；
- 输出未来平均窗口：2 min；
- 目标：未来 2 min 平均直径相对当前 2 min 平均直径的变化。

\[
z_t
=
\overline y_{t+20\mathrm{min}}^{(2\mathrm{min})}
-
\overline y_t^{(2\mathrm{min})}.
\tag{2.4}
\]

## 2.4 严格因果

预测时刻 \(t\) 只允许使用 \(t\) 及以前的输入和输出历史。

定义残差：

\[
r_s=z_s-\widehat K(U_s^-).
\tag{2.5}
\]

只有满足：

\[
s+h+W\le t
\tag{2.6}
\]

的残差才在时刻 \(t\) 成熟并可进入 AR/PS。

---

# 3. 完整多通道 Urysohn 面

## 3.1 核空间

对通道 \(j\)，定义时滞域：

\[
\mathcal T_j=[0,T_j],
\]

训练幅值支持：

\[
\mathcal U_j=[a_j,b_j].
\]

设 \(\nu_j\) 为训练棒上通道 \(j\) 的经验幅值测度，\(\omega_j\) 为时滞权重。基础 Hilbert 空间：

\[
\mathcal H_j
=
L_2\!\left(
\mathcal T_j\times\mathcal U_j,
\omega_j(d\tau)\nu_j(du)
\right).
\tag{3.1}
\]

核内积：

\[
\langle K,G\rangle_{\mathcal H_j}
=
\int_{\mathcal T_j}
\int_{\mathcal U_j}
K(\tau,u)G(\tau,u)\,
\nu_j(du)\omega_j(d\tau).
\tag{3.2}
\]

完整物理层：

\[
\widehat z_t^K
=
b
+
\sum_{j=1}^{p}
\int_{\mathcal T_j}
K_j(\tau,u_j(t-\tau))\,d\tau.
\tag{3.3}
\]

## 3.2 不对 Urysohn 面预先降秩

正式模型不预先假设：

\[
K_j(\tau,u)=q_j(\tau)f_j(u),
\]

也不预先假设多个通道共享同一 \(q(\tau)\)。

Rank 只在完整面拟合后用于：

- 描述；
- 诊断；
- 压缩；
- PLC 编译误差控制。

不再用 Rank 决定完整 Urysohn 面是否有资格存在。

## 3.3 加性通道的科学边界

由于闭环输入可能高度共线，单个通道面的归因未必稳定。即使总预测算子可稳定，通道分解也可能在近似等价方向间旋转。

因此必须区分：

\[
\text{total input-history predictor stability}
\]

和

\[
\text{channel-specific surface stability}.
\]

只有当通道面在跨棒、fold、common-support 和 placebo 下稳定时，才允许对该通道作物理归因。

---

# 4. Urysohn 面内的唯一正交幅值分解

## 4.1 标准化幅值坐标

对通道 \(j\)，训练幅值均值和标准差为：

\[
\mu_j=\int u\,\nu_j(du),
\qquad
s_j^2=\int(u-\mu_j)^2\,\nu_j(du)>0.
\tag{4.1}
\]

定义：

\[
\xi_j(u)=\frac{u-\mu_j}{s_j}.
\tag{4.2}
\]

于是：

\[
\langle 1,\xi_j\rangle_{\nu_j}=0,
\qquad
\|\xi_j\|_{\nu_j}=1.
\tag{4.3}
\]

## 4.2 常数截面、线性幅值截面和纯非线性残差

对几乎处处的 \(\tau\)，定义：

\[
m_j(\tau)
=
\int K_j(\tau,u)\,\nu_j(du),
\tag{4.4}
\]

\[
\beta_j(\tau)
=
\int
\bigl[K_j(\tau,u)-m_j(\tau)\bigr]
\xi_j(u)\,\nu_j(du),
\tag{4.5}
\]

\[
N_j(\tau,u)
=
K_j(\tau,u)
-
m_j(\tau)
-
\beta_j(\tau)\xi_j(u).
\tag{4.6}
\]

则：

\[
\int N_j(\tau,u)\,\nu_j(du)=0,
\tag{4.7}
\]

\[
\int N_j(\tau,u)\xi_j(u)\,\nu_j(du)=0.
\tag{4.8}
\]

## 4.3 常数规范

对固定历史长度，\(m_j(\tau)\) 在输入幅值上为常数，其总贡献：

\[
c_j=\int_{\mathcal T_j}m_j(\tau)\,d\tau
\tag{4.9}
\]

可吸收到总截距：

\[
b^\star=b+\sum_jc_j.
\tag{4.10}
\]

因此可采用规范：

\[
m_j(\tau)\equiv0,
\tag{4.11}
\]

并写成：

\[
\boxed{
K_j(\tau,u)
=
\beta_j(\tau)\xi_j(u)
+
N_j(\tau,u).
}
\tag{4.12}
\]

该规范消除通道—时滞常数截面的不可辨识自由度。

## 4.4 定理 4.1：唯一正交分解

**定理。**  
设 \(K_j(\tau,\cdot)\in L_2(\nu_j)\)，且 \(s_j>0\)。则对几乎处处的 \(\tau\)，存在唯一的三元组：

\[
m_j(\tau)\in\mathbb R,\qquad
\beta_j(\tau)\in\mathbb R,\qquad
N_j(\tau,\cdot)\in
\operatorname{span}\{1,\xi_j\}^{\perp},
\]

使得：

\[
K_j(\tau,u)
=
m_j(\tau)
+
\beta_j(\tau)\xi_j(u)
+
N_j(\tau,u).
\tag{4.13}
\]

### 证明

由于 \(\{1,\xi_j\}\) 在 \(L_2(\nu_j)\) 中标准正交，子空间：

\[
\mathcal L_j=\operatorname{span}\{1,\xi_j\}
\]

是闭的。Hilbert 空间正交投影定理保证：

\[
K_j(\tau,\cdot)
=
P_{\mathcal L_j}K_j(\tau,\cdot)
+
(I-P_{\mathcal L_j})K_j(\tau,\cdot),
\]

且分解唯一。

标准正交基下：

\[
P_{\mathcal L_j}K_j
=
\langle K_j,1\rangle_{\nu_j}\,1
+
\langle K_j,\xi_j\rangle_{\nu_j}\,\xi_j.
\]

这正是式 (4.4)–(4.6)。证毕。

## 4.5 Pythagorean 恒等式

对每个 \(\tau\)：

\[
\|K_j(\tau,\cdot)\|_{\nu_j}^2
=
m_j(\tau)^2
+
\beta_j(\tau)^2
+
\|N_j(\tau,\cdot)\|_{\nu_j}^2.
\tag{4.14}
\]

积分后：

\[
\|K_j\|_{\mathcal H_j}^2
=
\|m_j\|_{\omega_j}^2
+
\|\beta_j\|_{\omega_j}^2
+
\|N_j\|_{\mathcal H_j}^2.
\tag{4.15}
\]

注意：这是核空间能量恒等式，不自动等于预测 MSE 的正交分解，因为闭环设计可能使不同通道和时滞特征相关。

## 4.6 解释

- \(m_j(\tau)\)：幅值常数截面，吸收进截距；
- \(\beta_j(\tau)\xi_j(u)\)：该 Urysohn 面的线性幅值投影；
- \(N_j(\tau,u)\)：与常数和线性幅值都正交的纯非线性曲面。

这样 NLinear 被放在完整 Urysohn 面内部，而不是放在其外部。

---

# 5. NLinear 作为派生 Rank-1 性质

## 5.1 通道—时滞线性投影矩阵

在离散时滞网格 \(\tau_1,\ldots,\tau_L\) 上，定义：

\[
B=
\begin{bmatrix}
\beta_1(\tau_1)&\cdots&\beta_p(\tau_1)\\
\vdots&&\vdots\\
\beta_1(\tau_L)&\cdots&\beta_p(\tau_L)
\end{bmatrix}
\in\mathbb R^{L\times p}.
\tag{5.1}
\]

一般线性 Urysohn 子模型为：

\[
K_j^{\mathrm{LIN}}(\tau,u)
=
\beta_j(\tau)\xi_j(u).
\tag{5.2}
\]

## 5.2 定理 5.1：NLinear 等价条件

**定理。**  
以下两条等价：

1. 存在某个时间形状 \(q\in\mathbb R^L\) 和通道坐标 \(a\in\mathbb R^p\)，使：
   \[
   \beta_j(\tau_\ell)=q_\ell a_j;
   \]
2. 
   \[
   \operatorname{rank}(B)\le1.
   \]

### 证明

若 \(B=qa^\top\)，则矩阵秩至多 1。

反之，若 \(\operatorname{rank}(B)\le1\)，则由秩一矩阵分解定理，存在 \(q,a\) 使 \(B=qa^\top\)。证毕。

这里 \(q\) 和 \(a\) 是已估计线性投影矩阵的坐标表达，不是完整 Urysohn 模型中新增的训练权重。

## 5.3 严格嵌套

定义：

### Zero

\[
K_j\equiv0.
\]

### NLinear 投影子类

\[
N_j\equiv0,
\qquad
\operatorname{rank}(B)\le1.
\]

### General Linear Urysohn

\[
N_j\equiv0,
\qquad
B\ \text{任意}.
\]

### Full Urysohn

\[
N_j\ \text{任意满足正交约束}.
\]

因此：

\[
\boxed{
\text{Zero}
\subset
\text{NLinear}
\subset
\text{Linear Urysohn}
\subset
\text{Full Urysohn}.
}
\tag{5.3}
\]

## 5.4 Gram-SVD 与最佳 Rank-1 派生投影

若时滞坐标基为 \(C(\tau)\)，Gram 矩阵为：

\[
G_\tau
=
\int C(\tau)C(\tau)^\top\,\omega(d\tau),
\tag{5.4}
\]

线性投影系数矩阵为 \(\Theta_\beta\)。白化：

\[
\widetilde B=G_\tau^{1/2}\Theta_\beta.
\tag{5.5}
\]

令：

\[
\widetilde B=U\Sigma V^\top.
\tag{5.6}
\]

最佳 Rank-1 派生近似为：

\[
\widetilde B_1
=
\sigma_1u_1v_1^\top.
\tag{5.7}
\]

由 Eckart–Young–Mirsky 定理：

\[
\widetilde B_1
=
\arg\min_{\operatorname{rank}(M)\le1}
\|\widetilde B-M\|_F.
\tag{5.8}
\]

Rank-1 能量比例：

\[
\rho_{\mathrm{NLinear}}
=
\frac{\sigma_1^2}{\sum_r\sigma_r^2}.
\tag{5.9}
\]

该量只回答：

> 完整 Urysohn 面中的线性幅值投影有多接近一个共享时滞方向？

它不回答：

- 完整 Urysohn 面是否线性；
- Rank-1 是否为唯一物理机制；
- 闭环共线性是否已经解除；
- NLinear 是否优于 full Urysohn 的外层预测。

## 5.5 不再直接训练 Rank-1 因子

正式辨识先得到完整 \(K_j\)，随后通过唯一投影得到 \(\beta_j\)，再计算 \(B\) 的谱。

不直接优化：

\[
q,\ a,\ N
\]

的联合非凸目标。

因此：

- 不新增随机种子；
- 不新增通道训练权重；
- 不新增 Rank 超参数；
- 不允许 Rank-1 失败后自动升级 Rank-2 救援；
- NLinear 只作为派生结构证据。

---

# 6. 完整 Urysohn 面的有限表示

## 6.1 固定数值坐标，不称为物理字典

为数值求解，取固定 tensor-product cubic B-spline 坐标：

\[
K_j(\tau,u)
=
C_\tau(\tau)^\top
\Theta_j
C_{u,j}(u).
\tag{6.1}
\]

这里 \(C_\tau,C_{u,j}\) 只是有限元/样条坐标，不是物理公共字典，也不承担“所有变量共享同一时间形状”的含义。

## 6.2 固定网格原则

- 时滞支持由冻结任务决定；
- 幅值 knots 只由训练棒分位数确定；
- 测试棒不能修改 knots；
- 不搜索公共/私有字典；
- 不搜索 Rank；
- 网格加密只作为数值敏感性审计，不作为测试性能调参。

## 6.3 唯一规范约束

离散幅值 Gram 矩阵为 \(G_{u,j}\)，常数向量为 \(c_{0,j}\)，标准线性坐标向量为 \(c_{1,j}\)。

纯非线性系数必须满足：

\[
c_{0,j}^\top G_{u,j}\Theta_{N,j}^\top=0,
\tag{6.2}
\]

\[
c_{1,j}^\top G_{u,j}\Theta_{N,j}^\top=0.
\tag{6.3}
\]

这样非线性曲面不能重复拟合常数和线性幅值分量。

---

# 7. 单一自动平滑几何

## 7.1 为什么仍需要正则化

完整二维 Urysohn 面是闭环病态逆问题。若完全不正则，常见结果是：

- 通道间代理旋转；
- 时滞方向高频振荡；
- 幅值边界爆炸；
- 跨棒核极不稳定；
- 极小训练误差但无外层迁移。

因此正则化是估计器的数值定义，不是额外物理层。

## 7.2 归一化坐标

定义：

\[
s=\tau/T_j\in[0,1],
\qquad
v=(u-\mu_j)/s_j.
\tag{7.1}
\]

在归一化坐标中定义统一 Sobolev 曲率：

\[
\mathcal J(K_j)
=
\int
\left[
\left(\partial_{ss}K_j\right)^2
+
2\left(\partial_{sv}K_j\right)^2
+
\left(\partial_{vv}K_j\right)^2
\right]\,ds\,d\nu_j(v).
\tag{7.2}
\]

为保证范数正定，定义：

\[
\|K_j\|_{\mathcal H_j^\star}^2
=
\|K_j\|_{L_2}^2
+
\mathcal J(K_j).
\tag{7.3}
\]

## 7.3 单一估计尺度

求解：

\[
\widehat K_\lambda
=
\arg\min_{b,K_1,\ldots,K_p}
\left\{
\frac1n
\sum_t
\left[
z_t-b-\sum_j\mathcal K_j[u_j](t)
\right]^2
+
\lambda
\sum_j
\|K_j\|_{\mathcal H_j^\star}^2
\right\}.
\tag{7.4}
\]

只有一个全局 \(\lambda\)。它不是手工模型超参数，而由训练棒内部 GCV 自动选择：

\[
\widehat\lambda
=
\arg\min_{\lambda>0}
\operatorname{GCV}(\lambda).
\tag{7.5}
\]

不使用：

- 通道权重；
- 独立 lag penalty；
- 独立 amplitude penalty；
- shared/private penalty；
- Rank penalty；
- 测试集调参。

## 7.4 定理 7.1：有限系数下的唯一解

在固定 spline 坐标、\(\lambda>0\) 且范数矩阵正定时，式 (7.4) 对 \((b,\Theta_1,\ldots,\Theta_p)\) 是严格凸二次问题，因此存在唯一解。

### 证明

目标 Hessian 为：

\[
H
=
\frac{2}{n}\Phi^\top\Phi
+
2\lambda P,
\]

其中 \(P\succ0\)。故：

\[
x^\top Hx
=
\frac{2}{n}\|\Phi x\|_2^2
+
2\lambda x^\top Px
>0
\]

对任意 \(x\ne0\) 成立，因此 \(H\succ0\)，解唯一。证毕。

## 7.5 可辨识性仍需外层证书

唯一的正则化解不等于唯一物理真值。必须继续报告：

- 有效设计 Gram 条件数；
- 通道删除敏感性；
- 跨棒面相关；
- common-support 一致性；
- placebo；
- block bootstrap；
- OOD 比例。

---

# 8. 训练后的派生对象

## 8.1 完整面

\[
\widehat K_j^{\mathrm{FULL}}
=
\widehat\beta_j\xi_j+\widehat N_j.
\tag{8.1}
\]

## 8.2 一般线性投影

\[
\widehat K_j^{\mathrm{LIN}}
=
\widehat\beta_j\xi_j.
\tag{8.2}
\]

## 8.3 派生 NLinear Rank-1 投影

对 \(\widehat B\) 做 Gram-SVD，得到：

\[
\widehat B_1.
\]

据此构造：

\[
\widehat K_j^{\mathrm{R1-LIN}}.
\tag{8.3}
\]

该对象不重新训练，只是完整面线性投影的最佳 Rank-1 核空间近似。

## 8.4 非线性增量

\[
\widehat K_j^{\mathrm{NL}}
=
\widehat N_j.
\tag{8.4}
\]

必须分别报告：

- \(\|\widehat\beta_j\|^2\)；
- \(\|\widehat N_j\|^2\)；
- \(\rho_{\mathrm{NLinear}}\)；
- full 与 linear 的逐样本误差差；
- full 与 Rank-1 linear 的误差差。

不能仅凭核能量比例宣称预测贡献。

---

# 9. 非线性曲面的谱

## 9.1 只对纯非线性残差分解

对：

\[
N_j(\tau,u)
\]

做 Hilbert–Schmidt 分解：

\[
N_j(\tau,u)
=
\sum_{r\ge1}
\sigma_{j,r}^{\mathrm{NL}}
q_{j,r}^{\mathrm{NL}}(\tau)
f_{j,r}^{\mathrm{NL}}(u),
\tag{9.1}
\]

其中：

\[
\langle f_{j,r}^{\mathrm{NL}},1\rangle_{\nu_j}=0,
\tag{9.2}
\]

\[
\langle f_{j,r}^{\mathrm{NL}},\xi_j\rangle_{\nu_j}=0.
\tag{9.3}
\]

因此非线性谱不能复制线性幅值投影。

## 9.2 Rank 的新语义

非线性 Rank 只用于：

- 描述曲面复杂度；
- 选择部署压缩阶数；
- 计算截断误差；
- 比较跨棒非线性子空间。

它不用于：

- 决定是否训练 full Urysohn；
- 替代完整面；
- 自动开放私有分支；
- 作为预测失败后的救援。

## 9.3 部署 Rank

给定部署误差预算 \(\varepsilon_{\mathrm{deploy}}\)，选择：

\[
R_{j,\mathrm{deploy}}
=
\min
\left\{
R:
\frac{
\sum_{r>R}
(\sigma_{j,r}^{\mathrm{NL}})^2
}{
\sum_r
(\sigma_{j,r}^{\mathrm{NL}})^2+\epsilon
}
\le
\varepsilon_{\mathrm{deploy}}
\right\}.
\tag{9.4}
\]

这是编译参数，不是科学模型 Rank。

---

# 10. 有限带 \(C^1\) 幅值延拓

## 10.1 适用对象

线性部分：

\[
\beta_j(\tau)\xi_j(u)
\]

在 \(u\in\mathbb R\) 上全局光滑，不需要延拓。

纯非线性曲面：

\[
N_j(\tau,u)
\]

只在训练支持 \([a_j,b_j]\) 内受数据约束，因此需要有限带 \(C^1\) 延拓。

## 10.2 左延拓

对固定 \(\tau\)，令：

\[
y_a=N_j(\tau,a_j),
\qquad
d_a=\partial_uN_j(\tau,a_j).
\]

取冻结带宽 \(\delta_{L,j}>0\)，定义：

\[
c_L(\tau)=y_a-\delta_{L,j}d_a.
\tag{10.1}
\]

三次 Hermite 基：

\[
h_{00}(s)=2s^3-3s^2+1,
\]

\[
h_{10}(s)=s^3-2s^2+s,
\]

\[
h_{01}(s)=-2s^3+3s^2,
\]

\[
h_{11}(s)=s^3-s^2.
\tag{10.2}
\]

当 \(u\in[a_j-\delta_{L,j},a_j]\)：

\[
s=\frac{u-(a_j-\delta_{L,j})}{\delta_{L,j}},
\]

\[
\widetilde N_j(\tau,u)
=
h_{00}(s)c_L(\tau)
+
h_{01}(s)y_a
+
h_{11}(s)\delta_{L,j}d_a.
\tag{10.3}
\]

当 \(u<a_j-\delta_{L,j}\)：

\[
\widetilde N_j(\tau,u)=c_L(\tau).
\tag{10.4}
\]

## 10.3 右延拓

令：

\[
y_b=N_j(\tau,b_j),
\qquad
d_b=\partial_uN_j(\tau,b_j),
\]

\[
c_R(\tau)=y_b+\delta_{R,j}d_b.
\tag{10.5}
\]

当 \(u\in[b_j,b_j+\delta_{R,j}]\)：

\[
s=\frac{u-b_j}{\delta_{R,j}},
\]

\[
\widetilde N_j(\tau,u)
=
h_{00}(s)y_b
+
h_{10}(s)\delta_{R,j}d_b
+
h_{01}(s)c_R(\tau).
\tag{10.6}
\]

当 \(u>b_j+\delta_{R,j}\)：

\[
\widetilde N_j(\tau,u)=c_R(\tau).
\tag{10.7}
\]

## 10.4 命题 10.1：总 Urysohn 面全局 \(C^1\)

若核心 \(N_j(\tau,\cdot)\in C^1([a_j,b_j])\)，则：

\[
\widetilde K_j(\tau,u)
=
\beta_j(\tau)\xi_j(u)
+
\widetilde N_j(\tau,u)
\tag{10.8}
\]

关于 \(u\) 属于 \(C^1(\mathbb R)\)。

### 证明

Hermite 延拓在内边界匹配 \(N_j\) 的函数值和一阶导数，在外边界匹配常值饱和段的函数值和零导数。线性部分全局 \(C^\infty\)。二者之和全局 \(C^1\)。证毕。

## 10.5 OOD 合同

1. \([a_j,b_j]\) 仅由训练棒冻结；
2. 延拓带宽仅由训练支持和物理范围冻结；
3. fit/common-support、extension-band、saturated 三个区域分别报告；
4. 饱和区不进入物理 K 认证；
5. 不允许测试后调带宽；
6. PLC LUT 必须通过函数值和一阶导连续测试；
7. 若 OOD 占比过高，结论登记为 `AMPLITUDE_SUPPORT_INSUFFICIENT`。

---

# 11. 冻结 K 后的 Residual Predictive-State/AR

## 11.1 OOF 物理残差

内层 rolling cross-fit 得到：

\[
r_s^{\mathrm{OOF}}
=
z_s
-
\widehat K^{\mathrm{OOF}}(U_s^-).
\tag{11.1}
\]

禁止用同一样本上的 in-sample K 残差训练 AR。

## 11.2 成熟残差集合

预测时刻 \(t\)：

\[
R_{t,\mathrm{mature}}^-
=
\left\{
r_s^{\mathrm{OOF}}:
s+h+W\le t
\right\}.
\tag{11.2}
\]

## 11.3 Canonical residual AR

\[
\widehat r_t
=
c
+
\sum_{k=1}^{P}
\phi_k
r_{s_k}^{\mathrm{OOF}}.
\tag{11.3}
\]

候选必须包含：

\[
\widehat r_t\equiv0.
\tag{11.4}
\]

## 11.4 词典序

1. 先冻结 full Urysohn K；
2. 再生成严格 OOF 成熟残差；
3. 再选择 residual PS/AR；
4. AR 不反向改变 K；
5. AR 不解释某个失败物理通道；
6. 若 AR 无稳定增益，则精确为零。

---

# 12. 数值求解

## 12.1 CPU FP64 参考

所有正式核、投影、GCV、谱、bootstrap 和最终指标以 CPU FP64 为准。

## 12.2 线性系统

在固定 basis 和 \(\lambda\) 下：

\[
(\Phi^\top\Phi+n\lambda P)\theta
=
\Phi^\top z.
\tag{12.1}
\]

可以使用：

- Cholesky；
- QR；
- symmetric eigensolver；
- matrix-free PCG。

必须输出：

- KKT residual；
- condition number；
- effective degrees of freedom；
- GCV 曲线；
- coefficient hash；
- prediction hash。

## 12.3 GCV

平滑矩阵：

\[
S_\lambda
=
\Phi
(\Phi^\top\Phi+n\lambda P)^{-1}
\Phi^\top.
\tag{12.2}
\]

\[
\operatorname{GCV}(\lambda)
=
\frac{
\|z-S_\lambda z\|_2^2/n
}{
\left(
1-\operatorname{tr}(S_\lambda)/n
\right)^2
}.
\tag{12.3}
\]

使用一维 deterministic bounded search 求 \(\widehat\lambda\)，不使用测试集和随机种子。

## 12.4 CUDA 的位置

CUDA 只用于：

- 大批量设计矩阵前向；
- tensor-product basis evaluation；
- bootstrap；
- surface evaluation；
- 大样本 matrix-free operator。

CUDA 不改变：

- Urysohn 面；
- 正交幅值分解；
- NLinear 的派生地位；
- GCV；
- 外层验证。

---

# 13. PLC / MCU 编译

## 13.1 线性投影

\[
\mathcal K_j^{\mathrm{LIN}}[u_j](t)
=
\sum_{\ell}
\beta_j(\ell)\xi_j(u_{j,t-\ell})\Delta t.
\tag{13.1}
\]

可直接编译为 FIR 或稳定 IIR 近似。

## 13.2 非线性曲面

完整在线形式：

\[
\mathcal N_j[u_j](t)
=
\sum_{\ell}
N_j(\ell,u_{j,t-\ell})\Delta t.
\tag{13.2}
\]

实现方式：

1. lag × amplitude LUT；
2. spline basis LUT；
3. 非线性谱压缩：
   \[
   N_j(\tau,u)
   \approx
   \sum_{r=1}^{R_{j,\mathrm{deploy}}}
   \sigma_{j,r}q_{j,r}(\tau)f_{j,r}(u).
   \]

这里的 Rank 仅由部署误差预算决定。

## 13.3 在线状态

部署状态包括：

- 各输入历史环形缓冲；
- 线性 FIR/IIR 状态；
- 非线性 LUT/spline 评价；
- 可选低秩非线性卷积状态；
- matured residual AR 状态；
- OOD 标志。

## 13.4 Bit-accurate

必须测试：

- FP64 reference；
- FP32；
- PLC float；
- fixed-point；
- \(C^1\) 边界；
- saturated OOD；
- FIR/IIR 截断误差；
- worst-case accumulator。

---

# 14. 当前 CZ 实验证据

## 14.1 已完成结果

当前冻结结果包括：

| 模型 | pooled RMSE | 解释 |
|---|---:|---|
| Persistence | 0.513061 | 零输入变化基线 |
| 旧单通道 K-only | 0.412383 | 方向不稳定 |
| Dynamic-PLS | 0.399628 | 传统输入模型 |
| Temporal Autoencoder | 0.384272 | 最佳稳定 GPU 深度动态模型 |
| NLinear-U，逐样本种子中位集成 | 0.365854 | 强低秩线性预测证据 |
| Joint-K+AR | 0.351642 | 当前总体预测冠军 |
| Shared–Private K confirm | 0.513061 | `LEVEL_D_REJECTED` |

## 14.2 Shared–Private 否定结果的含义

本轮确认实验否定的是：

\[
\text{full K}
\rightarrow
\text{post-hoc shared SVD}
\rightarrow
\text{private orthogonal rescue}.
\]

它不否定：

- 完整 Urysohn 面；
- full 面内部的线性投影；
- NLinear 的预测有效性；
- 纯非线性面可能存在；
- Residual PS/AR 的独立预测作用。

## 14.3 当前允许声明

1. NLinear 在冻结任务上具有很强预测性能；
2. 其结构应作为 full Urysohn 线性投影的 Rank-1 假设来审计；
3. 不能把 GPU NLinear 的成功直接解释为共享物理时滞核已认证；
4. shared/private 字典未通过；
5. 当前需要直接测试 full Urysohn 是否能在保留二维面的同时达到或超过 NLinear；
6. 即使 full Urysohn 预测更好，通道物理面仍需独立跨棒证书。

---

# 15. v3.2 的实验合同

## 15.1 Stage U0：协议冻结

冻结：

- L6；
- 两 outer directions；
- sample IDs；
- target；
- PCA；
- scaler；
- history；
- horizon；
- output window；
- no-future-input；
- 断点规则。

## 15.2 Stage U1：固定 spline 坐标

冻结：

- 一套时滞 basis；
- 每通道一套训练分位数 amplitude basis；
- 一套归一化 Sobolev 几何；
- 一个由 GCV 自动选择的全局平滑尺度。

不搜索：

- V0/V1；
- shared/private；
- Rank；
- 通道权重；
- 两个 penalty；
- 测试集最优 basis。

## 15.3 Stage U2：完整 Urysohn 面

在每个 outer training rod 内拟合：

\[
\widehat K_j^{\mathrm{FULL}}.
\]

输出：

- GCV \(\lambda\)；
- KKT；
- condition number；
- full predictions；
- surfaces；
- OOD。

## 15.4 Stage U3：正交幅值分解

由同一个 \(\widehat K_j\) 派生：

\[
\widehat\beta_j,\qquad
\widehat N_j.
\]

不重新训练。

## 15.5 Stage U4：NLinear 结构审计

计算：

- \(\widehat B\)；
- singular values；
- \(\rho_{\mathrm{NLinear}}\)；
- 最佳 Rank-1 派生投影；
- 两方向 rank-1 time shape correlation；
- channel loading sign consistency；
- Rank-1 projection RMSE。

## 15.6 Stage U5：full vs linear vs nonlinear

比较：

- derived Rank-1 linear projection；
- full linear projection；
- full Urysohn；
- full Urysohn + matured residual AR。

非线性是否有贡献必须由逐样本 paired bootstrap 判断，不能由核能量单独决定。

## 15.7 Stage U6：物理证书

分别评估：

- total input operator；
- 每个 \(\beta_j\)；
- 每个 \(N_j\)；
- linear rank-1 subspace；
- full surface common-support；
- placebo；
- OOD。

## 15.8 Stage U7：部署压缩

只有科学结构冻结后，才对 \(N_j\) 做谱压缩并测试 PLC 误差。

---

# 16. 声明矩阵

| 证据 | 允许声明 | 禁止声明 |
|---|---|---|
| full Urysohn RMSE 优于 NLinear | full 面提供额外预测信息 | 非线性物理机制已证明 |
| full 与 linear 无显著差异 | 当前数据下线性投影足够 | 真实过程不存在非线性 |
| \(\rho_{\mathrm{NLinear}}\) 高 | 线性投影近似 Rank 1 | 多通道共享 plant kernel 已证明 |
| \(\rho_{\mathrm{NLinear}}\) 低 | full 面线性投影非 Rank 1 | NLinear GPU 结果错误 |
| \(N_j\) 跨棒稳定且有增益 | 该通道存在可复现非线性观测响应 | 开放环物理因果已证明 |
| residual AR 精确零 | K 后无稳定残差增益 | 系统没有内部状态 |
| residual AR 有增益 | 成熟残差包含额外预测状态 | AR 解释某个控制变量 |
| C1 延拓通过 | 轻度幅值 OOD 平滑可部署 | 远离训练域仍可信 |

---

# 17. 关键性质汇总

## 17.1 完整面不被 NLinear 替代

正式模型始终是：

\[
K_j(\tau,u).
\]

NLinear 只是：

\[
N_j=0,\quad\operatorname{rank}(B)\le1
\]

这一派生子类。

## 17.2 分解唯一

给定训练测度 \(\nu_j\) 和规范，常数—线性—非线性分解是 Hilbert 正交投影，唯一。

## 17.3 无外挂训练权重

\(q\) 和 \(a\) 仅是 \(\widehat B\) 的 Rank-1 坐标，不进入 full Urysohn 训练目标。

## 17.4 无 shared/private 超参数

不存在：

\[
R_s,\quad R_{p,j},\quad\lambda_s,\quad\lambda_p.
\]

## 17.5 单一自动估计尺度

只有一个 GCV 自动确定的 \(\lambda\)，用于定义病态逆问题的平滑解；它不是物理结构参数。

## 17.6 非线性谱不复制线性项

纯非线性幅值函数与 \(1,\xi_j\) 正交。

## 17.7 Rank 仅用于派生诊断和部署

不再用 Rank 救援预测失败。

## 17.8 Physics-first 冻结

Residual AR 不反向改写 K。

## 17.9 可保证与不可保证

可保证：

- 固定坐标下正则解唯一；
- 幅值投影分解唯一；
- Rank-1 最佳近似在核空间范数下最优；
- C1 延拓连续；
- 成熟残差因果；
- 部署压缩误差可量化。

不可保证：

- 被动闭环数据恢复开放环 plant；
- 两根晶棒足以证明普适物理规律；
- 高 Rank-1 能量即因果共享机制；
- 低 RMSE 即通道归因正确；
- 训练支持外远距离外推可信。

---

# 18. 最终语义链

\[
\boxed{
\begin{aligned}
&\text{冻结闭环预测任务}
\\
&\Downarrow
\\
&\text{直接拟合每个输入通道的完整 Urysohn 面 }K_j(\tau,u)
\\
&\Downarrow
\\
&\text{在幅值 Hilbert 空间中唯一投影为}
\\
&K_j=\beta_j\xi_j+N_j
\\
&\Downarrow
\\
&\text{对 }B=[\beta_1,\ldots,\beta_p]\text{ 做派生 Rank-1 审计}
\\
&\Downarrow
\\
&\text{比较 Rank-1 linear、general linear 和 full Urysohn}
\\
&\Downarrow
\\
&\text{冻结 full }K
\\
&\Downarrow
\\
&\text{只对成熟 OOF 残差建立可精确为零的 PS/AR}
\\
&\Downarrow
\\
&\text{最后做非线性谱压缩、}C^1\text{ 延拓和 PLC 编译}.
\end{aligned}
}
\]

---

# 附录 A：v3.1 → v3.2 迁移表

| v3.1 | v3.2 |
|---|---|
| 公共字典 | 删除 |
| 私有字典 | 删除 |
| shared/private Rank | 删除 |
| participation gate | 删除 |
| post-hoc full-K shared SVD | 降级为否定历史 |
| shared/private exact-zero | 删除该结构 |
| 完整 Urysohn 面 | 恢复为正式核心 |
| NLinear | full 面线性投影的 Rank-1 派生性质 |
| 两个 penalty | 单一 GCV 自动 Sobolev 尺度 |
| Rank 选择 | 只用于派生诊断和部署压缩 |
| C1 延拓 | 保留，只作用于纯非线性曲面 |
| Residual PS/AR | 保留，严格成熟和冻结 K |

---

# 附录 B：最小实现对象

```python
@dataclass(frozen=True)
class UrysohnSurfaceArtifact:
    horizon_minutes: float
    output_window_minutes: float
    lag_support_minutes: float
    cadence_seconds: float

    input_names: tuple[str, ...]
    input_mean: tuple[float, ...]
    input_scale: tuple[float, ...]

    lag_basis_spec: dict
    amplitude_basis_specs: tuple[dict, ...]
    coefficients_fp64: tuple[list[list[float]], ...]

    gcv_lambda: float
    kkt_residual: float
    condition_number: float
    effective_df: float

    linear_projection_coefficients: tuple[list[float], ...]
    nonlinear_projection_coefficients: tuple[list[list[float]], ...]

    nlinear_singular_values: list[float]
    nlinear_rank1_energy_ratio: float
    nlinear_rank1_time_shape: list[float]
    nlinear_rank1_channel_coordinates: list[float]

    c1_extension_specs: tuple[dict, ...]
    residual_model: dict

    protocol_sha256: str
    sample_id_sha256: str
    coefficient_sha256: str
    prediction_sha256: str
```

---

# 附录 C：当前冻结名称

正式模型名称：

\[
\boxed{
\text{OPS-UOI Orthogonally Decomposed Full Urysohn Surface}
\rightarrow
\text{Matured Residual PS/AR}
}
\]

简称：

```text
OD-FUOI-PSAR
```

其中：

- `OD`：Orthogonally Decomposed；
- `F`：Full；
- `UOI`：Urysohn Operator Identification；
- `PSAR`：Predictive-State / Autoregressive residual。

NLinear 不出现在正式模型名中，因为它不是额外模型层，而是 full Urysohn 面的派生线性 Rank-1 审计。
