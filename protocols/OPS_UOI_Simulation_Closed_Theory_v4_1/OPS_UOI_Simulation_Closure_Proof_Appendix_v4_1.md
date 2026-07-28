# OPS-UOI v4.1 Simulation Closure 详细证明附录

> 本附录只证明 v4.1 新增的 simulation closure、域内不变性、递推稳定性和域外非识别性。  
> v4.0 的 operator-first、正交 score、sieve 主不等式和 rank 证明保持原文。

---

# A. 符号

训练支撑为：

\[
I=[a,b],\qquad a<b.
\]

域内响应：

\[
f\in C^1(I).
\]

左右 continuation 尺度：

\[
\rho_->0,\qquad \rho_+>0.
\]

定义：

\[
\widetilde f(u)
=
\begin{cases}
f(a)+\rho_-f'(a)\tanh((u-a)/\rho_-),&u<a,\\
f(u),&a\le u\le b,\\
f(b)+\rho_+f'(b)\tanh((u-b)/\rho_+),&u>b.
\end{cases}
\]

---

# B. \(C^1\) 拼接

## 引理 B.1：函数值连续

当 \(u\uparrow a\) 时：

\[
\tanh((u-a)/\rho_-)\to0,
\]

故：

\[
\lim_{u\uparrow a}\widetilde f(u)=f(a).
\]

域内右极限也是 \(f(a)\)。所以在 \(a\) 连续。右边界同理。

## 引理 B.2：导数连续

左 continuation 导数为：

\[
\widetilde f'(u)
=
f'(a)\operatorname{sech}^2((u-a)/\rho_-).
\]

因此：

\[
\lim_{u\uparrow a}\widetilde f'(u)=f'(a).
\]

域内右导数由 \(f\in C^1\) 也为 \(f'(a)\)。右边界同理。

## 定理 B.3

由区间内部 \(C^1\)、两端函数值和导数拼接，得：

\[
\widetilde f\in C^1(\mathbb R).
\]

---

# C. 全局有界性

因为：

\[
|\tanh z|\le1,
\qquad
0<\operatorname{sech}^2z\le1,
\]

在左域：

\[
|\widetilde f(u)|
\le
|f(a)|+\rho_-|f'(a)|.
\]

在右域：

\[
|\widetilde f(u)|
\le
|f(b)|+\rho_+|f'(b)|.
\]

在紧区间 \([a,b]\) 上，连续函数有界。因此：

\[
\|\widetilde f\|_\infty<\infty.
\]

同理：

\[
\|\widetilde f'\|_\infty
\le
\max\{
\|f'\|_{\infty,[a,b]},
|f'(a)|,
|f'(b)|
\}.
\]

---

# D. free-run 的存在唯一性

考虑：

\[
\widehat y_{t+1}
=
b+g_t^x+
\sum_{\ell=1}^{L_y}
q_\ell\widetilde f(\widehat y_{t+1-\ell}),
\]

其中 \(g_t^x\) 对给定外生序列是已知有限实数。

给定初始历史：

\[
\widehat y_{t_0-L_y+1},\ldots,\widehat y_{t_0},
\]

右端只依赖已知历史，因此唯一确定 \(\widehat y_{t_0+1}\)。归纳即可唯一确定任意有限时刻。

该结论依赖模型为显式递推，不需要 Banach 不动点定理。

---

# E. 全局 simulation 界

若：

\[
|g_t^x|
\le
M_x,
\qquad
|\widetilde f(u)|\le M_y,
\]

则：

\[
|\widehat y_{t+1}|
\le
|b|+M_x+M_y\sum_{\ell=1}^{L_y}|q_\ell|.
\]

对 full Urysohn 外生分支：

\[
M_x
\le
\sum_{j,\tau}
\|\widetilde K_j(\tau,\cdot)\|_\infty.
\]

所以可取：

\[
B_{\mathrm{sim}}
=
|b|
+
\sum_{j,\tau}
\|\widetilde K_j(\tau,\cdot)\|_\infty
+
\|\widetilde f^y\|_\infty
\sum_\ell|q_\ell^y|.
\]

从第一步递推开始，所有新输出均落入 \([-B_{\mathrm{sim}},B_{\mathrm{sim}}]\)。联合有限初始 history 得统一界。

---

# F. 增量稳定

取相同外生输入的两条轨迹，令：

\[
\delta_t=\widehat y_t-\widehat y_t'.
\]

由均值定理：

\[
|\delta_{t+1}|
\le
\sum_{\ell=1}^{L_y}
|q_\ell|L_f
|\delta_{t+1-\ell}|,
\]

其中：

\[
L_f=\|\widetilde f'\|_\infty.
\]

定义非负状态：

\[
d_t=
(
|\delta_t|,
|\delta_{t-1}|,
\ldots,
|\delta_{t-L_y+1}|
)^\top.
\]

则逐分量有：

\[
d_{t+1}\le A_{\mathrm{AR}}d_t.
\]

若：

\[
\rho(A_{\mathrm{AR}})<1,
\]

则有限维线性系统稳定，存在矩阵范数和常数 \(C,\varrho<1\) 使：

\[
\|A_{\mathrm{AR}}^k\|
\le C\varrho^k.
\]

故：

\[
\|d_t\|
\le
C\varrho^{t-t_0}\|d_{t_0}\|.
\]

这证明全局初值遗忘。

---

# G. 外生扰动 ISS 界

若外生贡献不同，记：

\[
\eta_t
=
g_t^x-g_t^{x\prime}.
\]

则：

\[
d_{t+1}
\le
A_{\mathrm{AR}}d_t
+
e_1|\eta_t|,
\]

其中 \(e_1=(1,0,\ldots,0)^\top\)。

迭代：

\[
d_t
\le
A_{\mathrm{AR}}^{t-t_0}d_{t_0}
+
\sum_{s=t_0}^{t-1}
A_{\mathrm{AR}}^{t-1-s}e_1|\eta_s|.
\]

使用稳定矩阵幂界即得：

\[
\|d_t\|
\le
C\varrho^{t-t_0}\|d_{t_0}\|
+
C\sum_{s=t_0}^{t-1}
\varrho^{t-1-s}|\eta_s|.
\]

---

# H. 域内训练不变性

训练样本满足：

\[
X_t\in[a,b].
\]

由 continuation 定义：

\[
\widetilde f(X_t)=f(X_t).
\]

所以每一个设计矩阵元素、预测值和平方损失完全相同。若正则项只在参考域 \([a,b]\) 上积分或由相同 basis 系数定义，则 objective 也相同。

因此 continuation 不改变：

- 训练最优解集合；
- fitted contribution；
- Gram；
- v4.0 的 Q/K 识别对象。

数值实现中应通过 machine-equivalence test 检验该结论。

---

# I. 结构谱不变性

结构核矩阵由训练参考域 quadrature 构造：

\[
T=W_\tau^{1/2}KW_x^{1/2}.
\]

只要 \(W_x\) 的 quadrature nodes 全在 \([a,b]\)，continuation 不改变任何 node 上的 \(K\)，故 \(T\) 不变。于是：

\[
\sigma_r(T),\quad
R_S^\star,\quad
d_R^{HS}
\]

全部不变。

---

# J. 训练支撑外非识别性

设训练输入分布支撑在 \(S\)。取任意非零函数 \(h\)，满足：

\[
h(u)=0,\quad u\in S,
\]

但在 \(S^c\) 上非零。令：

\[
f_2=f_1+h.
\]

则：

\[
f_2(X_t)=f_1(X_t)
\quad\text{a.s.}
\]

所以任意基于训练样本的 likelihood、经验风险和条件矩均相同。训练数据无法区分 \(f_1,f_2\) 在支撑外的值。

因此任何域外 continuation 都来自模型类选择或先验，而非数据识别。

---

# K. 有界不等于准确

全局有界性只给出：

\[
|\widehat y_t|\le B_{\mathrm{sim}}.
\]

它不保证：

\[
|\widehat y_t-y_t|
\]

小，也不保证 continuation 进入真实系统支撑外时仍合理。

所以实验必须同时报告：

- boundedness；
- free-run RMSE；
- continuation usage；
- 最大域外距离；
- 域内/域外误差；
- 初值敏感性。

---

# L. 结论

v4.1 新增理论完成了四个严格分离：

\[
\boxed{\text{可计算性}}
\neq
\boxed{\text{递推稳定性}}
\neq
\boxed{\text{预测准确性}}
\neq
\boxed{\text{结构可解释性}}.
\]

这四者分别由：

1. global continuation；
2. companion spectral certificate；
3. validation/test simulation；
4. support + Gram/coercivity；

进行认证。
