# 自回归秩自适应并联 Hammerstein–Urysohn 方法（AR-RAPHU）

> **版本**：v2.0  
> **状态**：方法定义与实验前冻结稿  
> **当前应用对象**：CZ 法单晶硅直拉过程中，单根晶棒等径阶段的晶体直径动态预测  
> **当前结论边界**：现有 CZ 数据只能支撑单轨迹、同阶段内的时间外推案例；尚不能证明跨晶棒、跨炉次或跨阶段泛化。  
> **方法主线**：方案 A 负责稀疏支持、主响应和主时滞发现；方案 B 负责固定主结构后的显式凸重建、可分离性审计和选择性低秩升级。

---

# 0. 当前项目事实、待确认事项与证据等级

## 0.1 已确认事实

当前已确认：

1. 数据来自 **一根 CZ 法硅单晶晶棒**；
2. 数据只覆盖 **等径阶段**；
3. 原始表格最后一列为输出，即晶体直径 \(y_t\)；
4. 原始表中除输出外有 9 个过程变量；
5. 历史模型把过去的晶体直径也作为一个输入通道，因此模型接口共有 10 个输入通道：
   \[
   9\text{ 个过程变量}+1\text{ 个历史输出通道};
   \]
6. 当前历史窗口长度记录为 \(L=32\)；
7. 当前数据只有一条连续轨迹，没有可用的晶棒 ID 或独立重复轨迹；
8. 氩气流量设定在当前晶棒中为常数，因此该分支在当前数据内不可辨识。

## 0.2 尚未确认的元数据

以下信息当前 **尚未取得**，不得在论文、报告或图表中写成已知事实：

- 各列准确工程单位；
- 物理采样周期；
- 晶体直径的具体测量方法；
- 直径信号是否经过设备端滤波、移动平均、估计或延迟补偿；
- 各传感器或执行器的安装位置；
- 各过程量是设定值、控制输出还是反馈测量值；
- 当前等径数据是否经过人工截取、去异常或重采样；
- 是否能够获得更多晶棒及其炉次 ID。

这些项目在后续文档中统一标记为：

```text
STATUS = NOT_YET_AVAILABLE
```

## 0.3 当前命名原则

在直径测量机制和延迟未确认前：

- 可以称为 **晶体直径动态预测**；
- 可以称为 **带历史输出的动态数据驱动模型**；
- 暂不把全部任务统称为“纯软测量”；
- 模型恢复的时滞应称为 **有效预测时滞**；
- 不把有效预测时滞直接解释成纯粹的热传播或物质输运时间。

---

# 1. 数据结构与预测任务

## 1.1 外生过程变量

设离散时刻为 \(t\)，九个过程变量构成

\[
\boldsymbol x_t
=
(x_{1,t},x_{2,t},\ldots,x_{9,t})^\top
\in\mathbb R^9.
\tag{1.1}
\]

其中 \(x_{j,t}\) 表示第 \(j\) 个过程变量在时刻 \(t\) 的观测值。

变量的准确单位和控制属性当前尚未确认，因此在正式元数据到达前，符号只表示数据列，不附加未经确认的物理量纲。

## 1.2 输出

令

\[
y_t\in\mathbb R
\tag{1.2}
\]

表示时刻 \(t\) 的晶体直径测量值。

需要区分：

- 真实但不可直接知道的几何直径 \(z_t\)；
- 数据表中的测量或估计直径 \(y_t\)。

在测量机制未知时，我们只对 \(y_t\) 建模，不假定

\[
y_t=z_t.
\]

更一般地，设备可能满足

\[
y_t=\mathcal H_m(z_{t-d_m:t})+\nu_t,
\tag{1.3}
\]

其中：

- \(\mathcal H_m\)：未知测量、滤波或估计算子；
- \(d_m\)：可能存在的仪表延迟；
- \(\nu_t\)：测量噪声。

式 (1.3) 当前只是风险模型，不代表已经确认存在某种滤波或延迟。

## 1.3 历史输出通道

第十个输入通道不是新的传感器变量，而是过去的输出：

\[
y_{t-1},y_{t-2},\ldots,y_{t-L_y}.
\tag{1.4}
\]

因此任何训练样本都必须保证当前标签 \(y_t\) 本身不进入输入。

对于一步预测，输入窗口可定义为

\[
\mathcal I_t
=
\left\{
x_{j,t-\tau}:
j=1,\ldots,9,\ 
\tau=0,\ldots,L_x-1
\right\}
\cup
\left\{
y_{t-\ell}:
\ell=1,\ldots,L_y
\right\}.
\tag{1.5}
\]

预测目标为

\[
\widehat y_t
=
F(\mathcal I_t).
\tag{1.6}
\]

如果旧代码把 \(y_t\) 直接放进同一窗口并用于预测 \(y_t\)，则属于标签泄漏，必须在代码审计中排除。

---

# 2. 完整符号表

| 符号 | 类型/范围 | 含义 |
|---|---|---|
| \(t\) | 整数 | 当前时刻 |
| \(h\) | 正整数 | 预测步长 |
| \(p_x\) | \(9\) | 外生过程变量数量 |
| \(L_x\) | 正整数 | 外生变量最大历史长度 |
| \(L_y\) | 正整数 | 输出历史长度 |
| \(\tau\) | \(0,\ldots,L_x-1\) | 外生变量时滞索引 |
| \(\ell\) | \(1,\ldots,L_y\) | 历史输出时滞索引 |
| \(\Delta t\) | 正数或未知 | 采样周期；当前尚未确认 |
| \(x_{j,t}\) | 实数 | 第 \(j\) 个过程变量 |
| \(\boldsymbol x_t\) | \(\mathbb R^9\) | 过程变量向量 |
| \(y_t\) | 实数 | 表格中的晶体直径输出 |
| \(z_t\) | 实数 | 潜在真实几何直径，仅用于讨论测量模型 |
| \(\widehat y_t\) | 实数 | 一步预测 |
| \(\widehat y_{t+h\mid t}\) | 实数 | 在时刻 \(t\) 对 \(t+h\) 的预测 |
| \(b\) | 实数 | 全局偏置 |
| \(f_j^x(u)\) | 一元函数 | 第 \(j\) 个过程变量的静态响应 |
| \(q_j^x(\tau)\) | 非负函数 | 第 \(j\) 个过程变量的主时滞核 |
| \(f^y(v)\) | 一元函数 | 历史直径的非线性自回归响应 |
| \(q^y(\ell)\) | 非负函数 | 历史直径的记忆核 |
| \(K_j^x(\tau,u)\) | 二元函数 | 第 \(j\) 个外生变量的时滞—幅值联合响应 |
| \(K^y(\ell,v)\) | 二元函数 | 历史输出分支的记忆—幅值联合响应 |
| \(\Delta_j^x(\tau,u)\) | 二元函数 | 外生分支相对 rank-1 主效应的残差 |
| \(\mathcal S_x\) | 子集 | 活跃外生变量集合 |
| \(R_j\) | 非负整数 | 第 \(j\) 个外生分支的有效秩 |
| \(\eta_j^x\) | \([0,1]\) | 外生分支非可分离度 |
| \(\rho_j^x\) | \([0,1]\) | 外生二维残差相对能量 |
| \(\mu_j^x(u)\) | 非负数 | 外生分支条件平均时滞 |
| \(\kappa_y\) | 非负数 | 自回归分支收缩系数 |
| \(B_{j,r}(u)\) | 基函数 | 输入幅值方向 B 样条 |
| \(C_a(\tau)\) | 基函数 | 时滞方向 B 样条 |
| \(G_x,G_\tau\) | 半正定矩阵 | 基函数 Gram 矩阵 |

---

# 3. 三类任务必须分开

## 3.1 X-only：只使用过程变量

\[
\widehat y_t^{X}
=
b+
\sum_{j=1}^{9}
\sum_{\tau=0}^{L_x-1}
K_j^x(\tau,x_{j,t-\tau}).
\tag{3.1}
\]

它回答：

> 不利用过去直径，仅依靠过程变量能解释和预测多少直径变化？

如果直径本身可实时测得，这不一定是最佳工程预测器，但它最接近过程变量软测量和工艺机理辨识。

## 3.2 AR-only：只使用历史直径

\[
\widehat y_t^{AR}
=
b_y+
\sum_{\ell=1}^{L_y}
K^y(\ell,y_{t-\ell}).
\tag{3.2}
\]

它回答：

> 只利用直径自身的持续性，能取得多高预测精度？

它是所有复杂模型必须击败的关键基线。

## 3.3 X+AR：十通道完整模型

\[
\widehat y_t^{X+AR}
=
b+
\sum_{j=1}^{9}
\sum_{\tau=0}^{L_x-1}
K_j^x(\tau,x_{j,t-\tau})
+
\sum_{\ell=1}^{L_y}
K^y(\ell,y_{t-\ell}).
\tag{3.3}
\]

它回答：

> 在已有历史直径的条件下，过程变量还能提供多少额外前瞻信息？

必须报告过程变量的边际增益：

\[
\Delta_{\mathrm{X\mid AR}}
=
\operatorname{Loss}(\mathrm{AR\text{-}only})
-
\operatorname{Loss}(\mathrm{X+AR}).
\tag{3.4}
\]

若 \(\Delta_{\mathrm{X\mid AR}}\) 很小，即使 X+AR 预测非常准确，也不能声称过程变量模型学到了强工艺机制。

---

# 4. 方案 A：自回归并联 Hammerstein 主模型

## 4.1 外生过程分支

对第 \(j\) 个过程变量，

\[
v_{j,t}^x=f_j^x(x_{j,t}),
\tag{4.1}
\]

\[
c_{j,t}^x
=
\sum_{\tau=0}^{L_x-1}
q_j^x(\tau)v_{j,t-\tau}^x.
\tag{4.2}
\]

其中：

- \(f_j^x\)：一元 KAN 或显式样条响应；
- \(q_j^x\)：结构化分布时滞核；
- \(c_{j,t}^x\)：该过程变量在时刻 \(t\) 的预测贡献。

## 4.2 历史输出分支

对直径自身历史，

\[
v_{t-\ell}^y=f^y(y_{t-\ell}),
\tag{4.3}
\]

\[
c_t^y
=
\sum_{\ell=1}^{L_y}
q^y(\ell)f^y(y_{t-\ell}).
\tag{4.4}
\]

这里 \(c_t^y\) 不应被解释成独立的工艺传递通道。它主要代表：

- 直径自身惯性；
- 未观测状态的预测代理；
- 闭环系统记忆；
- 可能存在的测量平滑或延迟；
- 其他未显式建模的慢动态。

## 4.3 完整 rank-1 主模型

\[
\boxed{
\widehat y_t
=
b+
\sum_{j=1}^{9}
\sum_{\tau=0}^{L_x-1}
q_j^x(\tau)
f_j^x(x_{j,t-\tau})
+
\sum_{\ell=1}^{L_y}
q^y(\ell)
f^y(y_{t-\ell})
}
\tag{4.5}
\]

式 (4.5) 是 v2 的方案 A。

它属于：

- 带自回归输出分支的并联 Hammerstein 模型；
- Parallel Hammerstein–NARX 型模型；
- AR-RAPHU 的 rank-1 主模型。

---

# 5. 时滞核

## 5.1 外生 Gamma 核

对外生分支可使用有限窗口内归一化 Gamma 形状：

\[
\widetilde q_j^x(\tau)
=
(\tau+\delta_x)^{\alpha_j^x-1}
\exp[-\beta_j^x(\tau+\delta_x)],
\tag{5.1}
\]

\[
q_j^x(\tau)
=
\frac{\widetilde q_j^x(\tau)}
{\sum_{s=0}^{L_x-1}\widetilde q_j^x(s)}.
\tag{5.2}
\]

因此

\[
q_j^x(\tau)\ge0,
\qquad
\sum_{\tau=0}^{L_x-1}q_j^x(\tau)=1.
\tag{5.3}
\]

当前代码究竟使用 shape–rate、shape–scale、点值离散还是区间质量离散，必须由代码审计确认。本文公式是方法定义，不代替实现清单。

## 5.2 自回归核

自回归分支使用

\[
q^y(\ell)\ge0,
\qquad
\sum_{\ell=1}^{L_y}q^y(\ell)=1.
\tag{5.4}
\]

第一版建议：

- 使用低自由度核；
- 不使用完整二维 B 残差；
- 对较近历史赋予更强先验；
- 增加收缩或导数约束，防止多步递归不稳定。

## 5.3 有效时滞解释

离散平均时滞为

\[
\bar\tau_j^x
=
\sum_{\tau=0}^{L_x-1}\tau q_j^x(\tau).
\tag{5.5}
\]

只有在采样周期 \(\Delta t\) 确认后，才可转换为物理时间：

\[
\bar d_j^x=\Delta t\,\bar\tau_j^x.
\tag{5.6}
\]

当前 \(\Delta t\) 尚未由数据来源确认，因此正式结果首先以“采样步”报告。

---

# 6. 方案 B：只优先升级外生过程分支

## 6.1 一般联合响应面

对外生变量，

\[
c_{j,t}^x
=
\sum_{\tau=0}^{L_x-1}
K_j^x(\tau,x_{j,t-\tau}).
\tag{6.1}
\]

标准 Hammerstein 假设为

\[
K_j^x(\tau,u)
=
q_j^x(\tau)f_j^x(u).
\tag{6.2}
\]

方案 B 放宽为

\[
K_j^x(\tau,u)
=
q_j^x(\tau)f_j^x(u)
+
\Delta_j^x(\tau,u).
\tag{6.3}
\]

## 6.2 为什么第一版不升级历史输出分支

第一版固定

\[
K^y(\ell,v)
=
q^y(\ell)f^y(v)
\tag{6.4}
\]

而不增加 \(\Delta^y\)，原因是：

1. 历史直径持续性极强，复杂 B 分支可能掩盖全部过程变量；
2. 研究的科学对象是过程变量动态，而不是历史输出自身的复杂记忆；
3. 复杂自回归面会增加多步预测中的递归不稳定；
4. 输出测量机制尚未确认，过早解释其二维结构风险很大。

## 6.3 低秩推广

对必要的外生变量，

\[
K_j^x(\tau,u)
\approx
\sum_{m=1}^{R_j}
q_{j,m}^x(\tau)f_{j,m}^x(u).
\tag{6.5}
\]

其中：

- \(R_j=0\)：变量无效；
- \(R_j=1\)：标准 Hammerstein；
- \(R_j=2\)：双动态机制；
- \(R_j>2\)：更复杂联合响应。

---

# 7. 可辨识约束

## 7.1 主核归一化

\[
\sum_\tau q_j^x(\tau)=1.
\tag{7.1}
\]

它消除 \(q_j^x\) 与 \(f_j^x\) 的任意尺度交换。

## 7.2 响应中心化

\[
\frac{1}{T_{\mathrm{tr}}}
\sum_{t\in\mathcal T_{\mathrm{tr}}}
f_j^x(x_{j,t})=0.
\tag{7.2}
\]

自回归响应也可中心化：

\[
\frac{1}{T_{\mathrm{tr}}}
\sum_{t\in\mathcal T_{\mathrm{tr}}}
f^y(y_{t-1})=0.
\tag{7.3}
\]

## 7.3 A 主效应与 B 残差正交

对所有输入值 \(u\)，

\[
\langle
q_j^x,
\Delta_j^x(\cdot,u)
\rangle_{W_\tau}
=0.
\tag{7.4}
\]

该约束保证二维残差不能重复表示主时滞方向上的一维响应。

---

# 8. 固定 A 后的凸重建

令

\[
f_j^x(u)
=
\sum_{r=1}^{K_x}
c_{j,r}B_{j,r}(u).
\tag{8.1}
\]

二维残差写成

\[
\Delta_j^x(\tau,u)
=
\sum_{a=1}^{K_\tau}
\sum_{r=1}^{K_x}
D_{j,a,r}
C_a(\tau)B_{j,r}(u).
\tag{8.2}
\]

固定方案 A 得到的：

\[
\widehat{\mathcal S}_x,
\qquad
\widehat q_j^x,
\qquad
\widehat q^y,
\]

后，求解

\[
\begin{aligned}
\min\quad&
\frac{1}{2T_{\mathrm{tr}}}
\sum_t
(y_t-\widehat y_t)^2
+
\lambda_f\sum_j\mathcal R_f(c_j)
+
\lambda_\Delta\sum_j\mathcal R_\Delta(D_j)
\\
\text{s.t.}\quad&
(\widehat q_j^x)^\top W_\tau C D_j=0.
\end{aligned}
\tag{8.3}
\]

在固定支持、核和样条结点后，式 (8.3) 对样条系数是凸问题。

它承担：

- KAN 响应的显式重建；
- 二维残差审计；
- 可检查的内层最优性；
- 为 rank 判断提供响应面。

---

# 9. 外生分支 rank 审计

定义基函数 Gram 矩阵

\[
[G_\tau]_{a,b}
=
\sum_\tau w_\tau C_a(\tau)C_b(\tau),
\tag{9.1}
\]

\[
[G_{x,j}]_{r,s}
=
\int B_{j,r}(u)B_{j,s}(u)\,d\widehat P_j(u).
\tag{9.2}
\]

对白化系数矩阵

\[
\widetilde\Theta_j
=
G_\tau^{1/2}\Theta_jG_{x,j}^{1/2}
\tag{9.3}
\]

做奇异值分解：

\[
\widetilde\Theta_j
=
U_j
\operatorname{diag}
(\sigma_{j,1},\sigma_{j,2},\ldots)
V_j^\top.
\tag{9.4}
\]

外生变量的 rank-1 解释率为

\[
E_{j,1}^x
=
\frac{\sigma_{j,1}^2}
{\sum_r\sigma_{j,r}^2},
\tag{9.5}
\]

非可分离度为

\[
\eta_j^x=1-E_{j,1}^x.
\tag{9.6}
\]

第一版不使用 \(\eta^y\) 作为主要科学结论。

---

# 10. 自回归分支的递归稳定性

## 10.1 一步预测

若一步预测使用真实历史输出 \(y_{t-\ell}\)，模型不会把自己的预测递归输入，因此不需要全局收缩条件才能计算。

## 10.2 多步递归预测

多步递归时，后续模型使用预测输出：

\[
\widehat y_{t+h\mid t}
=
F_x(\boldsymbol x)
+
F_y(
\widehat y_{t+h-1\mid t},
\ldots
).
\tag{10.1}
\]

若

\[
|f^y(a)-f^y(b)|
\le
L_f^y|a-b|,
\tag{10.2}
\]

定义

\[
\kappa_y
=
L_f^y
\sum_{\ell=1}^{L_y}|q^y(\ell)|.
\tag{10.3}
\]

当

\[
\kappa_y<1
\tag{10.4}
\]

时，自回归部分在相应范数下具有收缩性质，预测误差不会仅因输出递归而无限放大。

若 \(q^y\) 非负归一化，则

\[
\kappa_y=L_f^y.
\tag{10.5}
\]

因此第一版建议对 \(f^y\) 加入导数上界或 Lipschitz 正则。

---

# 11. 常量通道与十通道接口

当前数据的氩气流量设定为常数。设其对应第 \(j_c\) 个变量：

\[
x_{j_c,t}\equiv c.
\tag{11.1}
\]

则

\[
f_{j_c}^x(x_{j_c,t})=f_{j_c}^x(c)
\tag{11.2}
\]

也是常数，经过归一化卷积后仍然只能与偏置项混合。

所以：

- 数据接口仍可保持 10 通道，以兼容旧代码；
- 科学模型中常量通道必须屏蔽、固定为零或标记为不可辨识；
- 不能报告该变量的时滞核和响应函数为有效发现；
- 如果未来其他晶棒中该设定发生变化，才重新开放该分支。

当前一根棒上的有效可学习通道上限实际为：

\[
8\text{ 个变化过程变量}
+
1\text{ 个历史输出通道}.
\]

---

# 12. 多预测步长

一步预测容易被历史直径持续性主导，因此必须定义

\[
\widehat y_{t+h\mid t},
\qquad
h\in\mathcal H.
\tag{12.1}
\]

在采样周期尚未确认时，\(\mathcal H\) 先按步数设置，例如

\[
\mathcal H=\{1,5,10,30,60\}.
\tag{12.2}
\]

采样周期确认后再换算为物理时间。

需要分别比较：

- persistence；
- AR-only；
- X-only；
- X+AR；
- 方案 A；
- A+B；
- direct multi-horizon；
- recursive multi-step。

过程变量是否真正有前瞻价值，应主要看较大 \(h\) 下的

\[
\Delta_{\mathrm{X\mid AR}}(h).
\tag{12.3}
\]

---

# 13. 完整训练流程

```text
输入：
    单根晶棒连续等径序列
    9 个过程变量
    晶体直径输出
    外生窗口 L_x
    输出历史窗口 L_y
    预测步长 h

1. 在原始时间轴上先划分 train/validation/test。
2. 在切分边界设置足够隔离带，防止窗口共享原始点。
3. 只用训练段拟合标准化参数。
4. 明确构造第十通道：
       channel_10[t, ell] = y[t-ell], ell >= 1
   禁止使用当前标签 y[t]。
5. 先训练三类基线：
       persistence
       AR-only
       X-only
6. 训练方案 A 的 X+AR 主模型。
7. 记录过程变量相对 AR-only 的增量性能。
8. 固定方案 A 的支持与时滞。
9. 对外生分支进行一维凸样条重拟合。
10. 只对外生分支拟合正交 B 残差。
11. 计算 eta_j^x、rho_j^x 和条件时滞。
12. 通过时间滚动、block bootstrap 和多步预测决定是否升级。
13. 第一版不升级输出历史分支。
```

---

# 14. 当前证据状态

| 项目 | 当前状态 |
|---|---|
| 原 V20 rank-1 合成结果 | 已有，但不含当前 AR-CZ 结构 |
| AR-RAPHU 合成真值实验 | **尚未执行** |
| TEP 公共大规模实验 | **尚未执行** |
| TEP 主动幅值扫描 | **尚未执行** |
| Debutanizer 公共软测量实验 | **尚未执行** |
| UCI Gas Turbine 外部回归实验 | **尚未执行** |
| OpenCGS 晶体物理补充 | **尚未执行** |
| 当前单根 CZ 的 AR/X/X+AR 对照 | **尚未执行** |
| 当前单根 CZ 的 A+B rank 审计 | **尚未执行** |
| 多根晶棒跨棒验证 | **尚无数据，未执行** |
| 各列单位、采样周期、测径方式、传感器位置 | **尚未取得** |

任何后续论文摘要都不能把“计划”写成“结果”。

---

# 15. 若获得更多晶棒时的自然扩展

对第 \(r\) 根晶棒，记

\[
\mathcal D_r
=
\{
\boldsymbol x_t^{(r)},
y_t^{(r)}
\}_{t=1}^{T_r}.
\tag{15.1}
\]

可以引入共享主效应和晶棒随机偏差：

\[
K_{j,r}^x(\tau,u)
=
K_{j,0}^x(\tau,u)
+
U_{j,r}^x(\tau,u).
\tag{15.2}
\]

其中：

- \(K_{j,0}^x\)：跨晶棒共享机制；
- \(U_{j,r}^x\)：第 \(r\) 根晶棒的偏差。

数据规模不同对应不同验证等级：

- 1 根：仅单轨迹案例；
- 2–4 根：探索性 leave-one-rod-out；
- 5–9 根：可做分组交叉验证；
- 10 根及以上：可研究层次模型、跨棒不确定性和稳定 rank；
- 更多阶段：再引入阶段条件 \(s_t\)，研究切换 Hammerstein/Urysohn。

---

# 16. 方法的最终定位

推荐名称：

> **Autoregressive Rank-Adaptive Parallel Hammerstein–Urysohn Model**

缩写：

\[
\boxed{\mathrm{AR\text{-}RAPHU}}
\]

中文：

> **自回归秩自适应并联 Hammerstein–Urysohn 模型**

方法贡献应表述为：

1. 将九个过程变量与历史输出分支明确分开；
2. 用结构化 Hammerstein 分支发现过程变量的主响应和有效时滞；
3. 用低复杂度自回归分支汇总输出持续性与未观测状态；
4. 把 Hammerstein 可分离性转化为外生联合响应面的 rank-1 假设；
5. 用 A 锚定凸 B 残差检验并选择性升级外生变量；
6. 通过 AR-only、X-only、X+AR 和多预测步长区分“目标持续性”与“过程前瞻信息”。

不应声称：

- 当前一根晶棒结果能够跨晶棒泛化；
- 当前结果覆盖放肩和收尾阶段；
- 历史输出分支对应独立物理机制；
- 有效预测时滞等同于纯过程时滞；
- 尚未运行的公共实验已经取得提升。

---

# 17. 外部参考资料与地址

[R1] M. Schoukens, R. Pintelon, and Y. Rolain, “Parametric Identification of Parallel Hammerstein Systems.”  
DOI: https://doi.org/10.1109/TIM.2011.2138370  
Publication page: https://research.tue.nl/en/publications/parametric-identification-of-parallel-hammerstein-systems/

[R2] M. Schoukens and K. Tiels, “Identification of Block-Oriented Nonlinear Systems Starting from Linear Approximations: A Survey.”  
https://arxiv.org/abs/1607.01217

[R3] G. H. Golub and V. Pereyra, “The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables Separate.”  
https://doi.org/10.1137/0710036

[R4] Z. Liu et al., “KAN: Kolmogorov–Arnold Networks,” ICLR 2025.  
https://proceedings.iclr.cc/paper_files/paper/2025/hash/afaed89642ea100935e39d39a4da602c-Abstract-Conference.html

[R5] R. S. Risuleo, G. Bottegal, and H. Hjalmarsson, “A Kernel-Based Approach to Hammerstein System Identification.”  
https://doi.org/10.1016/j.ifacol.2015.12.263  
https://arxiv.org/abs/1412.4055

[R6] X. Chen et al., “Identification of MISO Hammerstein System Using Sparse Multiple Kernel-Based Hierarchical Mixture Prior and Variational Bayesian Inference.”  
https://doi.org/10.1016/j.isatra.2023.02.004  
https://pubmed.ncbi.nlm.nih.gov/36801139/

---

# 18. 仍待数据来源确认的清单

```text
[NOT_YET_AVAILABLE] 每列准确工程单位
[NOT_YET_AVAILABLE] 物理采样周期
[NOT_YET_AVAILABLE] 晶体直径测量方式
[NOT_YET_AVAILABLE] 设备端滤波或估计规则
[NOT_YET_AVAILABLE] 直径信号固定延迟
[NOT_YET_AVAILABLE] 传感器安装位置
[NOT_YET_AVAILABLE] 设定值/操纵量/反馈量分类
[NOT_YET_AVAILABLE] 是否可提供更多晶棒
[NOT_YET_AVAILABLE] 新晶棒是否仍只包含等径阶段
[NOT_YET_AVAILABLE] 新晶棒之间的设备和工艺条件是否一致
```
