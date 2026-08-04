# PRISM Theory v2.0
## Physics-first Response Identification with Scale-specific Multirate Modular Operators
### 物理优先、尺度专属、多速率、可装配响应辨识理论体系

> **正式中文名**：PRISM 物理优先尺度专属多速率模块化响应辨识理论  
> **正式英文名**：Physics-first Response Identification with Scale-specific Multirate Modular Operators  
> **简称**：PRISM v2 / PRISM-M  
> **版本日期**：2026-08-04  
> **文档性质**：理论对象、模块接口、解释权边界、结构选择、认证条件和部署合同  
> **直接理论母本**：`PRISM_Theory_v1_3_Theory_Only.md`  
> **母本 SHA256**：`32833133c4a05d08bd6bf3060bf1057da808860218894efc7bd8d825fd3beeff`  
> **版本边界**：本文保留 v1.3 的通道专属多速率 profile、Urysohn 有限模型、Physics-First 与 K-Joint 双路线、成熟残差、商空间/Schur 可辨识性、因果 split/purge 和部署合同；新增模块代数、Wiener 读出、通道联合基、状态-only 面孔和受约束装配选择。  
> **理论边界**：本文不写入任何数据集排名、误差数值或经验胜负；实验只能检验模块是否被激活及其预测与结构证书，不能反向定义理论。
> **配套数值冻结**：本轮 benchmark 的全部门槛由 `PRISM_V2_ASSEMBLY_CONFIG_FROZEN.json` 与 `PRISM_V2_NUMERICAL_FREEZE.md` 唯一规定；理论正文不替代实验数值配置。
> **配套数值冻结**：本轮 benchmark 的全部门槛由 `PRISM_V2_ASSEMBLY_CONFIG_FROZEN.json` 与 `PRISM_V2_NUMERICAL_FREEZE.md` 唯一规定；理论正文不替代实验数值配置。

---

# 0. v2.0 的核心修订

## 0.1 PRISM 不再被定义为一个固定模型

PRISM v1.3 的正式解释路线可以概括为

\[
K\longrightarrow \text{freeze }K\longrightarrow A_{\mathrm{res}}\;\text{or exact-zero},
\]

并保留 K-Joint AR 预测路线。v2.0 不删除这两条路线，而是把它们放入一个更一般的装配空间：

\[
\boxed{
\mathfrak P
=
\mathfrak T
\circ
\mathfrak E
\circ
\bigl(
\mathfrak K,
\mathfrak C,
\mathfrak W,
\mathfrak A,
\mathfrak J
\bigr)
}
\tag{0.1}
\]

其中：

- \(\mathfrak T\)：不可关闭的时间因果与信息可用性合同；
- \(\mathfrak E\)：单尺度、固定多分辨率或通道专属多速率编码；
- \(\mathfrak K\)：过程变量输入响应算子，可取 exact-zero；
- \(\mathfrak C\)：通道融合，可取纯加性；
- \(\mathfrak W\)：Wiener 型静态读出，可取 identity；
- \(\mathfrak A\)：状态记忆或成熟残差记忆，可取 exact-zero；
- \(\mathfrak J\)：联合预测路线，只输出总预测，不自动分配物理解释权。

因此 PRISM 的正式对象是

\[
\boxed{
\text{固定接口、固定中性元、固定因果合同和固定选择规则下的模块化模型族}
}
\tag{0.2}
\]

而不是一个始终包含相同层数的网络。

## 0.2 v2.0 的两副正式面孔

### 面孔 S：状态/系统辨识模式

当没有可信过程变量，或过程变量不满足时间、完整性、代理变量和可辨识性合同，关闭 \(K\) 支路：

\[
\widehat z_t=\mathcal A(Y_t^{-},C_t^{-}),
\qquad \mathcal K=0,
\qquad \mathcal W=I.
\tag{0.3}
\]

这里 \(C_t^{-}\) 表示允许使用的已知上下文，如阶段、运行编号或已冻结工况标签。该模式可以是 AR、目标-only NAR、稳定状态空间或其他严格过去状态预测器，但不能声称恢复了输入物理响应。

### 面孔 P：过程变量/工业软测量模式

当过程变量满足合同，输入-only 路线为

\[
\widehat z_t^{\mathrm{input}}
=
\mathcal W\!\left(
\mathcal C\{\mathcal K_j[u_j^-]\}_{j=1}^{p}
\right),
\tag{0.4}
\]

不得读取目标历史。若目标历史也在线可用，则解释路线为

\[
\widehat z_t^{\mathrm{PF}}
=
\underbrace{\mathcal W\!\left(
\mathcal C\{\mathcal K_j[u_j^-]\}
\right)}_{\text{冻结输入响应}}
+
\underbrace{\mathcal A(\widetilde R_t^{\mathrm{mature},-})}_{\text{冻结后成熟残差状态}},
\tag{0.5}
\]

联合预测路线为

\[
\widehat z_t^{\mathrm{J}}
=
\mathcal J(\Phi_t^{K},Y_t^{-}),
\tag{0.6}
\]

其中 \(\Phi_t^K\) 为冻结 profile 定义下的输入响应特征。式 (0.6) 的总预测可以验证，但其内部 K/状态分解不自动获得物理解释权。

## 0.3 v2.0 的中性元

每个可拆模块必须拥有显式中性状态：

| 模块 | 中性状态 | 中性语义 |
|---|---|---|
| 时间合同 \(T\) | 无 | 必须开启 |
| 多尺度编码 \(E\) | 单尺度 | 不引入通道专属多速率 |
| 输入响应 \(K\) | exact-zero | 不使用过程变量 |
| 通道融合 \(C\) | additive | 无跨通道交互 |
| Wiener 读出 \(W\) | identity | 无后置静态曲率 |
| 状态模块 \(A\) | exact-zero | 无状态补偿 |
| 工况模块 \(R\) | single regime | 不做混合专家 |
| 在线适配 \(D_a\) | frozen | 不在线改参数 |

中性元使所有复杂装配精确包含更简单装配，避免“模型必须非零”“每个模块都必须有贡献”的实现偏见。

## 0.4 v1.3 的保留、扩展与禁止回退

### 保留

1. 通道专属 \((\Delta,h,W,W_0,T,\mathcal B)\) profile；
2. exact-zero、线性分布时滞、rank-1、rank-\(R\)、完整有限 Urysohn 阶梯；
3. Gram、Schur、支持稳定性和 rank margin；
4. 物理层先冻结、OOF 残差、成熟条件；
5. K-Joint AR 仅作联合预测路线；
6. forward/group split、purge、OOD、moving-block bootstrap；
7. CPU FP64、matrix-free、CUDA/PLC 编译与误差预算；
8. predictive rank、structural rank、deployment rank 分离。

### 扩展

1. 输入响应特征不再必须先压缩为每通道一个标量；
2. 新增通道联合基与稀疏交互融合；
3. 新增可关闭 Wiener 读出；
4. 新增状态-only 正式面孔；
5. 新增模块装配图、模块中性元和装配卡；
6. 新增物理空间与状态空间的解释权正交合同；
7. 新增主解释路线、输入-only 路线、状态-only 路线和联合预测路线的统一接口。

### 禁止回退

1. 不恢复独立 Q 层；统计证书不是预测模块；
2. 不恢复按变量顺序抢残差；
3. 不允许 W、A 或联合回调反向改写已认证的 K；
4. 不允许用测试/OOD 选择模块；
5. 不允许把状态-only 模式写成物理输入识别；
6. 不允许把联合预测路线内部参数自动解释为物理机制；
7. 不允许在同一解释路线中同时启用重复的输入静态非线性和 Urysohn 幅值非线性而不做归属约束。

---

# 1. 不可变时间因果合同

## 1.1 目标头

对目标头 \(m\)，定义

\[
\eta_m=(h_m,W_m,W_{0,m},D_m),
\tag{1.1}
\]

其中：

- \(h_m\)：预测提前步；
- \(W_m\)：未来目标平均窗口；
- \(W_{0,m}\)：当前状态锚点窗口；
- \(D_m\)：目标或残差可用性的额外成熟延迟。

当前窗口和未来窗口使用半开区间：

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

主预测目标为变化量

\[
z_t^{(m)}
=
\bar y_{t+h_m}^{(W_m)}-\bar y_t^{(W_{0,m})}.
\tag{1.4}
\]

即便 \(h_m=0\)，状态输入也最多使用 \(y_{t-1-D_m}\)，不得读取目标窗口内样本。

## 1.2 通道 profile

对通道 \(j\) 和目标头 \(m\)，定义

\[
\pi_{j,m,r}
=
(\Delta_{j,r},h_m,W_m,W_{0,m},T_{j,r},\mathcal B_{j,r}),
\tag{1.5}
\]

其中 \(r\) 为候选尺度编号。不同通道可以拥有不同：

- 重采样步长 \(\Delta_{j,r}\)；
- 历史覆盖 \(T_{j,r}\)；
- 时滞基 \(\mathcal B_{j,r}\)。

但同一兼容目标头必须共享相同的 \((h_m,W_m,W_{0,m},D_m)\)。

## 1.3 合法信息集

输入-only 信息集：

\[
\mathcal I_t^{U}
=
\sigma\{u_{j,s},x_{k,s},c_s:s<t\},
\tag{1.6}
\]

状态信息集：

\[
\mathcal I_t^{Y}
=
\sigma\{y_s:s\le t-1-D_m\},
\tag{1.7}
\]

动态信息集：

\[
\mathcal I_t^{D}=\mathcal I_t^{U}\vee\mathcal I_t^{Y}.
\tag{1.8}
\]

所有模块都必须是对应信息集上的可测函数。模块化不改变因果边界。

## 1.4 primitive support 与 purge

设装配 \(a\) 的最大向后依赖为 \(L_a\)，则预测原点 \(t\) 的 primitive support 为

\[
I_t^{(a)}
=
[t-L_a,\;t+h_m+W_m+D_m).
\tag{1.9}
\]

相邻开发/评估集合之间使用

\[
G_a=L_a+h_m+W_m+D_m+b,
\tag{1.10}
\]

其中 \(b\) 为冻结的额外依赖间隔。完整 run/profile 之间禁止任何输入、目标、残差或状态窗口跨界。

---

# 2. 模块代数与装配图

## 2.1 模块接口

每个模块写成带类型的映射：

\[
E_j:\mathcal I_t^U\to\mathbb R^{d_{E,j}},
\tag{2.1}
\]

\[
K_j:\mathbb R^{d_{E,j}}\to\mathbb R^{d_{K,j}},
\tag{2.2}
\]

\[
C:\prod_j\mathbb R^{d_{K,j}}\to\mathbb R^{d_C},
\tag{2.3}
\]

\[
W:\mathbb R^{d_C}\to\mathbb R,
\tag{2.4}
\]

\[
A:\mathbb R^{d_A}\to\mathbb R,
\tag{2.5}
\]

\[
J:\mathbb R^{d_C+d_Y}\to\mathbb R.
\tag{2.6}
\]

解释路线的装配顺序固定为

\[
\mathcal I_t^U
\xrightarrow{E}
\Phi_t
\xrightarrow{K}
Q_t
\xrightarrow{C}
q_t
\xrightarrow{W}
p_t
\xrightarrow{\text{freeze}}
R_t^{\mathrm{OOF}}
\xrightarrow{A}
a_t.
\tag{2.7}
\]

最终预测

\[
\widehat z_t=p_t+a_t.
\tag{2.8}
\]

## 2.2 有效装配集合

核心装配集合为

\[
\mathcal G_{\mathrm{core}}
=
\{
A,
K,
W\circ K,
K\oplus A,
(W\circ K)\oplus A,
J(K,Y^-),
J(W\circ K,Y^-)
\}.
\tag{2.9}
\]

其中：

- \(A\)：状态-only；
- \(K\)：输入-only、identity Wiener、无状态；
- \(W\circ K\)：输入-only、带静态读出；
- \(K\oplus A\)：Physics-First v1.3 的一般化；
- \((W\circ K)\oplus A\)：v2.0 的完整解释路线；
- \(J(\cdot,Y^-)\)：联合预测路线。

并非所有数学组合都允许。以下组合禁止：

1. \(A\) 读取原始过程变量，同时仍声明其为残差状态模块；
2. W 在 K 冻结前后反复联合回调并仍声称 K 可解释；
3. 将同一静态输入非线性同时记入独立 Hammerstein 前端和 Urysohn 幅值基；
4. 工况门读取未来标签或测试分区标识；
5. 先在测试集挑装配，再回写装配卡。

## 2.3 装配偏序

定义复杂度偏序

\[
a_1\preceq a_2
\]

当且仅当 \(a_1\) 可由 \(a_2\) 的某些模块设为中性元精确得到。例如：

\[
K\preceq W\circ K\preceq (W\circ K)\oplus A,
\tag{2.10}
\]

\[
A\preceq J(K,Y^-),
\quad \text{当 }K=0.
\tag{2.11}
\]

装配选择使用该偏序优先简单模型，而不是只按参数个数排序。

## 2.4 装配卡

每个目标头最终必须输出规范化装配卡：

```json
{
  "time_contract": "HALF_OPEN_CAUSAL",
  "scale_encoder": "CHANNEL_SPECIFIC",
  "input_operator": "RANK_2_URYSOHN",
  "channel_fusion": "ADDITIVE",
  "wiener_readout": "IDENTITY",
  "state_module": "MATURE_RESIDUAL_AR",
  "joint_route": "OFF",
  "regime_module": "SINGLE",
  "selection_status": "FROZEN_ON_DEVELOPMENT"
}
```

装配卡是结构结果的一部分，不是便于展示的附属元数据。

---

# 3. 多尺度编码与输入响应模块

## 3.1 编码模块 \(E\)

对通道 \(j\)，多分辨率时滞块为

\[
\phi_{j,b}(t)
=
\frac{1}{|I_{j,b}|}
\sum_{\tau\in I_{j,b}}u_j(t-\tau).
\tag{3.1}
\]

编码候选：

1. `SINGLE_SCALE`：所有通道共享单一 \(\Delta,T,\mathcal B\)；
2. `FIXED_MULTIRESOLUTION`：所有通道共享一组固定多分辨率块；
3. `CHANNEL_SPECIFIC`：每个通道独立选择 \(\Delta_j,T_j,\mathcal B_j\)。

编码模块只决定历史表示，不决定幅值非线性和通道是否 active。

## 3.2 输入响应模块 \(K_j\)

每个通道候选阶梯为

\[
\mathcal K_j^{(0)}=0,
\tag{3.2}
\]

\[
\mathcal K_j^{(L)}[u_j^-]
=
\sum_b\beta_{j,b}\phi_{j,b}(t),
\tag{3.3}
\]

\[
\mathcal K_j^{(R)}[u_j^-]
=
\sum_{r=1}^{R}
\left(
\sum_b a_{j,r,b}\psi_{j,b}(t)
\right)
\left(
\sum_k c_{j,r,k}\chi_{j,k}(u_j)
\right),
\tag{3.4}
\]

以及完整有限 Urysohn

\[
\mathcal K_j^{(F)}[u_j^-]
=
\sum_{b,k}\Theta_{j,bk}\psi_{j,b}(t)\chi_{j,k}(u_j).
\tag{3.5}
\]

模型阶梯必须满足

\[
0\subset L\subset R1\subset R2\subset\cdots\subset F.
\tag{3.6}
\]

## 3.3 输入静态非线性的归属规则

经典 Hammerstein 前端可写为

\[
v_{j,t}=H_j(u_{j,t}),
\qquad q_{j,t}=L_j[v_j^-].
\tag{3.7}
\]

在 PRISM v2 中，若 \(K_j\) 使用 Urysohn 幅值基，则 \(H_j\) 被视为 \(K_j\) 的内部参数化，不再作为独立可解释模块。只有在线性动态核 \(L_j\) 与显式静态前端 \(H_j\) 的专门 Hammerstein 装配中，才单独登记 \(H_j\)。

因此同一通道不允许同时把相同曲率归入 \(H_j\) 和 Urysohn \(K_j\)。

## 3.4 通道 active、exact-zero 与可分离性

通道 \(j\) 的总设计空间为 \(\mathcal V_j\)，其相对其他通道的 Schur 信息量为

\[
S_j
=
G_{jj}-G_{j,-j}G_{-j,-j}^{\dagger}G_{-j,j}.
\tag{3.8}
\]

只有当：

1. \(S_j\) 的有效特征值超过阈值；
2. 外层运行方向稳定；
3. 支持与幅值域覆盖稳定；
4. exact-zero 不在 one-SE 可接受集合；

才允许将通道声明为 active。否则输出 `EXACT_ZERO` 或 `UNRESOLVED`。

---

# 4. 通道融合模块 \(C\)

## 4.1 为什么保留完整通道基

v1.3 允许每个通道先形成标量贡献 \(q_j(t)\)，再进行加和。v2.0 同时允许保留通道内部已冻结的基表示：

\[
Q_t
=
[Q_{1,t},\ldots,Q_{p,t}],
\qquad Q_{j,t}\in\mathbb R^{d_{K,j}}.
\tag{4.1}
\]

这样最终物理层可以在不改变通道 profile 和基族的前提下，进行一次兼容目标头内的联合固定支持重拟合。

## 4.2 加性融合

\[
C_{\mathrm{add}}(Q_t)
=
\beta_0+
\sum_j\beta_j^\top Q_{j,t}.
\tag{4.2}
\]

这是中性融合形式，也是所有更复杂融合的精确子模型。

## 4.3 稀疏二阶交互

对候选通道对 \((j,k)\)，构造残差化交互基

\[
B_{jk}^{\perp}
=
(I-P_{[1,Q_j,Q_k]})B_{jk}(Q_j,Q_k).
\tag{4.3}
\]

融合为

\[
C_{\mathrm{pair}}(Q_t)
=
C_{\mathrm{add}}(Q_t)
+
\sum_{(j,k)\in\mathcal P_*}
\gamma_{jk}^\top B_{jk}^{\perp}(Q_{j,t},Q_{k,t}).
\tag{4.4}
\]

条件

\[
\mathbb E[B_{jk}^{\perp}\mid Q_j]=0,
\qquad
\mathbb E[B_{jk}^{\perp}\mid Q_k]=0
\tag{4.5}
\]

保证交互项不能重新伪装成单通道主效应。

## 4.4 低秩融合

当通道数较多时，可使用

\[
C_{\mathrm{LR}}(Q_t)
=
C_{\mathrm{add}}(Q_t)
+
\sum_{r=1}^{R_C}
\prod_j f_{j,r}(Q_{j,t}),
\tag{4.6}
\]

但其结构解释要求额外的旋转、尺度和符号规范化。v2.0 理论允许该模块，首轮核心 benchmark 不强制启用。

## 4.5 融合模块的解释边界

- \(K_j\) 解释单通道内部的历史与幅值响应；
- \(C\) 解释多个已冻结通道表示之间的组合；
- 交互项不允许被称为单通道物理核；
- 若 \(C\) 使用高自由度黑箱，则只能输出 predictive fusion，不能升级为单通道物理认证。

---

# 5. Wiener 读出模块 \(W\)

## 5.1 广义 Wiener 读出

经典标量 Wiener 模型为

\[
q_t=L[u^-](t),
\qquad \widehat z_t=w(q_t).
\tag{5.1}
\]

PRISM v2 使用其向量潜变量一般化：

\[
q_t=C(Q_t)\in\mathbb R^{d_C},
\qquad p_t=W(q_t).
\tag{5.2}
\]

当 \(d_C=1\) 时退化为经典 Wiener 读出。

## 5.2 identity 中性元

\[
W_0(q)=q
\tag{5.3}
\]

或在向量表示下

\[
W_0(q)=\alpha_0+\alpha^\top q.
\tag{5.4}
\]

因此关闭 Wiener 模块不改变前级 K/C 的预测语义。

## 5.3 正交曲率基

设线性物理空间

\[
\mathcal P
=
\operatorname{span}\{1,q_1,\ldots,q_{d_C}\}.
\tag{5.5}
\]

对原始非线性基 \(B(q)\)，定义训练几何下的残差化基

\[
\widetilde B(q)
=
(I-P_{\mathcal P})B(q).
\tag{5.6}
\]

Wiener 读出写为

\[
W(q)
=
\alpha_0+\alpha^\top q
+
\theta^\top\widetilde B(q).
\tag{5.7}
\]

于是训练样本上

\[
\langle \widetilde B,1\rangle=0,
\qquad
\langle \widetilde B,q_i\rangle=0.
\tag{5.8}
\]

这使 \(W\) 只能解释 K/C 线性物理空间之外的静态曲率。

## 5.4 Wiener 候选

1. `IDENTITY`；
2. `MONOTONE_I_SPLINE`：适合已知单调测量或饱和关系；
3. `NATURAL_CUBIC_SPLINE`：允许非单调平滑曲率；
4. `ANOVA_SPLINE`：单变量曲率加少量残差化二阶交互；
5. `ISOTONIC`：仅作为非光滑单调工程候选，部署参数量按实际断点计数。

## 5.5 W 何时可以激活

W 只有在开发数据上同时满足以下条件才激活：

1. identity 不在 one-SE 可接受集合；
2. 相对 identity 的验证 MSE 改善超过冻结实用门槛；
3. 正交残差不超过数值阈值；
4. 曲率符号或单调方向在 outer folds 稳定；
5. OOD 幅值范围未大面积超出训练支持，或已声明 extrapolation contract。

W 被关闭不是模型失败，而是正式结构结论 `WIENER_IDENTITY`。

## 5.6 W 的科学边界

W 表示冻结输入响应到目标观测之间的静态读出曲率。它可能来自：

- 传感器或实验室分析映射；
- 饱和或阈值；
- 未显式分解的稳态非线性；
- 多通道潜变量到目标的静态投影。

除非额外实验与机理支持，不能把 W 自动写成唯一物理构成方程。

---

# 6. 状态与残差模块 \(A\)

## 6.1 状态-only 模式

当 \(K=0\) 时，定义

\[
\widehat z_t=A(Y_t^-).
\tag{6.1}
\]

候选包括：

- 线性 AR；
- 目标-only 二次 NAR；
- 稳定状态空间；
- 其他严格过去、按目标头独立注册的状态预测器。

该模式输出 predictive state identification，不输出输入物理贡献。

## 6.2 Physics-First 成熟残差

先在滚动 OOF 中获得输入响应预测

\[
p_s^{\mathrm{OOF}}
=W(C(K(E(U_s^-)))).
\tag{6.2}
\]

定义

\[
r_s^{\mathrm{OOF}}
=z_s-p_s^{\mathrm{OOF}}.
\tag{6.3}
\]

在预测原点 \(t\)，只有满足

\[
s+h_m+W_m+D_m\le t
\tag{6.4}
\]

的残差可以进入状态模块。

## 6.3 状态特征的物理残差化

设成熟残差历史特征为 \(H_t\)，冻结物理表示为 \(\Phi_t^{PW}\)。在开发集上计算

\[
\Gamma
=
\arg\min_G\|H-\Phi^{PW}G\|_F^2,
\tag{6.5}
\]

并定义

\[
\widetilde H
=H-\Phi^{PW}\Gamma.
\tag{6.6}
\]

残差状态预测为

\[
a_t=A(\widetilde H_t).
\tag{6.7}
\]

在线性情形，训练几何下有

\[
(\Phi^{PW})^\top\widetilde H=0.
\tag{6.8}
\]

这给出模块间的经验解释权分隔：状态模块不能再利用与冻结物理空间线性重合的方向。

## 6.4 A 的 exact-zero

\[
A_0=0.
\tag{6.9}
\]

若成熟残差历史不能在开发数据中稳定改善预测，最终装配退化为 input-only \(W\circ C\circ K\)。

## 6.5 A 的解释边界

成熟残差可能包含：

- 未测内部状态；
- 未测扰动；
- 控制器内部状态；
- 模型截断误差；
- 测量误差的可预测部分。

因此 A 可以称为状态/残差记忆，不应被命名为某一具体物理机制，除非有独立观测支持。

---

# 7. 联合预测模块 \(J\)

## 7.1 定义

\[
\widehat z_t^{J}
=
J(\Phi_t^K,Y_t^-).
\tag{7.1}
\]

固定基下可以采用带块惩罚的联合最小二乘：

\[
\min_{\beta_K,\beta_Y}
\|z-\Phi^K\beta_K-H^Y\beta_Y\|_2^2
+
\lambda_K\|\beta_K\|_2^2
+
\lambda_Y\|\beta_Y\|_2^2.
\tag{7.2}
\]

也可以加入冻结 Wiener 基，但必须把总模型标记为 predictive joint。

## 7.2 精确退化候选

联合路线至少包含：

\[
(K=0,A=0),
\quad
(K=0,A\ne0),
\quad
(K\ne0,A=0),
\quad
(K\ne0,A\ne0).
\tag{7.3}
\]

## 7.3 不可归因性

若 \(\operatorname{span}(\Phi^K)\cap\operatorname{span}(H^Y)\ne\{0\}\)，则存在非零 \(v\) 使

\[
\Phi^K\delta_K=H^Y\delta_Y=v.
\tag{7.4}
\]

于是

\[
(\beta_K,\beta_Y)
\mapsto
(\beta_K+\delta_K,\beta_Y-\delta_Y)
\tag{7.5}
\]

不改变总预测。正则可以选择一个数值代表，但不能制造唯一物理分解。

因此 J 的用途是工程预测和结构上限评估，不是替代 Physics-First 认证。

---

# 8. 工况模块、适配模块与审计外壳

## 8.1 工况模块 \(R\)

当单一装配在开发运行域中出现稳定的多工况失配时，可定义

\[
\widehat z_t
=
\sum_{r=1}^{R}\pi_r(c_t^-)
\left[p_{r,t}+a_{r,t}\right],
\qquad
\sum_r\pi_r=1.
\tag{8.1}
\]

要求：

1. 门控只读严格过去和已知上下文；
2. `single regime` 是正式候选；
3. 工况数由开发数据选择；
4. 每个局部模块仍遵守 K/W/A 解释权合同；
5. 首轮 PRISM v2 核心 benchmark 将该模块置于 `DEFERRED`，避免用混合专家掩盖基础结构问题。

## 8.2 在线适配模块 \(D_a\)

在线适配只能修改预先允许的参数子集，例如偏置或读出缩放：

\[
\theta_{t+1}
=
\Pi_{\Theta_{\mathrm{safe}}}
(\theta_t-\eta_t\nabla\ell_t).
\tag{8.2}
\]

若适配会改变 K 的形状，则原物理证书失效，必须重新认证。v2.0 核心理论保留接口，但 benchmark 默认 `frozen`。

## 8.3 不确定性不是独立预测层

bootstrap、运行域离散、OOD 幅值距离、支持覆盖和数值条件属于审计外壳：

\[
\mathcal V(\widehat z_t)
=
(\text{interval},\text{support},\text{OOD flag},\text{certificate}).
\tag{8.3}
\]

它们不定义新的 Q 预测层，也不参与抢占目标残差。

---

# 9. 解释路线的严格训练顺序

## 9.1 Stage K0：注册与数据冻结

冻结：

- 目标头；
- 时间换算；
- run/profile split；
- proxy policy；
- 标签成熟；
- purge；
- train-only scaler；
- 候选模块格。

## 9.2 Stage K1：单通道 profile 审计

对每个通道和候选 profile：

1. exact-zero；
2. 线性分布时滞；
3. rank 阶梯；
4. 数值证书；
5. 运行域稳定性；
6. 与尺度匹配状态模型的条件新颖性诊断。

输出候选通道及固定支持，不生成最终 test 预测。

## 9.3 Stage K2：兼容目标头联合物理重拟合

保留每通道已冻结基，构造

\[
\Phi^K=[\Phi_1^K,\ldots,\Phi_p^K]
\tag{9.1}
\]

并在开发集固定支持重拟合。可选择 additive 或稀疏 interaction 融合。不得重新搜索未注册 profile。

## 9.4 Stage W：冻结 K/C 后选择 Wiener

使用 rolling OOF K/C 潜变量和预测，构造正交曲率基。比较 identity 与注册 W 候选。W 选择后冻结，不回调 K/C。

## 9.5 Stage A：冻结 K/C/W 后选择成熟残差状态

生成 OOF 成熟残差，构造残差化状态特征，比较 exact-zero 与注册 A 候选。A 选择后冻结。

## 9.6 Stage F：最终开发集重拟合

在结构与超参数全部冻结后：

1. K/C 在 train+validation 固定支持重拟合；
2. W 在固定 OOF 规则与冻结基上重拟合；
3. A 在冻结成熟残差合同上重拟合；
4. 禁止最终端到端联合优化。

## 9.7 联合预测路线

J 使用同一开发 split 和冻结特征族，但可以联合优化 K/W/state 参数。其输出文件必须单独标记 `PREDICTIVE_JOINT`。

---

# 10. 模块选择理论

## 10.1 有限装配格

设候选装配集合为有限集

\[
\mathcal A=\{a_1,\ldots,a_M\}.
\tag{10.1}
\]

每个装配的开发风险为

\[
\widehat R(a)
=
\frac1K\sum_{k=1}^{K}L_k(a),
\tag{10.2}
\]

其中折由完整实体或严格 forward/purge 规则生成。

## 10.2 one-SE 与偏序选择

设最小平均风险模型为 \(a_*\)，其折间标准误为 \(\mathrm{SE}_*\)。可接受集合为

\[
\mathcal A_{1\mathrm{SE}}
=
\{a:\widehat R(a)\le\widehat R(a_*)+\mathrm{SE}_*\}.
\tag{10.3}
\]

从中选择偏序最小、参数更少、正则更强、历史更短、分辨率更粗的装配。

因此模块只有在中性模型不再属于可接受集合时才被激活。

## 10.3 实用改善门槛

one-SE 控制统计波动，实用门槛控制极小但无工程意义的复杂化。模块 \(M\) 相对中性元 \(M_0\) 的开发改善定义为

\[
\Delta_M
=
\frac{R(M_0)-R(M)}{R(M_0)}.
\tag{10.4}
\]

只有 \(\Delta_M\ge\epsilon_M\) 且满足 one-SE、数值和稳定性门时才激活。\(\epsilon_M\) 必须在实验协议中预先冻结，理论本文不指定具体数值。

## 10.4 装配稳定性

令 \(a^{(k)}\) 为第 \(k\) 个 outer-development 方向选择的装配。模块 \(M\) 的激活稳定率为

\[
\rho_M
=
\frac1K\sum_{k=1}^{K}\mathbf 1\{M\in a^{(k)}\}.
\tag{10.5}
\]

低稳定率结构只能标记 `PREDICTIVE_SELECTED` 或 `UNRESOLVED`，不能声明为稳定物理模块。

---

# 11. 可辨识性与解释权

## 11.1 总预测、模块投影与内部参数

PRISM v2 区分：

1. 总预测函数 \(\widehat z\)；
2. 冻结信息几何下的模块投影；
3. 基函数系数；
4. 连续核代表；
5. 机理解释。

总预测唯一不意味着模块分解唯一，模块投影唯一不意味着内部低秩因子唯一。

## 11.2 K 的商空间

若两个核 \(K,K'\) 在观测设计支持上产生相同输出，则

\[
K\sim K'
\iff
\mathcal A(K-K')=0.
\tag{11.1}
\]

可识别对象首先是商空间 \(\mathcal H/\ker\mathcal A\) 中的代表，而非全局连续核。

## 11.3 W 的条件唯一性

在 K/C 冻结、\([1,q,\widetilde B]\) 满列秩且正则严格凸时，式 (5.7) 的有限维预测系数唯一。该唯一性只属于冻结潜变量几何，不证明真实过程存在唯一静态 Wiener 方程。

## 11.4 A 的条件唯一性

在物理空间冻结、残差历史经式 (6.6) 残差化且设计 Gram 正定时，线性 A 的预测投影唯一。它仍只是未由物理输入表示解释的可预测状态方向。

## 11.5 闭环边界

被动闭环数据默认识别

\[
\boxed{
\text{当前控制策略、运行域和观测分布下的预测响应算子}
}
\tag{11.2}
\]

除非存在外生激励、控制器知识、工具变量、跨控制策略复现或其他因果设计，不得把 K/W/A 自动写成开放环 plant 结构。

---

# 12. 认证门槛与状态标签

## 12.1 模块级证书

### K 证书

- profile 合法；
- exact-zero 对照；
- Gram/Schur 条件；
- 支持覆盖；
- rank margin；
- 外层运行稳定；
- placebo 通过；
- 数值残差通过。

### C 证书

- 加性基固定；
- 交互残差化通过；
- 交互对在开发折稳定；
- 不改变单通道主效应定义。

### W 证书

- identity 对照；
- 正交曲率；
- knot/shape train-only；
- 单调约束若被声明则全部通过；
- 支持外推合同；
- 参数量按真实自由度计数。

### A 证书

- rolling OOF；
- 成熟条件；
- 物理空间残差化；
- exact-zero 对照；
- 不读取原始 U/X；
- 不使用测试/OOD 选择。

## 12.2 装配状态

| 状态 | 含义 |
|---|---|
| `ASSEMBLY_FROZEN` | 开发数据上结构与超参数已冻结 |
| `STATE_ONLY` | K=0，仅状态预测 |
| `INPUT_ONLY` | A=0，不用目标历史 |
| `PHYSICS_FIRST` | K/C/W 先冻结，A 只用成熟残差 |
| `PREDICTIVE_JOINT` | 联合优化，只解释总预测 |
| `WIENER_IDENTITY` | W 被正式关闭 |
| `STATE_EXACT_ZERO` | A 被正式关闭 |
| `CHANNEL_EXACT_ZERO` | 某通道 K 被正式关闭 |
| `PHYSICAL_CERTIFIED` | 对应模块完整证书通过 |
| `PREDICTIVE_VALIDATED` | 预测证据通过但结构证据不足 |
| `UNRESOLVED` | 证据不足 |
| `OOD_UNSTABLE` | OOD 方向失败 |

## 12.3 论文声明矩阵

| 证据 | 允许声明 |
|---|---|
| 状态-only 预测通过 | 该目标存在可预测状态记忆 |
| input-only 预测通过 | 过程变量历史含预测信息 |
| K 完整证书通过 | 当前运行域下存在稳定通道预测响应结构 |
| W 通过 | 冻结输入响应到目标之间存在稳定静态曲率 |
| A 通过 | 物理层之外仍存在成熟可预测状态 |
| J 通过 | 联合信息集提高总体预测 |
| OOD 失败 | 模型在该运行域迁移下不稳定 |
| 模块 exact-zero | 开发证据不支持激活该模块 |

禁止从单一结果跳跃到：开放环因果、唯一机理、全工况通用或控制器类型识别。

---

# 13. 数值实现

## 13.1 训练几何

所有标准化、幅值 knots、白化矩阵、正交投影和候选筛选只使用当前开发训练部分。最终系数使用 FP64。

## 13.2 固定支持直接求解

线性/样条固定基下：

\[
(\Phi^\top\Phi+\Lambda)\theta=\Phi^\top z.
\tag{13.1}
\]

求解顺序：Cholesky、pivoted QR、SVD rescue。必须输出相对 KKT、条件数和有效秩。

## 13.3 Matrix-free

大样本下只实现

\[
v\mapsto \Phi^\top(\Phi v)+\Lambda v,
\tag{13.2}
\]

并用 PCG/LSMR。所有 chunk 必须保持 sample ID 顺序。

## 13.4 正交基的数值构造

W 或交互基残差化应使用 QR/SVD：

\[
\widetilde B=B-Q(Q^\top B),
\tag{13.3}
\]

其中 \(Q\) 是基础空间的正交基。禁止显式构造大型 \(I-P\) 矩阵。

## 13.5 参数量

总参数量必须分解为

\[
N_{\mathrm{total}}
=
N_E+N_K+N_C+N_W+N_A+N_R.
\tag{13.4}
\]

至少同时报告：

- stored parameter count；
- active coefficient count；
- effective degrees of freedom；
- deployment state count。

样条 knots、低秩因子、isotonic 断点和状态矩阵都必须计入，不能只统计最后一个 readout。

---

# 14. 部署合同

## 14.1 在线装配执行

在线时按装配卡执行：

1. 多速率缓存更新；
2. K 通道响应；
3. C 融合；
4. W 查表/样条；
5. 成熟残差状态更新；
6. 输出总预测和审计标志。

未激活模块不占用运行时分支。

## 14.2 LUT/FIR/IIR 编译

- 线性 K：FIR/IIR；
- 有限 Urysohn：幅值 LUT + 时滞 FIR；
- W：一维 LUT、分段多项式或小型 ANOVA LUT；
- A：AR/IIR 或稳定状态空间。

## 14.3 误差预算

部署误差上界写为

\[
\varepsilon_{\mathrm{deploy}}
\le
\varepsilon_E+
\varepsilon_K+
\varepsilon_C+
\varepsilon_W+
\varepsilon_A+
\varepsilon_q,
\tag{14.1}
\]

分别对应重采样、核截断、融合近似、W 查表、状态近似和量化。模块关闭时对应误差项为零。

## 14.4 OOD 合同

对每个模块记录训练支持：

- 输入幅值范围；
- 潜变量 \(q\) 范围；
- 残差状态范围；
- 工况标签范围。

超出支持时输出 OOD 标志。有限带延拓只能保证数值连续，不能保证科学有效。

---

# 15. 理论命题

## 命题 1：装配因果闭包

若 E、K、C、W、A 分别只读取其注册的合法信息集，且 A 只读取满足成熟条件的残差，则任意允许的核心装配都是 \(\mathcal I_t^U\)、\(\mathcal I_t^Y\) 或其并集上的因果可测函数。

**证明要点**：合法信息集对可测函数复合与有限加法封闭；成熟残差由不晚于 \(t\) 可获得的历史目标构成。

## 命题 2：中性元嵌套

核心装配格中的简单模型可以由复杂模型将相应模块设置为 exact-zero、identity 或 additive 精确得到。

**意义**：one-SE 能在真正嵌套的候选之间优先简单结构，而不是比较不相容模型。

## 命题 3：两副面孔是同一模型族的限制面

状态-only 模式是 \(K=0,W=I\) 的限制；输入-only 模式是 \(A=0\) 的限制；Physics-First 是 K/C/W 冻结后添加 A 的限制；联合路线共享相同输入特征族但不共享解释权。

## 命题 4：W 正交曲率不能重写线性物理空间

按式 (5.6) 构造时，W 的非线性校正与 \(\operatorname{span}\{1,q\}\) 正交。因此在训练几何下，W 不能通过添加常数或线性斜率重新表示 K/C 的一阶贡献。

## 命题 5：残差化状态与冻结物理空间正交

按式 (6.6) 构造时，\((\Phi^{PW})^\top\widetilde H=0\)。在线性 A 中，状态预测不使用冻结物理空间的线性重合方向。

## 命题 6：正交不等于因果独立

命题 4、5 只给出开发分布和选定特征空间下的经验几何分离，不证明 W、A 与真实物理机制统计独立或因果独立。

## 命题 7：联合路线的总预测可以唯一而分解不唯一

当联合设计的正规方程严格凸时，总预测系数代表可唯一；若 K 与状态子空间相交，内部块贡献仍可依赖正则和参数化，因此不能作物理归因。

## 命题 8：模块 exact-zero 是结构结果

若中性元位于 one-SE 可接受集合并按偏序被选择，则关闭该模块是正式开发结论，不应被当作训练失败或缺失结果。

## 命题 9：模块化不允许测试集定制

模型族可按过程选择不同装配，但选择映射必须在开发数据上冻结。若先观察测试结果再增加模块，则同一测试只能作 post-hoc exploratory 评价，不能作原始确认性证据。

## 命题 10：部署近似不能增强可辨识性

LUT、FIR、状态空间或量化只能近似已冻结装配，不能把 `UNRESOLVED` 模块升级为 `PHYSICAL_CERTIFIED`。

---

# 16. 正式研究流程

1. 冻结数据、目标头、split、purge 和模块格；
2. 审计是否允许过程变量面孔；
3. 同时保留状态-only 基线；
4. 单通道多尺度 K 审计；
5. 兼容目标头联合物理基重拟合；
6. 选择 C 的 additive/interaction；
7. 冻结 K/C，选择 identity/非线性 W；
8. 冻结 K/C/W，生成 rolling OOF 成熟残差；
9. 选择 exact-zero/状态 A；
10. 生成装配卡；
11. 独立运行联合预测路线；
12. 冻结后访问测试/OOD；
13. 报告预测、模块激活、参数量、证书和 OOD；
14. 编译到部署目标。

---

# 17. 版本边界和经验中立性

## 17.1 本文不声称

- Wiener 在所有工业过程上有效；
- 过程变量一定优于状态历史；
- PRISM 一定优于 Hammerstein、ARX、NARX 或神经网络；
- 模块化选择会自动恢复真实机理；
- 同一模块在所有目标头都应激活。

## 17.2 本文正式声称

- PRISM 可以在固定因果合同下同时容纳状态-only、输入-only、Physics-First 和联合预测；
- W、A、交互和工况模块可以拥有显式中性元；
- 不同过程可以通过统一开发规则选择不同装配；
- 解释路线可以通过冻结顺序与正交化分配经验解释权；
- 模块化结构不会取消闭环、可辨识性和 OOD 边界。

---

# 18. 最终语义链

\[
\boxed{
\text{冻结时间与信息合同}
\rightarrow
\text{选择状态面孔或过程变量面孔}
\rightarrow
\text{通道专属编码}
\rightarrow
\text{K 单通道响应}
\rightarrow
\text{C 联合物理基}
\rightarrow
\text{W identity 或静态曲率}
\rightarrow
\text{冻结物理表示}
\rightarrow
\text{成熟残差 A 或 exact-zero}
\rightarrow
\text{装配卡}
\rightarrow
\text{测试/OOD 与部署}
}
\tag{18.1}
\]

PRISM v2 的核心不再是“总要使用哪一层”，而是：

\[
\boxed{
\text{什么信息进入什么模块，}
\text{每个模块拥有何种中性元，}
\text{何种证据允许激活它，}
\text{激活后能声明到什么程度。}
}
\tag{18.2}
\]

---

# 附录 A：核心符号

| 符号 | 含义 |
|---|---|
| \(z_t\) | 未来目标窗口与当前锚点之差 |
| \(E_j\) | 通道多尺度编码 |
| \(K_j\) | 通道输入响应算子 |
| \(Q_j\) | 通道冻结基表示 |
| \(C\) | 通道融合 |
| \(q_t\) | 融合后潜变量 |
| \(W\) | Wiener 静态读出 |
| \(p_t\) | 冻结输入物理预测 |
| \(r_t^{OOF}\) | OOF 物理残差 |
| \(A\) | 状态或成熟残差预测器 |
| \(J\) | 联合预测模块 |
| \(R\) | 工况门控模块 |
| \(\pi_{j,m,r}\) | 通道尺度 profile |
| \(D_m\) | 标签/残差成熟延迟 |
| \(G_a\) | 装配对应 purge 距离 |
| \(\mathcal A\) | 候选装配集合 |

# 附录 B：从 v1.3 到 v2.0 的最小映射

| v1.3 对象 | v2.0 对象 |
|---|---|
| 通道专属多速率 profile | 保留为 E/K 接口 |
| Urysohn-first K | K + additive C + identity W |
| Mature residual AR | A 模块 |
| K-Joint AR | J 模块的一种实现 |
| Input-only leaderboard | A=0 的装配面 |
| Dynamic leaderboard | A 或 J 激活的装配面 |
| exact-zero channel | K 中性元 |
| exact-zero residual | A 中性元 |
| 线性 K | Urysohn 阶梯子模型 |
| 部署 LUT/FIR | 模块化部署编译 |

# 附录 C：首轮核心 benchmark 中暂不激活的扩展

下列对象属于 v2 理论接口，但不进入首轮核心装配搜索：

- 多工况混合专家 R；
- 在线 K 形状适配；
- 高阶全通道交互；
- 深度 W 或深度 J；
- 控制闭环主动激励模块；
- 将不确定性作为预测层。

这些对象必须在核心 K/C/W/A 装配得到稳定证据后另行预注册。
