# PRISM Theory v2.1.1
## Implementation-Safe Stagewise-Routed Physics-first Response Identification with Scale-specific Multirate Modular Operators
### 实现安全、逐级目标路由、物理优先、尺度专属、多速率模块化响应辨识理论体系

> **正式中文名**：PRISM v2.1.1 实现安全逐级目标路由物理优先尺度专属多速率模块化响应辨识理论  
> **正式英文名**：Implementation-Safe Stagewise-Routed Physics-first Response Identification with Scale-specific Multirate Modular Operators  
> **简称**：PRISM v2.1.1 / PRISM-SR-IS  
> **版本日期**：2026-08-06  
> **文档性质**：理论对象、模块接口、解释权边界、局部选择规则、可执行算法合同、数值不变量、联合预测路线、认证条件与部署合同  
> **直接理论母本**：`PRISM_Theory_v2_1_Stagewise_Routed_Modular_Assembly_Theory_Only.md`  
> **母本 SHA256**：`4b113cc9ff88a416b762a16a603ba477e1fef9f29760448e6da53e1c4e6e3575`  
> **上游理论母本**：`PRISM_Theory_v2_0_Modular_Assembly_Theory_Only.md`  
> **继承边界**：完整保留 v2.1 的 K→ΔW→A 逐级 OOF 目标路由、模块局部 guarded one-SE、可关闭软重叠惩罚、A-only 外置、Joint-KWA 非 AR-only 合同，以及 v2.0 的 E/K/C/W/A/J 模块代数、时间合同、通道专属多速率、Urysohn 阶梯、商空间/Schur 可辨识性和部署合同。  
> **核心修订**：把开发运行中暴露出的选择器与物化实现风险上升为正式算法不变量：profile one-SE 必须增加相对遗憾门；结构选择与数值正则必须分离；C 不得静默抹除已激活 K；W 准入必须尺度无关并允许部分 fold 不适用；PF 与 Joint 必须共享同一输入支路判定；选择损失、practical gate、物化预测和合同必须属于同一候选；Joint 必须真正联合拟合 K/W/AR 系数。  
> **理论边界**：本文不写入任何数据集排名、误差数值、具体通道胜负或经验 pass/fail；实际运行只能检验本文合同，不能反向修改理论。  
> **算法边界**：本文给出参考算法和必须满足的不变量，但不把某个 Python 文件、某个库版本或某个数据集专用常数定义为理论本身。所有数值阈值必须由配套冻结配置预注册。  
> **版本状态**：v2.1 的理论路线继续成立；v2.1.1 是实现与选择语义修正版本，不继承不满足本文不变量的旧实验结论。

> **Practice Contract Amendment（2026-08-08）**：实践执行语义补充如下。Physical-First 与 Joint 是两个分层证据路线。PF 是可以独立满足合同、独立冻结并进入正式 test/OOD 评价的结构路线；Joint 是建立在冻结输入支持与基构造规则之上的可选预测增强路线。Joint 未通过自身的 development 稳定性门时，不得进入正式预测、test 或 OOD，但这不否定已经通过自身全部合同的 PF。本补充只澄清冻结、装配和访问资格，不改变任何估计器、候选集合、超参数或判定阈值。

> **Practice Contract Amendment（2026-08-09，Sample Support）**：K 单通道审计与多通道装配的样本支持正式解耦。C1 只建立满足预测头自身目标/可用性/分割边界约束的最大许可 anchor universe；K 候选按自身历史长度取得 candidate-native fit support，并在同一通道的 local common scoring support 上进行公平选择；不同通道分别完成独立 K 审计后，C 才对已激活且已选择的通道取 assembly common support 并重拟合。该补充旨在避免短历史通道因其他通道或更长候选的历史需求而提前丢失合法训练样本，同时保持 one-SE、C 融合与后续 PF/Joint 比较的同支持可比性。它不改变任何已完成历史实验的证据含义；旧结果仍解释为 head-level common-support protocol 下的结果。

---

# 0. v2.1.1 的核心修订

## 0.1 停止的是 v2.0 的旧选择路线，不是 v2 的模块代数

PRISM v2.1.1 保留：

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

v2.1.1 继续停止：

1. 在 \(A\)、\(K\)、\(K+A\)、\(K+W+A\) 之间做一次全局 winner-takes-all one-SE；
2. 把 A-only 规定为比 K-only 更简单的物理装配；
3. 通过 \((I-P)\) 对 W、A 特征进行硬投影并把该投影当作主要解释权保障；
4. 允许下游 W/A 的不稳定反向撤销已经通过独立门槛的 K；
5. 把 Joint 路线中的 \(K=0\)、AR-only 作为可选最终 PRISM 装配。

因此 v2.1.1 的结构是：

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

v2.1.1 继续将 A-only 移出物理装配格：

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

v2.1.1 的主要归因方式为：

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

v2.1.1 不在所有装配之间做单次全局 one-SE。选择作用域分别为：

\[
\mathcal A_K,\quad
\mathcal A_{C|K},\quad
\mathcal A_{W|K,C},\quad
\mathcal A_{A|K,C,W},\quad
\mathcal A_J.
\tag{0.18}
\]

每个模块只与同层中性元和同层复杂度阶梯比较。W/A 不能反向撤销 K。

## 0.6 v2.1.1 的正式装配集合

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


## 0.7 v2.1.1 的六条实现不变量

### 不变量 I：结构选择与数值稳定分离

profile、K family、C representation、W/A 是否激活属于结构选择；ridge、jitter、QR/SVD rescue 只属于数值稳定。不得把“更大的 ridge”解释为“更简单的物理结构”，也不得通过超强正则把已选择的输入支路静默缩成零。

### 不变量 II：已激活 K 的预测路径必须被保存

若至少一个通道已通过 K 局部门，则后续 C 必须产生可检测的非零 OOF 输入预测，或者显式回退到已通过的 K 预测。不存在“结构标签仍为 active，但实际输入预测近常数”的合法状态。

### 不变量 III：W 必须真正进入候选比较

W 的准入不能依赖原始尺度上的单一绝对方差阈值。单个 fold 退化只允许记为 `NOT_APPLICABLE`；满足冻结的最少可用 fold 数时，非线性 W 候选必须实际拟合、产生 fold loss 并参加 guarded one-SE。

### 不变量 IV：PF 与 Joint 共享输入支路判定函数

PF 和 Joint 不得分别以“active channel 数量”和“预测方差”定义输入是否存在。二者必须调用同一个、基于实际 OOF 输入贡献的判定函数，并对同一输入预测给出同一状态。

### 不变量 V：选择对象与物化对象一致

每个候选必须拥有唯一 `candidate_id`。one-SE、practical activation、最终合同、fold losses、OOF prediction 和 validation/test prediction 必须由同一 `candidate_id` 关联。任何 gate 后改成 neutral、却继续沿用 gate 前复杂候选损失的实现均为硬错误。

### 不变量 VI：Joint 中 W 必须以基系数参与联合优化

Joint-KWA 可以固定 K profile、K 支持和 W basis construction rule，但不得先得到一个预拟合的 `kw_scalar` 再把它当作单列特征冒充联合 W。正式 Joint 必须在同一目标函数中联合估计 K block、W basis coefficients 与 AR block。

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

通道 joint basis 可以保留。高阶 pairwise interaction 不作为首轮 v2.1.1 核心搜索的必要组成，但理论接口保留。

## 2.3 Wiener 读出修正

v2.1.1 将 W 明确写成 identity 基线加残差修正：

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

## 3.1 通道专属 profile 与遗憾保护

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

仅使用 one-SE 可能在 fold 波动较大时接受风险明显更高的极短历史。v2.1.1 因而定义 profile 双门。设最佳 profile 为 \(p_*\)，平均风险为 \(R_*\)，其标准误为 \(SE_*\)。profile \(p\) 只有同时满足

\[
R(p)\le R_*+SE_*,
\tag{3.1a}
\]

和

\[
\operatorname{Regret}(p)
=
\frac{R(p)-R_*}{\max(R_*,\varepsilon_R)}
\le \delta_{profile}
\tag{3.1b}
\]

才可进入可接受集合。\(\delta_{profile}\) 由冻结配置预注册；首轮实现合同可采用 0.02，但该数值不由理论正文永久固定。

每个通道最多保留：

1. 平均风险最小的 profile；
2. 双门集合内与最佳 profile 不同的最简单 profile。

K family、resolution 与 penalty 必须在保留 profile 上联合比较。禁止先无条件选最短历史，再在该单一 profile 上完成全部 K 结构选择。

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

## 3.4 固定支持最小稳定重拟合

K 的 profile、rank 与 active support 冻结后，应在每个 OOF fit fold 内进行固定支持重拟合：

\[
\widehat\theta_K^{refit}(\lambda)
=
\arg\min_{\theta\in\mathcal S_K}
\|z-\Phi_K\theta\|_2^2
+
\lambda\|\theta\|_2^2,
\tag{3.6}
\]

其中 \(\mathcal S_K\) 是冻结支持。设冻结的非降序 ridge 网格为

\[
0\le \lambda_1<\lambda_2<\cdots<\lambda_L.
\tag{3.6a}
\]

定义数值可接受集合

\[
\Lambda_{safe}
=
\{\lambda_\ell:
\text{KKT、条件数、有效秩、有限系数和幅值门全部通过}\}.
\tag{3.6b}
\]

最终取

\[
\lambda_{refit}=\min\Lambda_{safe}.
\tag{3.6c}
\]

若 \(0\in\Lambda_{safe}\)，必须选择 \(\lambda_{refit}=0\)。ridge 不进入结构复杂度 tie-break，也不得以“更强正则更简单”为由选择最大值。

若 \(\Lambda_{safe}=\varnothing\)，该固定支持候选标记为 `NUMERICALLY_INVALID`，不得通过选择超强 ridge 把其伪装成 exact-zero。该规则的目的不是提高自由度，而是避免 W/A 仅仅恢复被 K/C 强 ridge 压缩掉的幅值。

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


## 3.7 C 表示选择与输入支路保存门

C 的 one-SE 只在表示族之间比较，不在 ridge 强度之间定义结构偏序。对每个表示族 \(c\)，先按式 (3.6a)–(3.6c) 得到最小稳定 ridge，再生成严格 OOF 预测 \(p_C^{OOF}\)。

设开发 OOF 上最佳已激活单通道 K 预测为 \(p_{bestK}^{OOF}\)，目标为 \(z\)。定义

\[
v_C=
\frac{\operatorname{Var}(p_C^{OOF})}
{\max(\operatorname{Var}(z),\varepsilon_v)},
\qquad
v_K=
\frac{\operatorname{Var}(p_{bestK}^{OOF})}
{\max(\operatorname{Var}(z),\varepsilon_v)}.
\tag{3.8}
\]

C 表示只有同时满足以下条件才可作为正式输入支路：

\[
v_C\ge \max(v_{abs}^{min},\rho_v v_K),
\tag{3.9}
\]

\[
R(p_C^{OOF})
\le
(1+\delta_C)R(p_{bestK}^{OOF}),
\tag{3.10}
\]

并且至少一个非截距系数超过冻结的数值零阈值，且数值证书通过。\(v_{abs}^{min},\rho_v,\delta_C\) 由配置冻结；实现建议值可分别为 \(10^{-8},0.1,0.02\)，但实验不得在查看 test 后调整。

若 joint-basis 表示未通过，允许回退 compressed；若所有 C 表示均未通过，但 \(p_{bestK}^{OOF}\) 已通过 K 门，则回退：

```text
C_FALLBACK_TO_BEST_ACTIVE_K
```

该回退保存 K 的预测路径，但不授予 C 成功证书。若结果记录 active K，却同时产生未显式回退的近零 C 输出，必须硬失败：

```text
C_INPUT_PATH_COLLAPSE_BUG
```

C 不得通过 ridge、标准化错误、列裁剪或候选物化差异静默撤销 K。

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

## 4.2 W 的 fold 可用性与候选

W 只能建立在实际非坍缩的 K/C 潜变量 \(q\) 上，但“非坍缩”必须按相对数值尺度定义，而不是使用固定原始方差阈值。

对 fold \(f\)，定义

\[
\tau_f
=
\kappa_{eps}\,\epsilon_{64}
\max\{1,\|q_f\|_\infty\},
\tag{4.3}
\]

其中 \(\epsilon_{64}\) 为 float64 machine epsilon，\(\kappa_{eps}\) 由冻结配置给出。fold \(f\) 只有同时满足：

1. 在容差 \(\tau_f\) 下至少有冻结数量的不同有限 latent 值；
2. \([\mathbf 1,q_f]\) 的数值秩为 2；
3. \(sd(q_f)>\tau_f\)；
4. fold-fit 标准化后 W 基全部有限且求解证书可计算；

才记为 `USABLE`。退化 fold 记为 `NOT_APPLICABLE`，不得用零损失、无穷损失或 identity 损失伪装成已比较的非线性 W。

只要可用 fold 数达到冻结下限，W 候选仍进入比较；首轮四折实现可要求至少 3 个可用 fold。若全部 K 为 exact-zero，或统一输入支路保存门失败，则正式 W 必须为 identity。

W 中性候选为

\[
\Delta W_0=0.
\tag{4.3a}
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

v2.1.1 不要求：

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


## 4.6 W 的正式结果与救援诊断分离

W 的非零候选只有在 C 输入支路保存门通过后，才可获得 `W_RESIDUAL_VALIDATED` 或更高状态。

若 C 未通过保存门，但研究者为了定位过度收缩问题而在标准化 latent 上运行 W，该结果只能标记：

```text
W_RESCUE_DIAGNOSTIC_ONLY
```

它可以回答“非线性读出能否恢复被压缩的排序信息”，但不能回答“冻结物理潜变量之后是否存在稳定 Wiener 曲率”。救援诊断不得进入 PF 正式装配、test 选择或物理证书。

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

首轮 v2.1.1 核心路线以线性 mature residual AR 为主，以减少重新引入高方差状态捷径。

## 5.4 不再硬投影 A 特征

v2.1.1 不使用：

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


## 5.7 A 的候选—损失—预测一致性

A 的 practical activation 可能把 one-SE 初选候选重置为 exact-zero。实现必须在 gate 完成后重新绑定：

- `final_selected_candidate`；
- `final_selected_fold_losses`；
- `final_selected_contract`；
- `final_selected_oof_prediction`；
- `final_selected_validation_prediction`。

若最终候选为 exact-zero，则 fold losses 必须是 exact-zero 候选的损失，物化 residual prediction 必须逐样本为零。禁止保留复杂 A 的 fold losses，却物化 exact-zero 预测。

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

1. 计算全部注册 profile 的 fold-local 风险；
2. 使用 one-SE + 相对遗憾双门，最多保留两个 profile；
3. 在保留 profile 上联合比较 exact-zero、linear/rank/full 阶梯；
4. 选择后固定支持，并使用最小稳定 ridge 生成 OOF K 预测；
5. 输出 numerical certificate、Gram/Schur、common support、outer stability 与 conditional novelty diagnostic；
6. 任何 profile、basis、scaler 和 penalty 均不得读取 evaluation fold。

## 6.3 Stage P2：K/C 固定支持联合重拟合

冻结通道 profile、基和 active support 后，在兼容目标头内拟合 C。one-SE 只选择 C 表示族；每个表示族内部使用最小稳定 ridge。

生成 OOF \(p_C\) 后必须执行输入支路保存门。未通过时按预注册顺序回退 compressed 或 `BEST_ACTIVE_K_CHANNEL`，不得继续把近零 C 交给 W/A。

输出：

- OOF \(p_K\) 与最终 OOF 输入预测；
- validation 输入预测；
- 通道贡献；
- K/C contracts；
- C 保存门各项统计；
- local one-SE audit；
- 明确的 fallback 或 hard-failure 状态。

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


## 6.8 PF 参考算法

```text
INPUT: frozen data contract, target head, K/C/W/A candidate grids
OUTPUT: PF artifact or PHYSICS_ROUTE_NOT_SUPPORTED

for each outer/inner fold:
    fit_scalers_and_bases_on_fit_only()

    # K
    for channel in registered_channels:
        profile_losses = evaluate_registered_profiles(channel)
        retained_profiles = one_se_plus_regret_guard(profile_losses, max_profiles=2)
        k_candidate = guarded_select_K_over_profiles_and_families(retained_profiles)
        k_oof[channel] = fixed_support_refit_with_smallest_stable_ridge(k_candidate)

    active_K = channels_passing_K_gate(k_oof)
    if active_K is empty:
        emit K_EXACT_ZERO and PHYSICS_ROUTE_NOT_SUPPORTED
        do not fit formal W/A PF assembly

    # C
    c_candidates = fit_C_representations(active_K, ridge_semantics="stability_only")
    c_selected = select_family_then_apply_input_preservation_gate(c_candidates)
    if no C candidate preserves input:
        c_selected = BEST_ACTIVE_K_CHANNEL
        emit C_FALLBACK_TO_BEST_ACTIVE_K
    assert shared_input_path_gate(c_selected.oof_prediction).valid

    # W
    r1 = y - c_selected.oof_prediction
    usable_folds = audit_W_fold_usability(c_selected.oof_latent)
    w_selected = guarded_select_W(r1, usable_folds, mu_grid_including_zero)
    if C was not formally preserved:
        mark W as diagnostic_only
    freeze W

    # A
    r2 = r1 - w_selected.oof_correction
    mature_features = build_from_latest_available_target_index(r2)
    a_selected = guarded_select_A(r2, mature_features, fold_local_centering=True)
    bind_final_candidate_loss_prediction_contract(a_selected)

materialize PF prediction from exactly the bound K/C/W/A contracts
write immutable assembly card and hashes
```

该算法中的 `assert` 不是调试建议，而是理论检验前置条件。断言失败时必须停止当前头，不能自动改成 A-only。

---

# 7. Joint-KWA 联合预测路线

## 7.1 目标

Joint 路线回答：

> 在相同输入 profile、W 基族和 AR profile 下，允许 K、W、AR 在共享目标空间中相互侵占和调节时，能达到怎样的预测上限？

Joint 是 optional predictive enhancement，而不是 PF 有效性的必要条件。PF 的正式结构路线为

```text
K -> C -> W -> A
```

Joint 的候选路线为 `J_K / J_KW / J_KA / J_KWA`，其职责是在冻结输入支持与基构造规则下评估联合预测上限。因此二者的证据关系是分层并列关系，而不是串联必要关系：

\[
\mathrm{PF\_VALID}\not\Rightarrow\mathrm{JOINT\_VALID},
\qquad
\mathrm{JOINT\_INVALID}\not\Rightarrow\mathrm{PF\_INVALID}.
\tag{7.0a}
\]

`PF_pass = true` 可以独立产生正式模型；`Joint_pass = true` 只决定 Joint 是否附加成为正式 predictive candidate。

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


## 7.2.1 W 基的正确联合方式

在 Joint 中，允许先用冻结 K/C 规则构造 seed latent \(q^{seed}\)，并据此冻结 W 的 knot 与基函数映射：

\[
B^W=B_W(q^{seed}).
\tag{7.2a}
\]

但 \(\gamma_W\) 必须与 \(\beta_K,\beta_A\) 在式 (7.2) 中共同求解。禁止流程：

```text
先单独拟合 W → 得到 kw_scalar → 把 kw_scalar 作为固定一列加入 Joint
```

因为该流程只允许 Joint 调节一个预拟合输出的缩放，不能让 W basis 与 K、AR 真正相互侵占。

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

若 Joint 的协议、候选绑定与数值求解均合法，但该预测门未通过，则标记：

```text
JOINT_NOT_SUPPORTED_ON_DEVELOPMENT
```

该状态只表示当前 Joint estimator 在注册 development 协议下没有获得足够稳定的预测证据；它不是 `PHYSICS_ROUTE_NOT_SUPPORTED`，也不是 `PF_FAILED`。此时不得自动回退 AR-only、改成 K-zero、放宽阈值或把 Joint 送入 test/OOD 再决定是否恢复。仍然不把 AR-only 作为 Joint 的替代结果。真正的几何、系数或数值失败继续保留各自明确的失败分类。

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


## 7.6 Joint-KWA 参考算法

```text
INPUT: frozen K profiles/support, frozen C representation rule,
       frozen W basis construction rule, frozen AR profile

for each selection fold:
    build Phi_K on fit only
    build q_seed and W knots on fit only
    build B_W(q_seed) without pre-fitting gamma_W
    build legal H_Y using latest available target index

    evaluate only J_K, J_KW, J_KA, J_KWA
    jointly solve each registered block model
    store block predictions and total prediction separately

select within route-local guarded one-SE
compute g_U = Phi_K beta_K + B_W gamma_W
apply the exact same shared_input_path_gate used by PF

if gate fails:
    emit JOINT_INPUT_PATH_COLLAPSED
    do not return AR-only
else:
    bind selected candidate, losses, contract and predictions by candidate_id
```

AR-only 可以单独计算条件新颖性诊断，但不得出现在 `candidates`、one-SE acceptable set 或最终 Joint 装配卡中。

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

这与 v2.0 “只要 neutral 在 one-SE 集合就立即选择 neutral”不同。v2.1.1 允许一个统计可接受、具有预注册实用增益且折间方向稳定的非中性候选被保留。

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


## 8.5 Profile 的 one-SE 不等于无限制简化许可

profile 使用式 (3.1a)–(3.1b) 的双门。one-SE 控制折间不确定性，相对遗憾门控制平均风险的实质退化。只有同时通过二者，复杂度偏序才可以选择更短历史或更粗分辨率。

## 8.6 Ridge 不属于模块复杂度偏序

对固定表示和固定支持，ridge 是求解路径，不是结构模块。因而：

- 不把更大 \(\lambda\) 排在更小 \(\lambda\) 前作为“更简单”；
- 不把 \(\lambda\to\infty\) 产生的近零预测称为中性元；
- exact-zero 必须由显式 exact-zero 候选产生；
- ridge 只通过 `SMALLEST_LAMBDA_PASSING_CERTIFICATES` 选择。

该规则适用于 K 固定支持重拟合、C 表示拟合，以及其他声明 ridge 仅用于数值稳定的模块。

---

# 9. 可辨识性与解释权

## 9.1 总预测与模块证书分离

v2.1.1 区分：

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
- 每 fold latent 可用性与 `NOT_APPLICABLE` 原因；
- 最少可用 fold 数；
- 非线性候选确实被拟合并产生 loss 的证据；
- shape/monotone constraint；
- effective degrees of freedom；
- support/extrapolation；
- soft overlap audit；
- C 输入支路保存门状态；
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

Joint 不授予物理证书。其合同必须证明：

- 候选集合中不存在 K-zero、AR-only 或 both-zero；
- W basis coefficients 与 K/AR block 共同优化；
- PF 与 Joint 调用同一输入支路 gate 版本与参数；
- block prediction 与 total prediction 均被落盘审计。

Joint 的合法输出状态包括：

- `JOINT_INPUT_PATH_VALIDATED`
- `JOINT_PREDICTIVE_VALIDATED`
- `JOINT_NOT_SUPPORTED_ON_DEVELOPMENT`
- `JOINT_PREDICTIVE_UNSTABLE`
- `JOINT_OOD_UNSTABLE`

真正的数值、结构和协议失败状态继续保留。只有 Joint 自己通过 development predictive gate，才能生成正式 Joint freeze certificate 和 test-eligible Joint candidate ID。development gate 未通过时可以保存诊断 artifact、已选 diagnostic route 和 fold loss，但不得把它们注册为正式预测候选。

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
| `C_FALLBACK_TO_BEST_ACTIVE_K` | C 未获证书，但保存已通过的 K 路径 |
| `C_INPUT_PATH_COLLAPSE_BUG` | active K 被 C 静默压成近零，硬失败 |
| `W_RESCUE_DIAGNOSTIC_ONLY` | W 仅用于诊断上游过度收缩，不进入正式装配 |
| `PF_ONLY_FROZEN` | PF 已通过全部正式合同并冻结；Joint 未获 development 支持，不进入正式预测候选 |
| `PF_AND_JOINT_FROZEN` | PF 与 Joint 均通过各自 development 合同，两条路线均被冻结 |
| `JOINT_NOT_SUPPORTED_ON_DEVELOPMENT` | Joint development 稳定性证据不足；不影响合法 PF |
| `PF_JOINT_INPUT_GATE_INCONSISTENT` | 同一 gate version、参数、input prediction/hash、target 与 best-K comparator 却产生不同 gate 结果，属于硬实现失败 |
| `NOT_APPLICABLE` | 某 fold/候选因数值退化不适用，不伪装成 neutral |
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

v2.1.1 的核心 W/A 不做硬投影。仍禁止构造大型 \(I-P\)。

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


# 11A. 可执行算法合同

## 11A.1 最小稳定 ridge 选择器

```python
# 伪代码；具体证书阈值由冻结配置提供
def select_smallest_stable_ridge(X_fit, y_fit, lambda_grid, certificate_cfg):
    for lam in sorted(lambda_grid):
        model = solve_fixed_support(X_fit, y_fit, ridge=lam)
        cert = certify(model, X_fit, y_fit, certificate_cfg)
        if cert.hard_pass:
            return model, lam, cert
    return NUMERICALLY_INVALID
```

必须从小到大扫描；不得先按 validation loss 找最佳 ridge，再因 one-SE 选择最大 ridge。若结构选择确实需要 ridge bias-variance 调节，则必须把它注册成另一条预测候选族，不能仍声称其语义只是数值稳定。

## 11A.2 统一输入支路 gate

实现中必须只有一个权威函数，例如：

```python
def evaluate_input_path(y, input_pred, best_k_pred, nonintercept_coef, cfg):
    var_y = max(var(y), cfg.variance_floor)
    ratio = var(input_pred) / var_y
    best_ratio = var(best_k_pred) / var_y
    mse_ok = mse(y, input_pred) <= (1 + cfg.max_mse_regret) * mse(y, best_k_pred)
    variance_ok = ratio >= max(cfg.abs_ratio_min, cfg.best_k_fraction * best_ratio)
    coef_ok = max_abs(nonintercept_coef) >= cfg.coef_abs_min
    return GateResult(valid=variance_ok and mse_ok and coef_ok,
                      variance_ratio=ratio,
                      best_k_variance_ratio=best_ratio,
                      mse_ok=mse_ok,
                      coefficient_ok=coef_ok)
```

PF 的 C 输出与 Joint 的 \(g_U\) 均调用该函数。若输入对象不同，可以传入不同 `input_pred`，但阈值、标准化、方差 floor 和状态语义必须一致。“共享 gate”只表示 same implementation、same configuration、same numerical semantics；不要求 PF 与 Joint 的 PASS/FAIL 结果相同，因为 \(g_C\neq g_U\) 时两条路线本来就可以得到不同结果：

\[
\boxed{\text{same gate contract}\neq\text{same gate outcome}.}
\tag{11A.0}
\]

只有 gate version、参数、被评估 prediction/hash、target 和 best-K comparator 全部相同，却返回不同结果时，才构成 `PF_JOINT_INPUT_GATE_INCONSISTENT`。

## 11A.2A 分层冻结实践合同

```text
PF_OK = (
    K/C input path valid
    and W contract valid
    and A contract valid
    and PF candidate binding valid
)

JOINT_OK = (
    Joint protocol valid
    and Joint candidate family valid
    and Joint W truly jointly fit
    and Joint predictive/input gate valid
    and Joint candidate binding valid
)

if not PF_OK:
    PRISM route unsupported
    STOP before test

if PF_OK and not JOINT_OK:
    freeze PF
    joint_status = JOINT_NOT_SUPPORTED_ON_DEVELOPMENT
    formal_routes = [PHYSICS_FIRST]

if PF_OK and JOINT_OK:
    freeze PF
    freeze Joint
    formal_routes = [PHYSICS_FIRST, JOINT]
```

其中 Joint 的协议、候选族、真实联合拟合、数值证书或候选绑定失败仍是硬实现/协议失败；只有这些条件均合法而 Joint 自己的 development predictive gate 未通过时，才应用 PF-only freeze。严禁用 `PF_OK and not JOINT_OK` 作为把 Joint 送入 test/OOD 再决定的理由。M7/M8 的物化集合必须完全由冻结清单中的 `formal_routes` 决定。

## 11A.3 候选对象的不可变数据结构

每个候选至少包含：

```text
candidate_id
stage
family
hyperparameters
fold_ids
fold_losses
usable_fold_mask
oof_prediction_path
evaluation_prediction_path
contract_path
numerical_certificate_path
activation_decision
artifact_sha256
```

`one_se.selected` 只是中间字段。最终结果必须另写 `final_selected_candidate_id`，并由该 ID 解析全部损失、预测与合同。不得在内存中修改 `selected` 字符串而不重建关联对象。

## 11A.4 W 可用性审计参考实现

```python
def w_fold_is_usable(q, cfg):
    q = finite_values(q)
    if len(q) == 0:
        return False, "NO_FINITE_LATENT"
    scale_floor = cfg.eps_multiplier * finfo(float64).eps * max(1.0, max(abs(q)))
    if tolerant_unique_count(q, tol=scale_floor) < cfg.min_distinct:
        return False, "INSUFFICIENT_DISTINCT_LATENT"
    if std(q) <= scale_floor:
        return False, "RELATIVE_SCALE_DEGENERATE"
    if numerical_rank(column_stack([ones(len(q)), q])) < 2:
        return False, "INTERCEPT_LATENT_RANK_DEFICIENT"
    return True, "USABLE"
```

候选 loss 的分母只使用共同可用 folds。identity 和非线性 W 必须在相同可用 fold 集上配对比较，避免因缺失模式不同产生伪增益。

## 11A.5 C 回退算法

```text
evaluate all registered C representations with smallest stable ridge
remove candidates failing input-path preservation
if at least one remains:
    apply family-local one-SE and deterministic representation tie-break
else if a K channel is formally active:
    return BEST_ACTIVE_K_CHANNEL with C_FALLBACK_TO_BEST_ACTIVE_K
else:
    return PHYSICS_ROUTE_NOT_SUPPORTED
```

回退预测必须直接复用最佳 active K 的 OOF/validation 样本 ID，不得在回退时重新搜索通道或参数。

## 11A.6 A 成熟特征构造

```python
def build_mature_residual_features(samples, residual_series, offsets):
    # latest_available_target_index 已由数据合同冻结
    latest = samples.latest_available_target_index
    for sample_i, entity in grouped_by_entity(samples):
        for lag in offsets:
            query_index = latest[sample_i] - lag
            use residual only when it exists in the same entity and same allowed fold history
```

不得使用全局 residual mean；中心和尺度必须在每个 fit fold 内估计，并原样应用于 evaluation fold。

## 11A.7 真实 Joint-KWA 设计矩阵

正式 Joint 设计矩阵为

\[
X_J=[\Phi^K,\;B_W(q^{seed}),\;H^Y].
\tag{11A.1}
\]

对应块 penalty 为

\[
\Lambda_J=
\operatorname{diag}
(\lambda_K I_K,\lambda_W R_W,\lambda_A I_A).
\tag{11A.2}
\]

一次求解得到 \((\beta_K,\gamma_W,\beta_A)\)。`kw_scalar` 可以作为消融或工程特征，但不得命名为 `JOINT_KW` 的正式实现。

## 11A.8 资源与缓存合同

- fold 结束后释放大型 design matrix；
- 不同时缓存全部 profile × family × penalty 的稠密矩阵，优先缓存可复用充分统计或按候选流式重建；
- 多进程与 BLAS 线程不得无限嵌套；
- 每个 worker 独立读取只读数据，禁止复制完整共享数据 base 后再切片；
- 任何 row cap、列裁剪或 QR reduction 必须 train-only，并写入合同；
- 资源降级不得改变候选语义，无法完成时标记 retained numerical failure。

## 11A.9 强制断言

开发阶段至少包含：

```text
assert selected_loss_candidate_id == selected_prediction_candidate_id
assert selected_contract_candidate_id == selected_prediction_candidate_id
assert no_test_access_before_freeze
assert PF_input_gate.version == Joint_input_gate.version
assert not(active_K and near_zero_C and no_explicit_fallback)
assert not(Joint_candidate_is_AR_only)
assert W_nonzero_candidate_has_actual_fold_losses
assert mature_residual_uses_latest_available_target_index
```

断言失败属于实现失败，不属于模型负结果。

## 11A.11 Joint 预测稳定性实践 Tips

本节是 PRISM v2.1.1 的实践／可执行算法合同，不建立平行理论版本，也不改变 K/C/W/A、数据划分、Joint route 集合或输入门阈值。

1. Joint 的 `numerical_alpha` 与 `predictive_eta` 必须区分：前者只负责数值稳定证书，后者负责 bias--variance 与预测稳定性；预测正则不得伪装成 numerical rescue。
2. 在平均损失尺度下，预测 penalty 为 \(P_{pred}=n_{fit}\eta_{pred}I\)。
3. `predictive_eta=0` 必须作为精确中性边界。
4. Joint K representation 可以是 `CHANNEL_COMPRESSED` 或 `FULL_BASIS`，但两者共享完全相同的 frozen raw K support。
5. `CHANNEL_COMPRESSED` 是每个 active K channel 一列，而不是最终 C scalar。
6. `FULL_BASIS` 使用 frozen K basis 的合法内部列，不得复活 K exact-zero channel。
7. `FULL_BASIS` 只有在 registered OOF evidence 支持额外自由度时才可击败 compressed representation。
8. 每个 fold 的 `fit_physical_features()` 只运行一次；compressed/full 从同一 feature construction 提取。
9. M2--M4 只有在 SHA 与 provenance 完全一致时才可复用；Joint stability 实践不得触发 K/C/W/A 重算。
10. legacy Joint anchor 必须复现，以证明实践 estimator 未改变已修正的原始四折 OOF protocol。
11. worst-fold loss 是强制 stability diagnostic，不能只看平均 fold loss。
12. coefficient norm、effective degrees of freedom 与 prediction variance 应同时保存为 Joint stability diagnostics。
13. Joint stability diagnosis 是 development predictive evidence，不是物理因果解释。
14. PF independent freeze semantics 继续成立。
15. Joint 通过可产生 `PF_AND_JOINT_FROZEN`；Joint 不通过仍可产生 `PF_ONLY_FROZEN`。
16. test/OOD 不得用于选择 eta、K representation、Joint route、threshold 或 candidate family。
17. M6 必须发生在全部代码、理论和 config 修正之后；M6 后 source/config/theory 不得变化。
18. M7 前必须执行不读取 test/OOD 的 lockbox preflight。
19. M7/M8 的 formal materialization set 只能来自 M6 freeze manifest。
20. estimator dispatch 不得依赖模型版本字符串，必须依赖显式 estimator semantics contract。
21. 正式语义为 `joint_estimator_semantics=PREDICTIVE_STABILITY_RIDGE_R1`；该字段要求按 representation、numerical alpha 与 predictive eta 重建 Joint。
22. development 选出的 eta 与 representation 在 M7 只能重放，不得重新搜索。
23. M7 只拟合 frozen hyperparameters；predictive eta 在预测阶段不得再次 shrink。
24. M7 一旦打开 test/OOD，不得再因指标结果修改 estimator 或 config。
25. 正式 primary 比较 `J_SELECTED` vs `PF_SELECTED` 必须在 M7 前注册。

因此总 penalty 明确分解为

\[
P=P_{num}+n_{fit}\eta_{pred}I,
\]

而 `same_gate_contract` 不要求不同 estimator prediction 得到 `same_gate_outcome`。Joint 仍是 PF 之外的可选 predictive enhancement。


## 11A.12 Sample-Support Practice Contract：Native-Support K Audit + Common-Support Assembly

本节规定 PRISM v2.1.1 的样本支持执行语义。它不建立新的理论版本，不改变 K/C/W/A/J 的数学对象、候选族、损失函数或选择阈值；它只规定“哪些合法时间锚点可以用于哪个候选的拟合、哪些时间锚点可以用于候选间比较、何时才必须取多通道交集”。

### 11A.12.1 四类支持对象

对给定预测头和一个连续合法区间，先定义不含 K 候选历史要求的最大许可锚点集合

\[
\mathcal S^{anchor}
=
\{t:\ t\text{ 满足 split/purge、目标窗口、availability 与当前目标窗口 }W_0\text{ 的合法性}\}.
\tag{11A.3}
\]

C1 的职责是物化 \(\mathcal S^{anchor}\)，而不是预先对所有可能 K profile 的历史长度取最大值。若区间左端加 purge 后的第一个允许依赖位置为 \(b\)，则 anchor 最早只需满足

\[
t-W_0\ge b,
\tag{11A.4}
\]

而不是满足所有未来可能 K 候选的最大历史长度。

对 K 候选 \(c=(j,p,f,\ldots)\)，其中通道为 \(j\)，profile 历史长度为 \(L_c\)，定义 candidate-native support

\[
\mathcal S_c^{native}
=
\{t\in\mathcal S^{anchor}:t-L_c\ge b\}.
\tag{11A.5}
\]

候选 \(c\) 的 fit fold 必须尽可能使用其自身 \(\mathcal S_c^{native}\) 内的合法训练锚点。一个短历史候选不得因为另一通道或同通道另一长历史候选的存在而失去本来合法的训练样本。

同一局部选择集合 \(\mathcal C_j\)（例如同一通道的 profile/family 候选）用于 one-SE 或 guarded one-SE 比较时，定义 local common scoring support

\[
\mathcal S_j^{score}
=
\bigcap_{c\in\mathcal C_j}\mathcal S_{c,eval}^{native}.
\tag{11A.6}
\]

各候选可以在不同的 candidate-native fit support 上拟合，但必须在相同的 \(\mathcal S_j^{score}\) 上计算配对 validation loss。由此同时满足

\[
\boxed{\text{native-support fitting}+\text{common-support scoring}.}
\tag{11A.7}
\]

当若干通道已分别完成 K 审计并激活，设其最终选择的 K 候选为 \(c_1^*,\ldots,c_m^*\)。只有进入 C 装配时才定义 assembly common support

\[
\mathcal S^{assembly}
=
\bigcap_{r=1}^{m}\mathcal S_{c_r^*}^{native}.
\tag{11A.8}
\]

C 及依赖完整多通道输入向量的后续路线只能在 \(\mathcal S^{assembly}\) 上拟合、评分和物化。

### 11A.12.2 K 审计的实践规则

1. C1 不得再用 head-level 最大注册 K history 作为所有 `sample_ids` 的统一左边界；C1 必须建立 maximally permissive anchor universe。
2. anchor universe 仍必须满足 `w0_steps`、预测 horizon/window、availability delay、split/purge 和连续区间边界；“最大许可”不等于允许跨越因果边界。
3. `base_origin_id` 的核心身份仍由 dataset/entity/head/origin 决定；support mask 不能通过重新编号 origin 改变样本身份。
4. 每个 K profile/family 候选在 fit fold 内先应用自己的 native mask，再应用任何 deterministic row cap。row cap 不能先于 native mask，否则新增合法样本可能永远无法进入拟合。
5. 同一通道不同 profile 可以拥有不同 native fit rows；短 profile 应获得它额外合法的较早训练锚点。
6. 同一局部 selection set 的 evaluation rows 必须取 local common scoring support，从而保持 fold loss 严格配对。
7. `EXACT_ZERO` 与非零 K 候选比较时必须使用同一 local common scoring rows；exact-zero 不得因使用更宽或更容易的 evaluation support 获得优势。
8. 各通道的 K activation 是通道局部门：通道 \(j\) 与自己的 neutral candidate 比较，不要求为了另一通道的长历史而预先收缩 \(\mathcal S_j^{native}\)。
9. 不得直接用来自不同 channel-native scoring support 的 K-stage raw MSE 对多个通道做 winner-takes-all 排名。
10. 每个 K RESULT 至少记录：selected profile、native fit row count、local scoring row count、native support 起点、scoring-support hash、fold-wise support hashes 与 row-cap-after-mask 语义。

### 11A.12.3 C 装配与 best-K comparator

当进入 C 时，所有已激活通道的最终 K 候选必须首先映射到同一个 \(\mathcal S^{assembly}\)。此后：

1. C 的 compressed/joint-basis 拟合只读取 assembly common support；
2. `best_active_k_channel` 不得复用不同 native support 上的 K-stage loss 排名；必须在 \(\mathcal S^{assembly}\) 上重新计算各 active selected-K 的 prediction/loss 后再确定 comparator；
3. C input-path preservation gate 的 `best_k_pred` 必须来自同一 assembly common support；
4. C fallback 到 `BEST_ACTIVE_K_CHANNEL` 时，fallback prediction 必须与该 common-support comparator 逐样本对齐；
5. assembly support 的样本 ID 顺序、hash 与行数必须写入 C contract，避免不同模块悄悄使用不同交集。

因此不同通道可以在 K 阶段获得各自完整的合法训练支持，但一旦它们作为一个多通道对象进入 C，所有结构比较重新恢复严格同支持语义。

### 11A.12.4 W、A 与 Joint 的支持继承

- W 继承 C 的 assembly support；W 不得为了自身基函数重新扩大 raw-input support。
- PF 中 A 在 assembly support 基础上继续施加 target-history maturity、availability delay 与 `latest_available_target_index` 约束，因此 A 的合法支持可以进一步缩小，但不能越过 C 的输入合法边界。
- Joint 必须继承冻结的 active K support、K representation 规则和 assembly common support。`CHANNEL_COMPRESSED` 与 `FULL_BASIS` 只改变表示自由度，不得改变 raw sample support。
- Joint 的 J_K/J_KW/J_KA/J_KWA 候选在同一 formal comparison 中必须保持注册的 common evaluation support；不能通过不同 support 获得表面预测优势。

### 11A.12.5 Fold、mask 与可复现性合同

推荐执行顺序为：

```text
C1 maximal anchor universe
    -> registered temporal folds
    -> candidate-native fit mask inside each fold
    -> channel-local common evaluation mask
    -> K local selection
    -> selected-K refit on its full native fit support
    -> active selected channels
    -> assembly common support
    -> C -> W -> A / Joint
```

实践中必须满足：

1. temporal fold 边界先注册，candidate mask 只能在 fold 内删除不合法 rows，不能移动 fold 边界以追求更好结果；
2. support mask 必须由结构合同和时间索引确定，不能读取目标误差后自适应改变；
3. fit/evaluation support 必须分别 hash；任何 one-SE loss 都应能够追溯到明确的 scoring-support hash；
4. deterministic subsampling/row cap 在 support mask 之后执行，并以稳定样本 ID 为输入；
5. 新增 anchor rows 后，原有相同 origin 的 `base_origin_id` 应保持稳定；若 `view_sample_id` 因 support protocol 版本化而变化，必须在 protocol registry 显式记录；
6. C1 应保存每行可用于 native-mask 计算的 causal left boundary（例如 `causal_history_floor`），而不是要求 K 从 head-global `lmax` 反推边界；
7. `dependency_start` / `lmax_steps` 若继续保留为兼容字段，必须明确其 anchor-level 或 candidate-level 语义，禁止一个字段同时被两种语义解释；
8. 任何读取 `dependency_start`、`lmax_steps`、`valid_origins_for_interval()` 的下游审计代码都必须与新的 support contract 同步，避免“样本已放宽但审计仍按旧全局历史”或相反的语义漂移。

### 11A.12.6 历史结果与证据边界

本实践修正不把旧 common-support 实验改写为 native-support 实验。对已经完成并打开 test/OOD 的历史运行，正确表述为：

\[
\boxed{\text{VALID UNDER HEAD-LEVEL COMMON-SUPPORT PROTOCOL, NOT NATIVE-SUPPORT OPTIMAL}.}
\tag{11A.9}
\]

它不构成标签泄漏、未来输入泄漏或 train/test contamination；其主要限制是短历史候选损失合法训练样本，可能降低统计功效并影响 K activation、profile/family selection 以及后续装配。若对已打开 lockbox 的数据重放新 support protocol，只能标记为 retrospective protocol-sensitivity analysis，不得重新宣称 untouched confirmatory evidence。

### 11A.12.7 强制实现不变量

```text
assert C1_support_semantics == MAXIMALLY_PERMISSIVE_ANCHOR_UNIVERSE
assert candidate_fit_rows <= candidate_native_support_rows
assert row_cap_applied_after_native_mask
assert all_local_candidates_share_identical_scoring_support_hash
assert exact_zero_scoring_support_hash == nonzero_scoring_support_hash
assert no_cross_channel_K_loss_ranking_before_common_support_replay
assert C_support_hash == intersection(selected_active_K_native_support_hashes)
assert best_active_K_comparator_support_hash == C_support_hash
assert W_input_support_hash == C_support_hash
assert Joint_raw_input_support_hash == C_support_hash
assert support_mask_does_not_read_test_or_target_error
```

这些断言属于实践执行不变量。其目标不是让所有候选拥有相同训练样本数，而是让每个候选充分使用自己的合法训练信息，同时保证任何直接的模型选择比较发生在完全相同的评分样本上。


# 11B. 必须通过的回归测试

1. `test_profile_one_se_regret_guard`：最简单 profile 虽在 one-SE 内，但超过冻结相对遗憾时必须排除。  
2. `test_profile_retains_best_and_near_simple`：最多保留最佳与双门内最简单的不同 profile。  
3. `test_minimal_stabilizing_ridge_prefers_zero`：\(\lambda=0\) 证书通过时必须选 0。  
4. `test_ridge_not_used_as_exact_zero`：超强 ridge 的近零预测不能冒充 exact-zero。  
5. `test_c_cannot_erase_active_k`：active K 存在时，C 不得静默输出近常数。  
6. `test_c_fallback_to_best_active_k`：所有 C 表示失败时，回退预测与冻结最佳 K 逐样本完全对齐。  
7. `test_w_three_of_four_usable_folds`：仅一折退化时，W 非线性候选仍实际运行。  
8. `test_w_exact_zero_k_forces_identity`：全部 K exact-zero 时，正式 W 必须 identity。  
9. `test_w_identity_and_nonlinear_share_fold_mask`：W 的配对 loss 使用共同可用 folds。  
10. `test_pf_and_joint_share_input_path_gate`：PF 与 Joint 使用同一 gate 版本和阈值。  
11. `test_final_loss_matches_materialized_prediction`：候选、loss、contract、预测的 ID 与 hash 一致。  
12. `test_joint_has_no_ar_only_candidate`：Joint 注册表中不存在 AR-only。  
13. `test_joint_w_coefficients_are_jointly_fitted`：改变 W block 应改变联合正规方程维数，而非只增加预拟合标量列。  
14. `test_a_uses_maturity_delay_and_latest_index`：成熟条件包含 \(D_m\) 且使用 latest index。  
15. `test_fold_local_residual_centering`：evaluation 残差不得影响 fit-fold 中心。  
16. `test_no_test_access_before_manifest_freeze`。  

这些测试全部通过是访问开发结果的前置条件；开发冻结清单完成是访问 test 的前置条件。

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

除非相关投影空间满足额外正交、嵌套或交换条件。因而 v2.1.1 使用“层级残差归属”，不声称“完备正交分解”。

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


## 命题 11：最小稳定 ridge 不改变结构中性元定义

当固定支持模型使用最小通过数值证书的 ridge 时，ridge 只选择该结构的数值代表；exact-zero 仍只能由显式 zero 候选产生。因而强正则极限不被误写成结构关闭。

## 命题 12：C 保存门阻止已激活 K 被静默撤销

若 K 已产生通过门槛的 \(p_{bestK}^{OOF}\)，且 C 只有通过式 (3.9)–(3.10) 或显式回退才可进入下游，则任意合法 PF 输入支路要么保存可检测的 K 信息，要么明确报告 C 回退/失败，不会出现标签 active 而预测近零的静默状态。

## 命题 13：尺度无关 W 准入对线性单位变换不敏感

若 \(q' = a q+b\) 且 \(a\ne0\)，则基于相对 machine-epsilon floor、不同值数量和 \([1,q]\) 数值秩的 W 可用性判定，在无数值下溢/上溢时保持相同结构结论；固定绝对方差阈值不具有该性质。

## 命题 14：共享输入 gate 保证 PF/Joint 状态语义一致

当 PF 与 Joint 对相应输入预测调用同一 gate 函数、同一阈值和同一 target normalization 时，状态差异只能来自输入预测本身，而不能来自两套互相矛盾的判定语义。

## 命题 15：候选 ID 绑定使选择损失可追溯

若候选的 fold losses、合同与预测均由不可变 `candidate_id` 绑定，且最终物化仅通过该 ID 解析，则 practical gate 后无法合法地使用另一候选的损失评价当前预测。

---

# 14. 正式研究流程

1. 冻结数据、目标头、split、sample IDs、baseline contracts 和 hashes；
2. 冻结 v2.1.1 配置、输入 gate 版本和回归测试版本；
3. 在不访问 test 的条件下运行全部强制回归测试；
4. 对 K profile 执行 one-SE + 相对遗憾双门；
5. 在最多两个保留 profile 上联合选择 K family/resolution；
6. 固定 K 支持并以最小稳定 ridge 生成 OOF 预测；
7. 运行 C representation 选择和输入支路保存门；
8. 若需要，显式回退 `BEST_ACTIVE_K_CHANNEL`；
9. 生成一级 OOF 残差并审计 W fold 可用性；
10. 运行 ΔW 局部 guarded one-SE，区分正式 W 与 rescue diagnostic；
11. 生成二级 OOF 残差，按 latest index 构造成熟历史；
12. 运行 A 局部选择并绑定最终 candidate/loss/path/contract；
13. 输出 PF 装配卡并运行统一输入 gate；
14. 使用相同注册特征运行真实 Joint-KWA 块联合拟合；
15. 用同一 gate 评价 Joint 输入贡献，并检查 PF/Joint 状态语义；
16. 若任一实现不变量失败，停止并标记 implementation failure；
17. 满足预注册开发继续门后，冻结最终 manifest；
18. 冻结后首次访问 test；
19. 与冻结逐样本基线做配对比较和 block bootstrap；
20. 输出预测、结构、证书、参数、资源、manifest/hash；
21. 打包时保留旧失败审计，不覆盖历史目录。

---

# 15. 经验中立性

## 15.1 本文不声称

- v2.1.1 一定优于 v1.3 或 v2.1；
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
- 所有 v2.1.1 PRISM 选择应在对应局部作用域内完成；
- profile 简化必须同时受 one-SE 与相对遗憾约束；
- 数值 ridge 不得代替 exact-zero 或撤销 active K；
- W 的准入应尺度无关，并允许部分 fold 为 `NOT_APPLICABLE`；
- PF 与 Joint 必须共享输入支路判定语义；
- Joint 可以允许 K/W/AR 相互侵占，但必须保留非零输入支路并真实联合拟合 W basis coefficients；
- 理论证书与预测基线必须分开报告；
- 实现不变量未通过时，实验结果不能用于否定理论模块。

---

# 16. 最终语义链

\[
\boxed{
\text{冻结时间、数据、选择与实现合同}
\rightarrow
\text{K profile 双门与局部评估}
\rightarrow
\text{固定支持最小稳定重拟合}
\rightarrow
\text{C 表示选择与输入保存门}
\rightarrow
r^{(1)}
\rightarrow
\text{W 尺度无关准入与局部选择}
\rightarrow
r^{(2)}
\rightarrow
\text{A 成熟残差选择}
\rightarrow
\text{候选 ID 一致性冻结}
\rightarrow
\text{PF 装配卡}
\rightarrow
\text{真实 Joint-KWA}
\rightarrow
\text{共享输入 gate}
\rightarrow
\text{冻结后 test}
}
\tag{16.1}
\]

PRISM v2.1.1 的核心问题是：

\[
\boxed{
\text{既要让 K/W/A/J 按理论分工参与竞争，}
\text{又要阻止选择器、正则、准入门和物化代码}
\text{在模型参赛之前静默改变其语义。}
}
\tag{16.2}
\]

因此 v2.1.1 把下式作为实验可信性的前置条件：

\[
\boxed{
\text{理论候选}
=
\text{被选择候选}
=
\text{被计分候选}
=
\text{被物化候选}
=
\text{被部署候选}.
}
\tag{16.3}
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


# 附录 B2：v2.1 到 v2.1.1 的算法修正映射

| v2.1 风险点 | v2.1.1 正式合同 |
|---|---|
| profile one-SE 可因宽 SE 过度偏向最短历史 | one-SE + 相对遗憾双门，最多保留两个 profile |
| ridge 同时参与结构偏序 | ridge 仅作数值稳定，选择最小通过证书值 |
| C 可把 active K 压成近零 | 输入支路保存门 + `BEST_ACTIVE_K_CHANNEL` 显式回退 |
| W 使用固定绝对 latent 方差准入 | float64 相对尺度、不同值数量、数值秩与共同可用 folds |
| 一个 fold 退化导致所有 W 非线性候选消失 | 退化 fold=`NOT_APPLICABLE`，满足最少可用 fold 即继续 |
| PF 依据 active channel，Joint 依据预测方差 | 统一权威 `evaluate_input_path` |
| practical gate 后预测与 one-SE loss 可能错配 | 不可变 `candidate_id` 绑定 loss/path/contract |
| Joint 把预拟合 KW 标量当 W | 固定 W basis construction，联合求解 \(\gamma_W\) |
| 实现错误可能被误判为模型失败 | 强制断言、回归测试与 implementation-failure 状态 |

# 附录 C：v2.1.1 首轮核心实验暂不激活

- 多工况 mixture-of-experts；
- 高阶全通道交互；
- 深度 W；
- 深度 Joint；
- 在线 K 形状适配；
- 测试时自适应；
- 把不确定性作为预测层；
- 用 AR-only 作为 Joint 退化候选。

> Joint predictive-stability estimator 的执行语义已收口至本理论第 11A.11 节；不存在独立的 canonical v2.2 理论版本。
