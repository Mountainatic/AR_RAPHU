# PRISM Theory
## Physics-first Response Identification with Scale-specific Multirate Urysohn Models
### 物理优先、通道尺度专属、多速率 Urysohn 与尺度匹配 AR 双路线理论体系 v1.3

> **正式中文名**：PRISM 物理优先尺度专属多速率响应辨识理论  
> **正式英文名**：Physics-first Response Identification with Scale-specific Multirate Urysohn Models  
> **简称**：PRISM  
> **版本日期**：2026-08-01  
> **文档性质**：项目总理论、模型语义、认证条件、数值实现与工业部署主文档  
> **蓝本**：`OPS_UOI_PS_AR_RAPHU_CUDA_PLC_Three_Layer_Complete_System_v2_0.md`
> **修订依据**：`OPS_UOI_PS_AR_RAPHU_CUDA_PLC_Three_Layer_Complete_System_v2_0.md` 的三层理论骨架，以及当前对话确认的通道专属多时间尺度、尺度匹配 AR、Urysohn 优先与 K-Joint AR 双路线
> **版本边界**：不采用任何高于 v2.0 的 OPS-UOI 理论文档；后续版本中的公共—私有分解、共享时滞主结构和功率专属预设热状态银行不属于本版  
> **主要应用实例**：CZ 法单晶硅等径阶段晶体直径的闭环多时间尺度预测与输入响应辨识  
> **文档边界**：本文只陈述理论对象、假设、模型路线、认证条件与部署合同；任何具体数据集上的模型排名、误差数值、通道通过/失败状态和经验结论均不属于本文

---

# 0. PRISM 的总定义、版本修订与三层接口

## 0.1 PRISM 研究的核心问题

PRISM 研究的不是普通的单步黑箱回归，而是下列问题：

> 在强闭环、强输出惯性、输入共线、有效激励有限、不同物理量响应尺度显著不同的工业时间序列中，怎样先按物理通道各自的时间尺度恢复可检验的输入历史响应，再用严格成熟的剩余信息补偿未测状态，同时不把高预测精度误写成开放环因果机制。

其正式结构为：

\[
\boxed{
\text{当前输出状态锚点}
+
\text{通道专属多速率物理 }K
+
\text{后置成熟残差预测器}
}
\]

在解释路线中，物理层必须先拟合、先认证、先冻结；残差层不得反向改写物理层。预测路线可以联合优化，但其内部物理归因不作结构解释。

PRISM 同时保留两条正式路线：

\[
\boxed{
\text{路线 I：Urysohn 优先 }K
\rightarrow
\text{冻结 }K
\rightarrow
\text{成熟残差 AR 或 exact-zero}
}
\tag{0.1}
\]

\[
\boxed{
\text{路线 II：K-Joint AR，物理输入项与 AR/预测状态项联合优化}
}
\tag{0.2}
\]

路线 I 用于物理响应认证和解释权分配；路线 II 用于工程预测。两条路线使用相同的时间因果、outer split、目标头和通道多速率合同，但不能互相替代证据。

## 0.2 PRISM 三层体系

PRISM 保留蓝本的三层结构，但重新定义每一层的正式对象。

### 第一层：PRISM 顶层响应可辨识理论

回答：

- 每个物理通道在什么时间尺度上接受审计；
- 被动闭环数据能识别总预测贡献、通道贡献，还是完整核；
- 多输入共线时怎样通过 Schur/Gram 条件判断单通道是否可分离；
- 哪些结果只能称为 predictive operator，哪些结果才允许称为稳定物理响应结构；
- 为什么 K-Joint AR 的总体预测可以有效，而内部 \(K\) 与 AR/状态分解仍不唯一。

### 第二层：PRISM 多速率 Urysohn–Residual 模型

回答：

- 如何为每个通道独立设定建模步长、预测提前量、目标平均窗口、历史范围和时滞基；
- 如何在每个通道内部从 exact-zero、线性分布时滞、rank-1 逐级扩展到自适应低秩或完整非线性 Urysohn；
- 如何进行通道独立认证、兼容目标头联合重拟合、物理层冻结和成熟残差建模；
- 如何区分解释路线与纯预测路线。

### 第三层：CPU / CUDA / MCU / PLC 执行与部署

回答：

- 如何构造多速率因果缓存；
- 如何用 FP64 直接求解、matrix-free PCG 和 CUDA 批处理估计模型；
- 如何将时滞—幅值算子编译为 LUT、FIR、IIR 或稳定状态空间；
- 如何给出重采样、截断、查表、量化和状态近似的总误差预算。

三层之间的语义链为：

\[
\boxed{
\text{注册物理任务}
\rightarrow
\text{合法预测响应对象}
\rightarrow
\text{有限维可认证 }K
\rightarrow
\text{冻结物理层}
\rightarrow
\text{成熟残差补偿}
\rightarrow
\text{CUDA/PLC realization}
}
\]

底层加速不能创造顶层可辨识性；高精度预测也不能自动升级为物理结构认证。

## 0.3 相对 OPS-UOI v2.0 的保留内容

PRISM 保留下列核心思想：

1. 算子优先，而不是先固定神经网络结构；
2. 使用与实现一致的加权 \(L^2\)/Hilbert–Schmidt 几何；
3. 区分总贡献、商空间代表、完整核和结构 rank；
4. 使用 Gram、广义 Schur 补、信号强度、支持稳定性和 rank margin 作为结构声明条件；
5. 正则化只解决有限样本和数值稳定，不能制造科学可辨识性；
6. 时间因果、forward split、purge、placebo、moving-block bootstrap 和多运行域外层验证；
7. CPU FP64、matrix-free forward/adjoint、PCG、CUDA 与 PLC 编译链；
8. 幅值域外的有限带 \(C^1\) 延拓与 OOD 合同；
9. predictive rank、structural rank 和 deployment rank 必须分开。

## 0.4 删除或降级的旧内容

PRISM 正式删除或降级以下内容：

1. **删除独立 Q 层**。贡献增量、AR 条件新颖性和门槛均为诊断或证书，不是模型层；
2. **删除 AR-first 双残差作为主结构**。条件中心化只保留为辅助诊断，不再定义主物理层；
3. **删除所有输入共用同一采样步长、同一预测视野和同一历史窗口的默认设定**；
4. **删除 Gamma 时滞先验、动态 scorer 和 KAN 响应作为主模型必备部件**；
5. **删除高自由度 Full-Urysohn 面作为默认入口**；
6. **删除“先 lift、再 power、再 rotation”按变量顺序抢残差的训练方式**；
7. **删除把未认证变量送入 AR/ARX 分支继续解释的做法**；
8. **删除强制非零 rank、强制非线性或强制每个变量至少保留一个模态的实现语义**；
9. **删除用 K-Joint AR 的低 RMSE 补足物理层证据的做法**。

## 0.5 PRISM 新增内容

本版新增：

1. 通道专属 profile：每个变量独立拥有 \((\Delta,h,W,T,\mathcal B)\)；
2. 单采样率高频日志上的多速率物理表示；
3. 当前平均输出作为状态锚点，模型预测未来窗口变化量；
4. Physics-First \(K\rightarrow\)Mature-Residual 与 K-Joint AR 双路线；
5. 兼容目标头：不同通道只有在目标语义相同的情况下才联合重拟合；
6. OOF 物理残差和严格成熟条件；
7. exact-zero 是正式候选和正式结论；
8. 线性分布时滞必须被非线性模型精确嵌套；
9. 每个通道独立选择 exact-zero、线性、rank-1、自适应 rank-\(R\) 或完整 Urysohn；
10. 每个候选物理尺度配套尺度匹配 AR；不同目标头不得共用一个万能 AR；
11. 理论陈述与经验结果分离：实验只负责实例化和检验理论条件，不反向定义理论结构。

## 0.6 理论与认证状态标签

| 标签 | 含义 |
|---|---|
| `DEFINITION` | 正式定义 |
| `DERIVED` | 在列出的假设下可推导 |
| `IMPLEMENTED` | 已有对应代码路径 |
| `BENCHMARKED` | 已按预注册协议完成实验运行（由实验报告填写） |
| `PREDICTIVE_VALIDATED` | 总体预测证据通过，但不作物理结构解释 |
| `PHYSICAL_CERTIFIED` | 通道物理结构证书全部通过 |
| `DIRECTION_UNSTABLE` | 至少一个跨运行方向失败 |
| `EXACT_ZERO` | 注册模型选择零通道/零残差 |
| `UNRESOLVED` | 证据不足，不能作正负结构声明 |
| `ARCHIVED` | 不再属于正式主路线 |

---

# 1. 系统、信息集、变量与预测目标

## 1.1 固定阶段受迫闭环系统

设真实过程隐状态为

\[
\xi_t\in\mathcal M_r,
\]

其中 \(r\) 表示固定工艺阶段。系统可写为

\[
\xi_{t+1}=F_r(\xi_t,u_t,w_t,c_t),
\tag{1.1}
\]

\[
x_t=h_x(\xi_t,u_t,c_t)+\nu_t^x,
\tag{1.2}
\]

\[
y_t=h_y(\xi_t)+\nu_t^y.
\tag{1.3}
\]

其中：

- \(u_t\)：操纵量、设定值或控制器输出；
- \(c_t\)：控制器内部状态；
- \(w_t\)：未测扰动；
- \(x_t\)：可测过程变量；
- \(y_t\)：晶体直径观测或设备估计值；
- \(\nu_t^x,\nu_t^y\)：测量误差。

被动闭环数据中的输入既包含操纵意图，也包含控制器对输出、状态和扰动的反馈反应。因此 PRISM 默认识别的是：

\[
\boxed{
\text{固定阶段、当前闭环观测分布下的预测响应算子}
}
\]

而不是未经额外实验设计证明的开放环 plant transfer。

## 1.2 CZ 应用中的注册输入

当前 CZ 主输入为：

1. 主加热功率 \(u_{P,t}\)；
2. 联合升速 \(u_{L,t}\)；
3. 晶转速度 \(u_{C,t}\)；
4. 埚转速度 \(u_{R,t}\)。

当前不进入主模型：

- 晶体长度；
- 加热元件温度；
- 氩气流量；
- 炉压。

不进入主模型不表示这些变量没有物理作用，只表示它们不属于当前冻结的信息合同。

## 1.3 联合升速

晶升与埚升作为一个联合升速控制变量处理。对 outer training rod 内的晶升与埚升分别标准化：

\[
\widetilde v_{1,t}=\frac{v_{\mathrm{crystal\ lift},t}-\mu_1}{\sigma_1},
\qquad
\widetilde v_{2,t}=\frac{v_{\mathrm{crucible\ lift},t}-\mu_2}{\sigma_2}.
\tag{1.4}
\]

联合升速定义为训练数据 PCA 第一主成分：

\[
u_{L,t}=w_1\widetilde v_{1,t}+w_2\widetilde v_{2,t}.
\tag{1.5}
\]

约束：

- \(\mu_1,\sigma_1,\mu_2,\sigma_2,w_1,w_2\) 仅由 outer training rod 估计；
- 主成分符号固定为与联合上升方向一致；
- validation/test 原样应用训练参数；
- 现有数据不再尝试分离晶升与埚升的独立贡献。

## 1.4 预测时刻的信息集

在预测原点 \(t\)，只允许使用

\[
\mathcal F_t=\sigma\{u_s,x_s,y_s:s\le t\}.
\tag{1.6}
\]

主轨道不使用未来输入。已知未来设定值只能建立单独的 known-future-input 协议，不能混入主结果。

## 1.5 输出平均与状态锚点

定义过去窗口平均：

\[
\overline y_t^{(W_0)}
=
\frac{1}{W_0}\int_{t-W_0}^{t}y(s)\,ds,
\tag{1.7}
\]

未来窗口平均：

\[
\overline y_{t+h}^{(W)}
=
\frac{1}{W}\int_{t+h}^{t+h+W}y(s)\,ds.
\tag{1.8}
\]

注册目标为未来窗口变化：

\[
\boxed{
z_m(t)
=
\overline y_{t+h_m}^{(W_m)}
-
\overline y_t^{(W_{0,m})}.
}
\tag{1.9}
\]

通常 \(W_{0,m}=W_m\)。

最终未来平均直径预测为

\[
\widehat{\overline y}_{t+h_m}^{(W_m)}
=
\overline y_t^{(W_{0,m})}
+
\widehat z_m(t).
\tag{1.10}
\]

因此模型不需要重新学习当前绝对直径水平；当前平均直径承担状态锚点，物理和残差分支只预测未来变化。

---

# 2. 通道专属多速率 profile 理论

## 2.1 单采样率日志与多速率物理表示

原始日志采样周期约为

\[
\Delta_0\approx2\ \mathrm{s}.
\tag{2.1}
\]

原始文件是 single-rate，但物理响应不是同尺度。PRISM 从同一高频日志构造多速率因果分支：

| 通道 | 建模步长 | 物理目的 |
|---|---:|---|
| 联合升速 | 10 s | 保留较快界面与拉速累积响应 |
| 晶转、埚转 | 30 s | 抑制高频纹波，保留中频熔体流动信息 |
| 主加热功率 | 120 s | 聚焦慢热惯性，避免数千冗余 lag |

分支必须由因果区间平均构造：

\[
\widetilde u_j(k)
=
\frac{1}{|I_{j,k}|}
\sum_{s\in I_{j,k}}u_j(s),
\qquad I_{j,k}\subset(-\infty,t].
\tag{2.2}
\]

禁止使用中心滑窗或任何包含 \(t\) 后数据的抗混叠处理。

## 2.2 通道 profile

对物理通道 \(j\) 的第 \(r\) 个 profile，定义

\[
\boxed{
\pi_{j,r}
=
(\Delta_j,h_{j,r},W_{j,r},W_{0,j,r},T_{j,r},\mathcal B_{j,r}).
}
\tag{2.3}
\]

其中：

- \(\Delta_j\)：通道专属建模步长；
- \(h_{j,r}\)：预测提前量；
- \(W_{j,r}\)：未来输出平均窗口；
- \(W_{0,j,r}\)：当前状态锚点窗口；
- \(T_{j,r}\)：输入历史范围；
- \(\mathcal B_{j,r}\)：多分辨率时滞块或紧凑时滞基。

离散步数为

\[
H_{j,r}=\left\lceil\frac{h_{j,r}}{\Delta_j}\right\rceil,
\qquad
L_{j,r}=\left\lceil\frac{T_{j,r}}{\Delta_j}\right\rceil.
\tag{2.4}
\]

正式任务族为

\[
\boxed{
\mathfrak M
=
\bigcup_{j=1}^{p}\mathfrak M_j,
\qquad
\mathfrak M_j=\{\pi_{j,r}\}_{r=1}^{R_j}.
}
\tag{2.5}
\]

这不是“每个统一 horizon 把所有输入一起跑一次”，而是“每个通道在自己的物理时间尺度族中接受审计”。


因此，“某通道在短尺度无效”不能推出“该物理量无效”。它只说明该通道在当前 \((\Delta,h,W,T,\mathcal B)\) 注册下没有稳定证据。相同通道必须允许在更匹配的慢尺度、平滑目标或更长历史中重新接受审计。

## 2.3 CZ 多尺度 profile 注册示例

下表仅作为 CZ 应用中的 profile 注册示例，不是 PRISM 的普遍常数，也不表示任何 profile 已经取得更好效果。它体现的理论原则是：不同物理量可以在不同采样步长、历史范围、预测提前量和目标平滑窗口下接受审计。具体 profile 的保留、归零或认证必须由独立实验报告给出。

### 联合升速 L1–L6

\[
\Delta_L=10\ \mathrm{s}.
\]

| Profile | \(h\) | \(W\) | 历史 \(T\) | 离散 \((H,W_s,L)\) |
|---|---:|---:|---:|---:|
| L1 | 30 s | 30 s | 10 min | \((3,3,60)\) |
| L2 | 1 min | 30 s | 10 min | \((6,3,60)\) |
| L3 | 2 min | 1 min | 20 min | \((12,6,120)\) |
| L4 | 5 min | 1 min | 20 min | \((30,6,120)\) |
| L5 | 10 min | 2 min | 40 min | \((60,12,240)\) |
| L6 | 20 min | 2 min | 40 min | \((120,12,240)\) |

### 晶转与埚转 R1–R6

\[
\Delta_{\mathrm{rot}}=30\ \mathrm{s}.
\]

| Profile | \(h\) | \(W\) | 历史 \(T\) | 离散 \((H,W_s,L)\) |
|---|---:|---:|---:|---:|
| R1 | 2 min | 1 min | 30 min | \((4,2,60)\) |
| R2 | 5 min | 2 min | 30 min | \((10,4,60)\) |
| R3 | 10 min | 2 min | 60 min | \((20,4,120)\) |
| R4 | 20 min | 5 min | 60 min | \((40,10,120)\) |
| R5 | 40 min | 5 min | 120 min | \((80,10,240)\) |
| R6 | 60 min | 10 min | 120 min | \((120,20,240)\) |

晶转和埚转共用候选尺度表，但始终作为两个独立通道拟合、归零和认证。

### 主加热功率 P1–P7

\[
\Delta_P=120\ \mathrm{s}=2\ \mathrm{min}.
\]

| Profile | \(h\) | \(W\) | 历史 \(T\) | 2 min 离散 \((H,W_s,L)\) | 状态 |
|---|---:|---:|---:|---:|---|
| P1 | 10 min | 5 min | 60 min | \((5,\approx3,30)\) | candidate |
| P2 | 20 min | 5 min | 60 min | \((10,\approx3,30)\) | candidate |
| P3 | 40 min | 10 min | 120 min | \((20,5,60)\) | candidate |
| P4 | 60 min | 10 min | 120 min | \((30,5,60)\) | candidate |
| P5 | 90 min | 15 min | 180 min | \((45,\approx8,90)\) | candidate |
| P6 | 120 min | 15 min | 180 min | \((60,\approx8,90)\) | candidate |
| P7 | 180 min | 20 min | 240 min | \((90,10,120)\) | exploratory |

P1–P7 均属于通道尺度候选。只有在训练集内筛选、冻结后进入独立确认的数据头，才可产生确认性结论。

## 2.4 兼容目标头

通道独立 profile 用于发现各通道的有效尺度。多个通道只有在目标语义相同时才能联合进入同一预测头。

定义目标头

\[
m=(h_m,W_m,W_{0,m}).
\tag{2.6}
\]

通道 \(j\) 在目标头 \(m\) 上使用的表示为

\[
\rho_{j,m}=(\Delta_j,T_{j,m},\mathcal B_{j,m}).
\tag{2.7}
\]

因此多个通道可以共享同一预测目标，但仍使用各自的建模步长、历史长度和时滞块：

\[
\boxed{
\text{共享目标语义，不强制共享输入时间网格。}
}
\]

## 2.5 尺度匹配 AR 配套原则

PRISM 不允许用一个固定的高频 AR 与所有物理 profile 比较。对每个候选物理 profile

\[
\pi_{j,r}^{K}
=
(\Delta_j,h_{j,r},W_{j,r},W_{0,j,r},T_{j,r},\mathcal B_{j,r}),
\tag{2.8}
\]

必须同时注册一个预测同一目标的尺度匹配状态模型 profile：

\[
\boxed{
\pi_{j,r}^{A}
=
(\Delta^{A}_{j,r},h_{j,r},W_{j,r},W_{0,j,r},T^{A}_{j,r},\mathcal L^{A}_{j,r},\mathscr A_{j,r}).
}
\tag{2.9}
\]

其中：

- \(\Delta^{A}_{j,r}\)：输出状态序列的因果建模步长；
- \(T^{A}_{j,r}\)：AR/预测状态历史覆盖范围；
- \(\mathcal L^{A}_{j,r}\)：离散 lag 集、lag blocks 或状态基；
- \(\mathscr A_{j,r}\)：AR、ridge-AR、低维 predictive-state 等候选类；
- \((h_{j,r},W_{j,r},W_{0,j,r})\) 必须与对应物理 profile 完全相同。

物理 profile 与 AR profile 构成注册对：

\[
\boxed{
\Pi_{j,r}
=
(\pi_{j,r}^{K},\pi_{j,r}^{A}).
}
\tag{2.10}
\]

“尺度匹配”不要求 \(\Delta^{A}_{j,r}=\Delta_j\) 数值相等，但要求 AR 的输出频带、历史覆盖和预测目标与该 profile 的物理时间尺度一致。例如，长提前量、平滑目标窗口的物理 profile 必须配套慢尺度输出状态历史，不能拿秒级短历史 AR 作为唯一条件基线；小时级物理候选也必须配套更长的状态历史，而不能复用另一目标头的 AR。

因此：

\[
\boxed{
\text{不同物理尺度}
\Longrightarrow
\text{不同的尺度匹配 AR/预测状态配置}.}
\tag{2.11}
\]

但需要区分“每个通道 profile 的诊断 AR”和“最终目标头的正式 AR”：

1. **profile 诊断 AR** \(A^{\mathrm{diag}}_{j,r}\)：在通道尺度扫描中先冻结，用于检验该通道在相同目标与频带下是否提供 AR 之外的条件增量；它是 AR-first 诊断，不是正式解释路线；
2. **Physics-First 残差 AR** \(A^{\mathrm{res}}_m\)：在目标头 \(m\) 的物理层联合重拟合并冻结后，基于该头的 OOF 成熟残差训练；
3. **K-Joint AR** \(A^J_m\)：与目标头 \(m\) 的 \(K^J_m\) 联合优化，服务于工程预测。

若多个通道虽然输入步长不同，但预测同一个目标头 \(m\)，则最终路线 I 和路线 II 分别使用一个头级 \(A^{\mathrm{res}}_m\) 与一个头级 \(A^J_m\)，而不是为每个通道重复建立可归因的 AR。AR 只能表示该目标头的总体状态信息，不能被解释为某个特定输入通道的物理作用。

## 2.6 严格样本合法性

样本 \((m,t)\) 只有在以下全部条件满足时才合法：

1. 当前锚点窗口 \([t-W_{0,m},t]\) 完整存在；
2. 未来目标窗口 \([t+h_m,t+h_m+W_m]\) 完整存在；
3. 每个激活通道历史 \([t-T_{j,m},t]\) 完整存在；
4. 所有窗口均位于同一连续稳定段；
5. 不跨缺失断点、异常修复边界或工艺阶段边界；
6. 所有预处理参数来自 outer training data；
7. sample ID 在模型间保持一致。

---

# 3. 第一层：PRISM 顶层算子与可辨识理论

## T1. 通道专属核空间

对目标头 \(m\) 和通道 \(j\)，设幅值参考域为 \(\mathcal I_j\)，参考测度为 \(\nu_j\)。定义

\[
\mathcal H_{j,m}
=
L^2([0,T_{j,m}]\times\mathcal I_j,
\omega_{j,m}(\tau)d\tau\otimes d\nu_j(u)).
\tag{T1.1}
\]

总核空间为

\[
\mathcal H_m
=
\bigoplus_{j=1}^{p}\mathcal H_{j,m}.
\tag{T1.2}
\]

范数

\[
\|K\|_{\mathcal H_m}^2
=
\sum_{j=1}^{p}
\int_0^{T_{j,m}}
\int_{\mathcal I_j}
K_{j,m}(\tau,u)^2
\,d\nu_j(u)\,\omega_{j,m}(\tau)d\tau.
\tag{T1.3}
\]

不同通道允许不同历史区间和不同离散网格，但统一在物理时间 \(\tau\) 中定义。

## T2. 多通道 Urysohn 映射

定义

\[
(\mathcal A_mK)_t
=
\sum_{j=1}^{p}
\int_0^{T_{j,m}}
K_{j,m}(\tau,u_j(t-\tau))\,d\tau.
\tag{T2.1}
\]

工程实现中的多分辨率块、FIR 或状态空间均是 \(\mathcal A_m\) 的有限维近似。

### 假设 A1：有界设计密度

对每个通道与时滞，\(u_j(t-\tau)\) 相对 \(\nu_j\) 的密度存在统一上界。

### 命题 T2.1：\(\mathcal A_m\) 有界

在 A1 下存在常数 \(C_{A,m}<\infty\)，使

\[
\|\mathcal A_mK\|_{L^2(P)}
\le
C_{A,m}\|K\|_{\mathcal H_m}.
\tag{T2.2}
\]

因此顶层目标可在 Hilbert 空间中定义，不需要把无限维 \(L^2\) 的点值评价误当成连续泛函。

## T3. 物理优先预测响应对象

对注册目标 \(z_m(t)\)，定义物理可表示闭包

\[
\mathcal G_m
=
\overline{\operatorname{ran}\mathcal A_m}
\subset L^2(P).
\tag{T3.1}
\]

PRISM 主物理层的总体 estimand 为

\[
\boxed{
g_m^{\mathrm{PF}}
=
P_{\mathcal G_m}z_m.
}
\tag{T3.2}
\]

它表示：在当前注册输入、当前通道时间尺度和当前闭环分布下，能够由加性多通道 Urysohn 历史稳定逼近的最佳输入型预测响应。

残余为

\[
\varepsilon_m^{K}
=z_m-g_m^{\mathrm{PF}},
\tag{T3.3}
\]

满足

\[
\langle \varepsilon_m^{K},\mathcal A_mK\rangle_{L^2}=0,
\qquad \forall K\in\mathcal H_m.
\tag{T3.4}
\]

该定义不要求真实系统完全加性，也不要求 \(g_m^{\mathrm{PF}}\) 等于开放环 plant 响应。

## T4. 商空间与完整核

若存在 \(K_m\in\mathcal H_m\) 使

\[
\mathcal A_mK_m=g_m^{\mathrm{PF}},
\tag{T4.1}
\]

则所有等价核形成商类

\[
[K_m]
=
K_m+\ker\mathcal A_m.
\tag{T4.2}
\]

数据首先识别的是 \([K_m]\) 或总响应 \(g_m^{\mathrm{PF}}\)。最小范数代表为

\[
K_m^\dagger
=
\mathcal A_m^\dagger g_m^{\mathrm{PF}}
\in(\ker\mathcal A_m)^\perp.
\tag{T4.3}
\]

只有当有限结构空间内满足 injectivity 时，核坐标才可稳定解释。

## T5. 有限 sieve injectivity

对有限维空间 \(V_m\subset\mathcal H_m\)，定义

\[
\Gamma_m=\mathcal A_m^\ast\mathcal A_m.
\tag{T5.1}
\]

若存在 \(\kappa_m>0\)，使

\[
\langle K,\Gamma_mK\rangle
\ge
\kappa_m\|K\|_{\mathcal H_m}^2,
\qquad \forall K\in V_m,
\tag{T5.2}
\]

则 \(V_m\cap\ker\mathcal A_m=\{0\}\)，有限核坐标稳定可辨识。

ridge 使

\[
(\widehat\Gamma_m+\lambda I)\widehat\theta=b
\]

可解，但不等价于式 (T5.2) 成立。

## T6. 多变量单通道可分离性

将有限设计写为

\[
\phi_t=(\phi_{1,t}^\top,\ldots,\phi_{p,t}^\top)^\top.
\tag{T6.1}
\]

总体 Gram 分块为 \(G_{jk}=E[\phi_j\phi_k^\top]\)。对通道 \(j\)，定义相对于其他通道的广义 Schur 补

\[
G_{j\cdot -j}
=
G_{jj}-G_{j,-j}G_{-j,-j}^{\dagger}G_{-j,j}.
\tag{T6.2}
\]

若

\[
\lambda_{\min}^{+}(G_{j\cdot -j})\ge\kappa_{j,m}>0,
\tag{T6.3}
\]

则通道 \(j\) 在当前有限空间中存在不能被其他输入线性重构的独立方向。

若式 (T6.3) 失败，可能出现：

\[
\sum_j\mathcal A_{j,m}K_{j,m}
=
\sum_j\mathcal A_{j,m}K'_{j,m},
\qquad
K_{j,m}\ne K'_{j,m}.
\tag{T6.4}
\]

此时总输入响应可能可预测，但 per-channel 归因不唯一。

## T7. 通道 active / exact-zero

对通道 \(j\) 定义零假设

\[
H_{0,j,m}:K_{j,m}\equiv0.
\tag{T7.1}
\]

PRISM 必须把 exact-zero 作为正式模型候选。若 zero support 在 one-SE、稳定支持枚举或外层证书中被选择，则只能登记：

```text
PHYSICAL_CHANNEL_EXACT_ZERO_UNDER_REGISTERED_PROTOCOL
```

其语义是：

> 当前注册目标、时间尺度、输入表示和跨棒协议没有稳定证据支持该通道的可辨识贡献。

禁止写成“该变量对真实物理过程没有作用”。

## T8. 条件新颖性算子只作诊断

为判断输入是否提供输出历史之外的信息，可定义预测状态 \(S_{m,t}\) 和投影

\[
\Pi_{S_m}f=E[f\mid S_{m,t}].
\]

条件新颖性算子

\[
\mathcal D_m
=(I-\Pi_{S_m})\mathcal A_m.
\tag{T8.1}
\]

其结果用于条件增量 gate、AR-first 诊断或辅助报告，但：

\[
\boxed{
\mathcal D_m\text{ 不是 PRISM 的 Q 层，也不定义主物理层。}
}
\]

主物理 estimand 仍是式 (T3.2) 的输入型物理优先投影。

## T9. K-Joint AR 分解不唯一

K-Joint AR 路线同时使用物理输入历史与 AR/预测状态。设

\[
\widehat z_m^{J}
=K_m^{J}(U_t^-)+A_m^{J}(S_{m,t}^-).
\tag{T9.1}
\]

若两个函数空间交集非零：

\[
\mathcal V_K\cap\mathcal V_A\ne\{0\},
\tag{T9.2}
\]

则对任意 \(\Delta\in\mathcal V_K\cap\mathcal V_A\)：

\[
K+A=(K-\Delta)+(A+\Delta).
\tag{T9.3}
\]

总预测不变，内部归因改变。因此 K-Joint AR 的 \(K\) 不能自动解释为物理核。

## T10. 闭环科学边界

即使一个通道通过所有预测和结构证书，也只允许称为：

\[
\boxed{
\text{当前闭环观测分布、固定阶段和注册任务下的稳定预测响应核。}
}
\]

要升级为开放环 plant 响应，仍需额外条件，例如：

- 外生激励或自然实验；
- 控制器结构与输入生成机制可识别；
- 未测混杂受控；
- 多工况重复；
- 干预或独立实验验证。

---

# 4. 第二层：PRISM 多速率 Urysohn 有限模型

## M1. 多分辨率时滞块

长历史不按原始 2 s 网格全部展开。对通道 \(j\)，定义因果历史块

\[
\mathcal I_{j,b}\subset[0,T_{j,m}],
\qquad b=1,\ldots,B_{j,m},
\]

以及块平均

\[
\bar u_{j,t-\mathcal I_{j,b}}
=
\frac{1}{|\mathcal I_{j,b}|}
\int_{\mathcal I_{j,b}}u_j(t-\tau)d\tau.
\tag{M1.1}
\]

默认分辨率：

### 联合升速

| 历史区间 | 分辨率 |
|---|---:|
| 0–2 min | 10 s/block |
| 2–10 min | 30 s/block |
| 10–40 min | 2 min/block |

### 晶转、埚转

| 历史区间 | 分辨率 |
|---|---:|
| 0–10 min | 30 s/block |
| 10–30 min | 1 min/block |
| 30–120 min | 5 min/block |

### 主加热功率

| 历史区间 | 分辨率 |
|---|---:|
| 0–10 min | 30 s/block |
| 10–30 min | 2 min/block |
| 30–120 min | 5 min/block |
| 120–240 min | 10 min/block |

短 profile 只截取其注册历史内的块。

## M2. 紧凑 Urysohn 表示

对通道 \(j\)、目标头 \(m\)：

\[
\mathcal K_{j,m}[u_j](t)
=
\sum_{b=1}^{B_{j,m}}
\omega_{j,m,b}
\Phi_{j,m,b}
\left(
\bar u_{j,t-\mathcal I_{j,m,b}}
\right).
\tag{M2.1}
\]

其中 \(\Phi\) 可为线性、样条或低复杂度一元非线性响应。

## M3. 通道内部的嵌套模型阶梯

对每个通道 \(j\) 和目标头 \(m\)，PRISM 独立选择模型复杂度。不同通道不被强制共享跨通道公共时间模态，也不存在公共分支与私有补充分支。

### M3.0 exact-zero

\[
K_{j,m}\equiv0.
\tag{M3.1}
\]

### M3.1 线性分布时滞

\[
\mathcal K_{j,m}^{\mathrm{lin}}[u_j](t)
=
\sum_b k_{j,m,b}\,\bar u_{j,t-\mathcal I_{j,m,b}}.
\tag{M3.2}
\]

### M3.2 通道独立 rank-1 Urysohn

\[
K_{j,m}(\tau,u)
=
\sigma_{j,m,1}q_{j,m,1}(\tau)f_{j,m,1}(u).
\tag{M3.3}
\]

### M3.3 通道独立自适应 rank-\(R\) Urysohn

\[
K_{j,m}^{(R)}(\tau,u)
=
\sum_{r=1}^{R_{j,m}}
\sigma_{j,m,r}q_{j,m,r}(\tau)f_{j,m,r}(u).
\tag{M3.4}
\]

这里的 \(R_{j,m}\) 对每个通道、每个目标头独立选择。一个通道的 rank 不能由另一个通道的 rank 强制决定。

### M3.4 完整有限维 Urysohn

\[
K_{j,m}(\tau,u)
=
\sum_{a=1}^{M_{\tau,j,m}}
\sum_{b=1}^{M_{x,j,m}}
\Theta_{j,m,ab}C_{j,m,a}(\tau)B_{j,b}(u).
\tag{M3.5}
\]

线性、rank-1 和低 rank 模型必须作为完整模型的精确嵌套子空间。若更高复杂度没有稳定外部增益，则保留较低阶模型或 exact-zero。

也可写成线性子空间加幅值非线性补充：

\[
K_{j,m}(\tau,u)
=
k_{j,m}(\tau)u+K_{j,m}^{\mathrm{nl}}(\tau,u),
\qquad
K_{j,m}^{\mathrm{nl}}\equiv0
\text{ 是正式候选。}
\tag{M3.6}
\]

任何更复杂模型必须精确包含更简单模型。若线性子空间被选中，必须报告“Urysohn 在当前数据与幅值范围内退化到线性分布时滞”，不得为了保留非线性名称强行加入曲面。

## M4. 有限 basis 与设计矩阵

时滞/块基向量为

\[
C_{j,m}(\tau)
=(C_{j,m,1}(\tau),\ldots,C_{j,m,M_{\tau,j,m}}(\tau))^\top,
\tag{M4.1}
\]

幅值基为

\[
B_j(u)
=(B_{j,1}(u),\ldots,B_{j,M_{x,j}}(u))^\top.
\tag{M4.2}
\]

核写为

\[
K_{j,m}(\tau,u)
=C_{j,m}(\tau)^\top\Theta_{j,m}B_j(u).
\tag{M4.3}
\]

单样本设计矩阵

\[
\Phi_{j,m,t}
=
\int_0^{T_{j,m}}
C_{j,m}(\tau)B_j(u_j(t-\tau))^\top d\tau.
\tag{M4.4}
\]

离散块实现中，积分变为带块宽权重的求和。

## M5. 幅值中心化与平移不唯一性

定义训练域幅值基均值

\[
\bar B_j
=
\frac1{n_{\mathrm{train}}}
\sum_{t\in\mathrm{train}}B_j(u_{j,t}),
\]

中心化基

\[
\widetilde B_j(u)=B_j(u)-\bar B_j.
\tag{M5.1}
\]

中心化用于消除核常数项与全局偏置之间的平移不唯一性。所有中心化统计只能来自 outer training data。

## M6. 函数空间正则

估计目标可写为

\[
\widehat\Theta
=
\arg\min_{\Theta}
\frac{1}{2n}\|z-\Phi\theta\|_2^2
+
\sum_j\mathcal R_j(\Theta_j).
\tag{M6.1}
\]

正则项可包含：

\[
\mathcal R_j(\Theta_j)
=
\lambda_0\|\Theta_j\|_F^2
+
\lambda_\tau\|D_\tau\Theta_j\|_F^2
+
\lambda_x\|\Theta_jD_x^\top\|_F^2.
\tag{M6.2}
\]

正则化仅用于有限样本方差、平滑和数值条件，不把不可辨识方向变成可辨识方向。

## M7. Gram 白化与谱分解

时滞与幅值 Gram 为

\[
G_{\tau,j,m}
=
\int C_{j,m}(\tau)C_{j,m}(\tau)^\top d\tau,
\tag{M7.1}
\]

\[
G_{x,j}
=
\int B_j(u)B_j(u)^\top d\nu_j(u).
\tag{M7.2}
\]

白化核矩阵

\[
\widetilde\Theta_{j,m}
=
G_{\tau,j,m}^{1/2}
\Theta_{j,m}
G_{x,j}^{1/2}.
\tag{M7.3}
\]

对其 SVD：

\[
\widetilde\Theta_{j,m}
=U\Sigma V^\top.
\tag{M7.4}
\]

得到与 Hilbert–Schmidt 几何一致的时滞模态、幅值响应和谱值。

## M8. 三类 rank

### 结构 rank

\[
R_{S,j,m}^{\star}
=\operatorname{rank}_{HS}(K_{j,m}).
\tag{M8.1}
\]

只有完整核、HS 误差和谱间隔证书通过时才允许报告。

### 预测 rank

\[
R_{P,j,m}^{\star}(\epsilon)
=
\min\left\{R:
\|\mathcal A_m(K-K_R)\|_{L^2(P)}\le\epsilon
\right\}.
\tag{M8.2}
\]

它依赖当前输入分布，不等于结构 rank。

### 部署 rank

\[
R_{D,j,m}^{\star}
=
\min\{R:\text{总部署误差}\le\epsilon_{\mathrm{deploy}}\}.
\tag{M8.3}
\]

预测、结构与部署 rank 不得混写。

## M9. 通道尺度由 profile 表示

主加热功率的慢热惯性在本版中仅通过其通道专属建模步长、历史范围、时滞块和目标尺度表示：

\[
(\Delta_P,T_{P,m},\mathcal B_{P,m},h_m,W_m).
\tag{M9.1}
\]

本版不额外引入预设指数热状态银行。若未来加入灰箱热状态，必须作为单独扩展接受外部验证，不能静默并入当前 PRISM 主模型。

---

# 5. 路线 I：PRISM Physics-First \(K\rightarrow\) Mature Residual

## P1. 通道独立 profile 审计

每个通道先在自己的 \(\mathfrak M_j\) 中独立运行，不允许其他物理变量或输出历史提前进入解释竞争。

对注册对 \(\Pi_{j,r}=(\pi^K_{j,r},\pi^A_{j,r})\)，先在完全相同的样本、目标头、purge 和 outer split 上拟合尺度匹配诊断 AR：

\[
\widehat A^{\mathrm{diag}}_{j,r}
=
\arg\min_{A\in\mathscr A_{j,r}}
\left[
\frac1n\sum_t
\left(z_{j,r,t}-A(S_{j,r,t}^-)\right)^2
+
\mathcal R^A_{j,r}(A)
\right].
\tag{P1.1}
\]

随后独立拟合通道物理算子：

\[
\widehat K_{j,r}
=
\arg\min_{K\in\mathscr K_{j,r}}
\left[
\frac1n\sum_t(z_{j,r,t}-K(U_{j,t}^-))^2
+
\mathcal R^K_{j,r}(K)
\right].
\tag{P1.2}
\]

AR 条件增量只作为诊断：冻结 \(\widehat A^{\mathrm{diag}}_{j,r}\) 后，再检验加入 \(K_{j,r}\) 是否在相同尺度上提供稳定增量。该步骤不得改变路线 I 的正式训练顺序，也不得把诊断 AR 的系数解释为物理状态参数。

输出包括：

- exact-zero/linear/nonlinear 模型选择；
- pooled 和双向 outer 改善；
- 核符号、峰值、质心和主要支持；
- common-support、placebo、bootstrap、正则和分辨率敏感性；
- Schur/Gram 可辨识诊断。

## P2. 固定候选集合的联合重拟合与外层认证

在 outer training data 内完成通道与 profile 选择，得到冻结候选集合

\[
\widehat{\mathcal C}_{m}^{\mathrm{train}}
=
\{j:\mathrm{SelectedOnTraining}_{j,m}=1\}.
\tag{P2.1}
\]

在固定候选集合上联合重拟合物理层：

\[
\boxed{
\widehat K_{\mathrm{PF},m}
=
\arg\min_{K_j,\,j\in\widehat{\mathcal C}_{m}^{\mathrm{train}}}
\left[
\frac1{n_m}\sum_t
\left(
 z_{m,t}-b_m-
 \sum_{j\in\widehat{\mathcal C}_{m}^{\mathrm{train}}}
 \mathcal K_{j,m}[u_j](t)
\right)^2
+
\mathcal R_{K,m}
\right].
}
\tag{P2.2}
\]

候选集合、profile、正则和支持在 outer test 前全部冻结。outer test 只用于确认，不允许根据测试结果删通道、改 profile 或重新拟合。

外层确认后，物理认证集合定义为

\[
\mathcal S_m
=
\{j\in\widehat{\mathcal C}_{m}^{\mathrm{train}}:
\mathrm{Certified}_{j,m}=1\}.
\tag{P2.3}
\]

未进入训练候选集合的通道不进入 Physics-First 候选模型，也不作为具名外生变量转交残差层；进入候选集合但未通过 outer 认证的通道可以报告预测结果与失败原因，但不得作为已认证物理响应解释或部署。

这实现：

\[
\boxed{
\text{物理优先，但不是物理变量顺序优先。}
}
\]

## P3. 先 K、后残差的两阶段冻结训练

Physics-First 不由下列联合加权目标定义：

\[
\min_{K,A}
L(z,K+A)+\lambda_K\Omega_K(K)+\lambda_A\Omega_A(A).
\tag{P3.1}
\]

正式定义为：

1. 选择、拟合并认证 \(K_{\mathrm{PF},m}\)；
2. 固定支持联合重拟合；
3. 冻结全部物理参数；
4. 生成 OOF 物理预测；
5. 构造严格成熟残差；
6. 拟合残差预测器，包括 exact-zero；
7. 禁止任何最终联合反向优化。

两阶段冻结的含义不是建立新的多目标优化术语，而是明确解释路线的训练合同：前一阶段得到的物理层在后一阶段不可被回调。

## P4. OOF 物理残差

对 inner rolling/cross-fit 中样本原点 \(s\)：

\[
r_{m,s}^{\mathrm{OOF}}
=
z_{m,s}
-
\widehat K_{\mathrm{PF},m}^{\mathrm{OOF}}(U_s^-).
\tag{P4.1}
\]

不得使用同一样本 in-sample residual 训练后置残差模型。

## P5. 残差成熟条件

残差 \(r_{m,s}\) 包含未来目标窗口，因此在预测时刻 \(t\) 可用的必要条件为

\[
\boxed{
s+h_m+W_m\le t.}
\tag{P5.1}
\]

成熟残差集：

\[
R_{m,t}^{\mathrm{mature},-}
=
\{r_{m,s}^{\mathrm{OOF}}:s+h_m+W_m\le t\}.
\tag{P5.2}
\]

一般地，当目标头的提前量为 \(h_m\)、未来平均窗口为 \(W_m\) 时，最新可用残差原点至少落后当前 \(h_m+W_m\)；任何更近的残差在在线时刻尚未成熟。

## P6. 目标头专属的后置残差预测器

每个目标头 \(m=(h_m,W_m,W_{0,m})\) 都必须拥有自己的残差预测器配置

\[
\alpha_m^{\mathrm{res}}
=
(\Delta_m^{A},T_m^{A},\mathcal L_m^{A},\mathscr A_m^{\mathrm{res}}).
\tag{P6.0}
\]

不同 \(m\) 不共享一个万能 residual AR。即使两个头使用同一物理通道，只要 \(h\)、\(W\) 或状态频带不同，就必须分别训练、选择和验证 \(A_{\mathrm{res},m}\)。冻结 \(K\) 后：

\[
\widehat A_{\mathrm{res},m}
=
\arg\min_A
\left[
\frac1n\sum_t
\left(
 r_{m,t}^{\mathrm{OOF}}-A(R_{m,t}^{\mathrm{mature},-})
\right)^2
+
\mathcal R_{A,m}(A)
\right].
\tag{P6.1}
\]

候选必须包含

\[
A_0\equiv0.
\tag{P6.2}
\]

最终解释路线：

\[
\boxed{
\widehat z_{m,t}^{\mathrm{PF}}
=
\widehat K_{\mathrm{PF},m}(U_t^-)
+
\widehat A_{\mathrm{res},m}(R_{m,t}^{\mathrm{mature},-}).
}
\tag{P6.3}
\]

完整输出：

\[
\boxed{
\widehat{\overline y}_{t+h_m}^{(W_m),\mathrm{PF}}
=
\overline y_t^{(W_{0,m})}
+
\widehat K_{\mathrm{PF},m}(U_t^-)
+
\widehat A_{\mathrm{res},m}(R_{m,t}^{\mathrm{mature},-}).
}
\tag{P6.4}
\]

## P7. K-only 退化

若

\[
\widehat A_{\mathrm{res},m}\equiv0,
\tag{P7.1}
\]

则 Physics-First 退化为 K-only：

\[
\widehat z_{m,t}^{\mathrm{PF}}
=
\widehat K_{\mathrm{PF},m}(U_t^-).
\tag{P7.2}
\]

登记：

```text
PHYSICS_FIRST_RESIDUAL_EXACT_ZERO
PHYSICS_FIRST_DEGENERATES_TO_K_ONLY
```

## P8. 残差解释边界

后置残差预测器只能解释为：

> 对已冻结物理输入层未解释的隐藏状态、未测扰动、测量动态和剩余可预测结构进行补偿。

禁止表述：

- AR 解释了功率；
- AR 替代了未通过认证的旋转作用；
- AR 系数是控制器或 plant 参数；
- 深度残差模型优于零模型即可升级为物理状态方程。

---

# 6. 路线 II：PRISM K-Joint AR

## J1. AR 与广义预测状态

K-Joint AR 的标准状态是过去输出历史；其广义形式允许严格过去的低维预测状态。每个目标头都注册独立的状态 profile：

\[
\alpha_m^{J}
=
(\Delta_m^{J},T_m^{J},\mathcal L_m^{J},\mathscr A_m^{J}),
\tag{J1.1}
\]

并定义

\[
S_{m,t}^{-}
=
\Psi_m^{\alpha_m^J}(y_{\le t},U_{\le t}).
\tag{J1.2}
\]

因此，不同提前量、目标窗口和状态频带的目标头不能复用同一个 AR 阶数、同一输出采样步长或同一状态历史。K-Joint AR 的“Joint”是同一目标头内部的联合优化，不是跨目标头共用一个 AR。

可以包含：

- 过去直径；
- 过去直径变化；
- 多尺度输出平均；
- 经过注册的通道专属历史；
- 严格成熟的历史误差；
- 低维神经或线性状态。

不得包含未来输入、未来目标、未成熟残差或测试统计量。

标准线性 K-Joint AR 写为

\[
\widehat z_{m,t}^{\mathrm{KJAR}}
=
\sum_j\widetilde{\mathcal K}_{j,m}[u_j](t)
+
\sum_{\ell=1}^{L_{y,m}}a_{m,\ell}\,y_{t-\ell},
\tag{J1.3}
\]

其中物理输入项与 AR 系数在同一预测目标下联合优化。更一般的 \(A_{J,m}(S_{m,t}^-)\) 只是这一路线的预测状态扩展。

## J2. 联合优化

\[
\boxed{
(\widehat K_{J,m},\widehat A_{J,m})
=
\arg\min_{K,A}
\left[
\frac1n\sum_t
(z_{m,t}-K(U_t^-)-A(S_{m,t}^-))^2
+
\mathcal R_{J,m}(K,A)
\right].
}
\tag{J2.1}
\]

最终：

\[
\widehat z_{m,t}^{\mathrm{KJAR}}
=
\widehat K_{J,m}(U_t^-)
+
\widehat A_{J,m}(S_{m,t}^-).
\tag{J2.2}
\]

## J3. K-Joint AR 的正式用途

允许报告：

- 总体 RMSE、MSE、MAE、\(R^2\)；
- 两个 outer 方向和 pooled 结果；
- 预测区间、训练数据量敏感性、资源成本；
- 输入/状态消融；
- 与深度网络、PLS、AR/ARX 等预测模型比较。

禁止报告：

- K-Joint AR 内部 \(K\) 的峰值就是物理时滞；
- K-Joint AR 内部 \(K\) 小就表示物理作用小；
- K-Joint AR 的 AR 部分属于某个未认证变量；
- K-Joint AR 总预测优胜证明物理层通过认证。

## J4. 两个 leaderboard

必须分开：

### Input-only

\[
\widehat z=f(U^-).
\]

不使用过去直径。

### Dynamic

\[
\widehat z=f(U^-,Y^-).
\]

允许过去输出或预测状态。

Input-only 与 Dynamic 不得混在一个排名中。

---

# 7. 数值估计、求解与低秩结构

## N1. 训练数据内预处理

以下操作只能在 outer training data 内完成：

- 标准化；
- PCA 联合升速；
- 抗混叠分支参数；
- 幅值域与 spline knots；
- lag/block basis；
- 正则强度；
- rank 和 support；
- early stopping；
- 模型选择。

验证和测试只能应用冻结参数。

## N2. 直接求解

对线性/固定 basis 模型，正规方程

\[
(\Phi^\top\Phi+nP)\widehat\theta
=
\Phi^\top z.
\tag{N2.1}
\]

小中型问题优先使用 FP64 Cholesky/QR/SVD；必须记录：

- KKT/正规方程残差；
- 条件数；
- 有效秩；
- 正则尺度；
- 解法和精度。

## N3. Matrix-free PCG

大设计矩阵使用算子：

\[
A(v)=\Phi^\top(\Phi v)+nPv.
\tag{N3.1}
\]

PCG 必须与 FP64 reference 在小规模问题上做 forward/adjoint、目标值、梯度和预测一致性审计。

## N4. 通道独立谱估计与自适应 rank

对式 (M3.4)，每个 \((j,m)\) 独立完成谱分解与 rank 选择：

\[
K_{j,m}^{(R)}(\tau,u)
=
\sum_{r=1}^{R_{j,m}}
\sigma_{j,m,r}q_{j,m,r}(\tau)f_{j,m,r}(u).
\tag{N4.1}
\]

优先比较：

1. Gram 白化后的 SVD；
2. FP64 ridge/QR/SVD reference；
3. 固定 rank 的交替最小二乘；
4. rank \(R_{j,m}=0,1,2,\ldots\) 的 nested validation；
5. rank margin、跨棒核稳定性和 common-support 证书。

这里的 rank 是通道内部复杂度，不定义跨通道公共时间基。任何跨通道共享时间基都必须作为独立扩展重新定义，并接受单独的可辨识性与认证审计，不能由预测效果反向推出。

## N5. 支持选择与固定支持 refit

正式流程：

1. training-only 候选 support；
2. one-SE 或稳定支持规则；
3. exact-zero 与非零模型并列；
4. 固定 support 后重新拟合；
5. outer test 只评估一次；
6. 不得通过 test 后删通道、改 horizon 或缩短区间。

## N6. OOD 幅值合同

训练核心域

\[
\mathcal I_j^{\mathrm{core}}
=[Q_{0.01},Q_{0.99}],
\tag{N6.1}
\]

拟合域

\[
\mathcal I_j^{\mathrm{fit}}
=[m_j-\delta_j,M_j+\delta_j].
\tag{N6.2}
\]

核心指标只报告 common-support 或明确 OOD 比例。禁止静默硬裁剪后把结果混入主指标。

有限带 \(C^1\) 延拓只属于部署合同，不创造域外可辨识性。

---

# 8. 时间因果、cross-fitting、purge 与不确定性

## C1. primitive support

对目标头 \(m\)，一个样本原点 \(t\) 使用的原始索引支持至少覆盖

\[
[t-L_m^{\max},\ t+h_m+W_m],
\tag{C1.1}
\]

其中

\[
L_m^{\max}
=
\max_j T_{j,m}
\]

还应计入状态锚点窗口和残差历史构造。

## C2. forward split 与 purge

第 \(k\) 个评价块原点为 \([s_k,e_k]\)。训练数据必须严格位于评价块之前，并保留

\[
G_{m,k}
\ge
L_m^{\max}+h_m+W_m+b_{m,k},
\tag{C2.1}
\]

其中 \(b_{m,k}>0\) 为额外依赖间隔。

禁止随机 K-fold，因为相邻窗口共享大量原始时间点。

## C3. moving-block bootstrap

主不确定性使用时间块重采样。block length 至少做下列敏感性：

\[
10,\ 22,\ 40,\ 60\ \mathrm{min}
\tag{C3.1}
\]

分别对应短相关、目标成熟、最长核心历史和更保守依赖尺度。

多 seed 深度模型应先固定 ensemble 规则后对时间块 bootstrap，或采用 seed–time 两层 bootstrap；不得把同一真实样本在多个 seed 下的预测直接当成独立观测。

## C4. 跨运行域外层验证

设可用运行域、晶棒、批次或设备集合为

\[
\mathcal E=\{e_1,\ldots,e_M\}.
\]

外层验证必须按完整运行域划分，而不是把同一运行域的相邻时间窗口随机拆分。对任意注册的训练—测试方向

\[
e_a\rightarrow e_b,\qquad a\neq b,
\tag{C4.1}
\]

均应分别报告预测与结构证书。结构认证不能只依赖 pooled 平均；当可用运行域数量有限时，应明确声明外层方向覆盖范围，而不能把有限方向外推为普遍跨域结论。

---

# 9. PRISM 认证门槛与声明矩阵

## G1. S1：尺度候选

至少满足：

- 两个 outer 方向 improvement 均为正；
- pooled MSE improvement \(\ge1\%\)；
- bootstrap 正改善概率 \(\ge0.80\)。

S1 只表示 profile 值得进入确认阶段。

## G2. S2：稳定结构候选

至少满足：

- 两个方向 improvement 均为正；
- pooled improvement \(\ge3\%\)；
- bootstrap 正改善概率 \(\ge0.95\)；
- 两方向核相关 \(\ge0.6\)；
- 主要支持区间重叠；
- common-support 内结果不显著下降。

## G3. C：输出历史之外的新颖性诊断

冻结尺度匹配状态预测器后，输入条件增量至少满足：

- 两方向均不为负；
- pooled improvement \(\ge0.5\%\)；
- bootstrap 正改善概率 \(\ge0.90\)；
- 不主要依赖异常段或 OOD。

C 是辅助诊断，不是模型层，也不改变 Physics-First 训练顺序。

## G4. 完整物理认证八条件

通道 \(j\) 在目标头 \(m\) 被称为可解释稳定响应，必须同时满足：

1. 双向跨棒改善；
2. 核形状、符号与主要峰值稳定；
3. 时间块 bootstrap 支持；
4. 时间错位 placebo 失败；
5. common-support 内仍有效；
6. 主要时滞支持重叠；
7. 排除慢工艺轨迹伪解释；
8. exact-zero 未被稳定选择。

并额外要求：

- solver/KKT 通过；
- Gram/Schur 条件通过；
- 样本和信号强度足够；
- rank/支持不依赖单一正则点；
- 若报告 structural rank，必须有 HS 误差和谱间隔 margin。

正式逻辑：

\[
\boxed{
\mathrm{Certified}_{j,m}
=S2_{j,m}\land C_1\land\cdots\land C_8
\land\mathrm{SolverOK}
\land\mathrm{SchurOK}.
}
\tag{G4.1}
\]

## G5. 状态代码

| 状态 | 含义 |
|---|---|
| `PHYSICAL_CERTIFIED` | 全部物理认证条件通过 |
| `PREDICTIVE_ONLY` | 预测有效，但不能作结构解释 |
| `DIRECTION_UNSTABLE` | 至少一个 outer 方向失败 |
| `EXACT_ZERO` | 零模型稳定入选 |
| `UNRESOLVED` | 证据不足或数值/可辨识条件失败 |
| `K_JOINT_AR_PREDICTION_ONLY` | K-Joint AR 总预测可用，内部不可解释 |

## G6. 论文声明矩阵

| 结果 | 允许声明 |
|---|---|
| 两方向预测均改善、核证书失败 | 输入型预测有效但核未认证（由实验报告判定） |
| pooled 改善但一方向为负 | 跨运行方向不稳定，不得称稳定物理响应 |
| exact-zero 稳定选择 | 注册协议未形成足够证据支持该通道（由实验报告判定） |
| K 证书和 rank margin 均通过 | 可报告闭环预测响应核和结构 rank |
| K 通过、rank margin 不足 | 报告核与 rank interval，不报唯一 rank |
| K-Joint AR 胜出 | 报告总体预测优胜，不解释内部 K |
| residual exact-zero | 冻结 Physics-First 候选退化为 K-only；不自动表示 K 已认证 |
| 多棒数量仅为 2 | 不声明普遍跨棒泛化或 SOTA |

---

# 10. 理论陈述与经验结果的分离原则

## 10.1 本文所陈述的对象

本文只陈述以下内容：

1. 预测目标、信息集、状态锚点和多时间尺度 profile 的定义；
2. 多通道 Urysohn 映射、商空间、Gram/Schur 条件与结构可辨识性；
3. Urysohn 优先路线与 K-Joint AR 路线的模型语义；
4. 通道独立审计、兼容目标头联合重拟合、冻结、OOF 残差和成熟条件；
5. exact-zero、线性、低秩和完整 Urysohn 之间的嵌套关系；
6. 时间因果、purge、bootstrap、外层验证与声明门槛；
7. CPU/CUDA/PLC 的数值实现与部署误差合同。

这些内容属于理论定义、推导条件或验证合同，不依赖某一次实验中哪个模型获胜。

## 10.2 不属于本文的内容

下列内容必须放在独立实验报告中，不得写入理论正文：

- 任何具体模型的 MSE、RMSE、MAE、\(R^2\) 或相对改善数值；
- 某个模型在 CPU、GPU 或某个数据集上的排名；
- 某个通道在特定数据上通过、失败、归零或方向不稳定的结论；
- 某个 profile 被选为最佳尺度的经验结果；
- 某个残差模型在特定任务中为 exact-zero 的结果；
- 某种低秩、共享时间基或非线性结构因效果较好而被视为物理机制的推断；
- 对特定晶棒、工艺阶段、设备或数据规模的泛化结论。

## 10.3 理论对实验结果保持中立

PRISM 不预设：

- 哪个物理通道一定有效；
- 哪个通道一定应当归零；
- 哪个 profile 一定优于其他 profile；
- Urysohn 优先路线一定优于 K-Joint AR；
- K-Joint AR 一定优于纯数据驱动模型；
- residual AR 一定非零或一定为零；
- 线性、rank-1、rank-\(R\) 或完整 Urysohn 中哪一类一定胜出。

理论只规定：这些结论应当通过什么信息集、什么时间尺度合同、什么外层验证和什么结构证书来产生。

## 10.4 经验结果的引用方式

实验报告可以引用本文中的定义、路线和状态代码，并将具体结果映射为：

\[
\texttt{EXACT\_ZERO},\quad
\texttt{PREDICTIVE\_VALIDATED},\quad
\texttt{PHYSICAL\_CERTIFIED},\quad
\texttt{DIRECTION\_UNSTABLE},\quad
\texttt{UNRESOLVED}.
\]

但这些状态必须由实验报告给出，理论文档本身不填写任何数据集的最终状态。

---

# 11. 正式实验流程

## E1. Stage 0：物理时间尺度预审计

每个通道输出：

- 实际非零更新间隔；
- 中位保持时间；
- 一阶差分幅值；
- ACF 衰减；
- 频谱能量；
- 与不同频带直径变化的滞后相关；
- 两棒共同幅值支持；
- 稳定段、断点和 OOD。

Stage 0 只验证 profile 合理性，不宣布物理响应成立。

## E2. Stage 1：线性多尺度扫描

每个 profile 只运行：

1. persistence；
2. mean drift；
3. local trend；
4. scale-matched AR；
5. raw-history AR；
6. channel-only linear K；
7. 条件新颖性诊断；
8. K-Joint AR/ARX 预测对照。

目标是每通道最多保留少量候选尺度，而不是直接运行高自由度非线性模型。

## E3. Stage 2：结构确认

候选 profile 运行：

- 双向跨棒；
- moving-block bootstrap；
- common-support；
- 时间错位 placebo；
- 核符号、峰值、质心、主要支持；
- 正则敏感性；
- 时间分辨率敏感性；
- detrend/局部差分/慢趋势排除；
- exact-zero 选择；
- Gram/Schur/KKT 审计。

## E4. Stage 3：通道内部升阶、兼容目标头与双路线确认

只有 Stage 2 通过的候选才允许：

1. 增加非线性幅值响应；
2. 对每个通道比较 rank-1、通道独立 rank-\(R\) 与完整 Urysohn；
3. 对兼容目标头的认证通道固定集合联合重拟合；
4. 冻结物理层；
5. 为每个兼容目标头单独注册并拟合 mature residual AR，含 exact-zero；
6. 为同一目标头单独注册并运行 K-Joint AR，不跨头复用 AR 配置，并与路线 I 分开报告。

## E5. 基线矩阵

Input-only 至少包括：

- persistence-derived zero change；
- ridge distributed lag；
- Dynamic-PLS；
- NLinear/DLinear；
- compact Urysohn；
- KAN/graph/KAN 适配基线。

Dynamic 至少包括：

- AR；
- ARX/NARX；
- Joint-K+AR；
- GRU/LSTM；
- TCN；
- Temporal Autoencoder；
- Transformer/PatchTST/TimesNet/S4D 等。

所有模型使用相同目标、样本掩码和 outer 方向。

## E6. 评估指标

至少报告：

- MSE、RMSE、MAE、\(R^2\)；
- relative improvement vs persistence；
- relative improvement vs AR；
- pooled 与两个方向；
- moving-block bootstrap 区间与正改善概率；
- seed 中位数与 IQR；
- 参数量、训练时间、推理时延、峰值显存；
- train-fraction robustness；
- OOD/common-support 指标。

---

# 12. 第三层：CUDA / PLC 多速率部署

## D1. 多速率在线调度器

PLC 主循环保持原始快速采样时钟 \(\Delta_0\)。各分支在自己的触发周期更新：

```text
每 2 s:
    读取原始传感器与质量标志
    更新因果聚合器

每 10 s:
    更新联合升速分支缓存/状态

每 30 s:
    更新晶转和埚转分支缓存/状态

每 120 s:
    更新主加热功率分支缓存/状态

当目标头需要输出:
    读取当前直径锚点
    汇总已认证 K 通道
    读取成熟残差状态（若非零）
    输出未来窗口平均直径预测
```

## D2. LUT + FIR

谱/低秩模态写为

\[
K_{j,m,R}(\tau,u)
=
\sum_{r=1}^{R_{D,j,m}}
\sigma_{j,m,r}q_{j,m,r}(\tau)f_{j,m,r}(u).
\tag{D2.1}
\]

在线先计算

\[
v_{j,m,r,t}=f_{j,m,r}(u_{j,t}),
\tag{D2.2}
\]

再用环形缓冲：

\[
c_{j,m,r,t}
=
\sum_{\ell=0}^{L_{j,m,r}-1}
q_{j,m,r}(\ell)v_{j,m,r,t-\ell}.
\tag{D2.3}
\]

## D3. 稳定状态空间

可将时滞核近似为

\[
q(0)=D,
\qquad
q(k+1)=C^\top A^kB,
\tag{D3.1}
\]

在线：

\[
s_{t+1}=As_t+Bv_t,
\tag{D3.2}
\]

\[
c_t=C^\top s_t+Dv_t,
\tag{D3.3}
\]

要求

\[
\rho(A)<1.
\tag{D3.4}
\]

## D4. 指数/Erlang 递推

一阶指数：

\[
s_t=\rho s_{t-1}+(1-\rho)v_t.
\tag{D4.1}
\]

多级级联构成 Erlang 型核，可用于低成本热惯性或长记忆实现。高阶谱模态允许 signed readout。

## D5. 部署误差预算

总误差可分解为

\[
\epsilon_{\mathrm{total}}
\le
\epsilon_{\mathrm{resample}}
+
\epsilon_{\mathrm{basis}}
+
\epsilon_{\mathrm{rank}}
+
\epsilon_{\mathrm{tail}}
+
\epsilon_{\mathrm{LUT}}
+
\epsilon_{\mathrm{quant}}
+
\epsilon_{\mathrm{state}}.
\tag{D5.1}
\]

若 \(|v_t|\le M_v\)，截断尾部误差满足

\[
|e_{\mathrm{tail}}|
\le
M_v\|q_{\mathrm{tail}}\|_1.
\tag{D5.2}
\]

若一元响应查表误差 \(|f-\widetilde f|\le\epsilon_f\)，则

\[
|e_{\mathrm{LUT}}|
\le
\epsilon_f\|q\|_1.
\tag{D5.3}
\]

## D6. 固定点与 OOD

每个通道必须记录：

- 输入量化范围；
- 状态量化范围；
- 累加器位宽；
- 饱和策略；
- OOD 标志；
- finite-band \(C^1\) 延拓区；
- 超出延拓区后的拒绝/降级策略。

硬裁剪不得静默发生。

---

# 13. 可复现性、打包与审计合同

每次正式运行必须保存：

```text
registered_profiles.yaml
sample_ids_and_hashes
train_only_scalers_and_pca
outer_split_definition
inner_folds_and_purge
model_config
solver_log
KKT_and_condition_report
per_direction_predictions
prediction_hashes
bootstrap_config_and_results
support_and_rank_certificate
OOD_and_common_support_report
resource_metrics
final_decision.json
MANIFEST.json
SHA256SUMS
```

要求：

1. 清理旧输出，禁止新旧结果混包；
2. 原始 Excel 不进入代码仓库或结果包；
3. manifest 覆盖全部文件；
4. 逐文件 SHA256 校验；
5. 压缩包 round-trip 校验；
6. 隐私与绝对路径检查；
7. tag、branch、commit 与结果包明确对应；
8. 报告中的 PASS 必须能由底层指标复算，不得只相信状态字段。

---

# 14. PRISM 的正式理论命题汇总

## 命题 1：通道专属多速率不改变因果信息集

只要每个重采样块完全位于 \(( -\infty,t]\)，多速率表示是 \(\mathcal F_t\)-可测的，因此不会引入未来泄漏。

## 命题 2：目标头兼容性

不同通道可以使用不同 \(\Delta_j,T_{j,m},\mathcal B_{j,m}\)，但只有预测目标 \((h_m,W_m,W_{0,m})\) 相同，通道贡献才可在同一加性方程中联合解释。

## 命题 3：物理优先投影唯一、核未必唯一

\(g_m^{\mathrm{PF}}=P_{\overline{\operatorname{ran}\mathcal A_m}}z_m\) 在 \(L^2\) 中唯一；完整核只在商空间意义下唯一，有限核坐标还需要 injectivity。

## 命题 4：多输入归因需要 Schur 条件

即使总输入预测有效，若通道 Schur 补退化，per-channel 核仍不可分离。多变量预测优胜不能替代通道可辨识证书。

## 命题 5：两阶段冻结定义了解释优先级

Physics-First 的解释优先级来自“先拟合并冻结 K、后拟合成熟残差”的训练合同，不来自联合损失中的软权重。任何最终联合回调都会把模型重新变为 K-Joint AR 路线。

## 命题 6：成熟残差条件保证在线可实现

只有满足 \(s+h_m+W_m\le t\) 的残差才属于时刻 \(t\) 的信息集。违反该条件的 residual AR 是目标泄漏。

## 命题 6A：AR 配套是目标头专属而不是全局共享

设两个目标头 \(m_1\neq m_2\)，且二者至少在 \(h\)、\(W\)、输出状态频带或历史覆盖上不同。则除非额外证明存在跨头充分统计量，否则不能把同一个有限阶 AR 配置同时视为二者的尺度匹配状态模型。PRISM 因而要求分别注册 \(\alpha_{m_1}\) 与 \(\alpha_{m_2}\)。

该命题是实验合同，而不是声称真实系统在不同尺度上拥有互不相关的内部状态。它只防止用一个任意 AR 配置偏袒或压制某个物理 profile。

## 命题 7：K-Joint AR 总体可预测但内部不可归因

当物理与状态函数空间相交时，K-Joint AR 分解存在等价变换，故只有总预测受监督，内部组件不具唯一物理意义。

## 命题 8：部署只能近似已识别算子

LUT、FIR/IIR、状态空间和定点量化可在可控误差内近似有限模型，但不能提升原始数据的可辨识性或外推范围。

---

# 15. 归档路线与禁止回退项

以下内容保留为历史或消融，不属于 PRISM 主线：

```text
Q_LAYER_AS_MODEL_LAYER = ARCHIVED
AR_FIRST_CONDITIONAL_RESIDUALIZATION_AS_MAINLINE = ARCHIVED
ALL_CHANNELS_SHARE_ONE_HORIZON_AND_CADENCE = REJECTED
GAMMA_DELAY_PRIOR_AS_REQUIRED_STRUCTURE = ARCHIVED
DYNAMIC_LAG_SCORER_AS_REQUIRED_STRUCTURE = ARCHIVED
FULL_URYSOHN_SURFACE_AS_DEFAULT = REJECTED
POSTHOC_RANK1_PROJECTION_AS_PHYSICAL_PROOF = REJECTED
FAILED_CHANNEL_ROUTED_TO_ARX = REJECTED
FORCED_NONZERO_RANK = REJECTED
JOINT_LOW_RMSE_AS_K_CERTIFICATE = REJECTED
SHARED_PRIVATE_DECOMPOSITION = REJECTED
SHARED_TIME_BASIS_AS_PRISM_CORE = NOT_ADOPTED
POWER_THERMAL_STATE_BANK_AS_FORMAL_COMPONENT = NOT_ADOPTED
K_JOINT_AR = FORMAL_ROUTE_II
```

---

# 16. 理论论文级表述

PRISM 的理论主张是：

> 提出一种面向被动闭环工业时间序列的物理优先、多通道尺度专属、多速率 Urysohn 响应辨识理论。该理论允许每个物理通道独立注册采样尺度、预测提前量、目标窗口、历史范围和时滞基；在 Urysohn 优先路线中，先进行通道级审计与兼容目标头联合重拟合，再冻结物理层，并仅使用严格成熟的 OOF 残差构造可精确归零的后置预测器；在 K-Joint AR 路线中，输入响应项与预测状态项可为工程预测联合优化，但其内部解释权分配不具物理归因资格。理论进一步区分预测有效性、结构可辨识性和开放环因果性，并给出时间因果、外层验证、结构证书与部署误差合同。

理论正文不得自行宣称：

- 某个具体物理通道已经有效或无效；
- 某个具体 profile 已经是最佳时间尺度；
- 某种模型在特定数据集上取得最低误差；
- 某个 residual 分支已经非零或精确归零；
- 某个核已经获得跨设备、跨批次或开放环因果解释；
- 某种低秩或非线性结构已由预测效果证明为真实物理机制。

这些都属于实验报告的职责。

---

# 17. 最终语义链

\[
\boxed{
\begin{aligned}
&\textbf{注册层：}
&&\{\pi_{j,r}\}
\rightarrow
\text{通道专属物理时间尺度}
\\
&\textbf{理论层：}
&&z_m
\rightarrow
g_m^{\mathrm{PF}}
\rightarrow
[K_m]
\rightarrow
\text{Schur/injectivity certificate}
\\
&\textbf{模型层：}
&&\text{exact-zero}
\subset
\text{linear lag}
\subset
\text{per-channel low-rank Urysohn}
\subset
\text{nonlinear Urysohn}
\\
&\textbf{训练层：}
&&\text{channel audit}
\rightarrow
\text{fixed certified set refit}
\rightarrow
\text{freeze }K
\rightarrow
\text{mature residual or zero}
\\
&\textbf{预测层：}
&&\text{Urysohn-first explanation route}
\parallel
\text{K-Joint AR route}
\\
&\textbf{执行层：}
&&\text{FP64/CUDA solve}
\rightarrow
\text{LUT+FIR/IIR/state-space}
\rightarrow
\text{PLC deployment}.
\end{aligned}
}
\]

PRISM 的三条最终原则为：

\[
\boxed{
\text{先按物理通道确定正确时间尺度，再讨论输入是否有效。}
}
\]

\[
\boxed{
\text{先冻结可认证物理响应，再允许剩余状态补偿。}
}
\]

\[
\boxed{
\text{预测成功、结构可辨识和开放环因果是三个不同层次。}
}
\]

---

# 附录 A：核心符号表

| 符号 | 含义 |
|---|---|
| \(j\) | 物理通道索引 |
| \(r\) | 通道内部 profile 索引 |
| \(m\) | 兼容目标头索引 |
| \(\Delta_j\) | 通道建模步长 |
| \(h_m\) | 预测提前量 |
| \(W_m\) | 未来输出平均窗口 |
| \(W_{0,m}\) | 当前锚点平均窗口 |
| \(T_{j,m}\) | 通道历史范围 |
| \(\mathcal B_{j,m}\) | 通道时滞块/基 |
| \(z_m(t)\) | 未来平均变化目标 |
| \(\mathcal H_{j,m}\) | 通道核 Hilbert 空间 |
| \(\mathcal A_m\) | 多通道 Urysohn 映射 |
| \(g_m^{\mathrm{PF}}\) | 物理优先最佳输入型响应 |
| \(K_{j,m}\) | 通道时滞—幅值核 |
| \(\widehat{\mathcal C}_{m}^{\mathrm{train}}\) | outer test 前冻结的训练候选通道集合 |
| \(\mathcal S_m\) | outer 确认后通过完整物理证书的通道集合 |
| \(r_{m,s}^{\mathrm{OOF}}\) | OOF 物理残差 |
| \(A_{\mathrm{res},m}\) | 后置成熟残差预测器 |
| \(\pi^A_{j,r}\) | 与通道物理 profile 配套的尺度匹配诊断 AR profile |
| \(A^{\mathrm{diag}}_{j,r}\) | 通道 profile 条件增量审计所用的冻结尺度匹配 AR |
| \(A^{\mathrm{res}}_m\) | 路线 I 中目标头专属的成熟残差预测器 |
| \(A^J_m\) | 路线 II 中目标头专属的联合 AR/预测状态项 |
| \(S_{m,t}^{-}\) | K-Joint AR 的严格过去预测状态 |
| \(R_S,R_P,R_D\) | 结构、预测、部署 rank |

# 附录 B：理论来源与版本边界

- 理论蓝本：`OPS_UOI_PS_AR_RAPHU_CUDA_PLC_Three_Layer_Complete_System_v2_0.md`；
- 当前修订只采用本对话确认的通道专属多时间尺度、尺度匹配 AR、Urysohn 优先与 K-Joint AR 双路线；
- 本文不引用任何具体 CPU/GPU 实验结果包、模型排名或性能数值作为理论组成部分；
- 本文不采用高于 v2.0 的 OPS-UOI 理论文档中的公共—私有分解、共享时滞主结构或功率预设热状态银行。

# 附录 C：版本修订摘要

```text
OPS-UOI v2.0 blueprint
    ↓ 保留 operator-first / Hilbert / Gram / Schur / CUDA / PLC
    ↓ 删除 Q layer / AR-first mainline / unified horizon
    ↓ 加入 channel-specific multirate profiles
    ↓ 为每个物理 profile 注册 scale-matched AR companion
    ↓ 加入 current-state anchor + future-window change target
    ↓ 加入 Physics-First K -> mature residual exact-zero route
    ↓ 加入 compatible target-head joint refit
    ↓ 保留 K-Joint AR 作为正式路线 II
    ↓ 排除公共—私有分解与功率专属预设热状态银行
    ↓ 将理论陈述与具体实验结果彻底分离
PRISM Theory v1.3
```
