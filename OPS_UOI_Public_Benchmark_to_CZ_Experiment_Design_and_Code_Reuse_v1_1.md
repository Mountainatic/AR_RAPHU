# OPS-UOI / Predictive-State Spectral AR-RAPHU
## 公开权威数据集 → CZ 工业数据的分层实验设计与现有代码迁移方案 v1.1

> **日期**：2026-07-27  
> **文档性质**：预注册式实验总方案 + 现有代码复用与启动规范  
> **目标**：先在公开、可复现、来源权威的动态系统与工业软测量数据上验证方法，再进入 CZ 等径阶段数据；同时严格区分“数学证明”“受控真值验证”“公开基准验证”和“最终工业案例”。

---

# 0. 总结性决定

后续实验不采用：

\[
\text{合成数据}\rightarrow\text{直接进入单根 CZ 数据}
\]

这一条过窄路线，而采用四级证据链：

\[
\boxed{
\text{受控真值实验}
\rightarrow
\text{公开物理系统基准}
\rightarrow
\text{公开闭环工业基准}
\rightarrow
\text{CZ 工业数据}
}
\]

四级证据的职责分别为：

| 证据层 | 主要作用 | 能否替代数学证明 |
|---|---|---:|
| 受控真值实验 | 精确验证 quotient、核、rank、正交余项和收敛趋势 | 否 |
| 公开物理系统基准 | 检查非线性动态预测、谱压缩和样本外泛化 | 否 |
| 公开闭环工业基准 | 检查强 AR、反馈相关、多变量共线和 predictive state | 否 |
| CZ 工业数据 | 证明方法对目标工艺有实际价值 | 否 |

顶层 OPS-UOI 定理的正确关系是：

\[
\boxed{
\text{定理由数学证明成立；实验只能检验条件、实现和有限样本预言。}
}
\]

---

# 1. 核心研究问题

公开基准实验必须回答以下七个问题。

## Q1. 贡献可辨识与完整核可辨识是否被正确区分

当

\[
\mathcal D K_1=\mathcal D K_2
\]

但

\[
K_1\neq K_2
\]

时，系统是否输出：

```text
QUOTIENT_ONLY_IDENTIFIED
```

而不是伪造唯一核？

## Q2. 强 AR 与闭环相关下，外生贡献能否稳定估计

比较：

\[
\text{naive XAR}
,\quad
Y\text{-only residualization}
,\quad
\text{double residualization}.
\]

检查双残差是否减少 AR 抢占外生贡献和共线伪归因。

## Q3. 完整 Urysohn 母空间是否优于固定低秩假设

比较：

- rank-1；
- fixed rank-2；
- full kernel；
- adaptive spectral rank。

重点检查高 rank、幅值相关时滞、硬非线性和过程噪声。

## Q4. 分辨率是否能由稳定性选择，而不是由已知真值峰宽指定

检查 Lepski / nested-sieve 选择是否在隐藏真值和公开测试集上接近最优风险。

## Q5. 结构 rank、SVD 预测 rank 与最优预测 rank是否被正确命名

公开数据默认只允许报告：

\[
R_{P,\mathrm{svd}}^\star
\]

及其稳定区间。只有 K 层证书通过时才报告：

\[
R_S^\star.
\]

## Q6. 真实过程创新、测量噪声和模型错设如何影响结论

过程噪声基准、硬非线性基准和真实传感器数据分别承担不同压力测试。

## Q7. 模型能否从通用动态系统迁移到工业软测量

公开基准通过后，再在 CZ 数据上验证：

\[
\Delta_{X\mid AR}(h)
=
L_{AR}(h)-L_{XAR}(h)
\]

以及跨晶棒泛化。

---

# 2. 数据集选择标准

进入核心实验矩阵的数据集必须满足至少四项。

## 2.1 来源权威

优先级：

1. 有持久 DOI 的官方研究数据仓库；
2. 原作者维护的官方 benchmark 网站或代码仓库；
3. 原始论文配套模拟器；
4. UCI 等长期维护的公共仓库；
5. 论文中常用但只有非官方镜像的数据，不能进入核心确认集。

## 2.2 时间顺序明确

必须能够：

- 保持原始时间顺序；
- 构造因果历史窗口；
- 避免随机行划分；
- 定义独立训练、验证和测试记录。

## 2.3 与理论问题相匹配

数据组合必须覆盖：

- 物理非线性；
- 过程噪声；
- 强输出历史；
- 多变量闭环；
- 部分可观测；
- 有限样本；
- 真实软测量。

## 2.4 可复现

必须记录：

```text
official_source
download_date
license
raw_file_hash
extraction_hash
loader_version
split_manifest
```

---

# 3. 最终选定的数据集组合

---

## D0. OPS-UOI 受控真值生成器

### 类型

项目自有、完全已知真值的动态系统生成器。

### 地位

不是公开数据集，但不可删除。公开数据没有完整真核，不能精确验证：

- quotient 反例；
- 核 HS 误差；
- rank 真实值；
- nuisance 二阶余项；
- 收敛率。

### 主要任务

- `E-Q-vs-K`
- `E2B-ID-SPACE`
- `E3-ORTH-AR`
- `E-SCALE-MIX`
- `E-LEPSKI-MASK`
- `E-RANK-MARGIN`
- `E-MISSPEC-INTERACTION`

---

## D1. Parallel Wiener–Hammerstein Benchmark

### 官方来源

- Nonlinear Benchmark 官方网站；
- 4TU.ResearchData；
- 数据 DOI：`10.4121/12950081`；
- 配套论文：Schoukens et al., *Automatica*, 2015。

### 数据性质

- 真实电子实验装置；
- 两条并联 Wiener–Hammerstein 分支；
- 提供 multisine estimation、multisine test 和 increasing-amplitude test；
- SISO，但包含多分支非线性动态。

### 在本项目中的角色

这是公开数据中最适合检查以下问题的基准：

1. fixed rank-1 / rank-2 是否会欠表示；
2. full kernel 是否获得稳定样本外收益；
3. HS-SVD 截断的预测损失如何随 rank 下降；
4. increasing-amplitude test 下是否发生幅值外推失效；
5. spectral model 是否优于普通 parallel Hammerstein 基线。

### 合法声明

允许：

- 样本外预测性能；
- kernel estimate 的跨记录稳定性；
- SVD 预测 rank；
- full kernel 相对低秩的容量收益。

不允许仅凭“两条物理分支”就声称 Urysohn 真实结构 rank 必定为 2。物理分支数与当前 HS 核 rank 不是同一个定义。

---

## D2. Wiener–Hammerstein Process Noise Benchmark

### 官方来源

- Nonlinear Benchmark 官方网站；
- 4TU.ResearchData；
- 数据 DOI：`10.4121/12952124`。

### 数据性质

- 真实电子实验系统；
- 主要过程噪声在静态非线性之前进入；
- 输入和输出测量通道另有较小测量噪声；
- 提供估计与测试记录。

### 在本项目中的角色

专门检查：

1. 过程创新是否被错误当作传感器异常；
2. AR / predictive state 是否能吸收已发生过程创新；
3. MSE、Huber 敏感性和前端质量层的边界；
4. full kernel 在内部过程噪声下是否稳定；
5. rank profile 的 bootstrap 不确定性。

### 关键原则

不能把大残差全部清洗掉。该数据的核心挑战就是过程噪声，预处理只能处理明确的文件错误或测量无效标志。

---

## D3. Cascaded Tanks with Overflow

### 官方来源

- Nonlinear Benchmark 官方网站；
- 4TU.ResearchData；
- 数据 DOI：`10.4121/12960104`；
- CC BY-SA 4.0。

### 数据性质

- 真实流体实验装置；
- 输入为泵控制；
- 输出为液位测量；
- 同时包含软非线性和溢流造成的硬非线性；
- 数据记录相对较短；
- 官方提供 estimation/test 数据。

### 在本项目中的角色

检查：

1. 短记录下的偏差—方差权衡；
2. Lepski 分辨率选择；
3. hard nonlinearity 下加性 Urysohn 的边界；
4. 一步预测与多步 rollout 的差异；
5. full kernel 是否因容量过大而过拟合；
6. OOD/overflow 区间能否被正确标记。

### 关键输出

- official test RMSE；
- non-overflow 与 overflow 分区误差；
- chosen sieve；
- full 与 adaptive rank 的差异；
- OOD coverage。

---

## D4. Silverbox Benchmark

### 官方来源

- Nonlinear Benchmark 官方网站；
- 配套 ECC 论文 DOI：`10.23919/ECC.2013.6669201`。

### 数据性质

- 真实电子 Duffing 振子；
- 线性动态与反馈三次非线性组合；
- 提供 multisine 训练记录和多种测试记录；
- 官方 benchmark 维护结果列表。

### 在本项目中的角色

检查：

1. 强历史状态下 AR-only 与 XAR；
2. 反馈非线性；
3. in-distribution 与 arrow/extrapolation 测试；
4. state initialization window；
5. predictive-state 长度和 Urysohn 历史长度的分离。

---

## D5. Tennessee Eastman Process（TEP）

### 官方/原始来源

- Downs–Vogel 原始工业过程控制基准；
- Braatz/Russell/Chiang 修改的闭环模拟代码与训练测试数据；
- 参考仓库：`jkitchin/tennessee-eastman-profbraatz`；
- 原始与修改代码均保留许可说明。

### 数据性质

- 大型闭环化工过程模拟器；
- 41 个测量变量；
- 11 或 12 个 manipulated variables，具体按采用的 simulator 版本冻结；
- 多个工况模式；
- 21 个标准扰动/故障；
- 可生成多随机种子、多独立轨迹；
- 适合正常工况、闭环反馈和产品组分软测量。

### 主软测量目标

主目标分别建模：

\[
Y_t^{(G)}=\mathrm{XMEAS}(40)
\]

和

\[
Y_t^{(H)}=\mathrm{XMEAS}(41),
\]

即产品流中的 G/H 组分。

主输入只采用：

- 连续过程测量 XMEAS(1–22)；
- manipulated variables XMV；
- 不使用其他组分分析 XMEAS(23–39)，避免分析仪信息泄漏。

### 标签更新问题

产品组分分析存在较慢采样/分析周期。若 simulator 在中间时刻以保持值输出，不能把保持值当作新标签。

必须生成：

```text
label_value
label_update_mask
label_age
```

主评估只在真实分析更新时刻进行。

### 固定阶段原则

主确认只使用：

```text
Mode 1 normal operation
```

不同 mode 不混合为同一动力学。跨 mode 仅作为 OOD/阶段切换压力测试。

### 三种激励环境

#### TEP-NAT

标准闭环控制器、正常运行、独立噪声种子。

目的：Q 层条件贡献和真实闭环预测。

#### TEP-EXC

在安全约束内，对选定 setpoint 或 manipulated variable 叠加独立小幅 PRBS / multisine。

目的：提高 Schur/injectivity，检查 K 层。

#### TEP-COL

对多个输入施加高度相关扰动。

目的：构造变量归因退化，验证：

```text
QUOTIENT_ONLY_IDENTIFIED
```

### 主要理论任务

- 多变量 Schur 可辨识；
- 强 AR 双残差；
- forward purge gap；
- Q 层与 K 层；
- NAT/EXC/COL 对比；
- 多轨迹 outer test；
- 结构结论随激励变化的合法边界。

---

## D6. Siemens Industrial Benchmark（IB）

### 官方来源

- Siemens 原作者公开 benchmark；
- 论文 DOI：`10.1109/SSCI.2017.8280935`；
- 官方代码仓库：`siemens/industrialbenchmark`；
- Java/Python 实现和 Gym 接口。

### 数据性质

- 不对应某一真实工厂，但由工业控制经验设计；
- 连续状态和控制；
- 部分可观测；
- 延迟效应；
- 隐状态；
- 复杂随机性；
- 可生成任意数量独立 rollout。

### 可观测变量

包括：

- velocity；
- gain；
- shift；
- set point；
- consumption；
- fatigue。

### 预测任务

分别预测：

\[
Y_t^{(c)}=\text{consumption}_{t+h}
\]

和

\[
Y_t^{(f)}=\text{fatigue}_{t+h}.
\]

外生历史包括：

- velocity；
- gain；
- shift；
- set point；
- action increments。

predictive state 使用 consumption/fatigue 历史。

### 三种 rollout

#### IB-NAT

安全策略附近的有界随机动作。

#### IB-EXC

velocity、gain、shift 使用独立、有界 PRBS。

#### IB-COL

三个 steering 高度相关。

### 主要理论任务

- 部分可观测 predictive state；
- delayed effects；
- 强 AR；
- 多输入 Schur；
- 多 rollout outer generalization；
- NAT 与 EXC 的 Q/K 层差异；
- 固定 residualizer 与 refit residualizer 的区别。

### 阶段定义

set point 的不同大区间视为不同 operating regime。主结构声明只在预注册 set-point 区间内成立，跨区间只做 OOD。

---

## D7. UCI Gas Turbine CO and NOx Emission Dataset

### 官方来源

- UCI Machine Learning Repository；
- 数据 DOI：`10.24432/C5WC95`；
- CC BY 4.0。

### 数据性质

- 土耳其燃气轮机真实运行数据；
- 2011–2015；
- 36,733 个小时聚合样本；
- 时间顺序已排序但没有逐行日期；
- 9 个环境/过程输入；
- 目标为 CO 与 NOx；
- 官方建议前三年用于训练/交叉验证，后两年用于测试。

### 主任务

分别预测：

\[
Y_t^{CO},\qquad Y_t^{NOx}.
\]

输入：

- AT；
- AP；
- AH；
- AFDP；
- GTEP；
- TIT；
- TAT；
- TEY；
- CDP。

### 三条可用性轨道

#### GT-X

严格软测量：

\[
Y_{t+h}=f(X_{\le t})
\]

不使用过去排放分析值。

#### GT-XAR

假设连续排放分析仪可用，加入过去 CO/NOx。

只用于研究 predictive state，不作为严格替代分析仪场景。

#### GT-SPARSE-Y

模拟周期性标定：只有每 \(q\) 个小时获得一次真排放标签，其余时刻只使用 last-observed value、age 和 mask。

### 在本项目中的角色

这是公开组合中最接近传统工业软测量的真实数据。

主要检查：

- 从系统辨识基准到真实软测量的迁移；
- 跨年分布漂移；
- X-only 与 XAR 的价值；
- OOD 和置信度；
- 动态模型是否真的优于静态模型。

### 限制

小时聚合会削弱短时动态；因此它不能证明亚小时内核结构，也不能作为 closed-loop K 层主证据。

---

# 4. 暂不进入核心确认集的数据

## 4.1 Debutanizer Column

该数据是经典真实炼油软测量基准，文献通常报告 7 输入、1 个丁烷含量输出、2,394 个样本。

但目前公开网络中经常出现的是：

- 二次转载；
- 论文附件镜像；
- GitHub/Kaggle 复制；
- 缺少统一持久 DOI 和原始 checksum。

处理原则：

\[
\boxed{
\text{在找到可核验的原始公开源之前，只作补充实验，不进入核心确认集。}
}
\]

## 4.2 Sulfur Recovery Unit

SRU 是高度相关的软传感器案例，但常见公开版本的来源、样本数和预处理不一致。

同样要求：

- 原始作者/机构来源；
- 变量定义；
- 时间顺序；
- checksum；
- 许可。

完成来源审计后再加入。

---

# 5. 统一数据处理协议

## 5.1 原始数据冻结

每个数据集建立：

```text
data/raw/<dataset>/<version>/
data/interim/<dataset>/<pipeline_hash>/
data/processed/<dataset>/<split_hash>/
```

记录：

```json
{
  "dataset": "...",
  "official_source": "...",
  "doi": "...",
  "license": "...",
  "download_utc": "...",
  "raw_sha256": "...",
  "loader_commit": "...",
  "preprocessing_config_hash": "..."
}
```

## 5.2 不允许随机行划分

所有数据保持时间顺序。

划分优先级：

1. 官方 estimation/test；
2. 独立物理记录；
3. 独立 simulator rollout；
4. 年份；
5. expanding-window 时间块。

## 5.3 warm-up

测试记录开头的前

\[
L_{\mathrm{warm}}
=
\max(L_x,L_y,L_{\mathrm{state}})
\]

只用于初始化，不计主指标。

若官方 benchmark 指定 initialization window，以官方值为最低要求。

## 5.4 归一化

所有 scaler 只在训练集拟合：

\[
\widehat z
=
\frac{z-\widehat\mu_{\mathrm{train}}}
{\widehat\sigma_{\mathrm{train}}}.
\]

测试超出训练域：

- 不静默裁剪；
- 标记 OOD；
- 继续按预注册 extrapolation 规则运行。

## 5.5 缺失与质量

必须保留：

```text
raw_value
model_value
quality_flag
missing_mask
age_since_measurement
```

不能利用未来值做无延迟插值。

---

# 6. 统一模型矩阵

不是每个模型都在每个数据集上全量运行。分为核心和扩展。

## 6.1 核心基线

所有动态数据集至少运行：

1. mean / persistence；
2. AR ridge；
3. FIR / ARX ridge；
4. NARX-MLP；
5. rank-1 PS-AR-RAPHU；
6. fixed rank-2；
7. full-kernel PS-AR-RAPHU；
8. adaptive spectral PS-AR-RAPHU。

## 6.2 结构匹配基线

- PWH：parallel Wiener–Hammerstein；
- Silverbox：polynomial NARX / nonlinear state-space；
- Tanks：NARX / neural state-space；
- TEP：PLS、subspace/state-space、NARX；
- IB：GRU/TCN；
- Gas Turbine：PLS、SVR、MLP、GRU/TCN。

## 6.3 深度学习基线

公开工业数据最终至少包括：

- GRU；
- TCN；
- Transformer 或稀疏 attention；
- KAN/GADKN-inspired dynamic baseline。

所有可训练模型获得相同：

- development 数据；
- tuning 次数；
- early stopping 规则；
- seed 数；
- test 访问次数。

---

# 7. 预测任务协议

## 7.1 直接多视野预测

统一采用：

\[
\widehat Y_{t+h\mid t}
=
F_h(X_{\le t},Y_{\le t}).
\]

不使用未来 \(X\)。

默认样本视野候选：

\[
h\in\{1,5,10,20\}.
\]

但最终每个数据集根据采样率映射为具有物理意义的视野，并写入参数卡：

```text
⟦P-HORIZON-<DATASET>⟧
```

TEP 组分目标按真实 analyzer update interval 定义视野。

## 7.2 三轨

每个适用数据集都运行：

- AR-only；
- X-only；
- XAR。

并报告：

\[
\Delta_{X\mid AR}(h)
=
L_{AR}(h)-L_{XAR}(h).
\]

## 7.3 一步预测与 free-run 分开

- one-step/direct prediction：每个时刻使用真实可用历史；
- free-run：模型递归使用自身输出。

主论文不得把两者混为同一指标。

---

# 8. 顶层定理与实验映射

| 顶层对象 | 精确真值验证 | 公开基准支持 |
|---|---|---|
| quotient 可辨识 | D0 `E-Q-vs-K` | TEP-COL、IB-COL |
| K 层 injectivity | D0、TEP-EXC | PWH 稳定性、TEP Schur |
| Neyman 正交 | D0 `E3-ORTH-AR` | TEP、IB |
| mixing / 样本率 | D0、多 simulator rollout | TEP、IB 的样本缩放 |
| Lepski | D0 hidden truth | Tanks、PWH、Silverbox |
| structural rank | D0 精确 | PWH 仅稳定性支持 |
| predictive SVD rank | 所有动态数据 | 所有动态数据 |
| 过程噪声边界 | D0 | WH Process Noise |
| 加性错设 | D0 | Tanks overflow |
| 真实软测量 | 无完整真核 | UCI Gas Turbine、CZ |

重要结论：

\[
\boxed{
\text{公开数据可以支持顶层理论的经验后果，但不能取代 D0 真值实验和数学证明。}
}
\]

---

# 9. 分阶段实验计划

---

## Phase PB0：数据源、loader 与协议冻结

### 任务

- 下载官方数据；
- 记录 license/DOI；
- 生成 raw checksum；
- 编写统一 loader；
- 校验官方 split；
- 输出数据卡；
- 测试因果窗口索引；
- 验证无 train/test overlap。

### 交付物

```text
PUBLIC_BENCHMARK_DATA_AUDIT_v1.zip
```

### Gate

所有核心数据均满足：

```text
SOURCE_VERIFIED
HASH_VERIFIED
TIME_ORDER_VERIFIED
SPLIT_VERIFIED
```

---

## Phase PB1：公开物理 SISO 动态系统

### 数据集

- Parallel Wiener–Hammerstein；
- Wiener–Hammerstein Process Noise；
- Cascaded Tanks；
- Silverbox。

### 目的

先排除多变量和 closed-loop nuisance，把问题集中到：

- 动态表达能力；
- full kernel；
- adaptive rank；
- Lepski；
- process noise；
- extrapolation。

### 实验

#### PB1.1 统一 one-step/direct benchmark

运行所有核心模型和官方 split。

#### PB1.2 rank profile

对 full kernel 计算：

\[
\tau_R
=
\frac{
\sqrt{\sum_{r>R}\sigma_r^2}
}{
\sqrt{\sum_{r\ge1}\sigma_r^2}
}
\]

和测试贡献损失。

#### PB1.3 nested resolution

候选 sieve 只在 development 内选择，test 一次评估。

#### PB1.4 process-noise sensitivity

只在 WH Process Noise：

- raw MSE；
- Huber sensitivity；
- 不做清洗；
- 比较 predictive-state 更新。

#### PB1.5 extrapolation

- PWH increasing-amplitude；
- Silverbox arrow；
- Tanks overflow。

### PB1 通过条件

必须同时满足：

1. full kernel 在至少一个公认高难基准上显著优于 rank-1；
2. adaptive rank 在主要测试记录上达到 full kernel 的预注册预测预算；
3. fixed rank-2 不能被默认宣称普适；
4. OOD 区间被单独报告；
5. process-noise 数据不因前置“去尖峰”被破坏。

主预算：

\[
\epsilon_{\mathrm{pred}}=0.05
\]

次级预算：

\[
0.02,\ 0.10.
\]

来源标签：

```text
⟦P-RANK-BUDGET⟧
```

---

## Phase PB2：公开闭环多变量基准

### 数据集

- Tennessee Eastman；
- Siemens Industrial Benchmark。

### 目的

验证 OPS-UOI 顶层最重要的部分：

- predictive-state conditioning；
- double residualization；
- Schur 可辨识；
- Q/K 层；
- NAT/EXC/COL；
- outer rollout generalization。

### PB2.1 多轨预测

对每个目标运行：

- AR；
- X；
- XAR；
- naive XAR；
- double-residual XAR。

### PB2.2 NAT/EXC/COL

对每种激励环境报告：

\[
\lambda_{\min}(\widehat\Gamma_m),
\]

\[
\lambda_{\min}(\widehat S_{j,m}),
\]

\[
\Delta_{X\mid AR},
\]

kernel/contribution stability。

### PB2.3 冻结与重估 residualizer

在 NAT 估计：

\[
\widehat\pi_{\mathrm{NAT}}.
\]

然后：

1. 固定 \(\widehat\pi_{\mathrm{NAT}}\) 到 EXC/COL；
2. 在 EXC/COL 重新估计 \(\widehat\pi\)。

两组结果分别报告，不混成同一 estimand。

### PB2.4 trajectory outer test

训练、验证、测试按 rollout seed 分开。

不能把同一轨迹的窗口随机分到不同集合。

### PB2 通过条件

1. double residualization 在强 AR 条件下不劣于 naive，并改善贡献稳定性；
2. EXC 的 Schur/injectivity 优于 COL；
3. COL 能正确触发 quotient-only 状态；
4. 多轨迹 test 仍有正的外生增量；
5. TEP 组分标签没有 hold-value 泄漏；
6. 不跨 mode 声称同一核。

---

## Phase PB3：真实公开软测量

### 数据集

UCI Gas Turbine CO/NOx。

### 划分

严格采用：

- 2011–2013：development；
- 2014–2015：test。

development 内：

- 2011–2012：训练；
- 2013：验证/模型选择。

每年单独 warm-up，不跨年度构造窗口。

### 实验

- GT-X；
- GT-XAR；
- GT-SPARSE-Y；
- static vs dynamic；
- AR-RAPHU vs PLS/SVR/MLP/GRU/TCN；
- annual drift；
- OOD；
- calibration/uncertainty。

### PB3 通过条件

1. 至少一个目标在 2014–2015 获得稳定正 skill；
2. dynamic model 的收益不能只来自泄漏或年份编码；
3. XAR 与 X-only 的差异被正确解释；
4. 跨年误差和 OOD 单独报告；
5. 若静态基线同样好，诚实结论应是该小时聚合任务不需要复杂时滞模型。

---

## Phase PB4：理论专项确认

在 D0、TEP 和 IB 上执行：

- `E-Q-vs-K`
- `E2B-ID-SPACE`
- `E2B-NAT-Q`
- `E2B-PERM-DIAG`
- `E3-ORTH-AR`
- `E-CF-GAP`
- `E-SCALE-MIX`
- `E-LEPSKI-MASK`
- `E-RANK-MARGIN`
- `E-MISSPEC-INTERACTION`
- `E-BOOT-RADIUS`

PB4 与 PB1–PB3 可以部分并行，但最终理论声明必须等 PB4 完成。

---

## Phase CZ0：CZ 数据准入

只有在 PB1–PB3 至少完成主结果后进入正式 CZ confirmation。

先做：

- 时间戳；
- 采样周期；
- 单位；
- 直径测量方式；
- 测量质量；
- 等径阶段；
- 多晶棒 ID；
- 控制/输出对齐；
- 标签后处理审计。

---

## Phase CZ1：单棒开发

单棒只用于：

- pipeline integration；
- within-run temporal prediction；
- history/resolution pilot；
- measurement-noise audit；
- Q 层探索。

不作为最终跨棒泛化证据。

---

## Phase CZ2：多棒确认

按晶棒 outer split：

\[
\text{train rods}
\rightarrow
\text{validation rods}
\rightarrow
\text{untouched test rods}.
\]

最终输出：

- AR/X/XAR；
- \(\Delta_{X\mid AR}\)；
- predictive SVD rank；
- kernel/contribution stability；
- K 层证书；
- 与 GADKN-inspired、NARX、Hammerstein、TCN/GRU 的比较；
- 计算与部署成本。

---

# 10. 统一指标

## 10.1 预测

\[
RMSE,\quad
MAE,\quad
NRMSE,\quad
R^2.
\]

相对基线 skill：

\[
Skill
=
1-
\frac{L_{\mathrm{model}}}
{L_{\mathrm{baseline}}}.
\]

## 10.2 多视野

每个 \(h\) 单独报告，不平均掩盖远期失效。

## 10.3 可辨识性

\[
\lambda_{\min}(\widehat\Gamma_m),
\quad
\operatorname{cond}(\widehat\Gamma_m),
\]

\[
\lambda_{\min}(\widehat S_{j,m}),
\quad
\operatorname{cond}(\widehat S_{j,m}).
\]

## 10.4 谱

\[
\sigma_r,
\quad
\tau_R,
\quad
R_{P,\mathrm{svd}}^\star,
\]

以及在合法时：

\[
R_S^\star
\quad\text{或}\quad
[R_L,R_U].
\]

## 10.5 稳定性

- seed；
- fold；
- rollout；
- operating range；
- resolution；
- bootstrap。

## 10.6 计算

- wall time；
- peak RAM/VRAM；
- PCG iteration；
- objective residual；
- CPU/GPU difference；
- deployment MAC；
- memory；
- quantization error。

---

# 11. 统计报告

## 11.1 随机模型

至少 5 个 confirmation seeds。

开发调试 seed 与 confirmation seed 分离。

## 11.2 置信区间

时间序列指标使用：

- moving block bootstrap；
- stationary bootstrap；
- trajectory bootstrap。

不使用普通 IID bootstrap。

## 11.3 多数据集汇总

除逐数据集结果外，使用：

- average rank；
- paired win/loss/tie；
- normalized skill；
- critical difference 作为补充。

不只报告把所有点拼接后的总 RMSE。

---

# 12. 超参数与防泄漏顺序

每个数据集严格执行：

1. 冻结官方/outer test；
2. 在 development 中选择 history；
3. 选择 basis resolution；
4. 选择 regularization；
5. 估计 full kernel；
6. 在 validation/selection block 选择 rank；
7. 冻结模型；
8. 一次性 test；
9. test 后不修改主模型。

所有非普适参数必须写参数卡：

```text
⟦P-HISTORY-DATASET⟧
⟦P-HORIZON-DATASET⟧
⟦P-RESOLUTION-DATASET⟧
⟦P-REG-DATASET⟧
⟦P-RANK-BUDGET⟧
⟦P-BOOTSTRAP-DATASET⟧
⟦P-OOD-DATASET⟧
```

---

# 13. 实现顺序

## Step 1：统一数据接口

```python
@dataclass
class DynamicDataset:
    u: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray | None
    sequence_id: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    label_mask: np.ndarray
    quality_mask: np.ndarray
    metadata: dict
```

## Step 2：CPU FP64 reference

先实现：

- causal window；
- AR/X/XAR；
- full kernel；
- double residualization；
- Gram/Schur；
- rank profile；
- Lepski；
- bootstrap。

## Step 3：公开基准结果

先完成 PB1，再完成 PB2/PB3。

## Step 4：CUDA

当 CPU estimator 和数据张量定义冻结后，再做 CUDA 等价实现。

## Step 5：CZ

公开基准暴露的问题修正后，再进入 CZ confirmation。

---

# 14. 预期论文结果结构

## 结果 I：受控理论实验

证明实现满足：

- quotient/K 层边界；
- orthogonal remainder；
- sample scaling；
- rank interval。

## 结果 II：公开物理系统

证明：

- full kernel 的必要性；
- adaptive compression；
- process noise；
- hard nonlinearity；
- extrapolation。

## 结果 III：公开闭环工业系统

证明：

- predictive state；
- double residualization；
- Schur 可辨识；
- Q/K 状态切换；
- rollout generalization。

## 结果 IV：公开真实软测量

证明：

- 跨年排放软测量；
- 动态方法是否真正必要；
- OOD 与稀疏标签适应。

## 结果 V：CZ

证明：

- 对目标工艺的实际价值；
- 多晶棒泛化；
- 与现有 CZ 方法比较；
- 工程可部署性。

---

# 15. 关键停止线

## Stop-A：公开动态基准无收益

若 full/adaptive AR-RAPHU 在所有物理动态基准上均不优于简单 NARX/ARX，停止扩大理论声明，先检查模型错设。

## Stop-B：双残差没有稳定作用

若 TEP/IB 强 AR 环境中双残差不改善贡献稳定性，停止 Neyman-orthogonal 经验主张，检查 nuisance 和 estimand 实现。

## Stop-C：rank 不稳定

若 rank interval 始终极宽，不输出单值结构 rank，论文改为 full-kernel + uncertainty-aware compression。

## Stop-D：Gas Turbine 动态结构无价值

若静态 PLS/SVR 与动态模型相当，保留该结果作为负对照，不强行宣称所有软测量都需要动态核。

## Stop-E：CZ 只有单棒

只能声称：

```text
within-run temporal generalization
```

不能声称跨棒工业泛化。

---

# 16. 下一步立即执行的实验包

下一版代码与结果包应只覆盖 PB0 + PB1，避免一次铺开所有数据集。

## 必做

1. 官方数据下载与 hash；
2. 统一 loader；
3. Parallel Wiener–Hammerstein；
4. Wiener–Hammerstein Process Noise；
5. Cascaded Tanks；
6. Silverbox；
7. AR / X / XAR；
8. rank-1 / fixed rank-2 / full / adaptive；
9. official test protocol；
10. 数据、模型、结果和环境打包。

## 结果包名称

```text
OPS_UOI_PUBLIC_BENCHMARK_PB1_RESULTS_bundle.zip
```

## PB1 返回文件

```text
README.md
CHANGELOG.md
config/
src/
tests/
data_manifests/
split_manifests/
results/PWH/
results/WHPN/
results/CascadedTanks/
results/Silverbox/
tables/
figures/
logs/
environment/
manifest.json
SHA256SUMS.txt
```

PB1 通过后，下一包进入：

```text
PB2_TEP_IB_CLOSED_LOOP
```

然后：

```text
PB3_UCI_GAS_TURBINE
```

最后才是正式 CZ confirmation。

---

# 17. 官方来源与引用

## Nonlinear Benchmark

- 官方网站：`https://www.nonlinearbenchmark.org/benchmarks`
- 官方 Python loader：`https://github.com/GerbenBeintema/nonlinear_benchmarks`

## Parallel Wiener–Hammerstein

- 数据 DOI：`10.4121/12950081`

## Wiener–Hammerstein Process Noise

- 数据 DOI：`10.4121/12952124`

## Cascaded Tanks

- 数据 DOI：`10.4121/12960104`

## Silverbox

- Wigren and Schoukens, 2013 ECC；
- DOI：`10.23919/ECC.2013.6669201`

## Tennessee Eastman

- 参考代码：`https://github.com/jkitchin/tennessee-eastman-profbraatz`
- Downs and Vogel, “A Plant-Wide Industrial Process Control Problem,” 1993。
- Russell, Chiang and Braatz 的闭环修改版与故障诊断数据。

## Industrial Benchmark

- 官方代码：`https://github.com/siemens/industrialbenchmark`
- Hein et al., 2017；
- DOI：`10.1109/SSCI.2017.8280935`

## UCI Gas Turbine CO/NOx

- UCI Dataset ID 551；
- DOI：`10.24432/C5WC95`
- CC BY 4.0。

---


# 19. 从当前 v0.3.4 实验代码启动公开基准方案

本章不是重新设计一套与现有工程无关的新代码，而是规定：

\[
\boxed{
\text{以当前 spectral v0.3.4 为唯一主干，}
\quad
\text{保留既有证据，增量扩展到 PB0/PB1。}
}
\]

当前可靠起点为：

```text
SPECTRAL_PS_AR_RAPHU_V034_RANK_PROFILE_RESULTS.zip
```

该结果包已经给出：

```text
FULL_STRUCTURAL_SURFACE_CAPACITY: PASS
ADAPTIVE_RANK_PROFILE_VALIDATED
UNIVERSAL_RANK2_HYPOTHESIS: REJECTED
NEXT_ALLOWED_STAGE: ALLOW_E2B
```

因此新的公开基准工作不应退回重新实现 Stage1，也不应重新选择 v0.3.3/v0.3.4 的合成实验平滑参数。正确做法是：

1. 冻结 v0.3.4 为只读证据；
2. 复用其 spectral 核心；
3. 增加统一数据接口和公开数据 loader；
4. 增加 PB1 runner；
5. 保留旧实验入口和测试；
6. 在公开数据上重新选择数据集相关的历史长度、分辨率和正则化；
7. 不把合成实验的 \(64/48/32\times28\) 直接当作公开数据默认值。

---

## 19.1 当前代码资产审计

v0.3.4 结果包中已经存在以下可直接复用资产。

### 核心 spectral 模块

| 当前文件 | 已有职责 | PB1 复用方式 |
|---|---|---|
| `spectral/amplitude_domain.py` | fit/core/run 幅值域和越界审计 | 直接复用，增加 dataset metadata 输入 |
| `spectral/spline_basis.py` | 幅值与时滞 B-spline | 直接复用 |
| `spectral/design.py` | 构造 spectral 设计 | 把 synthetic 输入改为通用窗口接口，不改变数学定义 |
| `spectral/penalties.py` | 时滞/幅值平滑惩罚 | 直接复用 |
| `spectral/solver.py` | FP64 强凸求解、KKT 检查 | PB1 首阶段保持不变 |
| `spectral/crossfit.py` | 时间 cross-fitting | 扩展 sequence ID、官方记录和 purge gap |
| `spectral/nuisance.py` | \(\mu,\pi\) nuisance | 复用接口，增加可选 AR/ridge nuisance |
| `spectral/gram_svd.py` | Gram 白化谱 | 直接复用 |
| `spectral/rank_profile.py` | 结构谱与有效 rank | 直接复用，但公开数据默认只解释为稳定性结果 |
| `spectral/rank_ladder.py` | rank ladder 与 modal gain refit | 直接复用 |
| `spectral/predictive_rank.py` | 自然输入上的预测 rank | 重命名输出为 \(R_{P,\mathrm{svd}}^\star\) |
| `spectral/rank_bootstrap.py` | block-bootstrap rank | 复用并增加 sequence-aware resampling |
| `spectral/weighted_projection.py` | 分辨率投影与误差分解 | 复用 |
| `spectral/resolution_roles.py` | mother/structural/predictive 角色 | 复用概念，不复用固定数字 |
| `spectral/operator_metrics.py` | 算子/贡献误差 | 公开数据删去依赖 truth 的指标 |
| `spectral/representation_certificate.py` | 表示能力证书 | 在公开数据改为稳定性证书 |
| `spectral/contracts.py` | 实验 gate 与状态 | 扩展 PB0/PB1 contract |
| `protocol_config.py` | 协议配置与解析 | 升级为 schema version 6 |
| `spectral/metrics.py` | 预测指标 | 增加 skill、分记录和 OOD 分区指标 |

### 继续保留但仅用于 D0 的模块

| 文件 | 处理方式 |
|---|---|
| `synthetic.py` | 保留，继续生成 D0 受控真值 |
| `spectral/synthetic_components.py` | 保留，不能用于公开数据预处理 |
| `spectral/truth_spectrum.py` | 只在已知真核时调用 |
| `spectral/scenario_registry.py` | 保留为 synthetic registry；另建 public dataset registry |
| `capacity_matrix.py` | D0 与 SPACE 容量实验继续使用 |
| `truth_rank_profile.csv` 相关逻辑 | 公开数据禁用 |

### 已有工具入口

当前入口为：

```text
tools/run_spectral_job.py
tools/run_spectral_suite.py
tools/summarize_spectral_suite.py
tools/report_spectral_v034.py
```

其中 `run_spectral_suite.py` 已经较大，并混合多个 schema 版本。PB1 不应继续把所有数据下载、loader、基线和报告逻辑塞进该文件。应新增独立 runner，并调用现有 spectral 库。

### 当前已知限制

1. schema 5 只实现：
   - `R0`
   - `E2A_SR`
   - `E2A_SRB`
   - `E2A_P_NAT`
   - `E2A_P_PERM`
2. `E2B/E3` 在当前 runner 中尚未实现；
3. `--device cuda` 目前只是 CLI 参数，spectral 求解并没有真实 CUDA dispatch；
4. v0.3.4 的固定角色分辨率：
   \[
   64_{\rm id}\times28,\quad48\times28,\quad32\times28
   \]
   只属于已完成合成实验；
5. 当前 operator metrics 中部分指标依赖真核，不能直接用于公开数据；
6. 当前 runner 以单一合成序列为中心，尚无多记录、官方 split、年份 split 和 sequence ID。

---

## 19.2 应从哪个代码包起步

### 首选基线

使用**实际生成 v0.3.4 结果的完整工程树**：

```text
V20/V21 base repository
+ spectral v0.3.4 source overlay
```

原因：

- 完整工程树包含依赖、旧基线、环境和打包脚本；
- v0.3.4 包更像可复核结果快照，虽然包含 spectral 源码，但未必包含全部 legacy baseline 和环境文件。

### 仅有 v0.3.4 包时

它已经包含 PB1 核心所需的：

- `src/ar_raphu/spectral/`
- `src/ar_raphu/synthetic.py`
- `tools/`
- `tests/`
- `configs/`

因此可重建 PB1 的 spectral reference，但需补：

- `pyproject.toml` 或 requirements；
- 通用数据 loader；
- 传统与深度基线；
- 下载和许可证记录；
- 完整打包脚本。

### 禁止基线

不得从以下对象重新开始：

- 早期 Stage1 单 Gamma 路线；
- 已确认存在 representation stop 的旧单核支持实验；
- 只含结果、没有对应源码 hash 的临时目录；
- 已被 v0.3.4 科学重解释否定的 universal rank-2 gate。

---

## 19.3 先冻结现有证据，再创建新工作区

推荐目录：

```text
OPS_UOI_WORKSPACE/
├── evidence_readonly/
│   ├── SPECTRAL_PS_AR_RAPHU_V03_RESULTS.zip
│   ├── SPECTRAL_PS_AR_RAPHU_V031_CORE_RESULTS.zip
│   ├── SPECTRAL_PS_AR_RAPHU_V032_CAPACITY_RESULTS.zip
│   ├── spectral_v033.zip
│   └── SPECTRAL_PS_AR_RAPHU_V034_RANK_PROFILE_RESULTS.zip
├── repo/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── results/
└── return/
```

初始化命令示例：

```bash
set -euo pipefail

mkdir -p OPS_UOI_WORKSPACE/{evidence_readonly,repo,data/raw,data/interim,data/processed,results,return}

cp SPECTRAL_PS_AR_RAPHU_V034_RANK_PROFILE_RESULTS.zip \
   OPS_UOI_WORKSPACE/evidence_readonly/

cd OPS_UOI_WORKSPACE
sha256sum evidence_readonly/*.zip > evidence_readonly/SHA256SUMS.txt

# 首选：复制实际生成 v0.3.4 的完整 repo。
cp -a /path/to/STAGE1_DUAL_SOLVER_V20_bundle/. repo/

# 将 v0.3.4 的冻结源码覆盖到新分支，而不覆盖旧结果。
mkdir -p /tmp/v034_overlay
unzip -q evidence_readonly/SPECTRAL_PS_AR_RAPHU_V034_RANK_PROFILE_RESULTS.zip \
      -d /tmp/v034_overlay
cp -a /tmp/v034_overlay/src/. repo/src/
cp -a /tmp/v034_overlay/tools/. repo/tools/
cp -a /tmp/v034_overlay/tests/. repo/tests/
cp -a /tmp/v034_overlay/configs/. repo/configs/
```

进入 Git 分支：

```bash
cd repo
git status
git switch -c public-benchmark-pb1
```

不得将 `evidence_readonly/` 中的旧结果复制到新 `results/` 后修改。

---

## 19.4 复现当前 v0.3.4 基线

在修改任何代码前，必须先证明工作区能复现现有接口。

```bash
cd OPS_UOI_WORKSPACE/repo

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

python -m pytest -q
```

最低要求：

```text
ALL_EXISTING_TESTS_PASS
```

然后只做一个快速只读 smoke，不重新生成正式 v0.3.4 结论：

```bash
python tools/run_spectral_job.py \
  --config configs/spectral_v034.yaml \
  --experiment R0 \
  --stage development \
  --device cpu \
  --force
```

确认：

- lag orientation；
- spline basis；
- solver KKT；
- Gram SVD；
- rank profile；
- bootstrap tests；

均未因迁移被破坏。

对已有正式 v0.3.4 结果，优先校验原包 hash，不要求重新跑全部 20 confirmation seeds。

---

## 19.5 目标代码结构

新增代码必须围绕现有 spectral 核心展开：

```text
repo/
├── src/ar_raphu/
│   ├── spectral/                  # 原有核心，尽量小改
│   ├── datasets/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── download.py
│   │   ├── lineage.py
│   │   ├── windowing.py
│   │   ├── splits.py
│   │   ├── scaling.py
│   │   ├── quality.py
│   │   └── loaders/
│   │       ├── pwh.py
│   │       ├── wh_process_noise.py
│   │       ├── cascaded_tanks.py
│   │       ├── silverbox.py
│   │       ├── tennessee_eastman.py
│   │       ├── industrial_benchmark.py
│   │       └── uci_gas_turbine.py
│   ├── benchmarks/
│   │   ├── protocol.py
│   │   ├── task_registry.py
│   │   ├── evaluator.py
│   │   ├── result_schema.py
│   │   ├── selection.py
│   │   └── reporting.py
│   └── baselines/
│       ├── persistence.py
│       ├── ar_arx.py
│       ├── narx.py
│       ├── parallel_hammerstein.py
│       └── deep_sequence.py
├── configs/public_benchmarks/
│   ├── pb0_data_audit.yaml
│   ├── pb1_pwh.yaml
│   ├── pb1_whpn.yaml
│   ├── pb1_tanks.yaml
│   └── pb1_silverbox.yaml
├── tools/
│   ├── download_public_benchmarks.py
│   ├── audit_public_benchmarks.py
│   ├── run_public_benchmark_job.py
│   ├── run_public_benchmark_suite.py
│   ├── summarize_public_benchmark_suite.py
│   └── package_public_benchmark_results.py
└── tests/
    ├── test_dataset_contract.py
    ├── test_public_split_integrity.py
    ├── test_sequence_windowing.py
    ├── test_no_future_x.py
    ├── test_train_only_scaling.py
    ├── test_public_runner_smoke.py
    └── test_legacy_v034_regression.py
```

### 设计原则

- `datasets/` 只负责数据语义，不包含模型；
- `benchmarks/` 只负责编排，不重写 spectral 算法；
- `spectral/` 保持数学核心；
- `baselines/` 与主模型共享相同 split、scaler 和指标；
- `tools/` 只做 CLI；
- 所有结果由统一 schema 输出。

---

## 19.6 统一 DynamicDataset 接口

第一项新实现不是下载数据，而是冻结通用数据合同：

```python
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class DynamicDataset:
    x: np.ndarray                 # [n_time, n_x]
    y: np.ndarray                 # [n_time, n_y]
    timestamps: np.ndarray | None
    sequence_id: np.ndarray       # [n_time]
    split: np.ndarray             # train / val / test / warmup
    label_mask: np.ndarray        # [n_time, n_y]
    quality_mask: np.ndarray      # [n_time, n_x+n_y]
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    metadata: dict[str, Any]
```

必须满足：

1. 一个对象可以包含多个独立记录；
2. 任何窗口不能跨 `sequence_id`；
3. test split 不能参与 scaler、basis、history、rank 选择；
4. `label_mask=False` 的时刻不能被当作新标签；
5. `quality_mask` 不自动删除真实过程变化；
6. 时间戳缺失时仍保留原始采样序号；
7. 官方 split 优先于自定义 split。

---

## 19.7 现有 design.py 的适配方式

当前 synthetic runner 通常直接给定数组。公开数据适配层应先产生统一监督样本：

```python
@dataclass(frozen=True)
class WindowedTask:
    x_history: np.ndarray     # [n_sample, p, L_x]
    y_history: np.ndarray     # [n_sample, L_y]
    target: np.ndarray        # [n_sample]
    origin_index: np.ndarray
    sequence_id: np.ndarray
    split: np.ndarray
    ood_mask: np.ndarray
```

再把：

```text
WindowedTask.x_history
```

送入现有 `spectral/design.py`。

禁止在 `design.py` 中加入：

- 数据下载；
- 年份切分；
- 缺失值插补；
- train/test scaler；
- benchmark 特殊规则。

这样才能保证同一个 spectral design 同时服务 D0、PB1、PB2、PB3 和 CZ。

---

## 19.8 crossfit.py 的最小必要扩展

现有 `crossfit.py` 可保留基本接口，但必须新增：

```python
make_forward_folds(
    origin_index,
    sequence_id,
    split,
    *,
    n_folds,
    primitive_left_support,
    forecast_horizon,
    extra_mixing_gap,
)
```

每个 fold 输出：

```text
nuisance_train_indices
evaluation_indices
purged_indices
primitive_support_bounds
```

必须通过以下测试：

1. 不跨 sequence；
2. 不使用未来原点训练 nuisance；
3. primitive 时间支持不重叠；
4. purge gap 使用：
   \[
   G_n=L_\star+h+b_n;
   \]
5. test 只在模型冻结后出现。

PB1 的 SISO official split 可以先不启用复杂多折 cross-fitting，但接口必须从第一版支持 sequence 和 gap，避免 PB2 时重写。

---

## 19.9 solver.py 的复用边界

PB1 第一阶段不得修改核心求解算法。保持：

- FP64；
- 强凸 ridge/smooth penalty；
- KKT relative residual；
- CPU reference。

只允许新增一个薄适配器：

```python
fit_spectral_from_windowed_task(
    task,
    basis_config,
    penalty_config,
    nuisance_config,
    solver_config,
)
```

返回统一结果：

```python
@dataclass
class SpectralFit:
    coefficients: np.ndarray
    intercept: float
    prediction: dict[str, np.ndarray]
    gram_lag: np.ndarray
    gram_amp: np.ndarray
    spectra: dict
    solver_diagnostics: dict
    provenance: dict
```

CUDA、mixed precision 和 matrix-free 重写不进入 PB1 首包。

---

## 19.10 rank 模块的公开数据适配

公开数据没有 `truth_rank_profile.csv`。因此 rank pipeline 分两类输出。

### 总能输出

- estimated normalized spectrum；
- rank ladder；
- \(R_{P,\mathrm{svd}}^\star\)；
- block-bootstrap interval；
- fold/record stability；
- weak-operator flag。

### 只有 D0 才输出

- truth rank；
- estimated-vs-truth rank error；
- exact HS kernel error；
- truth tail curve error。

### 只有 K 层证书通过才允许输出

- structural rank；
- per-variable structural kernel；
- rank margin recovery。

代码上应在 `contract.json` 中写：

```json
{
  "truth_available": false,
  "k_level_certificate": false,
  "structural_rank_claim_allowed": false,
  "predictive_svd_rank_claim_allowed": true
}
```

---

## 19.11 schema version 6 配置

PB1 不改写 `spectral_v034.yaml`，新增独立配置。

示例：

```yaml
schema_version: 6
suite: OPS_UOI_PUBLIC_BENCHMARK_PB1
stage: development

dataset:
  id: parallel_wiener_hammerstein
  version: 4tu_12950081
  root: data/raw/pwh
  official_split: true
  target_column: output
  input_columns: [input]
  sequence_policy: official_record
  license_required: true
  hash_required: true

task:
  track: XAR
  horizons: [1, 5, 10, 20]
  use_future_x: false
  history:
    lx_grid: [16, 32, 64, 128]
    ly_grid: [1, 4, 8, 16, 32, 64]

basis:
  lag:
    role: mother_first
    candidates:
      - {type: discrete_identity}
      - {type: cubic_bspline, count: 32}
      - {type: cubic_bspline, count: 48}
  amplitude:
    type: cubic_bspline
    count_grid: [16, 20, 24, 28, 32, 40]

models:
  - persistence
  - ar_ridge
  - arx_ridge
  - narx_mlp
  - rank1_ar_raphu
  - fixed_rank2_ar_raphu
  - full_spectral_ar_raphu
  - adaptive_spectral_ar_raphu

selection:
  split: validation
  lepski_enabled: true
  rank_budgets: [0.10, 0.05, 0.02]
  primary_rank_budget: 0.05
  test_access: once

solver:
  dtype: float64
  device: cpu
  kkt_relative_residual: 1.0e-8

output:
  root: results/public_benchmarks/pb1/pwh
  save_predictions: true
  save_coefficients: true
  save_data_lineage: true
```

注意：

- `L_x,L_y` 不再固定为 64/32；
- 公开数据的历史长度必须由 development 选择；
- mother/structural/predictive 角色保留，但分辨率重新认证；
- `rank_budgets` 可以复用已预注册的 10%/5%/2%，因为这是误差语义，不是数据分辨率。

---

## 19.12 PB0 的实际启动

### 第一步：只建数据审计，不训练模型

```bash
python tools/download_public_benchmarks.py \
  --config configs/public_benchmarks/pb0_data_audit.yaml \
  --dataset all_pb1

python tools/audit_public_benchmarks.py \
  --config configs/public_benchmarks/pb0_data_audit.yaml
```

预期目录：

```text
data/raw/
├── pwh/
├── whpn/
├── cascaded_tanks/
└── silverbox/

data_manifests/
├── pwh.json
├── whpn.json
├── cascaded_tanks.json
└── silverbox.json
```

PB0 必须先生成：

```text
SOURCE_VERIFIED
HASH_VERIFIED
LICENSE_RECORDED
OFFICIAL_SPLIT_VERIFIED
TIME_ORDER_VERIFIED
```

任何一项失败，该数据集不能进入 PB1 confirmation。

---

## 19.13 PB1 的实际启动顺序

### 19.13.1 单元测试

```bash
python -m pytest -q \
  tests/test_dataset_contract.py \
  tests/test_public_split_integrity.py \
  tests/test_sequence_windowing.py \
  tests/test_no_future_x.py \
  tests/test_train_only_scaling.py \
  tests/test_legacy_v034_regression.py
```

### 19.13.2 每个数据集 smoke

```bash
for ds in pwh whpn cascaded_tanks silverbox; do
  python tools/run_public_benchmark_job.py \
    --config "configs/public_benchmarks/pb1_${ds}.yaml" \
    --stage smoke \
    --device cpu
done
```

smoke 只运行：

- 一个短记录；
- 一个 horizon；
- 一个 history；
- persistence / ARX / full spectral；
- 单 seed。

smoke gate：

```text
LOADER_PASS
NO_LEAKAGE_PASS
SOLVER_KKT_PASS
OUTPUT_SCHEMA_PASS
```

### 19.13.3 development

```bash
python tools/run_public_benchmark_suite.py \
  --config-dir configs/public_benchmarks \
  --suite PB1 \
  --stage development \
  --device cpu \
  --workers 8
```

development 允许：

- 选择 history；
- 选择 basis；
- 选择 penalty；
- 调整实现错误；
- 使用 development seeds。

不得查看 official test 的模型排名。

### 19.13.4 protocol freeze

development 完成后生成：

```text
results/public_benchmarks/pb1/PB1_PROTOCOL_FREEZE.json
```

至少包含：

```json
{
  "datasets": {},
  "selected_histories": {},
  "selected_resolutions": {},
  "selected_regularization": {},
  "rank_budgets": [0.10, 0.05, 0.02],
  "confirmation_seeds": [],
  "test_access_count": 0,
  "source_commit": "...",
  "config_hashes": {}
}
```

### 19.13.5 confirmation

```bash
python tools/run_public_benchmark_suite.py \
  --config-dir configs/public_benchmarks \
  --suite PB1 \
  --stage confirmation \
  --device cpu \
  --workers 8 \
  --require-protocol-freeze
```

confirmation 阶段：

- 禁止自动重新选择 basis；
- 禁止修改 gate；
- official test 只访问一次；
- 任何失败都保留。

---

## 19.14 如何复用旧的 AR/KAN/Stage1 代码

旧代码不再承担主模型，但仍有三个用途。

### 用途 A：rank-1 基线

原：

\[
q_j(\tau)f_j(x)
\]

可以直接作为 `rank1_ar_raphu` 基线。

### 用途 B：GADKN/KAN 系列对照的组件

旧 `UnivariateKANResponse` 可以复用为 KAN 响应基线，但：

- 不复用旧的 support pruning 结论；
- 不把非凸训练失败算作 spectral 方法失败；
- 必须多初始化并报告方差。

### 用途 C：CUDA 对照

旧 Stage1/V20 的 sequence-first CUDA 路线可以作为：

- rank-1 计算速度参考；
- spectral CUDA 未来实现的单模态极限测试。

### 不再复用的部分

- 旧硬门控；
- 单 Gamma 表示能力假设；
- shared homotopy support 路径；
- 直接用 raw parameter norm 做 group prox；
- universal rank-2 假设。

这些可以留在 `legacy/` 或原路径，不能进入 PB1 主估计器。

---

## 19.15 旧结果与新结果的关系

新实验必须引用旧证据，但不得覆盖旧结果。

| 旧结果 | 在新方案中的地位 |
|---|---|
| v0.3 E0/E1 | operator identity 与初始表示诊断 |
| v0.3.1 | projection repair 历史证据 |
| v0.3.2 | mother/full surface capacity |
| v0.3.3 | resolution role 与 full structural surface |
| v0.3.4 | adaptive rank profile 与 bootstrap |
| PB1 | 公开物理系统外部验证 |
| PB2 | 闭环多变量验证 |
| PB3 | 真实软测量验证 |
| CZ | 目标工艺验证 |

报告中应明确：

\[
\boxed{
\text{v0.3.4 解决“能否表示和压缩”；}
\quad
\text{PB1 解决“能否迁移到权威物理基准”。}
}
\]

---

## 19.16 结果目录与不可变性

新结果目录：

```text
results/public_benchmarks/
├── pb0/
├── pb1/
│   ├── pwh/
│   ├── whpn/
│   ├── cascaded_tanks/
│   ├── silverbox/
│   ├── runtime/
│   ├── tables/
│   └── figures/
├── pb2/
└── pb3/
```

每个 job 使用：

```text
dataset/model/track/horizon/seed/
```

并保存：

```text
config_resolved.yaml
data_lineage.json
split_manifest.csv
fit.npz
predictions.parquet
metrics.json
solver_diagnostics.json
rank_profile.csv
runtime.json
decision.json
```

已存在目录默认拒绝覆盖。只有显式：

```text
--force-development
```

可覆盖 development；confirmation 永不覆盖，只能生成新的 run ID。

---

## 19.17 兼容性与回归测试

新增 PB1 代码后，以下旧测试必须持续通过：

```text
test_amplitude_domain
test_spectral_design
test_spectral_solver
test_weighted_projection
test_operator_closure
test_rank_profile
test_rank_bootstrap
test_resolution_roles
test_v033_reinterpretation
```

还需增加固定 snapshot 回归：

```text
test_legacy_v034_regression.py
```

它读取一个小型冻结输入，检查：

- coefficient hash 或容差；
- prediction；
- KKT；
- normalized spectrum；
- rank profile。

允许数值容差，不要求不同 BLAS 下 bitwise 相同。

---

## 19.18 当前代码到 PB1 的最小改动集

为了避免大规模重写，PB1 第一包只允许以下新增/修改。

### 必须新增

1. `datasets/base.py`
2. 四个 PB1 loader
3. split/window/scaler 工具
4. public benchmark config schema
5. public runner
6. persistence、AR、ARX、NARX 基线
7. public result schema
8. loader/split/no-leakage tests
9. 打包器

### 允许小改

1. `spectral/design.py`：接受标准窗口数组；
2. `spectral/crossfit.py`：sequence-aware；
3. `spectral/rank_bootstrap.py`：record-aware block；
4. `spectral/contracts.py`：PB1 状态；
5. `spectral/metrics.py`：skill/OOD/record metrics。

### PB1 禁止改

1. 核心正规方程语义；
2. Gram 白化定义；
3. rank tail 定义；
4. v0.3.4 既有配置；
5. 旧结果；
6. CUDA 求解器重写；
7. E2B/E3 与 PB1 混在同一 runner 一次完成。

---

## 19.19 资源与并行策略

PB1 的主要开销来自：

- 多数据集；
- 多 horizon/history；
- basis/penalty 网格；
- 多模型；
- bootstrap。

但单个 SISO spectral fit 仍以 CPU FP64 线性代数为主。最合理的并行方式是：

\[
\boxed{
\text{数据集}\times\text{模型}\times\text{horizon}\times\text{seed}
}
\]

做进程级并行，而不是立即开发 GPU kernel。

推荐：

```text
smoke: 1–2 workers
development: 8 CPU workers
confirmation: 8–16 CPU workers
bootstrap: 独立批次并行
```

当 PB2 的多变量 full-kernel 使单次矩阵/PCG 成为热点后，再正式进入 spectral CUDA。

若 PB1 全网格在本机需要数十小时以上，可以用服务器做 confirmation 批量并行；此时服务器收益来自并发实验，而不是单次小模型的 GPU 加速。

---

## 19.20 PB1 完成后的代码门

PB1 只有在以下条件同时满足时才允许进入 PB2：

```text
PB0_SOURCE_AUDIT_PASS
ALL_LEGACY_REGRESSION_TESTS_PASS
ALL_PB1_LOADERS_PASS
ALL_PB1_SMOKE_PASS
PB1_PROTOCOL_FROZEN
OFFICIAL_TEST_ACCESSED_ONCE
PB1_RESULT_PACKAGE_VALID
```

科学 gate：

1. 至少一个数据集显示 full kernel 相对 rank-1 的明确收益；
2. adaptive rank 在主预算下接近 full model；
3. process-noise 数据没有被错误清洗；
4. OOD/overflow/extrapolation 单独报告；
5. 负结果被保留；
6. fixed rank-2 未被重新设为普适结论。

若代码门通过而科学 gate 未通过，仍需打包结果，但下一阶段为：

```text
STOP_AND_REVISE_MODEL
```

而不是自动进入 PB2。

---

## 19.21 最终打包规范

用户返还给分析方的 PB1 包必须包含源码、配置、数据 lineage、结果和 hash。

推荐打包脚本：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
OUT="$ROOT/return/OPS_UOI_PUBLIC_BENCHMARK_PB1_RESULTS_bundle"
ZIP="$ROOT/return/OPS_UOI_PUBLIC_BENCHMARK_PB1_RESULTS_bundle.zip"

rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT"

cp -a README.md CHANGELOG.md "$OUT/" 2>/dev/null || true
cp -a src tools tests configs "$OUT/"
cp -a data_manifests split_manifests "$OUT/" 2>/dev/null || true
cp -a results/public_benchmarks/pb0 "$OUT/results_pb0"
cp -a results/public_benchmarks/pb1 "$OUT/results_pb1"
cp -a environment "$OUT/" 2>/dev/null || true

find "$OUT" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$OUT" -type f \( -name '*.pyc' -o -name '*.tmp' \) -delete

python tools/build_manifest.py \
  --root "$OUT" \
  --output "$OUT/manifest.json"

(
  cd "$OUT"
  find . -type f ! -name 'SHA256SUMS.txt' -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS.txt
)

python - <<'PY'
from pathlib import Path
import json
root = Path("return/OPS_UOI_PUBLIC_BENCHMARK_PB1_RESULTS_bundle")
required = [
    root / "src",
    root / "tools",
    root / "tests",
    root / "configs",
    root / "results_pb0",
    root / "results_pb1",
    root / "manifest.json",
    root / "SHA256SUMS.txt",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"Missing required package entries: {missing}")
json.loads((root / "manifest.json").read_text(encoding="utf-8"))
print("PACKAGE_STRUCTURE_OK")
PY

(
  cd "$ROOT/return"
  zip -qr "$(basename "$ZIP")" "$(basename "$OUT")"
  sha256sum "$(basename "$ZIP")" > "$(basename "$ZIP").sha256"
  unzip -t "$(basename "$ZIP")"
)

echo "FINAL_PACKAGE=$ZIP"
echo "FINAL_SHA256=$ZIP.sha256"
```

压缩包内部不得包含：

- 原始受许可证限制、不能再分发的数据；
- API key；
- 用户私有路径；
- 旧结果的可写副本；
- 超大缓存；
- 未声明的临时模型。

原始公开数据若许可允许再分发，也优先只返还：

```text
download URL + DOI + raw hash + loader
```

避免包体积失控。

---

## 19.22 从今天开始的实际任务清单

按照依赖关系，下一步代码工作应严格是：

### 第 1 步

冻结完整 v0.3.4 工程树，运行旧测试。

### 第 2 步

实现 `DynamicDataset`、sequence-aware window 和 split integrity。

### 第 3 步

实现 PB1 四个 loader 和 PB0 audit。

### 第 4 步

让 PWH 在现有 full spectral solver 上完成一个 smoke：

```text
X-only, h=1, one history, full kernel, CPU FP64
```

### 第 5 步

接入 AR/X/XAR、rank-1、fixed rank-2 和 adaptive rank。

### 第 6 步

接入 WHPN、Tanks、Silverbox。

### 第 7 步

跑 PB1 development，冻结协议。

### 第 8 步

跑 confirmation 和打包。

只有这八步完成后，才启动 TEP/IB 的 E2B/E3 级多变量闭环工作。



# 20. 最终结论

在复用 v0.3.4 现有 spectral 主干的前提下，接下来的正确顺序是：

\[
\boxed{
\text{PB0 数据冻结}
\rightarrow
\text{PB1 公开物理动态基准}
\rightarrow
\text{PB2 TEP/IB 闭环基准}
}
\]

\[
\boxed{
\rightarrow
\text{PB3 真实公开软测量}
\rightarrow
\text{PB4 理论专项确认}
\rightarrow
\text{CZ 多棒确认}.
}
\]

这条路线能够把论文的证据从“我们自己设计的合成例子”扩展为：

\[
\boxed{
\text{公开系统辨识 benchmark}
+
\text{公开闭环工业 benchmark}
+
\text{公开真实软测量}
+
\text{CZ 目标工艺}.
}
\]

它仍然不会用实验替代定理证明，但会显著增强理论的可信度、方法的一般性和 Automatica 投稿时的防守能力。
