# PRISM v2 Numerical Freeze
## 全部数值门槛、选择语义与停止条件

> **协议 ID**：`PRISM_V2_MODULAR_ASSEMBLY_NUMERICAL_FREEZE_V1`  
> **冻结日期**：2026-08-04  
> **状态**：`FROZEN_BEFORE_IMPLEMENTATION_AND_V2_DEVELOPMENT_ACCESS`  
> **机器可读唯一真值源**：`PRISM_V2_ASSEMBLY_CONFIG_FROZEN.json`

本文把 PRISM v2 实现中此前仍可能被 Codex 自行补齐的数值语义全部冻结。理论文档继续描述一般对象；本文件和 JSON 决定本轮五数据集、七任务 CPU 实验的具体数值实现。若正文与 JSON 冲突，以 JSON 为准，并必须在实现审计中报错，禁止静默选取其中之一。

# 1. 统一选择规则

每个候选使用 4 个 inner folds；至少 3 折有限才可比较。折损失为 MSE：

\[
\bar L=\frac1n\sum_{k=1}^nL_k,
\qquad
\operatorname{SE}(L)=\frac{s(L_1,\ldots,L_n)}{\sqrt n},
\]

其中样本标准差使用 `ddof=1`。one-SE 可接受集合满足

\[
\bar L_a\le \bar L_{\min}+\operatorname{SE}(L_{\min}).
\]

浮点比较使用相对容差 `1e-12`、绝对容差 `1e-15`。中性元若位于 one-SE 集合，按偏序优先中性元。

非中性模块还必须同时满足：

1. 相对中性元平均 validation MSE 改善至少 `1%`；
2. 至少 `3/4` 折改善方向为正；
3. 数值证书通过；
4. 对物理结构作声明时，激活稳定率至少 `0.75`。

相对改善的分母使用

\[
\max\{L_0,10^{-12}\max(\operatorname{Var}_{train}(z),1)\},
\]

避免近零目标下比例爆炸。

# 2. 行数与样本规则

| 用途 | 上限 |
|---|---:|
| 单通道 K 拟合 | 100,000 |
| K/C 联合物理拟合 | 250,000 |
| W 拟合 | 250,000 |
| A 拟合 | 250,000 |
| J 联合预测拟合 | 250,000 |
| 每折 validation 选择 | 50,000 |
| QR/SVD 基审计 | 250,000 |
| test/OOD 最终预测 | 全部 immutable rows |

任何拟合候选至少需要 `200` 行，且至少满足每个自由参数 `20` 行。截断仅作用于拟合，不作用于最终指标。子样本严格按 `base_origin_id` 的 SHA256 升序选取。

# 3. E/K 数值门槛

## 3.1 时间 profile

- FAST：\(\Delta\) 候选 `[1,2,4,8]`；
- MEDIUM：`[2,4,8,16]`；
- SLOW：`[4,8,16,32,64]`；
- 正提前量历史覆盖：`[2h,4h,8h]`；
- \(h=0\)：历史覆盖 `[4Δ,16Δ,64Δ]`；
- 每通道最多保留 2 个 profile；
- 跨折历史尺度在至少 3/4 折内不得相差超过 2 倍。

## 3.2 基函数和 rank

- `m_tau ∈ {4,8,12}`；
- 连续幅值 `m_x ∈ {4,6,8}`；
- rank 候选 `{0,1,2,3,FULL}`；
- 二值变量（唯一值数 ≤2）使用中心化指示基；
- 3–8 个唯一值使用中心化 drop-one one-hot；
- 唯一值数 ≥9 才使用连续样条；
- 标准化方差下限 `1e-12`；
- 标准化 knot 去重距离 `1e-6`。

进入 FULL 的条件是 rank 1–3 均不在 one-SE 可接受集合且 FULL 数值有效。

## 3.3 K 证书

- relative KKT hard fail：`1e-8`；
- HS 相对重构误差：≤`0.05`；
- 保留 rank 的最小相对奇异值：≥`0.05`；
- 下一奇异值/末个保留奇异值：≤`0.80`；
- rank 跨折差异：最多 ±1；
- active channel 跨折激活率：≥`0.75`；
- active channel 集平均两两 Jaccard：≥`0.60`；
- active channel 总数硬上限：32，超出时按 train-only 平均改善排序，名称作最终确定性 tie-break。

# 4. C 联合基与交互

## 4.1 Joint basis

- 每通道 pivoted-QR 相对对角阈值：`1e-8`；
- 每通道最多 12 列；
- 全局最多 384 列；
- 超限时用 train-only pivoted QR 压缩，每个 active channel 至少保留 1 列；
- ridge alpha 使用冻结的 12 点网格；
- joint basis 替换 compressed 必须通过 one-SE、1% 和 3/4 正向折。

## 4.2 Sparse pairwise ANOVA

- strong heredity：两通道都必须 active；
- train-only 残差条件相关绝对值至少 `0.05`；
- 候选 pair pool 最多 12；
- 最终最多 3 对；
- 每轴 4 个样条 knots，原始 tensor 最多 16 列；
- 对截距和两个单变量主效应空间做 ANOVA 残差化；
- 每新增一对都必须单独满足 one-SE、1% 和 3/4 正向折；
- 最少 1000 拟合行，并满足每原始 pair 列至少 20 行。

# 5. W 模块

本轮 W 只读取 C 后的单个标量物理预测；多变量 W 延后。

- latent 先按 inner-train 标准化；
- knot 个数 `{4,6,8}`；
- knots 为 \(j/(k+1)\) 分位数；
- 标准化 knot 去重距离 `1e-6`；
- 至少 4 个不同 knots；
- 三次样条；
- natural cubic 边界线性延拓；
- I-spline 系数在冻结方向下非负；
- smoothness penalty `{0,1e-4,1e-3,1e-2,1e-1,1}`；
- effective degrees of freedom ≤12。

单调候选方向由 inner-train 的 Spearman 相关决定：绝对值至少 `0.05` 且同号折比例至少 `0.75`。否则单调候选 `NOT_APPLICABLE`，validation 不得反转方向。

W 的非线性基对 `[1,q]` 残差化，证书阈值为 `1e-8`。W 激活还要求 identity 不在 one-SE 集、改善 ≥1%、至少 3/4 折正向。

支持域：

- hard support：train min/max；
- tail support：train `[0.005,0.995]` 分位；
- hard exceedance >5% 或 tail exceedance >20% 时，W 仍可预测，但不得作稳定结构声明，必须标记 `SUPPORT_LIMITED`。

# 6. A 和 J

A/J 的 ridge alpha 均使用：

```text
1e-8,
2.848035868435799e-7,
8.111308307896872e-6,
2.310129700083158e-4,
6.579332246575682e-3,
1.8738174228603832e-1,
5.336699231206313,
151.99110829529332,
4328.7612810830615,
123284.67394420659,
3511191.7342151273,
1e8
```

A 使用 4 折 rolling OOF，至少 3 折可用，最大 64 个残差 lag。成熟特征实际观测比例低于 `0.80` 时可保留预测，但不能作结构声明。

正交 residual AR 对冻结 K/C/W 空间做 QR/SVD 投影，正交证书阈值 `1e-8`，投影 rank 相对下限 `1e-10`。

J 的 K/state penalty ratio 为 `{0.25,1,4}`，必须包含 exact K zero、exact state zero 和 exact both zero；J 相对最佳嵌套中性候选至少改善 1% 才激活。

# 7. 通用数值求解

- FP64；
- solver 顺序：Cholesky → pivoted QR → SVD rescue；
- relative KKT warning `1e-10`，hard fail `1e-8`；
- condition warning `1e12`，hard fail `1e14`；
- relative Gram eigenvalue floor `1e-10`；
- SVD `rcond=1e-12`；
- pivoted QR relative rank tolerance `1e-10`；
- Cholesky relative jitter grid `{0,1e-12,1e-10,1e-8}`；
- 标准化系数 L2 范数硬上限 `1e6`；
- ALS：3 个固定初始化、最多 100 次、相对目标容差 `1e-8`、连续 5 次上升即失败、目标爆炸到初值 `1e6` 倍即失败；
- 低秩候选必须 fixed-support refit。

# 8. 配对统计

- paired block bootstrap：500 次；
- seed：20260804；
- 95% CI：`[0.025,0.975]`；
- block：`h+W`、`2(h+W)`、`ceil(L_core/4)`；
- grouped 数据先完整实体有放回，再实体内 block；
- tie-safe two-sided p；
- Holm alpha：0.05；
- family：target head × split × information set × block length。

一个对比称为 `STATISTICALLY_SUPPORTED` 必须：

1. one-SE 与实用门通过；
2. positive probability ≥0.95；
3. 三个 block 中至少 2 个 Holm reject；
4. reject 中必须包含最长 block。

# 9. OOD 与停止门

`OOD_NOT_MATERIALLY_WORSE` 同时要求：

- 点估计相对 MSE 恶化不超过 10%；
- 最长 block 不得出现 Holm 显著恶化。

停止扩展的数值条件：

- joint basis 在 Level C 支持视图数为 0；
- W 在 Level C 激活视图数为 0；
- orthogonal A 支持视图数为 0；
- 超过 50% 最终装配卡的稳定率低于 0.50；
- 全部候选中数值硬失败率 >20%；
- selected/final 候选数值硬失败率 >5%；
- 至少 2 个 OOD 视图出现 >10% 且最长 block Holm 显著恶化，并且没有任何 OOD 视图改善。

结构成功至少需要 2 个受支持视图，且来自至少 2 个数据集。W 的模块性成功最低要求是至少 1 个 Level C 视图被支持，同时其他视图允许自动选择 identity。

# 10. 参数量

必须同时报告：

- stored parameters；
- active parameters；
- effective degrees of freedom/rank；
- deployment parameters。

绝对值 ≤`1e-12` 的标准化系数视为非 active；数值 rank 相对阈值 `1e-10`。knots、缩放统计、低秩因子、isotonic/I-spline 断点、投影矩阵和状态系数全部计入，禁止只统计最终 readout。

# 11. 不再留给实现自行决定的项目

本冻结已经确定：fold 数、one-SE 公式、实用改善、稳定率、行数上限、时间网格、K 基宽、rank、正则、channel gate、joint-basis 维数、pair 筛选、W knots/方向/支持域、A/J alpha、正交容差、求解器、bootstrap、Holm、OOD 恶化和停止条件。

机器配置中的：

```json
"unresolved_numeric_semantics": []
```

必须保持为空；实现遇到缺失数值只能停止并报告，禁止补默认值。
