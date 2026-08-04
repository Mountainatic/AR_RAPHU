# PRISM v2 Modular CPU Benchmark Protocol
## 既有五数据集、七任务、冻结划分下的模块装配验证方案

> **协议版本**：v2.0-numerical-freeze-20260804  
> **理论依据**：`PRISM_Theory_v2_0_Modular_Assembly_Theory_Only.md`  
> **数据继承**：PRISM Industrial Benchmark V1 C0/C1 immutable data  
> **运行定位**：CPU-only，完整 GPU benchmark 暂缓  
> **数值唯一真值源**：`PRISM_V2_ASSEMBLY_CONFIG_FROZEN.json`  
> **数值说明**：`PRISM_V2_NUMERICAL_FREEZE.md`  
> **数值唯一真值源**：`PRISM_V2_ASSEMBLY_CONFIG_FROZEN.json`  
> **数值说明**：`PRISM_V2_NUMERICAL_FREEZE.md`  
> **实验性质**：结构升级后的探索性主头复测 + 未访问注册视图的内部确认  
> **禁止事项**：不得重分 train/validation/test/OOD，不得重算目标、sample ID、purge、scaler 或 proxy policy，不得根据 test/OOD 调整模块格。

---

# 0. 本轮要回答的问题

PRISM v2 不以“某个固定结构是否统一获胜”为唯一问题，而检验下列五个命题：

1. **模块化必要性**：同一装配是否不可能同时适合全部过程；
2. **过程变量双面孔**：没有过程变量时，状态-only 面孔能否正常退化；有过程变量时，输入-only 和 Physics-First 能否提供额外信息；
3. **联合物理基价值**：保留完整通道基并联合固定支持重拟合，是否优于“每通道先压缩为单个标量再融合”；
4. **Wiener 模块价值**：identity 与非线性读出能否由统一开发规则自动选择，而不是按数据集手工添加；
5. **解释权分隔**：K/C/W 冻结后，成熟残差 A 是否能在不重叠解释物理空间的条件下提高动态预测。

本轮不以证明 PRISM 为跨任务 SOTA 为门槛。正式结果可以是：

- 不同目标头选择不同装配；
- 多数任务关闭 W；
- 某些任务关闭 K，退化为状态-only；
- 某些任务关闭 A，退化为 input-only；
- 联合预测强于解释路线，但解释路线结构更稳定；
- v2 无法稳定改善，模型升级被证伪。

---

# 1. 数据与划分完全继承

## 1.1 数据冻结原则

本轮直接读取原 C1 immutable package。以下对象不得重建：

- raw source hash；
- run/profile/month split；
- target heads；
- half-open target windows；
- physical-time-to-step conversion；
- availability delay；
- purge masks；
- base origin ID 和 view sample ID；
- train-only scaler metadata；
- proxy-excluded/full-sensor views。

所有新预测必须与原 sample ID 顺序逐行一致。

## 1.2 原始 split registry 哈希

| 数据集 | 冻结 split registry SHA256 |
|---|---|
| TEP | `298f075084adb9af507721dfdcec66c53d2a0365751f0ad662bcb86d535e46c2` |
| Debutanizer | `81187ee4b431aa55d07ae265986a663bb81355516337c893c171bc0fcb2b1e7c` |
| SRU | `84cb3c0596c426483216af6587c51dc01351b89d7878758cc70c96d9aaaca09d` |
| PMSM | `a571e744884e58817ade9512f97cf3e89e9e18b493d9d9a49182aac21a1a9842` |
| MetroPT | `145a264f35f58c3839812b02b50d9097053a035bcfbb73d8f9c62454216347c9` |

## 1.3 精确划分语义

### TEP

- Train：Training 文件中 fault 0–15，run 1–400；
- Validation：Training 文件中 fault 0–15，run 401–500；
- Main Test：Testing 文件中 fault 0–15，全部 run；
- Unseen-disturbance OOD：Testing 文件中 fault 16–20，全部 run；
- fault 16–20 不参与任何开发、profile、module 或 threshold 选择；
- 完整 `(partition, fault, run)` 为实体，不跨 run 构造窗口。

### Debutanizer

- Train：原始索引 `[0,1436)`；
- Validation：`[1436,1915)`；
- Test：`[1915,2394)`；
- 四折 expanding inner validation；
- 主视图保留发布文件已平移 8 步的目标；
- delay-10-steps 只改变目标历史/残差可用性，不改变预测目标。

### SRU

- Train：`[0,6048)`；
- Validation：`[6048,8064)`；
- Test：`[8064,10081)`；
- H2S 与 SO2 共用相同时间划分；
- 四折 expanding inner validation。

### PMSM

继续使用 `PRISM_PMSM_SPLIT_V1` 精确 profile 清单：

- Train profile：`[7,19,67,5,43,71,52,46,11,57,59,55,61,50,31,41,14,69,42,72,29,32,81,17,63,10,15,16,27,44,66,62,56,74,58,12,65,70,13,18,79,6]`；
- Validation profile：`[23,78,49,47,64,3,26,45,24,9,76,80,30,48,53]`；
- Test profile：`[51,75,60,54,8,21,2,73,68,36,20,4]`；
- 完整 profile 为实体；
- 不按模型表现调换 profile。

### MetroPT

- Train：2020-02、2020-03、2020-04；
- Validation：2020-05；
- Test：2020-06、2020-07、2020-08；
- OOD fault windows：
  - 2020-04-18 00:00 至 23:59；
  - 2020-05-29 23:30 至 2020-05-30 06:00；
  - 2020-06-05 10:00 至 2020-06-07 14:30；
  - 2020-07-15 14:30 至 19:00；
- fault windows 单列 OOD，不与正常 test pooled。

---

# 2. 任务、目标头与确认层次

## 2.1 七个主任务

| Task | Target | Cadence | 主提前量 | 目标窗口 | 注册提前量扫描 |
|---|---|---:|---:|---:|---|
| TEP_G12 | XMEAS(40) | 180 s | 12 min | 6 min | 3 / 12 / 36 min |
| DEB_C4 | y | 360 s | 30 min | 6 min | 0 / 30 / 60 min |
| SRU_H2S | y1 | 60 s | 5 min | 1 min | 0 / 5 / 30 min |
| SRU_SO2 | y2 | 60 s | 5 min | 1 min | 0 / 5 / 30 min |
| PMSM_PM5 | pm | 0.5 s | 5 min | 30 s | 30 s / 5 min / 20 min |
| METRO_P60 | Reservoirs | 10 s | 60 s | 10 s | 10 s / 60 s / 5 min |
| METRO_OIL20 | Oil temperature | 10 s | 20 min | 2 min | 5 / 20 / 60 min |

## 2.2 三层评价语义

### Level A：开发选择

只使用 train 和 validation，包括 inner folds。所有模块、knots、rank、interaction、W、A 和装配卡在这里冻结。

### Level B：主头复测

对七个主头原 test/OOD 运行 v2。由于 v2 架构受到 C6 V2 主头结果启发，该层正式标记：

```text
POST_HOC_EXPLORATORY_PRIMARY_HEADS
```

可与 C6 V2 基线作同 sample ID 比较，但不能写成从未看过测试结果的确认性证据。

### Level C：未访问注册视图内部确认

在访问以下 test/OOD 预测前冻结本协议和代码 hash：

1. 每个任务的两个非主提前量；
2. TEP analyzer maturity 5-step 视图；
3. Debutanizer delay-10-step 视图；
4. PMSM full-sensor-secondary 视图；
5. Metro pressure full-sensor-secondary 视图。

若这些视图此前没有生成最终 test/OOD 预测，则标记：

```text
PROSPECTIVE_INTERNAL_CONFIRMATION
```

它们不是新数据集，但比重复主头更强，因为结构冻结发生在预测访问之前。

---

# 3. 两个 leaderboard 继续分离

## 3.1 Input-only

\[
\widehat z=f(U^-,X^-),
\qquad Y^-\text{ forbidden}.
\]

包含：

- C6 V2 冻结基线；
- v1 PRISM Channel-Specific；
- v2 K joint-basis；
- v2 K + interaction；
- v2 K + W。

## 3.2 Dynamic

\[
\widehat z=f(U^-,X^-,Y^-_{\mathrm{available}}).
\]

包含：

- C6 V2 冻结动态基线；
- state-only；
- v1 Physics-First；
- v1 K-Joint AR；
- v2 K/C + A；
- v2 K/C/W + A；
- v2 predictive joint。

Input-only 与 Dynamic 不计算混合平均排名。

---

# 4. 冻结基线

## 4.1 直接复用 C6 V2 主头预测

主头 test/OOD 的以下预测文件不重跑，直接按 SHA256 复用：

### Input-only

- Persistence；
- Mean；
- Ridge；
- PLS；
- DPLS；
- RBF-SVR；
- XGBoost；
- Parallel Hammerstein；
- Hammerstein-Wiener；
- PRISM v1 Channel-Specific。

### Dynamic

- Persistence；
- Mean；
- Local Linear Trend；
- AR；
- ARX；
- Linear NARX；
- N4SID（仅开发成功目标头）；
- PRISM v1 Physics-First；
- PRISM v1 K-Joint AR。

这样避免因为重新运行软件版本或浮点路径改变基线。

## 4.2 非主头基线

对 Level C 新视图，只运行最小完整强基线：

### Input-only

- Persistence；
- DPLS；
- XGBoost；
- Parallel Hammerstein；
- Hammerstein-Wiener；
- PRISM v1 Channel-Specific。

### Dynamic

- Persistence；
- AR；
- ARX；
- Linear NARX；
- PRISM v1 Physics-First；
- PRISM v1 K-Joint AR。

N4SID 继续按原开发失败保留策略，不为补齐表格而伪造结果。

---

# 5. v2 模块候选

## 5.1 E：尺度编码

候选：

1. `SINGLE_SCALE`；
2. `FIXED_MULTIRESOLUTION`；
3. `CHANNEL_SPECIFIC`。

时间 profile 只在 train/inner folds 选择，候选网格继承 v1 C4 的 FAST/MEDIUM/SLOW 分类和历史覆盖规则。

## 5.2 K：单通道响应

嵌套阶梯：

1. `EXACT_ZERO`；
2. `LINEAR_DISTRIBUTED_LAG`；
3. `RANK_1_URYSOHN`；
4. `RANK_2_URYSOHN`；
5. `RANK_3_URYSOHN`；
6. `FULL_FINITE_URYSOHN`。

首轮不新增 rank 4 以上；只有前三阶都不能覆盖且数值条件通过时进入 full。

## 5.3 C：通道融合

候选：

1. `ADDITIVE_COMPRESSED`：复现 v1，每通道先压成一个输出；
2. `ADDITIVE_JOINT_BASIS`：保留所有 active 通道冻结基，联合固定支持重拟合；
3. `SPARSE_PAIRWISE_ANOVA`：在 joint basis 上最多增加 3 个残差化通道对。

通道对候选只由 train-only 条件相关排序产生；不得看 validation/test 决定候选名单。若 active 通道少于 2，第三项自动 `NOT_APPLICABLE`。

## 5.4 W：可拆 Wiener 读出

候选：

1. `IDENTITY`；
2. `MONOTONE_I_SPLINE_4`；
3. `MONOTONE_I_SPLINE_6`；
4. `NATURAL_CUBIC_SPLINE_4`；
5. `NATURAL_CUBIC_SPLINE_6`；
6. `NATURAL_CUBIC_SPLINE_8`。

规则：

- knots 来自当前 inner-train 潜变量分位数；
- 重复 knots 删除；
- 非线性基先对 `[1,q]` 或 `[1,Q]` QR 残差化；
- monotone 方向仅可由 train 中的整体方向冻结；
- validation 不能反转单调方向；
- 外推默认边界线性延拓并输出 support flag；
- `IDENTITY` 是正式候选。

## 5.5 A：状态与残差

### 状态-only

1. `AR_LINEAR`；
2. `NAR_TARGET_QUADRATIC`；
3. `EXACT_ZERO` 仅用于统一接口。

### Physics-First

1. `EXACT_ZERO`；
2. `MATURE_RESIDUAL_AR`；
3. `ORTHOGONAL_MATURE_RESIDUAL_AR`。

第三项先把成熟残差历史对冻结 K/C/W 物理空间残差化。

## 5.6 J：联合预测

候选：

1. `JOINT_K_STATE_LINEAR`；
2. `JOINT_KW_STATE_LINEAR`；
3. exact K zero；
4. exact state zero；
5. exact both zero。

J 只进入 Dynamic 预测榜，不作为物理认证结果。

## 5.7 暂不激活

- regime mixture；
- online adaptation；
- 深度 W；
- 高阶全通道交互；
- GPU 神经网络；
- 根据数据集名称直接指定装配。

---

# 6. 分阶段搜索，避免组合爆炸

## Stage V0：继承审计

校验：

- C1 package hash；
- split registry hash；
- task registry；
- sample count；
- sample ID；
- target/scaler/purge；
- C6 V2 基线 prediction hash。

任一不匹配立即停止。

## Stage V1：状态-only 面孔

每个目标头独立选择 AR 或 target-only quadratic NAR。该阶段不读取 U/X，用于建立没有过程变量时的合法 PRISM 面孔。

## Stage V2：单通道 E/K 审计

复用 v1 profile 网格，但重新在 v2 代码路径中：

1. exact-zero；
2. linear；
3. rank ladder；
4. fixed-support refit；
5. 数值证书；
6. active channel gate。

不访问 test/OOD。

## Stage V3：C 联合物理基

在 active 通道与 profile 冻结后比较：

- compressed additive；
- joint-basis additive；
- sparse pairwise ANOVA。

这一步直接检验 v2 的主要结构修订。

## Stage V4：W 模块

冻结 E/K/C，生成 rolling OOF 物理潜变量和预测。比较 identity 与注册 spline。W 只拟合 OOF 剩余曲率，不回调 K/C。

## Stage V5：A 模块

冻结 E/K/C/W，生成成熟 OOF 残差，比较 exact-zero、普通成熟 residual AR 和正交成熟 residual AR。

## Stage V6：装配选择

解释路线的最终候选只保留：

1. `A_ONLY`；
2. `K_COMPRESSED`；
3. `K_JOINT_BASIS`；
4. `K_JOINT_BASIS_W`；
5. `K_JOINT_BASIS_A`；
6. `K_JOINT_BASIS_W_A`；
7. `K_PAIRWISE_W_A`（仅 V3 激活 pairwise 时）。

按 one-SE + 模块偏序选择装配卡。

## Stage V7：联合预测路线

在完全相同开发样本、K profile 和 feature family 上拟合 J，作为工程预测上限。

## Stage V8：冻结与评估

生成 `V2_FINAL_FREEZE_MANIFEST.json` 后：

1. 先运行 Level C 未访问注册视图；
2. 再运行 Level B 主头复测；
3. OOD 不做适配；
4. 全部逐样本预测落盘。

先运行 Level C 是为了避免主头复测再次影响对内部确认结果的解释。

---

# 7. 数值冻结

本协议的全部实现数值已经冻结在：

- `PRISM_V2_ASSEMBLY_CONFIG_FROZEN.json`：机器可读唯一真值源；
- `PRISM_V2_NUMERICAL_FREEZE.md`：公式、阈值和失败语义说明。

任何实现不得继续使用旧 `PROPOSED` 配置，也不得在代码中增加未登记默认值。启动时必须验证：

```text
protocol_id = PRISM_V2_MODULAR_ASSEMBLY_NUMERICAL_FREEZE_V1
status = FROZEN_BEFORE_IMPLEMENTATION_AND_V2_DEVELOPMENT_ACCESS
unresolved_numeric_semantics = []
```

以下为核心数值摘要；未展开字段以冻结 JSON 为准。

## 7.1 选择与激活

- 4 个 inner folds，至少 3 折有限；
- one-SE：样本标准差 `ddof=1` 除以 \(\sqrt n\)；
- K、C、W、A 和 J 的非中性激活门均为相对 validation MSE 改善 ≥1%；
- 至少 3/4 folds 改善方向为正；
- 物理结构声明稳定率 ≥0.75；低于 0.75 只可标记预测性不稳定结构；低于 0.50 视为严重不稳定。

## 7.2 拟合上限

- 单通道 K：100,000；
- K/C、W、A、J：各 250,000；
- 每折 validation：50,000；
- 最终 test/OOD：全部 immutable rows；
- 每候选至少 200 行，且每自由参数至少 20 行。

## 7.3 K/C/W/A/J

- `m_tau={4,8,12}`，`m_x={4,6,8}`，rank `{0,1,2,3,FULL}`；
- active channel 上限 32；
- joint basis 每通道最多 12 列、全局最多 384 列；
- pair 候选相关阈值 0.05，候选池 12，最终最多 3 对；
- W knots `{4,6,8}`，monotone Spearman 阈值 0.05，同号稳定率 0.75；
- W/A 正交容差 `1e-8`；
- A 最大 64 lags，成熟特征结构声明覆盖率至少 0.80；
- J penalty ratio `{0.25,1,4}`；
- W/C smoothness penalty `{0,1e-4,1e-3,1e-2,1e-1,1}`；
- A/J ridge alpha 使用冻结 12 点网格。

## 7.4 数值证书

- relative KKT warning `1e-10`，hard fail `1e-8`；
- condition warning `1e12`，hard fail `1e14`；
- Gram eigen floor `1e-10`；
- SVD rcond `1e-12`；
- rank 最小相对奇异值 0.05，next/kept ≤0.80；
- HS 相对误差 ≤0.05；
- solver：Cholesky → pivoted QR → SVD rescue；
- ALS：3 初始化、100 次、容差 `1e-8`。

## 7.5 统计与 OOD

- paired block bootstrap 500 次；
- Holm alpha 0.05；
- 统计支持要求 3 个 block 中至少 2 个 Holm reject，且必须包含最长 block；
- positive probability ≥0.95；
- OOD 实质恶化门为相对 MSE 10%，并结合最长 block Holm 显著性；
- hard support exceedance >5% 或 tail support exceedance >20% 时降低结构等级。

完整停止条件、参数量口径和支持域语义见数值冻结文件。

---

# 8. 模型选择与防止按数据集定制

## 8.1 同一候选格

七个任务面对同一 E/K/C/W/A/J 候选规则。允许因通道数不足或连续实体条件不满足而 `NOT_APPLICABLE`，不允许根据任务历史表现删除候选。

## 8.2 one-SE tie break

顺序冻结为：

1. 更少激活模块；
2. exact-zero / identity / additive；
3. 更低 K rank；
4. compressed 优先于 joint basis，仅当二者 one-SE 等价；
5. 无 interaction；
6. monotone W 优先于 unconstrained W；
7. 更少 knots；
8. A exact-zero；
9. 更强正则；
10. 更短历史；
11. 更粗时间分辨率；
12. 预注册候选顺序。

第 4 条意味着 joint basis 必须提供稳定增益才取代 v1 compressed 结构。

## 8.3 装配卡稳定性

报告每个模块在 inner folds 和 outer-development directions 的激活率。物理结构声明要求：

\[
\rho_M\ge0.75.
\]

低于 0.75 仍可用于最终预测，但状态为 `PREDICTIVE_SELECTED_UNSTABLE_STRUCTURE`。

---

# 9. 评估指标

## 9.1 预测指标

- MSE；
- RMSE；
- MAE；
- \(R^2\)；
- NRMSE；
- relative Persistence skill；
- Dynamic 榜 relative AR skill；
- relative best-C6-baseline skill。

## 9.2 结构指标

- active channel 数；
- K rank 分布；
- joint-basis 相对 compressed 改善；
- pairwise interaction 数；
- W 是否 identity；
- W effective degrees of freedom；
- A 是否 exact-zero；
- module activation stability；
- K/W/A orthogonality certificate；
- stored/active/effective/deployment parameter count。

## 9.3 资源指标

- wall-clock fit time；
- prediction time；
- peak RSS；
- CPU thread-hours；
- model artifact bytes；
- per-sample latency。

## 9.4 OOD 指标

- OOD MSE；
- ID-to-OOD relative degradation；
- input support exceedance rate；
- W latent support exceedance rate；
- module-wise OOD failure；
- calibration/interval coverage（若输出）。

---

# 10. 配对统计

## 10.1 对比家族

### 主结构对比

1. v2 joint-basis K vs v1 compressed K；
2. K+W vs K；
3. K+A vs K；
4. K+W+A vs K+A；
5. orthogonal A vs ordinary residual A；
6. Physics-First v2 vs J v2；
7. v2 best assembly vs best frozen C6 baseline。

### 负对照

- exact-zero K；
- identity W；
- exact-zero A；
- shuffled/placebo channel profiles；
- forbidden future target leakage test，应强制失败。

## 10.2 Bootstrap

- paired block bootstrap；
- 500 replicates；
- seed：20260804；
- block lengths：
  - \(h+W\)；
  - \(2(h+W)\)；
  - \(\lceil L_{\mathrm{core}}/4\rceil\)；
- grouped datasets：先完整实体有放回抽样，再实体内 block；
- Holm family：target head × split × information set × block length；
- two-sided tie-safe p-value；
- 同时报告 CI、positive probability、raw p 和 Holm adjusted p。

## 10.3 确认性主比较

Level C 的预先指定主比较只有四类：

1. joint-basis K vs compressed K；
2. W-activated assembly vs identity assembly；
3. A-activated assembly vs A exact-zero；
4. final v2 assembly vs best frozen strong baseline。

其他全部标记 secondary，避免全对全检验稀释主要问题。

---

# 11. 判定门

## Gate G0：继承完整性

所有 C1 hash、sample ID、target、split、purge、scaler 一致。否则 `STOP`。

## Gate G1：代码与测试

至少覆盖：

- 中性元嵌套；
- no-U/X state-only；
- no-Y input-only；
- W 正交；
- A 成熟；
- A 正交；
- entity boundary；
- exact model inventory；
- cached prediction SHA；
- parameter count；
- deterministic rerun。

## Gate G2：模块开发有效性

每个模块必须有：

- neutral candidate；
- finite validation losses；
- numerical certificate；
- no test access；
- stable candidate serialization。

## Gate G3：最终冻结

`V2_FINAL_FREEZE_MANIFEST.json` 必须包括：

- theory hash；
- protocol hash；
- code commit；
- C1 hashes；
- C6 V2 baseline hashes；
- candidate config；
- all development result hashes；
- exact assembly card per head/view；
- `test_accessed=false`。

## Gate G4：Level C 内部确认

未访问注册视图先运行。若 freeze 后发现已有旧 test 预测，则对应视图降级为 exploratory，并在报告披露。

## Gate G5：Level B 主头复测

主头结果只更新 v2 的探索性对比，不覆盖 C6 V2 原结果。

---

# 12. 成功与失败标准

## 12.1 最低工程成功

满足全部：

1. 7 个主头和 Level C 注册视图均生成合法装配卡；
2. 无数据/因果/配对错误；
3. 模块能在不同任务中自动选择 identity/exact-zero/active；
4. 参数量审计正确；
5. 结果可复现。

即使误差没有提高，也说明模块系统实现正确。

## 12.2 结构成功

至少满足：

1. joint-basis K 在多个未访问视图上稳定优于 compressed K；或
2. W 在至少一个未访问视图上通过 1% + one-SE + Holm，而在其他视图自动 identity；或
3. orthogonal A 在不破坏 K/W 解释权的情况下稳定提高动态预测。

## 12.3 预测成功

v2 final assembly 在 Level C 中：

- Input-only 平均 rank 不差于 Parallel Hammerstein/H-W；或
- Dynamic 平均 rank 不差于 ARX/Linear NARX；
- 且至少一个 OOD 方向不比 v1 明显恶化。

这不是理论成立的必要条件，但决定论文能否以性能为主线。

## 12.4 停止条件

满足任一即停止继续扩展：

- joint-basis 在所有 Level C 视图均无稳定改善；
- W 只在主头复测激活，在未访问视图全部 identity；
- A 正交化后增益消失；
- 结构选择跨折极不稳定；
- 新模块只提高 ID、系统性恶化 OOD；
- 数值条件频繁硬失败。

此时应保留 v1 和强基线，而不是继续堆更复杂模块。

---

# 13. 输出目录

```text
PRISM_V2_MODULAR_CPU_RESULTS/
├── FREEZE/
│   ├── V2_FINAL_FREEZE_MANIFEST.json
│   ├── DATA_INHERITANCE_AUDIT.json
│   └── ASSEMBLY_CONFIG.json
├── DEVELOPMENT/
│   ├── STATE_ONLY/
│   ├── CHANNEL_AUDIT/
│   ├── JOINT_BASIS/
│   ├── WIENER/
│   ├── RESIDUAL_STATE/
│   └── JOINT_PREDICTIVE/
├── ASSEMBLY_CARDS/
├── PREDICTIONS/
│   ├── LEVEL_C_CONFIRMATION/
│   └── LEVEL_B_PRIMARY_EXPLORATORY/
├── MODEL_AUDIT/
├── ORTHOGONALITY_CERTIFICATES/
├── PARAMETER_COUNTS/
├── ENTITY_METRICS/
├── BOOTSTRAP/
├── RESOURCE_USAGE/
├── PRISM_V2_FINAL_METRICS.csv
├── PRISM_V2_MODULE_ACTIVATION.csv
├── PRISM_V2_CROSS_TASK_RANKS.csv
├── PRISM_V2_FINAL_REPORT.md
└── PRISM_V2_FINAL_DECISION.json
```

逐样本预测字段：

```text
sample_id,base_origin_id,dataset,task,target_head,split,
information_set,availability_scenario,proxy_policy,model,assembly_id,
y_true,y_pred,parameter_count,dtype,core_history_steps,
support_flag,ood_flag
```

---

# 14. 打包与返还

最终包必须：

1. 清理旧临时输出；
2. 收集代码、配置、测试、装配卡、报告、指标和证书；
3. 不包含原始数据；
4. 预测文件可分卷；
5. 生成 `RELEASE_ASSET_MANIFEST.json`；
6. 每个文件记录 bytes + SHA256；
7. 重组后再次校验；
8. 输出最终 zip/tar 和分卷命令；
9. Release 标记是否 exploratory/confirmation；
10. 保留全部失败与 N/A，不静默删除。

---

# 15. 本轮资源决议

\[
\boxed{\texttt{CPU\_ONLY\_GO}}
\]

理由：

- K/C/W/A 均可采用低维固定基和 FP64 线性/样条求解；
- TEP 可使用 streaming Gram，不必把设计矩阵常驻内存；
- 主头基线直接复用 C6 V2；
- 不需要多卡服务器；
- GPU 只在 v2 结构通过 Level C 后，再用少量本地单卡 sanity baselines。

正式顺序：

\[
\boxed{
\text{先写代码与测试}
\rightarrow
\text{冻结装配格}
\rightarrow
\text{开发选择}
\rightarrow
\text{冻结 manifest}
\rightarrow
\text{未访问视图}
\rightarrow
\text{主头复测}
}
\]
