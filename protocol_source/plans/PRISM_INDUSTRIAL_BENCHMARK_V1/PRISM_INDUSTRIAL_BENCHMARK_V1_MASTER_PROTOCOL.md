# PRISM INDUSTRIAL BENCHMARK V1：总实验协议

版本：V1.0  
状态：`PRE-REGISTERED / NO RESULTS`  
理论依据：`PRISM_Theory_v1_3_Theory_Only.md`  
研究对象：多通道尺度专属 Urysohn、Urysohn-First 成熟残差路线与 K-Joint AR 工程预测路线

---

## 1. 研究目标

本基准只回答以下问题，不根据实验结果修改问题本身：

1. 不同物理变量分别使用专属建模步长、历史范围和时滞基，是否优于所有变量共用一个窗口与步长？
2. Urysohn-First 路线能否在保持输入响应可审计性的同时，达到有竞争力的预测性能？
3. K-Joint AR 路线能否在不要求内部归因唯一的条件下取得更好的工程预测？
4. PRISM 的优势来自结构、时间尺度选择，还是仅来自更大的调参预算与模型容量？
5. 在经典过程工业、真实炼油软测量、电机热过程和真实运营设备上，结论是否一致？
6. 当某一通道在一个时间尺度上无效时，它是否会在另一个物理时间尺度上表现出稳定作用？

本协议不预设 PRISM 一定获胜。所有失败任务、归零通道、未收敛模型和方向性不一致结果均必须保留。

---

## 2. 冻结的“3+2”数据集

### 2.1 三个权威锚点

1. **Tennessee Eastman Process（TEP）**  
   过程控制领域经典高保真模拟基准。主用途是验证标准过程变量、操纵量、分析仪变量和多运行工况下的结构泛化。必须在论文中明确其为模拟过程，而非实厂历史数据。

2. **Debutanizer Column**  
   真实炼油厂脱丁烷塔软测量数据。目标为塔底产品 C4 含量，输入为七个在线过程变量。主用途是经典真实工业软测量与分析仪延迟场景。

3. **Sulfur Recovery Unit（SRU）**  
   真实炼油厂硫回收装置数据。目标为尾气 H2S 与 SO2 浓度。主用途是多输出质量变量、闭环补偿与动态软测量。

### 2.2 两个真实多尺度验证集

4. **PMSM Electric Motor Temperature**  
   永磁同步电机试验台多运行 profile 数据。主目标为永磁体温度 `pm`，用于检验电气量、机械量、冷却量与热状态的跨时间尺度响应。

5. **MetroPT-3**  
   波尔图地铁运营列车空气压缩机数据。主目标为储气罐压力与油温，用于在同一真实设备上同时检验快速气动响应与慢速热响应。

任何实验后新增的数据集只能进入下一版协议，不能加入 V1 主表。

---

## 3. 数据源、许可与版本冻结

每个数据集在任何训练前必须生成：

```text
DATASET_REGISTRY/<dataset>/
├── SOURCE_AND_LICENSE.md
├── RAW_FILE_HASHES.json
├── VARIABLE_DICTIONARY.csv
├── CADENCE_AUDIT.json
├── RUN_BOUNDARIES.csv
├── MISSING_AND_DUPLICATE_AUDIT.json
├── TARGET_AVAILABILITY.json
├── SPLIT_REGISTRY.json
└── FREEZE_DECISION.md
```

硬规则：

- 原始文件 SHA256、下载日期、来源论文或仓库、许可证必须登记；
- 对公开描述存在冲突的采样周期，只相信原始时间戳或官方变量说明；
- 无法确认物理时间尺度时，任务只能按“样本步”表述，不得伪装成分钟或小时；
- 不允许跨断点插值、平滑连接或让历史窗口跨越独立运行边界；
- 许可不允许再分发时，返回包只包含哈希、索引、脚本与结果，不包含原始数据；
- 数据清洗、删除规则和运行边界必须在模型训练前冻结。

---

## 4. 冻结任务矩阵

### 4.1 主论文 primary heads

主表只使用以下预注册任务。所有模型必须在同一任务定义上比较。

| Task ID | 数据集 | 目标 | 主提前量 | 主目标窗口 | 主信息场景 |
|---|---|---|---:|---:|---|
| `TEP_G12` | TEP | 产品流 G 组分，标准实现中对应 `XMEAS(40)` | 12 min | 6 min | 记录时刻可用 + 分析仪成熟敏感性 |
| `DEB_C4` | Debutanizer | 塔底 C4 含量 | 30 min | 1 个经确认采样间隔 | 记录时刻可用 + 60 min 标签延迟敏感性 |
| `SRU_H2S` | SRU | 尾气 H2S 浓度 | 5 min | 1 min 或一个经确认采样间隔 | 记录时刻可用 |
| `SRU_SO2` | SRU | 尾气 SO2 浓度 | 5 min | 1 min 或一个经确认采样间隔 | 记录时刻可用 |
| `PMSM_PM5` | PMSM | 永磁体温度 `pm` | 5 min | 30 s | 当前目标可用 |
| `METRO_P60` | MetroPT-3 | 储气罐压力 `Reservoirs` | 60 s | 10 s | 当前目标可用 |
| `METRO_OIL20` | MetroPT-3 | 油温 `Oil_temperature` | 20 min | 2 min | 当前目标可用 |

若原始数据采样周期不能精确支持某个物理时间，按最近的因果整数样本数实现，并在 `TASK_REALIZATION.json` 中记录实际时间误差。若误差超过 10%，该任务标记 `UNSUPPORTED_BY_CADENCE`，不得私自更换为更有利的 horizon。

### 4.2 多尺度 sweep

多尺度实验用于验证“不同物理量在不同时间尺度上才表现出明显作用”。它不用于在 test set 上重新挑选主任务。

| 数据集/目标 | 预注册提前量集合 |
|---|---|
| TEP 产品 G | 3 min、12 min、36 min |
| Debutanizer C4 | 0、30 min、60 min |
| SRU H2S/SO2 | 0、5 min、30 min |
| PMSM `pm` | 30 s、5 min、20 min |
| MetroPT `Reservoirs` | 10 s、60 s、5 min |
| MetroPT `Oil_temperature` | 5 min、20 min、60 min |

每个提前量拥有独立目标头、独立尺度匹配 AR、独立 residual AR 和独立 K-Joint AR。禁止跨目标头复用一个固定 AR 阶数或固定输出采样步长。

---

## 5. 变量与信息集合同

### 5.1 三类变量

对每个数据集建立：

- `U`：操纵量、设定量、外生量和物理上先行的在线输入；
- `X`：在线过程状态测量，但不是目标本身；
- `Y`：目标及其严格过去历史。

### 5.2 两张正式排行榜

#### 输入型软测量榜

\[
\widehat y_{t+h}=f(U_t^-,X_t^-),
\]

不允许读取历史目标 `Y^-`。该榜比较 PLS、DPLS、SVR、XGBoost、输入型深度网络和 PRISM-Urysohn。

#### 动态预测榜

\[
\widehat y_{t+h}=f(U_t^-,X_t^-,Y_t^-),
\]

允许使用当时已经可用的目标历史。该榜比较 AR/ARX/NARX、动态深度网络、PRISM-Urysohn-First 与 PRISM-K-Joint AR。

两榜不得混排，且必须共享同一 test sample ID。

### 5.3 防止代理变量泄漏

- TEP 预测产品 G 时，主输入集排除全部产品组分 `XMEAS(37:41)`；动态榜只允许使用目标 G 自身的成熟历史，不允许使用其他产品组分通过和约束间接重建目标；
- PMSM 主输入集使用 `ambient, coolant, u_d, u_q, i_d, i_q, motor_speed, torque`，不使用内部定子温度；完整传感器集只作为 secondary sensitivity；
- MetroPT 预测 `Reservoirs` 时，主输入集排除高度同源的 `TP3`；包含 `TP3` 的结果单列为 `FULL_SENSOR_PROXY_ALLOWED`；
- 任何与目标通过代数恒等式、校准公式或同一分析仪直接关联的变量必须登记并从主榜排除。

---

## 6. PRISM 的正式模型组

### 6.1 PRISM-Urysohn

只使用多通道物理历史：

\[
\widehat z_m(t)=\sum_{j\in\mathcal S_m}\mathcal K_{j,m}^{(\Delta_j,T_{j,m},\mathcal B_{j,m})}[u_j](t).
\]

每个通道独立选择：

\[
\Delta_j,\quad T_{j,m},\quad \mathcal B_{j,m},
\]

以及复杂度：

\[
0\rightarrow\text{linear distributed lag}
\rightarrow\text{rank-1}
\rightarrow\text{channel-specific rank-}R
\rightarrow\text{finite Urysohn}.
\]

### 6.2 PRISM-Urysohn-First

路线 I：

\[
K\rightarrow\text{freeze }K
\rightarrow A_m^{\mathrm{res}}(R_{m,t}^{\mathrm{mature},-}).
\]

必须使用 OOF 物理残差；在线可用残差满足：

\[
s+h_m+W_m\le t.
\]

`A_m^{res}=0` 是正式候选。K 与 residual AR 禁止最终联合回调。

### 6.3 PRISM-K-Joint AR

路线 II：

\[
\widehat z_m(t)=K_m^J(U^-,X^-)+A_m^J(Y^-).
\]

物理输入项与状态项在同一目标头内联合优化，以工程预测为目标。其内部 K 不自动获得物理解释资格。

### 6.4 AR-first conditional audit

每个通道 profile 配套同目标、同频带的诊断 AR：

\[
\Pi_{j,r}=(\pi^K_{j,r},\pi^A_{j,r}).
\]

该实验只回答“输入是否提供状态历史之外的增量信息”，不进入正式 PRISM 路线排序。

---

## 7. 通道专属时间尺度与公平对照

### 7.1 通道分类与候选步长

在查看模型性能前，根据变量语义、单位和传感器类型冻结通道类别：

- `FAST`：阀门、开关、电流、电压、转速、快速压力、流量；
- `MEDIUM`：液位、主体温度、压力状态、慢流量；
- `SLOW`：环境温度、冷却温度、油温、组分分析、热状态。

设原始可靠采样周期为 \(\delta\)，候选建模步长：

| 类别 | 候选 \(\Delta_j/\delta\) |
|---|---|
| FAST | 1, 2, 4, 8 |
| MEDIUM | 2, 4, 8, 16 |
| SLOW | 4, 8, 16, 32, 64 |

若某数据本身采样很慢，超过目标头可用分辨率的候选自动删除，但不得加入未注册步长。

### 7.2 历史覆盖

对目标头 \(m\)，每个通道候选历史：

\[
T_{j,m}\in\{2h_m,4h_m,8h_m\},
\]

并受数据运行段长度和预注册最大历史上限约束。`h=0` 的软测量任务使用：

\[
T_{j,m}\in\{4\Delta_j,16\Delta_j,64\Delta_j\}.
\]

### 7.3 三个关键消融

1. **Single-Scale PRISM**：所有通道共享一个 \(\Delta,T,\mathcal B\)；
2. **Fixed Multi-Resolution**：所有通道接受相同多分辨率金字塔，但不做通道独立选择；
3. **Channel-Specific PRISM**：每个通道独立选择步长、历史和时滞基。

只有第 3 项相对前两项有稳定提升，才能把优势归因于“不同物理量使用不同时间尺度”。

### 7.4 尺度匹配 AR

每个物理 profile 必须配套同一目标头的诊断 AR。AR 候选输出步长：

\[
\Delta^A\in\{\max(\delta,W/4),\max(\delta,W/2),\max(\delta,W),2W\},
\]

历史覆盖：

\[
T^A\in\{2h,4h,8h\},
\]

并在训练折内选择。不同 horizon 不得共用一个万能 AR。

---

## 8. 冻结 baseline

### 8.1 CPU：经典且广泛使用

- Mean / Persistence / Seasonal Persistence；
- Ridge；
- PLS；
- Dynamic PLS；
- RBF-SVR；
- XGBoost；
- AR；
- ARX；
- Linear NARX；
- N4SID；
- Parallel Hammerstein；
- Hammerstein-Wiener；
- PRISM-Urysohn；
- PRISM-Urysohn-First；
- PRISM-K-Joint AR。

### 8.2 GPU：公认核心 baseline

- MLP；
- LSTM；
- GRU；
- TCN；
- DLinear；
- NLinear；
- Causal Transformer Encoder；
- PatchTST。

### 8.3 GPU：有潜力的现代 baseline

- iTransformer；
- ModernTCN；
- TimeMixer；
- TimesNet；
- S4D；
- Temporal Autoencoder；
- AKGNN（窗口适配版）；
- T-AKGNN（显式时序扩展版）。

AKGNN 与 T-AKGNN 必须分别标记 `ADAPTED_STATIC_WINDOW` 和 `ADAPTED_TEMPORAL`，不得把适配实现冒充原论文原生时序任务。

---

## 9. 数据划分

### 9.1 TEP

- 以完整 run 为最小单位；
- train/validation/test 不共享 run；
- 主结果按 nominal 与 disturbance 分层报告；
- 另设未见扰动类型 OOD 测试；
- 不随机拆行。

### 9.2 Debutanizer 与 SRU

- 单连续序列时采用 60%/20%/20% chronological split；
- inner 4-fold expanding-window；
- purge 覆盖最大历史、目标提前量、目标窗口和标签可用延迟；
- 若 SRU 版本包含不同 sulfur line，优先增加跨 line 外层验证，但不得替代时间外推主结果。

### 9.3 PMSM

- `profile_id` 整段分组；
- 具体 profile 列表在模型训练前冻结；
- 分配仅使用 profile ID、时长和采样完整性，不使用模型误差；
- 报告跨 profile 与跨运行范围结果。

### 9.4 MetroPT-3

- 按连续月份划分：早期月份训练，中间月份验证，后期月份测试；
- 推荐初始冻结为 2020-02 至 2020-04 训练、2020-05 验证、2020-06 至 2020-08 测试；
- 文档记录的故障窗口从主正常预测榜中单列，作为 OOD/异常工况评估；
- 不允许随机行切分。

---

## 10. 调参与计算公平

### 10.1 统一原则

- 所有 scaler、PCA、通道分类、profile 选择、早停和超参数只使用 outer-training；
- 测试集不参与 profile、horizon、输入集或模型容量选择；
- 所有模型共享同一任务 sample ID；
- 不允许未来输入、未来目标或未成熟残差；
- 不允许实验后删除失败数据集；
- 结果报告使用任务级 win/tie/loss 与平均 rank，不把不同单位的数据直接 pooled MSE。

### 10.2 调参预算

CPU：

- 线性/统计模型每模型每任务最多 30 个配置；
- 非线性系统辨识每模型每任务最多 24 个配置；
- one-SE 优先低阶、强正则和少参数方案。

GPU：

- 核心模型每模型每 primary task 最多 24 个 trial；
- 现代扩展模型每模型每 primary task 最多 16 个 trial；
- screening 3 seeds；
- core confirmation 5 seeds；
- 按全部 primary tasks 的平均 validation rank 选前 6 名，运行 10 seeds 最终确认。

### 10.3 容量

主榜默认参数上限：

\[
N_{param}\le 250,000.
\]

超过上限的 paper-faithful 复现单列为 `LARGE_REPRODUCTION`，必须报告参数量、训练耗时、显存与推理延迟。

---

## 11. 指标与统计

每个任务报告：

- MSE、RMSE、MAE、\(R^2\)；
- nRMSE：\(\mathrm{RMSE}/\mathrm{std}(y_{test})\)；
- relative Persistence skill；
- dynamic leaderboard 中 relative AR skill；
- 每个 run/profile/month 的误差分布；
- 参数量、训练时间、推理延迟、CPU 内存或 GPU 峰值显存。

跨数据集汇总：

- 平均 rank；
- median rank；
- win/tie/loss；
- 每个数据集单独结果；
- 不使用跨单位 pooled MSE。

最终模型比较使用配对时间块 bootstrap，至少 500 次。块长度敏感性至少覆盖：

1. \(h+W\)；
2. \(2(h+W)\)；
3. 目标头最长核心历史的一个保守比例；
4. 完整运行/profile 重采样（有独立运行时）。

多模型比较使用 Holm 校正。深度模型 seed 不得当作独立时间样本重复计入 bootstrap；先形成预注册 seed ensemble，或采用 seed×time 两层 bootstrap。

---

## 12. 执行阶段

### Stage 0：数据冻结

完成五个数据集的来源、许可、哈希、变量、采样周期、运行边界、目标可用性与 split 冻结。任何一个数据集未通过，不启动正式模型比较。

### Stage 1：CPU 共享数据包与 smoke

生成统一：

- sequence view；
- multiresolution tabular view；
- graph view；
- target heads；
- sample IDs；
- masks；
- split metadata；
- scaler metadata；
- proxy-excluded/full-sensor 两套输入视图。

### Stage 2：CPU 正式实验

运行简单基线、PLS/DPLS、SVR、XGBoost、AR/ARX/NARX、系统辨识、PRISM 两条路线、尺度匹配 AR 与结构消融。

### Stage 3：GPU primary-head screening

在 7 个 primary tasks 上运行全部 GPU baseline，3 seeds。

### Stage 4：多尺度核心比较

只对下列代表模型运行全部 scale sweep：

- DPLS；
- ARX；
- NLinear；
- TCN；
- PatchTST；
- iTransformer 或 ModernTCN 中 validation 平均 rank 更高者；
- PRISM-Urysohn；
- PRISM-Urysohn-First；
- PRISM-K-Joint AR。

该阶段直接检验通道专属多尺度主张，避免所有大模型在全部 horizon 上无上限扩张计算量。

### Stage 5：finalists

按全部 primary tasks 的平均 validation rank 选前 6 名 GPU 模型，统一 FP32、TF32 关闭、10 seeds。PRISM 两条路线和 DPLS/ARX 不受前 6 名筛选限制，始终进入最终报告。

### Stage 6：鲁棒性与解释性

- 25%/50%/100% 训练数据；
- 输入噪声；
- 单通道缺失；
- proxy sensor allowed/excluded；
- single-scale/fixed-multiresolution/channel-specific；
- 时间错位 placebo；
- nominal/OOD；
- 核符号、支持、峰值、rank、exact-zero 与跨 split 稳定性。

### Stage 7：CPU/GPU 合并

统一评估器读取逐样本预测，校验 protocol hash、task hash、split hash、sample IDs 后生成最终两张 leaderboard。

---

## 13. 代码与返回包

项目根目录：

```text
PRISM_INDUSTRIAL_BENCHMARK_V1/
├── configs/
├── dataset_registry/
├── shared/
├── src/
├── scripts/
├── tests/
├── results_cpu/
├── results_gpu/
├── results_final/
└── return/
```

返回包：

```text
PRISM_INDUSTRIAL_SHARED_DATA_V1_bundle.zip
PRISM_INDUSTRIAL_CPU_RESULTS_V1_bundle.zip
PRISM_INDUSTRIAL_GPU_RESULTS_V1_bundle.zip
PRISM_INDUSTRIAL_FINAL_RESULTS_V1_bundle.zip
```

每个包必须包含：

- `MANIFEST.json`；
- 每个文件 SHA256；
- protocol/split/task/sample-ID hashes；
- commit SHA；
- 环境与硬件信息；
- round-trip 解压校验；
- raw-data/privacy/license exclusion 检查。

---

## 14. 完成标准

V1 只有在以下条件全部满足时结束：

1. 5 个数据集均有冻结来源和 split；
2. 7 个 primary tasks 均保留全部 baseline 成败记录；
3. 输入型与动态排行榜完全分开；
4. 每个物理 profile 有尺度匹配诊断 AR；
5. 每个目标头有独立 residual AR 与 K-Joint AR；
6. CPU/GPU 使用一致 sample IDs；
7. 最终比较基于逐样本预测和配对统计；
8. 失败数据集、失败方向和 exact-zero 结果未被删除；
9. 返回包 manifest、SHA256、隐私与许可检查全部通过。

