# PRISM v0.6 离散观测域最小兼容实施 Prompt（可直接交给代码 Agent）

> 身份：你是 PRISM 项目的实现与验证 agent。请在**现有代码库**内完成本 Prompt 的 Phase 0–5。用户有一份完整理论母版 `PRISM_Theory_v0_6_Discrete_Observation_Compatible_Minimal_Extension.md`，其直接基础是 v0.5。**先定位项目已有代码与数据，不得假设本文列举的示例路径必然存在。**
>
> 任务目标：用最小、可回退、可审计的改动，把 PRISM 对**离散记录目标的时间索引、观测元数据、区段合法支持与残差诊断**显式化；继续保留现有 K/C/W/A 与 H1/H4 的模型行为。验证 2 秒采样、记录格点候选与相邻差分是否可能造成约 7 秒谱峰，但**绝不先验宣布其为真实峰来源**。
>
> 执行模式：**快速可交付、无长耗时重训**。先完成只读审计和小型合成实验，优先利用已冻结的预测、OOF、test 与历史缓存；禁止为了本任务触发 54,096 级候选遍历、完整 recursive nested-OOF 递归搜索或 10–20 分钟多步全量重训。若基础测试可在合理时间内运行，可照常执行。默认增量诊断预算 30 分钟以内（不是等待/后台工作的要求）；超出需在交付报告中明确停止点、可复现命令和原因，不要以“已验证”替代未运行的事实。

## 0. 不可违反的理论与项目约束

1. 正式结构为 `E → K → C → W → A`；K 为包含 lag–amplitude 维度的注册有限 Urysohn 响应结构，不得未经批准简化为线性 K；C 是冻结 K 通道响应的装配；W 仅读取当前冻结响应坐标 `q_t`，补足可选静态非线性；A 仅读取合法成熟的 W 后 CF/prequential 残差历史。
2. 保留 v0.5 tagged BYPASS/MODEL/N/A/UNRESOLVED、零元独立于非零参数候选、C BYPASS 不阻断 W、W BYPASS 不阻断 A；**不加入 one-SE、白化指标或量化检测门禁**。
3. 保留严格同域 parent–child OOF risk 差、fold-local 上游重建/调参、inner/outer 隔离、最后 test lockbox；不允许将全数据估计的采样/格点统计泄漏给折内自适应学习。
4. 任何新增功能**默认关闭或只读**；默认模型预测、route、训练样本行、H1/H4 target、已有 RMSE 必须保持与旧版本一致（浮点容差需报告）。严禁“顺手修复”H1 的标签索引造成旧实验不可比。
5. 不添加隐含连续状态识别、量化似然模型、Bayesian filter、量化预测后强制 rounding、全局平滑、双向滤波或通过未来数据补缺。v0.6 **不是** PRISM-Q。
6. 数据中小数记录分辨率不等于传感器真实 ADC 分辨率。数据若为两位小数，只能记作 `observed_grid_candidate=0.01`，除非找到仪表原始说明/配置才标记为 `instrument_verified`。
7. 不用任何表面相似性推断“7 秒峰已经消除”“Rod2 达到白化标准”“A 吃的是量化误差”等未经验证的结论。

## Phase 0 — 代码和数据资产发现（**先做，不许猜**）

遍历项目已有目录（不要越界访问），给出 `CODE_AUDIT.md`：

- 入口/训练/预测/加载器/历史编码 E/多率历史对齐/K/C/W/A/CF residual/目标构造器/H1/H4/谱分析与白化报告/OOF splits/评分支持的实际文件路径和函数名；
- 样本的输入频率来源、输出时间戳与单位、采样间隔检测方式、是否存在断裂段、是否仅保留了 0.01 格点、输出记录是否进一步缩放或变换；
- H1、H4 的**真实索引**：`prediction_origin_idx`、`anchor_indices`、`label_indices`、`target_ready_time`、`feedback_available_time`、`horizon_steps`、`stride`、`target_formula`，以及这些字段当前来自何处；
- 有无已冻结 `K/C/W/A` 逐 stage 预测及 H1 innovation 文件；有无 `Rod_1 → Rod_2` 与 `Rod_2 → Rod_1` 两个方向的开发集/测试集证据；标注每份资产 train/val/OOF/test 和其是否独立。

**关键检查点：** 母版使用 `mean(y[t+h : t+h+W]) - mean(y[t-W0 : t])` 的窗口公式。在 `W=W0=1` 下 `h=0` 才是 `y[t]-y[t-1]`，而 `h=1` 则是 `y[t+1]-y[t-1]`。项目名 `H1` 不代表公式一定是相邻差分！**以实际目标生成代码为权威，记录母版与实现是否等价；不要直接替换它。**

检查原 §3.2 的 `u[j,t-lag]` 在 lag=0 时是否已经在预测原点合法可用：必须满足项目真实的输入可用性政策，而不能仅依据 dataframe 列存在。

如果仓库缺少真实数据/旧预测文件：直接开展 Phase 1–3 的合成测试，Phase 4 标记 `DATA_NOT_AVAILABLE`，不伪造真实实验。

## Phase 1 — 最小类型/元数据增量（不改变数值路径）

按照现有项目语言和风格新增或扩展一个**不可变观测合同**（名字建议 `DiscreteObservationSpec`，可以用 Python dataclass / 类型定义 / JSON 配置）。至少提供：

```python
# 伪接口：按仓库实际结构改写。所有字段禁止被用来暗改旧预测。
DiscreteObservationSpec(
    sample_period_seconds: float | None,
    time_origin_policy: str,
    timestamp_field: str | None,
    output_unit: str,
    output_record_grid_candidate: float | None,
    grid_evidence_status: str,  # unknown / observed_grid_candidate / instrument_verified
    segment_policy: str,
    gap_rule: dict | None,  # 明确时间戳差阈值/数据来源/基准
    observation_availability_policy: str,
    head_contract_id: str,
)
```

要求：

- 默认旧配置解析行为与旧版完全相同；新字段可选、可序列化、有单元测试。
- 明确区分**输出记录格点**与原模型 K 输入幅值编码 `Q_j`，变量不得复用导致含义相反。
- 输出样本预测保持浮点连续值，不因 0.01 格点就 `round(pred, 2)`。
- 所有输入索引与标签依赖以**原始采样 index + 时间戳 + segment id**表示；可只在审计层导出，不强制重构训练代码。
- 若推断 grid 候选，必须标注算法和证据，不将检测结果自动喂回 W/A/route；若用于训练，视为另一个严格折内实验，本次不做。
- 对不规则采样或断裂，先产生 `valid_support` 或警告及可审计掩码。**默认旧运行行为不变**；如果旧代码会跨断裂取滞后，不可默默改变训练行，须输出 `POTENTIAL_LEAK_OR_INVALID_SUPPORT` 和后续整改建议。可以提供一个需显式开启的 `strict_segment_support` 模式，但这时训练与测试评分支持变化必须单独验收与重新评价。

## Phase 2 — 目标时间索引与信息合法性审计

实现只读函数 `audit_target_contract(data, head_config, observation_spec)`，输出逐行 `target_alignment.csv`（若样本太大允许按折导出）或等价结构：

```text
entity_id, segment_id, split_id, head_id, origin_index,
anchor_start_index, anchor_end_index,
target_start_index, target_end_index,
last_input_available_index, last_output_available_index,
label_maturity_index, prediction_value_unit, target_value_unit,
legal_support, invalid_reason
```

检查：

1. 每个目标的锚点/未来窗口、成熟/交付延迟、预测原点是否与原标签代码完全一致；
2. 对 H1/H4 与历史窗口的边界、短段、purge、跨棒隔离、split 边界、末尾未知未来控制 scenario 检测非法行；
3. `anchor_end_index` 不可在无相应权限时超过输入可见时刻；`target_start_index` 不可误作已知 predictor feature；
4. 断裂/缺失段不得自动拼接成连续时间序列用于 **新诊断**；
5. 若不合法是旧实现问题，仅审计报告提出补救，不能自动重写项目 baseline。

**注意**：每次目标定义必须有唯一 `head_id`。`recorded_level`、`registered_target`、`post-K`、`post-C`、`post-W`、`post-A innovation` 在输出列名中严格分开。

## Phase 3 — 低成本合成实验与数学单元测试（无需 PRISM 重训）

新建只读诊断模块，保持与训练包低耦合。固定 random seed，保存参数、数组摘要、图和数值表。实现如下四个反事实场景，至少使用 `N=4096, Ts=2.0 s` 的固定合成规则序列，并做必要的 N/seed 稳健性对照：

**S1：无周期输入的斜坡量化**

- `s[n] = s0 + b * n * Ts`，不添加正弦；
- 取 `delta=0.01`，分别验证 `b*Ts/delta = 1/7`、`2/7`、`3/8` 及一个非简单分数比；
- 量化 `Q_delta(s) = delta * floor(s/delta + 0.5)`（规定 tie rule/浮点容差）；
- 对照 `s[n]`、`Q_delta(s[n])` 及相邻差分 `Δs[n]`、`ΔQ[n]`，计算 ACF/PSD/step distribution；
- 断言当 `b*Ts/delta=p/q` 且算子平移等变时 `ΔQ[n+q]=ΔQ[n]`（有限浮点容差），但不要保证每组都出现**指定** 7 秒峰；周期 q=7 个采样点对应 14 秒基本周期，7 秒可能仅作为谐波出现。

**S2：平滑慢变 + 噪声 + 记录舍入**

- 使用固定慢速趋势（或二阶平滑响应）+ 已知零均值测量噪声；
- 开关是否量化，记录 `0`, `+delta`, `-delta` 的频率及其他步长；
- **不可宣称 iid 量化误差**。

**S3：真实低频微弱正弦**

- 频率分别用 `1/7 Hz` 和邻近频率做敏感性分析；比较采样前/采样后/量化后/差分后的 PSD 与 ACF；
- 说明加入 7 秒正弦再观察到 7 秒峰，**不证明**量化是根因。

**S4：奈奎斯特以上振荡引起的混叠**

- 以 `f0=0.357142857 Hz`, `fs=0.5 Hz` 为例（以实际离散 alias 计算为准），展示约 `0.142857 Hz` 的混叠峰；
- 不得据此断言实测传感器存在该连续高频分量。

**数学测试：**

- 对 iid `v[n]` 的相邻差分，理论 `rho_1=-0.5`；用足够 N 的合成重复数据检查经验值接近 -0.5，断言给出统计容差而非浮点完全相等。
- `|1-exp(-i omega)|^2 = 4 sin^2(omega/2)` 在随机频点数值一致。
- 任意 head 的 `registered_target` 必须按实际目标构造器计算；只有在其为相邻差分时才复用 Δ 的频响结论。
- 对打断时间戳的序列，诊断程序分段计算 PSD/ACF，并明确短区段无法支撑哪些 lag 或频率分辨率；**不得拼接断裂样本后当作规则连续采样**。
- 在诊断开启/关闭的条件下，固定输入调用旧模型得到完全相同的逐样本预测（或说明预注册的极小浮点差异）。

输出至少一份 `SYNTHETIC_DISCRETE_DIAGNOSTICS.md`、四组场景的 `png`/`svg` 图（图中单位、频率/lag、采样率、量化条件和样本长度清楚标注）和 `tests` 的执行记录。

## Phase 4 — 已冻结真实数据对照（若数据/预测已存在）

**零重训优先**。仅使用已冻结预测与合法样本行，不搜索新 K、W、A。分别做 `Rod_1 → Rod_2` 与 `Rod_2 → Rod_1`（若确有两方向输出）。各方向输出：

1. `OBSERVATION_AUDIT`: 时间戳差分的统计/不连续段、被记录在网格上的数值比例、0.01 仅为格点候选的依据、原始 step 的 `0/±0.01/other` 直方图；
2. `TARGET_AUDIT`: H1/H4 目标的严格定义、行数、区段、有效 lag、预测起点、成熟延迟；
3. `FROZEN_SPECTRAL_REPORT`: `recorded_level`、`registered_target` 以及逐 stage `post-K/C/W/A` 残差的 PSD、ACF、PACF（存在有效预测时）；每张图明确 split、N、采样率、滤波/去趋势、时间区段；
4. 在 **约 0.143 Hz** 附近的频率坐标处仅作标注性对照，谱峰位置/能量/频率分辨率给出具体数值，若根本不存在此峰要如实说明；
5. 将 `post-A` **一步预测创新**的白化程度与 raw recorded difference 的谱结构分开，不可混合训练内 residual 与 OOF/test innovation；
6. 对独立同分布的白噪声 ACF 参考带可报告粗略 `±1.96/sqrt(N)`，但时间依赖、折内参数估计、多重 lag 与有效样本长度会影响解释；该带不能成为接纳规则。

如没有 stage-by-stage 冻结逐样本预测，只做能确认的部分，缺项标记 `NOT_AVAILABLE`；不允许在测试集上新拟合模型或为寻找白化最优结果而调整参数。

## Phase 5 — 回归、审计与最终交付

最小验收清单（有无法运行的必须精确说明）：

- [ ] **兼容性**：默认关闭诊断时 H1/H4、K/C/W/A route 和预测结果不变；如无历史运行结果，仅做同一 checkout 的前后对照，不能伪称与原论文结果完全相等。
- [ ] **H1 时间索引**：文档列出实际公式（不是只写 H1）并给出至少一个手工算例的逐行索引检查。
- [ ] **区段**：断裂数据不会被新诊断错误拼接；严格支持模式必须显式配置，且标为可能影响训练评价的变更。
- [ ] **防泄漏**：新的观察 metadata 不使用评价标签；所有自适应估计都留在训练域或保持只读诊断。
- [ ] **角色边界**：W 仍只读取冻结 q；A 仍只读取成熟 CF/prequential 残差；不把记录差分直接冒充最终创新。
- [ ] **量化审计**：仪器原生量化步长与小数格式被区分，未知时不硬编码 0.01。
- [ ] **数学自测**：iid 差分 -0.5、量化斜坡周期、alias 与差分频响测试通过。
- [ ] **白化**：只针对冻结样本外 `post-A` 创新进行正式诊断，不将合成场景相似性视为真因鉴定。
- [ ] **时效**：不触发全量长期 K 搜索/recursive outer-fold 训练，记录实际命令与耗时。

**预期交付结构（按项目已有约定可调整，但报告文件内容不可省略）：**

```text
delivery/
  CODE_AUDIT.md
  OBSERVATION_CONTRACT.md
  TARGET_INDEX_AUDIT.md
  SYNTHETIC_DISCRETE_DIAGNOSTICS.md
  REAL_DATA_FROZEN_DIAGNOSTICS.md  # 没有数据时清楚写 NOT_AVAILABLE
  REGRESSION_TEST_RESULTS.md
  IMPLEMENTATION_DIFF_SUMMARY.md
  artifacts/
    synthetic/ ...
    real/ ...  # 仅当确有真实数据和冻结结果
```

### 代码实现建议（不强加项目目录命名）

```python
# 核心原则：新审计模块只“读取旧过程”，不截断旧训练和预测主路径。
def build_discrete_audit(data, existing_head_builder, observation_spec, segment_policy):
    align = existing_head_builder.explain_indices(data)  # 无此接口时，建立只读 adapter
    assert_no_unauthorized_cross_segment_use(align, segment_policy)
    return align  # 不改写已有 target 数值


def diagnostics_for_registered_target(recorded, aligned_target, frozen_predictions, spec):
    # 按实体/段/split/合法样本分别计算，不以未来数据填补
    # raw recorded -> target -> post-K/C/W/A residual 逐类命名
    # 只有 registered_target 真正为 adjacent difference 才调用 Δ 分析假设
    return build_psd_acf_step_report(...)


def assert_backward_compatibility(old_preds, new_preds, old_routes, new_routes):
    assert_equal_sample_ids_and_head_ids(...)
    assert_same_routing_states(old_routes, new_routes)
    assert_allclose(old_preds, new_preds, atol=registered_tolerance)
```

### 交付报告末尾强制回答的 6 个问题

1. 现有 PRISM 是否已经在离散索引上建模？有什么实际代码证据？
2. H1 的真实目标公式是什么？是否对应母版相邻记录差分特例？
3. 0.01 是已证实的仪表量化步长，还是只有数据格式候选？
4. 理论上无真实 7 秒正弦时，是否能从斜坡/量化/差分合成类似谱结构？能复现哪些特征，不能复现哪些？
5. 真实数据的 7 秒峰究竟存在于**原始记录、注册 target**还是**最终样本外创新**？结果是否足以确定来源？
6. 在不重训、不更改模型角色与门禁条件的前提下，本次代码增量到底改善了哪些能力，哪些仍属于后续 PRISM-Q 研究？

**禁止的结论**：仅凭合成信号相似就写“根因是量化”；只凭观测数据有小数格点就写“ADC 量化精度为 0.01”；只凭某几个 ACF 点越过/缩进参考带就宣布全域白化；把 v0.5 正式结构选择结果因诊断峰值而改写。

**执行顺序**：先完成能立即交付的 Phase 0–3；有真实冻结资产则做 Phase 4；最后完成 Phase 5 并给出产物相对路径、测试命令和所有失败项。不要把任何需要额外批准的大规模计算包装成“最小离散兼容改造”。
