# AR-RAPHU 三层验证方案 v2

> **版本**：v2.0  
> **执行原则**：先在已知真值和公共数据上确定方法是否有效，再回到当前单根 CZ 晶棒；如果后续取得更多晶棒，则升级为跨晶棒验证。  
> **结果状态原则**：没有运行的实验必须明确写为“尚未执行”，不得提前填入预期结果。

---

# 0. 当前状态总览

| 验证模块 | 数据状态 | 实验状态 | 当前能否下结论 |
|---|---|---|---|
| 旧 V20 非自回归合成审计 | 已有 | 已完成 | 仅支持原方案 A 的部分能力 |
| AR-RAPHU 完全合成 | 可生成 | **尚未执行** | 否 |
| CZ 输入半合成 | 当前一根棒可用 | **尚未执行** | 否 |
| TEP 长期公共数据 | 公开可得 | **尚未执行** | 否 |
| TEP 主动干预 | 模拟器公开可得 | **尚未执行** | 否 |
| Debutanizer | 公开小型真实软测量数据可得 | **尚未执行** | 否 |
| UCI Gas Turbine CO/NOx | 公开真实数据可得 | **尚未执行** | 否 |
| OpenCGS | 开源模拟器可得 | **尚未执行** | 否 |
| 当前单根 CZ | 已有 | 新方案 **尚未执行** | 否 |
| 多根 CZ | 可能后续获得 | 当前尚无 | 否 |

---

# 1. 需要验证的核心命题

\[
\mathrm{P1}:
\quad
\text{方案 A 能在强自回归持续性下恢复真实外生变量。}
\tag{1.1}
\]

\[
\mathrm{P2}:
\quad
\text{固定 A 的支持和时滞后，凸样条内层能稳定重建响应。}
\tag{1.2}
\]

\[
\mathrm{P3}:
\quad
\text{A+B 能区分 Gamma 错设、一般 rank-1 和真正 rank-2。}
\tag{1.3}
\]

\[
\mathrm{P4}:
\quad
\text{过程变量在 AR-only 之外提供稳定的预测增量。}
\tag{1.4}
\]

\[
\mathrm{P5}:
\quad
\text{上述结构在公共独立轨迹和主动干预中可复现。}
\tag{1.5}
\]

\[
\mathrm{P6}:
\quad
\text{若取得多根晶棒，结构可跨晶棒泛化。}
\tag{1.6}
\]

当前只有 P1–P5 可以先行验证；P6 取决于后续数据。

---

# 2. 统一模型矩阵

## B0：Persistence

\[
\widehat y_t=y_{t-1}.
\tag{2.1}
\]

多步直接基线：

\[
\widehat y_{t+h\mid t}=y_t.
\tag{2.2}
\]

## B1：线性 AR

\[
\widehat y_t
=
b+
\sum_{\ell=1}^{L_y}a_\ell y_{t-\ell}.
\tag{2.3}
\]

## B2：线性 ARX

\[
\widehat y_t
=
b+
\sum_\ell a_\ell y_{t-\ell}
+
\sum_j\sum_\tau h_{j,\tau}x_{j,t-\tau}.
\tag{2.4}
\]

## M1：经典样条/FIR 并联 Hammerstein

不使用历史输出，作为 X-only 系统辨识基线。

## M2：Stable-Spline/Kernel Hammerstein

用于检验 Gamma 动态先验过窄的问题。

## M3：方案 A，X-only

\[
\widehat y_t^X
=
b+
\sum_jq_j^x*f_j^x(x_j).
\tag{2.5}
\]

## M4：方案 A，AR-only

\[
\widehat y_t^{AR}
=
b_y+q^y*f^y(y_{\mathrm{past}}).
\tag{2.6}
\]

## M5：方案 A，X+AR

\[
\widehat y_t^{X+AR}
=
b+
\sum_jq_j^x*f_j^x(x_j)
+
q^y*f^y(y_{\mathrm{past}}).
\tag{2.7}
\]

## M6：自由平滑动态核 rank-1

用于区分单 Gamma 错设和 rank 错设。

## M7：A 锚定一维凸样条重拟合

固定 M5 的支持和时滞，只重建显式一维响应。

## M8：A 锚定正交 B 残差

仅升级外生过程变量。

## M9：Rank-2 外生并联 Hammerstein

只对通过审计的变量启用。

## M10：完整外生 Urysohn 面

作为高灵活上界，不默认用于历史输出分支。

## M11：普通时序深度基线

至少包括 TCN、GRU/LSTM 和一个 Transformer 类模型。

---

# 3. 统一任务轨道

每个能够提供历史顺序的数据集，都至少建立：

## Track-X

只用过程变量：

\[
X_{\le t}\rightarrow y_{t+h}.
\]

## Track-AR

只用历史目标：

\[
y_{\le t}\rightarrow y_{t+h}.
\]

## Track-XAR

同时使用：

\[
(X_{\le t},y_{\le t})\rightarrow y_{t+h}.
\]

必须报告：

\[
\Delta_{\mathrm{X\mid AR}}(h)
=
\operatorname{Loss}_{AR}(h)
-
\operatorname{Loss}_{XAR}(h).
\tag{3.1}
\]

---

# 4. Layer I：完全合成与半合成真值

## 4.1 AR-S0：纯自回归

\[
y_t=F_y(y_{t-1:t-L_y})+\varepsilon_t.
\tag{4.1}
\]

要求：

- 外生变量不得稳定进入支持；
- B 不得凭空制造外生二维结构。

## 4.2 AR-S1：自回归 + Gamma rank-1 外生机制

\[
y_t
=
F_y(y_{\mathrm{past}})
+
\sum_jq_j^\Gamma*f_j(x_j)
+
\varepsilon_t.
\tag{4.2}
\]

用于验证强持续性下的外生变量恢复。

## 4.3 AR-S2：非 Gamma rank-1

\[
K_j^\star(\tau,u)=h_j^\star(\tau)f_j(u).
\tag{4.3}
\]

要求自由核 rank-1 能解释，而不能误判为 rank-2。

## 4.4 AR-S3：Rank-2

\[
K_j^\star
=
q_{j,1}f_{j,1}
+
q_{j,2}f_{j,2}.
\tag{4.4}
\]

用于验证第二机制检测能力。

## 4.5 AR-S4：幅值依赖时滞

\[
K_j^\star(\tau,u)
=
q_j(\tau\mid u)f_j(u).
\tag{4.5}
\]

用于验证条件平均时滞曲线。

## 4.6 AR-S5：测量滤波

先生成潜在真实直径 \(z_t\)，再生成观测：

\[
y_t
=
\sum_{r=0}^{R_m-1}
h_m(r)z_{t-r}
+\nu_t.
\tag{4.6}
\]

用于检验模型是否把传感器滤波误当成过程时滞。

## 4.7 AR-S6：测量延迟

\[
y_t=z_{t-d_m}+\nu_t.
\tag{4.7}
\]

## 4.8 AR-S7：闭环控制与变量共线

用过去输出生成部分操纵量，检验闭环内生性导致的假结构。

## 4.9 半合成输入来源

分别使用：

- AR/VAR 随机输入；
- 当前 CZ 一根棒的真实输入；
- TEP 真实或模拟输入。

这样既有真值，又保留工业自相关和共线结构。

## 4.10 Layer I 尚未完成

```text
STATUS: NOT_YET_RUN
```

在该层完成前，不冻结最终 rank 阈值。

---

# 5. Layer II-A：Tennessee Eastman 主公共基准

## 5.1 数据来源

Tennessee Eastman 公开模拟器提供完整工业过程动态、41 个测量变量、12 个操纵变量和多类扰动。经典观测向量通常使用 41 个 XMEAS 与 11 个 XMV，共 52 个变量。公开代码还允许修改模拟长度并主动施加扰动。

来源：

- https://github.com/jkitchin/tennessee-eastman-profbraatz
- https://github.com/anasouzac/new_tep_datasets
- 原始问题 DOI：https://doi.org/10.1016/0098-1354(93)80018-I

## 5.2 TEP-X

使用在线过程量预测产品 G/H 组分，不使用历史质量标签。

用途：

- 检验纯软测量；
- 比较结构化动态模型和普通深度模型；
- 检验外生 rank。

## 5.3 TEP-XAR

加入过去 G/H 组分。

用途：

- 检验 AR 持续性是否掩盖过程变量；
- 测量 \(\Delta_{\mathrm{X\mid AR}}\)；
- 检验 X-only 与 XAR 的结构差异。

## 5.4 主动幅值扫描

对操纵变量施加多个幅值的阶跃/脉冲，构造

\[
K_j^{\mathrm{emp}}(\tau,a)
=
y_{j,a}(t_0+\tau)-y_0(t_0+\tau).
\tag{5.1}
\]

它为 rank 检验提供近似干预真值。

## 5.5 数据规模

执行顺序：

1. 先运行现成公开数据；
2. 再生成 \(5\times10^6\) 时间步；
3. 方法成立后再扩展到 \(2\times10^7\) 时间步。

## 5.6 当前状态

```text
DATA: AVAILABLE
EXPERIMENT: NOT_YET_RUN
RESULT: NOT_AVAILABLE
```

---

# 6. Layer II-B：Debutanizer 真实软测量数据

## 6.1 数据特征

公开文献中的 Debutanizer 数据包含：

- 7 个过程输入；
- 1 个底部产品丁烷含量目标；
- 2394 条工业过程记录。

来源：

- 论文：https://doi.org/10.1016/j.aej.2016.02.016
- 数据镜像与工程示例：
  https://github.com/Ujjwal-1267/industrial-debutanizer-soft-sensor

## 6.2 在本项目中的作用

它是一个真实软测量外部验证集，但规模较小。

必须先确认：

- 数据原始顺序是否为时间顺序；
- 是否有采样周期；
- 目标是否有实验室分析延迟；
- 是否允许构造 AR 轨道。

若时间顺序不能可靠确认：

- 只用于静态/滞后特征软测量对照；
- 不用于严格 rank 时滞审计；
- 不把随机切分结果作为动态泛化证据。

## 6.3 当前状态

```text
DATA: PUBLICLY ACCESSIBLE
CHRONOLOGY AUDIT: NOT_YET_DONE
EXPERIMENT: NOT_YET_RUN
RESULT: NOT_AVAILABLE
```

---

# 7. Layer II-C：UCI Gas Turbine CO/NOx

## 7.1 数据特征

UCI 数据包含：

- 36,733 个小时级观测；
- 11 个传感器/运行变量；
- CO 和 NOx 排放目标；
- 2011–2015 五个年度文件。

来源：

- https://archive.ics.uci.edu/dataset/551/gas%2Bturbine%2Bco%2Band%2Bnox%2Bemission%2Bdata%2Bset
- DOI：https://doi.org/10.24432/C5WC95

## 7.2 作用和限制

作用：

- 检验真实工业多变量回归的一般化；
- 按年份做跨时间测试；
- 比较 X-only 与简单 AR/XAR；
- 检查外生变量稀疏与函数稳定性。

限制：

- 数据为小时聚合；
- 不一定具有适合 Hammerstein 物理时滞解释的高频连续动态；
- 不能承担主动干预 rank 真值；
- 主要作为外部泛化补充，而不是核心 rank 证据。

## 7.3 当前状态

```text
DATA: AVAILABLE
EXPERIMENT: NOT_YET_RUN
RESULT: NOT_AVAILABLE
```

---

# 8. Layer II-D：OpenCGS 晶体生长物理补充

OpenCGS 是公开晶体生长模拟框架，可进行 CZ 热场模拟。

来源：

- https://github.com/nemocrys/opencgs
- 验证论文：https://doi.org/10.1016/j.jcrysgro.2022.126750

用途：

- 检查加热功率、几何和热梯度响应方向；
- 构造不同工作点；
- 为真实 CZ 解释提供物理一致性证据。

限制：

- 它不是现成的大规模动态直径时序；
- 第一版不把它当主要预测基准。

当前状态：

```text
SIMULATOR: AVAILABLE
EXPERIMENT: NOT_YET_RUN
RESULT: NOT_AVAILABLE
```

---

# 9. Layer III：当前单根 CZ

## 9.1 已确认的数据范围

```text
crystal_count = 1
stage = constant-diameter stage only
process_channels = 9
historical_output_channels = 1
model_input_channels = 10
target = final spreadsheet column
```

尚未确认：

```text
units = NOT_YET_AVAILABLE
sampling_period = NOT_YET_AVAILABLE
diameter_measurement_method = NOT_YET_AVAILABLE
sensor_locations = NOT_YET_AVAILABLE
measurement_filtering = NOT_YET_AVAILABLE
measurement_delay = NOT_YET_AVAILABLE
```

## 9.2 当前能够做的验证

只能做：

- 单晶棒内部 expanding-window；
- 不同后段的时间外推；
- AR-only/X-only/XAR 对比；
- 多预测步长；
- 同一阶段结构稳定性；
- block bootstrap；
- 半合成真值桥梁。

不能声称：

- 跨晶棒泛化；
- 跨炉次泛化；
- 跨阶段泛化；
- 工业部署可靠性。

## 9.3 当前 CZ 任务

### CZ-T1：一步预测

用于复现旧任务，但必须加入 persistence 和 AR-only。

### CZ-T2：多步预测

当前先用步数：

\[
h\in\{1,5,10,30,60\}.
\]

采样周期确认后转换为物理时间。

### CZ-T3：X-only

检查过程变量自身的解释力。

### CZ-T4：XAR

检查过程变量对 AR-only 的边际收益。

### CZ-T5：A+B 外生 rank 审计

固定历史输出分支为 rank-1，只审计过程变量。

## 9.4 常量通道

当前氩气流量设定常数：

- 在接口中保留；
- 在科学训练中屏蔽；
- 不报告其响应和时滞。

## 9.5 当前状态

```text
DATA: AVAILABLE
METADATA: PARTIALLY_AVAILABLE
NEW_MODEL_EXPERIMENT: NOT_YET_RUN
RESULT: NOT_AVAILABLE
```

---

# 10. 如果取得更多晶棒

## 10.1 2–4 根

进行探索性 leave-one-rod-out：

\[
\text{train on }R-1\text{ rods},
\quad
\text{test on one rod}.
\]

由于验证集不足，结果仍需谨慎。

## 10.2 5–9 根

可进行：

- grouped cross-validation；
- 独立验证棒；
- 独立测试棒；
- 支持概率和 rank 稳定性统计。

## 10.3 10 根以上

可研究：

\[
K_{j,r}
=
K_{j,0}+U_{j,r},
\]

即共享机制加晶棒随机偏差。

同时可以回答：

- 哪些时滞跨棒稳定；
- 哪些变量只在部分棒中有效；
- rank 是否是共性还是个体差异；
- 新棒少样本适配。

## 10.4 若获得多阶段

再加入阶段变量 \(s_t\)，研究：

\[
K_j(\tau,u\mid s_t).
\]

这属于后续切换 Hammerstein/Urysohn 扩展，不与当前第一阶段混合。

---

# 11. 实验执行顺序

## Phase 0：代码和语义冻结

- 审计第十通道的对齐；
- 确认不存在当前标签泄漏；
- 冻结 Gamma 离散化；
- 冻结 KAN 结构；
- 冻结数据切分和 scaler 顺序。

## Phase 1：AR-RAPHU 合成真值

先确定方法是否能在强 AR 情形下恢复外生结构。

## Phase 2：TEP

先完成公开大规模数据和主动干预。

这是当前最优先的外部证据。

## Phase 3：Debutanizer 与 Gas Turbine

补充真实工业泛化。

## Phase 4：当前单根 CZ

在公共实验规则冻结后再运行，避免看到 CZ 结果后调整阈值。

## Phase 5：多根 CZ

数据一旦到达，追加 grouped split，不重写前面公共实验规则。

---

# 12. 第一轮最小实验矩阵

| 编号 | 数据 | 任务 | 模型 | 状态 |
|---|---|---|---|---|
| E1 | AR-S0 | AR-only 真值 | B0/B1/M4/M5 | 未运行 |
| E2 | AR-S1 | Gamma rank-1 | M3/M5/M7/M8 | 未运行 |
| E3 | AR-S2 | 非 Gamma rank-1 | M5/M6/M8 | 未运行 |
| E4 | AR-S3 | rank-2 | M6/M8/M9 | 未运行 |
| E5 | TEP | X/XAR | B2/M3/M5/M7/M8/M11 | 未运行 |
| E6 | TEP 主动扫描 | 经验核 | M3/M8/M9/M10 | 未运行 |
| E7 | Debutanizer | X 或 XAR | M1/M3/M5/M11 | 未运行 |
| E8 | Gas Turbine | 年度外推 | M1/M3/M5/M11 | 未运行 |
| E9 | CZ | AR/X/XAR | B0/B1/B2/M3/M4/M5 | 未运行 |
| E10 | CZ | 外生 rank | M6/M7/M8/M9 | 未运行 |

---

# 13. 通过标准

## 13.1 合成层

- 纯 AR 真值下，外生误选率低；
- rank-1 假阳性率不超过预定阈值；
- rank-2 检出率达到预定功效；
- 能区分 Gamma 错设和 rank 错设；
- 条件时滞恢复方向正确。

阈值需由实验前预注册。

## 13.2 TEP

- 公共脚本可复现；
- XAR 相对 AR-only 在多个 horizon 上有稳定收益；
- 主动响应面与估计面主要结构一致；
- rank 决策跨轨迹稳定；
- 大规模训练资源可控。

## 13.3 当前单根 CZ

- 复杂模型必须击败 persistence 或明确承认未击败；
- XAR 必须报告相对 AR-only 的增量；
- 至少多个滚动折方向一致；
- B 升级必须有外层测试收益和 bootstrap 证据；
- 结论限定为该晶棒等径阶段。

## 13.4 多根 CZ

- 必须在未见晶棒上评估；
- 所有预处理只 fit 训练晶棒；
- 支持和 rank 的稳定性按晶棒统计，而非按时间点统计。

---

# 14. 结果记录规范

任何实验结果必须保存：

```text
dataset_manifest.json
variable_dictionary.csv
split_manifest.json
preprocessing_manifest.json
model_semantics.json
config.yaml
seed.json
train_log.csv
validation_selection.json
test_metrics.json
predictions.parquet
support.json
lag_kernel.npy
response_grid.npz
surface_grid.npz
singular_values.json
conditional_delay.csv
contributions.parquet
runtime.json
environment.json
SHA256SUMS.txt
```

结果汇总文档必须区分：

```text
COMPLETED
FAILED
NOT_YET_RUN
NOT_APPLICABLE
BLOCKED_BY_MISSING_METADATA
BLOCKED_BY_MISSING_DATA
```

---

# 15. 资源计划

本地 RTX 5080 足以进行：

- 最小合成矩阵；
- TEP 初始长期数据；
- 当前单根 CZ；
- 5–10 个随机种子；
- 初步 bootstrap。

以下阶段开始值得租高核心 CPU/大内存服务器：

- TEP 达到 500 万至 2000 万时间步；
- 完整模型矩阵和多 horizon；
- 500 次以上 bootstrap；
- 多根晶棒 grouped cross-validation；
- 大规模多超参数并行。

---

# 16. 公开数据地址

## Tennessee Eastman

- https://github.com/jkitchin/tennessee-eastman-profbraatz
- https://github.com/anasouzac/new_tep_datasets
- https://doi.org/10.1016/0098-1354(93)80018-I

## Debutanizer

- https://doi.org/10.1016/j.aej.2016.02.016
- https://github.com/Ujjwal-1267/industrial-debutanizer-soft-sensor

## Gas Turbine CO/NOx

- https://archive.ics.uci.edu/dataset/551/gas%2Bturbine%2Bco%2Band%2Bnox%2Bemission%2Bdata%2Bset
- https://doi.org/10.24432/C5WC95

## OpenCGS

- https://github.com/nemocrys/opencgs
- https://doi.org/10.1016/j.jcrysgro.2022.126750

---

# 17. 最终研究结构

当前研究按以下顺序形成证据：

\[
\boxed{
\text{AR 合成真值}
\rightarrow
\text{TEP 大规模公开验证}
\rightarrow
\text{真实外部工业数据}
\rightarrow
\text{单根 CZ 案例}
\rightarrow
\text{未来多根 CZ 跨棒验证}
}
\]

这允许在多根晶棒尚未取得时先推进方法，又不会把单根晶棒的证据范围夸大。
