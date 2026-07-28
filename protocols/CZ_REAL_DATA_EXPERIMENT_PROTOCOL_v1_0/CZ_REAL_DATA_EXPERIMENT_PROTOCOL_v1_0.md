# CZ 等径阶段真实数据实验协议 v1.0
## OPS-UOI / Spectral PS-AR-RAPHU

> 日期：2026-07-28  
> 阶段：真实工业数据 development protocol  
> 研究范围：单晶硅直拉等径阶段  
> 目标变量：晶体直径，单位 mm  
> 炉压单位：Torr（托），不是 Pa  
> 主结论范围：固定阶段系统辨识与软测量，不外推到引晶、放肩、转肩和收尾阶段

---

# 1. 数据审计结论

## 1.1 实际只有两条独立时间序列

- `实验数据1(3).xlsx / Sheet1`：20,103 个样本；
- `实验数据1-张.xlsx / Sheet1`：与上一条完全相同；
- `实验数据1-张.xlsx / Sheet2`：20,627 个样本，为另一炉数据。

因此：

\[
\boxed{\text{独立序列数}=2}
\]

不能把两个文件中的 Sheet1 当成两根晶棒。

## 1.2 采样周期

第一炉具有晶体长度和晶升速度。利用：

\[
\Delta L
\approx
\sum_t v_{\mathrm{pull},t}\Delta t
\]

反推：

\[
\Delta t\approx1.997\text{ s}.
\]

因此实验暂按：

\[
\boxed{\Delta t=2\text{ s}}
\]

解释物理时间。第二炉没有时间列和长度列，暂按相同记录周期处理；正式运行前必须由数据提供方确认。

第一炉约 11.15 h，第二炉约 11.46 h。

## 1.3 单位

| 变量 | 单位 |
|---|---|
| 加热元件温度 | °C |
| 主加热功率 | kW |
| 晶升速度 | mm/min |
| 晶转速度 | rpm |
| 埚升速度 | mm/min |
| 埚转速度 | rpm |
| 氩气流量设定 | L/min |
| 晶体长度 | mm |
| 炉压 | Torr |
| 晶体直径 | mm |

---

# 2. 变量角色冻结

## 2.1 主实验输入 \(U_5\)

主实验只使用五个主要操纵变量：

\[
U_5=
\{
P_{\mathrm{heater}},
v_{\mathrm{crystal}},
\omega_{\mathrm{crystal}},
v_{\mathrm{crucible}},
\omega_{\mathrm{crucible}}
\}.
\]

即：

1. 主加热功率；
2. 晶升速度；
3. 晶转速度；
4. 埚升速度；
5. 埚转速度。

目标：

\[
y_t=\text{晶体直径}_t.
\]

这是两炉共同具备的输入集合，也是跨炉测试的唯一主输入集合。

## 2.2 加热元件温度

文件中加热元件温度并非常数，第一炉范围约 1245–1271.6 °C，第二炉约 1246–1273.3 °C。

但它不是主要操纵量，因此冻结为：

```text
PRIMARY_MODEL: exclude
SENSITIVITY_BRANCH: include as measured state
```

对应两条输入实验：

\[
M_0:U_5\rightarrow y,
\]

\[
M_T:(U_5,T_{\mathrm{heater}})\rightarrow y.
\]

主结论基于 \(M_0\)，\(M_T\) 用于判断温度测量是否提供额外预测信息。

## 2.3 其余变量

### 氩气流量

第一炉中严格恒定：

\[
130\ \mathrm{L/min}.
\]

删除，不进入任何模型。

### 炉压

炉压单位为 Torr。第一炉中均值约 23 Torr、标准差约 0.044 Torr，而且第二炉没有该变量。

因此：

- 不进入跨炉主模型；
- 可做第一炉 A-only 辅助消融；
- 不据此声称跨炉增益。

### 晶体长度

晶体长度从 1 mm 单调增长到 385 mm。它是工艺进程坐标，不是操纵变量。

主模型删除，原因不是“完全恒定”，而是：

1. 它近似编码时间；
2. 可能让模型通过进度查表而不是学习动力学；
3. 第二炉缺少该变量；
4. 会削弱跨炉可迁移性。

它只用于：

- 样本周期核验；
- 时间分层；
- 按晶棒位置报告误差；
- 检查模型误差是否随进程漂移。

---

# 3. 科学问题

## Q1：预测增益

五个操纵变量在强 AR 背景下是否提供真实增量预测价值：

\[
\Delta_{U|AR}(h)
=
L_{AR}(h)-L_{U+AR}(h)?
\]

## Q2：完整系统辨识

固定辨识模型能否完成：

\[
\text{输入序列}
+
\text{初始输出历史}
\longrightarrow
\text{完整 free-run 直径轨迹}?
\]

## Q3：谱秩

full kernel、rank-1、rank-2 和 adaptive rank 的预测损失分别是多少？

## Q4：跨炉泛化

在第一炉上冻结的模型能否迁移到第二炉？

## Q5：温度变量的地位

加入加热元件温度后，增益是稳定的还是只在同炉内有效？

## Q6：解释性

哪些输入通道在什么时滞和幅值范围内具有：

- Q 层预测贡献；
- K 层局部可解释性；
- 弱激励或不可辨识警告？

---

# 4. 数据划分

严禁随机切分。晶体直径的一阶自相关超过 0.999，随机切分会严重泄漏。

## 4.1 第一炉：development 序列

第一炉用于所有结构和超参数选择。

### Development 区

前 80%：

\[
\mathcal D_A^{\mathrm{dev}}
=
[0,0.8N_A).
\]

在其中执行 expanding-window folds：

| Fold | Train | Validation |
|---|---|---|
| F1 | 0–40% | 40–50% |
| F2 | 0–50% | 50–60% |
| F3 | 0–60% | 60–70% |
| F4 | 0–70% | 70–80% |

### A 炉内部 confirmation

后 20%：

\[
\mathcal D_A^{\mathrm{conf}}
=
[0.8N_A,N_A).
\]

在结构全部冻结前不得访问。

## 4.2 第二炉：跨炉 outer test

第二炉整条序列作为外部泛化轨道：

\[
\mathcal D_B^{\mathrm{outer}}.
\]

主结果：

```text
TRAIN/FREEZE: Furnace A
OUTER TEST: Furnace B
```

所有 scaler、basis domain、history、penalty、rank 和 continuation 参数均由第一炉冻结。

## 4.3 跨炉轻量校准轨道

在零样本跨炉结果完成后，增加工业部署轨道：

- 使用第二炉前 5% 或 10% 样本校准；
- 只允许更新 intercept、output scale 或预注册低维 adapter；
- 其余 kernel/history/rank 冻结；
- 在后续 90% 上评价。

该轨道与 zero-shot 分开：

```text
ZERO_SHOT_CROSS_FURNACE
FEW_SHOT_CALIBRATED_CROSS_FURNACE
```

## 4.4 Purge gap

每一 fold 的训练与验证之间设置：

\[
G=L_\star+h_{\max},
\qquad
L_\star=\max(L_x-1,L_y-1).
\]

禁止历史窗口跨越 split 边界。

---

# 5. 预测任务

按暂定 2 s 采样周期：

## 5.1 Primary direct horizons

\[
h\in\{1,5,15,30,60\}
\]

分别对应约：

\[
2,\ 10,\ 30,\ 60,\ 120\text{ s}.
\]

## 5.2 Long-horizon sensitivity

\[
h=150
\]

约 5 min，只作探索性结果，不阻塞主协议。

## 5.3 Free-run simulation

给定：

- validation/test 的真实操纵输入序列；
- 开始处的真实 \(L_y\) 个输出初始化；

之后：

\[
\widehat y_{t+1}
=
\widehat{\mathcal M}
(U_{\le t},\widehat Y_{\le t}),
\]

不得读取中间真实直径，不得在线更新参数。

---

# 6. 模型矩阵

## 6.1 最低基线

1. Mean predictor；
2. Persistence：
   \[
   \widehat y_{t+h}=y_t;
   \]
3. AR-only；
4. X-only/FIR；
5. linear ARX。

## 6.2 非线性基线

1. pNARX；
2. MLP-NARX；
3. GRU 或 LSTM，二选一；
4. 原 AKGNN/reference implementation；
5. Stage1TargetDelayKAN，若代码可复现。

## 6.3 我们的方法

1. rank-1 Spectral PS-AR-RAPHU；
2. fixed rank-2；
3. full spectral；
4. adaptive predictive rank；
5. H3 shared-history 消融；
6. H2 native-history 主模型。

所有模型必须使用同一输入集合、同一 split 和同一评价区间。

---

# 7. H2 原生模型选择

## 7.1 History 候选

由于采样周期约 2 s，不能继续把 \(L_x=32\) 当成默认真值。

外生历史：

\[
L_x\in\{32,64,128,256,512\}
\]

对应约：

\[
1.1,\ 2.1,\ 4.3,\ 8.5,\ 17.1\text{ min}.
\]

AR 历史：

\[
L_y\in\{8,16,32,64,128\}
\]

对应约：

\[
16,\ 32,\ 64,\ 128,\ 256\text{ s}.
\]

## 7.2 Resolution 候选

历史长度与 lag basis 数量分离：

\[
M_\tau\in\{16,32,48,64\},
\]

\[
M_x\in\{16,20,24,28,32\}.
\]

不使用“历史长度等于谱分辨率”的混合参数化。

## 7.3 选择顺序

\[
\boxed{
\text{history}
\rightarrow
\text{resolution}
\rightarrow
\text{penalty}
\rightarrow
\text{simulation closure}
\rightarrow
\text{rank}
}
\]

使用：

- blocked validation；
- one-SE rule；
- exact-zero penalty endpoint；
- FP64 KKT certification；
- v4.1 bounded \(C^1\) continuation；
- full-first-then-compress。

---

# 8. 预处理

## 8.1 原始值保存

保存：

```text
raw_value
model_value
quality_flag
```

不覆盖原数据。

## 8.2 标准化

每个 fold 只使用 train 部分计算：

\[
\mu_j^{\mathrm{tr}},\quad
s_j^{\mathrm{tr}}.
\]

validation、A-confirmation 和 B-outer 全部沿用 train scaler。

## 8.3 异常值

不做全局平滑和粗暴删除。

只标记：

- 非有限值；
- 完全重复时间行；
- 传感器尖峰；
- 不可能的执行器跳变。

主实验使用原始时序；异常处理作为 sensitivity。

## 8.4 不允许的操作

- 随机 shuffle；
- 全数据归一化；
- 用第二炉扩展第一炉 basis domain；
- 使用未来直径；
- direct 预测使用未来未知输入；
- 把晶体长度当普通预测变量；
- 把 continuation 域外曲线解释为已识别 K。

---

# 9. 指标

## 9.1 原始单位指标

\[
RMSE,\quad MAE,\quad R^2,\quad
P_{50}(|e|),P_{90}(|e|),P_{95}(|e|),\max|e|.
\]

全部以 mm 报告。

## 9.2 标准化指标

\[
NRMSE_{\sigma}
=
\frac{RMSE}{\operatorname{Std}(y)}.
\]

## 9.3 强 AR 条件下的增量指标

\[
\Delta_{U|AR}(h)
=
\frac{MSE_{AR}(h)-MSE_{U+AR}(h)}
{MSE_{AR}(h)}.
\]

## 9.4 增量轨迹指标

为了防止高自相关掩盖模型能力，增加：

\[
RMSE_{\Delta y}
=
RMSE\left(
(\widehat y_t-\widehat y_{t-1})
-
(y_t-y_{t-1})
\right).
\]

## 9.5 free-run

分别报告：

- 完整 free-run RMSE；
- 漂移斜率；
- 最大连续偏差；
- 初值扰动敏感度；
- continuation 使用率和最大外推距离。

---

# 10. 统计比较

Development：

\[
B=250
\]

移动块或 stationary block bootstrap。

最终 confirmation：

\[
B=1000.
\]

比较单位为连续时间块，不把 20,000 个高度相关样本当作独立样本。

主比较：

\[
\Delta RMSE
=
RMSE_{\mathrm{baseline}}
-
RMSE_{\mathrm{ours}}.
\]

报告 95% block-bootstrap 区间。

---

# 11. 消融实验

## A. 输入角色

| 编号 | 输入 |
|---|---|
| A0 | \(U_5\) |
| A1 | \(U_5+\)加热元件温度 |
| A2 | 第一炉 \(U_5+\)温度+炉压，仅辅助 |
| A3 | 加入晶体长度，泄漏风险对照，不作主模型 |

A3 的作用是证明“进程坐标查表”是否虚高，不允许成为主结论。

## B. 动力结构

- X-only；
- AR-only；
- XAR；
- rank-1；
- rank-2；
- full；
- adaptive rank。

## C. History

- H3 baseline-shared；
- H2 native。

## D. Simulation closure

- hard fail；
- boundary clip；
- linear \(C^1\)；
- bounded \(C^1\)。

主模型使用 bounded \(C^1\)。

## E. 跨炉

- zero-shot；
- intercept-only calibration；
- 5% adapter；
- 10% adapter。

---

# 12. 可解释性输出

每个操纵变量报告：

1. 时间核或 lag basis；
2. 幅值响应；
3. full kernel surface；
4. singular spectrum；
5. rank-1/rank-2 重构；
6. bootstrap band；
7. support mask；
8. weak-operator/inactive 状态。

K 层只能在：

\[
\mathcal S^{\mathrm{cert}}
\]

内解释。

跨炉测试后，可分别在 A、B 上重新拟合模型并比较：

- singular value profile；
- leading mode correlation；
- principal angle；
- lag peak consistency；
- amplitude response shape。

该步骤属于 post-confirmation mechanism analysis，不参与模型选择。

---

# 13. 成功门槛

## 必须门槛

```text
NO_RANDOM_SPLIT
TRAIN_ONLY_SCALER
PURGE_PASS
OFFICIAL_OUTER_FURNACE_NOT_USED_IN_TUNING
ALL_PRIMARY_KKT_PASS
H2_NATIVE_COMPLETE
FREE_RUN_COMPLETE
INTERPRETATION_FIREWALL_PASS
```

## 科学成功

至少满足：

1. \(U_5+AR\) 在多个 horizon 上稳定优于 AR-only；
2. H2 不劣于 H3，或给出明确更小复杂度；
3. rank-2/full 相对 rank-1 有可重复增益；
4. 第一炉内部 confirmation 稳定；
5. 第二炉 zero-shot 或轻量校准后仍有增益；
6. 解释结果在 bootstrap 中具有方向稳定性。

## 失败也有价值的状态

```text
AR_DOMINATES_EXTERNAL_INPUT
CROSS_FURNACE_DOMAIN_SHIFT
TEMPERATURE_STATE_REQUIRED
WEAK_OPERATOR_OR_INACTIVE
DIRECT_GOOD_FREE_RUN_POOR
K_LEVEL_NOT_IDENTIFIED
```

这些结果必须保留，不能通过删除 horizon 或改变 split 规避。

---

# 14. 执行顺序

```text
CZ-R0  data audit and schema freeze
CZ-R1  persistence/AR/ARX smoke
CZ-R2  H3 shared-history spectral smoke
CZ-R3  H2 native history-resolution-penalty
CZ-R4  rank/full/continuation/free-run
CZ-R5  complete nonlinear baselines
CZ-R6  Furnace A internal confirmation
CZ-R7  Furnace B zero-shot outer test
CZ-R8  5%/10% few-shot calibration
CZ-R9  mechanism and interpretability audit
```

第二炉在 CZ-R7 之前保持锁定。

---

# 15. 当前最重要的研究定位

本轮不声称：

- 多阶段全过程模型；
- 跨所有晶棒的普适模型；
- 因果 plant kernel；
- 在线控制器；
- 主动辨识。

本轮准确表述为：

\[
\boxed{
\text{面向等径阶段、闭环弱激励数据的可解释非线性系统辨识}
}
\]

并以另一炉数据检验跨炉迁移能力。

这已经比单一晶棒随机切分强得多。
