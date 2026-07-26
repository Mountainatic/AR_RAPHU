# Predictive-State AR-RAPHU v3：D1–D6 诊断实验完整执行方案

> **文档性质**：实现合同，不是方向性建议。  
> **目标**：用六组最小诊断实验明确区分以下五种失败来源：  
> 1. rank-2 模型容量不足；  
> 2. Scheme A 的 rank-1 结构盲区；  
> 3. 强 AR 分支吸收外生增量信息；  
> 4. 联合训练中的优化捷径与训练顺序问题；  
> 5. 现有 group-prox 稀疏惩罚的尺度失衡与梯度饿死。  
> **禁止事项**：Codex 不得自行增加实验、调整阈值、改动数据生成器、改变训练预算、引入新网络、启动 M7/M8/M9/M10、运行公共数据或重新解释停止线结果。  
> **当前基线来源**：`AR_RAPHU_STOPLINE_20260725.zip/source/`。  
> **核心原则**：复用现有底层实现，新建独立诊断入口；不在 v2 停止线脚本中打补丁。

---

# 0. 本轮只回答什么问题

本轮不重新验证完整 AR-RAPHU，不追求论文最终结果，只回答：

\[
\boxed{
\text{为什么强 AR 下外生支持和 rank 恢复失败？}
}
\]

六项实验的职责固定如下。

| 实验 | 唯一主问题 | 场景 |
|---|---|---|
| D1 | 正确 rank-2 模型在无 AR、无稀疏时是否具备容量和可优化性？ | AR-S3 / E4 |
| D2 | 即使给出真实支持，rank-1 模型类是否无法表达 rank-2 真值？ | AR-S3 / E4 |
| D3 | 冻结 AR 后，外生变量是否仍能解释 AR 创新残差？ | AR-S3 / E4 |
| D4 | 在模型类正确时，训练顺序是否决定 X 能否在 XAR 中存活？ | AR-S1 / E2 |
| D5 | 统一分支尺度并只惩罚显式 gate 后，是否出现高召回低假阳性的支持区间？ | AR-S1 / E2 |
| D6 | X 梯度是否在贡献形成前被 AR 和 proximal 收缩压制？ | AR-S1 / E2 |

D1–D3 使用 rank-2 真值场景；D4–D6 使用 Gamma rank-1 真值场景，以避免把“rank 模型类错误”和“AR/优化错误”混在一起。

---

# 1. 现有代码从哪里启动

## 1.1 工作目录

从停止线包解压后的 `source/` 开始：

```bash
cd AR_RAPHU_STOPLINE_20260725/source
```

不要修改以下目录：

```text
STAGE1_DUAL_SOLVER_V20_bundle/
tools/run_phase1_scheme_a.py
tools/run_phase1_m6.py
tools/run_phase1_m7.py
tools/run_phase1_m8.py
configs/protocol_v2.yaml
```

这些文件作为 v2 参照，只读使用。

## 1.2 新建文件

只新增以下文件：

```text
configs/v3_diagnostics.yaml

src/ar_raphu/diagnostics/
├── __init__.py
├── config.py
├── rank2_model.py
├── residual_data.py
├── train_utils.py
├── truth_metrics.py
├── gate_fista.py
└── instrumentation.py

tools/
├── run_v3_diagnostic_job.py
└── run_v3_diagnostic_suite.py

tests/
├── test_v3_rank2_model.py
├── test_v3_residual_data.py
├── test_v3_gate_fista.py
└── test_v3_instrumentation.py
```

允许对 `src/ar_raphu/synthetic.py` 做且只做一个兼容性改动：公开两个只读包装函数：

```python
def truth_response(variable: int, values: np.ndarray) -> np.ndarray:
    return _truth_response(variable, values)

def second_truth_response(variable: int, values: np.ndarray) -> np.ndarray:
    return _second_truth_response(variable, values)
```

不得改变合成数据生成逻辑。

## 1.3 必须复用的现有组件

直接复用：

```python
from ar_raphu.synthetic import generate_synthetic_sequence
from ar_raphu.sequence_data import PreparedDirectForecastData
from ar_raphu.model import ARRAPHURank1
from ar_raphu.training import seed_everything, evaluate_rmse
from ar_raphu.phase1_evidence import partition_predictions_and_contributions
from STAGE1_DUAL_SOLVER_V20_bundle.stage1.model import Stage1TargetDelayKAN
```

不得重写：

- 合成数据生成器；
- scaler；
- 滑动窗口；
- sequence-first grouped convolution；
- KAN；
- Gamma 核；
- 当前 AR 分支。

---

# 2. 统一诊断配置

创建 `configs/v3_diagnostics.yaml`，内容必须等价于：

```yaml
schema_version: 1
status: DIAGNOSTIC_ONLY

common:
  seeds: [0, 1, 2, 3, 4]
  d6_seeds: [0, 1, 2]
  n_samples: 10000
  external_variables: 10
  active_support: [0, 1, 2]
  L_x: 64
  L_y: 32
  primary_horizon: 1
  conditional_horizons: [5, 10]
  hidden_kan: 8
  grid_size: 7
  spline_order: 3
  response_execution_mode: vectorized
  physical_chunk: 4096
  deterministic_algorithms: true
  dtype_neural: float32
  validation_interval: 10
  response_learning_rate: 0.003
  lag_learning_rate: 0.0005
  ar_learning_rate: 0.003
  joint_finetune_learning_rate: 0.0003
  free_lag_smoothness: 0.001
  min_learning_rate: 0.00001
  scheduler_factor: 0.5
  scheduler_patience_validations: 20

D1:
  scenario: AR-S3
  variants:
    - rank2_oracle_q
    - rank2_learned_q_truth_init
  fixed_component_weights: [0.6, 0.4]
  oracle_epochs: 2000
  learned_q_epochs: 3000
  patience: 300

D2:
  scenario: AR-S3
  variants:
    - rank1_free_q
    - rank2_free_q
  epochs: 3000
  patience: 300
  rank1_q_initialization: weighted_truth_mixture
  rank2_q_initialization: truth_components

D3:
  scenario: AR-S3
  ar_epochs: 2000
  residual_model_epochs: 2500
  patience: 250
  variants:
    - residual_rank1_free_q
    - residual_rank2_free_q
  conditional_extension_trigger:
    minimum_failed_seed_count: 4
    innovation_r2_threshold: 0.10
    horizons: [5, 10]

D4:
  scenario: AR-S1
  variants:
    - simultaneous
    - x_first
    - ar_first
  simultaneous_epochs: 2500
  x_first:
    x_pretrain_epochs: 1500
    ar_fit_epochs: 500
    joint_epochs: 500
  ar_first:
    ar_pretrain_epochs: 1000
    x_fit_epochs: 1500
    joint_epochs: 500
  patience: 250

D5:
  scenario: AR-S1
  source_checkpoint: D4_x_first_best
  lambda_ratios: [0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0]
  fista_max_iterations: 10000
  fista_tolerance: 1.0e-9
  support_threshold: 1.0e-8

D6:
  scenario: AR-S1
  warmup_epochs: 2000
  pruning_epochs: 1200
  pruning_scale: 0.003
  ramp_epochs: 300
  log_interval: 10
  diagnostic_sample_count: 1024
  starvation:
    gradient_ratio_threshold: 0.10
    contribution_signal_threshold: 0.05
    consecutive_logs: 5
    shrink_threshold: 0.99

runtime:
  gpu_workers: 8
  oom_fallback_workers: 4
  cpu_threads_per_worker: 1
  use_cuda_mps: true
```

Codex 不得修改这些值。

---

# 3. 本轮精简审计政策

## 3.1 仍然保留的四项快速检查

正式运行前只运行：

```bash
conda run -n Env_pytorch --no-capture-output \
  python -m pytest \
  tests/test_ar_raphu_model.py \
  tests/test_sequence_training.py \
  tests/test_v3_rank2_model.py \
  tests/test_v3_residual_data.py \
  tests/test_v3_gate_fista.py \
  tests/test_v3_instrumentation.py \
  -q
```

必须检查：

1. 输入截止索引小于目标索引；
2. rank-2 前向形状正确；
3. residual dataset 的目标和原目标索引完全对齐；
4. FISTA 在小型人工 Lasso 上收敛并产生精确零系数。

## 3.2 本轮删除的审计

本轮不得执行：

- 全部 118/全仓库测试；
- 每个中间文件 SHA256；
- `PACKAGE_MANIFEST.json`；
- HTML 报告；
- 每个作业的环境快照；
- Git commit/tag 冻结检查；
- M7/M8 manifest；
- Fold 4 lockbox 守卫；
- 公共数据下载校验；
- 每阶段压缩包；
- checkpoint 重放审计；
- 前向和梯度双路径全量等价审计。

## 3.3 每个作业只保存

```text
config.json
summary.json
training_log.csv
best.pt
```

只有确实需要结构指标的作业额外保存：

```text
surface_metrics.json
lag_kernels.npz
gradient_timeline.csv
gate_path.csv
```

不生成 SHA 文件。

---

# 4. 新增 rank-2 模型的精确定义

## 4.1 类名

在 `src/ar_raphu/diagnostics/rank2_model.py` 实现：

```python
class ARRAPHURank2Diagnostic(nn.Module):
    ...
```

## 4.2 结构

外生部分必须由两个相互独立的 `ARRAPHURank1(track="X")` 实例组成：

```python
self.component_1
self.component_2
```

两个实例：

- `external_channels=10`；
- `L_x=64`；
- 独立 KAN；
- 独立 lag kernel；
- 只允许真实支持 `[0,1,2]` 活跃；
- 内部 wrapper bias 置零并冻结。

固定权重：

```python
self.register_buffer("component_weights", torch.tensor([0.6, 0.4]))
```

预测：

\[
\widehat y_t^X
=
b+
0.6\,C_{1,t}
+
0.4\,C_{2,t}.
\]

若 `include_ar=True`，再增加一个现有 `ARRAPHURank1(track="AR")`：

\[
\widehat y_t
=
b+
0.6\,C_{1,t}
+
0.4\,C_{2,t}
+
C_t^{AR}.
\]

不得增加 component attention、mixture-of-experts、交叉变量网络或额外 MLP。

## 4.3 q 模式

支持且只支持：

```text
oracle_fixed
free_truth_init
```

### `oracle_fixed`

调用：

```python
component_1.external_branch.set_fixed_delays(q_primary)
component_2.external_branch.set_fixed_delays(q_secondary)
```

### `free_truth_init`

两个 component 使用：

```python
external_delay_mode="free_static_logits"
```

初始化：

```python
delay_logits.copy_(torch.log(q.clamp_min(1e-8)))
```

非真实支持行保持零 logits，但因 active mask 不参与输出或梯度。

## 4.4 优化器参数组

不得把所有参数放入同一个学习率组。

```python
optimizer = torch.optim.Adam([
    {"params": response_parameters, "lr": 3e-3},
    {"params": lag_logit_parameters, "lr": 5e-4},
    {"params": bias_and_ar_parameters, "lr": 3e-3},
])
```

free lag penalty：

\[
\lambda_q
\sum_{j,r}
\|\Delta^2 \operatorname{logit}q_{j,r}\|_2^2,
\qquad
\lambda_q=10^{-3}.
\]

---

# 5. 统一训练函数

在 `train_utils.py` 实现：

```python
@dataclass
class DiagnosticTrainResult:
    best_state: dict[str, Tensor]
    best_epoch: int
    best_validation_rmse: float
    history: list[dict]

def train_diagnostic_model(
    model,
    data,
    *,
    max_epochs: int,
    patience: int,
    response_lr: float,
    lag_lr: float,
    ar_lr: float,
    joint_lr: float | None,
    lag_smoothness: float,
    validation_interval: int,
    batch_size: int,
    device: torch.device,
) -> DiagnosticTrainResult:
    ...
```

每个 epoch 使用现有：

```python
data.iter_contiguous_batches(...)
model.forward_contiguous(...)
```

进行全训练段梯度累积。

早停依据只能是 validation RMSE。

不得：

- 使用 test 早停；
- 运行稀疏 proximal；
- 自动改变支持；
- 自动改变 rank；
- 自动扩展 epoch；
- 自动重新初始化失败种子。

---

# 6. D1：rank-2 容量与可优化性

## 6.1 D1 的目的

D1 只判断：

\[
\text{给出真实支持和正确 rank 后，模型是否能拟合 rank-2 真值？}
\]

D1 不包含 AR，不包含稀疏，不包含 one-SE。

## 6.2 数据

```python
sequence = generate_synthetic_sequence(
    "AR-S3",
    seed=seed,
    n_samples=10000,
    external_variables=10,
)
```

数据：

```python
PreparedDirectForecastData.from_sequence(
    sequence.x,
    sequence.y_observed,
    track="X",
    horizon=1,
    L_x=64,
    L_y=32,
    split_target_intervals=sequence.split_target_intervals,
)
```

虽然模型不使用 `L_y`，仍传入 32 保持接口一致。

## 6.3 D1-A：`rank2_oracle_q`

### 模型

- `ARRAPHURank2Diagnostic(include_ar=False)`；
- support 固定 `[0,1,2]`；
- component q 使用 `sequence.truth["q_primary"]` 和 `q_secondary`；
- q 冻结；
- KAN 和总 bias 可训练；
- component 权重固定 0.6/0.4。

### 训练

- 2000 epoch 上限；
- patience 300；
- response lr 0.003；
- 无 lag penalty；
- validation interval 10。

### 输出指标

1. validation/test RMSE、R²；
2. 每个真实变量的整体 surface NRMSE；
3. surface correlation；
4. 预测贡献 closure；
5. 各 component 的响应曲线，仅作诊断，不要求 component 唯一对应。

整体真实 surface：

\[
K_j^\star(\tau,u)
=
0.6q_{j,1}^\star(\tau)f_{j,1}^\star(u)
+
0.4q_{j,2}^\star(\tau)f_{j,2}^\star(u).
\]

学习 surface：

\[
\widehat K_j(\tau,u)
=
0.6q_{j,1}^\star(\tau)\widehat f_{j,1}(u)
+
0.4q_{j,2}^\star(\tau)\widehat f_{j,2}(u).
\]

幅值网格使用每个变量训练段原始值的 1%–99% 分位区间，201 点。

### 诊断通过条件

至少 4/5 seeds 同时满足：

```text
validation R2 >= 0.95
mean active-variable surface NRMSE <= 0.20
mean active-variable surface correlation >= 0.90
```

## 6.4 D1-B：`rank2_learned_q_truth_init`

与 D1-A 相同，但：

- q 为 free logits；
- 从真实 q 初始化；
- q 可训练；
- lag lr 0.0005；
- lag smoothness 0.001；
- 3000 epoch。

额外指标：

- 每个 component 的 Wasserstein-1 lag error；
- mean lag error；
- q entropy；
- q boundary mass。

### 诊断通过条件

至少 4/5 seeds 同时满足：

```text
validation RMSE <= 1.10 * 同 seed 的 D1-A validation RMSE
mean component W1 lag error <= 3.0 samples
mean active-variable surface NRMSE <= 0.25
```

## 6.5 D1 结论编码

```text
D1_CAPACITY_PASS
D1_CAPACITY_FAIL
D1_LAG_OPTIMIZATION_FAIL
```

判定：

- D1-A 失败：`D1_CAPACITY_FAIL`；
- D1-A 通过、D1-B 失败：`D1_LAG_OPTIMIZATION_FAIL`；
- 两者都通过：`D1_CAPACITY_PASS`。

---

# 7. D2：rank-1 结构盲区

## 7.1 D2 的目的

排除 Gamma 错设，只比较：

\[
\text{自由 rank-1}
\quad\text{vs}\quad
\text{自由 rank-2}.
\]

两者都给真实支持、都无 AR、都无稀疏。

## 7.2 D2-R1：自由 rank-1

使用现有：

```python
ARRAPHURank1(
    track="X",
    external_delay_mode="free_static_logits",
    inactive_external_channels=(3,4,5,6,7,8,9),
    ...
)
```

q 初始值固定为：

\[
q_j^{\mathrm{mix}}
=
0.6q_{j,1}^\star+0.4q_{j,2}^\star.
\]

初始化：

```python
delay_logits.copy_(log(q_mix.clamp_min(1e-8)))
```

训练：

- 3000 epoch；
- response lr 0.003；
- lag lr 0.0005；
- smoothness 0.001；
- patience 300。

## 7.3 D2-R2：自由 rank-2

使用 D1-B 同一模型和初始化。

## 7.4 比较指标

同 seed 成对比较：

\[
G_{\mathrm{RMSE}}
=
\frac{\mathrm{RMSE}_{R1}-\mathrm{RMSE}_{R2}}
{\mathrm{RMSE}_{R1}},
\]

\[
G_{\mathrm{surface}}
=
\frac{\mathrm{NRMSE}_{R1}-\mathrm{NRMSE}_{R2}}
{\mathrm{NRMSE}_{R1}}.
\]

另外报告：

- rank-1 最佳 surface NRMSE；
- rank-2 surface NRMSE；
- 每个真实变量 contribution correlation；
- rank-1 q 是否形成两个峰之间的折中核。

## 7.5 结构盲区判定

若至少 4/5 seeds 满足任一条件：

### 条件 A：预测和结构都明显改善

```text
G_RMSE >= 0.05
G_surface >= 0.30
```

### 条件 B：预测近似等价但结构恢复明显不同

```text
abs(G_RMSE) < 0.05
G_surface >= 0.30
```

则输出：

```text
D2_RANK1_BLIND_SPOT_CONFIRMED
```

条件 B 的含义是：

> rank-1 预测可能足够，但不具备生成结构恢复能力。

若 rank-1 与 rank-2 的 surface 都恢复良好，则输出：

```text
D2_RANK1_BLIND_SPOT_NOT_CONFIRMED
```

---

# 8. D3：AR 创新残差路径

## 8.1 D3 的目的

判断在给定历史输出以后，X 是否仍保留预测信息：

\[
r_{t,h}^{AR}
=
y_{t+h}-\widehat y_{t+h\mid t}^{AR}.
\]

## 8.2 第一步：训练 AR-only

每个 seed、每个当前 horizon 单独训练：

```python
ARRAPHURank1(track="AR", horizon=h, L_y=32, ...)
```

主轮先只运行 `h=1`。

训练：

- 2000 epoch；
- lr 0.003；
- patience 250；
- 无稀疏。

保存：

```text
ar_best.pt
ar_predictions_train.npy
ar_predictions_validation.npy
ar_predictions_test.npy
```

## 8.3 第二步：构造 residual dataset

在 `residual_data.py` 实现：

```python
@dataclass
class PreparedExternalResidualData:
    x_scaled: np.ndarray
    residual_scaled: np.ndarray
    ...
```

要求：

1. X scaler 继续使用原始 `PreparedDirectForecastData` 的 train-only scaler；
2. residual 使用原目标的 scaled 单位：
   \[
   r^{scaled}=y^{scaled}-\hat y^{AR,scaled};
   \]
3. 不再次对 residual 标准化；
4. target index 和 origin index 与原任务完全一致；
5. 只提供 X sequence，不提供 y window。

## 8.4 第三步：训练 residual rank-1

- 真实支持 `[0,1,2]`；
- X-only；
- free q；
- q 初始化为 weighted truth mixture；
- 2500 epoch；
- response lr 0.003；
- lag lr 0.0005；
- smoothness 0.001。

## 8.5 第四步：训练 residual rank-2

- 真实支持 `[0,1,2]`；
- rank-2；
- q 从真实 components 初始化；
- 2500 epoch；
- 其余同上。

## 8.6 指标

创新解释率：

\[
R^2_{\mathrm{innov}}
=
1-
\frac{\sum(r-\widehat r)^2}
{\sum(r-\bar r)^2}.
\]

联合预测：

\[
\widehat y^{AR+innovation}
=
\widehat y^{AR}+\widehat r^X.
\]

报告：

- validation/test innovation \(R^2\)；
- AR RMSE；
- AR+innovation RMSE；
- rank-1 和 rank-2 residual surface NRMSE；
- \(\Delta_{\mathrm{X\mid AR}}\)。

## 8.7 条件扩展规则

主轮 h=1 完成后，若 rank-2 residual model 在至少 4/5 seeds 上：

```text
validation innovation R2 < 0.10
```

则自动且只自动追加：

```text
h = 5
h = 10
```

重复完整 D3 流程。

不得追加 h=30 或 h=60。

## 8.8 D3 结论

```text
D3_X_INFORMATION_REMAINS_AT_H1
D3_X_INFORMATION_EMERGES_AT_LONGER_HORIZON
D3_AR_MEDIATES_MOST_X_INFORMATION
```

判定：

- h=1 中位 innovation R² ≥0.10：第一项；
- h=1 <0.10，但 h=5 或 10 ≥0.10：第二项；
- h=1、5、10 均 <0.10：第三项。

---

# 9. D4：训练顺序与 AR shortcut

## 9.1 D4 的目的

D4 使用 AR-S1，因为 rank-1 模型类与真值完全匹配。若这里仍出现 X 弱化，就能归因于联合优化而非 rank 错设。

所有模型：

- XAR；
- Gamma rank-1；
- 10 个外生变量全部存在；
- 无稀疏；
- 不剪枝；
- h=1。

## 9.2 Variant A：`simultaneous`

随机初始化完整 XAR，一次性联合训练：

```text
2500 epochs
response lr = 0.003
Gamma lr = 0.003（保持当前 v2 行为）
AR lr = 0.003
```

这作为当前联合训练参照。

## 9.3 Variant B：`x_first`

### Phase B1：X-only 预训练

```text
track = X
1500 epochs
lr = 0.003
```

### Phase B2：组装 XAR

创建新的 XAR：

- 复制 X-only 的 `external_branch.state_dict()`；
- 复制 X-only wrapper bias 到 joint bias；
- 冻结 external branch 和 joint bias；
- AR branch 随机初始化。

训练 AR：

```text
500 epochs
AR lr = 0.003
```

### Phase B3：联合微调

解冻全部参数：

```text
500 epochs
all branch lr = 0.0003
```

禁止稀疏。

## 9.4 Variant C：`ar_first`

### Phase C1：AR-only 预训练

```text
1000 epochs
lr = 0.003
```

### Phase C2：组装 XAR

- 复制 AR branch；
- 复制 AR bias 到 joint bias；
- 冻结 AR branch 和 joint bias；
- X branch 随机初始化。

训练 X：

```text
1500 epochs
X response/Gamma lr = 0.003
```

### Phase C3：联合微调

```text
500 epochs
all branch lr = 0.0003
```

## 9.5 D4 指标

由于没有稀疏，不能用“是否 active”作为支持。

计算每个外生变量贡献能量：

\[
E_j
=
\mathbb E_{\mathrm{val}}[c_{j,t}^2].
\]

指标：

1. true-support energy fraction：
   \[
   \frac{\sum_{j\in S^\star}E_j}{\sum_jE_j};
   \]
2. top-3 contribution recall；
3. active vs inactive median energy ratio；
4. active-variable response NRMSE；
5. lag W1；
6. validation RMSE；
7. X 总贡献 RMS / target std；
8. AR 总贡献 RMS / target std。

## 9.6 shortcut 判定

若至少 4/5 seeds 中 `x_first` 相对 `simultaneous` 同时满足：

```text
true-support energy fraction 增加 >= 0.20
active response NRMSE 降低 >= 20%
validation RMSE 不恶化超过 2%
```

则：

```text
D4_AR_SHORTCUT_CONFIRMED
```

若三种顺序表现接近，则：

```text
D4_TRAINING_ORDER_NOT_PRIMARY
```

若 `ar_first` 明显优于 `x_first`，记录：

```text
D4_AR_RESIDUALIZATION_HELPFUL
```

但不得自动改变最终架构。

---

# 10. D5：标准化 gate + 凸 Lasso 支持路径

## 10.1 D5 的目的

D5 不再直接惩罚 KAN 参数块。它固定已经训练好的分支，只解决：

\[
\text{统一尺度后，线性 gate 能否正确分离支持？}
\]

D5 使用每个 seed 的 D4 `x_first` 最优 checkpoint。

## 10.2 提取贡献设计矩阵

从训练段提取外生贡献：

\[
C\in\mathbb R^{n\times10},
\qquad
C_{tj}=c_{j,t}.
\]

提取固定 AR contribution：

\[
a\in\mathbb R^n.
\]

中心化：

\[
\bar c_j=\frac1n\sum_tC_{tj},
\]

\[
s_j=
\sqrt{
\frac1n
\sum_t(C_{tj}-\bar c_j)^2
}
+
10^{-8}.
\]

标准化设计：

\[
Z_{tj}
=
\frac{C_{tj}-\bar c_j}{s_j}.
\]

调整目标：

\[
r_t=y_t-a_t.
\]

## 10.3 Lasso 模型

\[
\min_{b,g}
\frac1{2n}
\|r-b\mathbf1-Zg\|_2^2
+
\lambda\|g\|_1.
\]

禁止继续训练 KAN、Gamma 或 AR。

## 10.4 lambda 路径

计算：

\[
\lambda_{\max}
=
\frac1n
\|Z^\top(r-\bar r)\|_\infty.
\]

按降序运行：

```text
0.32 λmax
0.16 λmax
0.08 λmax
0.04 λmax
0.02 λmax
0.01 λmax
0.005 λmax
0
```

每个 lambda 使用前一个较大 lambda 解 warm start。

## 10.5 FISTA

在 `gate_fista.py` 实现标准 FISTA：

- \(g\) 使用 soft-threshold；
- \(b\) 不惩罚；
- Lipschitz 常数：
  \[
  L=\lambda_{\max}(Z^\top Z/n);
  \]
- 最大 10000 iteration；
- 相对参数变化小于 \(10^{-9}\) 停止；
- 每 100 iteration 检查目标单调性；
- 若 FISTA 动量导致目标增加，重启动量，但不改变步长。

## 10.6 支持定义

\[
\widehat S(\lambda)
=
\{j:|g_j|>10^{-8}\}.
\]

对每个 lambda 保存：

- support；
- recall；
- FPR；
- precision；
- validation RMSE；
- gate values；
- KKT residual。

不使用 one-SE，不自动选择 lambda。

## 10.7 D5 判定

若至少 4/5 seeds 的 gate 路径中存在至少一个 lambda 同时满足：

```text
support recall >= 0.80
false positive rate <= 0.10
```

则：

```text
D5_SCALE_NORMALIZED_GATE_PATH_SUCCESS
```

否则：

```text
D5_GATE_PATH_STILL_NOT_SEPARABLE
```

D5 只判断路径是否存在，不选择论文最终 lambda。

---

# 11. D6：梯度饿死与 proximal 时间线

## 11.1 D6 的目的

直接记录：

\[
\frac{\|\nabla_X\mathcal L\|}
{\|\nabla_{AR}\mathcal L\|},
\]

以及 X 贡献形成时间和 proximal 收缩时间。

只用 seeds `[0,1,2]`，场景 AR-S1。

## 11.2 训练流程

### Phase W：联合 dense warmup

完整 XAR 随机初始化：

```text
2000 epochs
lr = 0.003
no sparsity
no early stopping
```

每 10 epoch 记录一次。

### Phase P：从 Phase W 最佳 validation checkpoint 开始 pruning

完全复现当前 pruning 计算：

```text
requested_scale = 0.003
ramp_epochs = 300
pruning_epochs = 1200
full-split gradient accumulation
apply_group_proximal_step
```

D6 不进行 5000 epoch fixed-support refit。

## 11.3 梯度记录位置

必须在：

```text
loss.backward()
之后
optimizer.step()
之前
```

读取梯度。

记录：

### AR

- AR KAN grad norm；
- AR Gamma grad norm；
- AR 总 grad norm。

### 每个 X 变量

- KAN branch grad norm；
- Gamma parameter grad norm；
- branch parameter norm；
- contribution RMS；
- q mean；
- q std；
- q entropy；
- q last-3 boundary mass。

### 全局

- train MSE；
- validation RMSE；
- X total grad norm；
- AR total grad norm；
- ratio \(G_X/G_{AR}\)；
- X total contribution RMS / target std；
- AR contribution RMS / target std；
- current support；
- proximal shrink factor；
- penalty scale。

诊断 contribution 使用固定的训练段前 1024 个合法 target 和 validation 前 1024 个 target。不得每次随机抽样。

## 11.4 starvation 事件定义

在某个真实变量 \(j\) 上，若连续 5 个记录点满足：

\[
\frac{
\|\nabla_{\theta_j^X}\mathcal L\|
}{
\|\nabla_{\theta^{AR}}\mathcal L\|+10^{-12}
}
<0.10,
\]

同时：

\[
\frac{\operatorname{RMS}(c_j)}
{\operatorname{Std}(y)}
<0.05,
\]

并且 pruning 阶段出现：

\[
\text{shrink}_j<0.99
\]

或变量被 prune，则定义该变量发生 starvation。

## 11.5 D6 判定

若至少 2/3 seeds 中，至少两个真实变量发生 starvation：

```text
D6_GRADIENT_STARVATION_CONFIRMED
```

若 X 梯度长期不低，但仍恢复失败：

```text
D6_NOT_A_GRADIENT_MAGNITUDE_PROBLEM
```

若 warmup 已形成明显 X contribution，但 pruning 后消失：

```text
D6_PROXIMAL_COLLAPSE_CONFIRMED
```

多个标签可以同时成立。

---

# 12. 运行器的精确 CLI

## 12.1 单作业入口

`tools/run_v3_diagnostic_job.py` 只允许：

```bash
python tools/run_v3_diagnostic_job.py \
  --experiment D1 \
  --variant rank2_oracle_q \
  --seed 0 \
  --horizon 1 \
  --device cuda
```

参数：

```text
--experiment {D1,D2,D3,D4,D5,D6}
--variant <该实验预声明 variant>
--seed <预声明 seed>
--horizon {1,5,10}
--device {cuda,cpu}
--force
```

不得提供任意 epoch、lr、grid、lambda 覆盖参数。所有数值只从 `v3_diagnostics.yaml` 读取。

## 12.2 整套入口

```bash
python tools/run_v3_diagnostic_suite.py \
  --experiment D1 \
  --device cuda
```

或：

```bash
python tools/run_v3_diagnostic_suite.py \
  --all \
  --device cuda
```

suite 按顺序运行：

```text
D1 → D2 → D3 → D4 → D5 → D6
```

D5 必须等待 D4 x_first checkpoint。

## 12.3 GPU 并发

`run_v3_diagnostic_suite.py` 使用子进程并发，不生成 manifest。

固定：

```text
workers = 8
CPU threads per worker = 1
```

若任何子进程返回 CUDA OOM：

1. 停止尚未启动的作业；
2. 已完成作业保留；
3. 只对未完成作业改用 4 workers；
4. 不改变 batch、模型或训练预算；
5. 不再进行第三次自动调整。

---

# 13. 本地与 AutoDL 启动命令

## 13.1 本地 Conda

```bash
cd <project-root>

conda run -n Env_pytorch --no-capture-output \
  python -m pytest \
  tests/test_ar_raphu_model.py \
  tests/test_sequence_training.py \
  tests/test_v3_rank2_model.py \
  tests/test_v3_residual_data.py \
  tests/test_v3_gate_fista.py \
  tests/test_v3_instrumentation.py \
  -q

conda run -n Env_pytorch --no-capture-output \
  python tools/run_v3_diagnostic_suite.py --all --device cuda
```

## 13.2 AutoDL/uv

若继续使用现有 AutoDL `.venv`：

```bash
cd <project-root>

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

bash deploy/autodl/mps.sh start

.autodl-tools/uv run python -m pytest \
  tests/test_ar_raphu_model.py \
  tests/test_sequence_training.py \
  tests/test_v3_rank2_model.py \
  tests/test_v3_residual_data.py \
  tests/test_v3_gate_fista.py \
  tests/test_v3_instrumentation.py \
  -q

.autodl-tools/uv run python \
  tools/run_v3_diagnostic_suite.py --all --device cuda

bash deploy/autodl/mps.sh stop
```

本轮不运行：

```text
deploy/autodl/preflight.py
verify_server.sh
make_phase1_*_manifest.py
build_stopline_artifact.py
package_stopline_release.py
```

---

# 14. 输出目录

固定为：

```text
results/v3_diagnostics/
├── D1/
│   ├── seed_0/
│   │   ├── rank2_oracle_q/
│   │   └── rank2_learned_q_truth_init/
│   └── aggregate_summary.json
├── D2/
├── D3/
├── D4/
├── D5/
├── D6/
├── diagnostic_summary.csv
└── DIAGNOSTIC_DECISION.md
```

每个 variant/seed 只保存：

```text
config.json
summary.json
training_log.csv
best.pt
```

D1/D2/D3 增加：

```text
surface_metrics.json
lag_kernels.npz
```

D5 增加：

```text
gate_path.csv
```

D6 增加：

```text
gradient_timeline.csv
```

---

# 15. 自动汇总，不允许 Codex自行解释

`run_v3_diagnostic_suite.py` 完成后按照本文阈值自动生成 `DIAGNOSTIC_DECISION.md`。

文件必须只包含以下字段：

```text
D1_STATUS:
D2_STATUS:
D3_STATUS:
D4_STATUS:
D5_STATUS:
D6_STATUS:

PRIMARY_FAILURE_CLASS:
SECONDARY_FAILURE_CLASS:
SUPPORTED_NEXT_ARCHITECTURE_CHANGE:
UNSUPPORTED_CHANGES:
```

## 15.1 主失败类型映射

### 情况 1

```text
D1_CAPACITY_FAIL
```

则：

```text
PRIMARY_FAILURE_CLASS = RANK2_MODEL_CAPACITY_OR_IMPLEMENTATION
SUPPORTED_NEXT_ARCHITECTURE_CHANGE = FIX_RANK2_MODEL_BEFORE_ANY_SCREENING_CHANGE
```

### 情况 2

```text
D1_CAPACITY_PASS
D2_RANK1_BLIND_SPOT_CONFIRMED
```

则：

```text
PRIMARY_FAILURE_CLASS = SCHEME_A_RANK1_HARD_GATE
SUPPORTED_NEXT_ARCHITECTURE_CHANGE = ADD_SCHEME_B_RESCUE_FOR_A_REJECTED_VARIABLES
```

### 情况 3

```text
D3_AR_MEDIATES_MOST_X_INFORMATION
```

则：

```text
PRIMARY_FAILURE_CLASS = CONDITIONAL_INFORMATION_LIMIT
SUPPORTED_NEXT_ARCHITECTURE_CHANGE = SEPARATE_PREDICTIVE_SUPPORT_FROM_GENERATIVE_SUPPORT
```

### 情况 4

```text
D4_AR_SHORTCUT_CONFIRMED
```

则：

```text
PRIMARY_FAILURE_CLASS = JOINT_OPTIMIZATION_SHORTCUT
SUPPORTED_NEXT_ARCHITECTURE_CHANGE = X_FIRST_OR_RESIDUALIZED_CURRICULUM
```

### 情况 5

```text
D5_SCALE_NORMALIZED_GATE_PATH_SUCCESS
```

则：

```text
SECONDARY_FAILURE_CLASS = GROUP_PROX_SCALE_MISMATCH
SUPPORTED_NEXT_ARCHITECTURE_CHANGE += NORMALIZED_SCALAR_GATE_SELECTION
```

### 情况 6

```text
D6_GRADIENT_STARVATION_CONFIRMED
```

则：

```text
SECONDARY_FAILURE_CLASS = GRADIENT_STARVATION
SUPPORTED_NEXT_ARCHITECTURE_CHANGE += DELAYED_SPARSITY_AND_BRANCHWISE_OPTIMIZATION
```

---

# 16. 明确禁止的下一步

D1–D6 完成前，Codex 不得：

- 重跑当前停止线 Phase 1；
- 扩展到 30 seeds；
- 启动 M7/M8 bootstrap；
- 实现最终 v3；
- 运行 TEP；
- 运行 Debutanizer；
- 运行 Gas Turbine；
- 运行 CZ；
- 实现 PLC/MCU；
- 改 one-SE；
- 扩大 penalty 网格；
- 增加 Transformer/GRU/TCN；
- 改合成数据 SNR；
- 改 AR 强度；
- 使用真实支持之外的信息改最终主模型；
- 把诊断 oracle 实验写成正式方法结果。

---

# 17. 预计计算规模

本轮不再产生百万级 refit epoch。

按最大预算估计：

| 实验 | 约模型训练数 | 单模型最大 epoch |
|---|---:|---:|
| D1 | 10 | 3000 |
| D2 | 10 | 3000 |
| D3 | 15（h=1） | 2500 |
| D4 | 15 schedules | 3000 总阶段 |
| D5 | 5 个凸路径 | 无神经 refit |
| D6 | 3 条 warmup+prune 时间线 | 3200 |

即使触发 D3 的 h=5/10，规模也远低于停止线的 247 万累计 epoch。

---

# 18. 最终交付内容

本轮完成后只返还：

```text
src/ar_raphu/diagnostics/
tools/run_v3_diagnostic_job.py
tools/run_v3_diagnostic_suite.py
configs/v3_diagnostics.yaml
tests/test_v3_*.py
results/v3_diagnostics/
DIAGNOSTIC_DECISION.md
```

不生成 HTML，不生成逐文件 SHA，不生成复杂 manifest。

最终目录可以简单压缩：

```bash
zip -r PS_AR_RAPHU_V3_DIAGNOSTICS_RESULTS.zip \
  src/ar_raphu/diagnostics \
  tools/run_v3_diagnostic_job.py \
  tools/run_v3_diagnostic_suite.py \
  configs/v3_diagnostics.yaml \
  tests/test_v3_rank2_model.py \
  tests/test_v3_residual_data.py \
  tests/test_v3_gate_fista.py \
  tests/test_v3_instrumentation.py \
  results/v3_diagnostics
```

压缩前只检查文件存在，不做 SHA：

```bash
test -f results/v3_diagnostics/DIAGNOSTIC_DECISION.md
test -f results/v3_diagnostics/diagnostic_summary.csv
unzip -l PS_AR_RAPHU_V3_DIAGNOSTICS_RESULTS.zip | tail -n 20
```

---

# 19. 最终执行顺序

```text
1. 从停止线 source 创建独立工作副本
2. 新增 diagnostics 模块和统一配置
3. 运行六个目标测试文件
4. D1：验证 rank-2 容量
5. D2：验证 rank-1 盲区
6. D3：验证 AR 创新信息
7. D4：验证训练顺序/shortcut
8. D5：验证标准化 gate 稀疏路径
9. D6：记录梯度与 proximal 时间线
10. 按冻结规则自动生成 DIAGNOSTIC_DECISION.md
11. 简单打包返还
12. 根据诊断结果另行设计最终 v3，不在本轮自行实现
```
