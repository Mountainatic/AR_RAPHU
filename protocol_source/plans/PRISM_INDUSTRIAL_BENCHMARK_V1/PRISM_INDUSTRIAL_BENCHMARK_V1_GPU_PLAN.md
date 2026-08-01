# PRISM INDUSTRIAL BENCHMARK V1：GPU 实验方案

版本：V1.0  
状态：`PRE-REGISTERED / NO RESULTS`  
建议硬件：RTX 5090 32 GB 可完成 smoke 与单卡正式运行；完整 V1 建议 2–4 张 24 GB 以上 GPU 并行  
训练精度：screening 使用 BF16 AMP + FP32 master；finalists 使用 FP32，TF32 关闭  
指标精度：FP64 汇总

---

## 1. GPU 批的职责

GPU 批只负责：

1. 公认深度 baseline；
2. 有潜力的现代长序列 baseline；
3. AKGNN 与 T-AKGNN 的明确适配对照；
4. primary tasks screening；
5. 多尺度核心比较；
6. finalists 多种子确认、资源统计与鲁棒性消融。

GPU 不得：

- 重新下载或清洗数据；
- 重新切 split；
- 重新生成 target；
- 改变 sample ID；
- 重新估计 CPU 冻结的 scaler/profile/PCA；
- 使用未来输入、未来目标或未成熟残差；
- 根据 test 结果删除模型或修改 horizon。

---

## 2. 启动前共享包校验

必须读取：

```text
PRISM_INDUSTRIAL_SHARED_DATA_V1_bundle.zip
PRISM_INDUSTRIAL_SHARED_DATA_V1_bundle.zip.sha256
```

启动时检查：

- protocol hash；
- dataset hash；
- task registry hash；
- split registry hash；
- sample ID hash；
- scaler metadata hash；
- 每个 view 的 shape/dtype/hash；
- proxy-excluded 与 full-sensor 标记；
- input-only 与 dynamic 信息集标记。

任一不一致立即退出。

---

## 3. GPU baseline

### 3.1 核心公认模型

- MLP；
- LSTM；
- GRU；
- TCN；
- DLinear；
- NLinear；
- Causal Transformer Encoder；
- PatchTST。

### 3.2 有潜力的现代模型

- iTransformer；
- ModernTCN；
- TimeMixer；
- TimesNet；
- S4D；
- Temporal Autoencoder。

### 3.3 图与 KAN 模型

- `AKGNN_WINDOW_ADAPTED`；
- `T_AKGNN_SEQUENCE_ADAPTED`。

AKGNN 静态窗口版使用冻结的 multiresolution summary；T-AKGNN 使用因果序列编码。二者均必须报告适配部分，不得声称为原始论文任务的完全复现。

---

## 4. 两张 GPU 排行榜

### 4.1 Input-only

\[
\widehat y_{t+h}=f(U^-,X^-).
\]

禁止历史目标。模型后缀统一为 `-U`。

### 4.2 Dynamic

\[
\widehat y_{t+h}=f(U^-,X^-,Y^-_{available}).
\]

只允许严格过去且当时可获得的目标历史。模型后缀统一为 `-UXY`。

每个模型、task、seed 必须保存两榜信息集标签。禁止同一 checkpoint 同时冒充两榜。

---

## 5. 输入视图

### 5.1 Unified causal sequence

所有模型共享 CPU 生成的因果序列：

```text
X: [batch, time, channels]
y: [batch, target_dim]
sample_id: [batch]
```

### 5.2 Fixed multiresolution

为公平对比 PRISM 的通道专属尺度，神经网络获得一套对所有通道相同的固定多分辨率金字塔。该视图不能根据通道单独重采样。

### 5.3 Channel-specific neural ablation

只在多尺度核心阶段，对 NLinear、TCN、PatchTST、ModernTCN/TimeMixer 增加通道独立分辨率输入。必须标记 `CHANNEL_SPECIFIC_INPUT_ADAPTED`，与原模型主结果分开。

### 5.4 Proxy views

主榜使用 proxy-excluded view。full-sensor 只进入 secondary sensitivity，不能替代主结果。

---

## 6. 模型配置原则

### 6.1 参数量

主榜：

\[
N_{param}\le 250,000.
\]

建议容量：

- Small：20k–80k；
- Medium：80k–250k。

超过 250k 的官方结构单列 `LARGE_REPRODUCTION`，不参与小模型效率排名。

### 6.2 因果性

- LSTM/GRU/TCN 只读过去；
- Transformer 使用 causal mask；
- PatchTST patch 不得覆盖预测原点之后；
- iTransformer 只对过去窗口变量 token 化；
- TimesNet/TimeMixer 不得读取 future covariates，除非任务明确注册且所有模型同享；
- 双向 RNN 不进入正式在线榜。

### 6.3 输出头

每个 horizon 是独立目标头。允许共享 backbone 的多头版本只能作为 secondary efficiency experiment；主榜使用单目标头独立训练，防止跨 horizon 信息和预算不公平。

---

## 7. 调参预算与种子

### 7.1 Stage G1：primary screening

- 7 个 primary tasks；
- 全部 16 个 GPU baseline；
- 每模型每任务最多 24 trial（核心）或 16 trial（现代扩展）；
- 3 screening seeds；
- inner validation 只使用 outer-training；
- 最多 150 epochs；
- patience 15。

### 7.2 Stage G2：core confirmation

所有核心 baseline 和按平均 validation rank 进入前 8 的现代模型：

- 5 seeds；
- 最大 200 epochs；
- patience 20；
- 保存 seed median ensemble。

### 7.3 Stage G3：finalists

按 7 个 primary tasks 的平均 validation rank 选择前 6 名 GPU 模型：

- 10 seeds；
- FP32；
- TF32=false；
- 完整 test；
- 逐样本预测；
- data-fraction、missing-channel、noise 和 proxy ablation。

模型不能因为某一数据集失败而被从该数据集结果中删除。

---

## 8. 多尺度核心比较

不是所有深度模型都运行全部 horizon。完整 scale sweep 固定比较：

- NLinear；
- TCN；
- PatchTST；
- iTransformer 与 ModernTCN 中平均 validation rank 更高者；
- PRISM-Urysohn；
- PRISM-Urysohn-First；
- PRISM-K-Joint AR；
- CPU DPLS 与 ARX。

GPU 负责前四项；CPU 提供其余项预测。

对每个模型报告：

1. short/mid/long 三个目标头；
2. fixed single-scale；
3. fixed multiresolution；
4. channel-specific input adaptation；
5. 性能随 horizon 的变化；
6. 参数量和推理代价随 horizon 的变化。

该阶段是检验 PRISM 核心主张的主实验，不以单个最佳 horizon 代替完整结果。

---

## 9. 精度与稳定性

### 9.1 Screening

```yaml
training_dtype: bf16_amp
master_weights: fp32
loss_dtype: fp32
metric_dtype: fp64
tf32: true
```

自动回退条件：NaN、Inf、梯度溢出、validation 震荡异常、与 FP32 smoke 差异超阈值。

### 9.2 Finalists

```yaml
training_dtype: fp32
metric_dtype: fp64
tf32: false
```

记录：

- GPU 型号；
- driver、CUDA、cuDNN、PyTorch；
- seed；
- deterministic flags；
- AMP/TF32；
- batch size；
- parameter count；
- peak VRAM；
- train seconds；
- inference latency。

### 9.3 训练保护

必须实现：

- gradient clipping；
- LR warmup；
- cosine 或 plateau scheduler；
- early stopping；
- NaN fail-fast；
- batch-size fallback；
- AMP→FP32 fallback；
- checkpoint/resume；
- 每 task/model/seed 独立状态；
- 单任务失败不终止全局。

---

## 10. 推荐 batch 与并发

初始 batch：

- MLP/DLinear/NLinear：512；
- LSTM/GRU/TCN：128–256；
- Transformer/PatchTST/iTransformer：64–128；
- ModernTCN/TimeMixer/TimesNet/S4D：64；
- AKGNN/T-AKGNN：32–64。

单 RTX 5090：

- 一个大模型训练 worker；
- 轻量 DLinear/NLinear/MLP 最多 2 个并发；
- dataloader workers 8；
- Optuna 默认串行，避免显存碎片；
- CPU 后台只做指标和压缩。

2–4 GPU：按 dataset/task 分片，不按 seed 共享同一 checkpoint。

---

## 11. GPU 运行阶段

### G0：共享包 smoke

验证所有 task shape、mask、因果性和 sample ID。

### G1：核心模型 smoke

每个模型选择一个小任务、1 seed、2 epochs，验证前向、反向、保存、恢复和无泄漏测试。

### G2：primary screening

运行全部 16 个模型、7 个 primary tasks、3 seeds。

### G3：core confirmation

核心 baseline + validation 前 8，5 seeds。

### G4：多尺度核心比较

运行预注册 short/mid/long heads 和 channel-specific input ablation。

### G5：finalists

前 6 名，10 seeds，FP32。

### G6：鲁棒性

- 25%/50%/100% data；
- Gaussian sensor noise；
- one-channel missing；
- contiguous missing blocks；
- proxy-excluded/full-sensor；
- nominal/OOD；
- inference benchmark。

### G7：打包

保存全部逐样本预测和资源信息，不包含原始数据和无用 checkpoint。

---

## 12. 逐样本输出

每个 task/model/seed：

```text
sample_id
dataset
task
split
model
information_set
seed
y_true
y_pred
training_dtype
parameter_count
train_seconds
infer_ms_per_1000
peak_vram_mb
checkpoint_hash
```

主报告使用预注册 seed median ensemble。bootstrap 由 CPU 统一执行，不能把 10 个 seed 当作 10 倍独立时间样本。

---

## 13. GPU 代码结构

```text
PRISM_INDUSTRIAL_BENCHMARK_V1/
├── configs/
│   ├── gpu_core.yaml
│   ├── gpu_frontier.yaml
│   ├── gpu_finalists.yaml
│   └── task_registry.yaml
├── src/
│   ├── data_views/
│   ├── neural_core/
│   ├── long_sequence/
│   ├── graph_kan/
│   ├── training/
│   ├── evaluation/
│   └── packaging/
├── scripts/
│   ├── validate_shared.py
│   ├── run_gpu_smoke.py
│   ├── run_gpu_primary.py
│   ├── run_gpu_confirm.py
│   ├── run_gpu_multiscale.py
│   ├── run_gpu_finalists.py
│   ├── aggregate_gpu.py
│   └── build_gpu_bundle.py
├── tests/
├── RUN_GPU.sh
└── RESUME_GPU.sh
```

---

## 14. GPU 打包

```bash
rm -rf return/PRISM_INDUSTRIAL_GPU_RESULTS_V1
rm -f return/PRISM_INDUSTRIAL_GPU_RESULTS_V1_bundle.zip*

python scripts/build_gpu_bundle.py \
  --results results_gpu \
  --output return/PRISM_INDUSTRIAL_GPU_RESULTS_V1 \
  --keep-best-checkpoints-only

python scripts/validate_package.py \
  --package-dir return/PRISM_INDUSTRIAL_GPU_RESULTS_V1 \
  --forbid-raw-data \
  --require-manifest \
  --require-predictions \
  --require-resource-usage

cd return
zip -r PRISM_INDUSTRIAL_GPU_RESULTS_V1_bundle.zip \
  PRISM_INDUSTRIAL_GPU_RESULTS_V1
sha256sum PRISM_INDUSTRIAL_GPU_RESULTS_V1_bundle.zip \
  > PRISM_INDUSTRIAL_GPU_RESULTS_V1_bundle.zip.sha256
python ../scripts/validate_zip_roundtrip.py \
  --zip PRISM_INDUSTRIAL_GPU_RESULTS_V1_bundle.zip
```

最终打印：

```text
FINAL_GPU_ZIP=
FINAL_GPU_SHA256=
SHARED_DATASET_SHA256=
PROTOCOL_HASH=
TASK_REGISTRY_HASH=
SPLIT_REGISTRY_HASH=
SAMPLE_ID_HASH=
VALIDATION_STATUS=PASS
```

---

## 15. GPU 完成条件

- 16 个 baseline 在 7 个 primary tasks 上均有成功或明确失败记录；
- 核心模型 5 seeds；
- 前 6 名 10 seeds、FP32；
- 多尺度核心比较完整；
- input-only 与 dynamic 完全分离；
- 逐样本预测与资源记录完整；
- sample IDs 与 CPU 完全一致；
- 原始数据、缓存和无用 checkpoint 未进入返回包；
- manifest、SHA256、round-trip 全部 PASS。

