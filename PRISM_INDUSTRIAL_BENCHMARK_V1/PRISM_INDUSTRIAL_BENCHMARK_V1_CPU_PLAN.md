# PRISM INDUSTRIAL BENCHMARK V1：CPU 实验方案

版本：V1.0  
状态：`PRE-REGISTERED / NO RESULTS`  
建议硬件：32–64 vCPU，128 GB RAM；MetroPT 阶段建议 NVMe 本地盘 500 GB 以上  
数值精度：物理算子、系统辨识、指标和 bootstrap 使用 FP64

---

## 1. CPU 批的职责

CPU 批负责：

1. 冻结五个数据集的原始版本、变量、采样周期、运行边界和 split；
2. 生成 CPU/GPU 共用的不可变共享数据包；
3. 运行经典工业软测量和系统辨识 baseline；
4. 运行 PRISM-Urysohn、PRISM-Urysohn-First 和 PRISM-K-Joint AR；
5. 完成通道专属时间尺度、尺度匹配 AR、exact-zero 和 Urysohn 复杂度审计；
6. 汇总 GPU 逐样本预测并执行最终 FP64 统计。

CPU 批不是 GPU 的预筛附属。PLS、DPLS、ARX、N4SID、Hammerstein 以及 PRISM 两条路线均是最终论文正式方法。

---

## 2. 数据冻结阶段

### 2.1 预检

每个数据集执行：

```bash
python scripts/audit_dataset.py \
  --dataset <name> \
  --raw-root <path> \
  --registry dataset_registry/<name>
```

预检必须输出：

- 文件哈希；
- 时间戳差分分布；
- 重复、缺失、常数段和异常跳变；
- run/profile/month 边界；
- 变量单位和角色 U/X/Y；
- 目标代理变量；
- 标签可用时间；
- 是否允许再分发。

### 2.2 数据集特定规则

#### TEP

- 使用标准 41 个测量变量与操纵量定义；
- 主目标为产品 G 组分；
- 主输入排除产品组分 37–41；
- split 以完整 simulation run 为单位；
- nominal 与 disturbance 分层；
- 未见 disturbance 类型单列 OOD。

#### Debutanizer

- 七个在线过程输入、一个 C4 质量目标；
- 不采用 Kennard–Stone 混合拆分作为主结果；
- 使用严格时间外推；
- 若版本无可靠时间戳，主任务保留样本步定义并在报告中标明物理时间不确定性；
- 另建 60 min 标签延迟敏感性视图。

#### SRU

- H2S 与 SO2 分别建立单目标任务；
- 若有不同 sulfur line，登记 line ID；
- 单 line 主结果使用时间外推，多 line 增加跨 line 外层验证；
- 不允许把两个目标互相作为主输入，除非单列 `MULTI_TARGET_PROXY_ALLOWED`。

#### PMSM

- 以 `profile_id` 整段划分；
- 主输入只含外部、冷却、电气与机械量；
- 内部定子温度只进入 secondary full-sensor 视图；
- profile 不得跨 train/val/test。

#### MetroPT-3

- 由原始时间戳确认 1 Hz/0.1 Hz 描述冲突；
- 主任务分别为 `Reservoirs` 与 `Oil_temperature`；
- `Reservoirs` 主视图排除 `TP3`；
- 故障窗口单列 OOD，不与正常过程主榜混合；
- 按月份连续划分。

---

## 3. 共享数据包

CPU 生成：

```text
shared/
├── PROTOCOL.json
├── DATASET_HASHES.json
├── TASK_REGISTRY.json
├── SPLIT_REGISTRY.json
├── SAMPLE_ID_REGISTRY.json
├── dataset_views/
│   ├── input_only/
│   ├── dynamic/
│   ├── proxy_excluded/
│   └── full_sensor_secondary/
├── sequence_views/
├── multiresolution_tabular_views/
├── graph_views/
├── targets/
├── masks/
├── scaler_metadata/
└── VALIDATION_REPORT.md
```

每个数组记录：dataset、task、split、sample ID、dtype、shape、SHA256、时间支持区间和是否包含历史目标。

GPU 不得重新计算 target、split、scaler、窗口、标签延迟或 profile_id 划分。

---

## 4. CPU baseline

### 4.1 简单基线

- Mean；
- Persistence；
- Seasonal Persistence，仅当训练数据内存在注册周期；
- Local linear trend，作为次级参考。

### 4.2 传统软测量

- Ridge；
- PLS；
- Dynamic PLS；
- RBF-SVR；
- XGBoost。

输入型榜禁止使用历史目标。Dynamic PLS 的滞后、窗口和 latent components 只在 inner folds 选择。

### 4.3 系统辨识

- AR；
- ARX；
- Linear NARX；
- N4SID；
- Parallel Hammerstein；
- Hammerstein-Wiener。

所有 AR/ARX/NARX 都按目标头注册独立的输出步长、历史覆盖和阶数；禁止一个固定 AR 配置跨所有 horizon。

### 4.4 PRISM 模型

1. `PRISM_U_ONLY`；
2. `PRISM_U_FIRST_RESIDUAL_AR`；
3. `PRISM_K_JOINT_AR`；
4. `PRISM_SINGLE_SCALE`；
5. `PRISM_FIXED_MULTIRESOLUTION`；
6. `PRISM_CHANNEL_SPECIFIC`；
7. `AR_FIRST_CONDITIONAL_AUDIT`，仅诊断；
8. `K_COMPLEXITY_ZERO_LINEAR_R1_RRANK_FULL`。

---

## 5. PRISM-Urysohn 的 CPU 实现

### 5.1 时滞表示

每个通道使用因果块平均或注册基函数：

\[
\phi_{j,b}(t)=\frac{1}{|I_{j,b}|}\sum_{\tau\in I_{j,b}}u_j(t-\tau).
\]

近端块更细，远端块可几何扩展。块边界仅由训练集内 profile 注册确定。

### 5.2 复杂度阶梯

对每个通道、每个目标头依次比较：

- exact-zero；
- 线性 distributed lag；
- rank-1 Urysohn；
- 通道独立 rank-R；
- 完整有限 Urysohn。

更复杂模型必须精确包含更简单模型。one-SE 规则优先选择更低复杂度。

### 5.3 通道选择

流程：

1. 每个通道 profile 独立审计；
2. 与尺度匹配 AR 做条件增量诊断；
3. 在 outer-training 内冻结候选通道集合；
4. 对兼容目标头内的候选通道联合重拟合；
5. 冻结物理层；
6. 生成 rolling OOF 物理残差；
7. 训练目标头专属 residual AR，含 exact-zero。

禁止按变量顺序逐层抢残差。

### 5.4 K-Joint AR

同一目标头内联合优化输入响应和 AR 状态项。报告总预测，不将内部 K 作为物理认证结果。

---

## 6. 尺度匹配 AR

对每个 \(\pi^K_{j,r}\) 同时注册 \(\pi^A_{j,r}\)：

- 相同目标；
- 相同 sample ID；
- 相同 outer split；
- 输出历史频带与目标窗口匹配；
- 历史覆盖至少覆盖 2h、4h 或 8h；
- purge 覆盖最大历史和目标成熟时间。

诊断 AR 只用于条件增量。路线 I 的 residual AR 在联合 K 冻结后单独训练；路线 II 的 Joint AR 与 K 联合训练。三者不能混用。

---

## 7. 超参数预算

### 7.1 线性与软测量

- Ridge alpha：对数网格 20 点；
- PLS components：1 至 min(20, p)；
- DPLS latent components：1–20；
- DPLS lag/profile：注册时间网格；
- SVR：最多 30 个配置；
- XGBoost：最多 30 个配置，限制树深与总叶数。

### 7.2 系统辨识

- AR/ARX 阶数按物理时间转换，最多 24 个配置；
- N4SID 状态阶数：2, 4, 6, 8, 12, 16；
- Hammerstein 非线性：线性、二次、三次、分段线性；
- 稳定性和正则单独登记。

### 7.3 数值要求

FP64 用于：

- Gram、Schur、正规方程；
- Urysohn 参数；
- AR/ARX/N4SID 最终拟合；
- KKT 残差与条件数；
- 逐样本预测最终落盘；
- 所有指标与 bootstrap。

---

## 8. CPU 运行阶段

### C0：preflight

验证五个数据集可读取、任务可实现、split 无交叉、目标无未来泄漏。

### C1：共享数据构建

```bash
python scripts/build_shared_dataset.py \
  --config configs/master_protocol.yaml \
  --output shared \
  --package return/PRISM_INDUSTRIAL_SHARED_DATA_V1_bundle.zip
```

### C2：简单基线与经典软测量

运行 Mean、Persistence、Ridge、PLS、DPLS、SVR、XGBoost。

### C3：系统辨识

运行 AR、ARX、NARX、N4SID、Hammerstein 系列。

### C4：PRISM profile audit

运行通道 profile、尺度匹配 AR、exact-zero/linear/rank 阶梯和三类尺度消融。

### C5：PRISM 双路线

完成 Urysohn-First 与 K-Joint AR。

### C6：CPU finalists 与统计

保存所有逐样本预测，执行 block bootstrap、paired difference、Holm correction 和跨任务 rank。

### C7：合并 GPU

读取 GPU 结果包，校验全部 hash 后生成最终报告。

---

## 9. CPU 输出

```text
results_cpu/
├── DATASET_FREEZE_REPORTS/
├── SIMPLE_BASELINES.csv
├── CLASSICAL_SOFT_SENSOR.csv
├── SYSTEM_IDENTIFICATION.csv
├── PRISM_PROFILE_AUDIT.csv
├── PRISM_MODELS.csv
├── PREDICTIONS/
├── KERNELS/
├── AR_PROFILES/
├── NUMERICAL_CERTIFICATES/
├── BOOTSTRAP/
├── RESOURCE_USAGE/
├── CPU_FINAL_REPORT.md
└── CPU_FINAL_DECISION.json
```

逐样本预测字段至少包含：

```text
sample_id,dataset,task,split,model,y_true,y_pred,
information_set,profile_id,seed,dtype,parameter_count
```

---

## 10. 推荐并发

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

建议：

- 32 vCPU：`--n-jobs 20`；
- 64 vCPU：`--n-jobs 40`；
- SVR/N4SID/Hammerstein 重任务并发不超过 8；
- MetroPT 特征构建使用 memory-map / parquet 分块；
- bootstrap 16–32 workers。

---

## 11. CPU 打包

```bash
rm -rf return/PRISM_INDUSTRIAL_CPU_RESULTS_V1
rm -f return/PRISM_INDUSTRIAL_CPU_RESULTS_V1_bundle.zip*

python scripts/build_cpu_bundle.py \
  --results results_cpu \
  --shared-metadata shared \
  --output return/PRISM_INDUSTRIAL_CPU_RESULTS_V1

python scripts/validate_package.py \
  --package-dir return/PRISM_INDUSTRIAL_CPU_RESULTS_V1 \
  --forbid-raw-data \
  --require-manifest \
  --require-sha256

cd return
zip -r PRISM_INDUSTRIAL_CPU_RESULTS_V1_bundle.zip \
  PRISM_INDUSTRIAL_CPU_RESULTS_V1
sha256sum PRISM_INDUSTRIAL_CPU_RESULTS_V1_bundle.zip \
  > PRISM_INDUSTRIAL_CPU_RESULTS_V1_bundle.zip.sha256
python ../scripts/validate_zip_roundtrip.py \
  --zip PRISM_INDUSTRIAL_CPU_RESULTS_V1_bundle.zip
```

最终打印：

```text
FINAL_CPU_ZIP=
FINAL_CPU_SHA256=
SHARED_DATASET_SHA256=
PROTOCOL_HASH=
TASK_REGISTRY_HASH=
SPLIT_REGISTRY_HASH=
SAMPLE_ID_HASH=
VALIDATION_STATUS=PASS
```

---

## 12. CPU 完成条件

- 五个数据集冻结通过；
- 7 个 primary tasks 全部有 CPU baseline；
- 每个目标头有独立 AR 配置；
- 每个 PRISM 通道 profile 有尺度匹配 AR；
- Urysohn-First 与 K-Joint AR 分开报告；
- 逐样本预测、核、AR 配置和数值证书完整；
- raw data 未进入返回包；
- manifest、SHA256、round-trip 全部 PASS。

