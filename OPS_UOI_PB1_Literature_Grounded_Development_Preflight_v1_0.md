# OPS-UOI PB1 文献约束的 Development Preflight v1.0

> 基线分支：`public-benchmark-pb1`  
> 用户报告的基线提交：`c93c8ed`  
> 正式 test 已访问行数：`0`

## 1. 结论

Codex 提议的若干数值不能原样冻结。ARX/NARX 应尽量复现同领域 benchmark 文献；spectral penalty、bootstrap 与 WHPN Huber 在相同数据上没有可直接照抄的标准数字，因此应冻结**选择算法**而非裸数值。

| 项目 | Codex 提议 | 决定 | 文献约束后的协议 |
|---|---|---|---|
| XAR history | 完整 \(L_x\times L_y\) 搜索，同误差取短窗口 | 修改 | baseline 用 ARX-AIC 联合选 lag；主方法保留 native history 搜索，并增加 shared-history 消融 |
| AR/ARX ridge | `[0,1e-8,...,100]` | 拒绝为主基线 | 主 ARX 用稳定 QR/SVD 最小二乘；ridge 仅作 GCV 次级版本 |
| spectral penalty | 两方向 `[1e-4,...,1e-1]`，ridge=`1e-6` | 拒绝 | penalty 归一化后，自动区间 + grouped/blocked development risk |
| NARX | `[32,64]×[2,3]`、lr=`1e-3`、200 epoch、patience 20、3 seeds | 拒绝 | 单隐藏层 tanh、宽度 `[2,5,7,10]`、20k Adam iterations、5 初始化、无 early stopping |
| bootstrap | block 64、100 次 | 拒绝 | 重采样单元按数据结构；block 自动选择；development 250、confirmation 1000 |
| WHPN Huber | 标准化目标下 delta=1.345 | 降级 | 主协议 raw-unshifted + MSE；Huber 只作 sensitivity，delta=1.345×训练 OOF 残差稳健尺度 |

## 2. 公平比较的两层含义

公平不等于所有方法共用一组随手超参数。

1. **协议公平**：相同 split、test 访问次数、输入、标签、scaler 与评价指标。
2. **方法忠实**：baseline 按其原论文或配套实现选择 lag、容量、优化器和停止规则。

因此采用：

\[
\boxed{\text{相同数据协议}+\text{各方法文献忠实训练协议}}
\]

并增加 shared-history / equal-budget 消融。

## 3. ARX 与 NARX：Champneys et al. 2024

主要依据：

- Champneys et al., *Baseline Results for Selected Nonlinear System Identification Benchmarks*, IFAC-PapersOnLine 58(15), 2024, DOI `10.1016/j.ifacol.2024.08.574`。
- 配套代码：`MDCHAMP/nonlinear_baselines`。
- 审计锁定代码提交：`d9c4972f6bd59cfdde23a014bc57579e2af957aa`。

### 3.1 baseline history

论文对所有 AR 类模型使用线性 ARX 在 validation AIC 上联合选择：

\[
(n_x,n_y)=\arg\min_{1\le n_x,n_y\le20}AIC_{val}.
\]

ARX 选择出的 lag 复用于 pNARX、GP-NARX 与 MLP-NARX。

论文自身 split 下报告：Silverbox `(10,10)`、ordinary WH `(8,15)`、Tanks `(9,8)`。这些结果不能硬编码进我们自己的冻结 split；应复现**选择规则**。

机器确定性 tie rule：

1. 更低 AIC；
2. 在 `aic_tie_tolerance` 内，参数更少；
3. 更小 \(n_x+n_y\)；
4. 更小 \(\max(n_x,n_y)\)；
5. `(nx,ny)` 字典序。

该 tie rule 标记为 `PROJECT_DETERMINISM_RULE`。

### 3.2 主方法 history

PS-AR-RAPHU 不应被 baseline 的 20 阶上限绑死。冻结三条 lane：

- `H1 baseline_faithful`：ARX/pNARX/MLP-NARX 使用 ARX-AIC lag。
- `H2 native_model`：PS-AR-RAPHU 对配置中已预注册的 `lx_candidates × ly_candidates` 完整搜索，使用 one-standard-error rule 选择最短复杂度。
- `H3 shared_history_fairness`：把 ARX-AIC lag 固定给 rank-1/fixed-rank2/full/adaptive，用于区分模型结构收益与 history 搜索收益。

复杂度键固定为：

\[
C(L_x,L_y)=(L_x+L_y,L_xL_y,\max(L_x,L_y),L_x,L_y).
\]

## 4. ARX 主基线不使用 ridge 网格

Champneys et al. 的 ARX 使用 QR 分解，没有 ridge。主 baseline 冻结：

```yaml
solver: pivoted_qr_or_svd_minimum_norm
scientific_ridge: 0
```

矩阵秩不足时报告有效秩和条件数，不静默加入 `1e-6`。

可增加 `ARX_RIDGE_SECONDARY`，其 lambda 由训练数据 GCV 选择。GCV 依据 Golub–Heath–Wahba 1979，不使用固定尺度相关网格。

## 5. MLP-NARX 文献协议

### 5.1 peer-reviewed profile

Champneys et al. 明确采用：

- 一个隐藏层；
- `tanh`；
- width `[2,5,7,10]`；
- 20,000 次 Adam iteration；
- 5 个随机初始化；
- validation 选择总体最佳网络；
- **无 early stopping**。

配套代码显示 MLP 使用训练拟合的 `MinMaxScaler(-1,1)`，Adam learning rate 为 `1e-2`。

主 profile：

```yaml
id: MLPNARX_CHAMPNEYS2024
hidden_layers: 1
activation: tanh
widths: [2, 5, 7, 10]
optimizer: Adam
learning_rate: 1e-2
iterations: 20000
early_stopping: false
initializations: 5
history: ARX_AIC_SELECTED
scaling: train_only_minmax_minus1_plus1
selection: validation_AIC
```

### 5.2 论文与当前代码不完全一致

锁定代码提交使用：

- widths `[2,4,8,16,32]`；
- 10 初始化；
- Adam `1e-2`；
- 无 early stopping；
- ParWH 10k iterations；
- Tanks 20k iterations。

因此主结果采用 peer-reviewed profile；可选审计 lane `MLPNARX_CODE_D9C4972` 单独分栏。

### 5.3 数据集适用边界

- Tanks：same-benchmark published baseline。
- PWH：配套仓库包含 ParWH runner，但正式论文表格未报告 ParWH，称为 associated-code profile。
- WHPN：Champneys profile 只能称 transferred baseline。

## 6. WHPN 增加 same-data 前沿 baseline

Weber & Gühmann 2021 在 WHPN 上给出 GRU/TCN 的同数据比较。其 WHPN GRU-NAR 最优配置：

```yaml
neurons_per_layer: 200
hidden_layers: 2
batch_size: 32
window_length: 9000
TBPTT_sequence_length: 500
```

训练流程包括 MSE、标准化、RAdam+Lookahead、cosine annealing、LR finder 与 ASHA。

因此最终 WHPN 表中应加入 `GRU_NAR_WEBER2021_WHPN`。PB1 第一轮 formal fit 不必因此阻塞，但投稿前必须补齐或说明无法复现。ASHA resource budget、LR finder 规则和实现 commit 必须在真正运行前再次锁定；不能用 `lr=1e-3, patience=20` 替换后仍称为复现。

## 7. spectral penalty：冻结选择算法，不冻结裸 lambda

二维 spectral penalty 是新方法，裸 lambda 依赖 basis scaling、样本数、单位、loss normalization 与 Gram 白化。没有相同 benchmark 文献能合法提供固定 `[1e-4,...]`。

### 7.1 penalty 归一化

对每个 sieve 计算：

\[
H=\Phi^\top\Phi/n,\qquad P_\tau,P_x,I.
\]

每个非零 penalty 按相对 \(H\) 的正 generalized eigenvalue 中位数归一化：

\[
\bar P_q=\frac{P_q}{\operatorname{median}^+\{\lambda(P_q,H)\}}.
\]

若不可计算，fallback 为正 trace/rank，并记录。

### 7.2 自动区间

依据 Eilers–Marx P-spline 与 Li–Cao 自动区间路线：

1. 自动生成每个 smoothing parameter 的可解且足够宽的 log interval；
2. 在 development 上做 grouped/blocked risk search；
3. 最优点在边界时扩展区间；
4. 两次扩展后仍在边界，输出 `PENALTY_INTERVAL_NOT_CERTIFIED`，禁止 confirmation。

development risk：

- PWH：phase-grouped，五 amplitude 与双 period 不拆；
- WHPN：完整 realization grouped；
- Tanks：训练 80% 内 forward blocked folds。

one-standard-error tie rule：更低 effective degrees of freedom、更强 structured smoothing、更低 isotropic ridge、固定字典序。

### 7.3 numerical jitter 与 scientific ridge 分离

\[
\epsilon_{num}=c_{num}\epsilon_{mach}\max(1,\|H+P\|_2).
\]

numerical jitter 必须单独记录、不参与模型选择、不称为 ridge。科学 ridge \(\lambda_0\) 必须由 development 选择，不能固定 `1e-6`。

## 8. Bootstrap

固定 block 64 不成立，因为最优 block 依赖样本量、相关强度与统计量。经典结果对不同任务甚至给出 \(n^{1/3},n^{1/4},n^{1/5}\) 的不同阶。

### 8.1 重采样单元

- PWH：phase cluster，包含全部五 amplitude 和双 period。
- WHPN：完整 realization；必要时嵌套自动 block bootstrap。
- Tanks：moving/stationary block bootstrap，block length 用 Politis–White 自动选择；Bühlmann–Künsch 作敏感性。

### 8.2 replicate 数

对 95% 双侧区间 2.5% 尾分位，Monte Carlo 标准误差近似：

\[
SE_{MC}=\sqrt{0.025(1-0.025)/B}.
\]

- development 要求 \(SE_{MC}\le0.01\)：`B=250`；
- confirmation 要求 \(SE_{MC}\le0.005\)：`B=1000`。

## 9. WHPN Huber 不阻塞主拟合

WHPN 的标志性困难是 process noise 在静态非线性之前进入，而不是少量独立传感器离群点。文献指出这种情况下普通 prediction-error 方法可能偏差，专门 maximum-likelihood/stochastic WH 更匹配。

主协议：

```yaml
loss: MSE
alignment: raw_unshifted
process_noise_preserved: true
outlier_filter: none
```

Huber 仅作 sensitivity：

\[
\widehat s=1.4826\operatorname{median}|r-\operatorname{median}(r)|,
\qquad
\delta=1.345\widehat s,
\]

其中残差来自 training OOF MSE。`1.345` 只是标准正态下约 95% 渐近效率的传统常数，不是 WHPN 特有最优阈值。

Huber 不参与主模型选择、主 gate 或 test retuning。

最终 WHPN 更应加入 process-noise-aware ML/EIV comparator 与 Weber 2021 GRU-NAR。

## 10. Machine-checkable preflight

必须通过：

```text
LITERATURE_PROFILE_PINNED
ARX_LAG_SELECTOR_FROZEN
ARX_PRIMARY_SOLVER_FROZEN
MLPNARX_PROFILE_FROZEN
NATIVE_HISTORY_RULE_FROZEN
SHARED_HISTORY_ABLATION_FROZEN
PENALTY_NORMALIZATION_FROZEN
PENALTY_INTERVAL_ALGORITHM_FROZEN
PENALTY_BOUNDARY_POLICY_FROZEN
BOOTSTRAP_UNIT_FROZEN
BOOTSTRAP_BLOCK_SELECTOR_FROZEN
BOOTSTRAP_MC_PRECISION_FROZEN
WHPN_PRIMARY_LOSS_FROZEN
WHPN_HUBER_ROLE_FROZEN
OFFICIAL_TEST_ACCESS_COUNT_ZERO
```

硬失败：

```text
EXTERNAL_BASELINE_COMMIT_UNPINNED
RAW_LAMBDA_GRID_WITHOUT_SCALE_CERTIFICATE
VALIDATION_WINDOW_SPLIT_LEAKAGE
PENALTY_OPTIMUM_ON_UNRESOLVED_BOUNDARY
BOOTSTRAP_BLOCK_LENGTH_HARDCODED_ACROSS_DATASETS
HUBER_USED_TO_CLEAN_WHPN_PRIMARY
EARLY_STOPPING_ADDED_TO_CHAMPNEYS_MLPNARX_PROFILE
```

## 11. 启动判断

立即允许启动：persistence、QR/SVD ARX、pNARX、MLP-NARX Champneys profile、rank-1、fixed rank-2、full spectral、adaptive spectral。

次级：ARX-ridge GCV、MLP-NARX code profile、WHPN Huber sensitivity。

投稿前补充：WHPN GRU-NAR Weber 2021、process-noise-aware structured comparator、许可证解决后的 Silverbox。

当前真正阻塞正式拟合的只剩：

1. spectral penalty 自动区间与边界检查实现；
2. 文献 profile/外部代码 commit 的机器 pinning。

WHPN Huber 不应再阻塞主线。

## 12. 参考文献

1. Champneys et al. 2024, DOI `10.1016/j.ifacol.2024.08.574`。
2. `MDCHAMP/nonlinear_baselines`, commit `d9c4972f6bd59cfdde23a014bc57579e2af957aa`。
3. Weber & Gühmann 2021, DOI `10.1016/j.ifacol.2021.11.252`。
4. Golub, Heath & Wahba 1979, DOI `10.1080/00401706.1979.10489751`。
5. Eilers & Marx 1996, DOI `10.1214/ss/1038425655`。
6. Li & Cao 2023, DOI `10.1007/s11222-022-10178-z`。
7. Bühlmann & Künsch 1999, DOI `10.1016/S0167-9473(99)00014-6`。
8. Politis & White 2004, DOI `10.1081/ETC-120028836`。
9. Hall, Horowitz & Jing 1995, DOI `10.1093/biomet/82.3.561`。
10. Huber 1964, DOI `10.1214/aoms/1177703732`。
11. Giordano & Sjöberg 2018, DOI `10.1016/j.ifacol.2018.09.178`。
