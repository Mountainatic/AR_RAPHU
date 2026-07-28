# OPS-UOI v4.1 定理—实验逐项合同
## operator-first 识别与 simulation closure 的逐项可失败合同

> 本文不再使用“整体 RMSE 好，所以理论成立”的验证方式。  
> 每个理论条件、结论和失效边界都有独立实验、停止线和允许声明。

---

# 0. 通用输出合同

每项实验必须输出：

```text
config_resolved.yaml
data_lineage.json
split_manifest.csv
metrics_per_seed.csv
aggregate_metrics.csv
decision.json
failure_modes.json
environment.json
manifest.json
SHA256SUMS.txt
```

`decision.json` 至少包含：

```json
{
  "stage": "E2B-ID-SPACE",
  "primary_finding": "...",
  "all_preregistered_gates_pass": true,
  "allowed_next_stage": "...",
  "claims_allowed": [],
  "claims_forbidden": []
}
```

禁止：

- test 参与参数选择；
- 静默裁剪；
- 看到失败后修改 gate 并复用同一 confirmation seed；
- 只保留成功 seed；
- 用 ridge 后条件数替代未正则 Gram/Schur 证书。

---

# 1. E2B-ID-SPACE：多变量 K 层联合可辨识性

## 目标

验证在独立空间填充激励下：

\[
\Gamma_m\succ0,
\qquad
S_{j,m}\succ0,
\]

并且多变量同时估计不会造成 kernel crosstalk。

## DGP

- \(p=3\) active variables；
- 可增加 3 个 inactive variables；
- 各变量输入独立 SPACE；
- 核族包含 rank-1、weak rank-2、strong rank-2、higher-rank；
- 无 AR 或弱 AR，先隔离多变量辨识问题。

## 必须报告

\[
\lambda_{\min}(\Gamma_m),
\quad
\operatorname{cond}(\Gamma_m),
\]

\[
\lambda_{\min}(S_{j,m}),
\quad
\operatorname{cond}(S_{j,m}),
\]

\[
E_{K,j}^{HS}
=
\frac{\|\widehat K_j-K_j^0\|_{HS}}
{\|K_j^0\|_{HS}},
\]

\[
E_{\mathrm{cross},j}
=
\frac{
\sum_{k\ne j}
|\langle\widehat K_j-K_j^0,K_k^0\rangle|
}{
\|\widehat K_j-K_j^0\|
\sum_{k\ne j}\|K_k^0\|+\epsilon
}.
\]

## 主 gate

- 至少 4/5 seeds；
- 每个 active block 的 Schur 最小特征值高于预注册数值分辨率阈值；
- full contribution capacity 通过；
- HS kernel error 达到预注册预算；
- inactive contribution 接近零；
- rank profile 复原。

## 失败解释

若总贡献好但 Schur 失败：

```text
PREDICTIVE_CAPACITY_PASS
VARIABLE_ATTRIBUTION_NOT_IDENTIFIED
```

不允许将其写成优化失败。

---

# 2. E2B-NAT-Q：自然相关输入下的 quotient 预测

## 目标

验证自然闭环相关输入下，即使 per-variable K 层不可辨识，总外生贡献仍可预测。

## 允许指标

- 总贡献 NRMSE；
- XAR 相对 AR 的增量；
- quotient/empirical operator error；
- predictive SVD rank；
- Gram/Schur 作为可辨识诊断。

## 禁止 gate

不得要求所有变量的 HS kernel 都恢复。若 Schur 低，完整核恢复不是合法目标。

## 决策分类

1. `K_LEVEL_IDENTIFIED`；
2. `QUOTIENT_ONLY_IDENTIFIED`；
3. `NO_RESIDUAL_EXCITATION`；
4. `MODEL_CAPACITY_FAIL`。

---

# 3. E2B-PERM-DIAG：分布诊断而非可辨识排序

## 目标

诊断时间相关和变量相关如何改变冻结核的 predictive SVD rank。

## 两种协议必须分开

### Frozen residualizer

冻结 NAT 的 \(\widehat\pi_{\mathrm{NAT}}\)，只改变评价输入分布。此时可以比较固定映射下的贡献范数。

### Refit residualizer

在 PERM 上重新估计 \(\pi_{\mathrm{PERM}}\)。此时 estimand 已变化，只作为新环境诊断。

报告中不得把二者混在一起。

---

# 4. E3-ORTH-AR：强 AR 下的双残差验证

## DGP

AR 强度：

\[
\rho_y\in\{0,0.5,0.9,0.98\}.
\]

外生独立激励强度：

\[
\kappa\in\{\text{strong},\text{medium},\text{weak},0\}.
\]

## 比较

1. naive simultaneous；
2. 只 residualize \(Y\)；
3. 双 residualize \(Y\) 与设计；
4. oracle nuisance；
5. AR-only。

## 指标

\[
\|\widehat g-g_0\|_{L^2},
\]

\[
\|\widehat K-K_0\|_{HS}
\quad\text{仅在 K 层场景},
\]

\[
r_\mu,\quad r_\pi,
\]

\[
r_\mu r_\pi+Rr_\pi^2,
\]

\[
\widehat\kappa_m,\quad
\widehat S_j.
\]

## 预期结论

- 双残差在 nuisance 改善时偏差按二阶下降；
- \(\kappa\to0\) 时 kernel error 必然放大；
- \(\kappa=0\) 时应自动输出不可辨识，而不是伪恢复。

---

# 5. E-CF-GAP：forward gap 与泄漏审计

## 原始支持

每个 origin 使用：

\[
[t-L_\star,t+h].
\]

测试：

\[
G_n\in
\{
L_\star+h,
L_\star+h+\lceil0.5\log N\rceil,
L_\star+h+\lceil\log N\rceil,
L_\star+h+\lceil2\log N\rceil
\}.
\]

## 检查

- train/eval primitive index 是否重叠；
- nuisance OOF error；
- kernel/contribution bias；
- nominal interval/coverage；
- coupling proxy 随 gap 稳定。

零额外 gap 只能用于展示泄漏风险，不能进入 confirmation。

---

# 6. E-SCALE-MIX：样本量和 mixing 速率

## 网格

\[
N\in\{2000,4000,8000,16000,32000\}.
\]

mixing 强度由 AR/Markov 参数控制，至少三档。

## 输出

绘制：

\[
\log
\|\widehat g-g_0\|
\quad\text{vs}\quad
\log(N/\log^2N).
\]

理论参考斜率：

\[
-\frac{s}{2s+1}.
\]

同时报告：

- \(d_m\)；
- \(\kappa_m\)；
- nuisance rate；
- raw error；
- 去除 nuisance 和 ill-conditioning 后的 normalized error。

不得仅在一个 \(N\) 上宣称收敛率。

---

# 7. E-LEPSKI-MASK：隐藏真值的分辨率选择

## 协议

选择阶段完全隐藏真核。

候选：

\[
M_x\in\{12,16,20,24,28,32,40\},
\]

lag 先保持 identity，避免二维偏序。

每个候选：

- 内部选择正则；
- 用独立 selection block 估计差异范数；
- 生成同时半径；
- 应用 Lepski。

选完后才揭示真值计算 oracle index。

## 指标

\[
\frac{
\|\widehat g_{\widehat m}-g_0\|
}{
\min_m\|\widehat g_m-g_0\|+\epsilon
},
\]

\[
|\widehat m-m_{\mathrm{oracle}}|,
\]

coverage：

\[
\Pr(
\|\widehat g_m-g_m^\star\|\le\rho_m,\forall m
).
\]

## gate

- oracle risk ratio 的中位数和 90% 分位数；
- 选择不系统卡在最大 grid；
- 两次相邻 refinement 稳定；
- coverage 达到预注册水平。

---

# 8. E-RANK-MARGIN：rank margin 与弱信号

## 场景

1. 大 margin；
2. 小 margin；
3. 无 margin；
4. near-zero operator；
5. inactive operator。

## 必须输出

\[
\delta_U,
\quad
\|\widehat K\|_{HS},
\quad
\eta_U=
\min(1,2\delta_U/\|\widehat K\|),
\]

\[
[R_L,R_U].
\]

## 正确决策

- 大 margin：单值 rank；
- 小/no margin：rank interval；
- \(\|\widehat K\|\le2\delta_U\)：`WEAK_OPERATOR_OR_INACTIVE`；
- 不允许给 inactive variable 强制 rank 1。

## 覆盖 gate

\[
\Pr(R_L\le R^\star\le R_U)
\]

必须达到 nominal 水平。

---

# 9. E-MISSPEC-INTERACTION：加性错设

## DGP

加入真实交互：

\[
Y_t
=
\sum_j\mathcal A_jK_j
+
\gamma_{12}
H(W_{1,t},W_{2,t})
+
\varepsilon_t.
\]

改变 \(\gamma_{12}\)。

## 目标

验证 OPS-UOI 收敛到最佳加性投影 \(g_0\)，而不是错误声称恢复完整真实机制。

## 输出

- 最佳加性 oracle；
- OPS-UOI 与 additive oracle 的差；
- 相对完整 truth 的不可约 misspecification error；
- residual interaction test。

正确结论：

```text
BEST_ADDITIVE_PROJECTION_RECOVERED
FULL_MECHANISM_MISSPECIFIED
```

---

# 10. E-Q-vs-K：贡献可辨识和核不可辨识反例

构造两个不同核：

\[
K_1\ne K_2,
\qquad
\mathcal DK_1=\mathcal DK_2.
\]

验证：

- 预测贡献完全一致；
- ridge/minimum-norm 代表不同；
- HS rank 可不同；
- 数据无法选择“真实结构 rank”。

该实验必须进入论文，用来证明我们理解自身边界。

---

# 11. E-BOOT-RADIUS：block/bootstrap 半径

## 目标

验证用于 Lepski 和 rank interval 的 \(\delta_U,\rho_m\) 覆盖率。

## 变化因素

- block length；
- mixing 强度；
- 样本量；
- nuisance 是否重估；
- fixed-smoothing 与 full-pipeline bootstrap。

## 至少比较

1. moving block bootstrap；
2. stationary bootstrap；
3. theorem plug-in conservative radius。

若 bootstrap coverage 不足，confirmation 必须使用保守 theorem radius 或扩大系数。

---

# 12. REAL-CZ-OUTER：多晶棒真实验证

## 数据划分

必须按晶棒或批次做 outer split：

\[
\text{train rods}
\rightarrow
\text{unseen validation rod}
\rightarrow
\text{unseen test rod}.
\]

单根棒内窗口不得跨 outer split。

## 三条预测轨道

- AR-only；
- X-only；
- XAR。

## 基线

传统：

- persistence；
- AR/ARX/ARMAX；
- NARX；
- parallel Hammerstein；
- regularized Urysohn。

现代：

- GRU/LSTM；
- TCN；
- Transformer；
- KAN/GADKN-inspired baseline。

## 结构输出

真实 NAT 默认只报告：

- quotient contribution；
- predictive SVD rank；
- Schur/Gram 可辨识诊断；
- fold/rod stability。

只有 K 层证书通过时才报告完整 kernel 和 structural rank。

---

# 13. 测量质量层

高置信传感器跳码可在模型前修复，但必须保留：

```text
raw_value
model_value
quality_flag
```

真实同阶段物理尖峰不得按幅值自动过滤。

正式实验比较：

1. raw + standard loss；
2. high-confidence measurement repair + standard loss；
3. repair + robust sensitivity fit。

robust loss 是剩余污染敏感性分析，不承担物理事件辨识。

---

# 14. 最终论文声明矩阵

| 实验结果 | 允许声明 |
|---|---|
| Q 层通过、K 层失败 | 条件贡献可预测；完整核/结构 rank 不可辨识 |
| K 层通过、rank margin 通过 | 完整核与单值结构 rank |
| K 层通过、无 margin | 完整核 + rank interval |
| predictive SVD rank 稳定 | 当前输入分布下谱截断复杂度 |
| PERM rank 改变 | 分布改变会改变预测所需谱模态 |
| 多棒 outer test 通过 | batch-to-batch 泛化 |
| 单棒时间切分通过 | within-run temporal generalization |
| interaction stress 失败 | 加性模型边界，不得扩展物理结论 |

---

# 15. 推荐执行顺序

\[
\boxed{
\text{E2B-ID-SPACE}
\rightarrow
\text{E2B-NAT-Q}
\rightarrow
\text{E3-ORTH-AR}
}
\]

\[
\boxed{
\rightarrow
\text{E-CF-GAP}
\rightarrow
\text{E-SCALE-MIX}
\rightarrow
\text{E-LEPSKI-MASK}
}
\]

\[
\boxed{
\rightarrow
\text{E-RANK-MARGIN}
\rightarrow
\text{E-BOOT-RADIUS}
\rightarrow
\text{REAL-CZ-OUTER}.
}
\]

任何阶段发生 `NOT_IDENTIFIED` 时，都应停止对应结构声明，但不必停止预测轨道。

---

# 16. E-SIM-SUPPORT：三域与首次越界审计

## 目标

验证：

\[
\mathcal S^{\mathrm{cert}}
\subseteq
\mathcal S^{\mathrm{train}}
\subseteq
\mathcal D^{\mathrm{model}}
\]

在代码、日志和报告中被逐通道实现。

## 必须输出

```text
support_domains.json
support_mask_per_time.csv
first_crossing_audit.json
continuation_segments.csv
```

`first_crossing_audit.json` 对每次首次越界至少包含：

```json
{
  "dataset": "...",
  "trajectory_id": "...",
  "channel": "x_j or recursive_y",
  "time_index": 0,
  "train_lower": 0.0,
  "train_upper": 0.0,
  "value": 0.0,
  "normalized_distance": 0.0,
  "teacher_forced_value": 0.0,
  "free_run_value": 0.0,
  "external_contribution": 0.0,
  "ar_contribution": 0.0
}
```

## gate

- 域由 train-only 数据生成；
- validation/test 不修改 knots/scaler/domain；
- 外生 X 越界与 recursive y 越界分开；
- 域外访问不静默裁剪；
- official test 前所有规则冻结。

---

# 17. E-CONT-C1：continuation 数学合同

## 比较

```text
A0_HARD_FAIL
A1_BOUNDARY_CLIP
A2_LINEAR_C1
A3_BOUNDED_C1_TANH
```

主方法是 A3；A0–A2 只作消融。

## 数学检查

在每个边界和每个核/响应曲线上检查：

\[
|\widetilde f(a^-)-\widetilde f(a^+)|\le10^{-10},
\]

\[
|\widetilde f'(a^-)-\widetilde f'(a^+)|\le10^{-8},
\]

右边界同理。

检查大幅域外输入时：

```text
all_outputs_finite = true
all_derivatives_finite = true
bounded_value_certificate = true
```

## 域内等价 gate

相同 fitted coefficients 下，在所有 train-domain nodes：

\[
\max|\widetilde f(u)-f(u)|\le10^{-12}.
\]

训练设计矩阵和 fitted values 的最大差：

\[
\le10^{-12}.
\]

若不通过：

```text
CONTINUATION_CHANGED_IN_SUPPORT_MODEL
```

硬失败。

---

# 18. E-FR-CLOSURE：free-run 全局闭合

## 数据集

- PWH；
- WHPN；
- Cascaded Tanks；
- Silverbox。

## 每套数据必须报告

```text
direct_metrics.json
free_run_metrics.json
simulation_bound.json
continuation_usage.json
initial_condition_sensitivity.json
local_jacobian_diagnostics.json
```

## 必须指标

\[
RMSE_{\mathrm{FR}},
\quad
MAE_{\mathrm{FR}},
\quad
\max_t|\widehat y_t|,
\]

\[
r_{\mathrm{cont}}
=
\frac{\#\{\text{continuation evaluations}\}}
{\#\{\text{all amplitude evaluations}\}},
\]

\[
d_{\max}^{\mathrm{cont}}
=
\max
\frac{\operatorname{dist}(u,\mathcal S^{\mathrm{train}})}
{\operatorname{range}_{\mathrm{train}}+\epsilon}.
\]

## closure gate

- 完整 validation trajectory 无 NaN/Inf；
- 不读取中间真实输出；
- 不在线更新 fitted parameters；
- simulation bound 文件与实际最大值一致；
- direct 与 free-run 分栏；
- 域内/域外误差分栏。

性能是否优于 baseline 不是数学 closure gate，但必须如实比较。

---

# 19. E-FR-STAB：递推稳定证书

## 全局证书

计算：

\[
\rho(A_{\mathrm{AR}}).
\]

若：

\[
\rho(A_{\mathrm{AR}})<1,
\]

输出：

```text
GLOBAL_INCREMENTAL_STABILITY_CERTIFIED
```

否则输出：

```text
GLOBAL_INCREMENTAL_STABILITY_NOT_CERTIFIED
SIMULATION_BOUNDED_BY_CONTINUATION
```

不得把证书失败写成实际发散。

## 初值扰动实验

对官方允许的 initialization window 添加预注册扰动：

\[
\delta_0\in
\{\pm0.1\sigma_y,\pm0.25\sigma_y,\pm0.5\sigma_y\}.
\]

报告轨迹差：

\[
D_k
=
|\widehat y_{t_0+k}^{(\delta_0)}
-\widehat y_{t_0+k}^{(0)}|.
\]

若有全局证书，验证指数上界；无证书时只作诊断。

## 局部 Jacobian

报告：

- \(\rho(J_t)\) 分位数；
- 最大 \(\|J_{t:s}\|\) 近似；
- 首次 continuation 前后局部增益；
- 不能升级为全局证明。

---

# 20. E-H2-NATIVE：独立系统辨识模型闭环

## 目标

主结果不依赖 ARX 提供 history。

## 顺序

```text
H2-A native history
H2-B resolution
H2-B penalty
H2-C continuation
H2-D rank
```

## 必须冻结

```text
history_selection.json
resolution_selection.json
penalty_search.json
continuation_selection.json
rank_profile.csv
```

## gate

- H2 在四数据集均有结果或显式 preregistered partial；
- H3 仅作 shared-history 消融；
- official test 不参与；
- exact-zero penalty 与原坐标 KKT 通过；
- continuation scale 只在 development 中选择。

---

# 21. E-INTERP-FIREWALL：解释防火墙

## 自动规则

若任一评价点位于：

\[
\mathcal D^{\mathrm{model}}
\setminus
\mathcal S^{\mathrm{train}},
\]

则对应输出必须包含：

```text
OUTSIDE_TRAIN_SUPPORT
Q_IDENTIFICATION_NOT_CLAIMED
K_INTERPRETATION_FORBIDDEN
```

若位于训练支撑但 Schur/Gram 不通过：

```text
PREDICTIVE_CONTRIBUTION_REPORTABLE
K_LEVEL_NOT_IDENTIFIED
```

## 审计

全文表格和图注不得出现：

- extrapolated physical kernel；
- learned out-of-domain dynamics；
- structural rank from continuation region；
- causal plant extension。

---

# 22. PB1-v4.1 统一执行矩阵

| 数据 | H2 | H3 | Direct | Free-run | C1 continuation | Baselines |
|---|---:|---:|---:|---:|---:|---:|
| PWH | 必做 | 保留 | 1/5/10/20 | 必做 | 必做 | ARX/pNARX/MLP-NARX |
| WHPN | 必做 | 保留 | 1/5/10/20 | 必做 | 必做 | ARX/pNARX/MLP-NARX/过程噪声 comparator |
| Tanks | 必做 | 9/8 | 1/5/10/20 | 必做 | 必做 | 权威论文参数 |
| Silverbox | 必做 | 10/10 | 1/5/10/20 | 必做 | 必做 | 权威论文参数 |

## official test gate

只有以下全部成立才允许 confirmation：

```text
H2_NATIVE_COMPLETE
SIMULATION_CLOSURE_COMPLETE
CONTINUATION_RULE_FROZEN
INTERPRETATION_FIREWALL_PASS
ALL_PRIMARY_KKT_PASS
BASELINE_MATRIX_COMPLETE
OFFICIAL_TEST_ACCESS_COUNT_ZERO
SELF_CONTAINED_PACKAGE_PASS
```

---

# 23. v4.1 状态码

```text
SIMULATION_CLOSED
SIMULATION_BOUNDED_BY_CONTINUATION
GLOBAL_INCREMENTAL_STABILITY_CERTIFIED
GLOBAL_INCREMENTAL_STABILITY_NOT_CERTIFIED
OUTSIDE_TRAIN_SUPPORT
K_LEVEL_NOT_IDENTIFIED
CONTINUATION_CHANGED_IN_SUPPORT_MODEL
DIRECT_ONLY_MODEL
FREE_RUN_COMPLETE
FREE_RUN_NUMERICAL_FAILURE
```

这些状态不得互相替代。
