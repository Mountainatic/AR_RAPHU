# PRISM v2.1.1 SRU 实现修复方案

## 0. 版本定位

- 版本：`PRISM v2.1.1 SRU Implementation Correction`
- 性质：对 v2.1 已发现的实现与选择器失配进行修复，不修改共享 C1 数据，不重调基线，不覆盖 v2.1 失败审计。
- 数据范围：仅 `SRU_H2S__H5__W1` 与 `SRU_SO2__H5__W1`。
- 运行范围：重新执行 E1–E5；E0 只复核哈希；baseline 逐样本产物直接复用。
- 测试边界：在新的开发冻结清单生成前，不读取、汇总或比较 V2.1.1 test 指标。

旧结果必须保留为：

```text
V2_1_DEVELOPMENT_STOP
IMPLEMENTATION_SELECTION_COLLAPSE
TEST_NOT_ACCESSED
NO_VALID_FINAL_CANDIDATE
```

---

## 1. 已确认的故障链

### 1.1 K profile 简化过度

旧选择在 profile 候选全部进入宽 one-SE 集合后，无条件优先最短历史，导致选择 `(8,10)`，即使它相对最佳 profile 的折均损失高出约 18%–26%。

### 1.2 C 将已发现的 K 信号重新压成近零

C 将 ridge 强度同时当作结构复杂度，one-SE 后优先最大正则，最终在 SO2 选择 `alpha=1e8`，使物理输出近常数。该行为违反“固定支持后的弱正则重拟合”语义。

### 1.3 W 被绝对方差门提前禁止

修复零 K 情况时加入了“所有折 latent variance 均大于固定绝对阈值”的条件。SO2 仅一折低于阈值，全部非线性 W 候选便未进入比较。此前保留的审计已经显示 monotone W 可在 4/4 折改善，并将 validation MSE 改善约 5.4%。

### 1.4 PF 与 Joint 对“输入支路非零”的判断不一致

PF 卡依据 active channel 数量判定输入非零；Joint 则依据实际预测方差判定坍缩，因而同一条输入支路得到相互矛盾的状态标签。

---

## 2. 总体修复原则

```text
K 负责选择输入响应结构；
C 只负责在已冻结 K 支持上融合，不得撤销 K；
W 只在有效 K/C 潜变量上拟合增量静态读出；
A 只拟合 K/C/W 后的成熟残差；
Joint-KWA 允许 K、W、AR 互相调节，但不得退化成 AR-only。
```

结构选择与数值稳定必须分离：

- one-SE 决定 profile、family、模块是否激活；
- ridge 仅解决条件数和 KKT，不再作为“越大越简单”的结构偏序；
- 已通过 K 激活门的输入支路不能被 C 的正则化静默压成零。

---

## 3. K 修复

### 3.1 Profile 采用 one-SE 与相对遗憾双门

设最佳 profile 为 `p*`，折均风险为 `R*`。候选 profile `p` 进入可接受集合必须同时满足：

1. `R(p) <= R* + SE*`；
2. `(R(p)-R*) / max(R*, eps) <= 0.02`。

即 one-SE 仍保留，但简化 profile 最多允许 2% 的平均风险代价，不能因为折间 SE 很宽而接受 18%–26% 的明显退化。

### 3.2 保留最多两个 profile

每个通道保留：

1. 折均风险最小的 profile；
2. 若存在，双门可接受集合中最简单且与最佳不同的 profile。

K family/resolution 在这两个 profile 上联合比较，而不是先把 profile 永久压缩成最短历史后再选择 family。

### 3.3 K 最终重拟合

K 结构冻结后使用最小稳定 ridge：

- 从 `lambda=0` 开始递增；
- 选择第一个满足 KKT、条件数、有效秩与有限系数门的 lambda；
- 禁止按“更强正则更简单”选择最终物理幅值；
- 若 `lambda=0` 已通过，必须选择 0。

---

## 4. C 修复

### 4.1 C 只选择表示族

C 候选仍为：

- `ADDITIVE_COMPRESSED`；
- `ADDITIVE_JOINT_BASIS`。

one-SE 只在这两个表示族之间选择。每个表示族内部的 ridge 使用最小稳定 ridge，不进入复杂度 tie-break。

### 4.2 输入支路保存门

设 `p_bestK` 为开发 OOF 上最佳已激活单通道 K 预测，`p_C` 为 C 融合后的 OOF 预测。C 必须满足：

1. `Var(p_C) / Var(y) >= max(1e-8, 0.10 * Var(p_bestK) / Var(y))`；
2. `MSE(p_C) <= 1.02 * MSE(p_bestK)`；
3. C 合同中至少一个非截距系数绝对值大于 `1e-10`；
4. 数值证书通过。

若 joint basis 不满足，回退 compressed；若 compressed 仍不满足，回退 `BEST_ACTIVE_K_CHANNEL`，而不是输出近常数 C。

该回退不表示 C 成功，只表示保存已认证 K：

```text
C_FALLBACK_TO_BEST_ACTIVE_K
```

### 4.3 禁止静默撤销 K

只要至少一个 K 通道通过激活门，C 不得输出 `input_path_nonzero=true` 但实际预测方差近零。此情况必须硬失败：

```text
C_INPUT_PATH_COLLAPSE_BUG
```

---

## 5. W 修复

### 5.1 准入门改为尺度无关

删除“所有折原始 latent variance 必须大于固定绝对阈值”。每折 W 可用性改为：

1. 标准化前 latent 至少有 20 个不同有限值；
2. `[1,q]` 的数值秩为 2；
3. `std(q) > 64 * eps_float64 * max(1, max(abs(q)))`；
4. 标准化后的 W 基有限且数值证书通过。

一折退化记为 `NOT_APPLICABLE`，只要至少 3/4 折可用，W 候选仍应参加比较。

### 5.2 W 选择规则

候选：

- `IDENTITY_CORRECTION`；
- `MONOTONE_I_SPLINE_CORRECTION`；
- `NATURAL_CUBIC_CORRECTION`。

保留 v2.1 guarded local one-SE：非零 W 可在 identity 仍位于 one-SE 集合时激活，但必须同时满足：

- 相对 OOF MSE 改善至少 1%；
- 至少 3/4 可用折改善；
- 最少 3 个可用折；
- 数值证书通过；
- C 输入支路保存门已通过。

### 5.3 防止 W 只是在放大被压扁的 C

若 C 保存门未通过，不允许给 W 结构解释；只能输出：

```text
W_RESCUE_DIAGNOSTIC_ONLY
```

正式 W 结果必须建立在非坍缩 C 上。继续保留 `mu=0`，并扫描 `{0, 0.03, 0.3, 3, 30}`。

---

## 6. A 修复与保留项

A 保留现有 v2.1 修正：

- 目标 `r2 = y - pK_oof - deltaW_oof`；
- 成熟条件 `s+h+W+D<=t`；
- 使用 `latest_available_target_index`；
- fold-local residual centering；
- 无硬特征投影；
- exact-zero 为正式候选。

新增一致性检查：A 的输入预测文件及 fold loss 必须来自 practical gate 后的同一最终候选。

---

## 7. Joint-KWA 修复

候选固定为：

- `J_K`；
- `J_KW`；
- `J_KA`；
- `J_KWA`。

要求：

1. 不注册 K-zero、AR-only 或 both-zero 候选；
2. AR-only 仅作为条件增益诊断，不参加最终选择；
3. 使用同一冻结 K 支持和 W 基，让 K/W/AR 系数在联合目标中共同优化；
4. 不得把预拟合 W 标量仅作为一个固定额外列来冒充 W 联合调节；
5. 输入支路有效性使用与 PF 完全相同的尺度无关保存门；
6. 若输入支路坍缩，输出 `JOINT_INPUT_PATH_COLLAPSED`，不回退 AR-only。

---

## 8. 必须新增的回归测试

1. `test_profile_one_se_regret_guard`：最简单 profile 虽在 one-SE 内但比最佳差 18%，必须被排除。
2. `test_profile_retains_best_and_near_simple`：最多保留最佳与 2% 内最简单 profile。
3. `test_minimal_stabilizing_ridge_prefers_zero`：lambda=0 数值通过时必须选 0，不能选 1e8。
4. `test_c_cannot_erase_active_k`：单通道 K 有显著预测时，C 不得输出近常数。
5. `test_c_fallback_to_best_active_k`：两个 C 表示均失败时回退最佳 K。
6. `test_w_three_of_four_usable_folds`：仅一折退化时，W 非线性候选仍运行。
7. `test_w_exact_zero_k_forces_identity`：全部 K exact-zero 时 W 必须 identity。
8. `test_pf_and_joint_share_input_path_gate`：PF 与 Joint 对输入非零状态必须一致。
9. `test_final_loss_matches_materialized_prediction`：选择、gate、落盘预测和 fold loss 属于同一候选。
10. `test_joint_has_no_ar_only_candidate`。
11. `test_no_test_access_before_v211_freeze`。

E1 必须全部通过后才能重跑开发阶段。

---

## 9. 最小重跑顺序

```text
E0R  复核 C1 与 baseline 哈希，不重建数据和 baseline
E1R  运行新增回归测试
E2R  重新运行两个 SRU 头的 K 与 C
E3R  重新运行 W
E4R  重新运行 A
E5R  重新运行 Joint-KWA
E5.5 生成开发停止/继续判定
E6R  仅在继续条件通过后生成 V2.1.1 冻结清单
E7R  冻结后首次访问 test
E8R  配对 block bootstrap 与最终报告
```

不要覆盖原目录，新的输出根目录：

```text
results_prism_v2_1_1_sru
```

---

## 10. 开发阶段继续门

只有同时满足以下条件才允许 E6R/E7R：

1. 至少一个 SRU 目标头存在非坍缩 K/C 输入支路；
2. PF 与 Joint 输入状态判断一致；
3. C 未触发 `C_INPUT_PATH_COLLAPSE_BUG`；
4. W 若激活，至少 3 个可用折且平均改善 >=1%；
5. 至少一个正式 V2.1.1 候选相对该头最强 development baseline 改善 >=1%，且至少 3/4 折改善；
6. 所有结果仍标记 `test_accessed=false`。

若不满足，直接停止：

```text
V2_1_1_DEVELOPMENT_STOP
NO_SUPPORTED_INCREMENT_AFTER_IMPLEMENTATION_REPAIR
TEST_NOT_ACCESSED
```

该停止结论表示在修复选择器后，SRU 仍不支持该路线；不再继续调参。

---

## 11. 本轮不做的事情

- 不重跑其他数据集；
- 不重建 C1；
- 不重调或删减 baseline；
- 不增加深度网络或新模型族；
- 不扩大 K/W/A 超参数网格；
- 不因 SO2 当前结果专门手调某个通道；
- 不访问 test 后返回修改开发规则。

