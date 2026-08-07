# PRISM v2.1.1 Metro-P60 必需输出合同

## 1. 顶层状态

必须输出 `RUN_STATUS.json`；停止时另有 `STOP_STATE.txt`。状态至少包含 protocol、branch/commit/git-clean、evidence class、active head、stage、development frozen、test/OOD access、数据/代码/配置 hash。

## 2. Development

建议目录：

```text
DEVELOPMENT/K/<head>/<view>/<channel>/RESULT.json
DEVELOPMENT/C/<head>/<view>/RESULT.json
DEVELOPMENT/W/<head>/<view>/RESULT.json
DEVELOPMENT/A/<head>/<view>/RESULT.json
DEVELOPMENT/JOINT/<head>/<view>/RESULT.json
```

K 必须保存全部 profile 风险、best/retained、regret、family/rank、exact-zero/active、refit lambda、数值证书、OOF prediction hash、candidate-id。

C 必须保存 family 候选、最小稳定 ridge、best-active-K 参考、方差/MSE 保存门、系数范数、fallback、input-path 状态、OOF hash、candidate-id。

W 必须保存三类候选清单、每折可用性及原因、row-id hash、latent distinct/rank/scale、knots/smoothness/mu、fold loss/增益、usable folds、one-SE/practical gate、identity 两个等价误差、support/derivative 审计、OOF hash、candidate-id。

A 必须保存 r2 来源 candidate IDs、成熟合同、latest index、fold-local centering、exact-zero/active 候选、practical gate、有效系数/预测方差、最终物化 candidate-id。

Joint 必须保存恰好 `J_K/J_KW/J_KA/J_KWA`、各 block 维度、W 系数联合求解证据、无 AR-only/K-zero、共享 gate、每折损失、全部消融预测 hash、selected candidate-id。

## 3. Freeze

必须输出：

```text
FREEZE/METRO_P60_V211_DEVELOPMENT_FREEZE.json
FREEZE/METRO_P60_V211_DEVELOPMENT_DECISION.json
```

并声明 `test_accessed=false`、`ood_accessed=false`。

## 4. Final predictions

逐样本至少包含：sample_id、base_origin_id、split、view、candidate_id、y_true、y_pred、prediction_available、latest_available_target_index。

必须物化：

```text
KC, KCW, KCA, KCWA, PF_SELECTED
J_K, J_KW, J_KA, J_KWA, J_SELECTED
```

## 5. Final tables

```text
FINAL/METRO_P60_V211_FINAL_METRICS.csv
FINAL/METRO_P60_V211_W_MARGINALS.csv
FINAL/METRO_P60_V211_BOOTSTRAP.csv
FINAL/METRO_P60_V211_SELECTION_TRANSFER_AUDIT.json
FINAL/METRO_P60_V211_REPORT.md
```

还要保存 peak RSS、各阶段 wall time、row cap 实际行数、chunk 数、solver failure inventory、数据 hash、code diff、tests、manifest 与 SHA256SUMS。
