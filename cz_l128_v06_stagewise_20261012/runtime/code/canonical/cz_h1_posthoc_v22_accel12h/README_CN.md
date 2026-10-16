# CZ H1 POST_HOC accelerated canonical protocol

版本：`CZ_H1_POSTHOC_ACCEL12H_V1`。基线为已验收的
`CZ_H1_POSTHOC_CANONICAL_FINAL_CONFORMANCE_CODE_REVIEW_20261004`；
`ACCEL_BASELINE.json` 登记其 commit、交付 ZIP、25 个 Python 文件和五份 registry。
本目录独立实现新的 hierarchical / reduced-refresh learning policy，科学来源仍为
`POST_HOC_PROTOCOL_REVISION`。它与 exhaustive policy 不保证选出相同模型。

K 先用 registration order 中第一条 rank-1 recipe 评估全部 69 个 temporal
coordinates，再在选中 coordinate 上评估全部 196 条 structural recipes。
screen recipe 的 exact ID 和原 registry SHA 已在读取本轮 development target
之前冻结于 `ACCEL_K_SEARCH_PROTOCOL.json`。512 s history、candidate-native fit、
exact common scoring support、FP64/tie rule、所有 nonlinear family 均保留。
exhaustive 只存在于独立 reporting audit 分支，无 routing authority、无自动回退。

final replay 每个 entity 从最早合法 anchor 开始，每 320 行 refit；B0、B4、B8
等 checkpoint 每 1280 行在严格 past-only prefix 上重新选择 MODEL recipe。
tails 和早期 identity 行保留。global ADMIT/BYPASS 不重新投票。
按用户确认，K BYPASS 保持预测 identity，但供 C/W 使用的 retained K 表征
仍按上述 cadence 合法选择/重拟合；C/W BYPASS 输出分支跳过搜索。
inner-selection stream 与 final-development replay 分开，最终 W/A 用完整合法
CF/prequential residual 拟合。A 原有 maturity、empty cold-start、identity completion
和 segment 隔离规则未改。

同一 exact learning node 由 content-addressed checkpoint 复用。K input-only
cache 与 target-dependent statistics 分开：lag cache 每个 immutable
entity/origin/profile/m_tau 只构造一次；basis cache 保存 basis、shape、精确 Gram
和 feature sum；独立 target cache 保存 unchanged authority batch statistics。
authority prepared solver 的统计量路径只读取设计矩阵形状，`DesignShape` 禁止
array fallback。缓存同步使用内容哈希和进程锁，不依赖文件时间戳。
C/A 多 alpha 共享原批量 Gram/RHS；不启用未经证明的 incremental estimator。

每个方向四个 spawn channel workers，两方向共八个 numerical workers，所有
BLAS/OMP/MKL/NumExpr 线程数为一。`ENVIRONMENT_FINGERPRINT.json` 冻结实际硬件
和数值依赖。所有原 namespace、H4、authority solver、旧 results 保留。

本轮只做 development-only benchmark 和代码验收。三个 formal authorization
均为 FALSE，validation/test parquet 字节读取由 guard 拦截。
benchmark 固定为两个方向的 3000/8000/15000-row prefixes，覆盖完整 policy、
cold/warm cache、reselection/refit-only 和 full replay；不生成 deployment seal。
只有完整验收通过且保守 projection（safety factor ≥1.25）≤36000 s 才能标记
`READY_FOR_USER_AUTHORIZATION_OF_12H_FORMAL_RUN`。这不授予运行权限。

未来经用户授权后，正式运行采用同 namespace 的累计 wall-time ledger；
11 h 不开始新 candidate batch，11.5 h supervisor 停止进程，resume 不重置预算。
失败 gate 必须报告 `NOT_READY_FOR_12H_FORMAL_RUN`。

从 benchmark 根目录运行：
`python -m canonical.cz_h1_posthoc_v22_accel12h.run_canonical --help`。
seal 必须传入已验收 registration；benchmark 只允许已注册 prefixes。
实际计时、科学差异、等价性、22 项 conformance 和交付 SHA 见返回包中的审计文件。
