# CZ raw-2s H4 权威 PRISM E1–E6 重跑计划

状态：`PLANNED_NOT_STARTED`

机器可读计划：
`configs/cz_raw2s_h4_authority_e1_e6_rerun_plan_20260922.json`

## 1. 目标与证据边界

本计划只针对私有 CZ raw-2s 任务：

```text
task       = CZ_DIAM_RAW2S_CURRENT_L256_H4
cadence    = 2 seconds
input      = [t-256,t)
anchor     = D[t-1]
target     = D[t+3] - D[t-1]
H/W/W0     = 4/1/1
lead       = 8 seconds
directions = Rod1->Rod2 and Rod2->Rod1, fitted independently
```

权威方法基线固定为分支
`prism-strict-oof-finalization-20260915` 的提交
`2ee6273b8f915cbcdff2f46d56bc80047ddae4a7`。K、C、W、A、Joint、严格
nested-OOF 选择器和 portable checkpoint 必须来自这条权威实现链。禁止再用
rolling-statistics/PCA/Ridge 代理模型替代 PRISM 模块。

E1 以已经封存并完成正式测试的 R3 为锚点：

```text
/root/autodl-tmp/PRISM_CZ_RAW2S_H4_AUTHORITY_E1_E6_20260922_R3/
```

E2–E6 只允许访问每个方向的 source-rod development 数据，不允许访问 formal
target rod，也不允许访问 OOD。这样后续适配器开发不会利用已经看到的正式目标棒
结果进行选择。纯 K 只允许从 R3 已封存 K 合同进行确定性 replay，不允许重新调参。

## 2. 总体执行顺序

```text
计划/代码/数据冻结
        |
        v
适配器单元测试与微型 pilot
        |
        +--> E1 纯 K replay 与完整阶梯
        |
        +--> E2 半合成恢复 ---- Phase-A GO/STOP gate
                                  |
                                  v
                         E3 多尺度 + E4 敏感性
                                  |
                                  v
                         E5 稳定性 + E6 鲁棒性
                                  |
                                  v
                   support/checkpoint/report/private package
```

只有 E2 的恢复与空模型校准门禁通过后，才进入大规模 E3–E6。任何阶段失败均保留
原始审计记录并停止下游，不覆盖、不删除、不把失败结果改名为成功结果。

## 3. 各实验的正式设计

### E1：权威逐阶段消融

研究问题：K、C、Delta-W、A 和 Joint 各自对同一支持集上的增量是什么。

计划：

1. 复用 R3 已封存 K/C/W/A/Joint 选择和检查点，不改变任何阶段决策。
2. 新增纯 K portable checkpoint codec。它只能组合已冻结的 active K channel
   contracts，不能包含 C/W/A/Joint 系数。
3. 在 OOF 上证明纯 K replay 与 C 模块实际接收的 K parent 逐元素一致。
4. 在两个方向的同一 sample-id 顺序和 common support 上报告：K、K+C、
   K+C+Delta-W、K+C+Delta-W+A、Joint。
5. 对 ZERO stage 要求与 parent bitwise identical；同时输出 delta/level 指标恒等式
   证书。

E1 不进行新的模型选择，正式测试只做封存 checkpoint 的确定性 replay。

### E2：半合成结构恢复

研究问题：当真实 K/C/W/A 结构已知时，权威严格 OOF 路由能否恢复正确结构，
以及样本增加时空阶段误接纳是否下降。

生成方式不是独立的简化 Ridge 模型，而是：

1. 只读取 source rod 的真实四路输入轨迹。
2. 在合法 segment 内对各通道做 seed-keyed block-circular shift，保持单通道时间
   结构，同时降低通道间偶然共线性。
3. 预先从权威候选宇宙登记 K、C、W、A truth operator；冻结 stage RMS 比例和
   innovation SNR。
4. 先生成完整的合成直径轨迹，再进入 C1、lag、anchor 和 target 构造。A 的历史
   因此来自一致的递归轨迹，而不是事后拼接特征。
5. 每个 seed 和方向完整重跑权威 K/C/W/A/Joint。

登记的 regime 为 S0=NULL、S1=K、S2=K+C、S3=K+C+W、S4=K+C+W+A。S1–S4
screening 使用 10 seeds，正式恢复使用 30 seeds、每组 2048 点。S0 NULL 校准使用
`n={1024,2048,4096,8192}`，每个 n 30 seeds。

Phase-A 门禁：

- 每个真实新增 stage 的恢复率必须严格大于 0.5；
- 每个 NULL stage 在每个 n 上的激活率必须严格小于 0.5；
- false-admission 对 `log2(n)` 的斜率必须为负，或整条曲线恒为零。

门禁不通过时，E3–E6 停止，结论写为 selector calibration 未通过。

### E3：等预算多尺度

研究问题：每个通道独立选择时间尺度是否比所有通道共享一个尺度更好。

候选仅来自 CZ 已登记的 11 个 K profile：

```text
(2,8) (4,8) (8,8)
(2,16) (4,16) (8,16) (16,16)
(2,32) (4,32) (8,32) (16,32)
```

其中二元组为 `(delta_steps, history_steps)`。

- Uniform arm：11 个 assignment，每个 assignment 让四个通道使用同一 profile。
- Multiscale arm：每个 seed 使用 11 个 balanced Latin-hypercube assignment；每个
  通道在 11 个 assignment 中恰好使用每个 profile 一次。
- 两个 arm 的 candidate-assignment budget 都是 11。
- K fit 只可按完全相同的 data/fold/candidate/code hash 做 content-addressed reuse；
  每个 assignment 的 C/W/A/Joint 必须重新运行。
- 使用 10 个预登记 seeds，主指标是 paired delta-RMSE gain；只作敏感性报告，
  不反向修改 E1。

### E4：候选宇宙与 family 消融

E4A 使用严格嵌套的 COARSE、STANDARD、EXPANDED 三个候选宇宙。同一方向的三个
variant 必须使用完全相同的 source rows、folds 和 support。K profile、W knot/
smoothness/soft-overlap、A profile、Joint route/eta 的具体集合已写入机器可读计划。
每个 variant 独立完整拟合，不能从完整模型中简单删列后评分。

E4B 使用五个 arm：

```text
FULL
NO_NONLINEAR_K
NO_C
NO_W
NO_A
```

每个 arm 只移除一个 family，随后从 K 开始重建所有受影响 parent，并在剩余权威
候选中重新选择。被移除 stage 用外部 exact identity 表示，不加入候选网格。

主要报告：delta RMSE、delta R2、level R2、persistence skill、stage vector、active
channels、selected profiles、参数量和 support hash。E4 只描述敏感性，不用于选择
“最好看的”正式模型。

### E5：结构稳定性

研究问题：预测误差接近时，结构是否也稳定。

来源包括 causal outer fold、10 个 moving-block seed、E4A candidate universe、
E4B family ablation 和两个方向。moving-block 只能在 source-rod 合法 segment 内
重采样，并保留时间顺序与 260 点 dependency purge。

必须分别报告：

- stage activation frequency；
- channel admission frequency；
- selected profile distribution；
- pairwise Jaccard；
- modal stage-vector frequency；
- `RMSE <= 1.01 * best RMSE` 的 Rashomon structure count。

方向、information set 和 H/W 不得混合计算稳定性。只有 modal stage-vector
frequency >= 0.8 且 median pairwise Jaccard >= 0.75 才能标记为 `STABLE`；否则
只能标记为 `SENSITIVE`，即使预测 RMSE 很接近也不能宣称结构稳定。

### E6：原始测量鲁棒性

扰动必须发生在 raw aligned measurement 层，然后重新执行 C1、normalization、
lag、anchor 和 target 构造。未来真值保持干净。

两种模式：

- `PROCESS_ONLY`：只扰动四路输入；
- `REALISTIC_PROCESS_PLUS_OBSERVED_DIAMETER_HISTORY`：同时扰动可见的历史直径和
  anchor，但绝不扰动或泄漏未来 target。

primary view 为 `dynamic/record_time`；`input_only/record_time` 只作为归因用的
secondary view。重建 level 始终与干净的 `D[t+3]` 比较。在 realistic 模式下，
delta truth 定义为干净的 `D[t+3]` 减去受扰动但在预测时可见的 anchor，避免把
未来真值本身也加入测量扰动。

条件：Gaussian `alpha={0,.01,.025,.05,.1}`、八个 signed bias、linear/random-walk
drift@0.05、四个 quantization resolution。噪声尺度只能来自相应 outer-train 的
unique raw measurements。

- N1：冻结 clean authority checkpoint，仅扰动 held-out development evaluation；
  30 seeds，不重拟合。
- N2：每个 condition/seed 从 raw 层重新物化数据并完整运行权威链；10 seeds。

alpha=0 必须满足跨 seed prediction hash 一致、与 clean replay 一致。任何扰动跨越
segment、改动未来真值或改变候选宇宙都立即停止。

## 4. 拟新增的适配器与测试

计划新增而尚未实现的文件：

```text
src/prism_benchmark/cz_authority_pure_k_checkpoint.py
src/prism_benchmark/cz_authority_semisynthetic.py
src/prism_benchmark/cz_authority_candidate_views.py
src/prism_benchmark/cz_authority_robustness.py
src/prism_benchmark/cz_authority_e1e6_reporting.py
scripts/run_cz_raw2s_authority_e1_e6_r2.py
tests/test_cz_authority_pure_k_checkpoint.py
tests/test_cz_authority_semisynthetic.py
tests/test_cz_authority_candidate_views.py
tests/test_cz_authority_robustness.py
tests/test_cz_authority_e1e6_end_to_end.py
```

这些文件只能负责数据物化、候选过滤、调用权威模块、缓存和报告；不得复制或重写
K/C/W/A/Joint estimator。权威 core blobs 必须继续通过 exact Git-blob 审计。

最低测试集：

1. H4 off-by-one、L256 strict-past 和 purge=260 边界测试；
2. Rod1/Rod2 独立性与 formal-target 禁读测试；
3. pure-K replay identity 测试；
4. E2 truth operator 与 C1 round-trip 测试；
5. E3 fit-budget equality 和 cache-key 完整性测试；
6. E4 nested-universe 与 one-family-only 测试；
7. E5 block 不跨 segment 测试；
8. E6 alpha=0 invariance、future truth 不变和 raw-before-C1 测试；
9. workers=1 与并发配置 prediction hash 等价测试；
10. 隐私扫描，禁止 raw workbook、原始行和凭证进入报告或 GitHub。

## 5. 并发、时间和存储

R3 实测约 23 分钟、162 MiB。服务器有 32 CPU 和充足内存，但当前私有盘只剩约
14 GiB。E6 N2 单独就包含 19 conditions × 10 seeds × 2 modes × 2 directions
= 760 个 raw-to-authority refit units；input-only 与 dynamic view 在同一 unit 内生成，
不能重复计为两个 refit。加上 E2–E5，完整 E1–E6 保守预计需要 96–160 小时和
200–300 GiB 私有可写空间。因此：

- 没有至少 300 GiB 私有空间时不得启动正式 run；
- 禁止把私有 raw 或可逆推出 raw 的中间产物放到公共挂载；
- 每个 seed/condition 使用独立目录和 immutable manifest；
- 最大四个 outer units 并发，默认每个 unit 内部 worker=1，避免 nested
  oversubscription；
- 只有 workers=1 与目标并发数通过 prediction-hash 等价 pilot 后，才允许提升
  inner workers；
- 只允许在 data/fold/candidate/code hashes 全部一致时复用 cache；
- 完成的 shard 可以做无损压缩和哈希封存，但不能丢弃必要审计证据。

## 6. 交付物和完成定义

每个 E 都必须有原始逐 seed/condition 表、聚合表、support/candidate hashes、状态
文件和审计证书。全局必须生成：

```text
PLAN_FREEZE.json
CODE_BINDINGS.json
SUPPORT_FREEZE.json
CHECKPOINT_MANIFEST.json
PRIVACY_AUDIT.json
RUN_STATUS.json
```

只有以下条件同时成立，才把总状态写成 `COMPLETED`：

1. E1 纯 K 到 Joint 阶梯完整；
2. E2 Phase-A gate 通过；
3. E3 等预算证书通过；
4. E4 每个 variant 都是独立权威 refit；
5. E5 没有跨方向混排；
6. E6 N1/N2 和 alpha=0 证书全部通过；
7. E2–E6 全程没有 formal target/OOD 访问；
8. checkpoint、代码、数据、support 和报告哈希全部可复验；
9. 隐私审计为 PASS。

否则必须使用 `PARTIAL`、`FAILED` 或具体 `STOP_*` 状态，不能用缺失实验的代理结果
补齐表格。
