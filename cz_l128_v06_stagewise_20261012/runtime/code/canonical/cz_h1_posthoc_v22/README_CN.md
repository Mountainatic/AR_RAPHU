# CZ H1 post-hoc canonical 修复代码

入口是 `canonical.cz_h1_posthoc_v22.run_canonical`。从包内
`code/PRISM_INDUSTRIAL_BENCHMARK_V1` 目录运行。旧 `cz_h1_registry_v2`
目录、权威数值求解器、K 23-profile registry、maximal C1 和 H4 均保留。

本次协议是 POST_HOC_PROTOCOL_REVISION，不能把历史 Rod1/Rod2 target test
重新称为未使用的 confirmatory test。历史 sealed false provenance flags
保留为审计快照，新 registration/protocol 是这次修订的有效声明。

正式 inner/outer 均为 4 个 320 行 contiguous scoring blocks，结束位置是
floor(N×40/60/80/100%)。不足完整互不重叠块时明确 N/A。selection CF 每块
最多使用之前 640 行 context，context 与真正 score 的 provenance 分开。
final-development CF 独立生成：320 行 contiguous prequential blocks，每段
tail 保留，每个 source-development anchor 都有合法 out-of-fit parent
预测；没有可拟合 prefix 时传播预先注册的 identity baseline。

A history registry 是 (delta=1,history=8/16/32/64/96/128)，cap=128 lags。
H1 的最新成熟 residual origin 是 t-1；A8 对应 2–16 s。缺少任何所需
mature lag 的训练行不进入该 candidate 的 fit。MODEL score 使用共同成熟
交集，其余行仍以 identity 留在同一 paired risk。部署 A 只使用当前部署
W 的成熟在线残差；CF training stream 不会作为 inference seed。

现有协议允许 `seal`、`preflight`、`smoke`，禁止 `train`、`validate`、
`test`。这些入口已经实现，但本次不会自动正式运行。未来若明确授权
正式实验，需要冻结新的执行授权协议和 release；CLI flag 不能绕过停止线。

先在服务器用既有 authority 和 maximal C1 创建不可覆盖的新 release：

```powershell
python -m canonical.cz_h1_posthoc_v22.run_canonical seal --authority-project AUTHORITY_ROOT --c1-root C1_H1_ROOT --release-dir NEW_RELEASE_DIR
```

记录输出的 release SHA，显式传给实际 runner：

```powershell
python -m canonical.cz_h1_posthoc_v22.run_canonical preflight --authority-project AUTHORITY_ROOT --c1-root C1_H1_ROOT --release-dir NEW_RELEASE_DIR --release-sha PINNED_RELEASE_SHA --case-id Rod_1_to_Rod_2 --output-root NEW_PREFLIGHT_DIR
python -m canonical.cz_h1_posthoc_v22.run_canonical smoke --authority-project AUTHORITY_ROOT --c1-root C1_H1_ROOT --release-dir NEW_RELEASE_DIR --release-sha PINNED_RELEASE_SHA --case-id Rod_1_to_Rod_2 --output-root NEW_SMOKE_DIR
```

smoke 固定使用 3000 行 source-development prefix 和预先声明的小候选
子集，仅用于数值/递归接口检查。所有 smoke 状态都带 formal_eligible=false，
不能封存为正式部署 checkpoint。正式模式仍遍历完整注册候选，没有继承
smoke 缩减。report-only bootstrap 使用 500 replicates、seed=20260804、
95% interval、H1 block lengths=[2,4,64]，永远不改变路由。

环境须包含原 authority 所需 numpy/pandas/scipy/pyarrow/pyreadr 等依赖；
本次使用服务器已有环境，无安装/升级。OMP/MKL/OpenBLAS/NumExpr 均为1。
CPU/FP64 authority solver 保持原证书要求。新 runner 并未把 K 数值求解器
重新实现成另一种算法。

validation/test API 在部署 manifest 和全部 artifact hashes 验证以后才
读相应标签。test 还需匹配的 source-validation receipt。推断 engine
禁用所有 learn/tune/refit，A history 保持 entity/window maturity 约束。
本次只测试 synthetic label fixtures，没有打开真实 validation/test。

包根目录 `validation_harness` 提供本次服务器执行的原始检查脚本，包含
两个真实 source-development smoke 和固定 joint-lift 合成目标夹具；
没有 SSH 凭据或数据。脚本中的路径对应本次服务器 workspace，需要按
复核环境调整路径。它们不属于正式 runner，不授予 training/test 权限。
