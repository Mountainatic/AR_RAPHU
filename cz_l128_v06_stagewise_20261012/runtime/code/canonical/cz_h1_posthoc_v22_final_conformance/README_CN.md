# CZ H1 POST_HOC canonical final conformance

本目录从 1bb5a2230fee45d6ebd1d35a624c698bf47bff88 创建。IMMUTABLE_BASELINE.json 登记上一轮代码和交付 ZIP；旧 namespace、authority solver、H4 和历史 release 均保留。新协议为 CZ_H1_POSTHOC_CANONICAL_FINAL_CONFORMANCE_V1。FINAL_PROTOCOL.json 与 CANONICAL_PROTOCOL.json 内容一致。

MODEL recipe 只在 exact ordered common score IDs 上比较；candidate-native fit 保留。outer admission 使用完整注册 320 paired rows，不可用行 child=parent 并保留。SUPPORT_LIMITED、DOMAIN_NOT_APPLICABLE、NUMERICALLY_INVALID 独立登记，不能用零预测冒充 recipe。

geometry 使用 ENTITY_AWARE_NEAREST_LEGAL_4X320_V1：四块联合最小化总位移、最大位移、lexical endpoints。每块 320 行、互不重叠、单 entity、origin +1。nominal/actual endpoints、位移和 support hash 全部输出。有限支持不足为 POLICY_NOT_ESTIMABLE_IN_PARTITION，保留合法 parent/neutral identity，不产生 admission vote。完整 formal top-level geometry 不足则 hard preflight fail。

K 的 channel recipe 先在 registry common support 选定，再用 selected representations 的 assembly intersection 选择 reference channel。各 fold 使用其实际 training contract，在 exact assembly IDs 上重预测；不读取 outer-held-out target。K BYPASS 保留 representations。C 的两种 family × authority alpha grid 全部用 inner common-support predictive risk 比较，不添加 alpha=0。

K_MODEL_REGISTRY、C_MODEL_REGISTRY、W_MODEL_REGISTRY、A_MODEL_REGISTRY 记录 exact candidate order、hyperparameter role、support role 和 payload SHA。其 authority source binding 通过 AUTHORITY_UNCHANGED_BINDINGS.json 传递到原 K/C/W/A 配置和 solver；release 再核验所有实际源文件。A_TEMPORAL_REGISTRY 保留原值。

W response coordinate 与 baseline 分开：C admitted 用 frozen Psi_C，C bypass/unavailable 用 retained Psi_K；baseline 始终 F_C。A 的 outer、validation、test 均 empty cold-start，无 warmup；先 issue prediction，再等待 target maturity 后加入 W residual。state 不跨 block/entity，缺完整成熟历史 identity 保留评分行。

final-development replay 仍为每 entity 的 320-row expanding/prequential blocks，包含 tails 和全部合法 anchors，且在 recipe freeze 后重建。inner selection stream 不代替 final replay。

validation/test maximal support 在 release 构建时只投影 metadata 和 raw row identities；保持原 split boundary，旧 IDs 是新 IDs 子集。真实 target 只能经 deployment / validation-before-test barriers 后读取。本轮三个 formal authorization 仍 FALSE，hash-only byte sealing 与 target-value projection 分开记录。

outer pointwise 的 global_routing_status 在 aggregate 前为 null，model_domain_status 独立记录；global freeze 不重写 outer_realization_id 或 fold MODEL predictions。递归对象只有 partition_model_id；deployment_model_id 仅 final deployment seal 创建。兼容 A state 字段 upstream_deployment_model_id 同时携带 upstream_model_scope，PARTITION scope 不声称部署。

resume 必须显式 --resume，复用相同 release/data/environment/mode/partition/registry 的已 seal candidate/block/model；未 seal 工作重算。numerical-invalid 候选也保留 checkpoint。full-registry mode 独立读取 full_registry_smoke.prefix_rows。

从 benchmark 根目录运行 python -m canonical.cz_h1_posthoc_v22_final_conformance.run_canonical。seal/preflight/smoke/full-registry-smoke 可执行；train/validate/test 受 frozen authorization 限制。不得用 CLI 绕过。参考最终 conformance matrix、proofs、UNIT_TESTS.xml 和 DELIVERY_REPORT_CN.md；扫描完成不表示每个候选都 AVAILABLE。
