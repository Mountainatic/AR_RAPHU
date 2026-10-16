# Phase 0 — 代码和资产审计

基准：新服务器实际运行版 `/root/autodl-tmp/CZ_CROSSROD_L128_20261008/code/canonical/cz_crossrod_l128_20261008`，入口 run_canonical.py SHA256 `9c9af7c6f39b83522dafd36cbed152e1d3177514b797ea340867287ca71c3b44`；本地旧副本有一处入口差异，本次已取服务器版。baseline_runtime 保留原样，runtime 在其上仅添加三个模块。代码绑定详见 SOURCE_BINDING.json。

|功能|实际路径/函数|
|---|---|
|入口/开发/推理|runtime/canonical/cz_crossrod_l128_20261008/run_canonical.py: main；canonical_engine.py: CanonicalPipeline.learn/tune/predict/emit/online_a|
|加载和测试隔离|runtime/canonical/cz_crossrod_l128_20261008/data.py: load_development/load_inference；deployment.py；release.py: build_release；final_development_cf.py: replay|
|历史 E/多率对齐|/root/autodl-tmp/CZ_CROSSROD_L128_20261008/code/src/prism_benchmark/cpu_data.py: BaseAccessor.block_means；v2_k.py: profile_values；v2_basis.py: causal_geometric_intervals|
|K|src/prism_benchmark/v2_urysohn.py: predict_contract，有限 lag×amplitude theta；canonical_engine._features/_values|
|C/W|src/prism_benchmark/v2_c.py: _ridge_predict；v211_w.py: predict_w_correction；canonical_engine.response_coordinate|
|A/成熟 CF|runtime/canonical/cz_crossrod_l128_20261008/a_state.py: cz_target_end/build_a_training_state_from_cf_residual/feature_rows；canonical_engine.online_a|
|目标 H1/H4|src/prism_benchmark/cz_l256_nowcast.py: _effective_h_steps/_sample_rows/_raw_segments/_base_frame|
|折/评分支持|runtime/canonical/cz_crossrod_l128_20261008/fold_geometry.py: folds；temporal_support.py: past_fit_positions；risk.py；pointwise.py: paired_pointwise|
|新增只读审计|runtime/canonical/cz_crossrod_l128_20261008/discrete_audit.py: audit_target_contract；discrete_diagnostics.py: segmented_diagnostics|

采样间隔来自冻结 CZ_TASK_REALIZATION.json 的 2 秒约定。base_data 没有测量时间戳、单位或 ADC 配置，不能从它检测物理采样间隔。原始工作簿 `实验数据1-张.xlsx` 与私有 raw/cz_raw.xlsx 为既有输入；本次只读已绑定 source C1，不打开 target test。实体来自显式原始区段；Rod1 断点 0/334/688/734/20103，Rod2 0/745/20627。这些不是时间戳检测证据。row_in_entity 是段内原始 index；raw_global_index 可由显式段起点加段内 index 恢复。

目标的 recorded_level 未因本次格点候选进行缩放、平滑或取整；模型输出为连续 FP64。lag=0 块 [origin-far, origin-near) 在 near=0 时仍以 origin 为排他端点，最后输入为 t-1，当前目标从未成为输入。

当前两方向 C1 和 CPU 冻结源验证预测存在；深度 120 checkpoint 回执存在。E1 尚无最终冻结 K/C/W/A，test 未打开，E2–E6 未执行。历史 analysis_outputs/cz_stagewise_diagnostics_20260928 是旧 H1/H4 资产；cz_h1_light_whitening_20261006 是同棒 raw-history Ridge 诊断，不能冒充本次 L128 跨棒最终创新。ASSET_INVENTORY.json 区分 source raw、source validation、冻结基线；没有将 train/OOF/test 混合成创新。

