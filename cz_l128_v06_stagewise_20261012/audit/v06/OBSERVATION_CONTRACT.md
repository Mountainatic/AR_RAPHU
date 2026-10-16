# Phase 1 — 不可变观测合同

DiscreteObservationSpec 为 frozen dataclass；gap_rule 深度冻结、显式 JSON 往返。十个字段：sample_period_seconds、time_origin_policy、timestamp_field、output_unit、output_record_grid_candidate、grid_evidence_status、segment_policy、gap_rule、observation_availability_policy、head_contract_id。缺失 optional_observation_spec 返回 None，不修改旧配置。默认不调用审计；现有训练、预测、选择入口没有导入新增模块。

output_record_grid_candidate 是输出记录格点，绝不复用 K 的幅值编码 Q_j。默认未知；显式 grid_candidate_evidence 使用零原点最近格点距离，基础容差 1e-8 recorded unit，分别输出严格十进制格点比例与表示误差兼容比例。C1是float64；若每个值恰能binary32往返，另报告半binary32 ULP下的格点兼容候选，明确这是经验精度兼容证据，不是原生存储/仪器证据。原始格点距离仍保留，兼容比例不足.999则unknown。没有仪表配置就不能自动标为 instrument_verified。单位 recorded_diameter_unit_unverified；不要写 mm。

原始 index、entity/segment/split 与记录时间戳在审计层导出。真实 C1 无记录时间戳，2*index 只标 NOT_RECORDED_PROTOCOL_COORDINATE。可选真实 timestamp_field 若不规则、非递增、缺失或偏离声明 cadence，分开新诊断；未知 cadence 不报告物理 PSD。不填补断裂。缺失记录或跨段产生 POTENTIAL_LEAK_OR_INVALID_SUPPORT，不改训练行/旧 RMSE。未增加 strict_segment_support 训练模式，不改变评分支持。

本次元数据与诊断不输入 W/A/route、折内调参或候选门禁；无 rounding、量化似然、未来补缺、one-SE 或白化门禁增量。既有基线选择协议（包括其历史选择规则）保持原样。

