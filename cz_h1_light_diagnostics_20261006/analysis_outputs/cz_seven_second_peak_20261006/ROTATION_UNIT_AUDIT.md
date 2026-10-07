# 旋转速度单位只读审计

结论：**UNCONFIRMED**。无法从当前项目证明晶转速度/埚转速度的物理单位。没有进行 rpm→Hz 换算、旋转相位积分或 phase model fitting。

证据来源：

- 原工作簿 Sheet1/Sheet2：字段只有“晶转速度”“埚转速度”；表头无单位注释，数值格式不含物理单位。工作簿描述属性没有单位说明，见 `ROTATION_UNIT_EVIDENCE.json`。
- 原 H1 `cz_l256_nowcast.py`：仅将上述字段映射为 crystal_rotation/crucible_rotation，并声明 RAW_PERIOD_SECONDS=2.0；没有旋转单位声明。
- `residual_rollout_authority_20260928/PRISM_INDUSTRIAL_BENCHMARK_V1/dataset_registry/cz_czochralski/` 和本项目 CZ data code：未发现 rpm、rev/min、r/min、rev/s、rad/s 或转/分证据。全项目可见文本中的 rad/s 命中来自 NanoDrone benchmark，与 CZ 无关，见 `UNIT_SEARCH.log`。
- 既有 `analysis_outputs/cz_heater_identifiability_20261001/REPORT.md` 的数据说明同样明确工作簿没有可验证的单位说明。

按用户要求，速度分布仅按原始记录单位报告（`ROTATION_SPEED_DISTRIBUTIONS.csv`）。数值大小与某种常见单位相似不构成单位证据。恒速旋转可具有相位相关输出但没有速度波动 coherence，因此未确认单位或低 speed coherence 均不能排除 H_R。
