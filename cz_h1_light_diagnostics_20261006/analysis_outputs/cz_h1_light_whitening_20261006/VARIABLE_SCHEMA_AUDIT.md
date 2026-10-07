# CZ H1 变量审计

训练前生成。Rod1 20103 行、10 个字段；Rod2 20627 行、7 个字段。没有缺失值。全部数值采用原始单位。

Rod1-only：氩气流量设定、晶体长度、炉压。温度是两棒公共观测变量，但不在 Common4 中。AllLegal 固定使用四个正式输入，加两种原始升速、温度及三个 Rod1-only 字段，共 10 个通道。晶体直径不作为 predictor。

合法性条件：沿用项目 record_time 约定，仅用 t-1 及更早记录；原工作簿没有独立传感器延迟或字段生成元数据。晶体长度按已记录工艺状态处理，没有利用直径构造它；如该字段实际为离线回填，需另一次预注册实验排除，不能根据本轮结果删变量。

joint_lift：原 cz_l256_nowcast.py 的 _fit_joint_lift / _apply_joint_lift 原样复用（AST 仅提取数据函数）；晶升/埚升训练历史行去重并标准化，取 PC1，晶升 loading 为非负。每个 fit 单独拟合 PCA，HISTORY_STEPS 设置为该候选 L；不读取未来/validation/test 行拟合。

|rod|field|dtype|missing|variance|unique|category|legal|
|---|---|---|---:|---:|---:|---|---|
|Rod_1|主加热功率|float64|0|1.27853|613|control|True|
|Rod_1|加热元件温度|float64|0|19.3218|10414|measurement_context|True|
|Rod_1|埚升速度|float64|0|0.000182517|620|actuator|True|
|Rod_1|埚转速度|float64|0|0.000794311|171|actuator|True|
|Rod_1|晶体直径|float64|0|0.332992|381|target_measurement|False|
|Rod_1|晶体长度|int64|0|12097.5|385|operating_state|True|
|Rod_1|晶升速度|float64|0|0.00832001|988|actuator|True|
|Rod_1|晶转速度|float64|0|0.0543081|902|actuator|True|
|Rod_1|氩气流量设定|int64|0|0|1|control_setpoint|True|
|Rod_1|炉压|float64|0|0.00196205|25|operating_state|True|
|Rod_2|主加热功率|float64|0|1.40129|647|control|True|
|Rod_2|加热元件温度|float64|0|24.192|11864|measurement_context|True|
|Rod_2|埚升速度|float64|0|0.000105746|502|actuator|True|
|Rod_2|埚转速度|float64|0|0.000567996|154|actuator|True|
|Rod_2|晶体直径|float64|0|0.1958|373|target_measurement|False|
|Rod_2|晶升速度|float64|0|0.00461418|786|actuator|True|
|Rod_2|晶转速度|float64|0|0.0933917|559|actuator|True|
