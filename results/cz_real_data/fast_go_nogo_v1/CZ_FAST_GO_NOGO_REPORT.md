# CZ 快速 GO / NO-GO 可辨识性审计报告

## 技术摘要

**自动判定：`AUDIT_INCOMPLETE`。** 本判定只适用于 Furnace A 前 80% development 轨迹上的快速筛查，不是最终模型选择、完整 K 面恢复或跨晶棒物理认证。

- 允许的下一阶段：`NONE_AMBIGUOUS_FAST_EVIDENCE`。
- 外生增量为正的固定 horizon：`[]`。
- Furnace B 访问次数：`0`。
- 总墙钟时间：`74.45` 秒。

**未闭合原因：** 三个固定任务的线性和粗非线性 XAR 增量均为负，但条件输入能量与条件 Gram 又不弱；现有快速证据既不满足继续完整 K 的 GO 条件，也不满足“激励和条件谱均弱”的 NO-GO 条件。


## 三个固定任务给出的预测证据

| 尺度 | horizon | 线性 Δ(X|AR) 两折均值 | 粗非线性 Δ(X|AR) 两折均值 | 两折方向一致 |
|---|---:|---:|---:|---|
| short | 1 | -1.2343 | -834.173 | 否 |
| medium | 15 | -13.3125 | -812.732 | 否 |
| long | 60 | -20.418 | -740.11 | 否 |

这里的 Δ 定义为 `(MSE_AR - MSE_XAR) / MSE_AR`；正值表示加入五个外生过程量后验证误差下降。该量是预测增量证据，不等同于物理因果效应。

## AR 条件化后仍剩多少独立输入信息

| 尺度 | 条件能量最高的输入 | 条件能量比 | 最大 AR 残差相关输入 | |corr| | lag（采样步） |
|---|---|---:|---|---:|---:|
| short | 埚升速度 | 1.89066 | 晶升速度 | 0.113214 | 41 |
| medium | 埚升速度 | 1.47111 | 晶转速度 | 0.305653 | 125 |
| long | 埚升速度 | 1.28501 | 晶转速度 | 0.520131 | 112 |

条件能量使用 train-only ridge 将输入历史对严格滞后的直径历史做残差化；FAST-B 再检查 AR 残差与输入滞后的相关。相关性仍是诊断量，不能单独证明工艺机制。

## 条件 Gram 与低阶 K 稳定性

- 联合条件 Gram 的折间中位 effective rank：`13.8865`。
- `1e-3` 相对谱阈值下的中位 coercive dimension：`99.5`。
- 低阶 K 稳定性：`K_NOT_TESTED_DUE_TO_NO_Q_GAIN`；FAST-E 没有正的 Q 增量，因此合同禁止将其解释为 K 不稳定。

低阶 K 只表示粗网格模型中的 leading lag/amplitude mode 在两折间是否稳定；报告不会将其称为完整物理核、因果对象或已恢复工厂机理。

## 数据范围、定义与执行边界

- 数据：Furnace A / Sheet1，仅前 80% development 区间。
- 输入：主加热功率、晶升速度、晶转速度、埚升速度、埚转速度。
- 目标：晶体直径；history 只使用预测原点及之前的数据。
- 固定任务：`(Lx,Ly,h)=(64,16,1),(256,32,15),(512,64,60)`。
- 固定两折：0–50%/50–60% 与 0–70%/70–80%，并应用`max(Lx-1,Ly-1)+h` purge。
- Furnace B、完整 ORSS、R3 全搜索、confirmation 均未执行。

## 方法与数值检查

- FAST-A：train-only 多目标 ridge 条件能量，moving-block bootstrap。
- FAST-B：线性 AR 残差的滞后相关、block bootstrap 与 block permutation。
- FAST-C：`Mtau=Mx=8` 粗特征经 AR 条件化后的 Gram/Schur 谱。
- FAST-D：Persistence、AR、ridge-ARX 的固定任务比较。
- FAST-E：`Mtau=Mx=8`、共享 penalty path 的粗非线性 XAR。
- FAST-F：跨折 Q contribution 与加权 SVD leading K mode 稳定性。

## 局限性与稳健性边界

- 当前只有单根晶棒的一段闭环运行轨迹；不支持跨晶棒、跨炉次或跨阶段泛化结论。
- 未知采样周期和设备端滤波/延迟，因此 lag 只以采样步报告，不能直接解释为热传播时间。
- 本轮 coarse resolution 和固定 penalty path 仅用于路线判定，不能代替完整超参数冻结和锁箱评估。
- 任一 `AUDIT_INCOMPLETE` 都意味着证据未形成闭合链，而不是阴性结论。

## 建议的下一步

严格只进入状态文件允许的阶段：`NONE_AMBIGUOUS_FAST_EVIDENCE`。若该值为 `NONE_*`，应先解决运行门禁或证据歧义，不得自动恢复完整 K 搜索。

## 仍需回答的问题

- 增加晶棒和工况后，Q 增量与低阶 K mode 是否保持跨轨迹稳定？
- 取得采样周期、测量滤波和控制回路元数据后，lag 能否获得受限的物理解释？
- 若继续完整实验，冻结模型能否在未参与选择的 confirmation 区间保持同方向收益？

生成时间（UTC）：`2026-07-29T15:33:43.364705+00:00`  
Furnace A SHA256：`c46e0d35d26903386fd80408f36660c4f8925a5dbc56c92527f020e433ef04de`  
执行器：`dense_batched`
