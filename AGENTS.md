# 项目持久约束：AR-RAPHU v2/v3 历史基线与 Spectral Predictive-State v0.3

本文件是本项目后续 Codex 会话必须遵守的项目记忆。开始任何实现、数据处理、下载或实验前，必须完整阅读：

1. 本文件；
2. `Spectral_PS_AR_RAPHU_Theory_v0_3.md`；
3. `Spectral_PS_AR_RAPHU_Validation_Plan_v0_3.md`；
4. `PS_AR_RAPHU_V3_D1_D6_Diagnostic_Execution_Plan.md`；
5. `AR_RAPHU_method_v2.md`；
6. `AR_RAPHU_three_layer_validation_plan_v2.md`；
7. `AR_RAPHU_v2_revision_notes.md`。

当前活动规范是 `Spectral_PS_AR_RAPHU_Theory_v0_3.md` 与
`Spectral_PS_AR_RAPHU_Validation_Plan_v0_3.md`。对 Spectral v0.3 的模型、
目标、文件、数值、执行、判定、输出或停止条件有冲突时，一律以验证计划为
最高优先级、理论文档为语义边界；未规定处才沿用本文件和 v2/v3 历史基线。
v3 D1--D6 已完成并作为只读诊断证据保留，不再是活动执行合同。不得用旧 v2
支持门控、M7/M8 路线或 v3 最小诊断实现覆盖 Spectral v0.3。

旧文件 `rank_adaptive_parallel_hammerstein_urysohn_method.md` 与
`three_layer_validation_plan.md` 仅用于追踪 v1 设计演化，不再作为当前执行规范。
若 v1 与 v2 冲突，一律以 v2 为准。

## 用户授权边界

- 当前只可严格按照 Spectral v0.3 的 E0--E8 执行合同推进；v2 停止线和
  v3 D1--D6 作为只读科学与实现基线。不得跳过进入条件、扩展实验或把
  计划/期望写成结果。
- 对文档未规定但执行所必需、且不同选择会实质影响结果的事项，不得自行决定；暂停并询问用户。
- 质量和可复核性优先于速度。任何加速不得改变科学模型、数据边界、数值精度要求或统计方案，并须通过等价性/一致性测试。
- 结果状态统一使用：`COMPLETED`、`FAILED`、`NOT_YET_RUN`、`NOT_APPLICABLE`、`BLOCKED_BY_MISSING_METADATA`、`BLOCKED_BY_MISSING_DATA`。

## 当前已确认的 CZ 数据事实

- 当前数据来自 **一根 CZ 法硅单晶晶棒**，只覆盖 **等径阶段**，是一条连续轨迹。
- 私有源文件为 `实验数据1.xlsx`；最后一列 `晶体直径` 是输出 \(y_t\)。
- 九个外生过程变量依次为：
  `加热元件温度`、`主加热功率`、`晶升速度`、`晶转速度`、
  `埚升速度`、`埚转速度`、`氩气流量设定`、`晶体长度`、`炉压`。
- 历史模型的第十个通道不是新传感器，而是历史输出：
  \(y_{t-1},\ldots,y_{t-L_y}\)。因此：
  - 外生过程变量数 \(p_x=9\)；
  - 模型接口通道数为 10；
  - 当前标签 \(y_t\) 绝不能进入预测 \(y_t\) 的输入；
  - 只允许严格滞后的输出进入 AR 通道。
- 历史复现轨道固定 \(L_x=L_y=32\)，只用于与旧实验兼容，不解释为物理合理或统计最优窗口。
- 正式 AR-RAPHU 轨道不强制 \(L_x=L_y\)：
  - \(L_y\in\{1,4,8,16,32,64\}\)；
  - \(L_x\in\{32,64,128,256\}\)；
  - 只有当 256 在开发验证中仍明显优于较短窗口时，才追加测试 \(L_x=512\)；
  - 不得从一开始把 512 放入全模型搜索。
- 窗口选择顺序固定为：
  1. 在开发折用 AR-only 选择 \(\widehat L_y\)；
  2. 在开发折用 X-only 选择 \(\widehat L_x\)；
  3. 组成 X+AR 后只允许一次小范围邻域复核，不做完整 \(L_x,L_y\) 笛卡尔积搜索；
  4. 打开 Fold 4 test 前冻结一组全局 \((L_x^\star,L_y^\star)\)；
  5. M3–M10 对各自适用通道使用同一组冻结窗口。
- `氩气流量设定` 在当前晶棒中为常数：
  - 数据接口保留该列；
  - 科学训练中屏蔽、固定为零贡献或明确标记为不可辨识；
  - 不得报告其响应函数、时滞核或 rank 为有效发现。
- 当前实际可学习通道上限是 8 个变化外生变量 + 1 个历史输出通道。
- 当前只能声称单晶棒、等径阶段内的时间外推；不得声称跨晶棒、跨炉次、跨阶段泛化或工业部署可靠性。

## 尚未取得但不得猜测的元数据

以下统一标记为 `NOT_YET_AVAILABLE`：

- 各列准确工程单位；
- 物理采样周期；
- 晶体直径测量方法；
- 设备端滤波、移动平均、估计或延迟补偿；
- 直径信号固定延迟；
- 传感器/执行器安装位置；
- 过程量的设定值、控制输出或反馈量分类；
- 数据是否经过人工截取、去异常或重采样；
- 更多晶棒及其炉次 ID。

这些缺失项不阻止以“采样步”为单位开展明确允许的实验，但禁止把有效预测时滞解释为纯热传播或物质输运时间，也禁止把 \(y_t\) 等同于未观测的真实几何直径。

## 私有 CZ 数据与防泄漏硬约束

- `实验数据1.xlsx` 不得上传、外发、提交到公开仓库或混入公开产物；只允许本地只读处理。
- 冻结 SHA256：
  `c46e0d35d26903386fd80408f36660c4f8925a5dbc56c92527f020e433ef04de`
- 必须先冻结原始时间轴上的目标区间，再按目标索引 \(t+h\) 判定样本属于 train/validation/test。
- 对预测原点 \(t\)，输入严格定义为：
  - \(X[t-L_x+1:t+1]\)；
  - \(y[t-L_y+1:t+1]\)；
  - 目标为 \(y[t+h]\)，\(h>0\)。
- 输入历史允许跨越目标分区的左边界，使用该目标发生前已经观测到的 train/validation 历史；这符合真实在线预测，不是泄漏，不机械丢弃边界 \(L-1\) 个样本。
- 严格禁止使用任何索引大于 \(t\) 的 X 或 y、禁止把目标 \(y[t+h]\) 放入输入。必须用单元测试证明：
  \(\max(\text{input index})=t<t+h=\text{target index}\)。
- 标准化、目标缩放、插补规则、异常阈值、样条结点、经验 Gram 矩阵、特征选择和其他数据驱动预处理只在每个外层折的训练段拟合，再原样应用于 validation/test。
- 插值必须因果，不能跨边界；缓存、索引、窗口和派生产物必须带 dataset/track/horizon/fold/partition 身份。
- validation 只用于预先允许的选择；test 在配置、阈值和结论规则冻结后只读评估一次。
- CZ 半合成与真实目标分开保存，不覆盖原始目标，不共享会泄漏真值的缓存。
- 公开数据与私有 CZ 使用不同目录、manifest、缓存和结果命名空间。

### CZ 四折协议（0-based、左闭右开）

| Fold | Train | Validation | Test | 角色 |
|---|---|---|---|---|
| 1 | `[0,10051)` | `[10051,12061)` | `[12061,14072)` | 开发 |
| 2 | `[0,12061)` | `[12061,14072)` | `[14072,16082)` | 开发 |
| 3 | `[0,14072)` | `[14072,16082)` | `[16082,18092)` | 开发 |
| 4 | `[0,16082)` | `[16082,18092)` | `[18092,20103)` | 最终锁箱 |

- Fold 1–3 用于窗口、正则、KAN 容量、rank 阈值、优化、早停和结构稳定性开发。
- Fold 4 validation 可按预注册流程用于最终冻结，但 Fold 4 test `[18092,20103)` 在全部模型、窗口、阈值和超参数冻结前不得读取。
- 每折 scaler、分位数、样条结点和异常阈值只拟合该折 train 原始区间；validation/test 仅 transform。
- 最终主结论的 Fold 4 test 访问必须由显式 `protocol_frozen=true` 守卫。

## AR-RAPHU v2 科学模型

必须分开建立三条轨道：

- `Track-X`：只使用九个过程变量；
- `Track-AR`：只使用历史直径；
- `Track-XAR`：同时使用过程变量和严格滞后的历史直径。

所有可提供历史顺序的数据集都必须报告：

\[
\Delta_{\mathrm{X\mid AR}}(h)
=
\operatorname{Loss}_{AR}(h)-\operatorname{Loss}_{XAR}(h).
\]

一步预测必须至少比较 persistence、线性 AR、AR-only、X-only 和 XAR。主多步协议固定为 direct forecast：

\[
\widehat y_{t+h\mid t}
=F_h\!\left(X_{t-L_x+1:t},y_{t-L_y+1:t}\right),
\quad
h\in\{1,5,10,30,60\}.
\]

- 所有 horizon 使用相同信息截止时刻 \(t\)；
- 不使用未来 \(X_{t+1:t+h}\)；
- 不使用中间真实 \(y_{t+1:t+h-1}\)；
- 第一版使用独立模型或共享 backbone 的独立 horizon head；
- recursive forecast 只作稳定性辅助实验，不作为主结果。
- 仅 TEP 可增加单独标记的 `Known-future-X / Oracle-X` 上界轨道；不得与无未来 X 主协议混合，也不得用于当前 CZ 主结论。

方案 A 为带自回归输出分支的并联 Hammerstein 主模型。自回归分支代表输出持续性、未观测状态代理、闭环记忆及可能的测量平滑/延迟，不解释为独立工艺机制。

方案 B 第一版只升级外生过程分支：

- 固定方案 A 的外生支持、外生核和 AR 核；
- 进行外生一维凸样条重建；
- 只对外生变量拟合 A 锚定正交二维残差；
- 使用 Gram 白化 SVD 审计外生 rank；
- 历史输出分支保持低复杂度 rank-1，不计算或解释主要科学结论 \(\eta^y\)；
- 多步递归必须审计自回归收缩系数 \(\kappa_y\)，必要时对 \(f^y\) 加导数上界或 Lipschitz 正则。

Gamma 错设必须用自由平滑动态核 rank-1（M6）与真实非可分离性区分。rank-2 或完整面只对通过稳定性、bootstrap、外层收益和数据密度审计的外生变量启用。

## 统一模型与合成场景

模型矩阵以 v2 为准：

- B0 Persistence；
- B1 线性 AR；
- B2 线性 ARX；
- M1 经典样条/FIR 并联 Hammerstein；
- M2 Stable-Spline/Kernel Hammerstein；
- M3 方案 A X-only；
- M4 方案 A AR-only；
- M5 方案 A X+AR；
- M6 自由平滑动态核 rank-1；
- M7 A 锚定一维凸样条重拟合；
- M8 A 锚定正交 B 残差；
- M9 rank-2 外生并联 Hammerstein；
- M10 完整外生 Urysohn 面；
- M11 TCN、GRU/LSTM 与 Transformer 类深度基线。

Layer I 至少覆盖 AR-S0 至 AR-S7：纯 AR、Gamma rank-1、非 Gamma rank-1、rank-2、幅值依赖时滞、测量滤波、测量延迟、闭环控制与共线。

## 强制执行顺序

1. Phase 0：冻结代码和语义；审计第十通道对齐及标签泄漏；按已冻结候选协议选择 \(L_x/L_y\)；落实四折、direct multi-horizon、无未来 X、Fold 4 锁箱与 train-only scaler；继续冻结 Gamma 离散化和 KAN 结构。
2. Phase 1：运行 AR-RAPHU 合成真值 E1–E4，并扩展覆盖 AR-S4 至 AR-S7；合成层完成前不冻结最终 rank 阈值。
3. Phase 2：TEP 公开长期数据与主动干预；先现成数据，再 \(5\times10^6\)，证据成立后才扩展到 \(2\times10^7\) 时间步。
4. Phase 3：Debutanizer 与 UCI Gas Turbine；先审计时间顺序和标签延迟。若 Debutanizer 时间顺序不可靠，不用于严格动态 rank 证据。
5. Phase 4：最后运行当前单根 CZ 的 E9/E10；公共实验规则冻结后才读取 CZ 外层结果。
6. Phase 5：只有取得更多晶棒后才追加 grouped/leave-one-rod-out 验证，不重写先前公共实验规则。

当前单根 CZ 只允许 expanding-window、后段时间外推、AR/X/XAR、多 horizon、同阶段稳定性、block bootstrap 和半合成桥梁。

## 运行环境与加速

- 所有 Python 命令、脚本、测试和训练统一使用 Anaconda 环境 `Env_pytorch`：
  `conda run -n Env_pytorch --no-capture-output python ...`
- 缺包只允许 `conda install` 到 `Env_pytorch`；不得使用系统 Python或 `pip`。涉及 channel/版本且会影响复现时先询问用户。
- 已确认环境：Python 3.10.18、PyTorch 2.8.0.dev20250611+cu128、CUDA 12.8、cuDNN 9.8、RTX 5080 Laptop GPU 16 GB、计算能力 12.0。
- 适合 GPU 的训练、卷积、样条批处理和 bootstrap 使用 CUDA；CPU 密集任务使用受控多线程/多进程并记录并行设置。
- 可在明确热点使用 Rust，但必须先有正确参考实现，并通过预测、梯度、边界、dtype 和 CPU/GPU 一致性测试。
- 优先任务级并行；只有规模需要时使用 DDP。必须记录正确性、峰值内存、吞吐和墙钟时间。

## 公开数据与复现

- 下载公开数据时记录 URL、版本/commit、许可证、下载时间、原始文件名、字节数和 SHA256；原始文件只读，派生数据另存。
- TEP 先读取表头再冻结列映射；不得假定长期数据的全部列都是过程变量。
- Debutanizer 必须先确认时间顺序、采样周期和实验室标签延迟；不可靠时只作静态/滞后特征对照。
- Gas Turbine 按年度外推，作为外部泛化补充，不承担主动干预 rank 真值。
- 配置由单一 YAML 驱动；训练与测试命令分离；聚合和绘图不重新训练。

## Phase 0 当前状态

- 已有只读导出工具 `tools/export_phase0.py`、`data_manifests/cz/` 清单和 `tests/test_phase0_manifests.py`。
- v1 清单已按 v2 和正式窗口/四折/direct-forecast 协议重新生成。
- 冻结索引参考实现为 `src/ar_raphu/data_protocol.py`；协议测试为 `tests/test_data_protocol.py`。
- 当前项目根目录 Phase 0、协议、V20 契约、双分支适配、train-only
  预处理、懒窗口、合成生成器、sequence-first 训练与 Scheme A 测试
  均须在每次正式实验前完整回归；原版 V20 包内回归测试基线为
  118 passed。
- 已确认源工作簿为 20,103 条记录、10 列、无公式、无空值；原始 SHA256 未改变。
- 当前清单已明确：
  - `external_process_channels=9`；
  - `historical_output_channels=1`；
  - `model_input_channels=10`；
  - `current_target_used_as_input=false`；
  - `strictly_lagged_target_used_as_input=true`；
  - `crystal_count=1`；
  - `stage=constant_diameter_only`；
  - 氩气常量分支不可辨识；
  - 原 `BLOCKED_PENDING_EXACT_ALIGNMENT_AND_ROLLING_FOLDS` 状态已解除；
  - `split_manifest.json` 当前 `protocol_status=FROZEN`，包含精确四折、目标归属、窗口候选、Fold 4 锁箱及无未来 X 规则；
  - 尚未物化任何私有数据窗口，也未访问 Fold 4 test 目标值。

## V20 历史实现冻结与迁移边界

- 原版实现位于 `STAGE1_DUAL_SOLVER_V20_bundle/`。随包 `SHA256SUMS.txt` 的 64 个文件已全部验证，原版三组回归测试当前为 118 passed。
- 历史复现固定使用 V20 Fast-KAN 语义：
  - 静态 Gamma，`epsilon=0`；
  - \(\tau=0,\ldots,L-1\)，顺序为 current-to-past；
  - 点值离散后 softmax 归一化；
  - \(\delta=10^{-3}\)；
  - \(\beta\) 是 scale，不是 rate；
  - 默认以 bounded mean/std 学习，再换算并截断
    \(\alpha\in[0.5,10]\)、\(\beta\in[0.1,L]\)；
  - 每变量 KAN 为 `1 -> 8 -> 1`，grid 7、三次样条、SiLU base；
  - 第一层 grid 仅由 train min/max 加 5% padding（最小 0.05）确定，第二层固定 `[-3,3]`，不动态更新 grid；
  - sequence-first 向量化 KAN 与 grouped causal convolution 是数学等价加速。
- 正式 AR-RAPHU 的容量审计候选从 V20 已审计范围冻结：
  - `hidden_kan in {4,8}`；
  - `grid_size in {5,7,11}`；
  - `spline_order=3` 固定；
  - 历史复现配置固定 `hidden_kan=8, grid_size=7`。
- 可直接继承：Gamma/KAN 数学实现、精确序列算子、共享 warmup 后独立剪枝分叉、跨种子 validation-only one-SE、固定时滞凸样条部件。
- 不得直接继承：
  - V20 合成数据的 `horizon=0` 对齐；
  - 31 点 embargo split；
  - `real.yaml` 的 6000:15000 与 8.5:1:0.5；
  - 所有十通道共用同一个 `max_lag`；
  - 十个分支完全相同的科学解释；
  - 未屏蔽常量氩气的 all-active mask。
- bundle 不含 `process_data.py`、真实 CZ 第十通道构造或 scaler 实现。因此正式数据管线必须使用本项目冻结的目标索引协议重建。
- 正式模型必须使用独立外生/AR 分支，以支持 \(L_x\ne L_y\)；AR 分支保持 rank-1，外生常量氩气屏蔽，B 只作用于外生分支。
- 主 direct-forecast 实验不递归输入预测值，因此递归 Lipschitz/收缩约束不阻塞主协议；若运行 recursive 辅助实验，须另行冻结该约束。

## V20 双分支适配实现状态

- `src/ar_raphu/model.py::ARRAPHURank1` 已作为 V20 的协议适配层实现：
  - `Track-X` 只实例化 9 通道外生 V20 内核；
  - `Track-AR` 只实例化 1 通道历史输出 V20 内核；
  - `Track-XAR` 同时实例化二者，允许 \(L_x\ne L_y\)；
  - 每个 direct horizon 是独立模型实例，horizon 仅允许
    `{1,5,10,30,60}`；
  - 子内核偏置冻结且不进入输出，唯一可学习截距由适配层持有；
  - 第一层 KAN grid range 必须由调用方显式提供，禁止退回默认范围，
    从而强制后续数据管线只用对应 fold train 拟合；
  - 当前 CZ 的氩气通道索引 6（0-based）固定为 inactive，响应贡献、
    输入梯度和 Gamma 参数梯度均为零；
  - 外生与 AR 的 10 个分量贡献必须精确闭合到预测加唯一偏置。
- `tests/test_ar_raphu_model.py` 已验证：
  - 不同 \(L_x/L_y\)；
  - X/AR/XAR 轨道参数与输入隔离；
  - V20 Gamma 核逐行归一化；
  - 氩气硬屏蔽；
  - 贡献闭合；
  - CPU legacy 与 CPU/CUDA vectorized 的预测、贡献和输入梯度一致性。
- `data_manifests/cz/model_semantics.json` 的 `v2_adapter` 已记录上述实现
  及源码/测试 SHA256；当前状态只表示结构适配完成并通过单元测试，
  不表示任何训练或科学实验已经运行。

## Phase 0 尚需处理但无需用户补充的实现项

以下实现已完成：

1. 用 V20 组件建立支持 \(L_x\ne L_y\) 的双分支 AR-RAPHU 封装；
2. 为严格目标对齐、氩气屏蔽、贡献闭合和不同窗口长度补齐单元测试；
3. `src/ar_raphu/preprocessing.py` 提供只读取对应 fold train 原始区间的
   scaler 与 V20 grid-range 拟合；
4. `src/ar_raphu/dataset.py` 按目标索引懒构造窗口，缓存身份包含
   privacy/dataset/track/horizon/fold/partition/Lx/Ly；
5. 开发折数据集不会对 Fold 4 数值执行完整性扫描，Fold 4 test 在样本
   对象产生前即被锁箱守卫拒绝；
6. `configs/protocol_v2.yaml` 是单一 JSON-compatible YAML 配置源，
   `src/ar_raphu/protocol_config.py` 在 Phase 1 预注册字段未冻结时拒绝
   启动实验。

## Phase 1 预注册与启动状态

- 用户已于 2026-07-25 明确授权启动；`configs/protocol_v2.yaml` 当前
  状态为 `PHASE1_PREREGISTERED`，此前所有 Phase 1 预注册阻塞项已冻结。
- 用户于 2026-07-25 明确要求抛弃当前私有 CZ 数据集；Phase 4 CZ 与
  Phase 5 多晶棒轨道均标记为 `SKIP`。后续不得读取、训练、评估或打开
  私有 CZ Fold 4；剩余执行范围仅为 Phase 1 合成、Phase 2 TEP 和
  Phase 3 Debutanizer/Gas Turbine。
- `data_manifests/cz/` 自此作为不可变的历史审计快照保留；后续合成和
  公开数据代码变更不得通过重新读取私有工作簿来刷新其中的源码哈希。
  Phase 0 测试在确认 `private_CZ=EXCLUDED_BY_USER_2026-07-25` 后只校验
  该快照结构和已有校验和，不再把后续公开代码绑定到 CZ 快照哈希。
- 冻结内容包括 train-only z-score、Phase 1 缺失值 fail-fast/异常值
  仅报告、screening 10 seeds、critical 30 seeds、Adam 0.003、
  full-train 有效 batch、FP32 神经训练、FP64 凸求解和统计、
  deterministic、V20 warmup/prune/refit 预算与 validation-only one-SE。
- AR-S0--AR-S7 的 core 与边界样本量、变量数、支持、稳定 AR 真值、
  SNR、连续 60/20/20 目标切分和分阶段因子扩展均已冻结。
- 支持/rank 假阳性上限 10%、rank-2 power 下限 80%、alpha/FDR 0.05、
  screening/formal bootstrap 100/500、残差 ACF 块长及最小外层收益均已
  冻结。合成层结束前不得根据结果更改这些规则。
- `src/ar_raphu/synthetic.py` 已实现 AR-S0--AR-S7；
  `src/ar_raphu/sequence_data.py` 与 `src/ar_raphu/training.py` 已实现
  sequence-first、全训练集梯度累积、warmup、组近端剪枝和固定支持 refit。
- `tools/run_phase1_scheme_a.py` 将 warmup、每个 penalty 分叉与聚合测试
  分为独立命令；聚合前只读 validation，选定全局 penalty 后才允许读
  test。`--smoke` 只验证流程，不得作为科学证据。
- AR-S0 Scheme A CPU 冒烟已验证 dense、共享 warmup、独立分叉、
  validation-only one-SE 以及选择后测试的完整执行链。此处是当时状态；
  v2 最终状态以本文末尾“v2 停止线最终状态”为准。
- 2026-07-25 正式 E1 的首轮生成器审计发现 AR-S0 因全零初值且没有持续
  过程激励而恒为零，按 clean variance 缩放的观测噪声也随之为零。
  `results/phase1/E1_AR-S0/` 已写入 `INVALID_GENERATOR_AUDIT.md`；
  其中已有 B0/B1 与 seed 0--3 warmup 全部隔离为无效预跑，不得选参、
  聚合或报告。修正必须先由用户冻结 AR 过程激励定义，并使用新结果
  namespace，不能覆盖这些审计文件。
- 用户随后授权直接继续；合成生成器冻结为 version 2：AR-S0--AR-S7
  每个生成时刻统一加入独立 `N(0, 0.2^2)` 潜在过程创新，且与按 clean
  target variance/SNR 生成的测量噪声分离。所有修正后结果必须写入带
  `_G2` 的新 namespace。
- sequence-first CUDA 基准（n=10,000、X+AR、10 epochs）约 2.15 秒、
  峰值显存约 220 MB；该加速必须继续保持 legacy/vectorized 数值等价测试。
- 用户于后续继续指令中确认此前建议的 M6--M9 冻结网格：
  - M6 从 M5 Gamma 核初始化，logit 二阶差分权重
    `{0,1e-4,1e-3,1e-2}`；
  - M7 幅值三次样条 grid `{8,12,16}`、平滑权重 `{0,1e-4,1e-3}`；
  - M8 固定 M7 后顺序选择时滞 grid `{5,8,12}` 与残差平滑权重
    `{1e-4,1e-3,1e-2}`，不做完整笛卡尔积；
  - M9 使用 M8 Gram 白化面的前两项 SVD 重构，不新增超参数。
- 用户随后进一步冻结 M7/M8 的 one-SE 与稳定性规则：
  - v2 编号不可混用旧版：M7 是 A 锚定一维凸样条重拟合，M8 是只升级
    外生过程分支的 A 锚定正交二维残差；
  - M7 必须把全部 `(amplitude_grid, smoothing_weight)` 联合放入
    validation-only one-SE；候选集内先选更小幅值 grid，再选更大平滑
    权重，最后按预声明配置顺序；
  - M8 冻结 Scheme A 支持/时滞及 M7 幅值 grid，以 `1e-3` 为 pilot
    平滑对 lag grid 做 one-SE 并优先更小 grid；
  - 必须在 `1e-4,1e-3,1e-2` 检查 lag-grid one-SE 稳定性，所选 grid
    至少进入三个集合中的两个；否则设置
    `M8_GRID_SMOOTHING_INTERACTION=TRUE`，对已预声明的完整
    lag-grid × smoothing grid 做联合 one-SE 回退；
  - lag grid 冻结后对平滑权重做 one-SE，候选集内优先更强平滑；
  - M8 当前只有一个共享二维平滑权重；不得运行后拆分或新增方向权重；
  - 超参数只由 validation prediction loss 冻结。SVD、rank 显著性、
    test、真函数和物理外观均不得参与选择；固定模型后才能进行
    Gram 白化 SVD 与 bootstrap rank 审计；
  - Phase 1 合成数据没有外层 expanding-window folds，one-SE 以独立
    seed replicate 为统计单位；对以后存在外层折的数据，先折内平均
    seeds，再在 folds 间计算标准误，绝不按时间点计算 SE。
- 用户明确指出 CPU 存在过热降频风险。自 2026-07-25 起所有新任务池
  固定 `workers_per_device=2`，每进程 PyTorch intra-op/interop、
  OMP、MKL、OpenBLAS、NumExpr 线程数均为 1；不得因 GPU 尚有余量提高
  CPU 并发。必须在 runtime/telemetry 中记录这些限制。
- 上述低温限制只适用于原本地 RTX 5080 Laptop 工作站。用户随后明确
  授权把后续任务迁移到 AutoDL：RTX 5090 32 GB、25 核 Xeon Platinum
  8470Q、90 GB 内存、驱动 595.71.05、宿主 CUDA 13.2。该服务器使用
  `deploy/autodl/runtime_profile.json` 的独立运行配置：
  - 环境由 `uv` 管理，显式设置 `AR_RAPHU_RUNTIME_MANAGER=uv`；这是对
    本节 Conda 规则的仅限 AutoDL 例外，不改变本地仍须使用
    `Env_pytorch` 的要求；
  - 默认 1 块 GPU 上 8 个任务进程，每进程 3 个 intra-op/BLAS 线程和
    1 个 inter-op 线程，总 CPU 线程预算不超过 24；
  - 必须成功启动并验证 NVIDIA MPS 后才允许训练，MPS pipe/log 使用
    每用户独立的 `/tmp` 目录；
  - 服务器 manifest 必须在目标 uv 环境内重新生成，禁止沿用含本地
    Conda Python 绝对路径的 manifest；
  - 迁移前必须排除 `实验数据1.xlsx` 和 CZ 数据目录；只携带 Phase 1
    合成/公开数据代码、E1 证据及 E2 可复用断点；
  - 首次训练前必须在 5090 上通过项目公共回归、V20 118 项回归、CUDA
    vectorized/legacy 等价性及 MPS 健康检查；
  - MPS、任务并行或 CPU 并发只可改变吞吐，不能改变确定性、种子、
    FP32/FP64、训练预算、早停、validation-only 选择或 test 访问顺序。
- 仍不得为 recursive 辅助实验或其他未冻结扩展自行增加结构。

## v2 停止线最终状态（2026-07-25，冻结只读）

- 用户已停止 v2 后续训练与审计；不得恢复 S0 critical 剩余任务、S1--S3
  critical、bootstrap、M7/M8 审计或任何 Phase 2/3/4/5 工作，除非用户
  后续明确重新授权。
- AR-S0--AR-S7 screening 已完成。S0 critical 留有 20/20 warmup 和
  55/180 fork，但没有完成选择；这些仅为停止线前的部分计算，不得作为
  完整科学证据。
- S1--S3 critical 为 `NOT_YET_RUN`；bootstrap 为 `NOT_YET_RUN`。
- v2 停止线累计实际优化 epoch 为 `2,476,482`。这是计算量记录，不代表
  证据充分性。
- v2 科学结论状态为 `FAILED`；M9/M10 为 `NOT_APPLICABLE`，Phase 2/3
  为 `NOT_YET_RUN`，CZ 与多晶棒轨道为 `NOT_APPLICABLE`。
- v2 停止线源码、结果、报告和打包产物只作为不可变历史证据保留。不得
  将未完成阶段重新解释为通过，也不得为 v3 诊断覆盖或改写其结果目录。

## v3 D1--D6 的唯一科学问题与执行边界

当前活动任务是 Predictive-State AR-RAPHU v3 的最小诊断，只回答：

> 强 AR 条件下，为什么外生支持与 rank 恢复会失败？

D1--D6 不是完整 v3、不是 v2 续跑、不是最终模型选择，也不是公共数据或
CZ 验证。任何超出这一问题的实现、训练、解释或产物均需用户另行授权。

v2 停止线源码基线为：

- `AR_RAPHU_STOPLINE_20260725.zip/source/`；
- `STAGE1_DUAL_SOLVER_V20_bundle/`；
- `tools/run_phase1_scheme_a.py`；
- `tools/run_phase1_m6.py`；
- `tools/run_phase1_m7.py`；
- `tools/run_phase1_m8.py`；
- `configs/protocol_v2.yaml`。

上述内容在 v3 D1--D6 中一律只读，不得就地修补。必须复用现有合成
生成器、数据协议、模型、训练、证据以及 V20 KAN/Gamma 组件；不得重写
scaler、窗口、sequence-first convolution、KAN、Gamma 或 AR 分支。

### v3 允许创建的文件

只允许新增：

```text
configs/v3_diagnostics.yaml
src/ar_raphu/diagnostics/__init__.py
src/ar_raphu/diagnostics/config.py
src/ar_raphu/diagnostics/rank2_model.py
src/ar_raphu/diagnostics/residual_data.py
src/ar_raphu/diagnostics/train_utils.py
src/ar_raphu/diagnostics/truth_metrics.py
src/ar_raphu/diagnostics/gate_fista.py
src/ar_raphu/diagnostics/instrumentation.py
tools/run_v3_diagnostic_job.py
tools/run_v3_diagnostic_suite.py
tests/test_v3_rank2_model.py
tests/test_v3_residual_data.py
tests/test_v3_gate_fista.py
tests/test_v3_instrumentation.py
```

唯一允许的兼容性修改是向 `src/ar_raphu/synthetic.py` 暴露只读包装：

- `truth_response`；
- `second_truth_response`。

包装只能读取既有真值响应，不得改变生成器逻辑、随机数消费顺序、场景
定义、噪声或目标。

## v3 冻结配置：禁止 CLI 覆盖或运行后修改

### 数据、模型与运行

- 通用 seeds：`0,1,2,3,4`；D6 seeds：`0,1,2`。
- `n=10000`，外生变量数 `10`，真支持 `[0,1,2]`。
- `Lx=64`，`Ly=32`，主 horizon `h=1`；只有 D3 的预声明失败分支可加
  `h=5,10`。
- `hidden_kan=8`，`grid_size=7`，`spline_order=3`，
  `vectorized=true`，`chunk_size=4096`，deterministic FP32。
- runtime：一块 GPU 默认 `workers=8`，OOM 后只允许回退为 `4`；
  每任务 CPU/BLAS 线程为 `1`；使用 MPS。

### 统一优化

- validation 间隔 `10` epochs。
- response learning rate `0.003`，lag learning rate `0.0005`，
  AR learning rate `0.003`，joint learning rate `0.0003`。
- lag smoothness `0.001`，minimum learning rate `1e-5`。
- scheduler factor `0.5`，patience `20` 次 validation。
- 全训练集梯度累积；只以 validation RMSE 早停。
- 禁止用 test 早停、用 proximal/support/rank 早停、临时增加 epoch、
  重启失败 seed 或按结果换初始化。

### 各诊断预算

- D1：oracle `2000` epochs，free `3000` epochs，patience `300`；
  固定两个分量权重 `0.6/0.4`。
- D2：`3000` epochs，patience `300`。
- D3：AR `2000` epochs，residual `2500` epochs，patience `250`。
  若至少 4 个 seed 的 innovation \(R^2<0.10\)，只允许追加 `h=5,10`。
- D4：simultaneous `2500`；X-first `1500+500+500`；
  AR-first `1000+1500+500`；patience `250`。
- D5：固定使用 D4 X-first 的贡献设计；lambda ratio 路径严格为
  `.32,.16,.08,.04,.02,.01,.005,0`；FISTA `10000` iterations，
  tolerance `1e-9`，support threshold `1e-8`。
- D6：warmup `2000`，prune `1200`，penalty scale `0.003`，
  ramp `300`，每 `10` epochs 记录，sample `1024`；starvation 阈值为
  gradient ratio `0.10`、contribution ratio `0.05`、连续 `5` 次、
  shrink ratio `0.99`。

## D1--D6 固定职责与顺序

必须严格按 `D1 -> D2 -> D3 -> D4 -> D5 -> D6` 推进，D5 必须等待 D4
X-first 产物。不得用后续结果回改前序配置。

- D1：AR-S3、无 AR、无稀疏，分离容量失败与 lag 优化失败。
- D2：AR-S3，对照自由 rank-1 与 rank-2，检查 rank-1 盲点。
- D3：AR-S3，先冻结 AR，再在 innovation residual 上拟合外生分支。
- D4：AR-S1，对照 simultaneous、X-first、AR-first，检查训练顺序捷径。
- D5：AR-S1，固定 D4 X-first contribution design，运行凸 Lasso gate path。
- D6：AR-S1，记录 gradient/prox 时间线，判断 starvation。

D1 rank-2 模型必须恰好由两个独立的 `ARRAPHURank1(Track-X)` 分量组成，
仅使用真支持 `[0,1,2]`，权重固定为 `0.6/0.4`，并遵守计划中的唯一
bias 规则。q-mode 只允许 `oracle_fixed` 和 `free_truth_init`。禁止
attention、MoE、跨变量交互、额外 MLP 或其他结构。response、lag、AR
使用分离的 optimizer parameter groups，并保留冻结的 lag smoothness。

## v3 冻结判定规则

所有判定必须由程序按计划生成，不得由 Codex 目测、补充解释或调整阈值。

- D1A 通过：至少 4/5 seeds 同时满足 validation \(R^2\ge0.95\)、
  mean surface NRMSE \(\le0.20\)、surface correlation \(\ge0.90\)。
- D1B 通过：至少 4/5 seeds 同时满足 RMSE 不超过 oracle 的 `1.10` 倍、
  mean lag W1 \(\le3\) samples、surface NRMSE \(\le0.25\)。
- D1 只允许产生 `capacity pass`、`capacity fail`、`lag optimization fail`
  等计划规定标签。
- D2：至少 4/5 seeds 的 surface gap \(\ge0.30\)，且 RMSE gap
  \(\ge0.05\) 或 \(|RMSE gap|<0.05\)，才确认 rank-1 blind spot。
- D3：以 median innovation \(R^2=0.10\) 为冻结分界；先判 h=1，只有
  预声明失败分支才运行 h=5/10。
- D4：至少 4/5 seeds 中，X-first 相对 simultaneous 的外生 energy
  fraction 增加 \(\ge0.20\)，response NRMSE 改善 \(\ge20\%\)，且
  validation RMSE 退化不超过 `2%`，才确认 optimization shortcut。
- D5：至少 4/5 seeds 的路径上存在某个 lambda 同时满足 recall
  \(\ge0.80\)、FPR \(\le0.10\)，才判设计中存在可恢复点。D5 只报告
  整条 path，禁止据此选择 lambda。
- D6 使用上述冻结 starvation 阈值；计划允许的标签可并存。

最终必须自动生成：

- `diagnostic_summary.csv`；
- `DIAGNOSTIC_DECISION.md`。

两者只能包含 v3 计划规定的九个字段及冻结 failure mapping。Codex 不得
在自动判定中添加新标签、因果故事、人工裁决或扩大结论。

## v3 CLI、检查、产物与禁止事项

### CLI

单任务 CLI 只允许：

- experiment；
- 计划声明的 variant；
- seed；
- horizon（仅 `1,5,10`）；
- device；
- force。

不得提供任何超参数覆盖。suite 必须按 D1--D6 顺序调度，使用 subprocess
任务并行，不生成环境/依赖 manifest。

### 启动前仅允许的六组检查

1. 既有 model 定向测试；
2. 既有 sequence training 定向测试；
3. `tests/test_v3_rank2_model.py`；
4. `tests/test_v3_residual_data.py`；
5. `tests/test_v3_gate_fista.py`；
6. `tests/test_v3_instrumentation.py`。

其中必须证明 causal input、rank-2 forward、residual alignment 和 FISTA
exact-zero。禁止运行 V20 全部 118 项、全仓测试、文件 SHA、包 manifest、
HTML、环境快照、Git commit/tag freeze、M7/M8 manifest、Fold 4、公开
数据下载、逐阶段 zip、replay audit 或完整双路径等价审计。

### 结果目录与单任务文件

所有结果只能写入 `results/v3_diagnostics/`。每个任务只生成：

- `config.json`；
- `summary.json`；
- `training_log.csv`；
- `best.pt`。

D1--D3 可额外输出 surface/lag；D5 可额外输出 gate path；D6 可额外
输出 gradient timeline。不得生成 SHA 文件。

### D1--D6 完成前禁止

- 重跑 v2 Phase 1、30 seeds、M7/M8 bootstrap 或最终 v3；
- TEP、Debutanizer、Gas Turbine、CZ、PLC/MCU；
- 修改 one-SE、penalty、SNR、AR 或增加深度学习基线；
- 把 oracle 结果用于最终模型。

最终交付只允许计划列出的 v3 diagnostic 源码、配置、测试、结果和一个
简单 zip；禁止 HTML、SHA 和复杂 manifest。

## v3 当前状态与正确承接

- `PS_AR_RAPHU_V3_D1_D6_Diagnostic_Execution_Plan.md`：`FROZEN`。
- D1--D6 与实现：`COMPLETED`；已发布 `v3-diagnostics-20260726`。
- 冻结结果：D1 `D1_CAPACITY_FAIL`；D2
  `D2_RANK1_BLIND_SPOT_NOT_CONFIRMED`；D3
  `D3_X_INFORMATION_REMAINS_AT_H1`；D4
  `D4_TRAINING_ORDER_NOT_PRIMARY`；D5
  `D5_SCALE_NORMALIZED_GATE_PATH_SUCCESS`；D6
  `D6_NOT_A_GRADIENT_MAGNITUDE_PROBLEM+D6_PROXIMAL_COLLAPSE_CONFIRMED`。
- D1/D2 的 X-only 模型与含强 AR 的完整目标不匹配，不能据此否定 rank-2
  容量；D3/D4 说明 X 信息仍存在且 dense simultaneous XAR 可学习外生
  贡献；D5 支持标准化函数贡献路径；D6 将原失败定位为 proximal 持续
  收缩而非长期梯度饥饿。

## Spectral Predictive-State AR-RAPHU v0.3 活动语义

- 状态：`THEORY_V0.3 / PROPOSED_AND_TESTABLE` 与
  `PROPOSED_CORE_VALIDATION`。
- v0.3 替换旧“Scheme A 硬支持筛选后 Scheme B 审计”的结构路线，但不
  篡改旧结果。预测路径使用 dense simultaneous XAR；结构路径使用前向
  cross-fitting、双残差正交化和全变量平滑 Urysohn 强凸估计。
- 删除训练期变量硬门控、KAN 参数块 group-prox、用 gate/激活权重解释
  rank，以及 Scheme A 对 Scheme B 的资格控制。
- Scheme A 是同一 Gram 白化完整核的第一谱模态；Scheme B 是该核谱尾。
  支持是评价层的核范数、实际贡献、块消融与 D5 路径证据；rank 证据与
  部署 rank 分离。
- Predictive State 只表示给定阶段、视野与分布下的预测商状态，不声称
  完整物理状态、全局 Takens 嵌入或无条件因果效应。
- 条件加性模型允许每变量独立的平滑时滞--幅值核与不同 rank；本版不允许
  外生变量间显式交互核、未观测阶段切换或不可稳定截断的无穷记忆。

## Spectral v0.3 科学合同

每个实验必须生成并验证 `contract.json`，至少声明：科学问题；target/model
是否含 AR/X；`truth_used_for_training=false`；truth 是否只用于评价；
`support_used_for_training=all|oracle`；超参数只由 validation prediction
loss 选择；rank 输入和 test 均不得参与选择。

- 完整目标含 AR 而模型不含 AR 时，只能标记
  `ORACLE_COMPONENT_DIAGNOSTIC`，不得用于完整目标容量验收。
- 正式结构恢复必须同时残差化目标 y 和外生设计 Phi；truth 仅用于合成
  评价和 oracle capacity。
- support、真核、奇异值、rank、test 与图形不得参与 grid/smoothing 选择。
- 失败必须按预注册映射解释，不得自动换网络、扩大 grid、增加 epoch 或
  改变目标。

## Spectral v0.3 分支、范围与实现边界

- 新分支固定为 `ps-ar-raphu-v4-spectral`，不得覆盖 v2/v3 diagnostics。
- 起点是 `PS_AR_RAPHU_V3_DIAGNOSTICS_RESULTS.zip` 中源码与旧停止线
  `source/`；本轮只做合成 E0--E8，不做公开数据、真实 CZ、MPC、PLC、
  最终论文图或全仓 SHA/manifest。
- 可复用现有 synthetic、sequence/data protocol、preprocessing、model、
  training、phase1 evidence、statistics、gate_fista 与 V20 spline；禁止
  复用 `training.prune_external_path`、A-support-only M8 dispatch 和旧
  group-prox 支持选择。
- 新实现限制在 `src/ar_raphu/spectral/`、三个 spectral tools、
  `configs/spectral_v03.yaml` 和七个 `tests/test_spectral_*.py`。

## Spectral v0.3 固定配置与数值规则

- development seeds `0..4`；confirmation `100..119`；null `200..249`；
  `n_samples=10000`，外生变量 10，`L_x=64`，`L_y=32`，horizons
  `[1,5,10,30,60]`，primary horizon 1。
- 外生三次样条候选 lag `[5,8]`、amplitude `[8,12]`；唯一 fallback
  `(12,16)`；幅值域为 train-only `[0.01,0.99]` 分位。
- AR nuisance basis `6x8`，ridge 候选 `[1e-4,1e-3,1e-2,1e-1]`；lag 与
  amplitude smoothing 候选均为 `[1e-4,1e-3,1e-2,1e-1]`；稳定 ridge
  `1e-6`。
- forward cross-fit：4 folds、initial prefix targets 2000、purge gap 65、
  nuisance selection tail fraction 0.20。
- block length 64；bootstrap development/confirmation `100/500`；jitter
  relative `1e-10`。
- support：max-null 0.95 分位、正消融折比例 0.80、recall 0.80、FPR 0.10。
- rank：alpha 0.05、BH-FDR 0.10、正 rank-2 gain 折比例 0.80、第二模态
  stability 0.70、邻近配置一致数 2。
- adaptive 仅作对照：FISTA ratios
  `[0.32,0.16,0.08,0.04,0.02,0.01,0.005,0.0]`，epsilon 0.05，权重
  `[0.25,4.0]`。
- 设计矩阵可 CUDA 分块构建后立即转 CPU FP64。维数 <2000 的主 full-kernel
  ridge 必须用 FP64 Cholesky，不得用 Adam/FISTA；relative KKT residual
  `<=1e-8`。`OMP_NUM_THREADS=1`，多 seed 并发 4--8，不用 DDP。
- 联合 one-SE 只看 validation MSE；候选集内依次选最少总核系数、最大
  `lambda_tau*lambda_x`、更小 lag basis、更小 amplitude basis、固定顺序。

## Spectral v0.3 E0--E8 顺序与停止线

1. E0 重放 AR/X/过程创新/测量噪声；任一恒等式误差 >`1e-10` 即停止。
2. E1 做 AR-S1/S2/S3/S4 投影 oracle；NRMSE >0.10 时只允许 `(12,16)`
   fallback，仍失败即停止。
3. E2 用纯外生真值目标、oracle active support 和 full tensor Urysohn 验证
   full/rank 容量；失败即停止，不讨论 support/rank。
4. E3 比较 oracle、只残差 y、双残差和联合 AR+X。双残差须至少 4/5 seeds
   更接近 oracle，median `d_D/d_Y<=0.85`，validation RMSE 恶化不超过 5%。
   residual Gram condition number >1e8 标记弱可辨识；与 nuisance basis 最大
   相关 >0.10 标记残差化不足。失败停止 E4/E5，不换网络。
5. E4 用 50 个 AR-S0 null seeds 建 max-null cutoff；所有变量一直保留。
   confirmation 上 AR-S0 FPR <=0.10，AR-S1/S2/S3 recall >=0.80 且
   FPR <=0.10。
6. E5 只对 E4 有支持证据的变量推断 rank；rank-2 同时要求 adjusted p
   <=0.10、正增益折比例 >=0.80、稳定性 >=0.70 和邻居一致。AR-S1/S2
   错误升级率 <=0.10，AR-S3 检出率 >=0.80。
7. E6 比较 uniform 与 D5-adaptive；只有验证计划所有升级条件同时满足才
   采用 adaptive，否则 uniform 保持默认。
8. E7 比较 AR-only、dense XAR no-prox、spectral fixed、anchored refit；
   rank 固定、全变量保留、无 group-prox。B3 须位于 B1 one-SE、至少三个
   horizon 优于 AR-only、anchor 偏离 <=20%、不删除变量。
9. E8 仅在 E7 通过后运行；依次尝试 Gamma/Erlang、多指数、Laguerre、
   generic stable state-space，要求核误差 <=0.05、递推 RMSE 增量
   <=`0.01*Std(y)`、谱半径 <=0.995。

严格按 E0--E7 development 进入条件运行，全部满足后才 confirmation；E8
仅由 E7 解锁。E4 失败只保留连续证据，E5 失败只保留预测器，E7 失败保持
结构/预测双模型，E8 失败保留窗口实现。

## Spectral v0.3 检查、输出与打包

- 启动前只运行验证计划第 2 节列出的九个定向测试。禁止全仓测试、文件
  SHA、manifest、HTML、Git tag 检查、旧 M7/M8 审计和 checkpoint replay。
- CLI 只允许 `experiment`、`stage`、`device`、`force`；禁止覆盖 grid、
  smoothing、threshold、seed、target semantics 和 bootstrap 次数。
- 结果只写 `results/spectral_v03/`；job 只保存 `contract.json`、
  `config.json`、`summary.json`、`metrics.csv`、`fit.npz`，bootstrap 时增加
  `bootstrap_statistics.npz`。自动决策只能使用计划字段与映射。
- 完成后只打包一次 `SPECTRAL_PS_AR_RAPHU_V03_RESULTS.zip`，只检查
  `DEVELOPMENT_DECISION.md`、`spectral_summary.csv` 和 `unzip -t`；不生成
  SHA 或 manifest。
- 第一批严格限于：component replay+E0、tensor design+E1、FP64 ridge+
  Gram SVD、正确外生目标 E2、cross-fit nuisance 与双残差测试、E3
  development 5 seeds。六项通过后才开始 E4/E5。

## Spectral v0.3 当前状态与本次授权边界

- 两份 v0.3 文档已于 2026-07-26 纳入项目持久约束。
- 分支 `ps-ar-raphu-v4-spectral`：`CREATED`。
- Spectral v0.3 convex core：`IMPLEMENTED`；九组规定定向测试共
  `35 passed`。
- E0：`E0_COMPONENT_IDENTITY_PASS`。30 个 development jobs 的 latent
  最大恒等误差为 `1.7763568394002505e-15`，measurement identity 为 0。
- E1：`E1_PROJECTION_FALLBACK_FAIL`。唯一 `(12,16)` fallback 的最坏
  projection NRMSE 为：AR-S1 `0.14709`、AR-S2 `0.44745`、AR-S3
  `0.46491`、AR-S4 `0.43720`，均高于冻结 `0.10` 门槛。
- E2--E8：`NOT_YET_RUN`；`NEXT_ALLOWED_STAGE=STOP_E1_PROJECTION_CAPACITY`。
- 按冻结停止线不得启动 E2，不得扩大 basis grid；如需修改 E1 basis
  空间，必须由用户给出新协议或明确授权重新预注册。

## Spectral v0.3.1 表示修复与核心验证

- v0.3.1 的唯一执行规范为
  `Spectral_PS_AR_RAPHU_Theory_v0_3_1.md`、
  `Spectral_PS_AR_RAPHU_Validation_Plan_v0_3_1.md` 与
  `Codex_Restart_Prompt_Spectral_v0_3_1.md`；旧 v0.3 仅作冻结历史。
- 新分支固定为 `ps-ar-raphu-v031-representation-repair`；旧
  `results/spectral_v03/` 与 `configs/spectral_v03.yaml` 不得修改或覆盖。
- 新结果只写 `results/spectral_v031/`。E0 复用旧通过结果，旧 E1 保持
  `E1_COMPRESSED_LAG_BASIS_UNDERSPECIFIED`，不得改写成通过。
- E1R 固定幅值基 16，三次样条时滞候选 `24,28,32,40`，离散 identity
  reference 为 `np.eye(64)`；两侧最小二乘投影，不构造巨大 Kronecker
  design。必须选择 `32x16` 且 regression table 在 `5e-6` 内才可继续。
- E2A 只拟合单一 oracle 变量的真实外生贡献；E2B 只拟合 oracle active
  support 的总外生贡献；二者都不含 AR，且只由 validation contribution
  MSE 选择平滑，CPU FP64 Cholesky，KKT residual `<=1e-8`。
- E3 固定 `32x16`、十个外生变量与 O/Y/D/J 四方法；D 必须同时残差化
  `y` 和 `Phi`。十变量解使用 matrix-free FP64 PCG、block-Jacobi，
  relative residual `<=1e-8`、最多 2000 次；未收敛结果不得用于结论。
- 本轮严格顺序为 `E1R -> E2A -> E2B -> E3`。任一级失败立即按预注册
  标签停止；本轮不启动 E4--E8，不自行改变 basis、阈值、目标或模型。
- 当前状态：v0.3.1 三份冻结文件已完整读取；独立分支已创建。
  E1R=`E1R_REPRESENTATION_CERTIFIED_32x16`，冻结表格最大复现误差
  `4.97e-11`，已选择 `32x16`。E2A=`E2A_ESTIMATOR_OR_DATA_EXCITATION_FAIL`：
  60 个场景/seed/变量结果仅 15 个通过，四个场景均未达到 4/5 seed；
  E2B/E3=`NOT_YET_RUN`，本轮停止线为
  `NEXT_ALLOWED_STAGE=STOP_SINGLE_KERNEL_CAPACITY`。
