# PRISM 原生流式版实现说明

版本 `prism-native-arx-20261119`，2026 年 11 月 19 日。该版本固定了三个探索性主配置及其原生推理实现。源码在公开仓库，已训练的模型和复测所需的私有数据由课题组单独保管。模型与发布对应关系见 [evidence.json](evidence.json)。

## 1. 任务与三个配置

每次调用处理一个采样点 $t$，输出当前目标的估计值 $\hat y_t$。过程量 $x_t$ 到达时，调用者同时提供刚成熟的实际目标 $y_{t-D}$。实验中的 $D$ 是标签可用性假设，实际使用时必须由调用者保证时间对齐。

| 配置 | 窗口 $L$ | 延迟 $D$ | 上游输出 | A 特征 | A 遗忘因子 | 上游／A 回归系数 |
|---|---:|---:|---|---|---:|---:|
| SRU H2S | 16 | 8 | 有限 Urysohn 幅值响应，asinh 逆链接 | `QC32_RLS` | 0.999 | 23／13 |
| CZ Rod1 | 128 | 4 | E128 后的有限 Urysohn 幅值响应，signed-log 逆链接 | `QD8_RLS` | 0.995 | 83／7 |
| CZ Rod2 | 128 | 4 | E128 后的线性响应，signed-log 逆链接，输出锚定增量 | `EX_RLS` | 0.999 | 43／23 |

每个配置输入五列过程量。SRU 的输入为 `u1` 至 `u5`。CZ 按冻结缓存顺序输入主加热功率、晶体提升、坩埚提升、晶体旋转、坩埚旋转五列。各头分别拟合，使用各自的训练常数和初始协方差。推理端保持原训练参数、输入窗口、更新频率和 FP64 精度。

## 2. 上游 E、KC 和 W

两个 CZ 头都用到 $x_t,\ldots,x_{t-127}$。把时滞划分为八个左闭右开区间。

```text
[0,1) [1,2) [2,4) [4,8) [8,16) [16,32) [32,64) [64,128)
```

第 $b$ 个分箱、过程通道 $c$ 的特征为 $E_{b,c}(t)=\sum_{\ell=a_b}^{b_b-1}x_{t-\ell,c}/(b_b-a_b)$。分箱近处细、远处粗，最大时间支持为 128 步。Rod1 的每个分箱均值进入两个一阶样条基，Rod2 直接使用均值。SRU 先逐点计算幅值样条，再取当前项与过去 15 项的平均。

上游还读取 $y_{t-D}$ 和 $y_{t-D-L+1}$ 两个成熟目标端点。岭回归预测变换坐标中的响应，随后由 W 对应的逆链接恢复量纲。SRU 使用 $s\sinh(z)$，CZ 使用 $\operatorname{sign}(z)\operatorname{expm1}(|z|)$。Rod2 的坐标响应是目标增量，恢复后加上最新成熟目标 $y_{t-D}$。把得到的上游预测记为 $q_t$。这三个主头的 W 是目标坐标链接，模型中没有额外的 `W` 样条修正对象。

编译前将冻结的标准化、回归系数和输出缩放折叠为 FP64 仿射系数。对于一维样条，先对实际冻结变换在结点处取值，再从相邻结点值计算各段斜率。执行时选择当前分段，使用结点值加斜率乘距离，区间外沿用边界线性延拓。[benchmark.py](benchmark.py) · [streaming_balance.py](streaming_balance.py)

## 3. 在线 A 与标签成熟

令 $r_s=y_s-q_s$。在 $t$ 时刻，最新能用的残差是 $r_{t-D}$，其中 $q_{t-D}$ 是当时已经发行的上游输出。A 的特征随配置变化。

- SRU `QC32_RLS` 使用当前 $q_t$、1／4／8／16／32 步响应差、响应与 16／32 步差的交互、最新成熟残差，以及该残差与当前响应、4／32 步响应差的三个交互项。
- Rod1 `QD8_RLS` 使用当前响应、1／4／8 步响应差、响应与 8 步差的交互及最新成熟残差。
- Rod2 `EX_RLS` 使用当前响应、最新成熟残差、五列当前过程量及 `[1,8)`、`[8,32)`、`[32,128)` 的过程均值。三个均值复用 E128 分箱的和。

特征使用冻结的均值和尺度归一化并首部加 1，记为 $\phi_t$。当标签 $y_s$ 在 $t=s+D$ 成熟后，使用**发行时保存的** $\phi_s$ 更新 A。令 $\tilde r_s=(r_s-\mu_A)/\sigma_A$，$P$ 是协方差，$\lambda$ 是遗忘因子。一次更新为

$$
u=P\phi_s,\qquad d=\lambda+\phi_s^{\mathsf T}u,
$$

$$
\theta\gets\theta+\frac{u}{d}(\tilde r_s-\phi_s^{\mathsf T}\theta),\qquad
P\gets\frac{P-uu^{\mathsf T}/d}{\lambda}.
$$

更新后输出 $\hat y_t=q_t+g(\mu_A+\sigma_A\phi_t^{\mathsf T}\theta)$，三个主头的 $g=1$。随后把 $\phi_t$ 写入延迟队列。前 $D$ 个调用可以从合法上游历史构造残差特征，A 的参数更新则从本次启动后发行的第一条特征成熟时开始。[native_stream.c](native_stream.c)

## 4. 原生实现如何减少耗时

原生内核每步操作预分配的环形缓冲，分箱和从上一时刻的和加上进入值、减去离开值。每 256 次调用从缓冲重新求和，限制舍入漂移。预测保存 64 个上游发行值，A 的待成熟特征只保留 $D$ 条，读写槽位由采样序号确定。

RLS 使用对称秩一更新。协方差按连续行存储，AVX-512 路径将特征维补齐到 8 的倍数，使投影和矩阵更新用连续向量操作。初始化时将协方差对称化，更新时使用对称外积。保留分母、有限值、参数范数及每 256 次 RLS 更新的 Cholesky 正定性检查。若状态失效，C 入口返回 `NaN`，Python 单步接口抛出异常。

给定 $B$ 个分箱、$d_x$ 个过程通道、$p$ 个 A 系数，常规单步计算由 $O(Bd_x+p^2)$ 主导。定期重算分箱和 Cholesky 分别为 $O(Ld_x)$ 与 $O(p^3)$。匹配窗口的密集 AR 为 $O(L)$，ARX 为 $O((d_x+1)L)$。本机录得的同量级耗时有明确窗口、模型和硬件条件。三个 PRISM 状态缓冲分别占 4,888、13,192、17,800 字节，这是原生状态结构的大小。

## 5. 编译、调用和复测

验证环境为 Linux x86-64、GCC 11.4、Python 3.10、NumPy、Pandas、SciPy 和 scikit-learn。公共合成用例从临时目录生成三个头和匹配的 AR/ARX 常数，不读取实验文件。

```sh
python smoke.py
python smoke.py --portable
```

有冻结文件访问权限时，用 `benchmark.py` 生成每个头的 `model_constants.h`、`stream.so` 和 `MODEL.json`。`--profile` 指向冻结配置目录，`--sru` 指向 SRU 匹配窗口基线目录，`--cz` 目录下包含 `CZ_Rod1` 和 `CZ_Rod2`。脚本核对模型哈希，使用本机 GCC 编译，并在计时前核对实验开发段的所有预测。

```sh
python benchmark.py \
  --profile "$PRISM_PROFILE" --sru "$SRU_BASELINES" --cz "$CZ_BASELINES" \
  --results ./build/native-benchmark
```

给定起点 `t`，两份启动历史都需要 `L + 63` 个采样，从旧到新排列。过程历史最后一项为 `X(t-1)`，成熟目标历史最后一项为 `y(t-D-1)`。加载器用冻结的上游模型复原前 64 个时间点的 `q`，送给 C 初始化。每条独立轨迹创建自己的状态。

```python
from native_predictor import NativePredictor

state = NativePredictor(
    artifact=benchmark_dir / "CZ_Rod1_D4",
    profile_dir=profile_dir,
    model_path=profile_dir / "CZ_Rod1_D4.pkl",
    process_history=process_history,
    mature_target_history=mature_target_history,
    first_origin=t,
)
y_hat = state.step(x_at_t, y_at_t_minus_4)
```

用 C 部署时，入口是 `prism_state_bytes`、`init_prism`、`prism_step`。`init_prism` 接收从旧到新的 `L-1` 条过程、成熟目标历史，以及按相对发行序号模 64 重排的前 64 条上游预测。参考实现可复用 Python 加载器生成初始发行历史。保管和传输生成的 `.so`、常数头文件及模型文件时按权重资产处理。[native_predictor.py](native_predictor.py) · [deploy_history.py](deploy_history.py)

## 6. 验证结果与论文引用口径

完整开发段原生预测与原流式参考通过 `rtol=1e-12, atol=1e-12`。SRU、Rod1、Rod2 最大绝对差分别为 `5.25e-15`、`2.29e-11`、`1.14e-13`。RMSE 分别为 `0.023238`、`0.014646`、`0.010970`。SRU 和 Rod1 相对现有最强匹配 Transformer 的开发段 RMSE 低 6.70% 和 8.88%。Rod2 高 0.12%，相对于 ARX 则低约 7.94%。这些结果来自此前使用过的开发段，Rod2 的严格不劣和正式 SOTA 仍待独立验证。

同一服务器核心上，三个 PRISM 头分别耗时 `0.290`、`0.253`、`0.396` 微秒／步，约为实际冻结 ARX 的 `4.80×`、`1.85×`、`2.79×`。三种方法使用同一编译设置，比较包含原生历史维护与推理，PRISM 另含 A 在线更新。每项 51 次整段重放，去掉前 64 步预热后取每次平均单步耗时的中位数。该实验的计时范围为原生内核，Python 封装、初始化和 I/O 在计时区间之外。

SRU 删除 KC、W、A 后的 RMSE 分别增加 106.11%、63.13%、31.07%，来自原主配置各独立重训的删除项。完全去 A 和冻结 A 参数是两种不同的对照。Rod2 的训练内更新频率搜索重新选回逐步更新。最佳稀疏更新候选为每两步更新一次，开发 RMSE `0.011134`，因此发布版采用原逐步策略。[完整复测报告](REPORT_CN.md) · [汇总与哈希](evidence.json)
