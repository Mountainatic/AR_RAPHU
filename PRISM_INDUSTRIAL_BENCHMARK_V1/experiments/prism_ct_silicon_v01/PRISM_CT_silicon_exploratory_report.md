# PRISM-CT 拉硅数据探索性原型

## 数据与协议

- Sheet1: 20,103 rows；Sheet2: 20,627 rows。
- 目标：晶体直径。
- 跨 Sheet 仅使用两边共有的 6 个工艺输入 + 当前晶体直径状态。
- 采样周期按 2 s 假设。
- 预测步长：1, 5, 15, 30, 60, 120, 300, 600 steps，对应约 2 s, 10 s, 30 s, 1 min, 2 min, 4 min, 10 min, 20 min。
- CT 时间常数：10, 30, 60, 120, 300, 600, 1200, 2400, 4800, 7200 s。
- 单 Sheet：严格因果 60/20/20 时间切分。
- 跨 Sheet：一个 Sheet 全部用于拟合，另一个 Sheet 仅评估；若两个 Sheet 实际是两根独立晶棒，则可解释为跨棒探索。
- 为了做结构对比，所有 Ridge 分支统一采用 `alpha = n_train * lambda`，本轮 `lambda=1`，不按 test horizon 调参。

## CT 状态

稳定一阶连续时间状态：

`z_tau[t] = exp(-dt/tau) z_tau[t-1] + (1-exp(-dt/tau)) x[t]`

CT-Absolute 直接使用多个 `z_tau`。

CT-Multires 使用：

`[x, x-z_tau1, z_tau1-z_tau2, ..., z_tau(n-1)-z_taun]`

预测目标是增量：

`Delta y_h(t) = y(t+h)-y(t)`，最后恢复 `y_hat(t+h)=y(t)+Delta y_hat_h(t)`。

## 核心结果

### Sheet1 内部

在这一根/这一段上，离散 Lag-Ridge 全程更强：

| horizon | Lag-Ridge persistence skill | R² |
|---:|---:|---:|
| 1 step / 2 s | 3.7% | ~1.000 |
| 60 / 2 min | 47.5% | 0.979 |
| 120 / 4 min | 61.6% | 0.950 |
| 300 / 10 min | 74.8% | 0.801 |
| 600 / 20 min | 83.2% | 0.492 |

说明 CT 并不会天然替代离散 delay；对 Sheet1，明确的离散延迟结构非常强。

### Sheet2 内部

短期 CT-Multires 最好；中长预测 CT-Absolute 明显更强：

| horizon | 最好结构 | persistence skill | R² |
|---:|---|---:|---:|
| 1 / 2 s | CT-Multires | 2.3% | 0.997 |
| 60 / 2 min | CT-Multires | 10.0% | 0.782 |
| 120 / 4 min | CT-Absolute | 17.7% | 0.634 |
| 300 / 10 min | CT-Absolute | 45.5% | 0.185 |
| 600 / 20 min | CT-Absolute | 63.0% | -0.009 |

20 min 的 R² 仍接近 0，说明该测试尾段本身很难，但相对 persistence 已显著改善。

### 跨 Sheet：CT-Multires

CT-Multires 在两个方向的所有 horizon 都得到正 persistence skill，并且优势随 horizon 增大：

| horizon | S1→S2 skill | S1→S2 R² | S2→S1 skill | S2→S1 R² |
|---:|---:|---:|---:|---:|
| 1 / 2 s | 1.4% | 0.999 | 2.3% | ~1.000 |
| 5 / 10 s | 4.0% | 0.996 | 6.3% | 0.998 |
| 15 / 30 s | 6.7% | 0.985 | 11.7% | 0.994 |
| 30 / 1 min | 4.6% | 0.964 | 14.1% | 0.985 |
| 60 / 2 min | 10.6% | 0.941 | 29.0% | 0.970 |
| 120 / 4 min | 23.8% | 0.897 | 42.6% | 0.927 |
| 300 / 10 min | 42.4% | 0.721 | 53.1% | 0.712 |
| 600 / 20 min | 38.1% | 0.225 | 62.7% | 0.348 |

## 结构性结论

1. **CT 的主要价值不是一步预测。** 一步情况下目标极平滑，persistence 已经接近饱和。
2. **绝对 CT 状态适合描述单域慢状态，但不适合直接跨域。** 多个 EMA/稳定极点状态高度共线，轻微工况偏移可能被线性读出放大。
3. **多分辨率状态更像 PRISM 应该使用的 CT 表示。** `x-z_fast` 和相邻尺度差分保留了“哪个尺度发生了变化”，同时比绝对状态更有迁移性。
4. **Sheet1 与 Sheet2 告诉我们不能只保留一种时间表示。** Sheet1 离散 delay 更强，Sheet2 慢 CT state 更强，因此下一版更合理的结构是 `discrete delay branch || stable CT branch`，再由 PRISM 的 route/assembly 机制决定是否启用。
5. **下一版本最值得加入 conditioning/support audit。** 对 CT basis 做共线性、系数范数、跨域 support 检查，不通过则禁止 absolute-state branch，只允许 multires/state-increment branch。

## 科学性说明

这是结构探索，不是冻结 benchmark。下一轮如果要形成论文证据，需要先冻结：时间常数集合、Ridge 正则化规则、是否包含 target-state、两个 Sheet 的实际物理身份、train/val/test 及跨棒协议，然后重新运行，不能继续根据测试结果调整结构后再把同一测试集当最终 holdout。
