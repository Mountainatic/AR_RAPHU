# Phase 3 — 合成反事实和数学检验

固定 seed=20261010，N=4096，Ts=2 秒，delta=.01，半向上 floor(x/delta+.5)，平移等变 tie 容差 1e-10 grid units。数组、SHA、参数、PSD/ACF/PACF/步长频率、四图 S1–S4 和 N=2048/4096/8192 × seed/seed+1 对照均在 artifacts/synthetic。未调用 PRISM.fit/learn/tune。

S1 无正弦，比例 1/7、2/7、3/8 和 sqrt(2)/10。ΔQ 的 q 点周期测试使用8192点、绝对容差2e-13；q=7 对应14秒，7秒是可能谐波。不是所有比例都应出现同一个7秒峰。S2 慢趋势+零均值高斯测量噪声 sd=.0025，报告量化前/后0/±delta/other频率，不称量化误差 iid。S3 正弦1/7、.137、.149Hz，前采样Ts=.2秒与实际Ts=2秒；注入频率不能证明量化根因。S4 f0=.357142857Hz,fs=.5Hz，实际 alias=abs((f0+fs/2)%fs-fs/2)≈.142857143Hz，不推断实测仪器存在高频分量。

|场景/序列|N|全局 PSD peak Hz|.133–.153 Hz 区间最大值位置|区间能量占比|分辨率 Hz|
|---|---:|---:|---:|---:|---:|
|S1 1_over_7_adjacent_difference_quantized|4095|0.0714285714|0.142857143|0.333333|0.000122100122|
|S1 2_over_7_adjacent_difference_quantized|4095|0.142857143|0.142857143|0.649396|0.000122100122|
|S1 3_over_8_adjacent_difference_quantized|4095|0.187545788|0.133089133|7.4812e-05|0.000122100122|
|S1 sqrt2_over_10_adjacent_difference_quantized|4095|0.0706959707|0.141391941|0.31401|0.000122100122|
|S4 sampled|4096|0.142822266|0.142822266|0.998503|0.000122070312|

注释区间总能量是描述量，不是显著性检验或门禁；即使区间里总有一个最大值，也不表示存在显著7秒峰。零方差 ACF/PACF 明确为未定义。PSD=boxcar periodogram，constant 去均值；PACF=有偏ACF的Yule–Walker估计。有限样本附近频率有泄漏/格点分辨率。

数学测试：iid adjacent difference 100000点三种seed，rho1与-.5距离<.015（统计容差）；随机频点差分频响等式容差2e-15；alias 峰落在一个 FFT bin 内；时间/数值缺口分段、不拼接、短段不承诺长 lag。频响结论只用于本项目相邻差分H1，不套用于H4。

