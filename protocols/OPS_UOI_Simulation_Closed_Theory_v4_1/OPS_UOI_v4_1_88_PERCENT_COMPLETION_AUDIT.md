# OPS-UOI v4.1：88% 理论完成度审计

## 冻结结论

v4.1 不再扩展到 MPC、主动控制或多阶段统一模型。当前论文对象固定为：

\[
\boxed{
\text{固定阶段、闭环观测数据下的系统辨识方法}
}
\]

PS-AR-RAPHU 与 ARX、pNARX、MLP-NARX、RNN 等 baseline 同级竞争。

## 完成度

| 模块 | 完成度 | v4.1 状态 |
|---|---:|---|
| Operator-first estimand | 95% | 冻结 |
| Q/K 分层与 quotient | 95% | 冻结 |
| 双残差正交化 | 90% | 冻结 |
| finite-sieve coercivity / Schur | 88% | 理论完成，待更多实证 |
| 分辨率与 rank | 88% | 主链完成 |
| baseline 同级定位 | 100% | 冻结 |
| 三域定义 | 95% | v4.1 新增 |
| bounded \(C^1\) continuation | 95% | v4.1 新增 |
| free-run 良定性与全局有界 | 95% | v4.1 新增 |
| 增量稳定充分证书 | 85% | v4.1 新增 |
| H2 原生选择链 | 85% | 理论合同完成 |
| 解释防火墙 | 95% | v4.1 新增 |
| finite-sample bootstrap/Lepski | 70% | 剩余工作 |
| PB1 四数据集实验闭合 | 55% | 需代码重跑 |
| CZ outer validation | 35% | 受数据限制 |

理论设计加权完成度：

\[
\boxed{88\%}.
\]

## 剩余 12%

1. block/bootstrap 同时半径的最终有限样本推导；
2. Lepski crossing regularity 的严格条件；
3. 多变量 Schur/coercivity 的公开数据证据；
4. 四数据集 H2/native 与 free-run 重跑；
5. official test 一次性 confirmation；
6. 多晶棒 CZ outer validation。

## v4.1 解决的核心缺口

此前有限 spline 域近似同时承担：

\[
\mathcal D_{\mathrm{model}}
=
\mathcal S_{\mathrm{train}}
=
\mathcal S_{\mathrm{cert}},
\]

导致预测值稍微出界就停止。

v4.1 冻结为：

\[
\boxed{
\mathcal S_{\mathrm{cert}}
\subseteq
\mathcal S_{\mathrm{train}}
\subseteq
\mathcal D_{\mathrm{model}}=\mathbb R
}
\]

并严格规定：

- 数学上全局可计算；
- 训练支撑外不声称已识别；
- K 层只在认证域解释；
- free-run 必须作为固定系统辨识模型完成；
- online learning 不是 PB1 的补救机制。

## 下一允许工程阶段

```text
PB1_DEVELOPMENT_REPAIR_V2_1_SIMULATION_CLOSURE
```

该阶段只实现 v4.1，不增加新架构。
