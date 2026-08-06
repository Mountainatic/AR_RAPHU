# PRISM v2.1 SRU Stagewise-Routed 设计包

本包包含：

1. `PRISM_Theory_v2_1_Stagewise_Routed_Modular_Assembly_Theory_Only.md`  
   完整 v2.1 理论：继承 v2.0 模块代数，改为 K→ΔW→A 逐级 OOF 目标路由、模块局部 guarded one-SE、可关闭软重叠惩罚，以及 mandatory-input Joint-KWA。

2. `PRISM_V2_1_SRU_EXPERIMENT_AND_IMPLEMENTATION_PLAN.md`  
   SRU 单数据集实验、基线比较、数据 base 保留、阶段门、代码修正、统计、输出与打包方案。

3. `PRISM_V2_1_CHANGELOG_FROM_V2_0_AND_CODE_AUDIT.md`  
   从 v2.0 到 v2.1 的变更与原实现问题审计，区分理论设计问题、实现不一致和明确代码正确性错误。

## 本轮冻结解释

- 只执行 SRU 数据集；
- 保留 SRU 两个既有主目标头：H2S 与 SO2；
- 不删除或重建其他数据集的 C1 base；
- A-only/AR 只作为外部预测基线；
- Physical-First 不允许回退 A-only；
- Joint 允许 K/W/AR 相互侵占，但不允许 AR-only；
- 旧 V2 路线保持停止和只读归档。

## 建议下一步

将本包放入新分支 `prism-v2-1-sru-stagewise-routed`，按实验方案先完成 E0/E1 数据继承与回归测试，再开始 SRU 训练。
