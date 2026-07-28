# OPS-UOI / PS-AR-RAPHU v4.1

本包把 v4.0 operator-first 理论推进到 v4.1 的 simulation-closed 版本。

核心冻结：

- 我们与 ARX/pNARX/MLP-NARX 等 baseline 同级；
- 三域关系：
  `certified support ⊂ train support ⊂ model domain`；
- bounded C1 continuation 提供全局数学定义和 free-run 有界性；
- continuation 不产生域外 Q/K 识别证据；
- H2 是主模型，H3 仅作公平消融；
- 当前理论设计完成度冻结为 88%。

文件：
1. 完整集成理论；
2. 新增定理详细证明附录；
3. v4.1 定理—实验合同；
4. 机器合同 YAML；
5. 88% 完成度审计。
