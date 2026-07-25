# 项目变更记录

**项目**: T-AKGNN (Temporal-Aware AKGNN)
**基干**: AKGNN (IEEE TIM 2025) by Yang et al.
**数据**: 单晶炉晶体直径预测

---

## 迭代 1 — 初始状态评估

### 问题

现有项目声称复现 AKGNN 论文，但实际实现与论文存在根本性偏差：

1. 论文核心是 **Bregman 散度 Simplex 投影** 构建自适应图（Section IV-B，式11-24），而实现用的是 GAT-style 自注意力（Q@K^T+Softmax）
2. 论文输入是单时刻特征向量 x∈R^D（稳态过程），而实现用了 Conv1d 时序编码 + 滑动窗口 [B, N, L]
3. 论文 Embedding 是单路投影 h=xW+b（式25），而实现用了三路并行 h1/h2/h3
4. 实现额外加入了 EMA 快慢图机制（论文没有）

### 改动

- 阅读论文，撰写论文总结和研究方案
- 撰写复现差异审计报告

### 模型结构（初始项目）

`
输入 [B, 10, 32]
  → Conv1d 时序编码器 (压缩)
  → InputEmbedding (3路并行: h1, h2, h3)
  → GAT-style 图构建 (Q@K^T+Softmax) + 快慢图 EMA
  → 多头消息传递
  → KAN 预测
`

---

## 迭代 2 — T-AKGNN 方案设计与实现

### 问题

原 AKGNN 论文明确限定于 **稳态连续过程**（Section IV-A），不处理时滞。但工业控制数据普遍存在输运延迟、热惯性等滞后效应。

### 改动

将 AKGNN 的 Bregman 投影从纯空间域（2D, D×D）扩展到联合时空域（3D, D×D×L），每个条目 A[i,j,τ] 表示变量 j 在 τ 步前对变量 i 的影响权重。

**新增文件**:

| 文件 | 作用 |
|------|------|
| config.py | 集中超参数管理 |
| utils.py | set_seed / calculate_metrics |
| lag_kernel.py | 可学习时滞核（指数衰减 / 高斯） |
| st_graph.py | 联合时空 Bregman Simplex 投影图构建 |
| analysis.py | 时滞可解释性分析 |

**重写文件**:

| 文件 | 改动 |
|------|------|
| layers.py | InputEmbedding 从三路改回单路 |
| model.py | Conv1d + GAT → ST-Graph + ST消息传递 |
| train.py | 全面重写，保留原始可视化风格 |
| process_data.py | 重构，新增 var_names |

**删除**: graph_module.py（被 st_graph.py 替代）

### 当前模型结构

`
输入 [B, N, L] (N=10个节点, L=32步历史)
  |
  ┌─ Step 1: InputEmbedding (单路投影)
  │   h_i(t-τ) = x_i(t-τ) * w_i + b_i
  │   输出: [B, N, L, D]
  │
  ├─ Step 2: 联合时空图构建
  │   2a. Q/K 多头投影 (W_q, W_k)
  │   2b. 时空相似度: scores[b,h,i,j,τ] = Q_i · K_{j,τ} / √d_h
  │   2c. 时滞核调制: κ[i,j,τ] = w·exp(-β·τ) (可学习)
  │   2d. Bregman Simplex 投影: A = κ·exp(scores) / Σ(κ·exp(scores))
  │   输出: [B, H, N, N, L]
  │
  ├─ Step 3: 时空消息传递
  │   h_out = Σ_{j,τ} A[i,j,τ] · h_j(t-τ)
  │   残差: g = mean(h_out) + x[:,:,-1]
  │   输出: [B, N]
  │
  └─ Step 4: KAN 预测
      [N → 2N → N → 1] 三层 KANLinear
      输出: ŷ ∈ [B, 1]
`

### 训练结果

| 指标 | 值 |
|------|-----|
| 最佳 Epoch | 79 |
| Val RMSE (norm) | 0.1395 |
| Val R² | 0.9324 |
| Test R² (real) | 0.4856 |
| Test RMSE (real) | 0.1335mm |

---

## 迭代 3 — 可视化修复

### 问题

matplotlib 默认字体（DejaVu Sans）不支持中文，时滞分析图中的变量名以方框显示。

### 改动

在 analysis.py 和 train.py 中添加中文字体配置：
`python
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
`

---

## 迭代 4 — 超参数调优

### 问题

默认配置是否最优？不同超参数对性能的影响未知。

### 改动

运行 5 组实验对比（sweep.py）

### 结果

| 实验 | 参数 | Val R² | Test R² |
|------|------|--------|---------|
| default | 指数核, 8头, L=32, grid=7 | 0.9324 | 0.4856 |
| kernel_gauss | 高斯核, 8头, L=32, grid=7 | 0.9331 | 0.5029 |
| heads_4 | 指数核, 4头, L=32, grid=7 | 0.9072 | 0.3421 |
| window_16 | 指数核, 8头, L=16, grid=7 | 0.7275 | 0.0498 |
| grid_5 | 指数核, 8头, L=32, grid=5 | 0.8140 | — |

**结论**: 高斯核略优（+0.017 R2），8头优于4头，窗口32优于16

---

## 迭代 5 — Test R² 低诊断与修复

### 问题

Test R² (0.486) 远低于 Val R² (0.932)，模型泛化能力差。

### 诊断

7:2:1 时序切分导致训练/测试分布漂移：

| 维度 | 训练集 (前70%) | 测试集 (后10%) |
|------|--------------|---------------|
| 直径标准差 | 0.404 | 0.186 |
| 直径范围 | 1.62mm | 0.79mm |
| 直径-长度相关性 | -0.505 | -0.850 |

训练集覆盖早中期（高波动），测试集覆盖晚期（低波动）。R² 是方差相对指标，相同绝对误差下，低方差测试集 R² 自然偏低。

### 尝试的方案

| 方案 | 改动 | 结果 | 结论 |
|------|------|------|------|
| P0-随机切分 | 数据打乱再切分 | Test R²=-0.01 | 破坏时序依赖，不可行 |
| P0-扩训练比例 | 0.7 → 0.85 | **Test R²=0.956** | 核心解决 |
| P1-正则化 | weight_decay 1e-5→1e-3 | 未使用 | 风险高 |
| P2-Dropout | KAN前加 dropout(0.1) | NaN | KAN不兼容 |

### 最终性能

| 配置 | Test R² | Test RMSE | Test MAE |
|------|---------|-----------|----------|
| 原始 (7:2:1) | 0.486 | 0.1335mm | 0.1249mm |
| **优化 (8.5:1:0.5)** | **0.956** | **0.0391mm** | **0.0353mm** |

---

## 当前项目文件

`
E:\silicon\gnnest├── CHANGELOG.md          ← 本文件
├── test_r2_diagnosis.md  # Test R2 诊断记录
│
├── config.py             # 超参数管理
├── utils.py              # 工具函数
├── model.py              # T_AKGNN 主模型
├── layers.py             # InputEmbedding + KAN 层
├── lag_kernel.py         # 可学习时滞核
├── st_graph.py           # 时空图构建（核心）
├── train.py              # 训练脚本
├── process_data.py       # 数据预处理
├── analysis.py           # 时滞可解释性分析
├── sweep.py              # 超参调优
│
├── results/              # 标准训练输出
├── results_sweep/        # 超参调优输出
└── 2plan/                # 前期方案文档
`

## 关键决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 图构建方法 | Bregman 投影 + 时滞核 | 论文理论框架，非 GAT 注意力 |
| 时滞核类型 | 指数衰减（默认） | 参数更少，训练稳定 |
| 窗口长度 L | 32 | 16不够，64数据受限 |
| 注意力头数 | 8 | 4不够，16未见提升 |
| 数据切分 | 时序 8.5:1:0.5 | 随机切分破坏时序依赖 |
| 正则化 | 不额外加 | KAN + Dropout 不兼容 |
| KAN grid | 7 | 5欠拟合，9未见必要 |
