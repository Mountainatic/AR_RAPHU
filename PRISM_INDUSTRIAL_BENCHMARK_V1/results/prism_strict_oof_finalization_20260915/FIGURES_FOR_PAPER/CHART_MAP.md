# Figure map and visual QA

| 文件 | 数据来源 | 用途 | QA 结论 |
|---|---|---|---|
| `admission_margin_by_stage_task.png` | `E1_STAGEWISE/admission_margin_distribution.csv` | 比较九个公开任务/视图中 C/W/A 的逐折 admission margin | 采用 symmetric-log 轴；Metro P60 C 的极端负 fold 与其余近零/正 margin 均可见；颜色和图例可辨 |
| `false_admission_vs_sample_size.png` | `E2_SYNTHETIC/null_calibration.csv` | 展示 null stage 误纳率随样本量变化 | 坐标、阶段曲线和样本量标签可读 |
| `noise_stage_activation_probability.png` | `E6_ROBUSTNESS/noise_robustness_summary.csv` | 展示不同扰动下 C/W/A 激活概率 | 分面、图例和扰动强度可读；仅用于 compact universe |
| `noise_median_admission_margin.png` | `E6_ROBUSTNESS/noise_robustness_summary.csv` | 展示扰动下 admission margin 中位数 | 正负边界与模式差异可辨；margin 仅为报告量 |
| `n1_prediction_robustness.png` | `E6_ROBUSTNESS/n1_prediction_robustness.csv` | 展示 N1 预测层面的 Level R² 变化 | view/mode/alpha 可辨；不能与 E1 生产候选宇宙直接比较 |

全部图片由冻结 R14 表格机械生成。唯一的实验后绘图变更是提交 `4bf13cd` 中 E1 y 轴的显示尺度；底层 CSV 和报告数值未改变。
