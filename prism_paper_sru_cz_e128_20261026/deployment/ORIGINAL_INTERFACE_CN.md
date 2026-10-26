# 两个探索性示例头的部署复核

`SRU_H2S_D8.pkl`为SRU H2S、直接历史16、标签成熟延迟8。
`CZ_Rod1_D4_E128.pkl`为CZ Rod1、直接历史128、标签成熟延迟4，包含8个多分辨率时滞块。

这两例按重复开发结果选出，不是正式测试或工业部署认证。其他失败设置在完整实验包中保留，不应自动部署。

将同目录加入Python模块搜索路径。依赖与原环境一致：NumPy、SciPy、scikit-learn；无需Torch训练或重新拟合。`stable_spline.py`与`streaming.py`必须随模型一起保存，前者用于恢复修正后的预处理对象。

```python
from deploy import load_predictor

# 对预测起点t，x_past为此前L-1条合法过程输入，按从旧到新排序；
# y_mature_past为截至t-delay-1的L-1条目标历史。
state = load_predictor(model_path, x_past, y_mature_past, first_origin=t)

# 每个采样点只交付当前X(t)和此时刚成熟的y(t-delay)。
prediction = state.step(x_current, y_newly_mature)
```

过程列顺序必须与源缓存一致。SRU为u1..u5；CZ依次为主加热功率、晶升速度、埚升速度、晶转速度、埚转速度。历史目标均为对应头的原始记录尺度。序列切换或采样不连续时必须新建状态，不跨晶棒或缺失区间继承状态。

`SELECTION_MANIFEST.json`列出源模型、部署文件哈希、指标与原数值容限下的逐点一致性。完整实验脚本保留实际服务器路径；其他机器完整重训前须按注册恢复数据位置和相同环境，不能直接把归档路径当成通用训练命令。
