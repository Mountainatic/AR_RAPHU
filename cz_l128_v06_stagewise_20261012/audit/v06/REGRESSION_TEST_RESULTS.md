# Phase 5 — 回归执行证据

命令：`CZ_AUTHORITY=/root/autodl-tmp/CZ_CROSSROD_L128_20261008/code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/test_discrete_v06.py test_l128_adapter.py -q`。
耗时 3.33555253999657 秒，退出码 0。原始日志保留在 TEST_EXECUTION.json，初次未通过的测试夹具/导入问题保留 TEST_ATTEMPT*.json；修复没有修改模型代码。

```text
...............................                                          [100%]
31 passed in 2.51s

```

验收：不可变合同JSON、可选默认、H1/H4与权威_sample_rows两方向逐值相等、手工索引、短段/缺口/未来信息、数学与alias、逐文件byte一致均被测试。真实BaseAccessor + 手工冻结非线性 finite Urysohn K/C/W/A合同（不fit）对比实际服务器旧入口所属代码副本与新增模块版，ADMITTED/BYPASSED/NOT_APPLICABLE/UNRESOLVED各route同样本逐值比较、RMSE相同；审计开启后再预测。注册容差：array_equal，最大绝对差为0；不是与论文正式重跑结果的比较。

限制：当前无最终L128主模型冻结/最终post-A样本外输出，故不能验收其真实测试RMSE或全域白化。没有额外H4主模型训练、没有新target测试、没有仪表ADC认证、没有正式E1–E6重跑完成证据。这些均不伪称已完成。

增量离线诊断命令 `python run_discrete_audit.py --old-root /root/autodl-tmp/CZ_CROSSROD_L128_20261008 --authority /root/autodl-tmp/CZ_CROSSROD_L128_20261008/code`，报告生成前实际耗时 21.549 秒；完整总耗时见 AUDIT_EXECUTION.json。没有长时搜索/recursive OOF/refit；30分钟诊断预算以该离线诊断任务的计时和测试日志记录。

