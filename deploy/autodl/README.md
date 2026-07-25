# AutoDL RTX 5090 运行说明

此包只包含 Phase 1 合成与后续公开数据工作所需代码、断点和结果。
私有 `实验数据1.xlsx` 与 CZ 数据不会被打包或上传。

## 一次性部署

```bash
tar -xzf AR_RAPHU_AUTODL_*.tar.gz
cd AR_RAPHU_AUTODL
bash deploy/autodl/bootstrap.sh
bash deploy/autodl/verify_server.sh
```

验证必须完成根项目公共测试、V20 的 118 项回归、CUDA/5090 检查和 MPS
检查。验证失败时不要启动科学训练。

## 后台断点续跑 E2

```bash
bash deploy/autodl/launch_e2_detached.sh
bash deploy/autodl/status.sh
```

默认调度为 8 个 CUDA worker，每个 worker 3 个 CPU intra-op 线程，
最多占用 24 个 CPU 核。可在启动前覆盖：

```bash
export AR_RAPHU_GPU_WORKERS=8
export AR_RAPHU_CPU_THREADS_PER_WORKER=3
```

脚本会在 uv 环境内重新生成 manifest，避免沿用本机 Conda Python
绝对路径；已有成功 `DONE.json` 会跳过，中断任务会重跑。X 和 XAR
都完成 validation-only 选择后，才会依次打开各自 test。

## 停止与取回结果

```bash
bash deploy/autodl/stop.sh
bash deploy/autodl/collect_results.sh
```

`stop.sh` 向整个训练进程组发送 TERM，并停止 MPS。再次运行启动脚本会
按成功记录继续。
