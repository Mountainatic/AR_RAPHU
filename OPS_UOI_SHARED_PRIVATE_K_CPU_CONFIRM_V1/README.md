# OPS-UOI Shared–Private K CPU Confirm V1

This directory implements the frozen L6 CPU FP64 confirmation protocol.
It consumes the already-published shared dataset and frozen CPU/GPU
prediction bundles. It never reads or packages the original Excel files.

The scientific order is E0 through E8. Each stage writes an atomic checkpoint
and may be resumed.

```bash
bash RUN_CPU_CONFIRM.sh \
  --shared /path/to/SHARED_BENCHMARK_DATASET_bundle.zip \
  --cpu-baselines /path/to/PHYSICS_FIRST_CPU_RESULTS_bundle.zip \
  --gpu-baselines /path/to/PHYSICS_FIRST_GPU_RESULTS_bundle.zip \
  --config configs/frozen_l6.yaml \
  --n-jobs 20 \
  --bootstrap-jobs 16
```

All linear algebra and predictions are FP64. BLAS threads are fixed to one;
parallelism is at the task level.
