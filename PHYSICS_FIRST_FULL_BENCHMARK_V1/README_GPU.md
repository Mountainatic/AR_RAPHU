# PHYSICS-FIRST K → Residual-AR GPU Benchmark V1

This is the GPU companion to the frozen L6 CPU benchmark. It reads only the immutable shared package and never rebuilds splits, targets, PCA, scalers, breakpoints, or the L6 time scale.

Target image: PyTorch 2.8.0, Python 3.12, CUDA 12.8, RTX 5090 32 GB. The formal main model remains the CPU `K → Residual AR`; GPU models are competitive baselines and ablations. Names ending in `_adapted` are compact protocol-faithful adaptations, not exact upstream paper reproductions.

## Server checkout

```bash
cd ~/autodl-tmp
git clone https://github.com/Mountainatic/AR_RAPHU.git
cd AR_RAPHU
git checkout physics-first-gpu-benchmark-v1
cd PHYSICS_FIRST_FULL_BENCHMARK_V1
bash SETUP_GPU_ENV.sh
source .venv-gpu/bin/activate
```

The repository is private, so GitHub may request your username and a fine-grained personal access token.

## Shared package

```bash
mkdir -p ~/autodl-tmp/physics_shared
cp /path/to/SHARED_BENCHMARK_DATASET_bundle.zip* ~/autodl-tmp/physics_shared/
cd ~/autodl-tmp/physics_shared
sha256sum -c SHARED_BENCHMARK_DATASET_bundle.zip.sha256
unzip -q SHARED_BENCHMARK_DATASET_bundle.zip
```

## Smoke test

```bash
cd ~/autodl-tmp/AR_RAPHU/PHYSICS_FIRST_FULL_BENCHMARK_V1
python scripts/validate_shared_dataset.py --shared ~/autodl-tmp/physics_shared/SHARED_BENCHMARK_DATASET
python scripts/run_gpu_smoke.py \
  --shared ~/autodl-tmp/physics_shared/SHARED_BENCHMARK_DATASET \
  --device cuda:0 --seeds 0 --epochs 2 --no-strict-folds
```

## Formal core and frontier run

```bash
tmux new -s physics_gpu
cd ~/autodl-tmp/AR_RAPHU/PHYSICS_FIRST_FULL_BENCHMARK_V1
source .venv-gpu/bin/activate
bash RUN_GPU.sh \
  --shared ~/autodl-tmp/physics_shared/SHARED_BENCHMARK_DATASET \
  --device cuda:0 \
  --screening-seeds 0,1,2,3,4 \
  2>&1 | tee results_gpu/logs/full_run_console.log
```

Detach with `Ctrl+B`, then `D`; re-enter with `tmux attach -t physics_gpu`. Monitor with `tail -f results_gpu/logs/core.log` and `watch -n 2 nvidia-smi`.

Completed direction/model/seed tasks are skipped automatically on resume:

```bash
bash RESUME_GPU.sh \
  --shared ~/autodl-tmp/physics_shared/SHARED_BENCHMARK_DATASET \
  --screening-seeds 0,1,2,3,4
```

K-residual GRU/TCN/Transformer tasks require frozen train rolling-OOF K predictions plus test K predictions. Supply `--cpu-results .../results_cpu`; if train OOF predictions are absent, the code fails closed rather than using leakage-prone in-sample residuals.

The launcher validates hashes, runs tests, uses four chronological folds with the frozen purge, saves per-sample predictions, aggregates rankings, creates a manifest, builds the result ZIP and SHA256, and verifies an independent unzip round trip.
