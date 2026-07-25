# STAGE1 DUAL SOLVER V20

V20 closes the two S0 optimization lines in one reproducible package.

## Line A: exact operator-accelerated KAN

The scientific model remains

\[
\hat y_t=b+\sum_j\sum_\tau q_{j\tau}f_j(x_{j,t-\tau}).
\]

Implemented improvements:

- evaluate each KAN response once on the unique raw timeline;
- exact grouped causal convolution for static Gamma lag kernels;
- vectorized per-variable KAN execution on CUDA;
- cached train/validation/test tensors on device;
- reduced validation, diagnostic and CPU synchronization frequency;
- one proximal branch-norm pass per epoch;
- one dense warmup per seed;
- **independent** pruning forks from the same warmup checkpoint;
- cross-seed validation-only one-standard-error selection;
- clean prediction, support, delay, response and contribution recovery outputs;
- CUDA profiler and peak-memory audit.

V19 used a nested homotopy path. That was fast but path-dependent. V20 keeps the
shared warmup while eliminating propagation of an incorrect earlier support.

## Line B: variational distributed-lag spline

Each response is replaced by an explicit cubic B-spline expansion:

\[
f_j(x)=\sum_m c_{jm}B_{jm}(x).
\]

For fixed Gamma delay kernels the response problem is convex:

\[
\min_{b,c}\frac{1}{2T}\|y-b-\Phi(q)c\|_2^2
+\lambda_g\sum_j\|c_j\|_2
+\frac{\lambda_s}{2}\sum_j c_j^\top R c_j.
\]

Implemented improvements:

- group-scale-normalized design blocks;
- monotone restarted FISTA;
- explicit spline second-difference roughness matrix;
- proximal-gradient/KKT stopping audit;
- warm-started inner solves;
- low-dimensional Gamma delay block optimization with Adam and focused strong-Wolfe L-BFGS probes;
- seed-0 validation screen followed by five-seed formal candidates;
- cross-seed one-standard-error selection;
- clean and noisy from-scratch evaluation;
- support, delay distribution, function grid, spline coefficient, contribution,
  KKT, outer-history and runtime artifacts.

PyTorch is only a tensor/autodiff backend for this line; a GPU is optional. The
formal jobs are deliberately independent so several can run concurrently.

## Raising GPU utilization

A single Stage1 model is tiny. DDP is the wrong tool. V20 instead runs several
independent jobs on the same GPU. The default launcher uses four workers per
GPU:

```powershell
.\RUN_V20_FULL.ps1 -Devices "0" -WorkersPerGpu 4
```

For an RTX 5080 Laptop GPU, start with 4. If telemetry shows mean utilization
below 60% and peak memory below 60%, retry with 5 or 6 workers. Separate CUDA
process contexts consume more memory, but the model/data tensors themselves are
small.

Telemetry is written under:

```text
results_stage1/STAGE1_DUAL_SOLVER_V20/job_records/*/gpu_telemetry.csv
```

Summarize one telemetry file with:

```powershell
python tools/summarize_gpu_telemetry.py --csv <gpu_telemetry.csv>
```

## Full run

Run from the project root:

```powershell
.\RUN_V20_FULL.ps1 -Devices "0" -WorkersPerGpu 4
```

The script performs, in order:

1. clean old V20 outputs;
2. environment capture;
3. 118-test regression suite;
4. CUDA profiling;
5. five KAN warmups in a GPU job pool;
6. 45 independent KAN pruning/refit forks;
7. cross-seed KAN aggregation;
8. 69 variational screen jobs (63 Adam grid points + 6 focused L-BFGS probes);
9. validation-only candidate creation;
10. five-seed formal variational jobs;
11. clean aggregation;
12. noisy selected-config jobs from scratch;
13. final comparison and gate report;
14. manifest/hash/checkpoint validation and ZIP packaging.

To skip the profiler:

```powershell
.\RUN_V20_FULL.ps1 -Devices "0" -WorkersPerGpu 4 -SkipProfiler
```

## Expected returned file

After a successful run, return only:

```text
STAGE1_DUAL_SOLVER_V20_RESULTS_bundle.zip
```

The ZIP contains all code, data manifests, selected and candidate checkpoints,
job logs, telemetry, predictions, lag distributions, function grids,
contributions, FISTA/KKT histories, final comparison, SHA256 manifest and the
complete report.

## Important scientific boundaries

- Configuration selection never reads support truth, delay truth, function
  truth or test metrics.
- Truth is used only after a configuration has been fixed.
- Smoke runs prove execution, not scientific recovery.
- Exact KAN convolution remains the main compatible model.
- The variational line becomes a replacement candidate only if its formal
  support, delay, response, contribution and convergence gates pass.
