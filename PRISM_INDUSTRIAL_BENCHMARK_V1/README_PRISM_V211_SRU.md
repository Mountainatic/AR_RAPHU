# PRISM v2.1.1 SRU implementation correction

This runner executes only the two frozen SRU heads and reuses the immutable
v2.1 baseline per-sample artifacts. It never rebuilds C1 and never retunes a
baseline.

The resumable chain is:

```text
E0R -> E1R -> E2R-K -> E2R-C -> E3R -> E4R -> E5R -> E5.5
    -> E6R -> E7R -> E8R
```

E5.5 is a hard development gate. If it fails, the runner does not enter E6R
or access candidate test data. It still writes a stop-state report and the
integrity-checked results ZIP.

On the registered 32-vCPU / 60-GiB CPU server, run:

```bash
cd /path/to/PRISM_INDUSTRIAL_BENCHMARK_V1
bash RUN_PRISM_V211_SRU.sh
```

Defaults are eight independent workers, four BLAS threads per worker, 4 GiB
per worker, and an 8 GiB reserve (32 threads and a 40 GiB declared budget).
The script uses `/root/AR_RAPHU_AUTODL/.autodl-tools/uv` and the existing
`/root/AR_RAPHU_AUTODL` uv project. All stages are marker-resumable.

The terminal package is:

```text
PRISM_V2_1_1_SRU_IMPLEMENTATION_CORRECTION_RESULTS_bundle.zip
PRISM_V2_1_1_SRU_IMPLEMENTATION_CORRECTION_RESULTS_bundle.zip.sha256
```
