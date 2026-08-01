# PRISM Industrial Benchmark V1 — CPU implementation

This directory implements the preregistered CPU stages C0–C6. Raw public
sources are read from a server-only data directory and are never copied into
the return bundle. All design matrices, physical operators, predictions,
metrics and bootstrap calculations use `float64`.

Typical commands from this directory:

```bash
python scripts/audit_dataset.py --dataset TEP --raw-root /path/raw_sources --registry shared/DATASET_REGISTRY/TEP
python scripts/build_shared_dataset.py --raw-root /path/raw_sources --output shared --sample-cap 50000
python scripts/run_cpu_benchmark.py --raw-root /path/raw_sources --output results_cpu --sample-cap 50000 --bootstrap 100
python scripts/build_cpu_bundle.py --project . --results results_cpu --shared shared --output return/PRISM_INDUSTRIAL_CPU_RESULTS_V1
python scripts/validate_package.py --package-dir return/PRISM_INDUSTRIAL_CPU_RESULTS_V1
```

The current first run is a CPU screening run (`sample_cap=50000` and
`bootstrap=100`). The preregistered final run must use all eligible samples
and 500 block-bootstrap replicates; screening results are not final evidence.

