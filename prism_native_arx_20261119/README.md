# PRISM native streaming — 2026-11-19

This snapshot implements the frozen SRU and strict CZ E128 inference profiles in C with FP64, rolling history bins and recursive least-squares feedback at label maturity. It includes a like-for-like benchmark against the frozen dense AR and ARX baselines.

| Task | RMSE | PRISM, µs/step | AR, µs/step | ARX, µs/step |
|---|---:|---:|---:|---:|
| SRU H2S, delay 8 | 0.023238 | 0.290 | 0.039 | 0.061 |
| CZ Rod1, delay 4, E128 | 0.014646 | 0.253 | 0.049 | 0.137 |
| CZ Rod2, delay 4, E128 | 0.010970 | 0.396 | 0.050 | 0.142 |

The timings measure the native kernel on one Xeon Platinum 8352V core. Accuracy uses the existing development segment. Rod2 remains 0.12% above the strongest tested Transformer. Both CZ heads use the process E128 branch.

## Run the self-contained smoke test

With Linux, GCC, Python, NumPy, pandas and scikit-learn available, run the following commands in this directory. The test generates toy models in a temporary directory and checks 768 one-step forecasts per profile against the Python reference, including AR and ARX replay.

```sh
python smoke.py
python smoke.py --portable
```

## Documentation and version identity

- [中文实现说明](IMPLEMENTATION_CN.md) covers the feature layout, label timing, RLS equations, native code and reproducibility.
- [中文实验报告](REPORT_CN.md) gives the complete accuracy, latency and ablation results.
- [Aggregate evidence and artifact identity](evidence.json) binds the original private weights to this public source snapshot.
- `benchmark.py` compiles and verifies on authorized frozen model and baseline folders. `native_predictor.py` exposes one-step calls to a compiled library.

The public snapshot contains source, aggregate results, model hashes and a synthetic example. Trained weights, generated C constants or binaries, and pointwise data and predictions are kept in the separately verified private package.
