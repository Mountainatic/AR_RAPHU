# PRISM TIM baseline addendum

This branch archives the clipped Elastic Net, dynamic PLS, and GRU comparison against the frozen PRISM Laguerre ECM. Its parent commit is `7bf849fa0c2453ea58412cc631c3cddf9b6d4e3c`, and the [source lock](SOURCE_LOCK.json) binds the original algorithm files.

| Artifact | Contents |
|---|---|
| [Clean-tail results](results/prism_tim_baselines_addendum_20261127/CLEAN.csv) | Six protocols, with PRISM and nine baselines |
| [Fault pressure results](results/prism_tim_baselines_addendum_20261127/PRESSURE.csv) | Matched 512-point oscillations and four-standard-deviation biases |
| [Paired tests](results/prism_tim_baselines_addendum_20261127/DM_HAC_HOLM.csv) | Bartlett HAC Diebold–Mariano tests with Holm adjustment |
| [Experiment notes](results/prism_tim_baselines_addendum_20261127/README.md) and [receipt](results/prism_tim_baselines_addendum_20261127/RECEIPT.json) | Scoring conventions, selected settings, and artifact hashes |
| [Benchmark runner](results/prism_tim_baselines_addendum_20261127/benchmark_addendum.py) and [TEP helper](benchmark_helpers/tep_baselines.py) | Training, replay, and scoring code |

The comparison reuses previously scored tails. The publication contains aggregate CZ results. Workbooks, row-level predictions, and fitted model binaries remain in their original archives. Exact replay requires the original prepared data, checkpoints, and baseline archives recorded in the receipts.
