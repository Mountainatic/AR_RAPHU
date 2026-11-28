# PRISM TIM experiment archive

The [source lock](SOURCE_LOCK.json) binds the PRISM Laguerre ECM snapshot at commit `7bf849fa0c2453ea58412cc631c3cddf9b6d4e3c`. Use the final tables and architecture evidence below for the manuscript.

| Final material | Contents |
|---|---|
| [Manuscript tables](results/prism_tim_final_tables_20261127/README.md) | Expanded DPLS and GRU search, three-metric tables, signed significance markers, and nine-way Holm adjustment |
| [IEEE LaTeX tables](results/prism_tim_final_tables_20261127/IEEE_TABLES.tex) | Three clean-tail tables and six matched fault tables |
| [Architecture evidence](results/prism_tim_architecture_evidence_20261128/README.md) | Six-row ablation, frozen channel attribution, support recovery, and zero-leakage probes |
| [Native complexity evidence](results/prism_tim_architecture_evidence_20261128/MEMORY_SPAN.pdf) | Xeon history-span sweep, million-step checks, and deployment inventory in the accompanying evidence directory |

## Earlier baseline addendum

These files preserve the initial clipped Elastic Net, dynamic PLS, and GRU comparison.

| Artifact | Contents |
|---|---|
| [Clean-tail results](results/prism_tim_baselines_addendum_20261127/CLEAN.csv) | Six protocols, with PRISM and nine baselines |
| [Fault pressure results](results/prism_tim_baselines_addendum_20261127/PRESSURE.csv) | Matched 512-point oscillations and four-standard-deviation biases |
| [Paired tests](results/prism_tim_baselines_addendum_20261127/DM_HAC_HOLM.csv) | Bartlett HAC Diebold–Mariano tests with Holm adjustment |
| [Experiment notes](results/prism_tim_baselines_addendum_20261127/README.md) and [receipt](results/prism_tim_baselines_addendum_20261127/RECEIPT.json) | Scoring conventions, selected settings, and artifact hashes |
| [Benchmark runner](results/prism_tim_baselines_addendum_20261127/benchmark_addendum.py) and [TEP helper](benchmark_helpers/tep_baselines.py) | Training, replay, and scoring code |

The comparison reuses previously scored tails. The publication contains aggregate CZ results. Workbooks, row-level predictions, and fitted model binaries remain in their original archives. Exact replay requires the original prepared data, checkpoints, and baseline archives recorded in the receipts.
