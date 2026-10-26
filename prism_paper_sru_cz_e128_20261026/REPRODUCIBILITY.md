# Verification and replay

## Public package verification (standard library only)

```bash
python -B tools/verify_publication.py
```

Checks every file against FILE_MANIFEST.json, Python syntax without writing bytecode, publication scope, input attribution closure, and correspondence of metric tables. It does not read data or train models.

## Source layout

- `source/PRISM_PAPER_OPTIMIZATION_20261026`: original selected-window ablations, bounded-innovation alternative, CPU benchmarks, streaming implementation.
- `source/PRISM_PAPER_SPLINE_REPAIR_20261026`: corrected spline runtime and independently fitted SRU ablations.
- `source/PRISM_CZ_E128_20261026`: fixed128 / E128 / E16 runtimes and registered train/evaluate runner.
- `source/PRISM_CZ_E128_TRANSPORT_20261026`: additional mature-residual transport candidates; negative outcome retained.
- `source/baselines`: executed matched neural/linear runners, PRISM reference runtime, and third-party vendor sources, preserving each cached source layout. Earlier complete SRU comparison results are also inherited from `../sru_matched_window_baselines_20261025` at base commit 44ae45fe342c17fa2d76216f76fbbb8c12001877.
- `protocol/original_registrations`: byte-preserved pre-fit registrations and code/data hashes; these describe the entire original server run, including jobs deferred from this publication.
- `provenance/FROZEN_MODEL_RECEIPTS.json`: selected training configurations and original model/ledger hashes, excluding deferred Rod2 ablation receipts.

## Scientific environment and private replay

`environment/SCIENTIFIC_ENVIRONMENT.json` records Python 3.10.12 and scientific package versions from the original runtime. `requirements-scientific.txt` pins the package versions. No packages were installed for publication. The CPU timing environment is separately recorded in the complexity JSON files (NumPy 2.4.6 / Torch 2.12.1+cu130); timing is not claimed identical across machines.

Historical runners preserve their original `/root/autodl-tmp` paths intentionally, to preserve registered code hashes. They are not portable one-command training tools. Full replay needs the legally accessible, hash-bound training/development caches and original frozen checkpoints. Raw CZ data, per-sample predictions and learned weights are excluded from this public package. Model hashes identify the private artifacts but do not replace access to them.

On the original authorized cache host, to repeat the aggregate-only diagnostics:

```bash
/root/autodl-tmp/PRISM_V06_SRU_CZ_SCALE_SEARCH_20261020/.venv/bin/python -B tools/replay_diagnostics.py --base /root/autodl-tmp > diagnostics.json
```

This recovers Rod1 constrained candidate parents in RAM using original recipes, reuses frozen A, checks full predictions against saved predictions, and exports aggregate quantities only. No hyperparameter grid or A selection is rerun. It reads only training and previously used development caches; no workbook or reserved tail reader is present. It does not perform Rod2 module ablation.

Training and evaluation entry points in each historical `suite.py` are separate (`train` and `evaluate`). Do not run `prepare` or `queue` inside an immutable original experiment folder. A fresh complete retrain requires a new experiment namespace and re-registered path/hash bindings; changing paths and then ignoring the hash assertions is not a valid reproduction.

The deployment interface code is included, with original interface documentation. Model files referenced there are not included in this public archive. Their private export hashes, original numerical-equivalence checks and compact byte counts are retained in provenance; no claim is made that a fresh GitHub clone can load a private checkpoint.
