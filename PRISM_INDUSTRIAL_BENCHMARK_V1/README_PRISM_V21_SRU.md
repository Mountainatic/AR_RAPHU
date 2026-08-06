# PRISM v2.1 SRU stagewise-routed execution

The implementation is restricted to `SRU_H2S__H5__W1` and
`SRU_SO2__H5__W1`. It preserves the complete C1 shared-data base and writes
only to `results_prism_v2_1_sru`.

## Server preflight (does not start E0)

```bash
cd /path/to/PRISM_INDUSTRIAL_BENCHMARK_V1
export PYTHONPATH="$PWD/src"
/root/AR_RAPHU_AUTODL/.autodl-tools/uv run --no-sync \
  --project /root/AR_RAPHU_AUTODL \
  python -m pytest tests -q
```

The frozen baseline root must contain sample-level `validation.parquet` and
`test.parquet` predictions in the existing V2 prediction layout. E6 reads
validation predictions to freeze the best baseline and records opaque hashes
for test predictions. Test rows are first parsed at E7, after the final freeze.
An aggregate-only baseline archive is intentionally rejected.

## Full automatic chain

```bash
export PRISM_V21_SHARED_ROOT=/root/autodl-tmp/PRISM_SHARED_DATA_C1
export PRISM_V21_BASELINE_ROOT=/path/to/frozen/sample_level_predictions
export PRISM_V21_OUTPUT_ROOT="$PWD/results_prism_v2_1_sru"
export PRISM_V21_THROUGH_STAGE=e8
bash RUN_PRISM_V21_SRU.sh
```

The chain is resumable by PASS markers and runs in the fixed order:

```text
E0 -> E1 -> E2-K -> E2-C -> E3-W -> E4-A -> E5-Joint -> E6 -> E7 -> E8
```

Set `PRISM_V21_THROUGH_STAGE=e6` to develop and freeze without opening test.
E7 refuses access unless `V21_SRU_FINAL_FREEZE_MANIFEST.json` is valid. E8
creates metrics, entity metrics, the 500-replicate paired moving-block
bootstrap with Holm correction, audit JSONL files, a final report, manifests,
and the verified ZIP bundle.
