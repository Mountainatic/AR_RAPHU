# INVALIDATED CZ RUNS — DO NOT CITE

Every artifact below is retained only as forensic evidence explaining why an
older result was rejected.  None of the metrics in this directory may be used
as a PRISM result, compared with the corrected authority run, or quoted in a
paper/table.

## Why these runs are invalid

The older private runners described their stages as "PRISM-like" or as explicit
auditable feature blocks.  They did not execute the K/C/W/A/Joint modules from
the `prism-strict-oof-finalization-20260915` authority line.  Some reports also
contain ordinary `RIDGE` models.  A strict selector around a surrogate feature
pipeline does not make that pipeline the authoritative PRISM implementation.

The two `EXEC_20260923_R1/R2` directories are incomplete diagnostic attempts:
they stopped while exposing and correcting the
`BEST_ACTIVE_K_CHANNEL` final-checkpoint binding defect.  They are not result
runs.

## Invalidated server directories

| Directory | Approximate size before deletion | Disposition/reason |
|---|---:|---|
| `PRISM_CZ_RAW2S_L256_H4_E1_E6_20260921_R2` | 37 MiB | Private custom adapter; explicitly not a byte-for-byte authority rerun |
| `PRISM_CZ_RAW2S_L256_H4_E1_E6_STRICT_AUTH_20260921_R1` | 245 MiB | Surrogate/PRISM-like experiment family despite the directory name |
| `PRISM_CZ_RAW2S_L256_H4_E1_E6_STRICT_AUTH_20260921_R2` | 209 MiB | Same non-authority experiment family |
| `PRISM_CZ_RAW2S_L256_H4_E1_E6_STRICT_AUTH_20260921_R3_PARALLEL` | 1.7 GiB | Parallelized non-authority family; `PASS` labels are not authority evidence |
| `PRISM_CZ_RAW2S_L256_H4_E1_E6_CORRECTED_20260922` | 250 MiB | Custom feature-block/Ridge-PCA correction, not authority K/C/W/A/Joint |
| `PRISM_CZ_RAW2S_H4_AUTHORITY_E1_E6_EXEC_20260923_R1` | 20 KiB | Incomplete pure-K certificate attempt |
| `PRISM_CZ_RAW2S_H4_AUTHORITY_E1_E6_EXEC_20260923_R2` | 8.2 MiB | Incomplete enum-binding diagnostic attempt |

The duplicate obsolete code worktrees
`PRISM_CZ_RAW2S_E1E6_CODE_20260921` and
`PRISM_CZ_RAW2S_RERUN_CODE_20260920` are also removable because their relevant
history is already in GitHub; they are not copied here.

## What is archived

Only small, public-safe forensic material is included: top-level status and
summary files, experiment status files, code-audit summaries, and the R3
privacy-audited aggregate report.  Raw CZ workbooks, C1 arrays, checkpoints,
Parquet predictions, per-sample data, and caches are deliberately excluded.

The server-side transfer archive was
`INVALIDATED_CZ_PUBLIC_EVIDENCE_20260923.tar.gz` with SHA-256:

```text
46c20945d75b8d8383a5acb8b3beecaaf34bfdfd84fbc04191f6deb3188d00be
```

## Correct replacement

Use `docs/CZ_RAW2S_H4_AUTHORITY_E1_EXECUTION_20260923.md` and the server output:

```text
/root/autodl-tmp/PRISM_CZ_RAW2S_H4_AUTHORITY_E1_E6_EXEC_20260923_R3/
```

That run uses raw-2s `L256/H4/W1/W0=1`, the authority modules plus pinned
reviewed corrections, and passes the pure-K/C/W exact-identity certificates.
