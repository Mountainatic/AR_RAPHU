# CZ raw-2s H4 E3–E6 authority-aligned extension

This correction continues the private CZ study without relabeling it as a
native result of `prism-strict-oof-finalization-20260915`.  The public authority
branch explicitly blocked private CZ because the private workbook was absent.
The present run therefore has the permanent scope label
`PRIVATE_CZ_AUTHORITY_ALIGNED_EXTENSION_NOT_AUTHORITY_BRANCH_OUTPUT`.

The authority base is commit
`2ee6273b8f915cbcdff2f46d56bc80047ddae4a7`.  Before execution, the runner
requires the exact SHA-256 values of `e1e6_phase_b.py`,
`e1e6_robustness.py`, and `strict_oof_selection.py` from that commit.

## Protocol

CZ remains fixed at 2-second sampling, L256 and H/W/W0=`4/1/1`.  The target is
`D[t+3]-D[t-1]`, so the prediction is eight seconds ahead.  Rod1→Rod2 and
Rod2→Rod1 remain separate throughout.  E3–E6 read development evidence only;
formal-target and OOD access are forbidden.

## Completed evidence

Server result root:

```text
/root/autodl-tmp/PRISM_CZ_RAW2S_H4_AUTHORITY_ALIGNED_E3_E6_20260924_R5/
```

E3 completed ten seeds per direction.  Each frozen CZ C result exposes only
`joint_lift` as an active K channel.  Under the authority rule, which compares
scales only across frozen active channels, the uniform and channel-specific
candidate universes are therefore identical.  Both directions have exactly
zero paired multiscale gain.  This is a structural non-applicability result,
not a failed fit.

E4A and E4B reran the exact dataset-independent authority synthetic audit.
The aggregate CSV is byte-identical to the sealed authority aggregate.  Raw
run CSVs differ only in wall-clock fields, prediction hashes, and floating
roundoff at at most approximately `2.22e-16`.

E5 records the E3 single-channel degeneracy separately by direction and then
summarizes the authority E4 candidate-universe and family-ablation stability.

E6 uses the authority development-only compact frozen-model/re-identification
logic.  For each direction it constructs four 512-row chronological blocks
from one legal source-rod segment, with a 260-point embargo between blocks.
It perturbs raw aligned measurements before lag summaries.  It produced 900
N1 rows and 4,560 N2 rows; all numeric evidence is finite and the alpha-zero
seed-invariance certificate passes.

No private raw data, per-sample predictions, or private checkpoints are
included in this branch.

## Relationship to E1 and E2

- Valid real-CZ E1 remains at
  `/root/autodl-tmp/PRISM_CZ_RAW2S_H4_AUTHORITY_E1_E6_20260922_R3/`.
- Exact authority E2 remains at
  `/root/autodl-tmp/PRISM_CZ_AUTHORITY_BRANCH_LOGIC_20260924_R4/`.
- The invalid raw-CZ E2 adapter remains `INVALIDATED_DO_NOT_CITE`.

Thus “E1–E6 complete” means E1 real-CZ, E2 exact authority synthetic, E3/E5/E6
private authority-aligned extensions, and E4 exact authority synthetic.  It
does not mean that the public authority branch originally contained private
CZ E3–E6.
