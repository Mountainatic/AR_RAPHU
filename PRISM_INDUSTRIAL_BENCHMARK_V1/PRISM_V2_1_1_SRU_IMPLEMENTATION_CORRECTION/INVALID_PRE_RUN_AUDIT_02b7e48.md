# Invalid pre-run audit: 02b7e48

The server run rooted at `PRISM_V211_SRU_02b7e48` is an invalid implementation
pre-run and must not be reported as PRISM v2.1.1 scientific evidence.

- Candidate test access: `false`.
- The registered OOF input-path preservation gate passed for the SO2 Joint
  candidate.
- The implementation incorrectly allowed a post-selection, full-validation
  materialization check to override that frozen OOF decision.
- The resulting `V2_1_1_DEVELOPMENT_STOP` and package SHA256
  `6755ceb5a0d278d6cf558cabc73d21937bde9475010af159a2108daa4d22a98e`
  are therefore invalid and retained only for audit.

The correction keeps the materialized-validation calculation as a diagnostic
with `selection_eligible=false`; only the preregistered OOF gate can determine
input-path preservation.
