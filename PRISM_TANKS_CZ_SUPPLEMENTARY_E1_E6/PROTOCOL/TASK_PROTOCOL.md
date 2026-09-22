# Task protocol

- PRISM base: strict nested OOF finalization branch.
- Tanks primary task: direct H16 from the official estimation/validation split,
  with observations through `origin` and target `y[origin+16]` (64 seconds).
- Tanks E1 uses the frozen K/W adapter. Corrected E2–E6 use protocol
  `PRISM_V211_CASCADED_TANKS_E2_E6_H16_STRICT_OOF_20260922_R3` and live in
  `TANKS_H16_CORRECTED_R3/`.
- CZ primary task: raw 2-second L256, H4/W1/W0=1, independently for both rod
  directions. The 1/2/4/8/16 H-scan is completed; corrected E1–E6 code is
  present but its R2 private-data rerun is not yet claimed complete.
- Historical CZ D20 and the superseded Tanks E2–E6 R2 outputs are excluded
  from the current primary protocol.
