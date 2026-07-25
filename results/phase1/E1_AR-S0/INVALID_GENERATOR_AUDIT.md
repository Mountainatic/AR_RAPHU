# INVALID — AR-S0 generator audit

Status: quarantined; these files are not scientific evidence and must not be
used for configuration selection, testing, aggregation, or reporting.

The 2026-07-25 pre-pruning audit found that AR-S0 used an all-zero latent
initial state and no continuing process innovation. Its stable autoregressive
recursion therefore remained identically zero. The configured observation
noise was scaled from the clean-signal variance, which was also zero.

Affected outputs:

- `B0/seed_0` through `B0/seed_9`
- `B1/seed_0` through `B1/seed_9`
- `Track-XAR/seed_0/warmup.pt` through
  `Track-XAR/seed_3/warmup.pt`

No penalty fork, validation selection, or formal test aggregation was run from
these warmups. The files are retained only for auditability. Corrected runs
must use a new result namespace and a newly frozen generator version.
