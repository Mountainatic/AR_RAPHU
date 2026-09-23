# CZ E2--E6 streaming execution amendment (2026-09-23)

Status: **ACTIVE EXECUTION AMENDMENT**.

The parent plan estimated 200--300 GiB of simultaneous private retention and
required 300 GiB free before E2--E6.  The selected server has a 50 GiB private
data disk.  That capacity gate is physically impossible on this host, even
after deleting obsolete runs.

This amendment changes only scheduling and retention.  It does **not** change
the raw-2-second CZ task (`L=256`, `H/W/W0=4/1/1`), the two independent
directions, the authority implementation, candidate universes, seeds,
selection rule, evidence boundary, metrics, or Phase-A decision rule.

Each experiment is executed as isolated units keyed by experiment, direction,
regime or condition, and seed.  A unit is collected in deterministic order.
Its certificates, finite metrics, and content hashes must pass before its
aggregate row is committed.  Only then may its scratch and row-level
intermediates be deleted.  Aggregate tables, manifests, checkpoints required
for replay, and privacy audits remain private and retained.

The live low-water marks are 8 GiB on the persistent private disk and 8 GiB on
scratch.  A unit may use at most 4 GiB persistent space and 12 GiB scratch.
At most two outer units may run concurrently, and two-way execution is not
allowed until a one-worker/two-worker prediction-hash equivalence test passes.

Public GitHub uploads may contain only aggregate non-row-level evidence,
protocol/status documents, sanitized hash manifests, and privacy audits.  Raw
CZ data, per-sample predictions, caches, and checkpoints derived from private
CZ data remain forbidden.  Obsolete invalid runs may be deleted from the
server only after a public-safe forensic archive has been marked
`INVALID_DO_NOT_CITE`, hashed, uploaded, and the exact deletion paths have been
allowlisted.
