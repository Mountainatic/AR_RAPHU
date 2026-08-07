# Invalid inner-parallelism pre-run audit

Status: `INVALID_RUNTIME_PRE_RUN_ISOLATED`.

The formal namespace was intentionally stopped during M2 after 23 of 27 K
channels had completed. Four long-running channels remained. The stop was
requested only to introduce a throughput scheduler that splits independent
candidate and inner-fold fits across bounded, deterministic worker threads.

This partial namespace is not scientific evidence: M2 did not finish, no
development freeze was written, and no test or OOD data was accessed. The 23
completed results and chain log are retained for execution audit only and will
not be mixed with the restarted formal namespace.

The new scheduler preserves the registered job order when collecting results,
uses the same FP64 fits, row caps, seeds, penalties, certificates, and selection
rules, and only changes task-level throughput. Accessor prefixes are warmed
before concurrent read-only evaluation so the parallel and serial paths are
equivalent. The restarted run records outer task workers and inner candidate
workers separately in runtime telemetry.
