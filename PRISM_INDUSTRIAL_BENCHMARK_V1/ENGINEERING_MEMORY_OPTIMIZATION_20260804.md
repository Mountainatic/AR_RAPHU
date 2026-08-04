# PRISM V2 CPU memory and throughput optimization

Status: implemented on `prism-v2-modular-cpu-memory-opt`.

This change is an engineering-only optimization of the numerically frozen PRISM V2 CPU
protocol. It does not alter immutable C1 sample IDs, train/validation/test boundaries,
candidate grids, FP64 requirements, one-SE rules, activation gates, or test-access order.

## Preserved pre-optimization evidence

Before changing the implementation, the 184 existing V2 channel results were copied into
an independently archived snapshot:

```text
/root/autodl-tmp/PRISM_V2_PREOPT_SNAPSHOT_20260804_184.tar
SHA256 3dd06e67906b13e8f9a9d53d287d20e8699f1519b38d969a43e78a88b4e23b3f
```

The snapshot contains 177 `PASS` results and 7 protocol-retained solver failures. The
active result directory is resumed in place, while the archive remains independent.

## Implemented changes

1. A validation-scope `BaseAccessor` now serves both train and validation samples. The old
   implementation loaded the training base twice inside every channel process.
2. V2 channel base arrays are loaded one dataset at a time in the parent and inherited by
   that dataset's Linux pool through copy-on-write pages. Dataset pools are ordered by
   decreasing pending-task count. Prefix statistics stay lazy so the parent never holds
   every channel prefix or every dataset simultaneously.
3. Prefix sums are cached per entity/channel instead of rebuilt for every candidate and
   fold.
4. V2 channel sample parquet reads request only the eight frozen columns actually used by
   the stage.
5. Centered FP64 ridge systems are accumulated as chunked Gram/RHS/sum statistics. This
   avoids allocating a second centered design matrix. Full Urysohn transforms and final
   predictions are also chunked.
6. Every process task runs final garbage collection and glibc `malloc_trim(0)` to prevent
   persistent workers from retaining previous high-water arenas.
7. All V1/V3--V8 and baseline pools use the 60 GiB cgroup limit, current usage, a 4 GiB
   reserve, and a stage-specific per-worker estimate to resolve safe concurrency. The
   V2 pool is additionally capped at 20 stable workers; other stages request all 31 CPU
   workers and are clamped only when memory or pending-task count requires it.
8. An optional Rust/Rayon prefix kernel is available with a complete Python fallback.
   Process-level parallel execution freezes `RAYON_NUM_THREADS=1` to avoid nested
   oversubscription. Rust dependencies and PyPI build tools use Tsinghua mirrors.
9. Every Linux pool worker installs `PR_SET_PDEATHSIG=SIGTERM` and verifies its parent PID,
   so it cannot survive as a PPID-1 orphan if the pool parent dies.
10. The chain uses `set -euo pipefail` and checkpoint-preserving bounded retries. A stage
    advances only after a zero exit; persistent failure still stops the chain.

## Verification

The server environment passed all tests with the compiled Rust extension:

```text
71 passed
```

Two end-to-end recomputations were compared with pre-optimization result files:

| Case | prediction rows | maximum absolute difference | selection fields | prediction SHA256 |
|---|---:|---:|---|---|
| SRU `u1` | 982 | 0.0 | identical | identical |
| TEP `xmeas_5` | 625,600 | 0.0 | identical | identical |

The compared selection fields were profile, retained profiles, family, lag/amplitude
resolution, penalties, active flag, and status. The TEP comparison exercises the heavy
sample path and the Rust-enabled prefix implementation.

## Runtime telemetry

Every pool emits a `PRISM_PROCESS_POOL_START` JSON line containing requested and resolved
workers, job count, cgroup limit/current bytes, and the per-worker budget. This makes later
resource decisions auditable from stage logs without changing scientific outputs.

The first all-dataset pressure launch showed that immutable sample parquet frames remain
private and reached the 60 GiB cgroup limit at 29 workers (no OOM or OOM kill). A later
23-worker cross-dataset run produced results but its large parent disappeared without a
cgroup OOM event, leaving active PPID-1 workers that could no longer refill the queue. The
orphan process group was stopped after preserving 257/322 results. This observation caused
the cross-dataset parent design to be rejected: V2 now runs one dataset at a time, caps at
20 workers, kills workers with their parent, and resumes completed channel files on retry.
