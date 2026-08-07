# K inner-parallel equivalence audit

Status: `PASS`.

The scheduler uses 8 outer K channel processes and 4 ordered inner candidate
threads per channel. Every BLAS/OpenMP backend remains fixed to one thread.
Candidate jobs are independent; their results are collected in the original
registered order before any one-SE, activation, stability, or ridge decision.
Accessor prefixes are fully warmed before the concurrent read-only calls.

The frozen regression suite passed with `62 passed`, including explicit serial
versus parallel ordering and ridge-selection equivalence tests.

Two real Metro-P60 validation-only K channels were recomputed with four inner
workers and compared against the isolated one-inner-worker pre-run:

| Channel | Path | Serial seconds | Parallel seconds | Validation SHA256 | OOF SHA256 |
|---|---|---:|---:|---|---|
| `Pressure_switch` | exact-zero selection | 52.8524 | 51.3558 | `3f26a7dcc822bf470ce7c665d03886b29d6b421df768a5e989055f3239229e9f` | `ad3d7cf44ae5bf55feacb9114c368986f48360f29103974a8d3f2fbc42046112` |
| `COMP` | active linear distributed lag | 315.3022 | 210.8686 | `0f76f19cfad17507b87f28300fff4a3f807f4061bdf26a2adc63f01065b2cfda` | `ca24452c4430f352d6cc5cb91bc1d4dde36514a72fc924efcec045bb0bdf0573` |

For both channels, the serial and parallel validation Parquet hashes and OOF
Parquet hashes match byte for byte. The selected candidate, fold losses,
contracts, prediction loss, dtype, rows, and candidate binding also match. The
equivalence run did not read test or OOD data and is retained only as runtime
verification, not scientific evidence.
