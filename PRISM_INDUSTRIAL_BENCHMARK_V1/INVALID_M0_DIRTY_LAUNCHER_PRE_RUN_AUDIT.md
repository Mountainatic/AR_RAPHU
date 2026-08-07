# Invalid pre-run audit: launcher log inside worktree

Date: 2026-08-07 (Asia/Shanghai)

The namespace
`results_prism_v2_1_1_metro_p60_w_audit_INVALID_M0_DIRTY_LAUNCHER_PRE_RUN_20260807`
is an immutable invalid orchestration pre-run and is excluded from formal
scientific evidence.

- Source HEAD: `4d44edf422827eb39fa3e73daf60b4e25ce5401c`.
- The run stopped in M0 before regression tests, fitting, or training.
- All inheritance/data checks passed except `source_tree_clean`.
- The dirty-tree observation was caused only by the outer launcher log and PID
  files being placed under the Git worktree root.
- Development was not frozen; M1--M8 were not run.
- Candidate test and OOD data were not accessed (`test_accessed=false`,
  `ood_accessed=false`).

The retry places launcher logs outside the Git worktree. No scientific model,
data, split, numerical setting, seed, selection rule, or access order changed.
