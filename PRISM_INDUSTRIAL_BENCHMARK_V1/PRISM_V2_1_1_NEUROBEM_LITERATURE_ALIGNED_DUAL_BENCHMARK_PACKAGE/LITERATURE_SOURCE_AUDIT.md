# Literature source audit

All entries were read from primary papers, official repositories, or official
project downloads before PRISM model implementation. Frozen hashes and commits
are recorded in `LITERATURE_SOURCE_AUDIT.json`.

## Track A

The official NeuroBEM loader uses body linear velocity, body angular velocity,
and four motor speeds. Its labels are six residual force/torque components.
Total predicted force/torque is the BEM base output plus that residual. The
published configuration fixes `history_len=20`; at the dataset cadence of
400 Hz this is 50 ms. The official test list contains 13 processed segments.

HDVIO2.0 Table I reports pooled RMSE. `Fxy` and `Mxy` pool the first two axes;
`F` and `M` pool all three axes. This interpretation is numerically consistent
with every rounded aggregate in the table. Exact-comparison status remains
conditional on reproducing the NeuroBEM row from official prediction files.

NeuroMHE uses related NeuroBEM material but does not satisfy the complete
target/frame/test/metric/information equivalence gate, so it is
`RELATED_BUT_NOT_DIRECTLY_COMPARABLE` and excluded from the primary ranking.

## Track B

The official repository resets each CSV time to zero, mean-resamples into
0.01-second bins, extracts `[v,q,omega,u]`, and scales motor speed by `1e-3`.
The paper and code fix `H=20`, `U=10`, and evaluation `T=60`. The official data
release contains 12 named test CSVs matching the paper's trajectory table.

The manuscript reports 67/17/12 train/validation/test trajectories, while the
downloaded release stores 236 train segment CSVs, 11 validation segment CSVs,
and 12 named test CSVs. Because segment files are not a documented one-to-one
encoding of manuscript flight counts, both the manuscript counts and exact
release file identities must be retained; reconciliation is not inferred.

The official decoupled rollout teacher-forces the complementary state: the
velocity branch receives measured future attitude and the attitude branch
receives measured future linear/angular velocity. The registered PRISM Track B
contract is stricter and forbids all future measured states. Hence comparisons
share source data, H/U/T, and metrics, but are not an exact information-set
ranking.
