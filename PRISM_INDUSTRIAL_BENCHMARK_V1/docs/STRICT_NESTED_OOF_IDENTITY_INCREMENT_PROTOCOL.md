# PRISM strict nested-OOF identity-increment protocol

Status: frozen for the `r3-strict-oof` branch before new experiment execution.

This amendment supersedes every earlier one-SE, guarded one-SE, practical-gain,
positive-fold, confidence-interval, bootstrap-probability, coefficient-size, and
prediction-variance rule wherever such a rule could decide whether K, C, W, or
A exists. Historical result files are retained as historical evidence; they are
not silently relabelled as results of this amendment.

For each stage `b`, `H_b+` contains only nonzero constructions. The zero object
`0_b` is an external additive identity and is not a hyperparameter candidate.
An arbitrarily strong regularizer is not an implementation of `0_b`.

For outer fold `k`, tune the nonzero winner using only the other folds. Score
that frozen winner and the unchanged parent on fold `k`. Concatenate/row-weight
the paired outer-fold risks and define

```text
D_b = R_OOF(parent) - R_OOF(parent + tuned_nonzero_increment)
epsilon_num = 1000 * eps(float64) * max(1, |R_parent|, |R_child|)
```

The only routing rule is `D_b > epsilon_num`. Positive gains of 2%, 0.5%, 0.1%,
or any other magnitude above machine tolerance are ACTIVE. Otherwise the stage
returns its external identity and the parent prediction remains unchanged.

SE, CI, bootstrap probability, and fold direction are evidence fields only.
They may determine wording such as supported/not confirmed, but cannot change
`routing_status` or `final_selected_candidate`.

An identity result never stops the chain. The next stage receives the unchanged
parent prediction and its residuals and performs its own nonzero tuning and OOF
admission. Thus zero does not propagate; residuals do.

Implementation authority is the machine-readable configuration
`configs/strict_nested_oof_selection_v1.json`. The shared selector is
`src/prism_benchmark/strict_oof_selection.py`; active v2.1.1 K/C/W/A and Joint
paths call it directly. Legacy selector modules remain only so historical
protocols and artifacts can still be audited.
