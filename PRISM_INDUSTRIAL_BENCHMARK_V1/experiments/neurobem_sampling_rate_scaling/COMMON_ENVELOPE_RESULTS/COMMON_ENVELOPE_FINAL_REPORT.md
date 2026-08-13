# NeuroBEM common-envelope reliability audit

## Decision

`ENVELOPE_NORMALIZATION_EXPLANATION_SUPPORTED`.

Using each rate's original frozen calibration bounds reproduces the prior
physical-horizon contraction: PF_KCW is 200/100/50 ms and J_KCW is 100/50/10
ms at 100/200/400 Hz. Replacing only the evaluator threshold with each route's
already-frozen 100-Hz bounds yields PF_KCW = 200/200/200 ms and J_KCW =
100/100/100 ms under the unchanged nested 90% reliability rule.

The rate-specific bounds shrink almost in inverse proportion to rate. At 400
Hz, PF velocity is 24.4% of its 100-Hz bound, J attitude 24.9%, and J body-rate
25.7%. At 200 Hz the corresponding ratios are 49.4%, 50.2%, and 51.0%.

Absolute maximum and RMS path errors at 100/200 ms remain close across rates.
Common-envelope crossing fractions likewise become nearly rate invariant for
the key channels, whereas rate-specific crossing fractions rise sharply. The
previous reliable-horizon shrinkage is therefore explained predominantly by
rate-dependent calibration-envelope normalization, not proportional worsening
of the absolute predictive time horizon.

This is an evaluation audit, not a change to any registered envelope. Original
rate-specific results remain valid under their own protocol. Existing summaries
lacked per-step paths, so one deterministic reproduction replay was required;
it reused the frozen test identities and models without fitting, selection,
tuning, stabilization, or new formal test decision access.
