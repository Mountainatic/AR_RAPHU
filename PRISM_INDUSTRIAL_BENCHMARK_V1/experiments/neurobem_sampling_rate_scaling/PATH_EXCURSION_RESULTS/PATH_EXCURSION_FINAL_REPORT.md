# NeuroBEM trajectory path-excursion diagnostic

## Decision

`HIGHER_RATE_TRANSIENT_PATH_EXCURSION_MIXED`.

The stronger hypothesis that higher sampling rate uniformly increases maximum
within-window excursion is not supported. PF velocity at 100 ms has
Emax400/Emax100 = 0.928 and Joint body-rate has 0.940. At 200 ms PF velocity is
essentially equal (1.002), while Joint body-rate is only mildly larger (1.022)
and non-monotone across 100/200/400 Hz.

Joint attitude is the supported channel: its median maximum path error is
strictly ordered 400 > 200 > 100 at both 100 and 200 ms, with ratios 1.042 and
1.071. Its frozen-envelope crossing fraction also increases strongly with rate.

Crossing fractions rise more broadly and first crossings tend to occur earlier
at 400 Hz, but the rate-specific frozen calibration envelopes also become much
narrower. Therefore increased crossing frequency cannot by itself be reported
as universally larger transient amplitude. The operational reliable-horizon
contraction is partly compatible with more frequent/earlier envelope crossings,
but window-internal path excursion is not a complete route-independent
explanation.

## Provenance

Existing endpoint logs were insufficient, so the diagnostic used a pure
deterministic reproduction replay of the already accessed 12 test trajectories.
It reused frozen models, configs, identities, anchors, sampling operators, and
R3/R4 calibration bounds. No model fitting, threshold change, stabilization,
selection, tuning, or new formal test decision access occurred.
