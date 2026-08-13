# NeuroBEM sampling-rate scaling audit

This experiment asks whether the finite recursive prediction horizon identified
by R3 is governed mainly by elapsed physical time or by recursive application
count. It evaluates 100, 200, and native 400 Hz under fixed-step and fixed-time
history contracts.

Track 0 calls the frozen R3 resampler and adapter exactly. The scaling comparison
then fits new 100/200/400-Hz maps on one common native-400-Hz train support. Four
segments from one anomalous 164-Hz training flight are excluded from all three
rates by explicit user direction; they are never interpolated. Changing sampling
interval changes the fitted discrete transition, but model family, ridge grid,
route set, split, and reliability rule are unchanged. Test data are unavailable
to fitting and calibration code.
