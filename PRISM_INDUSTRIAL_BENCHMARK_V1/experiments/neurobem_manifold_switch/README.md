# NeuroBEM manifold-aware PRISM switching R1

This extension leaves the PRISM estimator and its frozen global Track-B
contracts unchanged. It evaluates causal local-validity monitoring, known
local-model switching, and post-alarm local re-identification. Detector
normalizers and thresholds use validation only. Each CSV is an isolated
entity; history never crosses trajectory boundaries.

The frozen release manifest contains 236/11/12 released CSV identities while
the paper reports 67/17/12 trajectories. This discrepancy is retained and
reported; identities are not silently dropped to make the counts agree.

An alarm based on an innovation can affect only the next prediction. Known
switching causally re-synchronizes the local state history from observations
available at the switch time. Detector-only ablations never re-synchronize.

