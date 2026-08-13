# NeuroBEM composition-consistency audit

## Decision

`COMPOSITION_CONSISTENCY_MIXED`.

For both PF_KCW and J_KCW, mean one-step MSE strictly improves with sampling
rate: 400 Hz < 200 Hz < 100 Hz. Cross-rate terminal prediction defect relative
to the 100-Hz map increases with rate separation and physical horizon, directly
showing that the frozen discrete maps are not exact semigroup-equivalent
representations of one common learned flow.

The stronger registered failure pattern—GT composition error strictly ordered
400 Hz > 200 Hz > 100 Hz at 100 ms—is not supported uniformly. At 100 ms PF
improves slightly with rate, while Joint is non-monotone. At 500 ms PF worsens
with rate, whereas Joint does not. The scientifically correct result is thus
mixed: measurable cross-rate composition defect is supported, but a universal
higher-rate composition failure is not.

PF velocity error and Joint attitude/body-rate errors are reported separately
in `COMPOSITION_CORE_TABLE.csv`. All evaluated endpoints through 500 ms were
finite on the formal test anchors.

## Contract

All rates share the same native-400-Hz measured initial and terminal states at
each physical timestamp. Frozen fixed-time history adapters (20/40/80 samples)
and frozen sampling operators are reused without refitting. The R3/R4
reproduction artifacts pass their registered value and SHA gate. Four 164-Hz
training segments remain excluded. Test was accessed once after the composition
calibration freeze. No clipping, stabilization, multi-step training, threshold
change, or PRISM-core modification was performed.
