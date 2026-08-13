# NeuroBEM sampling-rate scaling audit

This experiment asks whether the finite recursive prediction horizon identified
by R3 is governed mainly by elapsed physical time or by recursive application
count. It evaluates 100, 200, and native 400 Hz under fixed-step and fixed-time
history contracts.

The 100-Hz path calls the frozen R3 resampler and adapter exactly. The 200/400-Hz
discrete maps are fitted only from the same frozen train-parent support because
changing sampling interval changes the fitted discrete transition. Model family,
ridge grid, route set, split, and reliability rule are unchanged. Test data are
unavailable to fitting and calibration code.

