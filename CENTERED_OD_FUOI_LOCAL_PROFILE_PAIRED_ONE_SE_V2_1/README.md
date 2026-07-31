# Centered OD-FUOI Local Profile + Paired One-SE V2.1

CPU FP64 patch of the frozen V2 experiment. It preserves the centered
full-Urysohn model, basis, data, folds, purge, penalty and all downstream
diagnostics. Only the selection procedure changes:

1. the V2 effective-DF cache is searched on a log-excess coordinate and each
   candidate basin is certified by three bounded local refinements;
2. the leftmost admissible effective DF is selected from paired, same-sample
   squared-error differences using a 40-minute moving-block bootstrap;
3. 22- and 60-minute block lengths are reported as frozen sensitivities.

Far-field quadratic interpolation is diagnostic only. A selected EDF change
larger than 0.05 forces E2D--E9 to refit/regenerate; old V2 fitted surfaces are
never silently reused after such a change.

Run `RUN_CPU_CONFIRM.sh` with the three frozen benchmark bundles and the V1
and V2 result bundles. `RESUME_CPU_CONFIRM.sh` resumes the same invocation.
Raw Excel files are never read or packaged.
