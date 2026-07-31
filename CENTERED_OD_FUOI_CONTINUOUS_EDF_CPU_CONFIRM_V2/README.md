# Centered OD-FUOI Continuous-EDF CPU Confirm V2

CPU FP64 implementation of the frozen V2 repair experiment. It preserves the
V1 full-Urysohn/C1/projection/residual protocol while changing only two
scientific choices:

1. every input history is represented as `x(t-lag)-x(t)`;
2. smoothing is selected on a continuous effective-df coordinate by four
   blocked folds, fold-specific `lambda_f(d)`, and the continuous one-SE rule.

GCV, REML and L-curve values are emitted only as diagnostics. There is one
global Sobolev penalty scale, no rank search, no rank-2 rescue, no manual
lambda floor and no manually fixed EDF.

Run `RUN_CPU_CONFIRM.sh` with the three frozen benchmark bundles and the V1
results bundle. Profile evaluations checkpoint after every actual EDF query;
`RESUME_CPU_CONFIRM.sh` resumes the same invocation. Raw Excel files are never
read or packaged.
