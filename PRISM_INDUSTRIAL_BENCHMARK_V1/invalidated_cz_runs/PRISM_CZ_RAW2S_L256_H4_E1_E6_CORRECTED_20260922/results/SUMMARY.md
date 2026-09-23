# Private CZ raw-2s L256 H4 E1-E6 Result Summary

Protocol: `PRISM_CZ_RAW2S_L256_H4_PRIVATE_E1_E6_STRICT_NESTED_OOF_R2`

Formal task: 2-second samples, input `[t-256,t)`, anchor `D[t-1]`, target `D[t+3]-D[t-1]` (h=4, 8 seconds), W/W0=1/1.

The workbook is not included in this result directory and no artifact is uploaded to GitHub.

## Formal target-rod metrics

| direction | information_set | rows | rmse_delta | mae_delta | r2_delta | r2_level_reconstructed | persistence_skill |
|---|---|---|---|---|---|---|---|
| Rod_1_to_Rod_2 | input_only | 20109 | 0.0160118 | 0.0123359 | -0.000861517 | 0.998116 | 0.000202635 |
| Rod_1_to_Rod_2 | dynamic | 20109 | 0.0151283 | 0.0120614 | 0.106547 | 0.998318 | 0.107497 |
| Rod_2_to_Rod_1 | input_only | 19280 | 0.0172995 | 0.0134952 | -0.000737951 | 0.999089 | -0.000728768 |
| Rod_2_to_Rod_1 | dynamic | 19280 | 0.0161298 | 0.0128932 | 0.130007 | 0.999208 | 0.130015 |

## Main readouts

- E2 semisynthetic recovery rate: 0.637
- E2 false-admission rate: 0.106
- E3 mean paired multiscale gain: 0.266%
- E5 evidence rows: 118
- E6 N1/N2 rows: 200/200
