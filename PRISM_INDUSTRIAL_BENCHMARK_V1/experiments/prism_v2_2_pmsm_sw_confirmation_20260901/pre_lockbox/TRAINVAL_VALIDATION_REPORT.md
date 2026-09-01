# PMSM SW pre-lockbox shared-data validation

Status: `PASS`

- Only registered train and validation profiles are present.
- Registered test profiles and their target rows are physically absent.
- Target construction uses `HALF_OPEN_V1` future-minus-current window change semantics.
- Sample support is `NATIVE_K_COMMON_ASSEMBLY_R1`; candidate-native history masks remain mandatory downstream.
- No model was fit and no target performance metric was computed.
