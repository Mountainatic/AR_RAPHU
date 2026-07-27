# PB0 regression status

Date: 2026-07-27  
Server: `connect.westd.seetacloud.com:52559`  
Branch: `public-benchmark-pb1`

## Main repository

Command:

```bash
AR_RAPHU_PUBLIC_RAW_ROOT=/root/OPS_UOI_WORKSPACE/data/raw \
/root/AR_RAPHU_AUTODL/.venv/bin/python -m pytest -q
```

Result:

```text
177 passed, 7 skipped, 17 warnings in 65.03s
```

The seven skips are the private-CZ tests whose dataset was explicitly excluded
by the user. All nine official-source integration tests ran and passed.

## V20 compatibility baseline

Command:

```bash
cd STAGE1_DUAL_SOLVER_V20_bundle
/root/AR_RAPHU_AUTODL/.venv/bin/python -m pytest -q
```

Result:

```text
118 passed, 14 warnings in 7.39s
```

## PWH full-kernel smoke

Status: `COMPLETED`

- Scope: X-only, horizon 1, history 16, first estimation record.
- Design: 4,080 samples by 256 FP64 columns.
- Device: CPU.
- Relative KKT residual: `4.4067989004909433e-16`.
- Wall time: 9.35 seconds.
- Official test accessed: false.
- Scientific evidence: false.
