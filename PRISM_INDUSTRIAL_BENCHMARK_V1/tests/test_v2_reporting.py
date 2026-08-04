from __future__ import annotations

import numpy as np
import pandas as pd

from prism_benchmark.v2_reporting import _block_draws, _holm


def test_block_bootstrap_preserves_positive_paired_gain() -> None:
    reference=np.ones(40,dtype=np.float64);difference=np.full(40,0.25,dtype=np.float64);entities=np.asarray(["a"]*20+["b"]*20)
    draws=_block_draws(difference,reference,entities,4,100,20260804)
    np.testing.assert_allclose(draws,0.25)


def test_holm_is_step_down_and_family_local() -> None:
    base={"target_head":"h","split":"test","information_set":"dynamic","availability_scenario":"record_time","proxy_policy":"primary","block_length":8}
    frame=pd.DataFrame([{**base,"p_two_sided":0.01},{**base,"p_two_sided":0.06}])
    result=_holm(frame)
    assert bool(result.iloc[0]["holm_reject"])
    assert not bool(result.iloc[1]["holm_reject"])
