from __future__ import annotations

import pandas as pd

from prism_benchmark.v2_a import mature_features


def test_mature_residual_features_never_use_unmatured_origin() -> None:
    residuals=pd.DataFrame({"entity_id":["e"]*20,"origin":range(20),"residual":[float(x) for x in range(20)]})
    samples=pd.DataFrame({"entity_id":["e"],"origin":[15]})
    features,observed=mature_features(samples,residuals,h_steps=3,w_steps=2,delta=1,history=4,maximum_lags=64,residual_mean=0.0)
    assert features.max()<=10.0
    assert observed==1.0
