from __future__ import annotations

import numpy as np

from prism_benchmark.v2_j import _joint_fit


def test_joint_model_contains_exact_zero_blocks() -> None:
    rng=np.random.default_rng(9);k=rng.normal(size=(500,3));state=rng.normal(size=(500,2));y=k[:,0]+state[:,1]
    prediction,contract=_joint_fit(k,state,y,k,state,1e-8,1.0)
    assert np.mean((prediction-y)**2)<1e-8
    assert contract["k_columns"]==3 and contract["state_columns"]==2
