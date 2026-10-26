from pathlib import Path
import sys,json
import numpy as np
from sklearn.preprocessing import SplineTransformer
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'robust'))
from stable_spline import StableLinearSpline
rng=np.random.default_rng(20261026);train=rng.normal(size=(400,5))*[1,2,3,4,5]+[0,10,20,30,40]
probe=rng.normal(size=(100,5))*[5,10,15,20,25]+[0,10,20,30,40]
kwargs=dict(n_knots=3,degree=1,include_bias=False,extrapolation='linear')
fixed=StableLinearSpline(**kwargs).fit(train);p=fixed.transform(probe)
independent=np.concatenate([SplineTransformer(**kwargs).fit(train[:,i:i+1]).transform(probe[:,i:i+1]) for i in range(5)],axis=1)
np.testing.assert_allclose(p,independent,rtol=1e-14,atol=1e-14)
permutation=np.array([4,2,0,3,1]);other=StableLinearSpline(**kwargs).fit(train[:,permutation]).transform(probe[:,permutation])
np.testing.assert_allclose(other,p.reshape(100,5,2)[:,permutation,:].reshape(100,10),rtol=1e-14,atol=1e-14)
old=SplineTransformer(**kwargs).fit(train).transform(probe)
assert np.max(np.abs(old[:,2:]-p[:,2:]))>.1
# One-column learned W already has correct semantics and must stay identical.
one=StableLinearSpline(**kwargs).fit(train[:,:1]).transform(probe[:,:1])
old_one=SplineTransformer(**kwargs).fit(train[:,:1]).transform(probe[:,:1])
np.testing.assert_allclose(one,old_one,rtol=1e-14,atol=1e-14)
r=dict(status='COMPLETED',multi_feature_equals_independent_columns=True,variable_permutation_equivariant=True,
       single_column_W_preserved=True,original_multi_feature_discrepancy=float(np.max(np.abs(old-p))),
       raw_training_or_development_read=False)
(ROOT/'SPLINE_SELFTEST.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
