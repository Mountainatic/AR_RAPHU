from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'E128'))
import model as m
m.L=128;m.configure(4)
bins=m.e_intervals(128);np.testing.assert_array_equal(np.concatenate([np.arange(lo,hi) for lo,hi in bins]),np.arange(128))
rng=np.random.default_rng(20261026);a=rng.normal(size=(900,6));D={'synthetic':a};V=m.frame(D,256,300)
cfg=dict(kind='URYSOHN',alpha=1.,own=True,memory=True);x,recipe=m.design(D,V,cfg)
assert x.shape==(44,82)
cut=int(V.origin.iloc[-1]);poison=a.copy();poison[cut-3:,0]+=1000;poison[cut+1:,1:]+=1000
np.testing.assert_array_equal(m.design({'synthetic':poison},V,cfg,recipe)[0],x)
one=V.iloc[:1];o=int(one.origin.iloc[0]);changed=a.copy();changed[o-127,1]+=100
assert not np.array_equal(m.design({'synthetic':changed},one,cfg,recipe)[0],x[:1])
# E128 changes under the oldest allowed X, but never consumes X older than128.
changed=a.copy();changed[:o-127,1]+=100
np.testing.assert_array_equal(m.design({'synthetic':changed},one,cfg,recipe)[0],x[:1])
assert all(c.get('memory',True) for c in m.configs('full',None))
r=dict(status='COMPLETED',maximum_history=128,bins=bins,complete_nonoverlapping_support=True,
       oldest_allowed_input_used=True,older_than128_input_excluded=True,current_target_and_future_X_excluded=True,
       finite_Urysohn_design_features=82,raw_or_development_values_read=False)
(ROOT/'E128_SELFTEST.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
