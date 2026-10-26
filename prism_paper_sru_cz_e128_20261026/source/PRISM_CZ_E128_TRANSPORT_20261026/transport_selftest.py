from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'transport'))
import mature_a as a
o=np.arange(1000,1100);f=pd.DataFrame(dict(origin=o,target_stop_exclusive=o+1,latest_available_target_index=o-4,residual=np.sin(o/7),residual_source_role='DEPLOYED_W_ONLINE'))
spec=dict(family='TRANSPORT',p=1,window=0,forget=1.,gain=.75)
p,log=a.predict_spec(f,dict(fit_last_target_end=500),spec)
np.testing.assert_array_equal(p[:4],np.zeros(4));np.testing.assert_allclose(p[4:],.75*f.residual.to_numpy()[:-4],rtol=0,atol=0)
for n in (1,4,5,33,65):
    v=f.iloc[:n].copy();v.loc[v.target_stop_exclusive-1>v.latest_available_target_index.iloc[-1],'residual']+=1e8
    q,_=a.predict_spec(v,dict(fit_last_target_end=500),spec);np.testing.assert_array_equal(q,p[:n])
assert log['updates']==0 and np.all(log['latest_feature_label']<=f.latest_available_target_index)
assert sum(map(len,(a.SPECS_RLS,a.SPECS_BIAS,a.SPECS_Q,a.SPECS_DYNAMIC,a.SPECS_CONDITIONAL)))==281
r=dict(status='COMPLETED',residual_transport_matches_maturity=True,no_parameter_updates=True,unmatured_label_poison_prefixes=5,added_candidates=8,reserved_tail_read=False)
(ROOT/'TRANSPORT_SELFTEST.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
