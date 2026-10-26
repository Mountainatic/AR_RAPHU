"""Meaningful checks of the modified RLS update and label-maturity contract."""
from pathlib import Path
import sys,importlib.util,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'robust'))
import rls_core as robust
import mature_a
spec=importlib.util.spec_from_file_location('reference_rls',ROOT/'standard/rls_core.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
rng=np.random.default_rng(20261026)
x=rng.normal(size=(600,4));y=x@np.array([.1,.3,-.1,.2])+rng.normal(0,.1,600)
init=original.initialize(x[:250],y[:250],fit_last_target_end=749)
o=np.arange(1000,1350);f=pd.DataFrame(dict(origin=o,target_stop_exclusive=o+1,latest_available_target_index=o-4,residual=y[250:]))
native=np.ones(len(f),bool)
expected,oldlog=original.replay(f,x[250:],native,init,.999)
actual,log=robust.replay(f,x[250:],native,init,.999)
np.testing.assert_array_equal(expected,actual)
np.testing.assert_array_equal(oldlog['latest_parameter_target_end'],log['latest_parameter_target_end'])
# A huge bound also preserves the exact original arithmetic for this stream.
unbound,_=robust.replay(f,x[250:],native,init,.999,innovation_clip=1e15)
np.testing.assert_array_equal(unbound,actual)
out,audit=robust.replay(f,x[250:],native,init,.999,innovation_clip=3.)
for n in (1,4,5,33,129,350):
    poison=f.iloc[:n].copy();poison.loc[poison.target_stop_exclusive-1>poison.latest_available_target_index.iloc[-1],'residual']+=1e8
    p,_=robust.replay(poison,x[250:250+n],native[:n],init,.999,innovation_clip=3.)
    np.testing.assert_array_equal(p,out[:n])
assert (audit['latest_parameter_target_end']<=f.latest_available_target_index).all()
spike=f.copy();spike.loc[100,'residual']+=1e5
normal,_=robust.replay(spike,x[250:],native,init,.999)
limited,_=robust.replay(spike,x[250:],native,init,.999,innovation_clip=3.)
assert np.max(np.abs(limited))<np.max(np.abs(normal))/10
for bad in (0.,-1.,float('nan')):
    try:robust.replay(f,x[250:],native,init,.999,innovation_clip=bad)
    except ValueError:pass
    else:raise AssertionError('invalid cap accepted')
count=sum(map(len,(mature_a.SPECS_RLS,mature_a.SPECS_BIAS,mature_a.SPECS_Q,mature_a.SPECS_DYNAMIC,mature_a.SPECS_CONDITIONAL)))
assert count==305
receipt=dict(status='COMPLETED',no_bound_bitwise_equal=True,large_bound_bitwise_equal=True,
    mature_only_updates=True,poison_prefixes=6,spike_influence_limited=True,invalid_caps_rejected=True,
    original_candidates=273,new_candidates=32,training_or_development_values_read=False)
(ROOT/'SELFTEST.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
