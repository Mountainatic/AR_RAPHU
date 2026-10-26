"""A small fixed mature-residual transport candidate, keeping E128 frozen."""
from pathlib import Path
import shutil,sys,subprocess
root=Path(__file__).resolve().parent;src=Path('/root/autodl-tmp/PRISM_CZ_E128_20261026/E128')
dest=root/'transport';dest.mkdir(exist_ok=False)
for n in ('model.py','mature_a.py','legacy.py','rls_core.py','stable_spline.py'):shutil.copy2(src/n,dest/n)
with (dest/'mature_a.py').open('a') as f:
    f.write('''
# Fixed operator candidates: no online parameter updates, only lawful mature state.
SPECS_RLS += [dict(family='TRANSPORT',p=p,window=0,forget=1.,gain=g) for p in [1,4] for g in [.25,.5,.75,1.]]
_original_predict_spec=predict_spec
def predict_spec(f,obj,spec,*,update=True):
    if spec['family']!='TRANSPORT':return _original_predict_spec(f,obj,spec,update=update)
    x,native,audit=rls.features(f,spec['p']);p=np.zeros(len(f));p[native]=x[native].mean(1)
    return spec['gain']*p,dict(latest_feature_label=audit['latest_feature_target_end'],latest_parameter_label=np.full(len(f),obj['fit_last_target_end'],int),updates=0,resets=len(np.unique(rls.runs(f))),stored_issued_features=True,current_issued_response=False)
''')
p=root/'suite.py';s=p.read_text()
s=s.replace("sys.path.insert(0,str(ROOT/('standard' if name=='standard' else 'robust')))","sys.path.insert(0,str(ROOT/name))")
s=s.replace("cases={e:geometry(e) for e in ('SRU_H2S','CZ_Rod1','CZ_Rod2')}","cases={e:geometry(e) for e in ('CZ_Rod1','CZ_Rod2')}\n    for g in cases.values():g['windows']={4:128,8:128,16:128}")
old="return Path(g['root'])/'runs'/f'D{d}'/f\"L{g['windows'][d]}\"/f'{variant}.pkl'"
assert old in s
s=s.replace(old,"return BASE/'PRISM_CZ_E128_20261026/runs/E128'/entity/f'D{d}'/variant/'model.pkl'")
start=s.index('    jobs=[]\n');end=s.index('    sources={}\n',start)
s=s[:start]+'''    jobs=[dict(runtime='transport',entity=e,delay=d,variant='full') for e in cases for d in (4,8,16)]
'''+s[end:]
s=s.replace("for mode in ('standard','robust'):","for mode in ('transport',):")
s=s.replace("for entity in ('SRU_H2S','CZ_Rod1','CZ_Rod2'):","for entity in ('CZ_Rod1','CZ_Rod2'):")
s=s.replace('Original273 A specs retained. Add32 QC8/QC32 specifications with innovation_clip3 or6 in training-residual standard deviations, window512/all,forget.995/.999,gain.5/1. Ridge100 initialization, stored-issued phi, original support/numerical gates and fully nested A policy selection remain. Only theta update innovation is bounded; no prediction/target clipping.',
    'E128 encoding and corrected spline retained. Add8 fixed mature-residual transport A policies: p1 or4, gain.25/.5/.75/1. All273 original policies retained. Same nested MSE selection, admission, maturity/support/numerical gates; no new gain grid edits after development evaluation. Transport has zero adaptive regression coefficients, no parameter updates; original initializer is retained only for common support/certificate bookkeeping. Refit6 full chains; reuse exact E128 no_A controls. Retrospective exploratory extension motivated by observed A instability; not independent validation.')
s=s.replace('every21 new job','every6 new job')
s=s.replace('Fixed original train-selected full window per task','User-fixed128 history and fixed8-bin E')
p.write_text(s)
subprocess.run([sys.executable,str(root/'transport_selftest.py')],check=True)
subprocess.run([sys.executable,str(root/'suite.py'),'prepare'],check=True)
print('REGISTERED6_TRANSPORT_FITS',flush=True)
