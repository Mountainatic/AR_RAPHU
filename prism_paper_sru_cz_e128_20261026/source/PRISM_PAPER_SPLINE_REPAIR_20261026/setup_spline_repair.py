"""Register a separate spline-semantics experiment with original273 A policies."""
from pathlib import Path
import shutil,sys,subprocess
root=Path(__file__).resolve().parent;source=Path('/root/autodl-tmp/PRISM_SRU_WINDOW_SELECTION_20261024/prism')
dest=root/'robust';dest.mkdir(exist_ok=False)
for n in ('model.py','mature_a.py','legacy.py','rls_core.py'):shutil.copy2(source/n,dest/n)
shutil.copy2(root/'stable_spline.py',dest/'stable_spline.py')
p=dest/'model.py';s=p.read_text();old='from sklearn.preprocessing import SplineTransformer';assert s.count(old)==1
p.write_text(s.replace(old,'from stable_spline import StableLinearSpline as SplineTransformer'))
subprocess.run([sys.executable,str(root/'spline_repair_selftest.py')],check=True)
p=root/'suite.py';s=p.read_text()
start=s.index('    jobs=[]\n');end=s.index('    sources={}\n',start)
s=s[:start]+'''    jobs=[]
    for e in cases:
        for d in (4,8,16):
            variants=('full','no_KC','no_W','no_A') if e=='SRU_H2S' else ('full','no_A')
            jobs.extend(dict(runtime='repaired',entity=e,delay=d,variant=v) for v in variants)
'''+s[end:]
s=s.replace("for mode in ('standard','robust'):","for mode in ('repaired',):")
s=s.replace("Original273 A specs retained. Add32 QC8/QC32 specifications with innovation_clip3 or6 in training-residual standard deviations, window512/all,forget.995/.999,gain.5/1. Ridge100 initialization, stored-issued phi, original support/numerical gates and fully nested A policy selection remain. Only theta update innovation is bounded; no prediction/target clipping.","Correct per-column degree1 spline transform only. Original273 A candidates, ridge100, gates, folds, windows and original RLS equations retained. All full/noKC/noW/noA models are refit as applicable. No Huber extension or development-driven recipe selection.")
s=s.replace('every21 new job','every24 new job')
s=s.replace('frozen full/no_A are reused only for standard','all applicable repaired variants are newly fitted')
p.write_text(s)
subprocess.run([sys.executable,str(root/'suite.py'),'prepare'],check=True)
print('REGISTERED24_SPLINE_REPAIR_FITS',flush=True)
