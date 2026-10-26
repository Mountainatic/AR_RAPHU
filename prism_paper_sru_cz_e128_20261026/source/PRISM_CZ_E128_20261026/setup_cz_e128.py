"""CZ128 main history, with a registered fixed-multiresolution E and E16 control."""
from pathlib import Path
import shutil,sys,subprocess
root=Path(__file__).resolve().parent
src=Path('/root/autodl-tmp/PRISM_PAPER_SPLINE_REPAIR_20261026/robust')
for name in ('fixed128','E128','E16'):
    dest=root/name;dest.mkdir(exist_ok=False)
    for n in ('model.py','mature_a.py','legacy.py','rls_core.py','stable_spline.py'):shutil.copy2(src/n,dest/n)
    if name!='fixed128':
        with (dest/'model.py').open('a') as f:
            f.write('\nE_HISTORY='+('128' if name=='E128' else '16')+'\n')
            f.write((root/'e_design_append.py').read_text())
p=root/'suite.py';s=p.read_text()
s=s.replace("sys.path.insert(0,str(ROOT/('standard' if name=='standard' else 'robust')))","sys.path.insert(0,str(ROOT/name))")
s=s.replace("m.L=g['windows'][d];m.configure(d)","m.L=getattr(m,'E_HISTORY',g['windows'][d]);g['windows'][d]=m.L;m.configure(d)")
s=s.replace("cases={e:geometry(e) for e in ('SRU_H2S','CZ_Rod1','CZ_Rod2')}","cases={e:geometry(e) for e in ('CZ_Rod1','CZ_Rod2')}\n    for g in cases.values():g['windows']={4:128,8:128,16:128}")
start=s.index('    jobs=[]\n');end=s.index('    sources={}\n',start)
s=s[:start]+'''    jobs=[]
    for name in ('fixed128','E128','E16'):
        for e in cases:
            for d in (4,8,16):
                jobs.extend(dict(runtime=name,entity=e,delay=d,variant=v) for v in ('full','no_A'))
'''+s[end:]
s=s.replace("for mode in ('standard','robust'):","for mode in ('fixed128','E128','E16'):")
s=s.replace("for entity in ('SRU_H2S','CZ_Rod1','CZ_Rod2'):","for entity in ('CZ_Rod1','CZ_Rod2'):")
s=s.replace('Original273 A specs retained. Add32 QC8/QC32 specifications with innovation_clip3 or6 in training-residual standard deviations, window512/all,forget.995/.999,gain.5/1. Ridge100 initialization, stored-issued phi, original support/numerical gates and fully nested A policy selection remain. Only theta update innovation is bounded; no prediction/target clipping.',
    'CZ fixed128 main input history, corrected degree1 spline. E128 uses raw block averages on lag bins[0,1),[1,2),[2,4),[4,8),[8,16),[16,32),[32,64),[64,128), then per-channel amplitude basis and learned ridge coefficients. This is FIXED_MULTIRESOLUTION E, not full original channel-specific profile/rank search. E16 is a short-history refit control only, never selected as the CZ main model. Candidate process families require memory=True; OWN bypass and exact-zero A remain. Original273 A policies and gates retained. No Huber extension.')
s=s.replace('every21 new job','every36 new job')
s=s.replace('Fixed original train-selected full window per task','CZ main128 imposed by user; E16 is an explicit ablation')
s=s.replace('frozen full/no_A are reused only for standard','all36 full/no_A chains are newly fitted')
p.write_text(s)
subprocess.run([sys.executable,str(root/'e128_selftest.py')],check=True)
subprocess.run([sys.executable,str(root/'suite.py'),'prepare'],check=True)
print('REGISTERED36_CZ128_E_FITS',flush=True)
