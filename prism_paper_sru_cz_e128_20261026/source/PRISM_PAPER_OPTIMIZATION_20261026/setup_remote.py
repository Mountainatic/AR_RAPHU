"""Create an isolated experiment copy; originals are never modified."""
from pathlib import Path
import shutil, sys, subprocess

root=Path(__file__).resolve().parent
base=Path('/root/autodl-tmp')
source=base/'PRISM_SRU_WINDOW_SELECTION_20261024/prism'
for folder in ('standard','robust'):
    dest=root/folder;dest.mkdir(exist_ok=False)
    for name in ('model.py','mature_a.py','legacy.py','rls_core.py'):
        shutil.copy2(source/name,dest/name)

p=root/'robust/rls_core.py';s=p.read_text()
old='def replay(frame,x,native,model,forget=.999,*,update=True,update_delay=0,return_states=False):'
assert s.count(old)==1
s=s.replace(old,old.replace('return_states=False','return_states=False,innovation_clip=None'))
old="    if not (0<forget<=1) or update_delay<0:raise ValueError('Invalid RLS policy')"
assert s.count(old)==1
s=s.replace(old,old+"\n    if innovation_clip is not None and (not np.isfinite(innovation_clip) or innovation_clip<=0):raise ValueError('Invalid innovation bound')")
old='            gain=pa/den;theta+=gain*(y-a@theta)'
assert s.count(old)==1
s=s.replace(old,"            gain=pa/den;innovation=y-a@theta\n            if innovation_clip is not None:innovation=float(np.clip(innovation,-innovation_clip,innovation_clip))\n            theta+=gain*innovation")
p.write_text(s)
p=root/'robust/mature_a.py';s=p.read_text()
old="p,a=rls.replay(f,x,native,obj,spec['forget'],update=update)"
assert s.count(old)==1
s=s.replace(old,"p,a=rls.replay(f,x,native,obj,spec['forget'],update=update,innovation_clip=spec.get('innovation_clip'))")
s+='\n# Registered extension: bound only the normalized parameter-update innovation.\n'
s+="SPECS_CONDITIONAL += [dict(family=k,p=1,window=w,forget=f,gain=g,innovation_clip=c) for k in ['QC8_RLS','QC32_RLS'] for w in [512,0] for f in [.995,.999] for g in [.5,1.] for c in [3.,6.]]\n"
p.write_text(s)
subprocess.run([sys.executable,str(root/'selftest.py')],check=True)
subprocess.run([sys.executable,str(root/'suite.py'),'prepare'],check=True)
print('READY:21 fixed-window fits; original assets unchanged',flush=True)
