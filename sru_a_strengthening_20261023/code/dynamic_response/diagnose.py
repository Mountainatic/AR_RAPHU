"""Frozen-parent residual diagnosis. No fits or source final access."""
from pathlib import Path
import json,hashlib,pickle,sys
import numpy as np
BASE=Path('/root/autodl-tmp');OLD=BASE/'PRISM_DELAYED_LABEL_SOFTSENSOR_20261023';ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(OLD));import model as m
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
out=[]
for delay in [4,16]:
    case=f'SRU_H2S_D{delay}';p=OLD/'cases'/case;r=read(p/'ALL_MODELS_FROZEN_BEFORE_DEVELOPMENT.json');assert all(sha(p/n)==h for n,h in r['files'].items())
    obj=pickle.loads((p/'full_FROZEN.pkl').read_bytes());m.configure(delay);D={case:np.load(OLD/'TRAINING_DATA.npz')[case]};cal=m.frame(D,3328+8+2*delay,6048-delay+1);e=cal.y.to_numpy()-m.predict_kcw(D,cal,obj['upper']);a=read(p/'full_TRAINING.json')['A']
    def summary(e):
        return dict(n=len(e),mean=float(e.mean()),std=float(e.std()),RMS=float(np.sqrt(np.mean(e*e))),mature_lag_correlation=float(np.corrcoef(e[delay:],e[:-delay])[0,1]))
    out.append(dict(case=case,calibration=summary(e),quarters=[summary(q) for q in np.array_split(e,4)],A_selected=a['selected'],outer=[dict(status=q['status'],selected=q.get('selected')) for q in a['outer']],parent_risk=a['parent_risk'],child_risk=a['child_risk'],A_admitted=a['admitted']))
(ROOT/'FROZEN_PARENT_A_DIAGNOSIS.json').write_text(json.dumps(dict(scope='Training/calibration values only, no new development access or fits; existing frozen parent hashes checked.',cases=out,source_final_numeric_read=False),indent=2));print(json.dumps(out,indent=2))
