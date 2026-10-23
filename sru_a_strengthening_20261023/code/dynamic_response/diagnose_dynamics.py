"""Training-only issued-response lag diagnostics, no new numerical fits."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'PRISM_SRU_ISSUED_RESPONSE_A_20261023';sys.path.insert(0,str(OLD));import model as m
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for delay in [4,8,16]:
    case=f'SRU_H2S_D{delay}';folder=OLD/'cases'/case;freeze=read(folder/'ALL_MODELS_FROZEN_BEFORE_DEVELOPMENT.json');assert all(sha(folder/n)==h for n,h in freeze['files'].items())
    obj=pickle.loads((folder/'full_FROZEN.pkl').read_bytes());m.configure(delay);D={case:np.load(OLD/'TRAINING_DATA.npz')[case]};T=m.frame(D,3328+8+2*delay,6048-delay+1);q=m.predict_kcw(D,T,obj['upper']);e=T.y.to_numpy()-q;part=[]
    for lag in [1,4,8,16,32]:
        delta=q[lag:]-q[:-lag];err=e[lag:];part.append(dict(lag=lag,correlation_e_delta_q=float(np.corrcoef(err,delta)[0,1]),quarters=[float(np.corrcoef(err[ix],delta[ix])[0,1]) for ix in np.array_split(np.arange(len(err)),4)]))
    rows.append(dict(case=case,correlation_e_q=float(np.corrcoef(e,q)[0,1]),lagged_response=part))
(ROOT/'TRAINING_DYNAMICS_DIAGNOSIS.json').write_text(json.dumps(dict(scope='Same existing training calibration only, no development parse, no new fit, no final values.',rows=rows),indent=2));print(json.dumps(rows,indent=2))
