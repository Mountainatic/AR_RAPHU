from pathlib import Path
import json,pickle,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'PRISM_SRU_ISSUED_RESPONSE_A_20261023';sys.path.insert(0,str(OLD));import model as m
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=[]
for delay in [4,8,16]:
    case=f'SRU_H2S_D{delay}';folder=OLD/'cases'/case;freeze=read(folder/'ALL_MODELS_FROZEN_BEFORE_DEVELOPMENT.json');assert all(sha(folder/n)==h for n,h in freeze['files'].items());obj=pickle.loads((folder/'full_FROZEN.pkl').read_bytes());m.configure(delay);D={case:np.load(OLD/'TRAINING_DATA.npz')[case]};T=m.frame(D,3328+8+2*delay,6048-delay+1);q=m.predict_kcw(D,T,obj['upper']);e=T.y.to_numpy()-q;bins=np.quantile(q,[0,.25,.5,.75,1]);records=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        ix=np.flatnonzero((q[delay:]>=lo)&(q[delay:]<=hi))+delay;records.append(dict(q_low=float(lo),q_high=float(hi),n=len(ix),mature_residual_correlation=float(np.corrcoef(e[ix],e[ix-delay])[0,1])))
    rows.append(dict(case=case,pooled_correlation=float(np.corrcoef(e[delay:],e[:-delay])[0,1]),training_response_quartiles=records))
(ROOT/'TRAINING_CONDITIONAL_DIAGNOSIS.json').write_text(json.dumps(dict(scope='Training calibration only; no new fit/development parse or final values',rows=rows),indent=2));print(json.dumps(rows,indent=2))
