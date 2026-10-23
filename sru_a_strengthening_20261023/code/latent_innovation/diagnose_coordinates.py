from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'PRISM_SRU_ISSUED_RESPONSE_A_20261023';sys.path.insert(0,str(OLD));import model as m
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=[]
for delay in [4,8,16]:
    case=f'SRU_H2S_D{delay}';p=OLD/'cases'/case;freeze=read(p/'ALL_MODELS_FROZEN_BEFORE_DEVELOPMENT.json');assert all(sha(p/n)==h for n,h in freeze['files'].items());obj=pickle.loads((p/'full_FROZEN.pkl').read_bytes());m.configure(delay);D={case:np.load(OLD/'TRAINING_DATA.npz')[case]};T=m.frame(D,3328+8+2*delay,6048-delay+1);q=m.predict_kcw(D,T,obj['upper']);w=obj['upper']['parent']['coordinate'];e=T.y.to_numpy()-q;z=m.old.forward(T.y.to_numpy(),w)-m.old.forward(q,w)
    rows.append(dict(case=case,link=w,physical_lag_correlation=float(np.corrcoef(e[delay:],e[:-delay])[0,1]),latent_lag_correlation=float(np.corrcoef(z[delay:],z[:-delay])[0,1]),latent_quarters=[float(np.corrcoef(z[ix[delay:]],z[ix[:-delay]])[0,1]) for ix in np.array_split(np.arange(len(z)),4)]))
(ROOT/'TRAINING_COORDINATE_DIAGNOSIS.json').write_text(json.dumps(dict(scope='Existing training calibration only,no new fits,development or final values',rows=rows),indent=2));print(json.dumps(rows,indent=2))
