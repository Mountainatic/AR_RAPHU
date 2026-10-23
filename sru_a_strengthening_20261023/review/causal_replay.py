"""Independent frozen-checkpoint replay; train/development parquet only."""
import sys,os,json,pickle,hashlib
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
branch=Path(sys.argv[1]);sys.path.insert(0,str(branch))
import numpy as np,pandas as pd
import model as m, data_view as d
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
records=[]
reg=json.loads((branch/'REGISTRATION_BEFORE_FITS.json').read_text())
assert all(sha(branch/k)==v for k,v in reg['code_sha256'].items())
for case in reg['cases']:
    cfg=d.records()[case];m.configure(cfg['delay']);D=d.prefix(case,True);V=m.frame(D,cfg['train'],cfg['dev']);dest=branch/'cases'/case
    saved=pd.read_parquet(dest/'DEVELOPMENT_predictions.parquet');frozen=json.loads((dest/'ALL_MODELS_FROZEN_BEFORE_DEVELOPMENT.json').read_text());before=m.FITS
    assert all(sha(dest/k)==v for k,v in frozen['files'].items())
    objs={k:pickle.loads((dest/(k+'_FROZEN.pkl')).read_bytes()) for k in reg['variants']}
    objs['ADAPTIVE_ARX']=pickle.loads((dest/'BASELINES_FROZEN.pkl').read_bytes())['ADAPTIVE_ARX']
    for name,obj in objs.items():
        q=m.predict_kcw(D,V,obj['upper']);f=m.source(V,q,'independent_replay');a=np.zeros(len(V)) if obj['A'] is None else m.a_predict(f,obj['A'])[0]
        np.testing.assert_allclose(q+a,saved[name],rtol=1e-13,atol=1e-13)
        if obj['A']:
            assert obj['A']['model']['fit_last_target_end']<=int(V.origin.iloc[0])-m.DELAY
            _,log=m.a_predict(f,obj['A']);assert log['resets']==1
            for key in ['latest_feature_label','latest_parameter_label']:assert (log[key]<=V.origin.to_numpy()-m.DELAY).all()
        for n in sorted(set([1,m.DELAY,m.DELAY+1,33,64,128,513,1024,len(V)])):
            P=V.head(n);cut=int(P.origin.iloc[-1]);poison={case:D[case].copy()}
            # All labels not yet mature at the last issue are unknowable to EVERY prefix issue.
            poison[case][cut-m.DELAY+1:,0]+=1234.567
            poison[case][cut+1:,1:]-=876.543
            Pp=m.frame(poison,cfg['train'],cut+1);qp=m.predict_kcw(poison,Pp,obj['upper'])
            qc=m.predict_kcw(D,P,obj['upper'])
            np.testing.assert_array_equal(qp,qc)
            # BLAS changes reduction order between batch sizes; keep poison comparison bitwise.
            np.testing.assert_allclose(qc,q[:n],rtol=1e-13,atol=1e-13)
            fp=m.source(Pp,qp,'independent_replay');ap=np.zeros(n) if obj['A'] is None else m.a_predict(fp,obj['A'])[0]
            np.testing.assert_allclose(qp+ap,saved[name].to_numpy()[:n],rtol=1e-13,atol=1e-13)
        records.append(dict(branch=branch.name,case=case,variant=name,full_saved_prediction_reproduced=True,prefixes=sorted(set([1,m.DELAY,m.DELAY+1,33,64,128,513,1024,len(V)])),all_unmatured_y_and_future_X_poison_passed=True,initial_label_maturity=True,parameter_and_feature_maturity=True))
    assert m.FITS==before and all(sha(dest/k)==v for k,v in frozen['files'].items())
out=dict(status='PASSED',branch=branch.name,records=records,source_final_numeric_read=False,no_refits=True)
Path(sys.argv[2]).write_text(json.dumps(out,indent=2));print(branch.name,len(records),'CAUSAL_REPLAY_PASSED',flush=True)
