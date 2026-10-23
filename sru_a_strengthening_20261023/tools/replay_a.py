"""Reproduce saved A outputs without refits or original raw-data access."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--branch',choices=['dynamic_response','conditional_feedback','latent_innovation'],default='conditional_feedback');p.add_argument('--case',choices=['SRU_H2S_D4','SRU_H2S_D8','SRU_H2S_D16'],default='SRU_H2S_D4');args=p.parse_args()
sys.path.insert(0,str(ROOT/'code'/args.branch))
import model as m
dest=ROOT/'results'/args.branch/args.case
saved=pd.read_parquet(dest/'DEVELOPMENT_predictions.parquet');selected=json.loads((dest/'SELECTED_MODELS.json').read_text(encoding='utf-8'));result=json.loads((dest/'DEVELOPMENT_RESULT.json').read_text(encoding='utf-8'))
m.configure(selected['label_delay']);before=m.FITS;errors={};prefix_checks=0
for name,record in selected['variants'].items():
    q=saved[name+'_upper'].to_numpy();f=m.source(saved,q,'portable_replay');a=record['A']
    correction=np.zeros(len(f)) if a is None else m.a_predict(f,a)[0]
    np.testing.assert_allclose(q+correction,saved[name],rtol=1e-12,atol=1e-12)
    errors[name]=float(np.sqrt(np.mean((saved.y.to_numpy()-q-correction)**2)))
    expected=next(r['RMSE'] for r in result['rows'] if r['variant']==name);assert np.isclose(errors[name],expected,rtol=1e-12,atol=1e-12)
    if a is not None:
        for n in sorted(set([1,m.DELAY,m.DELAY+1,33,64,128,513,1024,len(f)])):
            clean=f.head(n).copy();poison=clean.copy();last=int(clean.latest_available_target_index.iloc[-1]);poison.loc[poison.target_stop_exclusive-1>last,'residual']+=1234.567
            aa=m.a_predict(clean,a)[0];ab=m.a_predict(poison,a)[0];np.testing.assert_array_equal(aa,ab);np.testing.assert_allclose(aa,correction[:n],rtol=1e-12,atol=1e-12);prefix_checks+=1
for name,expected in result['baselines'].items():assert np.isclose(float(np.sqrt(np.mean((saved.y-saved[name])**2))),expected,rtol=1e-12,atol=1e-12)
assert m.FITS==before
print(json.dumps(dict(status='PASSED',branch=args.branch,case=args.case,variants_RMSE=errors,baseline_RMSE_checked=len(result['baselines']),unmatured_feedback_prefix_checks=prefix_checks,no_refits=True,original_raw_or_final_data_read=False),indent=2))
