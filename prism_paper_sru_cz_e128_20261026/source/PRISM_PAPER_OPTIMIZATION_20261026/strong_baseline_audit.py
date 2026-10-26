"""Include historical strong/adaptive comparators rather than only new neural nets."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from suite import effect,dump
BASE=Path('/root/autodl-tmp');ROOT=Path(__file__).resolve().parent
FIX=BASE/'PRISM_PAPER_SPLINE_REPAIR_20261026'
names=['Persistence','OWN','ARX','WARX','HGB','SVR','ONLINE_ARX','OWN_AR128','ARX148','WARX148','ADAPTIVE_ARX','ONLINE_ARX148','ARX163','WARX163','HGB_LONG_MEMORY','SVR_LONG_MEMORY','ONLINE_ARX163']
records=[];best_records=[]
for d in (4,8,16):
    current=pd.read_parquet(FIX/f'results/repaired/SRU_H2S/D{d}_predictions.parquet');y=current.y_true.to_numpy();p=current.full.to_numpy()
    baseline={}
    old=pd.read_parquet(BASE/f'PRISM_SRU_CONDITIONAL_FEEDBACK_A_20261023/cases/SRU_H2S_D{d}/DEVELOPMENT_predictions.parquet')
    np.testing.assert_array_equal(current.origin,old.origin);np.testing.assert_allclose(y,old.y,rtol=0,atol=0)
    baseline.update({f'V9_prior_{k}':old[k].to_numpy() for k in names})
    for label,root in [('V6',BASE/'PRISM_SRU_A_STRENGTHENING_REVIEW_20261023/previous_frozen_references/V6'),
                       ('V8',BASE/'PRISM_SRU_DYNAMIC_RESPONSE_A_20261023/cases'),
                       ('V10',BASE/'PRISM_SRU_LATENT_INNOVATION_A_20261023/cases')]:
        f=pd.read_parquet(root/f'SRU_H2S_D{d}/DEVELOPMENT_predictions.parquet')
        np.testing.assert_array_equal(current.origin,f.origin);np.testing.assert_allclose(y,f.y,rtol=0,atol=0)
        baseline[f'{label}_ADAPTIVE_ARX']=f.ADAPTIVE_ARX.to_numpy()
    nn=pd.read_csv(BASE/f'PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025/results/point_predictions_D{d}.csv')
    np.testing.assert_array_equal(current.origin,nn.origin);np.testing.assert_allclose(y,nn.y_true,rtol=1e-12,atol=1e-12)
    baseline.update({f'matched_{k}':nn[k].to_numpy() for k in ('AR','ARX','LSTM','Transformer','iTransformer','TimeXer')})
    for name,pred in baseline.items():
        rmse=float(np.sqrt(np.mean((y-pred)**2)))
        records.append(dict(delay=d,baseline=name,RMSE=rmse,repaired_RMSE=float(np.sqrt(np.mean((y-p)**2))),repaired_relative_reduction_pct=float(100*(1-np.sqrt(np.mean((y-p)**2))/rmse))))
    best=min(baseline,key=lambda k:np.mean((y-baseline[k])**2))
    best_records.append(dict(delay=d,comparator=best,all_available_count=len(baseline),full_interval=effect(y,p,baseline[best],0),
        startup_interval=effect(y,p,baseline[best],32+d-1),purpose='Descriptive comparison on reused development; no independent/search-adjusted significance'))
dump(ROOT/'STRONG_BASELINE_AUDIT.json',dict(status='COMPLETED',records=records,best=best_records,reserved_tail_read=False,raw_workbook_read=False))
pd.DataFrame(records).to_csv(ROOT/'STRONG_BASELINE_AUDIT.csv',index=False)
print(json.dumps(best_records,indent=2))
