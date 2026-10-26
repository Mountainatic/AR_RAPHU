from pathlib import Path
import json,hashlib,math
import numpy as np,pandas as pd
from suite import dump,effect
BASE=Path('/root/autodl-tmp');ROOT=Path(__file__).resolve().parent;EXP=BASE/'PRISM_CZ_E128_20261026'
assert (EXP/'QUEUE_COMPLETED.json').exists()
rows=[];paired=[];metric_checks=[]
for e in ('CZ_Rod1','CZ_Rod2'):
    baseline=pd.read_csv(BASE/'cz_sru_matched_20261024/results/metrics.csv');baseline=baseline[baseline.rod==e]
    configs={mode:json.loads((EXP/f'results/{mode}/{e}/RESULTS.json').read_text()) for mode in ('fixed128','E128','E16')}
    for d in (4,8,16):
        values={}
        for mode,result in configs.items():
            metrics=[x for x in result['metrics'] if x['delay']==d]
            wide=pd.read_parquet(EXP/f'results/{mode}/{e}/D{d}_predictions.parquet');values[mode]=wide
            for m in metrics:
                recomputed=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(wide.y_true,wide[m['variant']]))/len(wide))
                assert math.isclose(recomputed,m['RMSE'],rel_tol=1e-12,abs_tol=1e-14);metric_checks.append(dict(entity=e,delay=d,mode=mode,variant=m['variant'],recomputed_RMSE=True))
            f=next(x for x in metrics if x['variant']=='full');noA=next(x for x in metrics if x['variant']=='no_A')
            b=baseline[(baseline.label_delay==d)&baseline.model.isin(['AR','ARX','Persistence','LSTM','Transformer','iTransformer','TimeXer'])];best=b.loc[b.RMSE.idxmin()]
            rows.append(dict(entity=e,delay=d,encoding=mode,history=f['window'],full_RMSE=f['RMSE'],no_A_RMSE=noA['RMSE'],
                best_baseline=best.model,baseline_RMSE=float(best.RMSE),relative_RMSE_reduction_pct=float(100*(1-f['RMSE']/best.RMSE)),
                A=f['A'],model_sha256=f['model_sha256']))
        for mode in ('fixed128','E16'):
            y=values['E128'].y_true.to_numpy();a=values['E128'].full.to_numpy();b=values[mode].full.to_numpy()
            np.testing.assert_array_equal(values['E128'].origin,values[mode].origin)
            paired.append(dict(entity=e,delay=d,reference=mode,E128_relative_RMSE_reduction_pct=float(100*(1-np.sqrt(np.mean((y-a)**2))/np.sqrt(np.mean((y-b)**2)))),
                full_interval=effect(y,a,b,0),startup_interval=effect(y,a,b,32+d-1)))
dump(ROOT/'CZ128_HISTORY_REVIEW.json',dict(status='COMPLETED',rows=rows,paired=paired,metric_checks=metric_checks,
    main_history=128,short_history_is_control_only=True,retrospective_development=True,reserved_tail_read=False))
pd.DataFrame([{k:v for k,v in x.items() if k!='A'} for x in rows]).to_csv(ROOT/'CZ128_HISTORY_REVIEW.csv',index=False)
print(json.dumps(dict(rows=rows,paired=paired),indent=2))
