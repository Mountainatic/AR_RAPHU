"""Aggregate all completed exploratory branches without discarding failures."""
from pathlib import Path
import json,csv,math,pickle,hashlib
import pandas as pd

BASE=Path('/root/autodl-tmp');ROOT=BASE/'PRISM_PAPER_OPTIMIZATION_20261026'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[];effects=[];checks=[]
sources=[('V9_matched_complete_ablation',ROOT/'results/standard'),('Huber_RLS',ROOT/'results/robust'),
         ('Spline_repaired',BASE/'PRISM_PAPER_SPLINE_REPAIR_20261026/results/repaired')]
for name,root in sources:
    for p in sorted(root.rglob('RESULTS.json')):
        j=read(p);groups={}
        for x in j['metrics']:
            entity=x['entity'];d=x['delay'];groups.setdefault((entity,d),[]).append(x)
            wide=pd.read_parquet(p.parent/f'D{d}_predictions.parquet')
            actual=math.sqrt(math.fsum((float(y)-float(yhat))**2 for y,yhat in zip(wide.y_true,wide[x['variant']]))/len(wide))
            assert math.isclose(actual,x['RMSE'],rel_tol=1e-12,abs_tol=1e-14)
            checks.append(dict(source=str(p),entity=entity,delay=d,variant=x['variant'],RMSE_recomputed=True))
        for (entity,d),ms in groups.items():
            f=next(x for x in ms if x['variant']=='full')
            bpath=(BASE/'PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025/results/metrics.csv' if entity=='SRU_H2S' else BASE/'cz_sru_matched_20261024/results/metrics.csv')
            df=pd.read_csv(bpath);df=df[df.label_delay==d]
            if entity!='SRU_H2S':df=df[df.rod==entity]
            df=df[df.model.isin(['Persistence','AR','ARX','LSTM','Transformer','iTransformer','TimeXer'])]
            best=df.loc[df.RMSE.idxmin()]
            ef=[e for e in j['effects'] if e['delay']==d]
            control={m['variant']:m['RMSE'] for m in ms}
            rows.append(dict(branch=name,entity=entity,delay=d,window=f['window'],RMSE=f['RMSE'],best_baseline=best.model,
                baseline_RMSE=float(best.RMSE),RMSE_reduction_pct=float(100*(1-f['RMSE']/best.RMSE)),
                no_KC_RMSE=control.get('no_KC'),no_W_RMSE=control.get('no_W'),no_A_RMSE=control.get('no_A'),
                all_three_descriptive_CI_positive=bool(ef) and all(e['CI95'][0]>0 for e in ef),
                selected_A=f['A'],frozen_model_sha256=f['model_sha256']))
        effects.extend(dict(branch=name,**e) for e in j['effects'])

# Latest input256 run is a distinct protocol, reported without pooling it.
p=BASE/'cz_sru_input256_20261025/results/metrics.csv'
if p.exists():
    df=pd.read_csv(p)
    for (entity,d),g in df.groupby(['rod','label_delay']):
        full=g[g.model=='PRISM'].iloc[0];b=g[g.model.isin(['Persistence','AR','ARX','LSTM','Transformer','iTransformer','TimeXer'])];best=b.loc[b.RMSE.idxmin()]
        rows.append(dict(branch='CZ_input256_existing',entity=entity,delay=int(d),RMSE=float(full.RMSE),
            best_baseline=best.model,baseline_RMSE=float(best.RMSE),RMSE_reduction_pct=float(100*(1-full.RMSE/best.RMSE)),
            all_three_descriptive_CI_positive=False,protocol='Different256-common-support geometry; do not mix with128 protocol ablations'))
out=ROOT/'comparison';out.mkdir(exist_ok=True)
(out/'ALL_BRANCH_RESULTS.json').write_text(json.dumps(dict(status='COMPLETED',rows=rows,effects=effects,metric_checks=checks,reserved_tail_read=False),indent=2))
pd.DataFrame([{k:v for k,v in r.items() if k!='selected_A'} for r in rows]).to_csv(out/'ALL_BRANCH_COMPARISONS.csv',index=False)
pd.DataFrame(effects).to_csv(out/'ALL_ABLATION_EFFECTS.csv',index=False)
print(json.dumps(dict(comparisons=len(rows),RMSE_rows_recomputed=len(checks),positive_cases=[{k:r[k] for k in ('branch','entity','delay','RMSE_reduction_pct')} for r in rows if r.get('all_three_descriptive_CI_positive')]),indent=2))
