"""Stdlib-only verification of Git publication, metrics and causal window selection."""
from pathlib import Path
import csv,json,hashlib,math,sys,collections
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)
manifest=read(ROOT/'PUBLICATION_MANIFEST.json')
assert all((ROOT/k).stat().st_size==v['bytes'] and sha(ROOT/k)==v['sha256'] for k,v in manifest.items())
reg=read(ROOT/'REGISTRATION.json');gate=read(ROOT/'ALL_SELECTIONS_FROZEN.json')
assert gate['all_fits_frozen_before_evaluation'] and not gate['development_values_read_for_window_selection']
for group in [reg['code_sha256'],gate['receipts']]:assert all(sha(ROOT/k)==v for k,v in group.items())
assert sha(ROOT/'WINDOW_CANDIDATES.csv')==gate['ledger_sha256']
baseline_checked=0
for name,digest in reg['baseline_files_sha256'].items():
    if name.endswith('.tar.gz'):continue
    assert sha(ROOT/name)==digest;baseline_checked+=1
choices=rows(ROOT/'WINDOW_CANDIDATES.csv');assert len(choices)==30
for delay in [4,8,16]:
    origins=None
    for window in [8,16,32,64,128]:
        frozen=read(ROOT/'runs'/f'D{delay}'/f'L{window}'/'FROZEN.json')
        assert frozen['status']=='COMPLETED' and not frozen['development_read_during_fit']
        assert frozen['latest_fit_target']<=6048-delay and frozen['common_support']==127+delay
        if origins is None:origins=frozen['score_origins']
        assert origins==frozen['score_origins'] and max(origins)<6048
    for variant in ['full','no_A']:
        candidates=[r for r in choices if int(r['label_delay'])==delay and r['variant']==variant]
        best=min(candidates,key=lambda r:(float(r['selection_MSE']),int(r['window'])))
        assert int(best['window'])==gate['choices'][str(delay)][variant]['window']
        assert best['selected']=='True' and close(best['selection_MSE'],gate['choices'][str(delay)][variant]['selection_MSE'])
    assert read(ROOT/'audits'/f'D{delay}_CAUSAL_REPLAY.json')['status']=='COMPLETED'
metrics=rows(ROOT/'results/metrics.csv');assert len(metrics)==27
model_names=['Persistence','AR','ARX','LSTM','Transformer','iTransformer','TimeXer','PRISM_no_A','PRISM']
expected={};checks=0
for delay in [4,8,16]:
    wide=rows(ROOT/'results'/f'point_predictions_D{delay}.csv');assert len(wide)==2016
    for i,row in enumerate(wide):
        assert int(row['origin'])==6048+i and int(row['label_delay'])==delay
        assert int(row['latest_available_y_index'])==6048+i-delay
    for name in model_names:
        y=[float(r['y_true']) for r in wide];p=[float(r[name]) for r in wide]
        assert all(math.isfinite(v) for v in y+p)
        errors=[a-b for a,b in zip(y,p)];sse=math.fsum(e*e for e in errors);mean=math.fsum(y)/len(y)
        valid=[i for i,v in enumerate(y) if abs(v)>1e-8]
        stats=dict(R2=1-sse/math.fsum((v-mean)**2 for v in y),RMSE=math.sqrt(sse/len(y)),MAE=math.fsum(abs(e) for e in errors)/len(y),
            MAPE_pct=100*math.fsum(abs(errors[i])/abs(y[i]) for i in valid)/len(valid),
            sMAPE_pct=200*math.fsum(abs(e)/max(abs(a)+abs(b),1e-8) for a,b,e in zip(y,p,errors))/len(y),
            WAPE_pct=100*math.fsum(abs(e) for e in errors)/math.fsum(abs(v) for v in y),bias=math.fsum(b-a for a,b in zip(y,p))/len(y))
        matches=[r for r in metrics if int(r['label_delay'])==delay and r['model']==name];assert len(matches)==1
        for key,value in stats.items():assert close(value,matches[0][key]),(delay,name,key)
        for i,row in enumerate(wide):expected[(delay,name,6048+i)]=(y[i],p[i],errors[i])
        checks+=1
long=rows(ROOT/'results/point_predictions_long.csv');assert len(long)==54432;seen=set()
for row in long:
    key=(int(row['label_delay']),row['model'],int(row['origin']));assert key not in seen;seen.add(key)
    y,p,e=expected[key]
    for name,value in [('y_true',y),('y_pred',p),('residual',e),('absolute_error',abs(e)),('squared_error',e*e)]:assert close(row[name],value)
assert seen==set(expected)
result=dict(status='COMPLETED',verified_publication_files=len(manifest),metric_rows=checks,point_rows=len(long),all_metrics_recomputed=True,
    train_only_window_selection_verified=True,common_score_origins_verified=True,preserved_baseline_metadata_files=baseline_checked,
    original_experiment_source_hashes_verified=True,reserved_tail_read=False)
if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
